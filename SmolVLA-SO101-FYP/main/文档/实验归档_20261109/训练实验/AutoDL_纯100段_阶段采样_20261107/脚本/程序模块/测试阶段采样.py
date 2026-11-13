"""核验完整标注、分层抽样、验证隔离与可重复恢复；不加载模型或更新权重。"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
from 阶段采样 import StageSampler, load_annotations
from 目标选择标注数据 import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def rejects(callback):
    try:
        callback()
    except (ValueError, KeyError):
        return
    raise AssertionError("无效配置没有被拒绝")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-batch", action="store_true", help="额外核验真实双摄CPU批次，不加载GPU模型")
    args = parser.parse_args(argv)
    split = json.loads((ROOT / "划分.json").read_text(encoding="utf-8"))
    annotation_path = ROOT / "标注/阶段标注_100段.json"
    annotations, report = load_annotations(ROOT / "训练数据", annotation_path, split)
    tables = [pq.read_table(p).to_pydict() for p in sorted((ROOT / "训练数据").glob("data/chunk-*/*.parquet"))]
    by_id = {int(t["episode_index"][0]): t for t in tables}
    train = [by_id[ep] for ep in split["train"]]
    counts = Counter()
    selected = set()
    crossing = 0
    for mode in ("stage", "full"):
        for batch in (16, 32, 64):
            sampler = StageSampler(train, annotations, batch, mode=mode)
            restored = StageSampler(train, annotations, batch, mode=mode)
            for step in range(1, 101):
                plan = sampler.plan(step)
                assert plan == restored.plan(step) and plan != sampler.plan(step + 1)
                groups = Counter((r["stage"], r["task_index"]) for r in plan)
                assert len(plan) == batch
                assert len(groups) == (4 if mode == "stage" else 2)
                assert set(groups.values()) == {batch // len(groups)}
                for record in plan:
                    ep, frame = record["episode"], record["frame"]
                    row = annotations[ep]
                    assert ep in split["train"] and ep not in split["validation"]
                    assert 0 <= frame <= row["length"] - 50
                    assert len(by_id[ep]["action"][frame:frame + 50]) == 50
                    assert row["target"] == ("RC" if record["task_index"] == 0 else "BC")
                    if record["stage"] == "pre_marker":
                        assert frame < row["markers"]["first_contact"]
                        crossing += int(frame + 49 >= row["markers"]["first_contact"])
                    selected.add(ep)
                    counts[f"{mode}_batch{batch}_{record['stage']}_{record['task_index']}"] += 1
            # 预读其它步及顺序不同不会消耗采样随机状态。
            sampler.plan(120)
            assert sampler.plan(88) == restored.plan(88)
    assert selected == set(split["train"]) and crossing > 0
    for batch in (0, 3, 15):
        rejects(lambda b=batch: StageSampler(train, annotations, b))
    sampler = StageSampler(train, annotations, 32)
    rejects(lambda: sampler.plan(0))
    wrong = copy.deepcopy(annotations)
    wrong[split["train"][0]]["target"] = "WRONG"
    rejects(lambda: StageSampler(train, wrong, 32))
    wrong = copy.deepcopy(annotations)
    wrong[split["train"][0]]["markers"]["first_contact"] = 0
    rejects(lambda: StageSampler(train, wrong, 32))
    raw = json.loads(annotation_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="阶段采样校验_") as temporary:
        p = Path(temporary) / "invalid.json"
        wrong = copy.deepcopy(raw)
        wrong["dataset_fingerprint"] = "wrong"
        atomic_json(p, wrong)
        rejects(lambda: load_annotations(ROOT / "训练数据", p, split))
        wrong = copy.deepcopy(raw)
        wrong["episodes"][0]["target"] = "WRONG"
        atomic_json(p, wrong)
        rejects(lambda: load_annotations(ROOT / "训练数据", p, split))
        wrong = copy.deepcopy(raw)
        wrong["episodes"][0]["video_sha256"] = {}
        atomic_json(p, wrong)
        rejects(lambda: load_annotations(ROOT / "训练数据", p, split))
    decoded = 0
    if args.decode_batch:
        sys.path.insert(0, str(ROOT / "上游框架/src"))
        import 百段阶段训练 as trainer
        info = json.loads((ROOT / "训练数据/meta/info.json").read_text(encoding="utf-8"))
        keys = ["observation.images.front", "observation.images.wrist"]
        cache = trainer.LockedVideoFrameCache(128 * 1024**2)
        videos = {k: trainer.preload_frames(ROOT / "训练数据", info, tables, k, cache) for k in keys}
        tasks = {0: raw["episodes"][next(i for i,r in enumerate(raw["episodes"]) if r["target"]=="RC")]["task"],
                 1: raw["episodes"][next(i for i,r in enumerate(raw["episodes"]) if r["target"]=="BC")]["task"]}
        plan = sampler.plan(1)
        cpu = trainer.read_cpu_batch(sampler.records(1), videos, keys, tasks, 50)
        assert cpu["action"].shape == (32, 50, 6) and cpu["observation.state"].shape == (32, 6)
        for i, r in enumerate(plan):
            table, frame = by_id[r["episode"]], r["frame"]
            assert np.array_equal(cpu["action"][i], np.asarray(table["action"][frame:frame + 50], dtype=np.float32))
            assert np.array_equal(cpu["observation.state"][i], np.asarray(table["observation.state"][frame], dtype=np.float32))
            assert cpu["task"][i] == tasks[r["task_index"]]
            for key in keys:
                assert np.array_equal(cpu[key][i], videos[key][r["episode"]][frame])
                assert cpu[key].shape[0] == 32 and cpu[key].shape[-1] == 3
        decoded = 64
        assert not trainer.torch.cuda.is_initialized()
        trainer.DECODE_POOL.shutdown(wait=True)
    report.update(status="passed", tested_modes=["stage", "full"], tested_batches=[16, 32, 64],
                  tested_steps_per_mode_batch=100, train_episodes_observed=len(selected),
                  pre_marker_windows_crossing_marker=crossing, exact_resume_sampling=True,
                  decoded_cpu_images=decoded, counts=dict(counts), gpu_model_loaded=False,
                  optimizer_updates=0, robot_commands_sent=0)
    atomic_json(ROOT / "阶段采样核验.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
