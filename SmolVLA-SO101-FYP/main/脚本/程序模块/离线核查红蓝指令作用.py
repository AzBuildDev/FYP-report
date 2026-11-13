"""固定录制画面、状态及随机种子，对照正确/相反指令；不连接相机或机械臂。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def ensure_no_live_inference():
    """实机运行期间禁止额外占用推理显卡；不终止已有进程。"""
    command = (
        "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' "
        "-and $_.CommandLine -match '实机抓取测试[.]py|录制任务演示[.]py' } "
        "| Select-Object -ExpandProperty ProcessId) | ConvertTo-Json -Compress"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError("无法确认实机进程状态，未加载GPU模型。")
    if result.stdout.strip():
        raise RuntimeError("仍有实机推理或录制进程，请先在窗口正常停止。本脚本没有终止它们。")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    ensure_no_live_inference()
    import av
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from 红蓝抓取任务 import TASKS
    from 选择推理模型 import validate_selection, load_local_policy

    if args.episodes_per_task < 1:
        raise ValueError("每种任务的段数必须为正")
    selection = validate_selection(args.model_dir, args.dataset_root)
    summary = json.loads((selection.model / "checkpoint_summary.json").read_text(encoding="utf-8"))
    held_out = summary["run_config"]["validation_episodes"]
    tasks = {r["task_index"]: r["task"] for r in (
        json.loads(line) for line in (selection.dataset / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip())}
    if set(tasks.values()) != set(TASKS.values()):
        raise ValueError("数据集任务文字与RC/BC不一致")
    info = json.loads((selection.dataset / "meta/info.json").read_text(encoding="utf-8"))
    tables = {}
    for p in sorted(selection.dataset.glob("data/chunk-*/*.parquet")):
        t = pq.read_table(p).to_pydict()
        if int(t["episode_index"][0]) in held_out:
            tables[int(t["episode_index"][0])] = t
    selected = []
    for obj, text in TASKS.items():
        candidates = sorted(ep for ep, t in tables.items() if tasks[t["task_index"][0]] == text)
        if not candidates:
            raise ValueError("验证集中缺少任务：" + obj)
        take = min(args.episodes_per_task, len(candidates))
        selected.extend((obj, candidates[int(i)]) for i in np.linspace(0, len(candidates)-1, take, dtype=int))
    stats = json.loads((selection.model / "dataset_stats.json").read_text(encoding="utf-8"))
    std = np.asarray(stats["action"]["std"], dtype=np.float32)
    if not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("动作归一化标准差无效")
    report = {
        "created_at": datetime.now().astimezone().isoformat(), "scope": "offline_instruction_ablation",
        "hardware_connected": False, "robot_commands_sent": 0, "physical_success_evaluated": False,
        "model": str(selection.model), "checkpoint_step": summary["step"],
        "checkpoint_sha256": file_hash(selection.model / "model.safetensors"),
        "script_sha256": file_hash(__file__), "dataset": str(selection.dataset),
        "held_out_episodes_selected": [ep for _, ep in selected],
        "seeds": args.seeds, "precision": "float32_like_local_inference",
        "positions_in_episode": ["first_frame", "15_percent_of_episode"],
        "limitations": [
            "相反指令没有本帧的另一条动作真值；比较其与原演示的误差只能诊断指令作用，不能计为选色准确率。",
            "两个时间点由录制进度确定，未人工标注下夹阶段；15%时间点的状态可能已经暴露演示方向。",
            "预测变化不能单独证明抓对目标；必须结合物体位置及闭环实测。",
            "不以任意阈值把结果判为模型成功或失败。",
        ], "records": [],
    }
    output = ROOT / "运行记录" / f"红蓝指令离线对照_{datetime.now():%Y%m%d_%H%M%S}.json"
    if args.resume_report:
        previous = json.loads(args.resume_report.read_text(encoding="utf-8-sig"))
        for key in ("model", "checkpoint_sha256", "dataset", "held_out_episodes_selected", "seeds", "positions_in_episode"):
            if previous[key] != report[key]:
                raise ValueError("恢复对照报告时配置不一致：" + key)
        report = previous
        output = args.resume_report
        report.setdefault("resumptions", []).append({
            "time": datetime.now().astimezone().isoformat(), "script_sha256": file_hash(__file__)})
    output.parent.mkdir(exist_ok=True)
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"离线加载{summary['step']}步模型，验证{len(selected)}段；不会连接硬件。", flush=True)
    policy = load_local_policy(selection.model, selection.processor)
    policy.eval()
    keys = list(policy.config.image_features)
    chunk = policy.config.chunk_size
    started = time.monotonic()
    for obj, ep in selected:
        table = tables[ep]
        indices = sorted({0, min(len(table["action"])-chunk, int(len(table["action"])*0.15))})
        previous_records = [r for r in report["records"] if r["episode"] == ep]
        expected_pairs = {(i, seed) for i in indices for seed in args.seeds}
        if {(r["frame"], r["seed"]) for r in previous_records} == expected_pairs:
            continue
        if previous_records:
            raise ValueError("存在未完整保存的段，请先核对报告：" + str(ep))
        ensure_no_live_inference()
        frames = {}
        for key in keys:
            path = selection.dataset / info["video_path"].format(
                episode_chunk=ep//info["chunks_size"], episode_index=ep, video_key=key)
            with av.open(str(path)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    if index in indices:
                        frames[key, index] = frame.to_ndarray(format="rgb24")
                    if index >= max(indices):
                        break
        for index in indices:
            batch = {"observation.state": torch.tensor(table["observation.state"][index],
                                                       dtype=torch.float32, device="cuda")[None]}
            batch.update({key: torch.from_numpy(frames[key, index]).permute(2, 0, 1)[None].to(
                "cuda", torch.float32)/255 for key in keys})
            expected = np.asarray(table["action"][index:index+chunk], dtype=np.float32)
            other = "BC" if obj == "RC" else "RC"
            for seed in args.seeds:
                predictions = {}
                for target in (obj, other):
                    policy.reset()
                    torch.manual_seed(seed)
                    with torch.inference_mode():
                        predictions[target] = policy.predict_action_chunk(
                            {**batch, "task": [TASKS[target]]})[0].float().cpu().numpy()
                    if predictions[target].shape != expected.shape or not np.isfinite(predictions[target]).all():
                        raise RuntimeError("预测形状或数值无效")
                errors = {target: float((np.abs(predictions[target][:20]-expected[:20])/std).mean())
                          for target in (obj, other)}
                report["records"].append({
                    "episode": ep, "frame": index, "original_target": obj, "seed": seed,
                    "correct_prompt_first20_mae": errors[obj],
                    "opposite_prompt_to_original_demo_first20_mae": errors[other],
                    "opposite_minus_correct_mae": errors[other]-errors[obj],
                    "prompt_change_first20_normalized_mae": float((
                        np.abs(predictions[obj][:20]-predictions[other][:20])/std).mean()),
                    "expected_first20": expected[:20].tolist(),
                    "predicted_RC_first20": predictions["RC"][:20].tolist(),
                    "predicted_BC_first20": predictions["BC"][:20].tolist(),
                })
            del batch
        save()
        print(f"已完成{obj} 视频编号{ep:06d}；累计{len(report['records'])}组对照。", flush=True)
    aggregates = {}
    for name, rows in [("all", report["records"])] + [
        (obj+"_"+phase, [r for r in report["records"] if r["original_target"] == obj and
                        ((r["frame"] == 0) if phase == "first_frame" else (r["frame"] != 0))])
        for obj in TASKS for phase in ("first_frame", "later_frame")]:
        if rows:
            aggregates[name] = {
                "comparisons": len(rows),
                "mean_correct_prompt_mae": float(np.mean([r["correct_prompt_first20_mae"] for r in rows])),
                "mean_opposite_prompt_mae": float(np.mean([r["opposite_prompt_to_original_demo_first20_mae"] for r in rows])),
                "mean_opposite_minus_correct_mae": float(np.mean([r["opposite_minus_correct_mae"] for r in rows])),
                "fraction_correct_prompt_closer_to_demo": float(np.mean([r["opposite_minus_correct_mae"] > 0 for r in rows])),
                "mean_prompt_change_normalized_mae": float(np.mean([r["prompt_change_first20_normalized_mae"] for r in rows])),
            }
    report.update(aggregates=aggregates, status="completed", elapsed_s=time.monotonic()-started)
    save()
    print(json.dumps(aggregates, ensure_ascii=False, indent=2), flush=True)
    print(f"报告：{output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "模型/新200段_batch32_20261102/step_020000")
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "AutoDL_新200段_4090D_20261102/训练数据")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--resume-report", type=Path, help="保持检查点、样本、随机种子不变，只补算未完成段")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
