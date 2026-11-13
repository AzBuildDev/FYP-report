"""用本机录制的任务演示微调 SmolVLA 预训练模型，保存可加载的微调检查点；不控制机械臂。"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(HF_HOME=str(ROOT / "运行环境/缓存/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

import av
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, load_smolvla
from 检查演示数据 import inspect_dataset
from 管理训练数据来源 import dataset_source, file_sha256, select_tables, training_stats


class VideoFrameCache:
    """跨演示、跨相机共用的解码帧缓存，默认上限 256 MiB。"""
    def __init__(self, max_bytes=256 * 1024**2):
        self.max_bytes, self.bytes = max_bytes, 0
        self.frames = OrderedDict()

    def get(self, key):
        if key not in self.frames:
            return None
        self.frames.move_to_end(key)
        return self.frames[key]

    def put(self, key, frame):
        if frame.nbytes > self.max_bytes:
            return
        if key in self.frames:
            self.bytes -= self.frames.pop(key).nbytes
        while self.frames and self.bytes + frame.nbytes > self.max_bytes:
            self.bytes -= self.frames.popitem(last=False)[1].nbytes
        self.frames[key] = frame
        self.bytes += frame.nbytes


class EpisodeVideoFrames:
    def __init__(self, path, count, fps, cache):
        self.path, self.count, self.fps, self.cache = path, count, Fraction(str(fps)), cache

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        key = (str(self.path.resolve()), index)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            origin = stream.start_time or 0
            target = origin + int(Fraction(index, 1) / self.fps / stream.time_base)
            container.seek(target, stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise RuntimeError(f"视频缺少帧时间戳：{self.path}")
                frame_index = round((frame.pts - origin) * frame.time_base * self.fps)
                if frame_index > index:
                    break
                if frame_index == index:
                    rgb = frame.to_ndarray(format="rgb24")
                    self.cache.put(key, rgb)
                    return rgb
        raise RuntimeError(f"无法按时间戳读取第 {index} 帧：{self.path}")


def preload_frames(dataset_root, info, tables, video_key, cache=None):
    """保留旧调用接口，改为按需解码；不把全部高清演示一次性载入内存。"""
    cache = VideoFrameCache() if cache is None else cache
    episodes = {}
    for table in tables:
        ep = int(table["episode_index"][0])
        video = dataset_root / info["video_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep, video_key=video_key)
        # 帧数、尺寸和节拍由训练前的完整数据检查验证。
        episodes[ep] = EpisodeVideoFrames(video, len(table["action"]), info["fps"], cache)
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / f"模型/红色方块微调_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, nargs="+", help="选择完整演示段，编号从 0 开始；其余段不参与训练")
    parser.add_argument("--dry-run", action="store_true", help="检查数据与训练计划，不加载 GPU 模型或修改检查点")
    parser.add_argument("--output-report", type=Path, help="独立任务结果报告路径，已有文件不会覆盖")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("训练步数和批量必须大于零")
    if not args.dry_run and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("模型输出目录非空，请用 --output-dir 指定新的模型目录，保留已有检查点")
    if args.output_report and args.output_report.exists():
        parser.error("结果报告已存在，请指定新文件")
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)

    dataset_check_path = (
        args.output_report.with_name(f"{args.output_report.stem}_dataset_check.json")
        if args.output_report
        else ROOT / "运行记录/dataset_check_before_training.json"
    )
    report, info, tables = inspect_dataset(args.dataset_root, dataset_check_path)
    if report["status"] != "passed":
        raise ValueError(report["errors"])
    cfg = PreTrainedConfig.from_pretrained(ROOT / "模型/基础预训练模型", local_files_only=True)
    video_keys = report["video_keys"]
    expected_video_keys = ["observation.images.front", "observation.images.wrist"]
    if video_keys != expected_video_keys:
        raise ValueError(f"双摄训练要求视频字段 {expected_video_keys}，实际为 {video_keys}")
    cfg.input_features = {
        **{key: feature for key, feature in cfg.input_features.items() if feature.type is not FeatureType.VISUAL},
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *info["features"][key]["shape"][:2]))
           for key in video_keys},
    }
    eligible = select_tables(tables, args.episodes, cfg.chunk_size)
    source = dataset_source(args.dataset_root, info, eligible, report["tasks"])
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.dry_run:
        plan = {"status": "passed", "scope": "training_plan_only", "training_data": source,
                "steps": args.steps, "batch_size": args.batch_size, "output_dir": str(args.output_dir.resolve()),
                "checkpoint_saved": False, "robot_commands_sent": 0}
        path = args.output_report or ROOT / "运行记录" / f"训练数据计划_{run_id}.json"
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"训练计划检查通过：{path}；没有训练或修改模型。", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("微调需要 CUDA GPU")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    stats = training_stats(eligible)

    cfg.device, cfg.push_to_hub = "cuda", False
    cfg.vlm_model_name = str(ROOT / "模型/视觉语言处理器")
    cfg.load_vlm_weights = False
    policy = SmolVLAPolicy(cfg, dataset_stats=stats)
    load_smolvla(policy, ROOT / "模型/基础预训练模型/model.safetensors", device="cpu", checkpoint_keys_mapping="model._orig_mod.//model.")
    policy.to("cuda").train()
    params = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([p for _, p in params], lr=cfg.optimizer_lr, betas=cfg.optimizer_betas, eps=cfg.optimizer_eps, weight_decay=cfg.optimizer_weight_decay)

    image_keys = list(cfg.image_features)
    tasks = {t["task_index"]: t["task"] for t in report["tasks"]}
    print("按需读取双路演示视频，共享解码缓存最多 256 MiB……", flush=True)
    frame_cache = VideoFrameCache()
    episodes = {key: preload_frames(args.dataset_root, info, eligible, key, frame_cache) for key in video_keys}
    table_by_ep = {int(t["episode_index"][0]): t for t in eligible}
    ep_ids = sorted(episodes[video_keys[0]])

    log_path = ROOT / "运行记录" / f"finetune_log_{run_id}.jsonl"
    log = log_path.open("a", encoding="utf-8", buffering=1)
    log.write(json.dumps({"type": "start", "time": datetime.now().isoformat(timespec="seconds"), "dataset": str(args.dataset_root), "steps": args.steps, "batch_size": args.batch_size, "episodes": ep_ids}) + "\n")

    losses = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        states, actions, images, texts = [], [], {key: [] for key in image_keys}, []
        for _ in range(args.batch_size):
            ep = ep_ids[int(rng.integers(len(ep_ids)))]
            table = table_by_ep[ep]
            i = int(rng.integers(len(table["action"]) - cfg.chunk_size + 1))
            states.append(table["observation.state"][i])
            actions.append(table["action"][i:i + cfg.chunk_size])
            for key in image_keys:
                images[key].append(episodes[key][ep][i])
            texts.append(tasks[table["task_index"][i]])
        batch = {
            "observation.state": torch.tensor(np.asarray(states), dtype=torch.float32, device="cuda"),
            "action": torch.tensor(np.asarray(actions), dtype=torch.float32, device="cuda"),
            "task": texts,
        }
        batch.update({key: torch.from_numpy(np.stack(images[key])).permute(0, 3, 1, 2).to("cuda", torch.float32) / 255
                      for key in image_keys})
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("训练损失不是有限值")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], cfg.optimizer_grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 50 == 0 or step == 0:
            recent = float(np.mean(losses[-50:]))
            elapsed = time.perf_counter() - start
            eta = elapsed / (step + 1) * (args.steps - step - 1)
            print(f"微调 {step + 1}/{args.steps}，近 50 步平均损失 {recent:.6f}，梯度范数 {float(norm):.3f}，已用 {elapsed/60:.1f} 分钟，预计剩余 {eta/60:.1f} 分钟", flush=True)
            log.write(json.dumps({"type": "step", "step": step + 1, "loss": losses[-1], "avg50": recent, "grad_norm": float(norm), "elapsed_s": elapsed}) + "\n")
    torch.cuda.synchronize()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(args.output_dir)
    stats_json = {k: {"mean": v["mean"].tolist(), "std": v["std"].tolist()} for k, v in stats.items()}
    (args.output_dir / "dataset_stats.json").write_text(json.dumps(stats_json, ensure_ascii=False, indent=2), encoding="utf-8")
    provenance = {"schema_version": 1, "status": "completed", "training_data": source,
                  "checkpoint_sha256": file_sha256(args.output_dir / "model.safetensors"),
                  "created_at": datetime.now().isoformat(timespec="seconds"), "seed": args.seed,
                  "steps": args.steps, "batch_size": args.batch_size}
    (args.output_dir / "training_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    seconds = time.perf_counter() - start
    result = {
        "status": "passed", "offline": True,
        "dataset_root": str(args.dataset_root.resolve()),
        "base_model": "模型/基础预训练模型", "output_dir": str(args.output_dir.resolve()),
        "steps": args.steps, "batch_size": args.batch_size, "precision": "bfloat16 autocast",
        "episodes": ep_ids, "total_frames": int(sum(len(t["action"]) for t in eligible)),
        "first_50_avg_loss": float(np.mean(losses[:50])), "last_50_avg_loss": float(np.mean(losses[-50:])),
        "seconds": seconds, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_saved": True, "robot_commands_sent": 0, "task_success_evaluated": False,
        "training_data": source, "checkpoint_sha256": provenance["checkpoint_sha256"],
    }
    result_path = args.output_report or ROOT / "运行记录" / f"finetune_result_{run_id}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log.write(json.dumps({"type": "done", **result}) + "\n")
    log.close()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
