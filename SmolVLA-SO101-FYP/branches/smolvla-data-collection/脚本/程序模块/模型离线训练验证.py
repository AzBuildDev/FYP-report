"""用本地演示验证 SmolVLA 前向、反向和参数更新；不控制机械臂。"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(HF_HOME=str(ROOT / "运行环境/缓存/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

import av
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, load_smolvla
from 检查演示数据 import DEFAULT_DATASET, inspect_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.steps <= 100:
        raise ValueError("训练验证限定 1 至 100 步；正式微调请使用独立配置。")
    report, info, tables = inspect_dataset(args.dataset_root, ROOT / "运行记录/演示数据检查.json")
    if report["status"] != "passed":
        raise ValueError(report["errors"])
    if not torch.cuda.is_available():
        raise RuntimeError("本验证需要 CUDA GPU")
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    stats = {}
    for key in ["observation.state", "action"]:
        values = np.concatenate([np.asarray(t[key], dtype=np.float32) for t in tables])
        stats[key] = {"mean": values.mean(axis=0), "std": np.maximum(values.std(axis=0), 1e-6)}
    cfg = PreTrainedConfig.from_pretrained(ROOT / "模型/基础预训练模型", local_files_only=True)
    cfg.device, cfg.push_to_hub = "cuda", False
    cfg.vlm_model_name = str(ROOT / "模型/视觉语言处理器")
    cfg.load_vlm_weights = False
    policy = SmolVLAPolicy(cfg, dataset_stats=stats)
    load_smolvla(policy, ROOT / "模型/基础预训练模型/model.safetensors", device="cpu", checkpoint_keys_mapping="model._orig_mod.//model.")
    policy.to("cuda").train()
    params = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([p for _, p in params], lr=cfg.optimizer_lr, betas=cfg.optimizer_betas, eps=cfg.optimizer_eps, weight_decay=cfg.optimizer_weight_decay)
    probe_name, probe = next((name, p) for name, p in params if "action_out_proj" in name and name.endswith("weight"))
    before = probe.detach().clone()
    image_key, video_key = next(iter(cfg.image_features)), report["video_keys"][0]
    tasks = {t["task_index"]: t["task"] for t in report["tasks"]}
    eligible = [t for t in tables if len(t["action"]) >= cfg.chunk_size]
    if not eligible:
        raise ValueError("演示长度必须至少容纳一个动作块")
    losses, steps = [], []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        table = eligible[int(rng.integers(len(eligible)))]
        i = int(rng.integers(len(table["action"]) - cfg.chunk_size + 1))
        ep = int(table["episode_index"][i])
        video = args.dataset_root / info["video_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep, video_key=video_key)
        with av.open(str(video)) as container:
            for j, frame in enumerate(container.decode(video=0)):
                if j == i:
                    rgb = frame.to_ndarray(format="rgb24")
                    break
        batch = {"observation.state": torch.tensor(table["observation.state"][i], dtype=torch.float32, device="cuda")[None], "action": torch.tensor(table["action"][i:i + cfg.chunk_size], dtype=torch.float32, device="cuda")[None], image_key: torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to("cuda", torch.float32) / 255, "task": [tasks[table["task_index"][i]]]}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("训练损失不是有限值")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], cfg.optimizer_grad_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        steps.append({"episode": ep, "frame": i, "gradient_norm": float(norm)})
        print(f"训练验证 {step + 1}/{args.steps}，损失 {losses[-1]:.6f}", flush=True)
    torch.cuda.synchronize()
    delta = float((probe.detach() - before).abs().max())
    if delta <= 0:
        raise RuntimeError("可训练参数没有更新")
    result = {"status": "passed", "offline": True, "dataset_root": str(args.dataset_root.resolve()), "steps": args.steps, "batch_size": 1, "precision": "bfloat16 autocast", "losses": losses, "samples": steps, "trainable_parameters": sum(p.numel() for _, p in params), "parameter_probe": probe_name, "max_parameter_change": delta, "seconds": time.perf_counter() - start, "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2, "gpu": torch.cuda.get_device_name(0), "camera_mapping": {video_key: image_key}, "robot_commands_sent": 0, "checkpoint_saved": False, "task_success_evaluated": False}
    (ROOT / "运行记录/模型离线训练验证.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
