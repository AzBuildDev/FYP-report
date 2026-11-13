"""Offline GPU inference on one saved SO101 observation. No robot or camera access."""
import argparse
import json
import os
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.environ["HF_HOME"] = str(root / "运行环境/缓存/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, load_smolvla

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", type=Path, default=root.parent / "训练数据采集/机械臂控制项目/历史演示/前视抓取_首批_20260613")
args = parser.parse_args()
dataset = args.dataset_root
info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
tables = sorted(dataset.glob("data/chunk-*/*.parquet"))
if not tables:
    raise FileNotFoundError(f"No local episode data: {dataset}")
first = pq.read_table(tables[0]).slice(0, 1).to_pylist()[0]
stats = {}
for key in ["observation.state", "action"]:
    values = np.asarray([v for path in tables for v in pq.read_table(path, columns=[key])[key].to_pylist()], dtype=np.float32)
    stats[key] = {"mean": values.mean(axis=0), "std": np.maximum(values.std(axis=0), 1e-6)}
video_key = next(key for key, value in info["features"].items() if value.get("dtype") == "video")
episode = int(first["episode_index"])
video = dataset / info["video_path"].format(episode_chunk=episode // info["chunks_size"], episode_index=episode, video_key=video_key)
with av.open(str(video)) as container:
    frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
tasks = [json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
task = next(item["task"] for item in tasks if item["task_index"] == int(first["task_index"]))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this verification")
torch.manual_seed(42)
config = PreTrainedConfig.from_pretrained(root / "模型/基础预训练模型", local_files_only=True)
config.device = "cuda"
config.push_to_hub = False
config.vlm_model_name = str(root / "模型/视觉语言处理器")
# The complete policy checkpoint already contains the VLM weights.
config.load_vlm_weights = False
print("Creating SmolVLA from local configs with all network access disabled", flush=True)
policy = SmolVLAPolicy(config, dataset_stats=stats)
policy = load_smolvla(policy, root / "模型/基础预训练模型/model.safetensors", device="cpu", checkpoint_keys_mapping="model._orig_mod.//model.")
policy = policy.to("cuda").eval()
image_key = next(iter(config.image_features))
batch = {
    "observation.state": torch.tensor(first["observation.state"], dtype=torch.float32, device="cuda").unsqueeze(0),
    image_key: torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.float32) / 255.0,
    "task": [task],
}
print("Inference input:", video_key, "->", image_key, "task:", task, flush=True)
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    chunk = policy.predict_action_chunk(batch)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
finite = bool(torch.isfinite(chunk).all())
expected_shape = [1, config.chunk_size, config.action_feature.shape[0]]
if list(chunk.shape) != expected_shape or not finite:
    raise RuntimeError(f"Invalid action output: {chunk.shape}, finite={finite}")
report = {
    "status": "passed", "offline": True, "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__, "dataset_root": str(dataset), "episode": episode,
    "task": task, "camera_mapping": {video_key: image_key}, "views_used": 1,
    "normalization": "mean/std computed from the existing local dataset; not a task-finetuned model",
    "action_chunk_shape": list(chunk.shape), "finite": finite, "inference_seconds": elapsed,
    "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    "first_action_for_inspection_only": chunk[0, 0].float().cpu().tolist(),
    "robot_commands_sent": 0, "task_success_evaluated": False,
}
(root / "运行记录/模型离线推理结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
