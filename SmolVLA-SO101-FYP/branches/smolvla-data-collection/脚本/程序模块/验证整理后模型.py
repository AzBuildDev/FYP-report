"""核对整理后的真实演示与微调模型；离线推理，不访问机械臂或相机。"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(HF_HOME=str(ROOT / "运行环境/缓存/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

import av
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from 检查演示数据 import inspect_dataset

dataset = ROOT.parent / "训练数据采集/机械臂控制项目/任务演示/单相机演示_20261005_205636_902311"
report, info, tables = inspect_dataset(dataset, ROOT / "运行记录/当前任务演示检查.json")
assert report["status"] == "passed", report["errors"]
model = ROOT / "模型/红色方块微调模型"
policy = SmolVLAPolicy.from_pretrained(model, local_files_only=True).to("cuda").eval()
video_key = report["video_keys"][0]
image_key = next(iter(policy.config.image_features))
ep = tables[0]["episode_index"][0]
video = dataset / info["video_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep, video_key=video_key)
with av.open(str(video)) as container:
    rgb = next(container.decode(video=0)).to_ndarray(format="rgb24")
with torch.inference_mode():
    action = policy.predict_action_chunk({"observation.state": torch.tensor(tables[0]["observation.state"][0], dtype=torch.float32, device="cuda")[None], image_key: torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to("cuda", torch.float32) / 255, "task": [report["tasks"][0]["task"]]})
assert list(action.shape) == [1, 50, 6] and bool(torch.isfinite(action).all())
result = {"status": "passed", "episodes": report["total_episodes"], "frames": report["total_frames"], "dataset": str(dataset), "model": str(model), "action_shape": list(action.shape), "offline": True, "robot_commands_sent": 0}
(ROOT.parents[1] / "公共工具/整理记录/模型重定位检查.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
