"""Read-only software and asset check; does not open cameras or move motors."""
from pathlib import Path
import argparse
import json
import sys
import cv2
import torch
import lerobot
from serial.tools import list_ports
from ultralytics import YOLO
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--policy", choices=["smolvla", "yolo"], default="smolvla")
args = parser.parse_args()
print("Python:", sys.version.split()[0])
print("LeRobot:", lerobot.__file__)
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.ones((16, 16), device="cuda")
    print("GPU:", torch.cuda.get_device_name(0), "compute:", (x @ x).mean().item())
print("OpenCV:", cv2.__version__)
print("GUI:", next((s.strip() for s in cv2.getBuildInformation().splitlines() if "GUI:" in s), "unknown"))
print("Ports:", [(p.device, p.description) for p in list_ports.comports()])
missing = []
if args.policy == "smolvla":
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    model_files = ["模型/基础预训练模型/model.safetensors", "模型/基础预训练模型/config.json", "模型/视觉语言处理器/config.json", "模型/视觉语言处理器/tokenizer.json", "模型/视觉语言处理器/preprocessor_config.json"]
else:
    model_files = ["旧视觉方案/模型/overhead_yolo_seg.pt", "旧视觉方案/模型/gripper_yolo_seg.pt"]
for name in model_files:
    p = root / name
    print("Model file:", name, "FOUND" if p.is_file() else "MISSING")
    if not p.is_file(): missing.append(name)
for rel in ["旧视觉方案/配置/桌面相机安全平面标定.json", "旧视觉方案/配置/夹爪目标轮廓配置.json"]:
    json.loads((root / rel).read_text(encoding="utf-8"))
print("Config JSON: OK")
print("Motor calibration:", len(list((root / "标定备份").rglob("*.json"))), "files")
print("Software imports: OK")
if missing:
    print("Policy assets are incomplete:", missing)
    sys.exit(2)
print("Selected policy software/assets: OK", args.policy)
print("Hardware readiness must still be checked separately before robot control.")
