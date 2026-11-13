"""SmolVLA 预训练模型实时画面推理：抓取桌面相机当前一帧，读取从臂关节状态（可选），
在 GPU 上预测 50 步动作。只读不写，绝不向机械臂发送任何动作指令。"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parents[2]
(root / "运行记录").mkdir(exist_ok=True)
if sys.stdout is None or sys.stderr is None:
    _window_log = (root / "运行记录/live-inference-window.log").open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = _window_log
    if sys.stderr is None:
        sys.stderr = _window_log
os.environ["HF_HOME"] = str(root / "运行环境/缓存/huggingface")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_LEROBOT_CALIBRATION"] = str(root / "标定备份")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, load_smolvla

CAMERA_NAME = "1080P USB Camera"
FOLLOWER_SERIAL = "5C82108242"  # 从臂 USB 序列号
WIDTH, HEIGHT = 640, 480


def capture_frame():
    """用 ffmpeg DirectShow 按设备名抓一帧 640×480 RGB 画面。"""
    import imageio_ffmpeg
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "warning",
        "-f", "dshow", "-rtbufsize", "32M", "-vcodec", "mjpeg",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", "25",
        "-i", f"video={CAMERA_NAME}", "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=30)
    expected = WIDTH * HEIGHT * 3
    if result.returncode != 0 or len(result.stdout) < expected:
        raise RuntimeError(f"相机 {CAMERA_NAME} 取帧失败：{result.stderr.decode('utf-8', 'ignore')[-300:]}")
    frame = np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape(HEIGHT, WIDTH, 3).copy()
    if frame.mean() < 3 and frame.std() < 1:
        raise RuntimeError("相机返回全黑画面，请检查镜头遮挡或连接。")
    return frame


def read_follower_state():
    """只读连接从臂，返回 6 个关节的当前位置；连接或读取失败时返回 None。"""
    try:
        from serial.tools import list_ports
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        from lerobot.robots.so101_follower.so101_follower import SO101Follower
        ports = {p.serial_number: p.device for p in list_ports.comports() if (p.vid, p.pid) == (0x1A86, 0x55D3)}
        if FOLLOWER_SERIAL not in ports:
            print("未检测到从臂 USB，改用数据集参考状态。", flush=True)
            return None
        follower = SO101Follower(SO101FollowerConfig(port=ports[FOLLOWER_SERIAL], id="my_awesome_follower_arm", max_relative_target=None))
        follower.bus.connect()
        try:
            if not follower.is_calibrated:
                print("从臂标定不匹配，改用数据集参考状态。", flush=True)
                return None
            observation = follower.get_observation()
            names = list(follower.action_features)
            state = [float(observation[k]) for k in names]
            if len(state) != 6 or not all(np.isfinite(state)):
                print("从臂关节读取异常，改用数据集参考状态。", flush=True)
                return None
            return state
        finally:
            try:
                follower.bus.disconnect()
            except Exception:
                pass
    except Exception as exc:
        print(f"从臂读取失败（{exc}），改用数据集参考状态。", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=root.parent / "训练数据采集/机械臂控制项目/历史演示/前视抓取_首批_20260613")
    parser.add_argument("--task", default="Pick up the object and place it in the target area")
    args = parser.parse_args()

    dataset = args.dataset_root
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    tables = sorted(dataset.glob("data/chunk-*/*.parquet"))
    if not tables:
        raise FileNotFoundError(f"缺少本地演示数据：{dataset}")
    fallback_state = pq.read_table(tables[0]).slice(0, 1).to_pylist()[0]["observation.state"]

    print("正在从桌面相机抓取当前画面……", flush=True)
    frame = capture_frame()

    print("正在读取从臂关节状态（只读，不启用扭矩）……", flush=True)
    state = read_follower_state()
    state_source = "follower_arm_live" if state is not None else "dataset_reference"
    if state is None:
        state = [float(v) for v in fallback_state]

    stats = {}
    for key in ["observation.state", "action"]:
        values = np.asarray([v for path in tables for v in pq.read_table(path, columns=[key])[key].to_pylist()], dtype=np.float32)
        stats[key] = {"mean": values.mean(axis=0), "std": np.maximum(values.std(axis=0), 1e-6)}

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 才能运行本验证")
    torch.manual_seed(42)
    config = PreTrainedConfig.from_pretrained(root / "模型/基础预训练模型", local_files_only=True)
    config.device = "cuda"
    config.push_to_hub = False
    config.vlm_model_name = str(root / "模型/视觉语言处理器")
    config.load_vlm_weights = False
    print("正在从本地文件加载 SmolVLA 预训练模型（已禁用联网）……", flush=True)
    policy = SmolVLAPolicy(config, dataset_stats=stats)
    policy = load_smolvla(policy, root / "模型/基础预训练模型/model.safetensors", device="cpu", checkpoint_keys_mapping="model._orig_mod.//model.")
    policy = policy.to("cuda").eval()

    image_key = next(iter(config.image_features))
    batch = {
        "observation.state": torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0),
        image_key: torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).to(device="cuda", dtype=torch.float32) / 255.0,
        "task": [args.task],
    }
    torch.cuda.reset_peak_memory_stats()
    print("正在进行 GPU 推理……", flush=True)
    start = time.perf_counter()
    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1024**2
    result = actions[0].float().cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError("推理输出包含无效数值")

    out_dir = root / "运行记录" / "实时推理" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(out_dir / "实时相机画面.png", frame)
    except Exception:
        pass
    report = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "purpose": "用实时相机画面验证 SmolVLA 预训练模型推理；未微调，不代表抓取成功率",
        "camera": CAMERA_NAME,
        "task": args.task,
        "state_source": state_source,
        "state_used": state,
        "action_chunk_shape": list(result.shape),
        "finite": True,
        "inference_seconds": elapsed,
        "peak_gpu_memory_mib": peak,
        "first_action_for_inspection_only": result[0].tolist(),
        "predicted_actions": result.tolist(),
        "robot_commands_sent": 0,
    }
    (out_dir / "实时推理结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "predicted_actions"}, ensure_ascii=False, indent=2), flush=True)
    print(f"画面和报告已保存：{out_dir}", flush=True)


if __name__ == "__main__":
    main()
