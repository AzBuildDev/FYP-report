"""实机闭环抓取测试：微调模型读取实时画面和关节状态，直接控制从臂。
控制限制：每步关节变化限量、运行范围包含当前姿态、限时运行、Ctrl+C 停止并释放扭矩。"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "上游框架/src"))
(ROOT / "运行记录").mkdir(exist_ok=True)
os.environ.update(
    HF_HOME=str(ROOT / "运行环境/缓存/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    HF_DATASETS_OFFLINE="1", HF_LEROBOT_CALIBRATION=str(ROOT / "标定备份"),
    TOKENIZERS_PARALLELISM="false",
)

import numpy as np
import pyarrow.parquet as pq
import torch
from 检查控制输入 import bounded_target, bounds_including_start, validate_frame
from 异步动作控制 import run_async_control
from 等待测试开始 import wait_for_start
from 管理训练数据来源 import file_sha256
from 预热模型与检查时效 import warm_model, join_warm_workers
from 检查与释放电机 import MotorHealthGuard, release_bus
from 实机测试监视窗口 import run_monitor

CAMERA_NAME = "1080P USB Camera"
FOLLOWER_SERIAL = "5C82108242"
MODEL_DIR = ROOT / "模型/红色方块微调模型"
DATASET_ROOT = ROOT.parent / "训练数据采集/机械臂控制项目/任务演示/单相机演示_20261005_205636_902311"
TASK = "Pick up the red cube and place it in the target area."
CONTROL_HZ = 20              # 控制频率
MAX_STEP_DELTA = 6.0         # 每步归一化位置变化限量；不等同于角度
RANGE_MARGIN = 8.0           # 演示活动范围之外的归一化位置余量
CHUNK_EXECUTE = 20           # 异步模式跳过过期前缀，每块最多执行 20 步
TEST_SECONDS = 60            # 单次测试最长秒数
RUN_KIND = "real_hardware"   # 模拟入口自检必须显式替换该标记
MOTOR_HEALTH_INTERVAL = 1.0  # 主线程每秒读取全部六个电机故障状态

latest_frame = {"time": 0.0, "image": None}
frame_lock = threading.Lock()
camera_stop = threading.Event()
user_stop = threading.Event()
monitor_status = {"text": "准备中"}


def monitor_loop():
    """Tk 在独立进程的主线程运行；父进程只传预览、文字和停止请求。"""
    try:
        run_monitor(ROOT, get_frame, camera_stop, user_stop, lambda: monitor_status["text"])
    except Exception as exc:
        monitor_status["error"] = str(exc)
        user_stop.set()
        print(f"监视窗口异常，已请求停止：{exc}", flush=True)


def close_monitors(threads):
    camera_stop.set()
    errors = []
    for thread in threads:
        thread.join(timeout=7)
        if thread.is_alive():
            errors.append("监视窗口通信线程未退出")
    return errors


def camera_callback(kind, value):
    if kind == "frame":
        stamp, image = value
        with frame_lock:
            latest_frame["time"], latest_frame["image"] = stamp, image


def get_frame():
    with frame_lock:
        return latest_frame["time"], latest_frame["image"]


def release_follower(follower):
    return release_bus(follower.bus)


def close_camera(thread):
    camera_stop.set()
    thread.join(timeout=2)


def make_predict(policy, image_key):
    def predict(state, image):
        with torch.inference_mode():
            chunk = policy.predict_action_chunk({
                "observation.state": torch.tensor(state, dtype=torch.float32, device="cuda")[None],
                image_key: torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1)[None].to("cuda", torch.float32) / 255,
                "task": [TASK],
            })
        return chunk[0].float().cpu().numpy()
    return predict


def main(mode="async", scene="演示起始位置"):
    if not scene.strip():
        raise ValueError("测试场景不能为空")
    user_stop.clear()
    camera_stop.clear()
    monitor_status["error"] = ""
    with ExitStack() as resources:
        resources.callback(camera_stop.set)
        run_test(resources, mode, scene.strip())


def run_test(resources, mode="async", scene="演示起始位置"):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from serial.tools import list_ports
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from 相机测试窗口 import named_capture

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = ROOT / "运行记录" / f"实机抓取测试_{run_id}.jsonl"
    log = resources.enter_context(log_path.open("w", encoding="utf-8", buffering=1))
    terminal = {"written": False, "event": None, "cleanup_errors": []}

    def finish_preparation():
        if not terminal["written"] and not log.closed:
            exception = sys.exc_info()[1]
            event = terminal["event"] or {"type": "preparation_end", "sent_commands": 0,
                                          "error": str(exception) if exception else "准备阶段退出，未启动控制"}
            if terminal["cleanup_errors"]:
                event["cleanup_errors"] = terminal["cleanup_errors"]
            log.write(json.dumps(event, ensure_ascii=False) + "\n")

    def cancel_preparation():
        terminal["event"] = {"type": "cancelled_before_start", "sent_commands": 0}

    resources.callback(finish_preparation)
    warm_workers = []
    # 后注册的串口、相机先释放，再有限等待不接触硬件的预热线程。
    resources.callback(join_warm_workers, warm_workers)
    monitor_threads = []
    resources.callback(lambda: terminal["cleanup_errors"].extend(close_monitors(monitor_threads)))
    log.write(json.dumps({"type": "配置", "task": TASK, "control_hz": CONTROL_HZ, "max_step_delta": MAX_STEP_DELTA,
                          "mode": mode, "scene": scene, "run_kind": RUN_KIND, "run_id": run_id,
                          "start_pose_policy": "current_pose", "range_margin": RANGE_MARGIN,
                          "chunk_execute": CHUNK_EXECUTE, "test_seconds": TEST_SECONDS}) + "\n")
    log.write(json.dumps({"type": "model_identity", "model_path": str(MODEL_DIR),
                          "checkpoint_sha256": file_sha256(MODEL_DIR / "model.safetensors")}, ensure_ascii=False) + "\n")

    print("【1/5】正在加载微调模型……", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU")
    torch.manual_seed(42)
    policy = SmolVLAPolicy.from_pretrained(MODEL_DIR, local_files_only=True).to("cuda").eval()
    image_key = next(iter(policy.config.image_features))

    print("【2/5】正在计算安全活动范围……", flush=True)
    tables = sorted(DATASET_ROOT.glob("data/chunk-*/*.parquet"))
    states = np.concatenate([np.asarray(pq.read_table(t, columns=["observation.state"])["observation.state"].to_pylist(), dtype=np.float32) for t in tables])
    lo, hi = states.min(axis=0) - RANGE_MARGIN, states.max(axis=0) + RANGE_MARGIN

    print("【3/5】正在连接从臂（只检测，不动）……", flush=True)
    ports = {p.serial_number: p.device for p in list_ports.comports() if (p.vid, p.pid) == (0x1A86, 0x55D3)}
    if FOLLOWER_SERIAL not in ports:
        raise RuntimeError("未检测到从臂 USB，请确认 USB 和电源已接好。")
    follower = SO101Follower(SO101FollowerConfig(port=ports[FOLLOWER_SERIAL], id="my_awesome_follower_arm", max_relative_target=None))
    resources.callback(lambda: terminal["cleanup_errors"].extend(release_follower(follower)))
    follower.bus.connect()
    health = MotorHealthGuard(follower.bus, MOTOR_HEALTH_INTERVAL)
    health.check(force=True)
    if not follower.is_calibrated:
        raise RuntimeError("从臂标定不匹配，已停止。")
    names = list(follower.action_features)
    info = json.loads((DATASET_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    if info["features"]["action"]["names"] != names or info["features"]["observation.state"]["names"] != names:
        raise RuntimeError("演示数据和从臂关节顺序不一致，已停止。")
    observation = follower.get_observation()
    last_sent = np.array([observation[k] for k in names], dtype=np.float32)
    print(f"      从臂 {ports[FOLLOWER_SERIAL]} 在线，当前关节：{np.round(last_sent, 1).tolist()}", flush=True)

    print("【4/5】正在开启桌面相机……", flush=True)
    camera_thread = threading.Thread(target=named_capture, args=(1, "MJPG", 25, camera_callback, camera_stop, CAMERA_NAME, None), daemon=True)
    camera_thread.start()
    resources.callback(close_camera, camera_thread)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if user_stop.is_set():
            cancel_preparation()
            return
        stamp, image = get_frame()
        if image is not None and time.monotonic() - stamp < 1.0 and not (image.mean() < 3 and image.std() < 1):
            break
        time.sleep(0.1)
    else:
        camera_stop.set()
        raise RuntimeError("相机没有提供有效画面，已停止。")
    print("      相机画面正常。", flush=True)
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_threads.append(monitor_thread)
    monitor_thread.start()
    monitor_status["text"] = "正在预热模型；尚未启用控制，可停止"

    print("      正在预热模型并检查推理时效（不发送动作）……", flush=True)
    stamp, image = get_frame()
    validate_frame(stamp, image, time.monotonic())
    try:
        warm_report = warm_model(make_predict(policy, image_key), last_sent, image, user_stop,
                                 chunk_size=policy.config.chunk_size, hz=CONTROL_HZ, workers=warm_workers)
    except KeyboardInterrupt:
        user_stop.set()
        warm_report = {"status": "cancelled", "prediction_reused": False}
    log.write(json.dumps({"type": "model_warmup", **warm_report}, ensure_ascii=False) + "\n")
    if warm_report["status"] != "ready" or user_stop.is_set():
        cancel_preparation()
        print("预热期间已取消，未启动控制。", flush=True)
        return
    policy.reset()
    monitor_status["text"] = "预热通过，等待确认开始；预热动作已丢弃"

    print("【5/5】准备就绪。", flush=True)
    print("=" * 60, flush=True)
    print(f"控制设置：从当前位置开始，每步归一化位置最多变化 {MAX_STEP_DELTA:g}。", flush=True)
    print("运行范围包含确认时的当前姿态，不要求摆回演示起始范围。", flush=True)
    print(f"最长运行 {TEST_SECONDS} 秒。测试期间从臂会自行运动，请全程看护。", flush=True)
    print("按 Ctrl+C 或监视窗口的“停止测试”停止并释放扭矩。确认周围安全后，按回车开始；等待期间 Esc 或关闭窗口可取消。", flush=True)
    print("异步模式每个控制节拍处理停止请求；串行模式需等当前推理结束。", flush=True)
    print("=" * 60, flush=True)
    try:
        confirmed = wait_for_start(user_stop)
    except (KeyboardInterrupt, EOFError):
        confirmed = False
    if not confirmed or user_stop.is_set():
        cancel_preparation()
        print("已取消，未启用扭矩。", flush=True)
        return
    stamp, image = get_frame()
    validate_frame(stamp, image, time.monotonic())
    observation = follower.get_observation()
    last_sent = np.array([observation[k] for k in names], dtype=np.float32)
    lo, hi = bounds_including_start(last_sent, lo, hi, RANGE_MARGIN)
    # 读取串口可能耗时：紧邻扭矩配置再次核对画面与停止请求。
    stamp, image = get_frame()
    validate_frame(stamp, image, time.monotonic())
    if user_stop.is_set():
        cancel_preparation()
        return
    health.check(force=True)

    sent_commands = 0
    inferences = 0
    async_stats, workers = {}, []
    start = time.monotonic()
    exit_reason = "time_limit"
    try:
        print("正在启用从臂扭矩……", flush=True)
        follower.bus.disable_torque()
        hold = follower.bus.sync_read("Present_Position")
        stamp, image = get_frame()
        validate_frame(stamp, image, time.monotonic())
        if user_stop.is_set():
            exit_reason = "user_stop"
            return
        follower.bus.sync_write("Goal_Position", hold)
        follower.configure()
        torque = follower.bus.sync_read("Torque_Enable", normalize=False)
        if len(torque) != 6 or any(v != 1 for v in torque.values()):
            raise RuntimeError("六个电机没有全部启用，已停止。")
        print(f"开始闭环控制，{TEST_SECONDS} 秒后自动停止。", flush=True)
        log.write(json.dumps({"type": "start", "initial_state": last_sent.tolist(),
                              "control_lower": lo.tolist(), "control_upper": hi.tolist()}) + "\n")

        if mode == "async":
            def read_state():
                health.check()
                observation = follower.get_observation()
                return np.array([observation[k] for k in names], dtype=np.float32)

            def send_target(target):
                # 位置命令可能清除设备保护标志，发送前先读取所有故障状态。
                health.check(force=True)
                stamp, image = get_frame()
                validate_frame(stamp, image, time.monotonic())
                if user_stop.is_set():
                    raise KeyboardInterrupt
                return follower.send_action({k: float(v) for k, v in zip(names, target)})

            def record_event(event):
                log.write(json.dumps(event, ensure_ascii=False) + "\n")
                monitor_status["text"] = (f"异步运行 {time.monotonic() - start:.0f}/{TEST_SECONDS} 秒 | "
                                          f"推理 {async_stats.get('inferences', 0)} 次 | "
                                          f"已发送 {async_stats.get('sent_commands', 0)} 步 | 停止测试 / Ctrl+C")

            run_async_control(make_predict(policy, image_key), read_state, get_frame, send_target,
                              user_stop, lo, hi, duration=max(0.001, TEST_SECONDS - (time.monotonic() - start)),
                              hz=CONTROL_HZ, max_delta=MAX_STEP_DELTA, execute_limit=CHUNK_EXECUTE,
                              on_event=record_event, stats=async_stats, workers=workers)
        else:
            while time.monotonic() - start < TEST_SECONDS:
                if user_stop.is_set():
                    break
                health.check()
                stamp, image = get_frame()
                validate_frame(stamp, image, time.monotonic())
                observation = follower.get_observation()
                state = np.array([observation[k] for k in names], dtype=np.float32)
                infer_start = time.perf_counter()
                with torch.inference_mode():
                    chunk = policy.predict_action_chunk({
                        "observation.state": torch.tensor(state, dtype=torch.float32, device="cuda").unsqueeze(0),
                        image_key: torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float32) / 255,
                        "task": [TASK],
                    })
                actions = chunk[0].float().cpu().numpy()
                inferences += 1
                monitor_status["text"] = (f"运行中 {time.monotonic() - start:.0f}/{TEST_SECONDS} 秒 | "
                                          f"推理 {inferences} 次（{time.perf_counter() - infer_start:.1f} 秒）| "
                                          f"已发送 {sent_commands} 步 | Ctrl+C 停止")
                if not np.isfinite(actions).all():
                    raise RuntimeError("模型输出无效数值，立即停止。")
                log.write(json.dumps({"type": "chunk", "t": round(time.monotonic() - start, 2),
                                      "inference_s": round(time.perf_counter() - infer_start, 2),
                                      "state": state.tolist(), "first_action": actions[0].tolist()}) + "\n")

                for target in actions[:CHUNK_EXECUTE]:
                    if user_stop.is_set() or time.monotonic() - start >= TEST_SECONDS:
                        break
                    health.check(force=True)
                    stamp, live_image = get_frame()
                    validate_frame(stamp, live_image, time.monotonic())
                    observation = follower.get_observation()
                    current = np.array([observation[k] for k in names], dtype=np.float32)
                    target = bounded_target(target, current, lo, hi, MAX_STEP_DELTA)
                    step_start = time.monotonic()
                    actual = follower.send_action({k: float(v) for k, v in zip(names, target)})
                    sent_commands += 1
                    last_sent = np.array([actual[k] for k in names], dtype=np.float32)
                    elapsed = time.monotonic() - step_start
                    if elapsed < 1 / CONTROL_HZ:
                        time.sleep(1 / CONTROL_HZ - elapsed)
        print("收到停止请求。" if user_stop.is_set() else f"{TEST_SECONDS} 秒测试时间到。", flush=True)
        if user_stop.is_set():
            exit_reason = "user_stop"
    except KeyboardInterrupt:
        exit_reason = "user_stop"
        print("\n收到 Ctrl+C，立即停止。", flush=True)
        log.write(json.dumps({"type": "interrupted", "t": round(time.monotonic() - start, 2)}) + "\n")
    except Exception as exc:
        exit_reason = "error"
        print(f"\n异常停止：{exc}", flush=True)
        log.write(json.dumps({"type": "error", "error": str(exc), "trace": traceback.format_exc()}) + "\n")
        raise
    finally:
        cleanup_errors = release_follower(follower)
        camera_stop.set()
        cleanup_errors.extend(close_monitors(monitor_threads))
        if monitor_status.get("error"):
            cleanup_errors.append("监视窗口异常：" + monitor_status["error"])
        if async_stats:
            sent_commands, inferences = async_stats["sent_commands"], async_stats["inferences"]
        # 串口关闭后再等待后台推理退出，停止控制不等待 GPU。
        for worker in workers:
            worker.thread.join(timeout=5)
        log.write(json.dumps({"type": "end", "sent_commands": sent_commands, "inferences": inferences,
                              "seconds": round(time.monotonic() - start, 1), "stop_reason": exit_reason,
                              "cleanup_errors": cleanup_errors}) + "\n")
        terminal["written"] = True
        log.close()
        print("资源清理异常：" + "；".join(cleanup_errors) if cleanup_errors else "已释放扭矩并断开连接。", flush=True)
        print(f"共发送 {sent_commands} 次动作、{inferences} 次推理。", flush=True)
        print(f"运行日志：{log_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["async", "serial"], default="async")
    parser.add_argument("--scene", default="演示起始位置", help="用于人工评测分组的物体位置或场景名称")
    args = parser.parse_args()
    main(args.mode, args.scene)
