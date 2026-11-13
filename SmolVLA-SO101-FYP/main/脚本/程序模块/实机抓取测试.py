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
(ROOT / "运行记录").mkdir(exist_ok=True)
os.environ.update(
    HF_HOME=str(ROOT / "运行环境/缓存/huggingface"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    HF_DATASETS_OFFLINE="1", HF_LEROBOT_CALIBRATION=str(ROOT / "标定备份"),
    TOKENIZERS_PARALLELISM="false",
)

import numpy as np
import pyarrow.parquet as pq
import torch
from 检查控制输入 import bounded_target, bounds_including_start, validate_frame, restrict_gripper_to_demo, validate_gripper_raw_position, SlewTrackingGuard
from 异步动作控制 import run_async_control
from 等待测试开始 import wait_for_start
from 管理训练数据来源 import file_sha256
from 预热模型与检查时效 import warm_model, join_warm_workers
from 检查与释放电机 import MotorHealthGuard, release_bus
from 实机测试监视窗口 import run_monitor, run_embedded_monitor, wait_for_preview
from 双摄采集配置 import CAMERAS as CAPTURE_CAMERAS
from 选择推理模型 import add_selection_arguments, choose_model, load_local_policy, selection_from_args
from 推理运行设置 import ControlSettings, task_inference_settings, trajectory_trial_settings
from 夹爪坐标映射 import attach_gripper_coordinates, native_gripper_feedback_position, prepare_initial_gripper_hold
from 推理过程记录 import ProgressReporter, InputArchive
from 红蓝抓取任务 import TASKS
from 运行中目标切换 import TaskSwitch
from 相机运行诊断 import CameraDiagnostics

CAMERAS = {
    spec["feature"]: {**spec, "fps": spec["input_fps"]}
    for spec in CAPTURE_CAMERAS.values()
}
FOLLOWER_SERIAL = "5C82108242"
TASK = TASKS["RC"]
CONTROL_HZ = 20              # 控制频率
MAX_STEP_DELTA = 1.0         # 首次实机测试：每步归一化位置变化限量；不等同于角度
RANGE_MARGIN = 3.0           # 演示活动范围之外的归一化位置余量
MAX_TOTAL_TRAVEL = 3.0       # 首次实机测试：相对确认时姿态的单关节总行程上限
CHUNK_EXECUTE = 1            # 首次实机测试：每个预测块只执行一步
TEST_SECONDS = 3             # 首次实机测试最长秒数
RUN_KIND = "real_hardware"   # 模拟入口自检必须显式替换该标记
MOTOR_HEALTH_INTERVAL = 1.0  # 主线程每秒读取全部六个电机故障状态

latest_frames = {key: {"time": 0.0, "image": None} for key in CAMERAS}
frame_lock = threading.Lock()
camera_stop = threading.Event()
camera_diagnostics = None
user_stop = threading.Event()
monitor_status = {"text": "准备中"}


def monitor_loop(preview_directory=None):
    """Tk 在独立进程的主线程运行；父进程只传预览、文字和停止请求。"""
    try:
        if preview_directory is None:
            run_monitor(ROOT, get_monitor_frame, camera_stop, user_stop, lambda: monitor_status["text"])
        else:
            run_embedded_monitor(ROOT, preview_directory, get_monitor_frame, camera_stop, user_stop,
                                 lambda: monitor_status["text"])
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


def camera_callback(camera, kind, value):
    if camera_diagnostics is not None:
        camera_diagnostics.update(camera, kind, value)
    if kind == "frame":
        stamp, image = value
        with frame_lock:
            latest_frames[camera]["time"], latest_frames[camera]["image"] = stamp, image


def get_frame():
    with frame_lock:
        frames = {key: dict(state) for key, state in latest_frames.items()}
    stamp = min(state["time"] for state in frames.values())
    if camera_diagnostics is not None and not camera_diagnostics.closed:
        return stamp, camera_diagnostics.read(frames)
    return stamp, {key: state["image"] for key, state in frames.items()}


def get_monitor_frame():
    with frame_lock:
        front = latest_frames["observation.images.front"]
        wrist = latest_frames["observation.images.wrist"]
        if front["image"] is None or wrist["image"] is None:
            return 0.0, None
        # 仅把监视窗口的副本补边成相同尺寸；模型仍接收各路原始分辨率。
        from PIL import Image, ImageOps
        previews = [np.asarray(ImageOps.pad(Image.fromarray(state["image"]), (640, 480)))
                    for state in (front, wrist)]
        return min(front["time"], wrist["time"]), np.hstack(previews)


def release_follower(follower):
    return release_bus(follower.bus)


def close_camera(threads):
    camera_stop.set()
    for thread in threads:
        thread.join(timeout=2)


def make_predict(policy, image_keys=None, task=TASK):
    if not isinstance(task, TaskSwitch) and task not in TASKS.values():
        raise ValueError("请选择规范的抓红或抓蓝任务")
    image_keys = list(policy.config.image_features) if image_keys is None else image_keys
    image_keys = [image_keys] if isinstance(image_keys, str) else list(image_keys)

    def predict(state, images):
        if not isinstance(images, dict):
            if len(image_keys) != 1:
                raise ValueError("双摄模型需要桌面和夹爪两路图像")
            images = {image_keys[0]: images}
        if set(images) != set(image_keys):
            raise ValueError(f"实时相机字段 {list(images)} 与模型输入 {image_keys} 不一致")
        batch = {
            "observation.state": torch.tensor(state, dtype=torch.float32, device="cuda")[None],
            "task": [task.task if isinstance(task, TaskSwitch) else task],
        }
        batch.update({key: torch.from_numpy(images[key][:, :, ::-1].copy()).permute(2, 0, 1)[None].to(
            "cuda", torch.float32) / 255 for key in image_keys})
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
        return chunk[0].float().cpu().numpy()
    return predict


def main(mode="async", scene="演示起始位置", selection=None, *, settings=None, window_start=False, stop_file=None,
         preview_directory=None, task=TASK):
    if task not in TASKS.values():
        raise ValueError("请选择规范的抓红或抓蓝任务")
    if not scene.strip():
        raise ValueError("测试场景不能为空")
    selection = selection if selection is not None else choose_model(ROOT)
    if selection is None:
        print("已取消模型选择，未连接硬件。", flush=True)
        return
    user_stop.clear()
    camera_stop.clear()
    monitor_status["error"] = ""
    settings = settings or ControlSettings(TEST_SECONDS, CHUNK_EXECUTE, MAX_TOTAL_TRAVEL)
    watcher_end = threading.Event()
    watcher = None
    if stop_file is not None:
        stop_file = Path(stop_file).resolve()
        if not stop_file.is_relative_to((ROOT / "运行环境/临时文件").resolve()):
            raise ValueError("推理停止请求文件必须在项目临时目录内")

        def watch_stop():
            while not watcher_end.is_set():
                if stop_file.exists():
                    user_stop.set()
                    return
                watcher_end.wait(0.03)

        watcher = threading.Thread(target=watch_stop, daemon=True)
        watcher.start()
    try:
        with ExitStack() as resources:
            resources.callback(camera_stop.set)
            if not user_stop.is_set():
                if preview_directory is None:
                    run_test(resources, mode, scene.strip(), selection, settings, window_start, task=task)
                else:
                    run_test(resources, mode, scene.strip(), selection, settings, window_start, preview_directory, task=task)
    finally:
        watcher_end.set()
        if watcher is not None:
            watcher.join(1)


def run_test(resources, mode, scene, selection, settings, window_start=False, preview_directory=None, *, task=TASK):
    global camera_diagnostics
    camera_diagnostics = None
    from serial.tools import list_ports
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    from 相机测试窗口 import named_capture

    model_dir, dataset_root = selection.model, selection.dataset
    test_seconds, chunk_execute, max_total_travel = settings.seconds, settings.chunk_execute, settings.max_total_travel
    duration_limit = float("inf") if test_seconds is None else test_seconds
    duration_label = "持续运行" if test_seconds is None else f"{test_seconds:g} 秒"
    step_delta = settings.max_step_delta

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = ROOT / "运行记录" / f"实机抓取测试_{run_id}.jsonl"
    log = resources.enter_context(log_path.open("w", encoding="utf-8", buffering=1))
    print(f"运行日志：{log_path}", flush=True)
    terminal = {"written": False, "event": None, "cleanup_errors": []}

    def finish_preparation():
        if not terminal["written"] and not log.closed:
            exception = sys.exc_info()[1]
            event = terminal["event"] or {"type": "preparation_end", "sent_commands": 0,
                                          "error": str(exception) if exception else "准备阶段退出，未启动控制"}
            if terminal["cleanup_errors"]:
                event["cleanup_errors"] = terminal["cleanup_errors"]
                print("资源清理异常：" + "；".join(terminal["cleanup_errors"]), flush=True)
            log.write(json.dumps(event, ensure_ascii=False) + "\n")

    def cancel_preparation():
        terminal["event"] = {"type": "cancelled_before_start", "sent_commands": 0}

    resources.callback(finish_preparation)
    warm_workers = []
    # 后注册的串口、相机先释放，再有限等待不接触硬件的预热线程。
    resources.callback(join_warm_workers, warm_workers)
    monitor_threads = []
    resources.callback(lambda: terminal["cleanup_errors"].extend(close_monitors(monitor_threads)))
    log.write(json.dumps({"type": "配置", "task": task, "control_hz": CONTROL_HZ, "max_step_delta": step_delta,
                          "target_strategy": settings.target_strategy,
                          "mode": mode, "scene": scene, "run_kind": RUN_KIND, "run_id": run_id,
                          "start_pose_policy": "current_pose", "range_margin": RANGE_MARGIN,
                          "max_total_travel": max_total_travel, "control_profile": settings.profile,
                          "start_requested_in_window": window_start,
                          "chunk_execute": chunk_execute, "test_seconds": test_seconds}) + "\n")
    log.write(json.dumps({"type": "model_identity", "model_path": str(model_dir),
                          "dataset_path": str(dataset_root), "processor_path": str(selection.processor),
                          "checkpoint_sha256": file_sha256(model_dir / "model.safetensors")}, ensure_ascii=False) + "\n")

    print("【1/5】正在加载微调模型……", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU")
    torch.manual_seed(42)
    print(f"模型：{model_dir}\n演示：{dataset_root}\n处理器：{selection.processor}", flush=True)
    policy = load_local_policy(model_dir, selection.processor)
    if user_stop.is_set():
        cancel_preparation()
        print("已停止，未连接从臂。", flush=True)
        return
    image_keys = list(policy.config.image_features)
    if image_keys != list(CAMERAS):
        raise RuntimeError(f"模型图像输入 {image_keys} 不是当前双摄配置 {list(CAMERAS)}")
    print(f"任务指令：{task}", flush=True)
    def record_task(event):
        log.write(json.dumps(event, ensure_ascii=False) + "\n")
        log.flush()
        print("目标状态：" + json.dumps(event, ensure_ascii=False), flush=True)
    task_switch = TaskSwitch(task, preview_directory if mode == "async" else None, record_task)
    evidence = InputArchive(log_path.with_suffix(""), make_predict(policy, image_keys, task_switch))
    predict = evidence
    print(f"推理输入存档：{evidence.directory}", flush=True)
    log.write(json.dumps({"type": "input_archive", "directory": str(evidence.directory),
                          "interval_s": evidence.interval}, ensure_ascii=False) + "\n")

    print("【2/5】正在计算安全活动范围……", flush=True)
    tables = sorted(dataset_root.glob("data/chunk-*/*.parquet"))
    data = [pq.read_table(t, columns=["observation.state", "action"]).to_pydict() for t in tables]
    states = np.concatenate([np.asarray(t["observation.state"], dtype=np.float32) for t in data])
    demo_actions = np.concatenate([np.asarray(t["action"], dtype=np.float32) for t in data])
    if states.ndim != 2 or states.shape[1] != 6 or not np.isfinite(states).all():
        raise ValueError("演示关节数据必须为有限数值的六维状态，未连接硬件。")
    lo, hi = states.min(axis=0) - RANGE_MARGIN, states.max(axis=0) + RANGE_MARGIN
    if demo_actions.shape != states.shape or not np.isfinite(demo_actions).all():
        raise ValueError("演示动作数据必须与六维状态对应且数值有限，未连接硬件。")
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    gripper_index = info["features"]["action"]["names"].index("gripper.pos")
    gripper_lower = max(0., float(min(states[:, gripper_index].min(), demo_actions[:, gripper_index].min())))
    gripper_upper = min(100., float(max(states[:, gripper_index].max(), demo_actions[:, gripper_index].max())))
    if user_stop.is_set():
        cancel_preparation()
        return

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
    progress = ProgressReporter(log, names)
    if info["features"]["action"]["names"] != names or info["features"]["observation.state"]["names"] != names:
        raise RuntimeError("演示数据和从臂关节顺序不一致，已停止。")
    def log_gripper_diagnostic(event):
        log.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"      夹爪原始位置：{event['signed_position']}；"
              f"标定范围：{event['calibration_min']}～{event['calibration_max']}。", flush=True)

    gripper_coordinates = attach_gripper_coordinates(follower.bus, ROOT, FOLLOWER_SERIAL,
                                                      require_hold=True, on_diagnostic=log_gripper_diagnostic)
    log.write(json.dumps({"type": "gripper_coordinate_mapping", **gripper_coordinates}, ensure_ascii=False) + "\n")
    if gripper_coordinates.get("closed_endpoint_correction"):
        print(f"      夹爪闭合端点偏差 {gripper_coordinates['closed_endpoint_correction']} 个计数，"
              "按零开度读取；电机目标仍使用合法标定范围，未修改标定。", flush=True)
    observation = follower.get_observation()
    last_sent = np.array([observation[k] for k in names], dtype=np.float32)
    print(f"      从臂 {ports[FOLLOWER_SERIAL]} 在线，当前关节：{np.round(last_sent, 1).tolist()}", flush=True)
    log.write(json.dumps({"type": "gripper_start_check", "phase": "connection",
                          "initial": float(last_sent[gripper_index]), "lower": gripper_lower,
                          "upper": gripper_upper, "expand_to_start": False}) + "\n")
    print(f"      夹爪当前开度：{last_sent[gripper_index]:.2f}；"
          f"本次演示允许：{gripper_lower:.2f}～{gripper_upper:.2f}。", flush=True)
    # 启动姿态不符时立即说明原因；确认后仍会重新检查，防止期间被移动。
    restrict_gripper_to_demo(last_sent, lo, hi, gripper_index, gripper_lower, gripper_upper)

    print("【4/5】正在开启桌面和夹爪相机……", flush=True)
    with frame_lock:
        for state in latest_frames.values():
            state.update(time=0., image=None)
    camera_diagnostics = CameraDiagnostics(CAMERAS, log_path.with_suffix("") / "相机日志", log)
    def finish_camera_diagnostics(diagnostics=camera_diagnostics):
        exception = sys.exc_info()[1]
        if exception is not None and not log.closed:
            diagnostics.fault(exception)
        diagnostics.close()
    resources.callback(finish_camera_diagnostics)
    log.write(json.dumps({"type": "camera_logs", "directory": str(camera_diagnostics.directory)}, ensure_ascii=False) + "\n")
    print(f"      逐路相机日志：{camera_diagnostics.directory}", flush=True)
    camera_threads = []
    resources.callback(close_camera, camera_threads)
    for key, spec in CAMERAS.items():
        thread = threading.Thread(
            target=named_capture,
            args=(spec["index"], spec["fourcc"], spec["fps"],
                  lambda kind, value, camera=key: camera_callback(camera, kind, value),
                  camera_stop, spec["device"], None),
            kwargs={"width": spec["width"], "height": spec["height"],
                    "log_path": camera_diagnostics.directory / f"{spec['label']}_驱动.log"},
            daemon=True,
            name=key,
        )
        camera_threads.append(thread)
        thread.start()
        if len(camera_threads) == 1:
            time.sleep(1.2)
    deadline = time.monotonic() + 10
    last_camera_error = None
    while time.monotonic() < deadline:
        if user_stop.is_set():
            cancel_preparation()
            return
        stamp, images = get_frame()
        try:
            validate_frame(stamp, images, time.monotonic(), max_age=1.0)
            break
        except RuntimeError as exc:
            last_camera_error = exc
        time.sleep(0.1)
    else:
        camera_stop.set()
        if last_camera_error is not None:
            raise last_camera_error
        raise RuntimeError("相机没有提供有效画面，已停止。")
    print("      两路相机画面正常。", flush=True)
    monitor_thread = threading.Thread(target=monitor_loop, args=(preview_directory,) if preview_directory else (), daemon=True)
    monitor_threads.append(monitor_thread)
    monitor_thread.start()
    monitor_status["text"] = "左：桌面相机 | 右：夹爪相机\n正在预热模型；尚未启用控制，可停止"
    if preview_directory is not None:
        print("      正在等待推理窗口显示双摄画面……", flush=True)
        if not wait_for_preview(preview_directory, user_stop):
            cancel_preparation()
            return
        print("      双摄画面已在推理窗口显示。", flush=True)

    print("      正在预热模型并检查推理时效（不发送动作）……", flush=True)
    stamp, images = get_frame()
    validate_frame(stamp, images, time.monotonic())
    try:
        warm_report = warm_model(predict, last_sent, images, user_stop,
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
    evidence.phase = "control"
    monitor_status["text"] = ("左：桌面相机 | 右：夹爪相机\n预热通过，即将按窗口的启动请求运行"
                              if window_start else "左：桌面相机 | 右：夹爪相机\n预热通过，等待确认开始；预热动作已丢弃")

    print("【5/5】正在核对启动姿态与最终设备状态……", flush=True)
    print("=" * 60, flush=True)
    print(f"控制设置：从当前位置开始，{'相邻发送指令' if settings.target_strategy == 'trajectory' else '相对当前反馈'}每步最多变化 {step_delta:g} 个归一化单位。", flush=True)
    print(f"单关节总行程不超过 {max_total_travel:g}。" if max_total_travel is not None
          else "任务推理：在演示活动范围与标定行程内运行。", flush=True)
    print("身体关节范围包含当前姿态；夹爪起点必须在演示范围内，超出时拒绝启用。", flush=True)
    print(f"运行时长：{duration_label}。运行期间从臂会自行运动，请全程看护。", flush=True)
    print("已收到窗口的“启动推理”请求，最终检查通过后开始；点击“停止推理”或关闭窗口可停止。" if window_start
          else "按 Ctrl+C 或监视窗口的“停止测试”停止并释放扭矩。确认周围安全后，按回车开始；等待期间 Esc 或关闭窗口可取消。", flush=True)
    print("异步模式每个控制节拍处理停止请求；串行模式需等当前推理结束。", flush=True)
    print("=" * 60, flush=True)
    try:
        confirmed = not user_stop.is_set() if window_start else wait_for_start(user_stop)
    except (KeyboardInterrupt, EOFError):
        confirmed = False
    if not confirmed or user_stop.is_set():
        cancel_preparation()
        print("已取消，未启用扭矩。", flush=True)
        return
    raw_gripper = follower.bus.read("Present_Position", "gripper", normalize=False)
    calibration = follower.bus.calibration["gripper"]
    effective_training_position = None
    try:
        if gripper_coordinates["shift"] == 0:
            effective_training_position = native_gripper_feedback_position(raw_gripper, calibration)
        else:
            validate_gripper_raw_position(raw_gripper + gripper_coordinates["shift"], calibration.range_min, calibration.range_max)
            effective_training_position = raw_gripper + gripper_coordinates["shift"]
    finally:
        log.write(json.dumps({"type": "gripper_raw_diagnostic", "phase": "final_confirmation",
                              "signed_position": raw_gripper,
                              "raw_training_position": raw_gripper + gripper_coordinates["shift"],
                              "training_position": effective_training_position,
                              "calibration_min": calibration.range_min, "calibration_max": calibration.range_max}) + "\n")
    stamp, images = get_frame()
    validate_frame(stamp, images, time.monotonic())
    observation = follower.get_observation()
    last_sent = np.array([observation[k] for k in names], dtype=np.float32)
    lo, hi = bounds_including_start(last_sent, lo, hi, RANGE_MARGIN)
    lo, hi = restrict_gripper_to_demo(last_sent, lo, hi, gripper_index, gripper_lower, gripper_upper)
    log.write(json.dumps({"type": "gripper_limits", "lower": gripper_lower, "upper": gripper_upper,
                          "initial": float(last_sent[gripper_index]), "expand_to_start": False}) + "\n")
    if max_total_travel is not None:
        lo = np.maximum(lo, last_sent - max_total_travel)
        hi = np.minimum(hi, last_sent + max_total_travel)
    target_controller = None
    if settings.target_strategy == "trajectory":
        def record_gripper_event(event):
            log.write(json.dumps({**event, "t": time.monotonic() - start}, ensure_ascii=False) + "\n")
            if event["event"] == "limited_hold":
                print(f"夹爪闭合受阻，疑似接触物体：开度 {event['contact_position']:.2f}，"
                      f"闭合目标不再低于 {event['hold_floor']:.2f}，按每步限速逐渐卸载；"
                      "其余关节继续执行模型动作，过载与跟随保护仍有效。", flush=True)
            else:
                print("模型要求张开夹爪，结束接触保持，按限速释放。", flush=True)

        tracking_limits = np.clip(np.percentile(np.abs(demo_actions - states), 95, axis=0) * 1.25, 2., 10.).astype(np.float32)
        tracking_limits[gripper_index] = min(3., tracking_limits[gripper_index])
        target_controller = SlewTrackingGuard(last_sent, lo, hi, step_delta, tracking_limits, names=names,
                                              gripper_index=gripper_index, on_gripper_event=record_gripper_event)
        log.write(json.dumps({"type": "tracking_guard", "tracking_limits": tracking_limits.tolist(),
                              "stall_timeout_s": target_controller.stall_timeout,
                              "max_step_delta": step_delta, "strategy": "previous_sent_command",
                              "gripper_contact": target_controller.gripper_contact.settings()}) + "\n")
        print(f"独立跟随误差上限：{np.round(tracking_limits, 2).tolist()}；持续卡住 {target_controller.stall_timeout:g} 秒即停止。", flush=True)
        print("夹爪闭合受阻时先限制继续夹紧；疑似接触保持不限时，模型要求张开时释放。", flush=True)
    # 读取串口可能耗时：紧邻扭矩配置再次核对画面与停止请求。
    stamp, images = get_frame()
    validate_frame(stamp, images, time.monotonic())
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
        raw_hold = follower.bus.sync_read("Present_Position", normalize=False)
        hold = prepare_initial_gripper_hold(raw_hold, calibration)
        if hold["gripper"] != raw_hold["gripper"]:
            print(f"闭合端点小偏差：实测 {raw_hold['gripper']}，预装合法夹爪目标 {hold['gripper']}；"
                  "启用时可能有轻微张开修正。", flush=True)
        log.write(json.dumps({"type": "initial_raw_hold", "positions": {key: float(value) for key, value in hold.items()},
                              "measured_positions": {key: float(value) for key, value in raw_hold.items()},
                              "gripper_endpoint_correction": float(hold["gripper"] - raw_hold["gripper"]),
                              "normalize": False}) + "\n")
        stamp, images = get_frame()
        validate_frame(stamp, images, time.monotonic())
        if user_stop.is_set():
            exit_reason = "user_stop"
            return
        follower.bus.sync_write("Goal_Position", hold, normalize=False)
        actual_hold_goal = follower.bus.read("Goal_Position", "gripper", normalize=False, num_retry=0)
        if actual_hold_goal != hold["gripper"]:
            raise RuntimeError(f"夹爪启动保持目标读回不一致：请求 {hold['gripper']}，读回 {actual_hold_goal}；未启用扭矩")
        follower.configure()
        torque = follower.bus.sync_read("Torque_Enable", normalize=False)
        if len(torque) != 6 or any(v != 1 for v in torque.values()):
            raise RuntimeError("六个电机没有全部启用，已停止。")
        print("开始闭环控制，持续运行，直到点击停止或发生故障。" if test_seconds is None
              else f"开始闭环控制，{test_seconds:g} 秒后自动停止。", flush=True)
        log.write(json.dumps({"type": "start", "initial_state": last_sent.tolist(),
                              "control_lower": lo.tolist(), "control_upper": hi.tolist()}) + "\n")

        if mode == "async":
            last_observed = last_sent.copy()

            def read_state():
                nonlocal last_observed
                health.check()
                observation = follower.get_observation()
                last_observed = np.array([observation[k] for k in names], dtype=np.float32)
                return last_observed.copy()

            def send_target(target):
                # 位置命令可能清除设备保护标志，发送前先读取所有故障状态。
                health.check(force=True)
                stamp, image = get_frame()
                validate_frame(stamp, image, time.monotonic())
                if user_stop.is_set():
                    raise KeyboardInterrupt
                actual = follower.send_action({k: float(v) for k, v in zip(names, target)})
                return actual

            latest_inference_s = 0.
            def record_event(event):
                nonlocal latest_inference_s
                if event["type"] == "control_step":
                    event = event.copy()
                    actual = event.pop("sent_action")
                    event["sent_target"] = [float(actual[k]) for k in names]
                    if target_controller is not None:
                        event["gripper_control"] = target_controller.gripper_status
                    progress.action(event, inference_s=latest_inference_s,
                                    inferences=async_stats.get("inferences", 0), fault_status=health.last_statuses)
                else:
                    log.write(json.dumps(event, ensure_ascii=False) + "\n")
                if event["type"] == "async_chunk":
                    latest_inference_s = event["inference_s"]
                monitor_status["text"] = (f"异步运行 {time.monotonic() - start:.0f} 秒 / {duration_label} | "
                                          f"推理 {async_stats.get('inferences', 0)} 次 | "
                                          f"已发送 {async_stats.get('sent_commands', 0)} 步 | 停止测试 / Ctrl+C")

            run_async_control(predict, read_state, get_frame, send_target,
                              user_stop, lo, hi, duration=None if test_seconds is None else max(0.001, test_seconds - (time.monotonic() - start)),
                              hz=CONTROL_HZ, max_delta=step_delta, execute_limit=chunk_execute,
                              on_event=record_event, stats=async_stats, workers=workers, target_controller=target_controller, task_switch=task_switch)
        else:
            while time.monotonic() - start < duration_limit:
                if user_stop.is_set():
                    break
                health.check()
                stamp, images = get_frame()
                validate_frame(stamp, images, time.monotonic())
                observation = follower.get_observation()
                state = np.array([observation[k] for k in names], dtype=np.float32)
                infer_start = time.perf_counter()
                actions = predict(state, images)
                inference_duration = time.perf_counter() - infer_start
                inferences += 1
                monitor_status["text"] = (f"运行中 {time.monotonic() - start:.0f} 秒 / {duration_label} | "
                                          f"推理 {inferences} 次（{time.perf_counter() - infer_start:.1f} 秒）| "
                                          f"已发送 {sent_commands} 步 | Ctrl+C 停止")
                if not np.isfinite(actions).all():
                    raise RuntimeError("模型输出无效数值，立即停止。")
                log.write(json.dumps({"type": "chunk", "t": round(time.monotonic() - start, 2),
                                      "inference_s": round(time.perf_counter() - infer_start, 2),
                                      "state": state.tolist(), "first_action": actions[0].tolist()}) + "\n")

                for model_target in actions[:chunk_execute]:
                    if user_stop.is_set() or time.monotonic() - start >= duration_limit:
                        break
                    health.check(force=True)
                    stamp, live_image = get_frame()
                    validate_frame(stamp, live_image, time.monotonic())
                    observation = follower.get_observation()
                    current = np.array([observation[k] for k in names], dtype=np.float32)
                    if target_controller is not None:
                        target_controller.observe(current, time.monotonic())
                    target = (target_controller.target(model_target, current) if target_controller is not None
                              else bounded_target(model_target, current, lo, hi, step_delta))
                    step_start = time.monotonic()
                    actual = follower.send_action({k: float(v) for k, v in zip(names, target)})
                    sent_commands += 1
                    last_sent = np.array([actual[k] for k in names], dtype=np.float32)
                    if target_controller is not None:
                        target_controller.commit(last_sent)
                    progress.action({"type": "action", "t": time.monotonic() - start,
                                     "observed_state": current.tolist(), "model_target": model_target.tolist(),
                                     "target": target.tolist(), "sent_target": last_sent.tolist(),
                                     "gripper_control": target_controller.gripper_status if target_controller else None},
                                    inference_s=inference_duration, inferences=inferences,
                                    fault_status=health.last_statuses)
                    elapsed = time.monotonic() - step_start
                    if elapsed < 1 / CONTROL_HZ:
                        time.sleep(1 / CONTROL_HZ - elapsed)
        print("收到停止请求。" if user_stop.is_set() else f"{test_seconds:g} 秒测试时间到。", flush=True)
        if user_stop.is_set():
            exit_reason = "user_stop"
    except KeyboardInterrupt:
        exit_reason = "user_stop"
        print("\n收到 Ctrl+C，立即停止。", flush=True)
        log.write(json.dumps({"type": "interrupted", "t": round(time.monotonic() - start, 2)}) + "\n")
    except Exception as exc:
        exit_reason = "error"
        camera_diagnostics.fault(exc)
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
        log.write(json.dumps({"type": "control_diagnostics", **progress.summary()}) + "\n")
        log.write(json.dumps({"type": "input_archive_end", "samples": evidence.counter,
                              "errors": evidence.errors}, ensure_ascii=False) + "\n")
        log.write(json.dumps({"type": "end", "sent_commands": sent_commands, "inferences": inferences,
                              "seconds": round(time.monotonic() - start, 1), "stop_reason": exit_reason,
                              "cleanup_errors": cleanup_errors}) + "\n")
        terminal["written"] = True
        log.close()
        print("资源清理异常：" + "；".join(cleanup_errors) if cleanup_errors else "已释放扭矩并断开连接。", flush=True)
        print(f"共发送 {sent_commands} 次动作、{inferences} 次推理。", flush=True)
        print(f"模型输入画面：{evidence.directory}（共 {evidence.counter} 组）", flush=True)
        if evidence.errors:
            print("输入画面存档异常：" + "；".join(evidence.errors), flush=True)
        print(f"运行日志：{log_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["async", "serial"], default="async")
    parser.add_argument("--target-object", choices=tuple(TASKS), default="RC", help="RC 抓红方块，BC 抓蓝方块")
    parser.add_argument("--scene", default="演示起始位置", help="用于人工评测分组的物体位置或场景名称")
    parser.add_argument("--window-start", action="store_true", help="由推理窗口的启动按钮请求运行，无需再按终端回车")
    parser.add_argument("--stop-file", type=Path, help="推理窗口使用的本机停止请求文件")
    parser.add_argument("--preview-directory", type=Path, help="向推理窗口发送双摄画面的临时目录")
    parser.add_argument("--task-inference", action="store_true", help="按演示范围连续运行，而非短行程检查")
    parser.add_argument("--trajectory-trial", action="store_true", help="轨迹限速首轮实测，固定3秒、起点总行程8、每步0.5")
    parser.add_argument("--seconds", type=float, default=30, help="任务推理秒数，无上限；0 表示持续运行直到停止")
    add_selection_arguments(parser)
    args = parser.parse_args()
    if len(sys.argv) == 1:
        from 实机推理窗口 import main as window_main
        window_main()
        sys.exit(0)
    selected = selection_from_args(args, ROOT)
    if selected is not None:
        main(args.mode, args.scene, selected, settings=trajectory_trial_settings() if args.trajectory_trial else task_inference_settings(args.seconds) if args.task_inference else None,
             window_start=args.window_start, stop_file=args.stop_file, preview_directory=args.preview_directory,
             task=TASKS[args.target_object])
    else:
        print("已取消模型选择，未连接硬件。", flush=True)
