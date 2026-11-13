"""Operator-started teleoperation checks with normal or limited following."""
import argparse
import json
import os
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.environ["HF_LEROBOT_CALIBRATION"] = str(root / "标定备份")

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader

parser = argparse.ArgumentParser()
parser.add_argument("--follower-port", default="COM13")
parser.add_argument("--leader-port", default="COM12")
parser.add_argument("--duration", type=float, default=15.0)
parser.add_argument("--mode", choices=["limited", "normal"], default="limited")
parser.add_argument("--wait-for-start", action="store_true", help="Wait for the operator before configuring or enabling motors")
args = parser.parse_args()
if not 0 < args.duration <= 30:
    raise ValueError("This short test must last 1 to 30 seconds")
normal_mode = args.mode == "normal"
control_fps = 60 if normal_mode else 10
report_name = "正常同步测试" if normal_mode else "低速同步测试"
if args.wait_for_start:
    from serial.tools import list_ports
    print("主臂带动从臂：" + ("正常速度同步测试。" if normal_mode else "低速、限行程同步测试。"), flush=True)
    if normal_mode:
        print("已取消短测的单步限幅和小行程限制，使用项目默认电机配置。", flush=True)
        print("开始前把两臂摆成相近姿态，从臂将直接跟随主臂的关节位置。", flush=True)
    print(f"测试持续 {args.duration:g} 秒。开始后轻轻摆动主臂，按 Ctrl+C 可提前停止。", flush=True)
    while True:
        input("接好两臂 USB 和电源、准备好后，按回车检查并开始：")
        devices = {device.serial_number: device.device for device in list_ports.comports() if (device.vid, device.pid) == (0x1A86, 0x55D3)}
        if "5C82108314" not in devices or "5C82108242" not in devices:
            missing = [name for serial, name in [("5C82108314", "主臂"), ("5C82108242", "从臂")] if serial not in devices]
            print("未检测到：" + "、".join(missing) + "。本次没有开启电机，请检查 USB 连接后重试。", flush=True)
            continue
        args.leader_port = devices["5C82108314"]
        args.follower_port = devices["5C82108242"]
        print(f"主臂 {args.leader_port}，从臂 {args.follower_port}；正在核对标定。", flush=True)
        break
leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id="my_awesome_leader_arm"))
follower = SO101Follower(SO101FollowerConfig(port=args.follower_port, id="my_awesome_follower_arm", max_relative_target=None if normal_mode else 1))
samples = []
report = {"leader_port": args.leader_port, "follower_port": args.follower_port, "mode": args.mode, "control_fps_requested": control_fps, "duration_requested_s": args.duration, "max_step_normalized": None if normal_mode else 1, "max_total_travel_normalized": None if normal_mode else 8, "status": "started"}
try:
    # Verify calibration before any motor configuration or torque enable.
    leader.bus.connect()
    follower.bus.connect()
    if not leader.is_calibrated or not follower.is_calibrated:
        raise RuntimeError("Calibration mismatch: short test stopped without recalibration")
    leader.configure()
    follower.bus.disable_torque()
    hold = follower.bus.sync_read("Present_Position")
    follower.bus.sync_write("Goal_Position", hold)
    follower.configure()
    enabled = follower.bus.sync_read("Torque_Enable", normalize=False)
    report["follower_torque_enabled"] = enabled
    if set(enabled) != set(follower.bus.motors) or any(value != 1 for value in enabled.values()):
        raise RuntimeError("Not all six follower motors have torque enabled")
    print("从臂全部 6 个电机已启用。", flush=True)
    initial = follower.get_observation()
    initial_leader = leader.get_action()
    for countdown in range(3, 0, -1):
        print(f"{countdown} 秒后开始跟随，请准备轻轻摆动主臂…", flush=True)
        time.sleep(1)
    print("正在同步：请轻轻摆动主臂。", flush=True)
    start = time.perf_counter()
    while time.perf_counter() - start < args.duration:
        tick = time.perf_counter()
        action = leader.get_action()
        target = action if normal_mode else {key: max(initial[key] - 8, min(initial[key] + 8, value)) for key, value in action.items()}
        sent = follower.send_action(target)
        observed = follower.get_observation()
        samples.append({"elapsed_s": time.perf_counter() - start, "leader": action, "command": sent, "follower": observed})
        time.sleep(max(0, 1 / control_fps - (time.perf_counter() - tick)))
    report.update(status="passed", duration_actual_s=time.perf_counter() - start, loop_count=len(samples), initial_follower=initial, initial_leader=initial_leader)
    report["follower_travel_normalized"] = {key: max(sample["follower"][key] for sample in samples) - min(sample["follower"][key] for sample in samples) for key in initial}
    report["leader_travel_normalized"] = {key: max(sample["leader"][key] for sample in samples) - min(sample["leader"][key] for sample in samples) for key in initial_leader}
    report["motion_observed"] = max(report["follower_travel_normalized"].values()) > 0.3
    report["manual_input_observed"] = max(report["leader_travel_normalized"].values()) > 0.3
    report["manual_following_verified"] = report["motion_observed"] and report["manual_input_observed"]
except BaseException as exc:
    report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=repr(exc))
    raise
finally:
    cleanup_errors = []
    for arm in [follower, leader]:
        if arm.bus.is_connected:
            try:
                arm.bus.disconnect(disable_torque=True)
            except Exception as exc:
                cleanup_errors.append(str(exc))
                if getattr(arm.bus.port_handler, "ser", None) is not None:
                    arm.bus.port_handler.closePort()
    report["cleanup_errors"] = cleanup_errors
    (root / f"运行记录/{report_name}结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / f"运行记录/{report_name}采样.json").write_text(json.dumps(samples, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
