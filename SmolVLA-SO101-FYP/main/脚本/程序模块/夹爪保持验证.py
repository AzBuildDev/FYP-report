"""默认仅输出只读计划；显式执行时只启用夹爪约半秒保持当前位置，不运行模型。"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from 检查与释放电机 import MotorHealthGuard, release_bus
from 夹爪坐标映射 import attach_gripper_coordinates
from 选择推理模型 import ROOT

SERIAL = "5C82108242"


def hold_start_allowed(normalized):
    # 保持测试只需远离机械两端；首帧样本的最小值不是机械限位。
    return 40.0 <= normalized <= 75.0


def require_native_hold_coordinates(mapping):
    if mapping["shift"] != 0 or mapping["native_position"] < 0:
        raise RuntimeError("当前保持验证仅支持标定范围内的原生正坐标；未发送目标、未启用扭矩。"
                           "旧偏移方案的失败记录仍保留，不在此入口重试。")


def hold_current_gripper(bus, native, *, seconds=0.5):
    if not 0 < seconds <= 0.5:
        raise ValueError("保持验证最多 0.5 秒")
    if native < 0:
        raise RuntimeError("当前负坐标写入未通过设备验证，保持测试已禁用；未发送位置或启用扭矩")
    if bus.read("Operating_Mode", "gripper", normalize=False, num_retry=0) != 0:
        raise RuntimeError("夹爪不是位置模式，停止保持验证")
    MotorHealthGuard(bus).check(force=True)
    torque = bus.sync_read("Torque_Enable", normalize=False)
    if len(torque) != 6 or any(value != 0 for value in torque.values()):
        raise RuntimeError("验证前必须确认从臂六电机扭矩均为 0")
    fresh = bus.read("Present_Position", "gripper", normalize=False, num_retry=0)
    if abs(fresh - native) > 5:
        raise RuntimeError("夹爪还在被移动，取消保持验证")
    # 原始保持位置与当前反馈相同；不向其他关节发送目标。
    bus.sync_write("Goal_Position", {"gripper": fresh}, normalize=False)
    goal = bus.read("Goal_Position", "gripper", normalize=False, num_retry=0)
    if goal != fresh:
        raise RuntimeError(f"夹爪目标读回不一致：请求 {fresh}，读回 {goal}；未启用扭矩")
    samples = []
    try:
        bus.write("Torque_Enable", "gripper", 1, normalize=False, num_retry=0)
        if bus.read("Torque_Enable", "gripper", normalize=False, num_retry=0) != 1:
            raise RuntimeError("夹爪扭矩启用未确认")
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            if bus.read("Status", "gripper", normalize=False, num_retry=0) != 0:
                raise RuntimeError("夹爪报告故障，停止保持验证")
            position = bus.read("Present_Position", "gripper", normalize=False, num_retry=0)
            samples.append({"t": time.monotonic() - start, "native_position": position})
            if abs(position - fresh) > 15:
                raise RuntimeError("保持当前位置时反馈变化超过 15 个计数，立即停止")
            time.sleep(0.02)
    finally:
        bus.write("Torque_Enable", "gripper", 0, normalize=False, num_retry=1)
    return {"target_native": fresh, "goal_readback": goal, "samples": samples,
            "max_position_change": max(abs(sample["native_position"] - fresh) for sample in samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-hold", action="store_true", help="显式授权执行夹爪保持；默认只读")
    parser.add_argument("--repaired-coordinates", action="store_true", help="使用已核对端点的新正坐标候选标定，尚不替换默认标定")
    args = parser.parse_args()
    repaired_state = None
    repaired_path = ROOT / "运行环境/夹爪坐标修复.json"
    if args.repaired_coordinates:
        repaired_state = json.loads(repaired_path.read_text(encoding="utf-8"))
        if repaired_state.get("serial") != SERIAL or repaired_state.get("endpoints_verified") is not True:
            raise RuntimeError("新坐标尚未核对两个人工端点，未连接设备或启用扭矩")
        if args.execute_hold and repaired_state.get("hold_attempt_failed") is True:
            raise RuntimeError("新坐标保持曾失败，禁止自动重试；先重新检查")
    os.environ["HF_LEROBOT_CALIBRATION"] = str(ROOT / "标定备份")
    from serial.tools import list_ports
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower
    ports = {port.serial_number: port.device for port in list_ports.comports()}
    if SERIAL not in ports:
        raise RuntimeError("未检测到指定从臂")
    follower = SO101Follower(SO101FollowerConfig(port=ports[SERIAL], id="my_awesome_follower_arm"))
    bus = follower.bus
    if repaired_state is not None:
        from lerobot.motors import MotorCalibration
        candidate_path = Path(repaired_state["candidate_calibration"]).resolve()
        if not candidate_path.is_relative_to((ROOT / "运行记录").resolve()):
            raise RuntimeError("候选标定位置不正确")
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        if set(candidate) != set(bus.calibration):
            raise RuntimeError("候选标定关节列表不一致")
        for name, calibration in bus.calibration.items():
            if name != "gripper" and candidate[name] != vars(calibration):
                raise RuntimeError("候选标定修改了其他关节")
        g = candidate["gripper"]
        plan = repaired_state["plan"]
        if g != dict(id=6, drive_mode=0, **{k: plan[k] for k in ("homing_offset", "range_min", "range_max")}):
            raise RuntimeError("候选夹爪标定与修复记录不符")
        bus.calibration = {name: MotorCalibration(**value) for name, value in candidate.items()}
        follower.calibration = bus.calibration
    result = {"scope": "gripper_hold_only" if args.execute_hold else "read_only_gripper_hold_plan",
              "status": "failed", "motion_requested": args.execute_hold, "model_loaded": False,
              "using_repaired_coordinates": args.repaired_coordinates}
    try:
        bus.connect()
        MotorHealthGuard(bus).check(force=True)
        if not follower.is_calibrated:
            raise RuntimeError("标定不一致，停止")
        mapping = attach_gripper_coordinates(bus, ROOT, SERIAL)
        if args.execute_hold:
            require_native_hold_coordinates(mapping)
        native = bus.read("Present_Position", "gripper", normalize=False)
        normalized = bus._normalize({6: native})[6]
        inverse = bus._unnormalize({6: normalized})[6]
        result.update(mapping=mapping, normalized_position=normalized, native_position=native,
                      inverse_target=inverse, all_torque=bus.sync_read("Torque_Enable", normalize=False))
        result["ready_for_hold"] = hold_start_allowed(normalized)
        if args.execute_hold and not result["ready_for_hold"]:
            raise RuntimeError(f"请先手动恢复中间开度（40～75），当前 {normalized:.2f}；未启用扭矩")
        if abs(inverse - native) > 1:
            raise RuntimeError("读写坐标往返不一致，停止")
        if args.execute_hold:
            result["hold"] = hold_current_gripper(bus, native)
        result["status"] = "passed"
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        if args.execute_hold:
            cleanup = release_bus(bus)
        else:
            cleanup = []
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        result["cleanup_errors"] = cleanup
        if cleanup:
            result["status"] = "failed"
        report = ROOT / "运行记录" / f"夹爪保持验证_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if repaired_state is not None and args.execute_hold:
            repaired_state.update(hold_verified=result["status"] == "passed",
                                  hold_attempt_failed=result["status"] != "passed", hold_report=str(report),
                                  inference_allowed=False)
            repaired_path.write_text(json.dumps(repaired_state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print(f"报告：{report}", flush=True)
    if result["status"] != "passed":
        raise RuntimeError("保持验证未通过")
    # 原生正坐标保持的结果仅保存在本次报告；不能据此认证旧负坐标偏移方案。


if __name__ == "__main__":
    main()
