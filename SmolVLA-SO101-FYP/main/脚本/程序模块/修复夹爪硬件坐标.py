"""仅平移从臂夹爪零位与限位；不发送目标、不启用扭矩，默认只生成计划。"""
import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERIAL = "5C82108242"
CALIBRATION = ROOT / "标定备份/robots/so101_follower/my_awesome_follower_arm.json"


def make_plan(native, old_offset, lower, upper):
    normalized = (native + 4096 - lower) / (upper - lower) * 100
    if not 45 <= normalized <= 75:
        raise RuntimeError(f"夹爪须保持之前确认的半开姿态，当前换算开度 {normalized:.2f}")
    feedback_delta = 2048 - native
    training_delta = feedback_delta - 4096
    offset = old_offset - feedback_delta
    new_lower, new_upper = lower + training_delta, upper + training_delta
    if not -2047 <= offset <= 2047 or not 0 <= new_lower < 2048 < new_upper <= 4095:
        raise RuntimeError("候选零位/限位不在设备合法范围内")
    return {"old_native": native, "new_native": 2048, "feedback_delta": feedback_delta,
            "training_delta": training_delta, "normalized_position": normalized,
            "homing_offset": offset, "range_min": new_lower, "range_max": new_upper,
            "span": new_upper - new_lower}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    import scservo_sdk as scs
    from serial.tools import list_ports
    from lerobot.motors.feetech.feetech import patch_setPacketTimeout
    from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE as TABLE
    from lerobot.utils.encoding_utils import decode_sign_magnitude, encode_sign_magnitude

    ports = [p for p in list_ports.comports() if p.serial_number == SERIAL]
    if len(ports) != 1:
        raise RuntimeError("指定从臂 USB 未唯一找到")
    old_calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    old = old_calibration["gripper"]
    if old != dict(id=6, drive_mode=0, homing_offset=1925, range_min=2046, range_max=3492):
        raise RuntimeError("原标定已改变，不应用旧测量推导的修正")
    endpoint_path = ROOT / "运行环境/夹爪端点核对.json"
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8-sig"))
    expected_identity = dict(serial=SERIAL, motor_id=6, model="sts3215", homing_offset=1925,
                             drive_mode=0, range_min=2046, range_max=3492)
    if endpoint.get("identity") != expected_identity or endpoint.get("endpoints_verified") is not True:
        raise RuntimeError("端点实测记录不匹配")

    port = scs.PortHandler(ports[0].device)
    port.setPacketTimeout = patch_setPacketTimeout.__get__(port, scs.PortHandler)
    packet = scs.PacketHandler(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    directory = ROOT / "运行记录" / f"夹爪坐标修复_{stamp}"
    directory.mkdir()
    report = dict(status="failed", serial=SERIAL, apply_requested=args.apply,
                  torque_enable_sent=False, position_target_sent=False, writes=[])

    def save():
        (directory / "修复报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def read(motor_id, address, length):
        function = packet.read2ByteTxRx if length == 2 else packet.read1ByteTxRx
        value, communication, error = function(port, motor_id, address)
        if communication != scs.COMM_SUCCESS or error:
            raise RuntimeError(f"READ id={motor_id} address={address}: communication={communication}, device_error={error} (0x{error:02X})")
        return value

    def health():
        for motor_id in range(1, 7):
            if read(motor_id, 40, 1) != 0 or read(motor_id, 65, 1) != 0:
                raise RuntimeError(f"电机 {motor_id} 扭矩未关闭或存在故障，停止修改")

    def snapshot():
        return {str(i): {name: read(i, address, length) for name, (address, length) in TABLE.items()}
                for i in range(1, 7)}

    def write(address, value):
        # Exactly three allowed configuration registers, motor ID 6 only. No broadcast.
        if address not in (9, 11, 31) or not 0 <= value <= 65535:
            raise RuntimeError("写入不在夹爪坐标修复白名单内")
        health()
        operation = dict(id=6, address=address, length=2, value=value, confirmed=False)
        report["writes"].append(operation)
        save()
        communication, error = packet.write2ByteTxRx(port, 6, address, value)
        operation.update(communication_status=communication, device_error=error)
        save()
        if communication != scs.COMM_SUCCESS or error:
            raise RuntimeError(f"WRITE id=6 address={address}: communication={communication}, device_error={error} (0x{error:02X}); 不继续写入")
        if read(6, address, 2) != value:
            raise RuntimeError(f"寄存器 {address} 写入读回不一致")
        operation["confirmed"] = True
        save()

    try:
        if not port.openPort() or not port.setBaudRate(1_000_000):
            raise RuntimeError("无法打开从臂串口")
        health()
        before = snapshot()
        report["before"] = before
        g = before["6"]
        if (g["Model_Number"], g["Firmware_Major_Version"], g["Firmware_Minor_Version"],
            g["Operating_Mode"], g["Phase"], g["Homing_Offset"], g["Min_Position_Limit"],
            g["Max_Position_Limit"], g["Lock"]) != (777, 3, 11, 0, 12, 1925, 2046, 3492, 0):
            raise RuntimeError("设备身份、固件或原配置改变，停止修改")
        positions = [decode_sign_magnitude(read(6, 56, 2), 15) for _ in range(5)]
        if max(positions) - min(positions) > 3:
            raise RuntimeError("夹爪尚未静止")
        native = positions[-1]
        plan = make_plan(native, old["homing_offset"], old["range_min"], old["range_max"])
        report["plan"] = plan
        candidate = json.loads(json.dumps(old_calibration))
        candidate["gripper"].update({k: plan[k] for k in ("homing_offset", "range_min", "range_max")})
        shutil.copy2(CALIBRATION, directory / "原从臂标定.json")
        shutil.copy2(endpoint_path, directory / "原端点核对与锁定.json")
        (directory / "候选从臂标定.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["candidate_calibration"] = str(directory / "候选从臂标定.json")
        report["rollback_registers"] = {"31": g["Homing_Offset"], "9": g["Min_Position_Limit"], "11": g["Max_Position_Limit"]}
        save()
        if args.apply:
            write(31, encode_sign_magnitude(plan["homing_offset"], 11))
            time.sleep(0.05)
            observed = decode_sign_magnitude(read(6, 56, 2), 15)
            report["position_after_homing"] = observed
            if abs(observed - 2048) > 5:
                raise RuntimeError(f"修改零位后读数 {observed}，与预期 2048 不符；不继续修改限位或启用")
            write(9, plan["range_min"])
            write(11, plan["range_max"])
            health()
            after = snapshot()
            report["after"] = after
            mutable = {"6": {"Homing_Offset", "Min_Position_Limit", "Max_Position_Limit"}}
            dynamic = {"Present_Position", "Present_Velocity", "Present_Load", "Present_Voltage", "Present_Temperature", "Present_Current", "Moving", "Status"}
            unexpected = [(i, k, v, after[i][k]) for i, params in before.items() for k, v in params.items()
                          if k not in dynamic | mutable.get(i, set()) and after[i][k] != v]
            report["unexpected_configuration_changes"] = unexpected
            if unexpected:
                raise RuntimeError("检测到计划外配置变化，保持推理锁定")
            report["status"] = "configuration_applied_pending_endpoint_and_hold_validation"
            report["predicted_endpoint_positions"] = [dict(label=m["label"], native=m["native"] + plan["feedback_delta"],
                                                           normalized=m["normalized"]) for m in endpoint["measurements"]]
            (ROOT / "运行环境/夹爪坐标修复.json").write_text(json.dumps(dict(
                schema_version=1, status=report["status"], endpoints_verified=False, hold_verified=False,
                inference_allowed=False, serial=SERIAL, report=str(directory / "修复报告.json"),
                candidate_calibration=report["candidate_calibration"], plan=plan), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        else:
            report["status"] = "read_only_plan"
    except BaseException as exc:
        report["error"] = str(exc)
        raise
    finally:
        if getattr(port, "ser", None) is not None:
            port.closePort()
        save()
        print(json.dumps({k: v for k, v in report.items() if k not in ("before", "after")}, ensure_ascii=False, indent=2), flush=True)
        print("报告：", directory / "修复报告.json", flush=True)


if __name__ == "__main__":
    main()
