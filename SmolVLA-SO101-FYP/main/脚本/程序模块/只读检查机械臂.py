"""只读检查两臂的位置、故障与扭矩状态；不写电机设置或运动目标。"""
import json
from pathlib import Path
from datetime import datetime

import scservo_sdk as scs
from serial.tools import list_ports

from lerobot.motors.feetech.feetech import patch_setPacketTimeout
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE
from lerobot.utils.encoding_utils import decode_sign_magnitude

root = Path(__file__).resolve().parents[2]
roles = {"5C82108314": "leader", "5C82108242": "follower"}
results = []
for device in list_ports.comports():
    if (device.vid, device.pid) != (0x1A86, 0x55D3):
        continue
    role = roles.get(device.serial_number, "unknown")
    if role == "unknown":
        continue
    port = scs.PortHandler(device.device)
    port.setPacketTimeout = patch_setPacketTimeout.__get__(port, scs.PortHandler)
    packet = scs.PacketHandler(0)
    arm = {"port": device.device, "serial": device.serial_number, "role_from_prior_usb_mapping": role, "motors": []}
    try:
        if not port.openPort() or not port.setBaudRate(1_000_000):
            raise RuntimeError("Cannot open port at 1 Mbps")
        for motor_id in range(1, 7):
            model, status, error = packet.ping(port, motor_id)
            motor = {"id": motor_id, "model": model, "ping_status": status, "error": error}
            if status == scs.COMM_SUCCESS:
                for key, address, length in [("position", 56, 2), ("torque_enabled", 40, 1), ("homing_offset_raw", 31, 2), ("range_min", 9, 2), ("range_max", 11, 2),
                                             ("fault_status", 65, 1), ("temperature", 63, 1), ("voltage_raw", 62, 1), ("load_raw", 60, 2),
                                             ("operating_mode", 33, 1), ("goal_position_raw", 42, 2), ("phase", 18, 1)]:
                    read = packet.read2ByteTxRx if length == 2 else packet.read1ByteTxRx
                    value, read_status, read_error = read(port, motor_id, address)
                    motor[key] = value if read_status == scs.COMM_SUCCESS else None
                    motor.setdefault("read_statuses", {})[key] = read_status
                    if read_error:
                        motor.setdefault("read_errors", {})[key] = read_error
                if model == 777 and motor.get("position") is not None:
                    motor["position_signed"] = decode_sign_magnitude(motor["position"], 15)
                    if motor.get("range_min") is not None and motor.get("range_max") is not None:
                        motor["position_within_calibration"] = motor["range_min"] <= motor["position_signed"] <= motor["range_max"]
                    if motor.get("goal_position_raw") is not None:
                        motor["goal_position_signed"] = decode_sign_magnitude(motor["goal_position_raw"], 15)
                if motor_id == 6 and model == 777:
                    # Only READ instructions: audit both grippers against the local control table.
                    parameters = {}
                    for name, (address, length) in STS_SMS_SERIES_CONTROL_TABLE.items():
                        read = packet.read2ByteTxRx if length == 2 else packet.read1ByteTxRx
                        value, read_status, read_error = read(port, motor_id, address)
                        parameters[name] = {"address": address, "length": length,
                                            "value": value if read_status == scs.COMM_SUCCESS else None,
                                            "communication_status": read_status, "device_error": read_error}
                    motor["full_parameters"] = parameters
            arm["motors"].append(motor)
            print(device.device, role, motor, flush=True)
    except Exception as exc:
        arm["connection_error"] = str(exc)
        print(device.device, "ERROR", exc, flush=True)
    finally:
        if getattr(port, "ser", None) is not None:
            port.closePort()
    results.append(arm)
output = root / "运行记录" / f"机械臂只读检查_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
success = len(results) == 2 and {arm["serial"] for arm in results} == set(roles) and all(
    not arm.get("connection_error") and len(arm["motors"]) == 6 and all(
    motor["ping_status"] == scs.COMM_SUCCESS and motor["error"] == 0 and motor.get("position") is not None
    and motor.get("fault_status") == 0 and not motor.get("read_errors")
    and all(status == scs.COMM_SUCCESS for status in motor.get("read_statuses", {}).values())
    for motor in arm["motors"]) for arm in results)
follower = [arm for arm in results if arm["serial"] == "5C82108242"]
disabled = len(follower) == 1 and len(follower[0]["motors"]) == 6 and all(
    motor.get("torque_enabled") == 0 and motor.get("read_statuses", {}).get("torque_enabled") == scs.COMM_SUCCESS
    and not motor.get("read_errors", {}).get("torque_enabled") for motor in follower[0]["motors"])
print("BOTH_ARMS_FAULT_FREE_READ_OK:", success, "FOLLOWER_TORQUE_DISABLED:", disabled, "REPORT:", output)
position_valid = all(motor.get("position_within_calibration", True) for arm in results for motor in arm["motors"])
print("POSITIONS_WITHIN_CALIBRATION:", position_valid)
raise SystemExit(0 if success and position_valid else 2)
