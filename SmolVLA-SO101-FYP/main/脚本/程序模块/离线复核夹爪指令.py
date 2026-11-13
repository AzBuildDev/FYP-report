"""使用实际 SDK 和替代串口重建目标数据包，不打开任何真实设备。"""
import json
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus

ROOT = Path(__file__).resolve().parents[2]


class CapturePort:
    is_open = True
    is_using = False

    def __init__(self):
        self.packets = []

    def clearPort(self):
        pass

    def writePort(self, packet):
        self.packets.append(list(packet))
        return len(packet)


def reconstruct():
    bus = FeetechMotorsBus("仅使用替代串口", {"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)},
                           {"gripper": MotorCalibration(6, 0, 1925, 2046, 3492)})
    capture = CapturePort()
    bus.port_handler = capture
    bus.sync_writer.port = capture
    bus.sync_write("Goal_Position", {"gripper": -1337}, normalize=False)
    packet = capture.packets[0]
    return {"scope": "offline_packet_reconstruction", "real_serial_opened": False,
            "original_wire_traffic_captured": False,
            "requested_signed_position": -1337, "packet_hex": " ".join(f"{value:02X}" for value in packet),
            "instruction": packet[4], "start_address": packet[5], "data_length": packet[6],
            "motor_ids": [packet[7]], "position_wire_value": packet[8] | (packet[9] << 8),
            "checksum_valid": (sum(packet[2:]) & 0xFF) == 0xFF,
            "contains_torque_register_write": packet[5] <= 40 < packet[5] + packet[6],
            "current_stored_position_limits": [2046, 3492],
            "signed_target_inside_stored_limits": False,
            "limits": "重建结果证明当前代码和 SDK 的编码路径，不是当时实际串口抓包；无法恢复原日志漏记的首次错误码，不能确定过载物理原因。"}


def main():
    result = reconstruct()
    path = ROOT / "运行记录/夹爪失败指令离线复核.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
