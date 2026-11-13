"""验证 STS 位置符号编码与原始夹爪范围检查；不连接串口。"""
import unittest
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from 检查控制输入 import validate_gripper_raw_position


class PositionEncodingTests(unittest.TestCase):
    def setUp(self):
        self.bus = FeetechMotorsBus("未使用的模拟端口", {"gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)},
                                   {"gripper": MotorCalibration(6, 0, 1925, 2046, 3492)})

    def test_observed_negative_position_is_not_a_large_positive_integer(self):
        self.assertEqual(self.bus._decode_sign("Present_Position", {6: 34042}), {6: -1274})

    def test_raw_hold_position_roundtrip_preserves_signed_value(self):
        encoded = self.bus._encode_sign("Goal_Position", {6: -1274})
        self.assertEqual(encoded, {6: 34042})
        self.assertEqual(self.bus._decode_sign("Present_Position", encoded), {6: -1274})

    def test_training_positive_coordinates_are_unchanged(self):
        for position in (2046, 2300, 2822, 3492, 4425):
            self.assertEqual(self.bus._decode_sign("Present_Position", {6: position}), {6: position})
            self.assertEqual(self.bus._encode_sign("Goal_Position", {6: position}), {6: position})

    def test_homing_offset_encoding_remains_unchanged(self):
        self.assertEqual(self.bus._decode_sign("Homing_Offset", {6: 2221}), {6: -173})

    def test_bad_raw_position_refuses_before_clipped_normalized_position(self):
        with self.assertRaisesRegex(RuntimeError, "原始位置异常"):
            validate_gripper_raw_position(-1274, 2046, 3492)
        with self.assertRaisesRegex(RuntimeError, "原始位置异常"):
            validate_gripper_raw_position(34042, 2046, 3492)
        validate_gripper_raw_position(2822, 2046, 3492)

    def test_real_sdk_packet_encoding_only_addresses_gripper_goal(self):
        from 离线复核夹爪指令 import reconstruct
        result = reconstruct()
        self.assertFalse(result["real_serial_opened"])
        self.assertEqual(result["start_address"], 42)
        self.assertEqual(result["data_length"], 2)
        self.assertEqual(result["motor_ids"], [6])
        self.assertEqual(result["position_wire_value"], 34105)
        self.assertTrue(result["checksum_valid"])
        self.assertFalse(result["contains_torque_register_write"])

    def test_unmapped_device_error_keeps_raw_error_code(self):
        self.assertIn("device_error=16", self.bus._format_device_error(16))
        self.assertIn("0x10", self.bus._format_device_error(16))
        self.assertIn("Overload", self.bus._format_device_error(32))
        self.assertIn("device_error=32", self.bus._format_device_error(32))


if __name__ == "__main__":
    unittest.main()
