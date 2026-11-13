"""回归本机圈数偏差，验证读取与发送双向映射；不连接电机。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from 夹爪坐标映射 import (attach_gripper_coordinates, native_gripper_feedback_position,
                       prepare_initial_gripper_hold, closed_endpoint_tolerance)
from 选择推理模型 import ROOT


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        self.bus = FeetechMotorsBus("模拟端口", {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100)}, {
            "shoulder_pan": MotorCalibration(1, 0, -173, 708, 3314),
            "gripper": MotorCalibration(6, 0, 1925, 2046, 3492)})
        temp = tempfile.TemporaryDirectory(dir=ROOT / "运行环境/临时文件")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "运行环境").mkdir()
        self.record = {"schema_version": 1, "endpoints_verified": True, "position_shift": 4096,
                       "identity": {"serial": "测试从臂", "motor_id": 6, "model": "sts3215", "homing_offset": 1925, "drive_mode": 0,
                                    "range_min": 2046, "range_max": 3492}}
        self.save()

    def save(self):
        (self.root / "运行环境/夹爪端点核对.json").write_text(json.dumps(self.record), encoding="utf-8")

    def attach(self, native=-1275):
        with patch.object(self.bus, "read", return_value=native):
            return attach_gripper_coordinates(self.bus, self.root, "测试从臂")

    def test_half_open_observation_recovers_training_coordinate(self):
        result = self.attach()
        self.assertEqual(result["training_position"], 2821)
        self.assertAlmostEqual(self.bus._normalize({6: -1275})[6], 53.5961272476)
        self.assertAlmostEqual(self.bus._normalize({6: -2037})[6], 0.8990318126)

    def test_targets_are_translated_back_to_native_frame_before_wire_encoding(self):
        self.attach()
        raw = self.bus._unnormalize({6: 53.59612724757953})
        self.assertLessEqual(abs(raw[6] - (-1275)), 1)
        wire = self.bus._encode_sign("Goal_Position", raw)
        self.assertLessEqual(abs(wire[6] - 34043), 1)

    def test_all_targets_roundtrip_in_same_frame(self):
        self.attach()
        for target in (7.17, 20., 50., 74.69, 84.97):
            raw = self.bus._unnormalize({6: target})
            self.assertTrue(-2050 <= raw[6] <= -604)
            recovered = self.bus._normalize(raw)[6]
            self.assertLess(abs(recovered - target), 0.08)

    def test_other_joints_are_unchanged(self):
        expected_state = self.bus._normalize({1: 1994})[1]
        expected_target = self.bus._unnormalize({1: 12})[1]
        self.attach()
        self.assertEqual(self.bus._normalize({1: 1994, 6: -1275})[1], expected_state)
        self.assertEqual(self.bus._unnormalize({1: 12, 6: 60})[1], expected_target)

    def test_unverified_or_mismatched_record_refuses(self):
        for change in ("verified", "identity", "shift"):
            with self.subTest(change=change):
                original = json.loads(json.dumps(self.record))
                if change == "verified":
                    self.record["endpoints_verified"] = False
                elif change == "identity":
                    self.record["identity"]["homing_offset"] = 0
                else:
                    self.record["position_shift"] = 1
                self.save()
                with self.assertRaises(RuntimeError):
                    self.attach()
                self.record = original

    def test_runtime_coordinate_jump_stops_instead_of_clipping(self):
        self.attach()
        with self.assertRaisesRegex(RuntimeError, "原始位置异常"):
            self.bus._normalize({6: 2821})

    def test_already_matching_frame_does_not_load_or_modify_mapping(self):
        (self.root / "运行环境/夹爪端点核对.json").unlink()
        self.assertEqual(self.attach(2821)["shift"], 0)
        self.assertAlmostEqual(self.bus._normalize({6: 2821})[6], 53.5961272476)

    def test_small_closed_endpoint_deviation_is_bounded_without_motor_writes(self):
        self.record.update(hardware_command_rejected=True, hold_verified=False)
        self.save()
        diagnostics = []
        with patch.object(self.bus, "read", return_value=2036), \
                patch.object(self.bus, "write") as write, patch.object(self.bus, "sync_write") as sync_write:
            mapping = attach_gripper_coordinates(self.bus, self.root, "测试从臂", require_hold=True,
                                                  on_diagnostic=diagnostics.append)
            self.assertEqual(mapping["native_position"], 2036)
            self.assertEqual(mapping["training_position"], 2046)
            self.assertEqual(mapping["closed_endpoint_correction"], 10)
            self.assertEqual(diagnostics[0]["signed_position"], 2036)
            self.assertFalse(diagnostics[0]["in_calibration_range"])
            write.assert_not_called()
            sync_write.assert_not_called()

    def test_native_feedback_tolerance_is_lower_endpoint_only(self):
        self.attach(2037)
        for value in (2030, 2036, 2037, 2046):
            with self.subTest(value=value):
                self.assertEqual(self.bus._normalize({6: value})[6], 0.)
        for value in (2029, -2037, 3493, 34042):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "原始位置异常"):
                self.bus._normalize({6: value})

    def test_initial_target_stays_legal_and_other_joint_targets_are_unchanged(self):
        calibration = self.bus.calibration["gripper"]
        original = {"shoulder_pan": 2092, "gripper": 2037}
        target = prepare_initial_gripper_hold(original, calibration)
        self.assertEqual(target, {"shoulder_pan": 2092, "gripper": 2046})
        self.assertEqual(original["gripper"], 2037)
        for value in (2029, -2037, 3493):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                prepare_initial_gripper_hold({"gripper": value}, calibration)
        self.attach(2037)
        self.assertEqual(self.bus._unnormalize({6: 0.})[6], 2046)

    def test_tolerance_does_not_apply_to_reversed_or_very_short_calibration(self):
        calibration = MotorCalibration(6, 1, 1925, 2046, 3492)
        self.assertEqual(closed_endpoint_tolerance(calibration), 0)
        with self.assertRaises(RuntimeError):
            native_gripper_feedback_position(2037, calibration)
        self.assertEqual(closed_endpoint_tolerance(MotorCalibration(6, 0, 0, 200, 300)), 1)

    def test_positive_upper_deviation_is_not_accepted_or_wrapped(self):
        self.record.update(hardware_command_rejected=True)
        self.save()
        with patch.object(self.bus, "read", return_value=3502):
            with self.assertRaisesRegex(RuntimeError, "高于上限 10 个计数"):
                attach_gripper_coordinates(self.bus, self.root, "测试从臂", require_hold=True)

    def test_measured_positive_positions_ignore_unrelated_rejected_wrap(self):
        self.record.update(hardware_command_rejected=True, hold_verified=False)
        self.save()
        for native in (2978, 3457):
            with self.subTest(native=native), patch.object(self.bus, "read", return_value=native):
                result = attach_gripper_coordinates(self.bus, self.root, "测试从臂", require_hold=True)
                self.assertEqual(result, {"shift": 0, "native_position": native, "training_position": native})
        persisted = json.loads((self.root / "运行环境/夹爪端点核对.json").read_text(encoding="utf-8"))
        self.assertTrue(persisted["hardware_command_rejected"])
        self.assertFalse(persisted["hold_verified"])

    def test_positive_feedback_cannot_bypass_pending_hardware_repair(self):
        (self.root / "运行环境/夹爪坐标修复.json").write_text(json.dumps({
            "serial": "测试从臂", "inference_allowed": False,
            "endpoints_verified": False, "hold_verified": False}), encoding="utf-8")
        with patch.object(self.bus, "read", return_value=2821), self.assertRaisesRegex(RuntimeError, "推理继续锁定"):
            attach_gripper_coordinates(self.bus, self.root, "测试从臂", require_hold=True)

    def test_inference_refuses_unverified_hardware_hold(self):
        with patch.object(self.bus, "read", return_value=-1275), self.assertRaisesRegex(RuntimeError, "保持验证"):
            attach_gripper_coordinates(self.bus, self.root, "测试从臂", require_hold=True)

    def test_rejected_hardware_mapping_blocks_inference_and_target_generation(self):
        self.record.update(hardware_command_rejected=True, hold_verified=True)
        self.save()
        with patch.object(self.bus, "read", return_value=-1337), self.assertRaisesRegex(RuntimeError, "保持验证失败"):
            attach_gripper_coordinates(self.bus, self.root, "测试从臂", require_hold=True)
        self.attach(-1337)
        with self.assertRaisesRegex(RuntimeError, "禁止生成发送目标"):
            self.bus._unnormalize({6: 50})


if __name__ == "__main__":
    unittest.main()
