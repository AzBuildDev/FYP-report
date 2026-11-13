"""用模拟输入验证黑帧、断帧、无效动作和关节滞后保护，不连接机械臂。"""
import unittest
from unittest.mock import patch
import numpy as np
from 检查控制输入 import bounded_target, bounds_including_start, validate_frame, SlewTrackingGuard
import 实机抓取测试 as grasp_test


class ControlInputTests(unittest.TestCase):
    def make_guard(self, tracking=4., timeout=.6):
        return SlewTrackingGuard(np.zeros(6), np.full(6, -100), np.full(6, 100), 1.,
                                 np.full(6, tracking), stall_timeout=timeout)

    def test_trajectory_advances_past_one_without_unbounded_windup(self):
        guard = self.make_guard()
        outputs = []
        for tick in range(12):
            guard.observe(np.zeros(6), tick * .01)
            target = guard.target(np.full(6, 30), np.zeros(6))
            outputs.append(target.copy())
            guard.commit(target)
        np.testing.assert_allclose(np.asarray(outputs)[:4, 0], [1, 2, 3, 4])
        self.assertLessEqual(np.max(outputs), 4)

    def test_motionless_motor_stops_after_bounded_error_timeout(self):
        guard = self.make_guard(timeout=.1)
        for tick in range(4):
            guard.observe(np.zeros(6), tick * .01)
            guard.commit(guard.target(np.full(6, 30), np.zeros(6)))
        guard.observe(np.zeros(6), .05)
        with self.assertRaisesRegex(RuntimeError, "跟随停滞"):
            guard.observe(np.zeros(6), .16)

    def test_measured_progress_resets_stall_timer(self):
        guard = self.make_guard(timeout=.1)
        guard.previous[:] = 4
        guard.observe(np.zeros(6), 0)
        guard.observe(np.full(6, .3), .09)
        guard.observe(np.full(6, .6), .18)

    def test_unexpected_feedback_jump_stops(self):
        with self.assertRaisesRegex(RuntimeError, "跟随误差超限"):
            self.make_guard().observe(np.full(6, 20), 0)

    def test_reversal_and_range_keep_command_slew(self):
        guard = self.make_guard()
        guard.commit(guard.target(np.full(6, 30), np.zeros(6)))
        target = guard.target(np.full(6, -30), np.zeros(6))
        np.testing.assert_allclose(target, 0)
        with self.assertRaisesRegex(RuntimeError, "相邻指令限速"):
            guard.commit(np.full(6, 10))

    def test_stale_or_missing_camera_frame(self):
        for stamp, image in [(1, np.full((4, 4, 3), 80)), (2, None), (float("nan"), np.ones((4, 4, 3)))]:
            with self.assertRaises(RuntimeError):
                validate_frame(stamp, image, 2)

    def test_black_frame(self):
        with self.assertRaises(RuntimeError):
            validate_frame(2, np.zeros((4, 4, 3)), 2)

    def test_valid_frame(self):
        image = np.full((4, 4, 3), 80)
        self.assertIs(validate_frame(1.8, image, 2), image)

    def test_limit_uses_measured_position_when_motor_lags(self):
        current = np.zeros(6)
        for max_delta in (1.5, 6.0):
            for _ in range(20):
                result = bounded_target(np.full(6, 30), current, np.full(6, -100), np.full(6, 100), max_delta)
                np.testing.assert_allclose(result, max_delta)

    def test_intersection_with_demonstration_range(self):
        result = bounded_target(np.full(6, 30), np.full(6, 9), np.zeros(6), np.full(6, 10), 1.5)
        np.testing.assert_allclose(result, 10)

    def test_invalid_values_and_unreachable_range(self):
        for target, current in [(np.full(6, np.nan), np.zeros(6)), (np.zeros(5), np.zeros(6)),
                                (np.zeros(6), np.full(6, 200))]:
            with self.assertRaises(RuntimeError):
                bounded_target(target, current, np.full(6, -100), np.full(6, 100), 1.5)

    def test_start_outside_demonstrations_is_accepted_and_still_bounded(self):
        current = np.array([-2, -23.7, -28, 97.5, 6.5, 100], dtype=np.float32)
        lo, hi = bounds_including_start(current, np.full(6, 10), np.full(6, 60))
        self.assertTrue(np.all(lo <= current))
        self.assertTrue(np.all(hi >= current))
        target = bounded_target(np.full(6, 1000), current, lo, hi, 6)
        self.assertLessEqual(float(np.max(np.abs(target - current))), 6)
        self.assertTrue(np.all(target <= 100))
        self.assertGreaterEqual(float(target[-1]), 0)

    def test_invalid_start_feedback_remains_rejected(self):
        with self.assertRaises(RuntimeError):
            bounds_including_start(np.full(6, np.nan), np.full(6, -100), np.full(6, 100))

    def test_prepare_failure_releases_acquired_resources(self):
        released = []

        def fail_prepare(resources, mode, scene, selection, settings, window_start, **kwargs):
            resources.callback(released.append, "serial_bus_closed")
            raise RuntimeError("模拟相机启动失败")

        with patch.object(grasp_test, "run_test", fail_prepare):
            with self.assertRaisesRegex(RuntimeError, "模拟相机启动失败"):
                grasp_test.main(selection=object())
        self.assertEqual(released, ["serial_bus_closed"])
        self.assertTrue(grasp_test.camera_stop.is_set())

    def test_disconnect_even_when_torque_release_fails(self):
        class Bus:
            is_connected = True
            motors = {"gripper": None}

            def write(self, *args, **kwargs):
                raise RuntimeError("模拟串口写失败")

            def read(self, *args, **kwargs):
                raise RuntimeError("模拟串口读失败")

            def disconnect(self, disable_torque=True):
                assert disable_torque is False
                self.is_connected = False

        follower = type("FakeFollower", (), {"bus": Bus()})()
        errors = grasp_test.release_follower(follower)
        self.assertTrue(any("模拟串口写失败" in error for error in errors))
        self.assertTrue(any("模拟串口读失败" in error for error in errors))
        self.assertFalse(follower.bus.is_connected)


if __name__ == "__main__":
    unittest.main()
