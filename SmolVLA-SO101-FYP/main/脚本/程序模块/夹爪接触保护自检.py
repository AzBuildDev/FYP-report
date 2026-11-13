"""验证夹爪闭合接触后的限幅、释放与故障停止；不访问真实设备。"""
import unittest

import numpy as np

from 检查控制输入 import SlewTrackingGuard
from 夹爪接触保护 import GripperContactGuard


class ContactTests(unittest.TestCase):
    def make_guard(self):
        self.events = []
        self.current = np.array([0., 0., 0., 0., 0., 19.01798], dtype=np.float32)
        guard = SlewTrackingGuard(self.current, [-100.] * 5 + [0.], [100.] * 5 + [69.17],
                                  .5, [5.] * 5 + [3.],
                                  names=[f"joint_{i}" for i in range(5)] + ["gripper.pos"],
                                  gripper_index=5, on_gripper_event=self.events.append)
        return guard

    def step(self, guard, tick, gripper_target=0., *, body_follows=True, gripper_follows=False):
        guard.observe(self.current, tick * .05)
        before = guard.previous.copy()
        target = guard.target(np.array([30., 0., 0., 0., 0., gripper_target]), self.current)
        self.assertLessEqual(float(np.max(np.abs(target - before))), .50001)
        guard.commit(target)
        if body_follows:
            self.current[:5] = target[:5]
        if gripper_follows:
            self.current[5] = target[5]
        return target

    def enter_contact(self):
        guard = self.make_guard()
        for tick in range(24):
            self.step(guard, tick)
        self.assertTrue(guard.gripper_contact.active)
        return guard

    def test_closing_contact_reduces_error_and_body_continues_past_old_timeout(self):
        guard = self.make_guard()
        for tick in range(60):
            self.step(guard, tick)
        self.assertGreater(float(self.current[0]), 20)
        self.assertEqual(len(self.events), 1)
        self.assertEqual(self.events[0]["event"], "limited_hold")
        self.assertAlmostEqual(float(guard.previous[5]), float(self.current[5]) - 1., places=4)

    def test_recorded_three_unit_closing_error_no_longer_stops_at_point_six_seconds(self):
        guard = self.make_guard()
        guard.previous[5] = self.current[5] - 3.
        guard.last_requested[5] = 6.04127
        for tick in range(40):
            self.step(guard, tick, 6.04127)
        self.assertTrue(guard.gripper_contact.active)
        self.assertLess(float(self.current[5] - guard.previous[5]), 1.01)

    def test_contact_floor_does_not_chase_closing_feedback(self):
        guard = self.enter_contact()
        floor = guard.gripper_contact.hold_floor
        self.current[5] -= .4
        for tick in range(24, 40):
            self.step(guard, tick)
        self.assertEqual(guard.gripper_contact.hold_floor, floor)
        self.assertGreaterEqual(float(guard.previous[5]), floor - 1e-4)

    def test_model_opening_releases_contact_and_opening_jam_still_stops(self):
        guard = self.enter_contact()
        self.step(guard, 24, 40.)
        self.assertFalse(guard.gripper_contact.active)
        self.assertEqual(self.events[-1]["event"], "model_opening")
        with self.assertRaisesRegex(RuntimeError, "gripper.pos 跟随停滞"):
            for tick in range(25, 90):
                self.step(guard, tick, 40.)
        self.assertEqual(sum(event["event"] == "limited_hold" for event in self.events), 1)

    def test_opening_jam_never_enters_contact_hold(self):
        guard = self.make_guard()
        with self.assertRaisesRegex(RuntimeError, "gripper.pos 跟随停滞"):
            for tick in range(50):
                self.step(guard, tick, 60.)
        self.assertEqual(self.events, [])

    def test_body_joint_jam_is_not_exempted_by_gripper_contact(self):
        guard = self.make_guard()
        with self.assertRaisesRegex(RuntimeError, "joint_0 跟随停滞"):
            for tick in range(50):
                self.step(guard, tick, body_follows=False)

    def test_feedback_jump_still_stops_during_contact(self):
        guard = self.enter_contact()
        self.current[5] += 5
        with self.assertRaisesRegex(RuntimeError, "跟随误差超限"):
            guard.observe(self.current, 1.2)

    def test_contact_without_release_continues_and_model_can_release_later(self):
        guard = self.enter_contact()
        for tick in range(24, 2401):
            self.step(guard, tick)
        self.assertTrue(guard.gripper_contact.active)
        self.assertGreater(guard.gripper_status["held_seconds"], 90.)
        self.assertIsNone(guard.gripper_status["max_hold_seconds"])
        self.assertAlmostEqual(float(guard.previous[5]), float(self.current[5]) - 1., places=4)
        self.step(guard, 2401, 40.)
        self.assertFalse(guard.gripper_contact.active)
        self.assertEqual(self.events[-1]["event"], "model_opening")

    def test_normal_closing_motion_does_not_latch_contact(self):
        guard = self.make_guard()
        for tick in range(60):
            self.step(guard, tick, gripper_follows=True)
        self.assertEqual(self.events, [])
        self.assertAlmostEqual(float(self.current[5]), 0.)

    def test_contact_configuration_must_be_finite_and_bounded(self):
        for kwargs in ({"closing_offset": 3.}, {"closing_offset": float("inf")},
                       {"detect_seconds": 0.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GripperContactGuard(0., 69.17, 3., **kwargs)


if __name__ == "__main__":
    unittest.main()
