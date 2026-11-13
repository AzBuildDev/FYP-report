"""以替代电机检查仅夹爪保持、目标回读和异常停扭矩。"""
import unittest
from 夹爪保持验证 import hold_current_gripper, hold_start_allowed, require_native_hold_coordinates


class FakeBus:
    motors = dict.fromkeys(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))

    def __init__(self, drift=False, bad_goal=False, fault=False):
        self.torque, self.goals, self.writes = 0, [], []
        self.drift, self.bad_goal, self.fault = drift, bad_goal, fault

    def sync_read(self, key, **kwargs):
        return {motor: self.torque if motor == "gripper" else 0 for motor in self.motors}

    def read(self, key, motor, **kwargs):
        if key == "Present_Position":
            return 2821 + (20 if self.drift and self.torque else 0)
        if key == "Goal_Position":
            return 2821 + (1 if self.bad_goal else 0)
        if key == "Torque_Enable":
            return self.torque
        if key == "Status" and self.fault and self.torque:
            raise RuntimeError("模拟过载")
        return 0

    def sync_write(self, key, values, **kwargs):
        assert key == "Goal_Position" and kwargs["normalize"] is False
        assert self.torque == 0
        self.goals.append(values)

    def write(self, key, motor, value, **kwargs):
        assert key == "Torque_Enable" and motor == "gripper"
        self.writes.append((motor, value))
        self.torque = value


class HoldTests(unittest.TestCase):
    def test_only_native_coordinates_can_reach_hold_execution(self):
        require_native_hold_coordinates({"shift": 0, "native_position": 2978})
        for mapping in ({"shift": 4096, "native_position": -1337},
                        {"shift": -4096, "native_position": 6917},
                        {"shift": 0, "native_position": -1337}):
            with self.subTest(mapping=mapping), self.assertRaisesRegex(RuntimeError, "原生正坐标"):
                require_native_hold_coordinates(mapping)

    def test_measured_half_open_position_is_accepted_but_endpoints_are_not(self):
        self.assertTrue(hold_start_allowed(44.32918395573997))
        self.assertTrue(hold_start_allowed(40.0))
        self.assertFalse(hold_start_allowed(39.99))
        self.assertFalse(hold_start_allowed(75.01))
        self.assertTrue(hold_start_allowed(49.30843706777317))
        self.assertTrue(hold_start_allowed(53.59612724757953))
        self.assertFalse(hold_start_allowed(0.9))
        self.assertFalse(hold_start_allowed(90.45643153526972))
        self.assertFalse(hold_start_allowed(float("nan")))

    def test_only_gripper_is_enabled_with_current_native_target(self):
        bus = FakeBus()
        result = hold_current_gripper(bus, 2821, seconds=0.03)
        self.assertEqual(bus.goals, [{"gripper": 2821}])
        self.assertEqual(bus.writes, [("gripper", 1), ("gripper", 0)])
        self.assertEqual(result["max_position_change"], 0)

    def test_wrong_goal_readback_prevents_enable(self):
        bus = FakeBus(bad_goal=True)
        with self.assertRaisesRegex(RuntimeError, "读回不一致"):
            hold_current_gripper(bus, 2821, seconds=0.03)
        self.assertEqual(bus.writes, [])

    def test_motion_during_hold_disables_torque(self):
        bus = FakeBus(drift=True)
        with self.assertRaisesRegex(RuntimeError, "超过 15"):
            hold_current_gripper(bus, 2821, seconds=0.03)
        self.assertEqual(bus.torque, 0)

    def test_fault_during_hold_disables_torque(self):
        bus = FakeBus(fault=True)
        with self.assertRaisesRegex(RuntimeError, "模拟过载"):
            hold_current_gripper(bus, 2821, seconds=0.03)
        self.assertEqual(bus.torque, 0)

    def test_observed_negative_coordinate_refuses_before_any_write(self):
        bus = FakeBus()
        with self.assertRaisesRegex(RuntimeError, "负坐标写入"):
            hold_current_gripper(bus, -1337)
        self.assertEqual(bus.goals, [])
        self.assertEqual(bus.writes, [])


if __name__ == "__main__":
    unittest.main()
