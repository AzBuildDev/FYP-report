"""模拟故障状态与停扭矩失败，验证全关节尝试和无重复写入的串口关闭。"""
import unittest
from 检查与释放电机 import MotorHealthGuard, release_bus


class FaultTests(unittest.TestCase):
    def bus(self):
        class Bus:
            motors = dict.fromkeys(["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"])
            is_connected = True
            writes, reads = None, None
            def __init__(self):
                self.writes, self.reads = [], []
            def write(self, key, motor, value, **kwargs):
                assert (key, value) == ("Torque_Enable", 0)
                self.writes.append(motor)
                if motor == "shoulder_lift":
                    raise RuntimeError("Overload error!")
            def read(self, key, motor, **kwargs):
                self.reads.append(motor)
                return 0
            def disconnect(self, disable_torque=True):
                assert not disable_torque
                self.is_connected = False
        return Bus()

    def test_single_bad_motor_does_not_skip_remaining_motors(self):
        bus = self.bus()
        errors = release_bus(bus)
        self.assertEqual(bus.writes, list(bus.motors))
        self.assertEqual(bus.reads, list(bus.motors))
        self.assertEqual(len(errors), 1)
        self.assertFalse(bus.is_connected)
        self.assertEqual(release_bus(bus), [])
        self.assertEqual(len(bus.writes), 6)

    def test_readback_nonzero_or_error_is_not_success(self):
        for value in (1, "error"):
            bus = self.bus()
            def read(*args, **kwargs):
                if value == "error":
                    raise RuntimeError("无法读回")
                return value
            bus.read = read
            errors = release_bus(bus)
            self.assertGreaterEqual(len(errors), 6)
            self.assertFalse(bus.is_connected)

    def test_periodic_fault_and_packet_error_reject(self):
        for value in (32, "packet_error"):
            bus = self.bus()
            def read(*args, **kwargs):
                if value == "packet_error":
                    raise RuntimeError("Overload error!")
                return value
            bus.read = read
            with self.assertRaisesRegex(RuntimeError, "停止控制"):
                MotorHealthGuard(bus).check()
            self.assertEqual(bus.writes, [])

    def test_check_interval_and_force(self):
        bus, now = self.bus(), [0]
        guard = MotorHealthGuard(bus, clock=lambda: now[0])
        self.assertTrue(guard.check())
        self.assertFalse(guard.check())
        now[0] = 1.1
        self.assertTrue(guard.check())
        self.assertTrue(guard.check(force=True))
        self.assertEqual(len(bus.reads), 18)


if __name__ == "__main__":
    unittest.main()
