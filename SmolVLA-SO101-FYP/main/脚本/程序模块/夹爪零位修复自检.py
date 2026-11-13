"""用替代串口验证零位平移、写入范围及异常中止，不连接实机。"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scservo_sdk as scs
from serial.tools import list_ports
import 修复夹爪硬件坐标 as repair


class RepairTests(unittest.TestCase):
    def test_translation_preserves_training_openings(self):
        plan = repair.make_plan(-1338, 1925, 2046, 3492)
        self.assertEqual((plan["homing_offset"], plan["range_min"], plan["range_max"]), (-1461, 1336, 2782))
        for native in (-2037, -1275, -742, -1338):
            expected = (native + 4096 - 2046) / 1446
            corrected = (native + plan["feedback_delta"] - plan["range_min"]) / plan["span"]
            self.assertAlmostEqual(expected, corrected)
        for normalized in (7.17336, 50, 84.9701):
            target = plan["range_min"] + normalized / 100 * plan["span"]
            self.assertTrue(1336 <= target <= 2782)

    def test_refuses_unknown_start(self):
        for native in (-500, 1000, -2048):
            with self.assertRaises(RuntimeError):
                repair.make_plan(native, 1925, 2046, 3492)

    def run_simulated(self, *, fault=0, feedback=2048, write_error=0):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root / "运行环境").mkdir()
        (root / "运行记录").mkdir()
        cal = root / "cal.json"
        original = {"gripper": dict(id=6, drive_mode=0, homing_offset=1925, range_min=2046, range_max=3492)}
        cal.write_text(json.dumps(original), encoding="utf-8")
        endpoint = dict(identity=dict(serial=repair.SERIAL, motor_id=6, model="sts3215", **{k: v for k, v in original["gripper"].items() if k != "id"}),
                        endpoints_verified=True, measurements=[])
        (root / "运行环境/夹爪端点核对.json").write_text(json.dumps(endpoint), encoding="utf-8")
        registers = {i: {3: 777, 0: 3, 1: 11, 18: 12, 31: 1925, 9: 2046, 11: 3492, 56: 32768 + 1338}
                     for i in range(1, 7)}
        writes = []

        class Port:
            ser = object()
            def __init__(self, *args): pass
            def openPort(self): return True
            def setBaudRate(self, *args): return True
            def closePort(self): pass

        class Packet:
            def read1ByteTxRx(self, port, motor, address):
                return self.read2ByteTxRx(port, motor, address)
            def read2ByteTxRx(self, port, motor, address):
                return (fault if address == 65 else registers[motor].get(address, 0)), 0, 0
            def write2ByteTxRx(self, port, motor, address, value):
                writes.append((motor, address, value))
                registers[motor][address] = value
                if address == 31:
                    registers[motor][56] = feedback
                return 0, write_error

        with patch.object(repair, "ROOT", root), patch.object(repair, "CALIBRATION", cal), \
             patch.object(scs, "PortHandler", Port), patch.object(scs, "PacketHandler", return_value=Packet()), \
             patch.object(list_ports, "comports", return_value=[SimpleNamespace(serial_number=repair.SERIAL, device="fake")]), \
             patch("sys.argv", ["repair", "--apply"]):
            error = None
            try:
                repair.main()
            except RuntimeError as exc:
                error = str(exc)
        self.assertEqual(json.loads(cal.read_text(encoding="utf-8")), original)
        return writes, error, root

    def test_only_three_gripper_registers_changed_without_installing_candidate(self):
        writes, error, root = self.run_simulated()
        self.assertIsNone(error)
        self.assertEqual(writes, [(6, 31, 3509), (6, 9, 1336), (6, 11, 2782)])
        state = json.loads((root / "运行环境/夹爪坐标修复.json").read_text(encoding="utf-8"))
        self.assertFalse(state["inference_allowed"])
        self.assertFalse(state["hold_verified"])

    def test_existing_fault_prevents_any_write(self):
        writes, error, _ = self.run_simulated(fault=32)
        self.assertEqual(writes, [])
        self.assertIn("故障", error)

    def test_unexpected_feedback_stops_before_limits(self):
        writes, error, _ = self.run_simulated(feedback=100)
        self.assertEqual(len(writes), 1)
        self.assertIn("预期 2048", error)

    def test_device_rejection_is_numeric_and_never_retried(self):
        writes, error, _ = self.run_simulated(write_error=16)
        self.assertEqual(len(writes), 1)
        self.assertIn("device_error=16 (0x10)", error)


if __name__ == "__main__":
    unittest.main()
