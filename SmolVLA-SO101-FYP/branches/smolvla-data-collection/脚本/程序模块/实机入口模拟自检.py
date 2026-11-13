"""用替代串口、相机和模型运行真实入口，检查确认、控制和退出；不访问硬件。"""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq
import 实机抓取测试 as grasp
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.so101_follower import so101_follower
from serial.tools import list_ports
import 相机测试窗口
from 等待测试开始 import wait_for_start


class ConfirmationTests(unittest.TestCase):
    def test_enter_confirms(self):
        self.assertTrue(wait_for_start(threading.Event(), lambda: "\r"))

    def test_window_stop_interrupts_wait_without_enter(self):
        stop = threading.Event()
        result = []
        thread = threading.Thread(target=lambda: result.append(wait_for_start(stop, lambda: None)))
        thread.start()
        try:
            stop.set()
            thread.join(timeout=0.5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result, [False])
        finally:
            stop.set()
            thread.join(timeout=1)

    def test_escape_and_ctrl_c_cancel(self):
        self.assertFalse(wait_for_start(threading.Event(), lambda: "\x1b"))
        with self.assertRaises(KeyboardInterrupt):
            wait_for_start(threading.Event(), lambda: "\x03")

    def test_stop_wins_over_enter(self):
        stop = threading.Event()

        def read_key():
            stop.set()
            return "\r"

        self.assertFalse(wait_for_start(stop, read_key))


class EntryTests(unittest.TestCase):
    def simulate(self, cancel=False, fail_state=False, initial_state=None,
                 warm_fail=False, warm_cancel=False, stop_after_confirmation=False, stale_after_confirmation=False,
                 motor_fault=False, cleanup_fault=False, stop_during_hold=False):
        info = json.loads((grasp.DATASET_ROOT / "meta/info.json").read_text(encoding="utf-8"))
        names = info["features"]["action"]["names"]
        table = next(grasp.DATASET_ROOT.glob("data/chunk-*/*.parquet"))
        initial = np.asarray(pq.read_table(table)["observation.state"][0].as_py() if initial_state is None
                             else initial_state, dtype=np.float32)
        sent, instances = [], []
        main_thread = threading.get_ident()
        confirmed, predictions = [], []

        class Bus:
            is_connected = False
            goal_writes = 0
            motors = dict.fromkeys([name.removesuffix(".pos") for name in names])

            def connect(self):
                self.is_connected = True

            def disconnect(self, disable_torque=True):
                if disable_torque:
                    self.disable_torque()
                self.is_connected = False

            def disable_torque(self):
                if cleanup_fault and sent:
                    raise RuntimeError("模拟过载使默认 disconnect 失败")

            def read(self, key, motor, **kwargs):
                if key == "Status" and motor == "gripper" and motor_fault and sent:
                    raise RuntimeError("Overload error!")
                return 0

            def write(self, key, motor, value, **kwargs):
                if cleanup_fault and sent and motor == "gripper":
                    raise RuntimeError("Overload error!")

            def sync_write(self, *args):
                self.goal_writes += 1

            def sync_read(self, key, **kwargs):
                if key == "Present_Position" and stop_during_hold:
                    grasp.user_stop.set()
                values = np.ones(6) if key == "Torque_Enable" else initial
                return dict(zip([n.removesuffix(".pos") for n in names], values))

        class Follower:
            def __init__(self, config):
                self.bus, self.state = Bus(), initial.copy()
                self.action_features = dict.fromkeys(names, float)
                self.is_calibrated = True
                self.configure_calls = 0
                instances.append(self)

            def get_observation(self):
                assert threading.get_ident() == main_thread
                if confirmed and stop_after_confirmation:
                    grasp.user_stop.set()
                if fail_state:
                    raise RuntimeError("模拟读取状态失败")
                return dict(zip(names, self.state))

            def configure(self):
                self.configure_calls += 1

            def send_action(self, action):
                assert threading.get_ident() == main_thread
                target = np.asarray([action[n] for n in names])
                assert np.max(np.abs(target - self.state)) <= grasp.MAX_STEP_DELTA + 0.00001
                self.state = target.copy()
                sent.append(action)
                return action

        class Policy:
            config = SimpleNamespace(image_features={"observation.images.front": None}, chunk_size=50)

            def to(self, device):
                return self

            def eval(self):
                return self

            def reset(self):
                pass

        def capture(index, codec, fps, callback, stop, *args):
            image = np.full((8, 8, 3), 80, dtype=np.uint8)
            while not stop.is_set():
                callback("frame", (time.monotonic(), image))
                stop.wait(0.01)

        def predictor(state, image):
            assert threading.get_ident() != main_thread
            predictions.append(True)
            if len(predictions) <= 2:
                assert instances[0].configure_calls == 0
                assert instances[0].bus.goal_writes == 0
                assert sent == []
            if warm_cancel:
                grasp.user_stop.set()
            if warm_fail:
                return np.full((50, 6), np.nan)
            time.sleep(0.04)
            return np.repeat((state + 30)[None], 50, axis=0)

        def confirmation(stop):
            assert len(predictions) == 2
            assert instances[0].configure_calls == 0
            confirmed.append(True)
            if cancel:
                stop.set()
            return not cancel

        original_get_frame = grasp.get_frame

        def frame():
            stamp, image = original_get_frame()
            return (time.monotonic() - 2 if confirmed and stale_after_confirmation else stamp), image

        temp_root = (grasp.ROOT / "运行环境/临时文件").resolve()
        assert temp_root.is_relative_to(grasp.ROOT.resolve())
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="控制入口自检_", dir=temp_root) as temp:
            temporary = Path(temp)
            (temporary / "运行记录").mkdir()
            with patch.object(grasp, "ROOT", temporary), patch.object(grasp, "TEST_SECONDS", 0.35), \
                 patch.object(grasp, "RUN_KIND", "simulation"), \
                 patch.object(grasp, "file_sha256", lambda path: "0" * 64), \
                 patch.object(grasp.torch.cuda, "is_available", return_value=True), \
                 patch.object(grasp.torch, "manual_seed"), \
                 patch.object(SmolVLAPolicy, "from_pretrained", return_value=Policy()), \
                 patch.object(so101_follower, "SO101Follower", Follower), \
                 patch.object(list_ports, "comports", return_value=[SimpleNamespace(vid=0x1A86, pid=0x55D3, serial_number=grasp.FOLLOWER_SERIAL, device="模拟串口")]), \
                 patch.object(相机测试窗口, "named_capture", capture), \
                 patch.object(grasp, "monitor_loop", lambda: None), \
                 patch.object(grasp, "make_predict", lambda *args: predictor), \
                 patch.object(grasp, "wait_for_start", confirmation), \
                 patch.object(grasp, "get_frame", frame):
                if fail_state:
                    with self.assertRaisesRegex(RuntimeError, "模拟读取状态失败"):
                        grasp.main("async")
                elif warm_fail or stale_after_confirmation:
                    with self.assertRaises((RuntimeError, ValueError)):
                        grasp.main("async")
                elif motor_fault:
                    with patch.object(grasp, "MOTOR_HEALTH_INTERVAL", 0.01), self.assertRaisesRegex(RuntimeError, "gripper"):
                        grasp.main("async")
                else:
                    grasp.main("async")
            logs = [json.loads(line) for line in next((temporary / "运行记录").glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
        self.assertFalse(instances[0].bus.is_connected)
        self.assertTrue(grasp.camera_stop.is_set())
        return sent, instances[0], logs

    def test_actual_entry_runs_and_releases_fake_hardware(self):
        sent, follower, logs = self.simulate()
        self.assertGreater(len(sent), 2)
        self.assertEqual(follower.configure_calls, 1)
        self.assertEqual(logs[-1]["sent_commands"], len(sent))
        self.assertEqual(logs[-1]["cleanup_errors"], [])
        self.assertEqual(logs[0]["mode"], "async")
        self.assertEqual(logs[0]["run_kind"], "simulation")
        self.assertEqual(logs[1]["type"], "model_identity")
        self.assertEqual(logs[0]["max_step_delta"], grasp.MAX_STEP_DELTA)
        initial = np.asarray(next(event["initial_state"] for event in logs if event["type"] == "start"))
        first = np.asarray(list(sent[0].values()))
        self.assertAlmostEqual(float(np.max(np.abs(first - initial))), min(grasp.MAX_STEP_DELTA, 30), places=4)

    def test_cancel_before_start_never_sends_actions(self):
        sent, follower, logs = self.simulate(cancel=True)
        self.assertEqual(sent, [])
        self.assertEqual(follower.configure_calls, 0)
        self.assertFalse(any(event["type"] == "start" for event in logs))
        self.assertEqual(logs[-1], {"type": "cancelled_before_start", "sent_commands": 0})

    def test_user_current_pose_outside_training_range_can_start(self):
        initial = [-2, -23.7, -28, 97.5, 6.5, 100]
        sent, follower, logs = self.simulate(initial_state=initial)
        self.assertGreater(len(sent), 2)
        self.assertEqual(follower.configure_calls, 1)
        self.assertEqual(logs[0]["start_pose_policy"], "current_pose")
        event = next(event for event in logs if event["type"] == "start")
        np.testing.assert_allclose(event["initial_state"], initial)
        self.assertTrue(np.all(np.asarray(event["control_lower"]) <= initial))
        self.assertTrue(np.all(np.asarray(event["control_upper"]) >= initial))
        self.assertEqual(logs[-1]["cleanup_errors"], [])

    def test_bad_warm_prediction_refuses_before_confirmation_and_torque(self):
        sent, follower, logs = self.simulate(warm_fail=True)
        self.assertEqual((sent, follower.configure_calls, follower.bus.goal_writes), ([], 0, 0))
        self.assertEqual(logs[-1]["type"], "preparation_end")
        self.assertIn("预热", logs[-1]["error"])

    def test_cancel_during_warmup_does_not_enable_or_write_positions(self):
        sent, follower, logs = self.simulate(warm_cancel=True)
        self.assertEqual((sent, follower.configure_calls, follower.bus.goal_writes), ([], 0, 0))
        self.assertEqual(logs[-1]["type"], "cancelled_before_start")

    def test_stop_during_post_confirmation_read_does_not_enable(self):
        sent, follower, logs = self.simulate(stop_after_confirmation=True)
        self.assertEqual((sent, follower.configure_calls, follower.bus.goal_writes), ([], 0, 0))
        self.assertEqual(logs[-1]["type"], "cancelled_before_start")

    def test_stale_frame_after_confirmation_refuses_before_torque(self):
        sent, follower, logs = self.simulate(stale_after_confirmation=True)
        self.assertEqual((sent, follower.configure_calls, follower.bus.goal_writes), ([], 0, 0))
        self.assertEqual(logs[-1]["type"], "preparation_end")

    def test_fault_during_control_stops_and_closes_bus(self):
        sent, follower, logs = self.simulate(motor_fault=True)
        self.assertGreater(len(sent), 0)
        self.assertEqual(logs[-1]["stop_reason"], "error")
        self.assertEqual(logs[-1]["sent_commands"], len(sent))
        self.assertTrue(any(event["type"] == "error" for event in logs))
        self.assertFalse(follower.bus.is_connected)

    def test_failed_torque_release_still_closes_bus_and_records_failure(self):
        sent, follower, logs = self.simulate(cleanup_fault=True)
        self.assertGreater(len(sent), 0)
        self.assertTrue(any("gripper" in message for message in logs[-1]["cleanup_errors"]))
        self.assertFalse(follower.bus.is_connected)

    def test_stop_during_initial_hold_read_prevents_torque_configuration(self):
        sent, follower, logs = self.simulate(stop_during_hold=True)
        self.assertEqual((sent, follower.configure_calls, follower.bus.goal_writes), ([], 0, 0))
        self.assertEqual(logs[-1]["stop_reason"], "user_stop")
        self.assertEqual(logs[-1]["sent_commands"], 0)
        self.assertFalse(any(event["type"] == "start" for event in logs))

    def test_prepare_failure_records_terminal_without_actions(self):
        sent, follower, logs = self.simulate(fail_state=True)
        self.assertEqual(sent, [])
        self.assertEqual(follower.configure_calls, 0)
        self.assertEqual(logs[-1]["type"], "preparation_end")
        self.assertEqual(logs[-1]["sent_commands"], 0)
        self.assertIn("模拟读取状态失败", logs[-1]["error"])


if __name__ == "__main__":
    unittest.main()
