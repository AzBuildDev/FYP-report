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
import pyarrow as pa
import 实机抓取测试 as grasp
from 选择推理模型 import InferenceSelection
from lerobot.robots.so101_follower import so101_follower
from serial.tools import list_ports
import 相机测试窗口
from 等待测试开始 import wait_for_start
from 推理运行设置 import task_inference_settings


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
    def setUp(self):
        # 自检不能依赖已清理的真实演示，也不能把模拟素材写入正式采集目录。
        temporary_root = grasp.ROOT / "运行环境/临时文件"
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="模拟控制演示_", dir=temporary_root)
        self.addCleanup(temporary.cleanup)
        dataset = Path(temporary.name)
        (dataset / "meta").mkdir()
        (dataset / "data/chunk-000").mkdir(parents=True)
        names = [f"{name}.pos" for name in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                            "wrist_flex", "wrist_roll", "gripper")]
        info = {"features": {key: {"names": names} for key in ("action", "observation.state")}}
        (dataset / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
        pq.write_table(pa.table({key: [[value] * 5 + [40 + value] for value in np.linspace(-5, 5, 50)]
                                for key in ("observation.state", "action")}),
                       dataset / "data/chunk-000/episode_000000.parquet")
        self.dataset = dataset

    def test_different_source_sizes_are_preserved_for_policy_and_fit_only_for_preview(self):
        front, wrist = np.full((1080, 1920, 3), 80, dtype=np.uint8), np.full((480, 640, 3), 100, dtype=np.uint8)
        states = {"observation.images.front": {"time": 1., "image": front},
                  "observation.images.wrist": {"time": 2., "image": wrist}}
        with patch.object(grasp, "latest_frames", states):
            stamp, preview = grasp.get_monitor_frame()
            self.assertEqual((stamp, preview.shape), (1., (480, 1280, 3)))
            _, images = grasp.get_frame()
            self.assertIs(images["observation.images.front"], front)
            self.assertIs(images["observation.images.wrist"], wrist)

    def simulate(self, cancel=False, fail_state=False, initial_state=None,
                 warm_fail=False, warm_cancel=False, stop_after_confirmation=False, stale_after_confirmation=False,
                 motor_fault=False, cleanup_fault=False, stop_during_hold=False,
                 window_start=False, task=False, stop_from_file=False, preview_missing=False, gripper_outside=False,
                 unlimited=False, mode="async", stop_after_actions=None, raw_gripper=None, closing_contact=False,
                 bad_hold_goal=False):
        info = json.loads((self.dataset / "meta/info.json").read_text(encoding="utf-8"))
        names = info["features"]["action"]["names"]
        table = next(self.dataset.glob("data/chunk-*/*.parquet"))
        initial = np.asarray(pq.read_table(table)["observation.state"][0].as_py() if initial_state is None
                             else initial_state, dtype=np.float32)
        sent, instances = [], []
        main_thread = threading.get_ident()
        confirmed, predictions = [], []

        class Bus:
            is_connected = False
            goal_writes = 0
            motors = {name.removesuffix(".pos"): SimpleNamespace(id=i + 1, model="sts3215")
                      for i, name in enumerate(names)}
            calibration = {"gripper": SimpleNamespace(range_min=2046, range_max=3492, drive_mode=0)}
            last_goals = {}

            def _normalize(self, values):
                return {i: (min(100., max(0., (v - 2046) / 1446 * 100)) if i == 6 else v)
                        for i, v in values.items()}

            def _unnormalize(self, values):
                return {i: (int(2046 + min(100., max(0., v)) / 100 * 1446) if i == 6 else int(v))
                        for i, v in values.items()}

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
                if (key == "Status" and motor == "gripper" and motor_fault and sent
                        and (not closing_contact or len(sent) >= 16)):
                    raise RuntimeError("Overload error!")
                if key == "Present_Position" and motor == "gripper":
                    if raw_gripper is not None:
                        return raw_gripper
                    return int(2046 + initial[5] / 100 * (3492 - 2046))
                if key == "Goal_Position":
                    return self.last_goals.get(motor, 0) + int(bad_hold_goal)
                return 0

            def write(self, key, motor, value, **kwargs):
                if cleanup_fault and sent and motor == "gripper":
                    raise RuntimeError("Overload error!")

            def sync_write(self, *args, **kwargs):
                if args[0] == "Goal_Position" and self.goal_writes == 0:
                    assert kwargs.get("normalize") is False, "启用时原始保持位置不能再次归一化"
                if args[0] == "Goal_Position":
                    self.last_goals = args[1].copy()
                self.goal_writes += 1

            def sync_read(self, key, **kwargs):
                if key == "Present_Position" and stop_during_hold:
                    grasp.user_stop.set()
                if key == "Present_Position":
                    assert kwargs.get("normalize") is False
                values = np.ones(6) if key == "Torque_Enable" else initial.copy()
                if key == "Present_Position":
                    values[5] = raw_gripper if raw_gripper is not None else int(2046 + initial[5] / 100 * 1446)
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
                reference = np.asarray([sent[-1][n] for n in names]) if task and sent else self.state
                assert np.max(np.abs(target - reference)) <= (.5 if task else grasp.MAX_STEP_DELTA) + 0.00001
                self.state = target.copy()
                if closing_contact:
                    self.state[5] = max(initial[5], self.state[5])
                sent.append(action)
                if stop_after_actions is not None and len(sent) >= stop_after_actions:
                    grasp.user_stop.set()
                return action

        class Policy:
            config = SimpleNamespace(image_features={
                "observation.images.front": None,
                "observation.images.wrist": None,
            }, chunk_size=50)

            def to(self, device):
                return self

            def eval(self):
                return self

            def reset(self):
                pass

        def capture(index, codec, fps, callback, stop, *args, **kwargs):
            image = np.full((8, 8, 3), 80, dtype=np.uint8)
            while not stop.is_set():
                callback("frame", (time.monotonic(), image))
                stop.wait(0.01)

        def predictor(state, image):
            if mode == "async" or len(predictions) < 2:
                assert threading.get_ident() != main_thread
            predictions.append(True)
            if stop_from_file and len(predictions) == 3:
                stop_path.touch()
            if len(predictions) <= 2:
                assert instances[0].configure_calls == 0
                assert instances[0].bus.goal_writes == 0
                assert sent == []
            if warm_cancel:
                grasp.user_stop.set()
            if warm_fail:
                return np.full((50, 6), np.nan)
            time.sleep(0.04)
            target = state + 30
            if closing_contact:
                target[5] = 0.
            return np.repeat(target[None], 50, axis=0)

        def confirmation(stop):
            assert not window_start, "窗口启动不应等待终端确认"
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
            stop_path = temporary / "运行环境/临时文件/请求停止"
            stop_path.parent.mkdir(parents=True)

            def run():
                grasp.main(mode, settings=task_inference_settings(1.3 if closing_contact else 0 if unlimited else 0.65) if task else None,
                           window_start=window_start, stop_file=stop_path if stop_from_file else None,
                           preview_directory=stop_path.parent if preview_missing else None)

            with patch.object(grasp, "ROOT", temporary), patch.object(grasp, "TEST_SECONDS", 0.35), \
                 patch.object(grasp, "RUN_KIND", "simulation"), \
                 patch.object(grasp, "file_sha256", lambda path: "0" * 64), \
                 patch.object(grasp.torch.cuda, "is_available", return_value=True), \
                 patch.object(grasp.torch, "manual_seed"), \
                 patch.object(grasp, "choose_model", return_value=InferenceSelection(temporary / "模拟模型", self.dataset, temporary)), \
                 patch.object(grasp, "load_local_policy", return_value=Policy()), \
                 patch.object(so101_follower, "SO101Follower", Follower), \
                 patch.object(list_ports, "comports", return_value=[SimpleNamespace(vid=0x1A86, pid=0x55D3, serial_number=grasp.FOLLOWER_SERIAL, device="模拟串口")]), \
                 patch.object(相机测试窗口, "named_capture", capture), \
                 patch.object(grasp, "monitor_loop", lambda *args: None), \
                 patch.object(grasp, "make_predict", lambda *args: predictor), \
                 patch.object(grasp, "wait_for_start", confirmation), \
                 patch.object(grasp, "get_frame", frame):
                if raw_gripper is not None and not 2030 <= raw_gripper <= 3492:
                    with self.assertRaisesRegex(RuntimeError, "夹爪原始位置异常"):
                        run()
                elif bad_hold_goal:
                    with self.assertRaisesRegex(RuntimeError, "启动保持目标读回不一致"):
                        run()
                elif gripper_outside:
                    with self.assertRaisesRegex(RuntimeError, "夹爪起点超出演示范围"):
                        run()
                elif preview_missing:
                    with patch.object(grasp, "wait_for_preview", side_effect=RuntimeError("双摄画面未显示")), \
                         self.assertRaisesRegex(RuntimeError, "画面未显示"):
                        run()
                elif fail_state:
                    with self.assertRaisesRegex(RuntimeError, "模拟读取状态失败"):
                        run()
                elif warm_fail or stale_after_confirmation:
                    with self.assertRaises((RuntimeError, ValueError)):
                        run()
                elif motor_fault:
                    with patch.object(grasp, "MOTOR_HEALTH_INTERVAL", 0.01), self.assertRaisesRegex(RuntimeError, "gripper"):
                        run()
                else:
                    run()
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
        actions = [event for event in logs if event["type"] == "action"]
        self.assertEqual(len(actions), len(sent))
        self.assertEqual(len(actions[0]["observed_state"]), 6)
        self.assertEqual(len(actions[0]["model_target"]), 6)
        self.assertTrue(all(actions[0]["limited"]))
        self.assertGreater(np.max(np.abs(np.asarray(actions[0]["model_target"]) - np.asarray(actions[0]["sent_target"]))), 1)
        self.assertTrue(any(event["type"] == "telemetry" for event in logs))
        self.assertEqual(logs[0]["mode"], "async")
        self.assertEqual(logs[0]["run_kind"], "simulation")
        self.assertEqual(logs[1]["type"], "model_identity")
        self.assertEqual(logs[0]["max_step_delta"], grasp.MAX_STEP_DELTA)
        initial = np.asarray(next(event["initial_state"] for event in logs if event["type"] == "start"))
        first = np.asarray(list(sent[0].values()))
        self.assertAlmostEqual(float(np.max(np.abs(first - initial))), min(grasp.MAX_STEP_DELTA, 30), places=4)

    def test_window_start_runs_task_without_terminal_and_exceeds_old_three_unit_limit(self):
        sent, follower, logs = self.simulate(window_start=True, task=True)
        self.assertGreater(len(sent), 6)
        self.assertEqual(follower.configure_calls, 1)
        self.assertEqual(logs[0]["control_profile"], "task_inference")
        self.assertEqual(logs[0]["chunk_execute"], 20)
        self.assertIsNone(logs[0]["max_total_travel"])
        initial = np.asarray(next(event["initial_state"] for event in logs if event["type"] == "start"))
        self.assertGreater(np.max(np.abs(np.asarray(list(sent[-1].values())) - initial)), 3)
        self.assertEqual(logs[-1]["stop_reason"], "time_limit")
        self.assertEqual(logs[-1]["cleanup_errors"], [])

    def test_window_stop_file_stops_task_and_releases_fake_hardware(self):
        sent, follower, logs = self.simulate(window_start=True, task=True, stop_from_file=True)
        self.assertEqual(logs[-1]["stop_reason"], "user_stop")
        self.assertEqual(logs[-1]["cleanup_errors"], [])
        self.assertFalse(follower.bus.is_connected)

    def test_continuous_task_stops_and_cleans_up_in_both_control_modes(self):
        for mode in ("async", "serial"):
            with self.subTest(mode=mode):
                sent, follower, logs = self.simulate(window_start=True, task=True, unlimited=True,
                                                     mode=mode, stop_after_actions=3)
                self.assertEqual(len(sent), 3)
                self.assertIsNone(logs[0]["test_seconds"])
                self.assertEqual(logs[-1]["stop_reason"], "user_stop")
                self.assertEqual(logs[-1]["cleanup_errors"], [])
                self.assertFalse(follower.bus.is_connected)

    def test_missing_visible_preview_prevents_torque_and_actions(self):
        sent, follower, logs = self.simulate(window_start=True, task=True, preview_missing=True)
        self.assertEqual((sent, follower.configure_calls, follower.bus.goal_writes), ([], 0, 0))
        self.assertEqual(logs[-1]["type"], "preparation_end")

    def test_window_mode_publishes_to_embedded_preview_without_extra_window(self):
        with patch.object(grasp, "run_embedded_monitor") as embedded, patch.object(grasp, "run_monitor") as separate:
            grasp.monitor_loop(self.dataset)
            embedded.assert_called_once()
            separate.assert_not_called()

    def test_cancel_before_start_never_sends_actions(self):
        sent, follower, logs = self.simulate(cancel=True)
        self.assertEqual(sent, [])
        self.assertEqual(follower.configure_calls, 0)
        self.assertFalse(any(event["type"] == "start" for event in logs))
        self.assertEqual(logs[-1], {"type": "cancelled_before_start", "sent_commands": 0})

    def test_user_current_pose_outside_training_range_can_start(self):
        initial = [-2, -23.7, -28, 97.5, 6.5, 40]
        sent, follower, logs = self.simulate(initial_state=initial)
        self.assertGreater(len(sent), 2)
        self.assertEqual(follower.configure_calls, 1)
        self.assertEqual(logs[0]["start_pose_policy"], "current_pose")
        event = next(event for event in logs if event["type"] == "start")
        np.testing.assert_allclose(event["initial_state"], initial)
        self.assertTrue(np.all(np.asarray(event["control_lower"]) <= initial))
        self.assertTrue(np.all(np.asarray(event["control_upper"]) >= initial))
        self.assertEqual(logs[-1]["cleanup_errors"], [])

    def test_fully_open_gripper_outside_demonstrations_refuses_before_torque(self):
        sent, follower, logs = self.simulate(initial_state=[0, -23, -28, 97, 6, 100], gripper_outside=True)
        self.assertEqual((sent, follower.configure_calls, follower.bus.goal_writes), ([], 0, 0))
        self.assertEqual(logs[-1]["type"], "preparation_end")
        check = next(event for event in logs if event["type"] == "gripper_start_check")
        self.assertEqual(check["initial"], 100)
        self.assertEqual(check["phase"], "connection")
        self.assertFalse(any(event["type"] == "model_warmup" for event in logs))

    def test_closed_position_beyond_tolerance_refuses_with_actual_raw_in_log(self):
        sent, follower, logs = self.simulate(raw_gripper=2029)
        self.assertEqual((sent, follower.configure_calls, follower.bus.goal_writes), ([], 0, 0))
        diagnostic = next(event for event in logs if event["type"] == "gripper_raw_diagnostic")
        self.assertEqual(diagnostic["signed_position"], 2029)
        self.assertEqual(diagnostic["calibration_min"], 2046)
        self.assertFalse(diagnostic["in_calibration_range"])
        self.assertEqual(logs[-1]["sent_commands"], 0)
        self.assertIn("低于下限 17 个计数", logs[-1]["error"])
        self.assertNotIn("保持验证失败", logs[-1]["error"])

    def test_closed_endpoint_can_start_with_legal_initial_goal(self):
        path = next(self.dataset.glob("data/chunk-*/*.parquet"))
        table = pq.read_table(path).to_pydict()
        for key in ("observation.state", "action"):
            for row in table[key]:
                row[5] -= 35.
        pq.write_table(pa.table(table), path)
        sent, follower, logs = self.simulate(raw_gripper=2037, window_start=True, task=True)
        self.assertTrue(sent)
        self.assertEqual(follower.configure_calls, 1)
        hold = next(event for event in logs if event["type"] == "initial_raw_hold")
        self.assertEqual(hold["measured_positions"]["gripper"], 2037)
        self.assertEqual(hold["positions"]["gripper"], 2046)
        self.assertEqual(hold["gripper_endpoint_correction"], 9)
        self.assertEqual(logs[-1]["stop_reason"], "time_limit")

    def test_wrong_initial_goal_readback_prevents_torque_enable(self):
        sent, follower, logs = self.simulate(bad_hold_goal=True)
        self.assertEqual((sent, follower.configure_calls, follower.bus.goal_writes), ([], 0, 1))
        self.assertEqual(logs[-1]["stop_reason"], "error")

    def test_task_gripper_targets_stay_inside_demonstrations(self):
        sent, follower, logs = self.simulate(window_start=True, task=True)
        self.assertTrue(sent)
        self.assertLessEqual(max(action["gripper.pos"] for action in sent), 45)
        limits = next(event for event in logs if event["type"] == "gripper_limits")
        self.assertFalse(limits["expand_to_start"])
        hold = next(event for event in logs if event["type"] == "initial_raw_hold")
        self.assertFalse(hold["normalize"])

    def test_closing_contact_does_not_abort_remaining_body_motion(self):
        sent, follower, logs = self.simulate(window_start=True, task=True, closing_contact=True,
                                             initial_state=[-5.] * 5 + [40.])
        contact = next(event for event in logs if event["type"] == "gripper_contact")
        self.assertEqual(contact["event"], "limited_hold")
        self.assertEqual(logs[-1]["stop_reason"], "time_limit")
        after = [event for event in logs if event["type"] == "action" and event["t"] > contact["t"] + .1]
        self.assertGreater(len(after), 2)
        self.assertGreater(after[-1]["sent_target"][0], after[0]["sent_target"][0])
        self.assertEqual(after[-1]["gripper_control"]["mode"], "contact_hold")
        self.assertLessEqual(abs(after[-1]["sent_target"][5] - after[-1]["observed_state"][5]), 1.01)

    def test_overload_during_contact_still_stops_and_releases(self):
        sent, follower, logs = self.simulate(window_start=True, task=True, closing_contact=True, motor_fault=True,
                                             initial_state=[-5.] * 5 + [40.])
        self.assertTrue(any(event["type"] == "gripper_contact" for event in logs))
        self.assertEqual(logs[-1]["stop_reason"], "error")
        self.assertTrue(any("Overload error" in event.get("error", "") for event in logs))
        self.assertFalse(follower.bus.is_connected)

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
