"""验证原始预测、受限目标与反馈不会混淆，并核对模型输入证据。"""
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from 推理过程记录 import ProgressReporter, InputArchive, PREFIX, export_run_report, format_progress


class ProgressTests(unittest.TestCase):
    def test_contact_state_is_preserved_in_log_and_shown_with_predictions(self):
        log, output = io.StringIO(), []
        reporter = ProgressReporter(log, [str(i) for i in range(6)], emit=lambda text, **kw: output.append(text))
        state = dict(mode="contact_hold", held_seconds=.5, max_hold_seconds=10., hold_floor=18.02)
        reporter.action(dict(t=1., observed_state=[19.] * 6, model_target=[6.] * 6,
                             target=[18.] * 6, sent_target=[18.] * 6, gripper_control=state))
        telemetry = json.loads(output[0][len(PREFIX):])
        self.assertEqual(telemetry["gripper_control"], state)
        self.assertIn("疑似接触，限幅保持", format_progress(telemetry))
        self.assertIn("/ 10s", format_progress(telemetry))
        telemetry["gripper_control"] = dict(state, held_seconds=120., max_hold_seconds=None)
        self.assertIn("120.0s / 不限时", format_progress(telemetry))
        self.assertEqual(telemetry["model_target"], [6.] * 6)

    def test_full_rate_actions_preserve_raw_model_targets_but_display_is_throttled(self):
        log, output = io.StringIO(), []
        reporter = ProgressReporter(log, [str(i) for i in range(6)], emit=lambda text, **kw: output.append(text))
        event = dict(t=0., observed_state=[0.] * 6, model_target=[15.] * 6, target=[1.] * 6, sent_target=[1.] * 6)
        for tick in range(20):
            reporter.action(dict(event, t=tick * .05), inference_s=.4, inferences=2)
        entries = [json.loads(s) for s in log.getvalue().splitlines()]
        actions = [e for e in entries if e["type"] == "action"]
        self.assertEqual(len(actions), 20)
        self.assertEqual(len(output), 2)
        self.assertTrue(output[0].startswith(PREFIX))
        self.assertEqual(actions[0]["model_target"], [15.] * 6)
        self.assertEqual(actions[0]["sent_target"], [1.] * 6)
        self.assertEqual(actions[0]["command_error"], [1.] * 6)
        self.assertEqual(reporter.summary()["limited_fraction"], [1.] * 6)

    def test_input_archive_saves_exact_state_and_two_actual_images_in_chinese_path(self):
        import cv2
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "实际模型输入"
            image = np.full((8, 8, 3), (40, 100, 220), dtype=np.uint8)
            images = {"observation.images.front": image, "observation.images.wrist": image}
            state = np.arange(6.)
            predictions = np.arange(300, dtype=np.float32).reshape(50, 6)
            recorder = InputArchive(root, lambda s, imgs: predictions)
            recorder.phase = "control"
            returned = recorder(state, images)
            records = [json.loads(s) for s in (root / "模型输入索引.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["state"], state.tolist())
            self.assertEqual(records[0]["phase"], "control")
            self.assertEqual(len(records[0]["images"]), 2)
            self.assertIs(returned, predictions)
            self.assertEqual(records[0]["first_action"], predictions[0].tolist())
            self.assertEqual(records[0]["actions"], predictions.tolist())
            self.assertGreaterEqual(records[0]["prediction_return_time_monotonic"],
                                    records[0]["model_input_time_monotonic"])
            for filename in records[0]["images"].values():
                decoded = cv2.imdecode(np.frombuffer((root / filename).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertLess(np.abs(decoded.astype(float) - image).max(), 4)
            self.assertEqual(recorder.errors, [])

    def test_export_keeps_raw_records_and_in_progress_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "run.jsonl"
            log.write_text('{"type":"action"}\n{"type":', encoding="utf-8")
            self.assertIn(log.read_text(encoding="utf-8"), export_run_report(log))


if __name__ == "__main__":
    unittest.main()
