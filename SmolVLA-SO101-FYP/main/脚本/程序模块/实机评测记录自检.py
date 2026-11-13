"""用临时软件日志检查人工评测、统计、修订和窗口；不启动真实机械臂。"""
import json
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from 管理实机评测记录 import EvaluationStore, SUCCESS, read_run

ROOT = Path(__file__).resolve().parents[2]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = ROOT / "运行环境/临时文件"
        temp.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="评测记录软件自检_", dir=temp)
        self.addCleanup(self.temporary.cleanup)
        self.logs = Path(self.temporary.name) / "仅软件测试日志"
        self.logs.mkdir()
        self.store = EvaluationStore(self.logs)
        self.sequence = 0

    def make_log(self, kind="real_hardware", commands=3, started=True, error=False):
        self.sequence += 1
        path = self.logs / f"实机抓取测试_软件占位_{self.sequence}.jsonl"
        config = {"type": "配置", "task": "仅软件测试任务", "mode": "async", "scene": "软件测试场景", "fixture_id": self.sequence}
        if kind:
            config["run_kind"] = kind
        events = [config, {"type": "model_identity", "model_path": "仅软件测试检查点", "checkpoint_sha256": "1" * 64}]
        if started:
            events.append({"type": "start"})
        if error:
            events.append({"type": "error", "error": "软件测试异常"})
        events.append({"type": "end", "sent_commands": commands, "seconds": 1, "cleanup_errors": []})
        path.write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n", encoding="utf-8")
        return path

    def mark(self, path, outcome=SUCCESS):
        return self.store.record(path, outcome, "软件测试场景", "软件自检", observed_real=True)

    def test_pending_is_not_success_and_real_observation_required(self):
        path = self.make_log()
        group = self.store.summary()["groups"][0]
        self.assertEqual(group["pending_attempts"], 1)
        self.assertIsNone(group["success_rate"])
        with self.assertRaisesRegex(ValueError, "本人观察"):
            self.store.record(path, SUCCESS, "场景", "记录者")
        self.assertFalse(self.store.path.exists())

    def test_simulation_rejected_and_excluded(self):
        path = self.make_log(kind="simulation")
        with self.assertRaisesRegex(ValueError, "模拟"):
            self.mark(path)
        self.assertEqual(self.store.summary()["groups"], [])
        self.assertEqual(self.store.summary()["excluded_simulation_runs"], 1)

    def test_legacy_requires_explicit_review(self):
        path = self.make_log(kind=None)
        self.assertEqual(self.store.summary()["unverified_legacy_runs"], 1)
        self.assertEqual(self.store.summary()["groups"], [])
        self.mark(path)
        self.assertEqual(self.store.summary()["unverified_legacy_runs"], 0)

    def test_interruptions_and_program_errors_remain_in_denominator(self):
        for outcome in (SUCCESS, "抓取失败", "人工中断", "程序异常"):
            self.mark(self.make_log(error=outcome == "程序异常"), outcome)
        group = self.store.summary()["groups"][0]
        self.assertEqual(group["attempts"], 4)
        self.assertEqual(group["successes"], 1)
        self.assertEqual(group["success_rate"], 0.25)

    def test_zero_actions_and_prestart_cancel_are_not_attempts(self):
        path = self.make_log(commands=0, started=False)
        with self.assertRaisesRegex(ValueError, "没有开始"):
            self.mark(path)
        self.mark(path, "开始前取消")
        group = self.store.summary()["groups"][0]
        self.assertEqual(group["attempts"], 0)
        self.assertEqual(group["non_attempts"], 1)
        self.assertIsNone(group["success_rate"])
        with self.assertRaisesRegex(ValueError, "已经开始"):
            self.mark(self.make_log(), "开始前取消")

    def test_error_log_cannot_be_marked_success(self):
        with self.assertRaisesRegex(ValueError, "程序或清理异常"):
            self.mark(self.make_log(error=True))

    def test_revision_preserves_history_without_duplicate_attempt(self):
        path = self.make_log()
        self.mark(path)
        self.mark(path, "放置失败")
        history = next(iter(json.loads(self.store.path.read_text(encoding="utf-8"))["records"].values()))
        self.assertEqual([row["outcome"] for row in history], [SUCCESS, "放置失败"])
        group = self.store.summary()["groups"][0]
        self.assertEqual(group["attempts"], 1)
        self.assertEqual(group["successes"], 0)

    def test_log_copy_is_not_double_counted(self):
        path = self.make_log()
        self.mark(path)
        (self.logs / "实机抓取测试_软件副本.jsonl").write_bytes(path.read_bytes())
        report = self.store.summary()
        self.assertEqual(report["groups"][0]["attempts"], 1)
        self.assertEqual(len(report["unreadable_or_duplicate_logs"]), 1)

    def test_changed_log_loses_annotation_binding(self):
        path = self.make_log()
        self.mark(path)
        path.write_text(path.read_text(encoding="utf-8").replace('"fixture_id": 1', '"fixture_id": 9'), encoding="utf-8")
        report = self.store.summary()
        self.assertEqual(report["groups"][0]["pending_attempts"], 1)
        self.assertIsNone(report["groups"][0]["success_rate"])
        self.assertEqual(len(report["annotations_without_matching_log"]), 1)

    def test_partial_corrupt_and_external_logs_cannot_be_marked(self):
        path = self.make_log()
        path.write_text('{"type":"配置"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "结束记录"):
            self.mark(path)
        path.write_text('{"type":', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "损坏"):
            self.mark(path)
        external = self.logs.parent / "实机抓取测试_外部.jsonl"
        external.write_text("", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "本项目"):
            self.mark(external)

    def test_concurrent_revisions_are_preserved(self):
        path = self.make_log()
        errors = []

        def mark():
            try:
                self.mark(path)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=mark) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(4)
            self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.store.scan()[0][0]["revision_count"], 2)

    def test_failed_atomic_save_keeps_previous_result(self):
        path = self.make_log()
        self.mark(path)
        original = self.store.path.read_bytes()
        with patch("管理实机评测记录.os.replace", side_effect=OSError("软件测试写入失败")):
            with self.assertRaises(OSError):
                self.mark(path, "抓取失败")
        self.assertEqual(self.store.path.read_bytes(), original)
        self.assertEqual(list(self.store.directory.glob("结果标记暂存_*.json")), [])

    def test_export_without_trials_has_no_success_rate(self):
        json_path, csv_path = self.store.export()
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["groups"], [])
        self.assertEqual(len(csv_path.read_text(encoding="utf-8-sig").splitlines()), 1)

    def test_hidden_window_uses_store_without_starting_robot(self):
        import tkinter as tk
        from 实机结果记录窗口 import ResultsWindow
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        app = ResultsWindow(root, self.store)
        root.update_idletasks()
        self.assertEqual(app.table.get_children(), ())
        self.assertFalse(app.observed.get())
        self.assertIn("尚无", app.summary_text.get("1.0", "end"))
        path = self.make_log()
        app.refresh()
        app.table.selection_set(read_run(path)["log_sha256"])
        app.select()
        self.assertEqual(app.outcome.get(), "")
        app.outcome.set(SUCCESS)
        app.observed.set(True)
        with patch("实机结果记录窗口.messagebox.showinfo"), patch("实机结果记录窗口.messagebox.showerror") as errors:
            app.save()
            app.export()
            errors.assert_not_called()
        self.assertEqual(self.store.summary()["groups"][0]["successes"], 1)
        self.assertFalse(app.observed.get())
        self.assertEqual(len(list(self.store.directory.glob("评测汇总_*.csv"))), 1)


def main():
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvaluationTests))
    report = {"status": "passed" if result.wasSuccessful() and result.testsRun else "failed",
              "created_at": datetime.now().isoformat(timespec="seconds"), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "scope": "result_recording_software_tests",
              "hardware_tested": False, "robot_commands_sent": 0, "task_success_evaluated": False}
    (ROOT / "运行记录/实机评测记录自检.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
