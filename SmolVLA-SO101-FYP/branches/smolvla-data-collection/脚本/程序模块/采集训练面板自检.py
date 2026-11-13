"""用临时元数据、替代软件进程和隐藏窗口检查面板；不训练模型、不连接硬件。"""
import io
import json
import sys
import tempfile
import threading
import tkinter as tk
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from 管理采集训练任务 import ProjectTasks, SoftwareJob
from 管理训练数据来源 import file_sha256
from 采集训练面板 import TrainingPanel

ROOT = Path(__file__).resolve().parents[2]


class FakeProcess:
    def __init__(self, code=0):
        self.pid, self.code, self.terminated = 999999, code, False
        self.stdout = io.StringIO("软件自检替代进程\n")

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -1


class PanelTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "运行环境/临时文件"
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="采集训练自检_", dir=directory)
        self.root = Path(self.temp.name) / "毕业设计项目/机械臂控制项目"
        self.root.mkdir(parents=True)
        self.project = ProjectTasks(self.root, sys.executable)
        self.dataset = self.project.data / "任务演示/模拟元数据批次"
        meta = self.dataset / "meta"
        meta.mkdir(parents=True)
        info = {"fps": 20, "total_episodes": 3, "total_frames": 180,
                "features": {key: {"names": list(range(6))} for key in ("action", "observation.state")}}
        (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
        (meta / "episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": ep, "length": 60}) for ep in range(3)), encoding="utf-8")
        self.model = self.root / "模型/整段留出实验模型"
        self.model.mkdir(parents=True)
        for filename in ("config.json", "dataset_stats.json"):
            (self.model / filename).write_text("{}", encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"software-fixture-not-real-model")
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def source(self, spec):
        return {"dataset_root": str(spec.dataset), "episodes": [{"episode": ep} for ep in spec.episodes]}

    def report(self, spec):
        result = {"status": "passed", "robot_commands_sent": 0}
        if spec.kind == "check":
            result["dataset_root"] = str(spec.dataset)
        elif spec.kind.startswith("train"):
            result.update(training_data=self.source(spec), steps=spec.steps, batch_size=spec.batch_size,
                          output_dir=str(spec.model), scope="training_plan_only", checkpoint_saved=False)
        else:
            result.update(evaluation_data=self.source(spec), model_dir=str(spec.model),
                          checkpoint_sha256=file_sha256(spec.model / "model.safetensors"),
                          data_assessment={"held_out_validation": True}, scope="evaluation_plan_only", model_inference_executed=False)
        return result

    def run_job(self, spec, result=None, code=0):
        def factory(argv, **options):
            self.calls.append((argv, options))
            if result is not None:
                spec.report.write_text(json.dumps(result), encoding="utf-8")
            return FakeProcess(code)
        job = SoftwareJob(self.project, spec, factory).start()
        job.thread.join(5)
        self.assertFalse(job.thread.is_alive())
        return job

    def test_catalog_keeps_failed_batch_and_does_not_call_it_ready(self):
        failed = self.project.data / "任务演示/保存失败"
        failed.mkdir()
        batches = self.project.batches()
        self.assertEqual(sum(item.ready for item in batches), 1)
        self.assertTrue(any(not item.ready and "保存失败" in item.label for item in batches))
        self.assertTrue(failed.exists())
        with self.assertRaises(ValueError):
            self.project.build("train", failed, [0])

    def test_rejects_other_project_and_invalid_episodes(self):
        for path, episodes in ((self.root, [0]), (self.dataset, []), (self.dataset, [0, 0]),
                               (self.dataset, [3]), (self.dataset, [True])):
            with self.assertRaises(ValueError):
                self.project.build("train", path, episodes)

    def test_command_is_argument_array_with_unique_outputs_and_no_shell(self):
        first = self.project.build("train_plan", self.dataset, [1, 0], name="红色方块 位置实验")
        second = self.project.build("train_plan", self.dataset, [0, 1])
        self.assertNotEqual(first.directory, second.directory)
        self.assertNotEqual(first.model, second.model)
        job = self.run_job(first, self.report(first))
        self.assertEqual(job.status, "succeeded")
        argv, options = self.calls[0]
        self.assertIn("--dry-run", argv)
        self.assertFalse(options["shell"])
        self.assertEqual(options["cwd"], self.root)
        self.assertEqual(options["env"]["HF_HUB_OFFLINE"], "1")
        self.assertTrue((first.directory / "控制台输出.log").is_file())
        self.assertEqual(json.loads((first.directory / "任务状态.json").read_text(encoding="utf-8"))["status"], "succeeded")

    def test_rejects_bad_name_or_training_parameters(self):
        for options in ({"name": "../实验"}, {"name": "English"}, {"steps": 0}, {"batch_size": 17}):
            with self.assertRaises(ValueError):
                self.project.build("train", self.dataset, [0], **options)

    def test_zero_exit_without_report_is_failed(self):
        spec = self.project.build("check", self.dataset)
        self.assertEqual(self.run_job(spec).status, "failed")

    def test_bad_reports_do_not_display_success(self):
        for update in ({"checkpoint_saved": True}, {"steps": 12}, {"output_dir": str(self.root)},
                       {"robot_commands_sent": 1}, {"training_data": {"dataset_root": str(self.dataset), "episodes": [{"episode": 2}]}}):
            spec = self.project.build("train_plan", self.dataset, [0, 1])
            result = self.report(spec)
            result.update(update)
            self.assertEqual(self.run_job(spec, result).status, "failed")

    def test_evaluation_checks_model_hash_and_held_out_result(self):
        for update, expected in (({}, "succeeded"), ({"checkpoint_sha256": "0" * 64}, "failed"),
                                  ({"data_assessment": {"held_out_validation": False}}, "failed"),
                                  ({"model_inference_executed": True}, "failed")):
            spec = self.project.build("eval_plan", self.dataset, [2], model=self.model)
            result = self.report(spec)
            result.update(update)
            self.assertEqual(self.run_job(spec, result).status, expected)

    def test_training_cannot_use_plan_as_completed_checkpoint(self):
        spec = self.project.build("train", self.dataset, [0, 1])
        self.assertEqual(self.run_job(spec, self.report(spec)).status, "failed")

    def test_completed_training_requires_bound_checkpoint_and_parameters(self):
        spec = self.project.build("train", self.dataset, [0, 1], steps=1000)
        spec.model.mkdir(parents=True)
        (spec.model / "model.safetensors").write_bytes(b"software-fixture-not-real-training")
        provenance = {"schema_version": 1, "status": "completed", "training_data": self.source(spec),
                      "steps": spec.steps, "batch_size": spec.batch_size,
                      "checkpoint_sha256": file_sha256(spec.model / "model.safetensors")}
        path = spec.model / "训练数据来源.json"
        path.write_text(json.dumps(provenance), encoding="utf-8")
        result = self.report(spec)
        result.update(scope="software_test_fixture", checkpoint_saved=True, checkpoint_sha256=provenance["checkpoint_sha256"])
        self.assertEqual(self.run_job(spec, result).status, "succeeded")
        provenance["steps"] += 1
        path.write_text(json.dumps(provenance), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "训练参数"):
            self.project.validate_result(spec)
        provenance["steps"] = spec.steps
        path.write_text(json.dumps(provenance), encoding="utf-8")
        (spec.model / "model.safetensors").write_bytes(b"changed-software-fixture")
        with self.assertRaisesRegex(ValueError, "不匹配"):
            self.project.validate_result(spec)

    def test_evaluation_cannot_use_plan_as_completed_inference(self):
        spec = self.project.build("eval", self.dataset, [2], model=self.model)
        result = self.report(spec)
        result.update(normalized_mae=0.3, samples=3)
        self.assertEqual(self.run_job(spec, result).status, "failed")

    def test_existing_job_directory_fails_without_overwriting(self):
        spec = self.project.build("check", self.dataset)
        spec.directory.mkdir(parents=True)
        spec.report.write_text("已有原文", encoding="utf-8")
        self.assertEqual(self.run_job(spec).status, "failed")
        self.assertEqual(spec.report.read_text(encoding="utf-8"), "已有原文")
        self.assertFalse(self.calls)

    def test_cancel_before_start_never_spawns(self):
        spec = self.project.build("check", self.dataset)
        job = SoftwareJob(self.project, spec, lambda *args, **kwargs: self.fail("不应创建进程"))
        self.assertTrue(job.cancel())
        job.start().thread.join(5)
        self.assertEqual(job.status, "cancelled")

    def test_cancel_only_terminates_own_live_process(self):
        spec = self.project.build("check", self.dataset)
        job = SoftwareJob(self.project, spec)
        owned, other = FakeProcess(None), FakeProcess(None)
        job.process = owned
        self.assertTrue(job.cancel())
        self.assertTrue(owned.terminated)
        self.assertFalse(other.terminated)
        self.assertFalse(job.cancel())

    def test_cancel_running_job_preserves_log_and_state(self):
        spec = self.project.build("check", self.dataset)
        entered, stopped = threading.Event(), threading.Event()
        class LiveProcess(FakeProcess):
            def __init__(self):
                super().__init__(None)
                self.stdout = self
            def __iter__(self):
                entered.set()
                stopped.wait(5)
                return iter(["已停止替代软件任务\n"])
            def terminate(self):
                super().terminate()
                stopped.set()
            def close(self):
                pass
        owned = LiveProcess()
        job = SoftwareJob(self.project, spec, lambda *args, **kwargs: owned).start()
        try:
            self.assertTrue(entered.wait(5))
            self.assertTrue(job.cancel())
            job.thread.join(5)
            self.assertFalse(job.thread.is_alive())
            self.assertEqual(job.status, "cancelled")
            self.assertTrue(owned.terminated)
            self.assertTrue((spec.directory / "任务状态.json").exists())
        finally:
            stopped.set()
            job.thread.join(5)

    def test_nonzero_exit_retains_failed_outputs(self):
        spec = self.project.build("train_plan", self.dataset, [0])
        self.assertEqual(self.run_job(spec, self.report(spec), code=2).status, "failed")
        self.assertTrue(spec.report.exists())

    def test_operator_whitelist_deduplicates_without_robot_autostart(self):
        opened = []
        def factory(*args, **kwargs):
            opened.append((args, kwargs))
            return FakeProcess(None)
        with self.assertRaises(ValueError):
            self.project.launch_operator("实机抓取测试.py", factory)
        one = self.project.launch_operator("录制任务演示.py", factory)
        self.assertIs(self.project.launch_operator("录制任务演示.py", factory), one)
        self.assertEqual(len(opened), 1)
        self.assertEqual(Path(opened[0][0][0][1]), self.root / "脚本/数据采集/录制任务演示.py")

    def test_held_out_batch_remains_evaluable_but_cannot_train(self):
        (self.dataset / "采集配置.json").write_text(json.dumps({"split": "held_out"}), encoding="utf-8")
        self.assertEqual(self.project.build("eval_plan", self.dataset, [0], model=self.model).kind, "eval_plan")
        with self.assertRaises(ValueError):
            self.project.build("train_plan", self.dataset, [0])

    def test_sealed_batch_is_visible_but_not_ready(self):
        (self.dataset / "批次封存状态.json").write_text("{}", encoding="utf-8")
        self.assertFalse(self.project.batches()[0].ready)
        with self.assertRaises(ValueError):
            self.project.build("train_plan", self.dataset, [0])

    def test_hidden_window_selects_disjoint_defaults_and_validates_job(self):
        window = tk.Tk()
        window.withdraw()
        jobs = []
        def factory(project, spec):
            result = self.report(spec)
            def spawn(argv, **options):
                spec.report.write_text(json.dumps(result), encoding="utf-8")
                self.calls.append((argv, options))
                return FakeProcess()
            job = SoftwareJob(project, spec, spawn)
            jobs.append(job)
            return job
        try:
            panel = TrainingPanel(window, self.project, factory)
            self.assertTrue(panel.strict.get())
            self.assertEqual(panel.train_list.curselection(), (0, 1))
            self.assertEqual(panel.eval_list.curselection(), (2,))
            self.assertFalse(self.project.operator_processes)
            panel.start("train_plan")
            jobs[-1].thread.join(5)
            panel.poll()
            self.assertIn("尚未训练", panel.status.get())
            panel.start("eval_plan")
            jobs[-1].thread.join(5)
            panel.poll()
            self.assertIn("尚未运行模型推理", panel.status.get())
            self.project.operator_processes["录制任务演示.py"] = FakeProcess(None)
            panel.close()
            self.assertFalse(self.project.operator_processes["录制任务演示.py"].terminated)
        finally:
            window.destroy()


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PanelTests))
    report = {"status": "passed" if result.wasSuccessful() and result.testsRun else "failed",
              "created_at": datetime.now().isoformat(timespec="seconds"), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "hardware_tested": False,
              "robot_commands_sent": 0, "scope": "collection_training_panel_software_tests"}
    target = ROOT / "运行记录/采集训练面板自检.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)
