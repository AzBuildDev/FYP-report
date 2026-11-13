"""用临时 parquet、归一化张量和软件报告验证对比规则；不训练、不推理、不连接硬件。"""
import copy
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import save_file

from 管理离线实验对比 import (UNITS, baseline_rows, compare_reports, sample_plan, summarize, validate_report, vector)
from 管理训练数据来源 import dataset_source, file_sha256, training_stats
from 汇总离线实验对比 import export_result

ROOT = Path(__file__).resolve().parents[2]


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "运行环境/临时文件"
        directory.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="离线对比自检_", dir=directory)
        self.root = Path(self.temp.name) / "毕业设计项目/机械臂控制项目"
        self.root.mkdir(parents=True)
        self.dataset = self.root.parent / "训练数据采集/机械臂控制项目/任务演示/软件测试数据"
        (self.dataset / "meta").mkdir(parents=True)
        (self.dataset / "data/chunk-000").mkdir(parents=True)
        names = [f"关节{i}.pos" for i in range(6)]
        self.info = {"fps": 20, "chunks_size": 1000, "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                     "features": {"action": {"dtype": "float32", "names": names}, "observation.state": {"dtype": "float32", "names": names},
                                  "observation.images.front": {"dtype": "video"}}}
        self.tasks = [{"task_index": 0, "task": "软件测试，不是真实演示"}]
        (self.dataset / "meta/info.json").write_text(json.dumps(self.info), encoding="utf-8")
        (self.dataset / "meta/tasks.jsonl").write_text(json.dumps(self.tasks[0]), encoding="utf-8")
        self.tables = []
        for ep in range(3):
            action = [[float(t * (ep + 1) + i - 80) for i in range(6)] for t in range(60)]
            table = {"episode_index": [ep] * 60, "action": action,
                     "observation.state": [[x + 1 for x in row] for row in action],
                     "timestamp": [t / 20 for t in range(60)], "task_index": [0] * 60}
            self.tables.append(table)
            pq.write_table(pa.table(table), self.dataset / f"data/chunk-000/episode_{ep:06d}.parquet")
            video = self.dataset / self.info["video_path"].format(episode_chunk=0, episode_index=ep, video_key="observation.images.front")
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"软件测试视频指纹 {ep}，不是真实视频".encode())
        self.model = self.make_model("模型甲", [0])
        self.report = self.make_report(self.model)
        self.path = self.root / "软件测试报告.json"
        self.save_report(self.path, self.report)

    def tearDown(self):
        self.temp.cleanup()

    def make_model(self, name, ids):
        model = self.root / "模型" / name
        model.mkdir(parents=True)
        selected = [self.tables[index] for index in ids]
        stats = training_stats(selected)
        tensors = {f"{prefix}.buffer_{feature.replace('.', '_')}.{field}": stats[feature][field]
                   for prefix, feature in (("normalize_inputs", "observation.state"), ("normalize_targets", "action"),
                                           ("unnormalize_outputs", "action")) for field in ("mean", "std")}
        save_file(tensors, str(model / "model.safetensors"))
        source = dataset_source(self.dataset, self.info, selected, self.tasks)
        record = {"schema_version": 1, "status": "completed", "training_data": source,
                  "checkpoint_sha256": file_sha256(model / "model.safetensors"), "software_test_fixture": True}
        (model / "训练数据来源.json").write_text(json.dumps(record), encoding="utf-8")
        (model / "config.json").write_text('{"chunk_size": 50}', encoding="utf-8")
        (model / "dataset_stats.json").write_text(json.dumps({key: {field: value.tolist() for field, value in item.items()}
                                                               for key, item in stats.items()}), encoding="utf-8")
        return model

    def make_report(self, model):
        stats = json.loads((model / "dataset_stats.json").read_text(encoding="utf-8"))
        raw = [1., 2., 3., 4., 5., 6.]
        normalized = float((np.asarray(raw) / stats["action"]["std"]).mean())
        return {"status": "passed", "scope": "held_out_demonstration_actions", "held_out_validation": True,
                "robot_commands_sent": 0, "task_success_evaluated": False, "model_inference_executed": True,
                "model_dir": str(model), "checkpoint_sha256": file_sha256(model / "model.safetensors"),
                "evaluation_data": dataset_source(self.dataset, self.info, [self.tables[1]], self.tasks),
                "seed": 42, "chunk_size": 50, "samples": 2, "joint_names": self.info["features"]["action"]["names"],
                "action_units": UNITS, "mae_per_joint": raw, "normalized_mae": normalized,
                "first_action_mae_per_joint": raw, "created_at": "软件测试", "software_test_fixture": True,
                "samples_detail": [{"episode": 1, "frame": frame, "mae_per_joint": raw} for frame in (0, 10)]}

    def save_report(self, path, report):
        path.write_text(json.dumps(report), encoding="utf-8")

    def validation(self, report=None):
        source = json.loads((self.model / "训练数据来源.json").read_text(encoding="utf-8"))["training_data"]
        stats = json.loads((self.model / "dataset_stats.json").read_text(encoding="utf-8"))
        return validate_report(report or self.report, self.model, self.report["evaluation_data"], source, stats)

    def test_baselines_use_current_state_and_training_mean(self):
        table = {"episode_index": [0, 0, 0], "action": [[2.] * 6, [4.] * 6, [6.] * 6],
                 "observation.state": [[1.] * 6, [100.] * 6, [200.] * 6]}
        rows = baseline_rows([table], [(0, 0)], 3, [-1.] * 6, [2.] * 6)
        self.assertAlmostEqual(rows[0]["common_scale_mae"], 1.5)
        self.assertAlmostEqual(rows[1]["common_scale_mae"], 2.5)
        self.assertEqual(rows[0]["first_action_mae_per_joint"], [1.] * 6)

    def test_common_scale_replaces_incomparable_model_scales(self):
        model2 = self.make_model("模型乙", [2])
        report2 = self.make_report(model2)
        path2 = self.root / "软件测试报告乙.json"
        self.save_report(path2, report2)
        result = compare_reports(self.root, [self.path, path2], self.model)
        one, two = result["rows"][:2]
        self.assertNotAlmostEqual(one["own_scale_mae"], two["own_scale_mae"])
        self.assertAlmostEqual(one["common_scale_mae"], two["common_scale_mae"])
        self.assertFalse(result["new_model_inference_executed"])

    def test_invalid_vectors_and_negative_error_rejected(self):
        for value in ([1.] * 5, [float("nan")] * 6, [-1.] * 6):
            with self.assertRaises(ValueError):
                vector(value, "测试误差")
        with self.assertRaises(ValueError):
            summarize("测试", "test", [[-1.] * 6], [1.] * 6)
        with self.assertRaises(ValueError):
            vector([0.] * 6, "标准差", positive=True)

    def test_plans_training_fit_and_hardware_reports_rejected(self):
        for update in ({"scope": "evaluation_plan_only"}, {"held_out_validation": False},
                       {"robot_commands_sent": 1}, {"model_inference_executed": False}, {"task_success_evaluated": True}):
            report = copy.deepcopy(self.report)
            report.update(update)
            with self.assertRaises(ValueError):
                self.validation(report)

    def test_sample_window_boundaries_duplicates_and_missing_rows_rejected(self):
        for change in ("outside", "duplicate", "missing", "boolean"):
            report = copy.deepcopy(self.report)
            if change == "outside":
                report["samples_detail"][-1]["frame"] = 11
            elif change == "duplicate":
                report["samples_detail"][-1]["frame"] = 0
            elif change == "missing":
                report["samples"] = 3
            else:
                report["samples_detail"][-1]["frame"] = True
            with self.assertRaises(ValueError):
                sample_plan(report)

    def test_wrong_aggregate_and_scale_rejected(self):
        for field, value in (("mae_per_joint", [99.] * 6), ("normalized_mae", float("nan"))):
            report = copy.deepcopy(self.report)
            report[field] = value
            with self.assertRaises(ValueError):
                self.validation(report)

    def test_different_windows_or_seeds_cannot_be_compared(self):
        model2 = self.make_model("模型乙", [2])
        for change in ("seed", "frame"):
            report2 = self.make_report(model2)
            if change == "seed":
                report2["seed"] = 24
            else:
                report2["samples_detail"][-1]["frame"] = 9
            path2 = self.root / "软件测试报告乙.json"
            self.save_report(path2, report2)
            with self.assertRaisesRegex(ValueError, "不能直接对比"):
                compare_reports(self.root, [self.path, path2], self.model)

    def test_same_checkpoint_is_not_two_models(self):
        with self.assertRaisesRegex(ValueError, "重复报告"):
            compare_reports(self.root, [self.path, self.path], self.model)

    def test_reference_mean_must_not_include_validation_data(self):
        unsafe_reference = self.make_model("重叠尺度模型", [1])
        with self.assertRaisesRegex(ValueError, "包含验证数据"):
            compare_reports(self.root, [self.path], unsafe_reference)

    def test_changed_evaluation_video_or_action_invalidates_old_report(self):
        video = next((self.dataset / "videos").rglob("episode_000001.mp4"))
        video.write_bytes(b"modified-test-content")
        with self.assertRaisesRegex(ValueError, "内容已变化"):
            compare_reports(self.root, [self.path], self.model)

    def test_modified_training_statistics_rejected(self):
        stats = json.loads((self.model / "dataset_stats.json").read_text(encoding="utf-8"))
        stats["action"]["std"][0] += 1
        (self.model / "dataset_stats.json").write_text(json.dumps(stats), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "统计文件"):
            compare_reports(self.root, [self.path], self.model)

    def test_unknown_training_provenance_cannot_enter_comparison(self):
        (self.model / "训练数据来源.json").unlink()
        with self.assertRaisesRegex(ValueError, "来源未知"):
            compare_reports(self.root, [self.path], self.model)

    def test_new_protocol_fingerprints_and_legacy_limitation(self):
        self.assertFalse(self.validation()[2])
        report = copy.deepcopy(self.report)
        report.update(metric_protocol_version=1, model_config_sha256=file_sha256(self.model / "config.json"),
                      dataset_stats_sha256=file_sha256(self.model / "dataset_stats.json"))
        self.assertTrue(self.validation(report)[2])
        (self.model / "config.json").write_text('{"chunk_size": 51}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "配置或统计"):
            self.validation(report)

    def test_csv_markdown_svg_export_and_no_overwrite(self):
        result = compare_reports(self.root, [self.path], self.model)
        output = self.root / "软件对比输出"
        export_result(result, output)
        self.assertEqual(len(list(output.iterdir())), 5)
        self.assertIn("不代表实机", (output / "对比说明.md").read_text(encoding="utf-8"))
        self.assertIn("<svg", (output / "动作误差对比.svg").read_text(encoding="utf-8"))
        self.assertEqual(ET.parse(output / "动作误差对比.svg").getroot().tag, "{http://www.w3.org/2000/svg}svg")
        self.assertFalse(any(line.endswith(" ") for line in (output / "动作误差对比.svg").read_text(encoding="utf-8").splitlines()))
        with self.assertRaises(FileExistsError):
            export_result(result, output)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonTests))
    report = {"status": "passed" if result.wasSuccessful() and result.testsRun else "failed",
              "created_at": datetime.now().isoformat(timespec="seconds"), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "scope": "offline_comparison_software_tests",
              "hardware_tested": False, "robot_commands_sent": 0, "model_inference_executed": False, "new_training_executed": False}
    (ROOT / "运行记录/离线实验对比自检.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)
