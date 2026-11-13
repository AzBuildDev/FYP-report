"""以替代预测缓存检查延迟索引、过期窗口、加权指标及来源绑定；不执行模型或硬件。"""
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

from 分析执行窗口误差 import analyze_report, load_predictions, summarize_case, window_indices
from 管理训练数据来源 import file_sha256
from 离线实验对比自检 import ComparisonTests
from 汇总执行窗口评估 import export_result

ROOT = Path(__file__).resolve().parents[2]


class WindowTests(unittest.TestCase):
    def sample(self, offset=2.):
        expected = np.repeat(np.arange(50, dtype=float)[:, None], 6, axis=1)
        return {"episode": 0, "frame": 0, "expected": expected, "predicted": expected + offset,
                "state": np.zeros(6), "train_mean": np.full(6, 5.)}

    def test_start_offset_and_execute_count_follow_control_formula(self):
        self.assertEqual(window_indices(50, 0), list(range(20)))
        self.assertEqual(window_indices(50, .8), list(range(16, 36)))
        self.assertEqual(window_indices(50, 2.4), [48, 49])
        self.assertEqual(window_indices(50, 2.5), [])
        for delay in (.799999, .8, .800001):
            indices = window_indices(50, delay)
            self.assertEqual(indices[0], int(delay * 20))
            self.assertEqual(len(indices), 20)

    def test_invalid_parameters_rejected(self):
        for options in ({"chunk_size": 0}, {"delay_s": -1}, {"delay_s": float("nan")},
                        {"execute_limit": 0}, {"execute_limit": True}, {"hz": 0}):
            args = {"chunk_size": 50, "delay_s": .8}
            args.update(options)
            with self.assertRaises(ValueError):
                window_indices(**args)

    def test_methods_use_same_indices_without_future_state_leak(self):
        case = summarize_case("测试", [self.sample()], [1.] * 6, [.8])
        self.assertEqual(case["scored_targets"], 20)
        self.assertEqual([row["common_scale_mae"] for row in case["rows"]], [2., 25.5, 20.5])
        self.assertEqual(case["windows"][0]["selected_indices"], list(range(16, 36)))

    def test_expired_block_has_no_zero_mae(self):
        case = summarize_case("过期", [self.sample()], [1.] * 6, [3.])
        self.assertEqual(case["expired_windows"], 1)
        self.assertEqual(case["scored_targets"], 0)
        self.assertTrue(all(row["common_scale_mae"] is None for row in case["rows"]))

    def test_partial_windows_are_weighted_by_target_count(self):
        case = summarize_case("加权", [self.sample(1.), self.sample(10.)], [1.] * 6, [0., 2.4])
        self.assertEqual(case["scored_targets"], 22)
        self.assertAlmostEqual(case["rows"][0]["common_scale_mae"], (20 + 20) / 22)

    def test_nonfinite_predictions_rejected_even_if_expired(self):
        sample = self.sample()
        sample["predicted"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            summarize_case("过期", [sample], [1.] * 6, [3.])

    def test_delay_count_matches_window_count(self):
        with self.assertRaises(ValueError):
            summarize_case("长度", [self.sample()], [1.] * 6, [])

    def test_expired_chart_export_is_valid_and_no_overwrite(self):
        target = ROOT / "运行环境/临时文件"
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="窗口图表自检_", dir=target) as directory:
            output = Path(directory) / "报告"
            result = {"model": "软件测试", "hz": 20, "execute_limit": 20, "source_evaluation_date": "软件测试",
                      "notes": ["不是真实控制"], "cases": [summarize_case("过期", [self.sample()], [1.] * 6, [3.])]}
            export_result(result, output)
            self.assertEqual(len(list(output.iterdir())), 5)
            self.assertIn("不计算", (output / "执行窗口说明.md").read_text(encoding="utf-8"))
            self.assertEqual(ET.parse(output / "执行窗口误差.svg").getroot().tag, "{http://www.w3.org/2000/svg}svg")
            with self.assertRaises(FileExistsError):
                export_result(result, output)


class CacheTests(unittest.TestCase):
    def setUp(self):
        # 复用现有临时数据构造器，不重复执行它的其他测试。
        self.fixture = ComparisonTests(methodName="runTest")
        self.fixture.setUp()
        self.root, self.path, self.report = self.fixture.root, self.fixture.path, self.fixture.report
        self.cache = self.path.with_name("软件预测缓存.npz")
        tables = self.fixture.tables
        predictions = [np.asarray(tables[1]["action"][row["frame"]:row["frame"] + 50], dtype=np.float32)
                       + np.arange(1, 7, dtype=np.float32) for row in self.report["samples_detail"]]
        np.savez_compressed(self.cache, predicted_actions=np.stack(predictions))
        for row in self.report["samples_detail"]:
            row["inference_s"] = .8
        self.report["prediction_cache"] = {"filename": self.cache.name, "sha256": file_sha256(self.cache),
                                           "format": "npz_no_pickle", "shape": [2, 50, 6], "local_only": True}
        self.fixture.save_report(self.path, self.report)

    def tearDown(self):
        self.fixture.tearDown()

    def test_bound_cache_can_be_reused_without_model_execution(self):
        result = analyze_report(self.root, self.path)
        self.assertFalse(result["new_model_inference_executed"])
        self.assertEqual(result["robot_commands_sent"], 0)
        self.assertEqual(result["cases"][0]["scored_targets"], 40)
        self.assertEqual(result["cases"][-1]["windows"][0]["selected_indices"], list(range(16, 36)))

    def test_missing_cache_cannot_be_reconstructed_from_mean(self):
        self.report.pop("prediction_cache")
        self.fixture.save_report(self.path, self.report)
        with self.assertRaisesRegex(ValueError, "不能从平均误差"):
            analyze_report(self.root, self.path)

    def test_modified_cache_and_wrong_shape_rejected(self):
        np.savez_compressed(self.cache, predicted_actions=np.zeros((2, 49, 6)))
        with self.assertRaisesRegex(ValueError, "内容指纹"):
            load_predictions(self.report, self.path)
        self.report["prediction_cache"]["sha256"] = file_sha256(self.cache)
        with self.assertRaisesRegex(ValueError, "形状"):
            load_predictions(self.report, self.path)

    def test_cache_cannot_escape_report_directory(self):
        self.report["prediction_cache"]["filename"] = "../其他缓存.npz"
        with self.assertRaisesRegex(ValueError, "缓存路径"):
            load_predictions(self.report, self.path)

    def test_cache_values_must_match_original_error_metrics(self):
        np.savez_compressed(self.cache, predicted_actions=np.zeros((2, 50, 6)))
        self.report["prediction_cache"]["sha256"] = file_sha256(self.cache)
        self.fixture.save_report(self.path, self.report)
        with self.assertRaisesRegex(ValueError, "原始评估明细"):
            analyze_report(self.root, self.path)

    def test_plan_cannot_enter_window_analysis(self):
        self.report["scope"] = "evaluation_plan_only"
        self.fixture.save_report(self.path, self.report)
        with self.assertRaises(ValueError):
            analyze_report(self.root, self.path)


if __name__ == "__main__":
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (WindowTests, CacheTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"status": "passed" if result.wasSuccessful() and result.testsRun else "failed",
              "created_at": datetime.now().isoformat(timespec="seconds"), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "scope": "execution_window_software_tests",
              "hardware_tested": False, "robot_commands_sent": 0, "model_inference_executed": False, "new_training_executed": False}
    (ROOT / "运行记录/执行窗口误差自检.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)
