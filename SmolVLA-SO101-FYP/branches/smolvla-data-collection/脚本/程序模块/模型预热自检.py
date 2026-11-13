"""检查模型启动时效、输入隔离与取消，不加载模型或连接硬件。"""
import threading
import time
import unittest

import numpy as np
from 预热模型与检查时效 import warm_model


class WarmupTests(unittest.TestCase):
    def run_warm(self, predict, **kwargs):
        return warm_model(predict, np.zeros(6), np.full((8, 8, 3), 80),
                          kwargs.pop("stop", threading.Event()), chunk_size=50, **kwargs)

    def test_cold_result_is_discarded_and_not_subject_to_warm_budget(self):
        calls = []
        def predict(state, image):
            calls.append(True)
            time.sleep(0.04 if len(calls) == 1 else 0.001)
            return np.zeros((50, 6))
        report = self.run_warm(predict, hz=1000, reserve=0.025)
        self.assertEqual(report["status"], "ready")
        self.assertGreater(report["samples"][0]["inference_s"], report["ready_budget_s"])
        self.assertFalse(report["prediction_reused"])
        self.assertNotIn("actions", report)

    def test_warm_slow_result_is_rejected(self):
        def predict(state, image):
            time.sleep(0.04)
            return np.zeros((50, 6))
        with self.assertRaisesRegex(RuntimeError, "超过启动预算"):
            self.run_warm(predict, hz=1000, reserve=0.025)

    def test_shape_nan_and_model_error_are_rejected(self):
        for output in (np.zeros((49, 6)), np.full((50, 6), np.nan)):
            with self.subTest(shape=output.shape), self.assertRaisesRegex(RuntimeError, "输出"):
                self.run_warm(lambda *args: output)
        def fail(*args):
            raise ValueError("模拟 GPU 错误")
        with self.assertRaisesRegex(RuntimeError, "模拟 GPU 错误"):
            self.run_warm(fail)

    def test_stop_does_not_wait_for_inference(self):
        stop, running, release = threading.Event(), threading.Event(), threading.Event()
        workers, results = [], []
        def predict(*args):
            running.set()
            release.wait(2)
            return np.zeros((50, 6))
        caller = threading.Thread(target=lambda: results.append(self.run_warm(predict, stop=stop, workers=workers)))
        caller.start()
        try:
            self.assertTrue(running.wait(1))
            stop.set()
            caller.join(0.5)
            self.assertFalse(caller.is_alive())
            self.assertTrue(workers[0].is_alive())
            self.assertEqual(results[0]["status"], "cancelled")
        finally:
            release.set()
            caller.join(2)
            for worker in workers:
                worker.join(2)

    def test_timeout_refuses_without_waiting_for_model(self):
        release, workers = threading.Event(), []
        def predict(*args):
            release.wait(2)
            return np.zeros((50, 6))
        try:
            with self.assertRaisesRegex(RuntimeError, "超时"):
                self.run_warm(predict, timeout=0.03, workers=workers)
            self.assertTrue(workers[0].is_alive())
        finally:
            release.set()
            for worker in workers:
                worker.join(2)

    def test_inputs_copied_and_pre_cancel_does_not_infer(self):
        state, image = np.zeros(6), np.full((8, 8, 3), 80)
        def predict(s, im):
            s[:] = 10
            im[:] = 0
            return np.zeros((50, 6))
        warm_model(predict, state, image, threading.Event(), chunk_size=50)
        self.assertTrue(np.all(state == 0))
        self.assertTrue(np.all(image == 80))
        stop = threading.Event()
        stop.set()
        report = self.run_warm(lambda *args: self.fail("已取消仍推理"), stop=stop)
        self.assertEqual(report["samples"], [])

    def test_invalid_budget_is_rejected(self):
        for kwargs in ({"hz": 0}, {"timeout": -1}, {"reserve": 2.5}, {"runs": 1}, {"hz": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_warm(lambda *args: np.zeros((50, 6)), **kwargs)


if __name__ == "__main__":
    unittest.main()
