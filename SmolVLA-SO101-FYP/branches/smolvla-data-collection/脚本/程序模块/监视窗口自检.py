"""隐藏的独立 Tk 进程退出和停止按钮检查，不加载模型或访问硬件。"""
from pathlib import Path
import threading
import time
import unittest
import numpy as np
from 实机测试监视窗口 import run_monitor

ROOT = Path(__file__).resolve().parents[2]


class MonitorTests(unittest.TestCase):
    def test_stop_button_request_returns_and_removes_temporary_preview(self):
        camera_stop, user_stop = threading.Event(), threading.Event()
        run_monitor(ROOT, lambda: (time.monotonic(), np.full((8, 8, 3), 80, dtype=np.uint8)),
                    camera_stop, user_stop, lambda: "仅软件测试", hidden=True, test_stop_after=0.25)
        self.assertTrue(user_stop.is_set())
        self.assertFalse(any((ROOT / "运行环境/临时文件").glob("实机监视预览_*")))

    def test_control_stop_closes_window_without_user_stop(self):
        camera_stop, user_stop, errors = threading.Event(), threading.Event(), []
        def run():
            try:
                run_monitor(ROOT, lambda: (time.monotonic(), np.full((8, 8, 3), 80, dtype=np.uint8)),
                            camera_stop, user_stop, lambda: "仅软件测试", hidden=True)
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(1)
        camera_stop.set()
        thread.join(8)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(user_stop.is_set())


if __name__ == "__main__":
    unittest.main()
