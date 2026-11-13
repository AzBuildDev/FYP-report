"""隐藏的独立 Tk 进程退出和停止按钮检查，不加载模型或访问硬件。"""
from pathlib import Path
import os
import tempfile
import threading
import time
import unittest
import numpy as np
from 实机测试监视窗口 import run_monitor, run_embedded_monitor, wait_for_preview

ROOT = Path(__file__).resolve().parents[2]


class MonitorTests(unittest.TestCase):
    def test_embedded_monitor_publishes_both_views_and_honors_stop(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT / "运行环境/临时文件") as temp:
            directory = Path(temp)
            (directory / "窗口心跳").touch()
            image = np.zeros((48, 128, 3), dtype=np.uint8)
            image[:, :64, 2] = 255
            image[:, 64:, 0] = 255
            camera_stop, user_stop = threading.Event(), threading.Event()
            errors = []
            def publish():
                try:
                    run_embedded_monitor(ROOT, directory, lambda: (time.monotonic(), image),
                                         camera_stop, user_stop, lambda: "双摄测试")
                except Exception as exc:
                    errors.append(exc)
            thread = threading.Thread(target=publish)
            thread.start()
            try:
                deadline = time.monotonic() + 2
                while not (directory / "当前预览.jpg").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                with Image.open(directory / "当前预览.jpg") as decoded:
                    self.assertGreater(decoded.getpixel((10, 20))[0], 200)
                    self.assertGreater(decoded.getpixel((100, 20))[2], 200)
                (directory / "请求停止").touch()
                thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertTrue(user_stop.is_set())
                self.assertEqual(errors, [])
            finally:
                camera_stop.set()
                thread.join(2)

    def test_unresponsive_window_rejects_preview(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "运行环境/临时文件") as temp:
            directory = Path(temp)
            heartbeat = directory / "窗口心跳"
            heartbeat.touch()
            os.utime(heartbeat, (time.time() - 10,) * 2)
            with self.assertRaisesRegex(RuntimeError, "停止响应"):
                run_embedded_monitor(ROOT, directory, lambda: (0, None), threading.Event(),
                                     threading.Event(), lambda: "")

    def test_preview_ack_required_and_cancellable(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "运行环境/临时文件") as temp:
            stop = threading.Event()
            with self.assertRaisesRegex(RuntimeError, "未启动控制"):
                wait_for_preview(temp, stop, timeout=0.01)
            stop.set()
            self.assertFalse(wait_for_preview(temp, stop))
            stop.clear()
            (Path(temp) / "预览已显示").touch()
            self.assertTrue(wait_for_preview(temp, stop))

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
