"""验证异步控制的节拍、停止、过期动作和线程隔离，不连接硬件。"""
import threading
import time
import unittest

import numpy as np
from 异步动作控制 import InferenceWorker, run_async_control


class AsyncTests(unittest.TestCase):
    def setUp(self):
        self.stop = threading.Event()
        self.sent = []
        self.events = []
        self.state = np.zeros(6, dtype=np.float32)
        self.main_thread = threading.get_ident()

    def predict(self, state, image):
        self.assertNotEqual(threading.get_ident(), self.main_thread)
        time.sleep(0.08)
        return np.repeat((state + 0.1)[None], 50, axis=0)

    def read_state(self):
        self.assertEqual(threading.get_ident(), self.main_thread)
        return self.state.copy()

    def frame(self):
        return time.monotonic(), np.full((8, 8, 3), 80, dtype=np.uint8)

    def send(self, target):
        self.assertEqual(threading.get_ident(), self.main_thread)
        self.sent.append(time.monotonic())
        self.state = target.copy()

    def run_control(self, **kwargs):
        return run_async_control(kwargs.pop("predict", self.predict), kwargs.pop("read_state", self.read_state),
                                 kwargs.pop("read_frame", self.frame), self.send, self.stop,
                                 np.full(6, -100), np.full(6, 100), on_event=self.events.append, **kwargs)

    def test_overlap_and_thread_isolation(self):
        result = self.run_control(duration=0.8, hz=40)
        self.assertGreater(result["inferences"], 4)
        self.assertGreater(result["sent_commands"], 20)
        self.assertGreater(result["active_control_hz"], 30)

    def test_stop_does_not_wait_for_inference(self):
        entered, release = threading.Event(), threading.Event()

        def blocked(state, image):
            entered.set()
            release.wait(2)
            return np.ones((50, 6))

        def cancel():
            entered.wait(1)
            self.stop.set()

        canceller = threading.Thread(target=cancel)
        canceller.start()
        start = time.monotonic()
        try:
            result = self.run_control(predict=blocked, duration=2)
            self.assertLess(time.monotonic() - start, 0.4)
            self.assertEqual(result["sent_commands"], 0)
        finally:
            release.set()
            canceller.join(1)

    def test_camera_disconnect_while_inference_pending(self):
        count = 0

        def disconnect():
            nonlocal count
            count += 1
            return (time.monotonic(), None) if count >= 2 else self.frame()

        with self.assertRaisesRegex(RuntimeError, "相机画面中断"):
            self.run_control(read_frame=disconnect, duration=1)
        self.assertFalse(self.sent)

    def test_expired_chunk_is_not_sent(self):
        def short_chunk(state, image):
            time.sleep(0.08)
            return np.ones((1, 6))

        with self.assertRaisesRegex(RuntimeError, "有效期"):
            self.run_control(predict=short_chunk, duration=0.5, hz=40)
        self.assertFalse(self.sent)

    def test_invalid_prediction_stops(self):
        for actions in (np.full((50, 6), np.nan), np.ones((50, 5))):
            with self.assertRaisesRegex(RuntimeError, "后台模型推理失败"):
                self.run_control(predict=lambda state, image: actions, duration=0.3)
        self.assertFalse(self.sent)

    def test_stop_during_read_prevents_send(self):
        def read():
            if self.events:
                self.stop.set()
            return self.state.copy()

        result = self.run_control(read_state=read, duration=0.5)
        self.assertEqual(result["stop_reason"], "user_stop")

    def test_worker_copies_inputs(self):
        allow = threading.Event()

        def predict(state, image):
            allow.wait(1)
            self.assertEqual(float(state[0]), 0)
            self.assertEqual(int(image[0, 0, 0]), 80)
            return np.ones((50, 6))

        worker = InferenceWorker(predict)
        state, image = self.state.copy(), self.frame()[1]
        worker.submit(state, image, time.monotonic())
        state[:] = 99
        image[:] = 0
        with self.assertRaises(RuntimeError):
            worker.submit(state, image, time.monotonic())
        allow.set()
        deadline = time.monotonic() + 1
        result = None
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            time.sleep(0.005)
        worker.close()
        self.assertIsNotNone(result)

    def test_inference_timeout(self):
        with self.assertRaisesRegex(RuntimeError, "推理超时"):
            self.run_control(predict=self.predict, duration=0.5, inference_timeout=0.02)
        self.assertFalse(self.sent)

    def test_keyboard_interrupt_is_logged_as_user_stop(self):
        def interrupted_frame():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            self.run_control(read_frame=interrupted_frame, duration=0.5)
        self.assertEqual(self.events[-1]["type"], "async_end")
        self.assertEqual(self.events[-1]["stop_reason"], "user_stop")
        self.assertFalse(self.sent)


if __name__ == "__main__":
    unittest.main()
