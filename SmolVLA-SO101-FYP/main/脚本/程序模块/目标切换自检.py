"""目标立即切换、活动动作清空和旧推理丢弃的离线测试，不连接硬件。"""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from 红蓝抓取任务 import TASKS
from 运行中目标切换 import TaskSwitch
from 异步动作控制 import run_async_control


class SwitchTests(unittest.TestCase):
    def test_switch_even_while_contact_guard_active(self):
        with tempfile.TemporaryDirectory() as d:
            switch = TaskSwitch(TASKS["RC"], d)
            guard = SimpleNamespace(active=True, contact_position=18.)
            state = np.array([0., 0, 0, 0, 0, 18])
            for i, target in enumerate(["BC", "RC", "RC"], 1):
                (Path(d)/"目标请求.json").write_text(json.dumps(dict(id=i,target=target)), encoding="utf-8")
                self.assertTrue(switch.update(state, i, guard))
                self.assertEqual(switch.current, target)
                self.assertEqual(switch.generation, i)
                self.assertTrue(guard.active)  # 任务切换不关闭夹爪接触保护。
                self.assertFalse(switch.update(state, i + .1, guard))

    def test_active_actions_are_cleared_before_new_inference(self):
        with tempfile.TemporaryDirectory() as d:
            switch = TaskSwitch(TASKS["RC"], d)
            stop = threading.Event()
            sent, events = [], []
            def predict(state, images):
                target = switch.current
                time.sleep(.03)
                return np.full((100, 6), 1. if target == "RC" else 2., dtype=np.float32)
            def send(action):
                sent.append((switch.generation, float(action[0])))
                if len(sent) == 1:
                    (Path(d)/"目标请求.json").write_text(json.dumps(dict(id=1,target="BC")), encoding="utf-8")
                if switch.generation == 1:
                    stop.set()
                return action
            workers = []
            run_async_control(predict, lambda: np.zeros(6),
                              lambda: (time.monotonic()-.002, np.full((4,4,3),100,dtype=np.uint8)),
                              send, stop, np.full(6,-10), np.full(6,10), duration=2, hz=100, max_delta=5,
                              task_switch=switch, on_event=events.append, workers=workers)
            for worker in workers:
                worker.thread.join(1)
            self.assertEqual(sent[0], (0, 1.))
            self.assertTrue(any(generation == 1 for generation, action in sent))
            self.assertTrue(all(action == 2. for generation, action in sent if generation == 1))

    def test_inflight_old_prediction_never_sent(self):
        with tempfile.TemporaryDirectory() as d:
            switch = TaskSwitch(TASKS["RC"], d)
            entered, finish, stop = threading.Event(), threading.Event(), threading.Event()
            calls, sent, events = [], [], []
            def predict(state, images):
                task = switch.current
                calls.append(task)
                if len(calls) == 1:
                    entered.set()
                    self.assertTrue(finish.wait(2))
                return np.full((50, 6), 1. if task == "RC" else 2., dtype=np.float32)
            def change():
                entered.wait(2)
                (Path(d)/"目标请求.json").write_text(json.dumps(dict(id=1,target="BC")), encoding="utf-8")
                time.sleep(.06)
                finish.set()
            t = threading.Thread(target=change)
            t.start()
            def send(action):
                sent.append(action.copy())
                stop.set()
                return action
            run_async_control(predict, lambda: np.zeros(6), lambda: (time.monotonic() - .002, np.full((4,4,3), 100, dtype=np.uint8)),
                              send, stop, np.full(6,-10), np.full(6,10), duration=2, hz=100, max_delta=5,
                              task_switch=switch, on_event=events.append)
            t.join()
            self.assertTrue(sent)
            np.testing.assert_equal(sent[0], np.full(6, 2.))
            self.assertTrue(any(e["type"] == "task_switch_discard" for e in events))


if __name__ == "__main__":
    unittest.main()
