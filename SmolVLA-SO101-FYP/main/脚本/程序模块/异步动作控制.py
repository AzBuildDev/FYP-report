"""后台推理、主线程控制；支持注入模拟设备，不直接依赖机械臂 SDK。"""
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
from 检查控制输入 import bounded_target, validate_frame


def copy_images(image):
    if isinstance(image, dict):
        return {key: value.copy() for key, value in image.items()}
    return image.copy()


@dataclass
class Prediction:
    source_time: float
    actions: np.ndarray
    inference_s: float
    generation: int = 0


class InferenceWorker:
    """最多一份请求在途。工作线程不读取串口，也不发送动作。"""

    def __init__(self, predict):
        self.predict = predict
        self.requests = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.pending = False
        self.thread = threading.Thread(target=self._run, name="Smol模型推理", daemon=True)
        self.thread.start()

    def submit(self, state, image, stamp, generation=0):
        if self.stopped.is_set() or self.pending:
            raise RuntimeError("推理请求尚未完成或工作线程已关闭")
        self.pending = True
        self.requests.put_nowait((state.copy(), copy_images(image), stamp, generation))

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.pending = False
        if isinstance(result, Exception):
            raise RuntimeError("后台模型推理失败") from result
        return result

    def close(self):
        # 不等待不可中断的 GPU 调用，确保主线程先释放机械臂资源。
        self.stopped.set()

    def _run(self):
        while not self.stopped.is_set():
            try:
                state, image, stamp, generation = self.requests.get(timeout=0.05)
            except queue.Empty:
                continue
            if self.stopped.is_set():
                return
            start = time.monotonic()
            try:
                actions = np.array(self.predict(state, image), dtype=np.float32, copy=True)
                if actions.ndim != 2 or actions.shape[1] != 6 or not len(actions) or not np.isfinite(actions).all():
                    raise ValueError("模型动作块的形状或数值无效")
                result = Prediction(stamp, actions, time.monotonic() - start, generation)
            except Exception as exc:
                result = exc
            if self.stopped.is_set():
                return
            self.results.put_nowait(result)


def run_async_control(predict, read_state, read_frame, send_action, stop, lo, hi,
                      duration=60, hz=20, max_delta=1.5, execute_limit=20,
                      inference_timeout=3, on_event=None, stats=None, workers=None, target_controller=None, task_switch=None):
    """在固定节拍检查输入并执行预测。推理慢或动作不足时保持最后目标，不回放旧步骤。"""
    if (duration is not None and (not np.isfinite(duration) or duration <= 0)) or hz <= 0 or execute_limit <= 0 or inference_timeout <= 0:
        raise ValueError("运行时间、频率、执行长度和推理超时必须大于零")
    # 无时限只影响自动停止，节拍等待和停止/故障处理仍逐步进行。
    duration = float("inf") if duration is None else duration
    stats = stats if stats is not None else {}
    stats.update(sent_commands=0, inferences=0, skipped_action_steps=0, missed_ticks=0,
                 first_action_time=None, last_action_time=None, stop_reason="running")
    on_event = on_event or (lambda event: None)
    start = time.monotonic()
    next_tick = start
    period = 1 / hz
    worker = InferenceWorker(predict)
    if workers is not None:
        workers.append(worker)
    active, previous_index, used = None, -1, 0
    request_time = None
    try:
        while not stop.is_set() and time.monotonic() - start < duration:
            now = time.monotonic()
            stamp, image = read_frame()
            validate_frame(stamp, image, time.monotonic())
            current = np.asarray(read_state(), dtype=np.float32)
            if current.shape != (6,) or not np.isfinite(current).all():
                raise RuntimeError("当前关节状态无效")
            if target_controller is not None:
                target_controller.observe(current, time.monotonic())
            if task_switch is not None:
                guard = target_controller.gripper_contact if target_controller is not None else None
                if task_switch.update(current, time.monotonic(), guard):
                    active, previous_index, used = None, -1, 0
            result = worker.poll()
            if result is not None and task_switch is not None and result.generation != task_switch.generation:
                on_event({"type": "task_switch_discard", "generation": result.generation})
                result = None
            if result is not None:
                now = time.monotonic()
                offset = max(0, int((now - result.source_time) * hz))
                if offset >= len(result.actions):
                    raise RuntimeError("推理结果已超出动作块有效期，停止控制")
                active, previous_index, used = result, offset - 1, 0
                stats["inferences"] += 1
                stats["skipped_action_steps"] += offset
                on_event({"type": "async_chunk", "t": now - start, "inference_s": result.inference_s,
                          "skip_prefix": offset, "first_action": result.actions[offset].tolist(),
                          "source_age_s": now - result.source_time})
            if worker.pending and time.monotonic() - request_time > inference_timeout:
                raise RuntimeError("模型推理超时，停止控制")
            if not worker.pending and not stop.is_set():
                request_time = time.monotonic()
                worker.submit(current, image, stamp, task_switch.generation if task_switch else 0)
            now = time.monotonic()
            if stop.is_set() or now - start >= duration:
                break
            if active is not None and used < execute_limit:
                # 每个控制节拍至少推进一步；避免毫秒级调度抖动令取整后重复索引。
                # 若线程曾错过节拍，按实际时间直接跳过已经过期的步骤。
                index = max(previous_index + 1, int((now - active.source_time) * hz))
                if previous_index < index < len(active.actions):
                    target = (target_controller.target(active.actions[index], current) if target_controller is not None
                              else bounded_target(active.actions[index], current, lo, hi, max_delta))
                    # 再次读取帧时间，覆盖串口读取或回调耗时造成的输入过期。
                    stamp, image = read_frame()
                    validate_frame(stamp, image, time.monotonic())
                    if stop.is_set() or time.monotonic() - start >= duration:
                        break
                    actual = send_action(target)
                    if target_controller is not None:
                        target_controller.commit(target)
                    previous_index = index
                    used += 1
                    stats["sent_commands"] += 1
                    action_time = time.monotonic()
                    if stats["first_action_time"] is None:
                        stats["first_action_time"] = action_time - start
                    stats["last_action_time"] = action_time - start
                    on_event({"type": "control_step", "t": action_time - start,
                              "observed_state": current.tolist(), "model_target": active.actions[index].tolist(),
                              "target": target.tolist(), "sent_action": actual,
                              "chunk_index": index, "source_age_s": now - active.source_time})
            next_tick += period
            now = time.monotonic()
            if now >= next_tick:
                missed = int((now - next_tick) / period) + 1
                stats["missed_ticks"] += missed
                next_tick += missed * period
            stop.wait(min(max(0, next_tick - now), max(0, start + duration - now)))
        stats["stop_reason"] = "user_stop" if stop.is_set() else "time_limit"
    except KeyboardInterrupt:
        stats["stop_reason"] = "user_stop"
        raise
    except BaseException:
        stats["stop_reason"] = "error"
        raise
    finally:
        worker.close()
        stats["seconds"] = time.monotonic() - start
        first, last = stats["first_action_time"], stats["last_action_time"]
        stats["active_control_hz"] = ((stats["sent_commands"] - 1) / (last - first)
                                      if first is not None and last > first else 0)
        on_event({"type": "async_end", **stats})
    return stats
