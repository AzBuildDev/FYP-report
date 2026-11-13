"""只对输入副本预热模型；不持有机械臂对象，不返回可执行动作。"""
import threading
import time

import numpy as np


def warm_model(predict, state, image, stop, *, chunk_size, hz=20, runs=2,
               timeout=30, reserve=0.25, workers=None):
    """首轮允许冷启动，后续轮次须留出时效余量；停止请求不等待 GPU。"""
    if (not isinstance(chunk_size, int) or chunk_size < 1 or
            not isinstance(runs, int) or runs < 2 or
            not np.isfinite([hz, timeout, reserve]).all() or
            hz <= 0 or timeout <= 0 or reserve < 0 or reserve >= chunk_size / hz):
        raise ValueError("预热次数或时效参数无效")
    state = np.array(state, dtype=np.float32, copy=True)
    image = np.array(image, copy=True)
    if (state.shape != (6,) or not np.isfinite(state).all() or image.ndim != 3 or
            image.shape[2] != 3 or not image.size or not np.isfinite(image).all()):
        raise ValueError("预热输入无效")
    budget = chunk_size / hz - reserve
    report = {"status": "cancelled", "samples": [], "horizon_s": chunk_size / hz,
              "reserve_s": reserve, "ready_budget_s": budget, "prediction_reused": False}
    deadline = time.monotonic() + timeout
    for index in range(runs):
        if stop.is_set():
            return report
        done, result = threading.Event(), {}
        round_started = time.monotonic()

        def infer():
            started = time.monotonic()
            try:
                actions = np.asarray(predict(state.copy(), image.copy()))
                if actions.shape != (chunk_size, 6) or not np.isfinite(actions).all():
                    raise RuntimeError("预热模型输出形状或数值无效")
                result["elapsed_s"] = time.monotonic() - started
            except BaseException as exc:
                result["error"] = exc
            finally:
                done.set()

        worker = threading.Thread(target=infer, daemon=True, name="模型启动预热")
        if workers is not None:
            workers.append(worker)
        worker.start()
        while not done.wait(0.02):
            if stop.is_set():
                return report
            if time.monotonic() >= deadline:
                raise RuntimeError("模型预热超时，未允许启动控制")
        if stop.is_set():
            return report
        if time.monotonic() >= deadline:
            raise RuntimeError("模型预热超时，未允许启动控制")
        if "error" in result:
            raise RuntimeError(f"模型预热失败：{result['error']}") from result["error"]
        elapsed = result["elapsed_s"]
        ready_latency = time.monotonic() - round_started
        report["samples"].append({"round": index + 1, "inference_s": elapsed,
                                  "ready_latency_s": ready_latency,
                                  "cold_start": index == 0})
        if index > 0 and ready_latency >= budget:
            raise RuntimeError(f"预热结果准备 {ready_latency:.3f} 秒，超过启动预算 {budget:.3f} 秒，未允许启动控制")
    report["status"] = "ready"
    return report


def join_warm_workers(workers):
    """应在硬件资源释放后调用；退出最多等待一秒。"""
    deadline = time.monotonic() + 1
    for worker in workers:
        worker.join(timeout=max(0, deadline - time.monotonic()))
