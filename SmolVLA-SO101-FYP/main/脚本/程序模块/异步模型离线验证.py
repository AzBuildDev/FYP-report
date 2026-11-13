"""本地微调模型驱动模拟关节，检查异步节拍和停止；不打开串口或相机。"""
import argparse
import json
import time
import threading
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from 实机抓取测试 import ROOT, make_predict
from 异步动作控制 import run_async_control
from 选择推理模型 import add_selection_arguments, load_local_policy, selection_from_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=8)
    add_selection_arguments(parser)
    args = parser.parse_args()
    if args.seconds < 4:
        parser.error("至少运行4秒，覆盖预热和连续推理")
    selection = selection_from_args(args, ROOT)
    if selection is None:
        print("已取消离线验证。", flush=True)
        return
    model_dir, dataset_root = selection.model, selection.dataset
    policy = load_local_policy(model_dir, selection.processor)
    torch.manual_seed(42)
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    tables = [pq.read_table(p).to_pydict() for p in sorted(dataset_root.glob("data/chunk-*/*.parquet"))]
    states = np.concatenate([np.asarray(t["observation.state"], dtype=np.float32) for t in tables])
    lo, hi = states.min(axis=0) - 8, states.max(axis=0) + 8
    current = np.array(tables[0]["observation.state"][0], dtype=np.float32)
    image_keys = list(policy.config.image_features)
    ep = int(tables[0]["episode_index"][0])
    frames = {}
    for key in image_keys:
        video = dataset_root / info["video_path"].format(
            episode_chunk=ep // info["chunks_size"], episode_index=ep, video_key=key)
        with av.open(str(video)) as container:
            frames[key] = [frame.to_ndarray(format="bgr24") for frame in container.decode(video=0)]
    stop, workers, events, commands = threading.Event(), [], [], []
    frame_start = time.monotonic()

    def frame():
        now = time.monotonic()
        i = min(int((now - frame_start) * info["fps"]), min(len(values) for values in frames.values()) - 1)
        return now, {key: values[i] for key, values in frames.items()}

    def send(target):
        nonlocal current
        assert np.all(target >= lo) and np.all(target <= hi)
        assert np.max(np.abs(target - current)) <= 1.50001
        current = target.copy()
        commands.append(time.monotonic())

    start = time.monotonic()
    try:
        stats = run_async_control(make_predict(policy, image_keys),
                                  lambda: current.copy(), frame, send, stop, lo, hi,
                                  duration=args.seconds, workers=workers, on_event=events.append)
        returned = time.monotonic() - start
    finally:
        for worker in workers:
            worker.thread.join(timeout=5)
            if worker.thread.is_alive():
                raise RuntimeError("后台推理未退出，停止本次模型验证")
    chunks = [event for event in events if event["type"] == "async_chunk"]
    assert stats["inferences"] >= 3 and stats["sent_commands"] > 40, stats
    assert stats["active_control_hz"] >= 18, stats
    assert returned < args.seconds + 0.5, returned

    # 再检查真实 GPU 推理在途时，控制线程能否响应窗口停止请求。
    policy.reset()
    entered = threading.Event()
    stop = threading.Event()
    workers = []
    original = make_predict(policy, image_keys)
    requested = []

    def infer(state, image):
        entered.set()
        return original(state, image)

    def cancel():
        if entered.wait(2):
            time.sleep(0.1)
            requested.append(time.monotonic())
            stop.set()

    canceller = threading.Thread(target=cancel)
    canceller.start()
    try:
        cancelled = run_async_control(infer, lambda: current.copy(), frame, send, stop, lo, hi,
                                      duration=3, workers=workers)
        return_time = time.monotonic()
    finally:
        canceller.join(2)
        for worker in workers:
            worker.thread.join(timeout=5)
            if worker.thread.is_alive():
                raise RuntimeError("后台推理未退出，停止本次模型验证")
    assert requested and cancelled["stop_reason"] == "user_stop" and cancelled["sent_commands"] == 0
    stop_delay = return_time - requested[0]
    assert stop_delay < 0.2, stop_delay
    result = {"status": "passed", "created_at": datetime.now().isoformat(timespec="seconds"),
              "scope": "real_model_simulated_control", "hardware_tested": False,
              "model": str(model_dir), "dataset": str(dataset_root), "processor": str(selection.processor),
              "robot_commands_sent": 0, "task_success_evaluated": False,
              "control": stats, "control_return_s": returned,
              "chunk_inference_median_s": float(np.median([c["inference_s"] for c in chunks])),
              "stop_response_s": stop_delay, "stop_test_simulated_commands": cancelled["sent_commands"],
              "clock_resolution_s": time.get_clock_info("monotonic").resolution,
              "notes": "关节即时跟随命令、图像来自已录制视频；只验证并发、节拍和限幅，不模拟接触、负载或任务成功。"}
    path = ROOT / "运行记录/异步模型验证.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
