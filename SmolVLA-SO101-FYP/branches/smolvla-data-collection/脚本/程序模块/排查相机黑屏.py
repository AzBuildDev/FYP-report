"""Bounded checks of actual image content using the same preview workers."""
import json
import argparse
import multiprocessing as mp
import queue
import time
from pathlib import Path

from 相机测试窗口 import capture_worker

ROOT = Path(__file__).resolve().parents[2]


def check(name, specs):
    import cv2
    ctx = mp.get_context("spawn")
    cameras = []
    for spec in specs:
        index, fmt, fps = spec[:3]
        backend = spec[3] if len(spec) > 3 else "DirectShow"
        messages, stop = ctx.Queue(3), ctx.Event()
        process = ctx.Process(target=capture_worker, args=(index, backend, fmt, fps, messages, stop), daemon=True)
        cameras.append(dict(index=index, backend=backend, format=fmt, fps=fps, messages=messages, stop=stop, process=process, metrics=[], error=None, last=None))
        process.start()
        time.sleep(1)
    start = time.monotonic()
    try:
        while time.monotonic() - start < 12:
            for camera in cameras:
                try:
                    kind, payload = camera["messages"].get_nowait()
                except queue.Empty:
                    continue
                if kind == "frame":
                    frame = payload[1]
                    camera["metrics"].append(dict(mean=float(frame.mean()), std=float(frame.std())))
                    camera["last"] = frame
                elif kind == "error":
                    camera["error"] = payload
            time.sleep(.015)
        result = dict(name=name, cameras=[])
        for camera in cameras:
            rows = camera["metrics"]
            last = camera["last"]
            if last is not None:
                cv2.imwrite(str(ROOT / f"运行记录/相机画面/black_check_{name}_{camera['index']}.jpg"), last)
            result["cameras"].append(dict(index=camera["index"], backend=camera["backend"], format=camera["format"], fps=camera["fps"], frames=len(rows), nonblack_frames=sum(r["std"] > 1 for r in rows), first=rows[0] if rows else None, last=rows[-1] if rows else None, error=camera["error"]))
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return result
    finally:
        # These bounded diagnostic workers have no robot connection.
        for camera in cameras:
            camera["stop"].set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and any(c["process"].is_alive() for c in cameras):
            for camera in cameras:
                try:
                    camera["messages"].get_nowait()
                except queue.Empty:
                    pass
            time.sleep(.02)
        for camera in cameras:
            if camera["process"].is_alive():
                camera["process"].terminate()
            camera["process"].join(timeout=2)
            camera["process"].close()
            camera["messages"].close()


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--named", action="store_true")
    parser.add_argument("--both-named", action="store_true")
    args = parser.parse_args()
    results = []
    cases = [
        ("mjpg_single", [(1, "MJPG", 25)]),
        ("auto_single", [(1, "自动", 30)]),
        ("mjpg_pair", [(1, "MJPG", 25), (2, "YUY2", 15)]),
        ("auto_pair", [(1, "自动", 30), (2, "YUY2", 15)]),
    ]
    if args.named:
        cases = [("named_single", [(1, "MJPG", 25, "按设备名")]), ("named_pair", [(1, "MJPG", 25, "按设备名"), (2, "YUY2", 15)])]
    if args.both_named:
        cases = [("gripper_named_single", [(2, "YUY2", 15, "按设备名")]), ("both_named", [(1, "MJPG", 25, "按设备名"), (2, "YUY2", 15, "按设备名")])]
    for name, specs in cases:
        results.append(check(name, specs))
        (ROOT / "运行记录/相机黑屏排查.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
