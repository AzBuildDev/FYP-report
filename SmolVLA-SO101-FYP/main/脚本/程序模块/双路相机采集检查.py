"""按设备名检查双摄规格、实际到帧节拍与 20 Hz 配对；不连接机械臂。"""
import argparse
import json
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from 相机测试窗口 import named_capture
from 双摄采集配置 import CAMERAS

ROOT = Path(__file__).resolve().parents[2]
DEVICES = {
    key: {**spec, "fps": spec["input_fps"]} for key, spec in CAMERAS.items()
}


def list_options():
    """读取驱动声明的模式，不把返回码或 OpenCV 的 set() 当作规格证据。"""
    import imageio_ffmpeg
    results = {}
    pattern = re.compile(
        r"(pixel_format|vcodec)=(\w+)\s+min s=(\d+)x(\d+) fps=([\d.]+)\s+"
        r"max s=(\d+)x(\d+) fps=([\d.]+)"
    )
    for key, spec in DEVICES.items():
        completed = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-list_options", "true",
             "-f", "dshow", "-i", f"video={spec['device']}"],
            capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        stderr = completed.stderr.decode("utf-8", errors="replace")
        modes = []
        for match in pattern.finditer(stderr):
            kind, fmt, w, h, low, mw, mh, high = match.groups()
            mode = {"format_type": kind, "format": fmt, "width": int(w), "height": int(h),
                    "min_fps": float(low), "max_width": int(mw), "max_height": int(mh),
                    "max_fps": float(high)}
            if mode not in modes:
                modes.append(mode)
        results[key] = {"device": spec["device"], "modes": modes,
                        "status": "listed" if modes else "failed", "driver_output": stderr}
    return results


def check_capture(specs, seconds, warmup):
    """到帧时间是主机收到完整图像的时刻，并非曝光时间或端到端延迟。"""
    stop, lock = threading.Event(), threading.Lock()
    states = {key: {"stamp": 0., "count": 0, "arrivals": [], "errors": [],
                    "invalid_frames": 0, "shape": None, "std_min": None}
              for key in specs}
    threads = []
    window = {"start": float("inf"), "end": float("inf")}

    def callback(key, kind, value):
        with lock:
            state = states[key]
            if kind == "error":
                state["errors"].append(str(value))
            elif kind == "frame":
                stamp, image = value
                state["stamp"], state["count"] = stamp, state["count"] + 1
                if window["start"] <= stamp < window["end"]:
                    state["arrivals"].append(stamp)
                    state["shape"] = list(image.shape)
                    # 定期检查非黑画面，避免检查本身挤占取帧线程。
                    if len(state["arrivals"]) % 10 == 1:
                        std = float(image.std())
                        state["std_min"] = std if state["std_min"] is None else min(std, state["std_min"])
                        if list(image.shape) != [specs[key]["height"], specs[key]["width"], 3] or std <= 1:
                            state["invalid_frames"] += 1

    ages = {key: [] for key in specs}
    repeats = {key: 0 for key in specs}
    last_counts = {key: None for key in specs}
    skews, missing, sample_times = [], 0, []
    cleanup_errors = []
    try:
        for key, spec in specs.items():
            thread = threading.Thread(
                target=named_capture,
                args=(spec["index"], spec["fourcc"], spec["fps"],
                      lambda kind, value, k=key: callback(k, kind, value),
                      stop, spec["device"], None),
                kwargs={"width": spec["width"], "height": spec["height"]},
                daemon=True, name=f"相机检查_{key}",
            )
            threads.append(thread)
            thread.start()
            # 与正式录制保持相同的错开开启顺序。
            if len(threads) == 1:
                stop.wait(1.2)
        stop.wait(warmup)
        window["start"] = time.monotonic()
        window["end"] = window["start"] + seconds
        next_sample = window["start"]
        while time.monotonic() < window["end"]:
            delay = next_sample - time.monotonic()
            if delay > 0:
                stop.wait(delay)
            now = time.monotonic()
            if now >= window["end"]:
                break
            with lock:
                snapshot = {key: (state["stamp"], state["count"]) for key, state in states.items()}
            sample_times.append(now)
            if any(count == 0 for stamp, count in snapshot.values()):
                missing += 1
            else:
                stamps = [stamp for stamp, count in snapshot.values()]
                skews.append(max(stamps) - min(stamps))
                for key, (stamp, count) in snapshot.items():
                    ages[key].append(max(0., now - stamp))
                    repeats[key] += count == last_counts[key]
                    last_counts[key] = count
            next_sample += .05
            # 不用突发补采掩盖采样调度落后。
            if next_sample < time.monotonic() - .05:
                next_sample = time.monotonic() + .05
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=3)
            if thread.is_alive():
                cleanup_errors.append(f"{thread.name} 未退出")

    results = {}
    for key, state in states.items():
        arrivals = state.pop("arrivals")
        rate = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]) if len(arrivals) > 1 else 0.
        gaps = np.diff(arrivals)
        age = ages[key]
        results[key] = {**specs[key], **state, "measured_frames": len(arrivals),
                        "measured_fps": rate,
                        "arrival_gap_p95_ms": float(np.percentile(gaps, 95) * 1000) if len(gaps) else None,
                        "arrival_gap_max_ms": float(max(gaps) * 1000) if len(gaps) else None,
                        "host_age_p95_ms": float(np.percentile(age, 95) * 1000) if age else None,
                        "host_age_max_ms": float(max(age) * 1000) if age else None,
                        "repeated_20hz_samples": repeats[key]}
        results[key]["passed"] = bool(
            not state["errors"] and not state["invalid_frames"]
            and state["shape"] == [specs[key]["height"], specs[key]["width"], 3]
            and rate >= max(19., specs[key]["fps"] * .9) and len(age) and max(age) < .25
        )
    sampler_gaps = np.diff(sample_times)
    sampler_fps = (len(sample_times) - 1) / (sample_times[-1] - sample_times[0]) if len(sample_times) > 1 else 0.
    passed = bool(all(result["passed"] for result in results.values()) and not missing
                  and len(skews) and max(skews) < .15 and not cleanup_errors
                  and sampler_fps >= 19 and len(sampler_gaps) and max(sampler_gaps) < .15)
    return {"passed": passed, "run_kind": "cameras_only", "seconds": seconds,
            "warmup_s": warmup, "record_sample_hz": 20, "samples": len(sample_times),
            "sampler_measured_hz": sampler_fps,
            "sampler_gap_max_ms": float(max(sampler_gaps) * 1000) if len(sampler_gaps) else None,
            "missing_samples": missing, "pair_skew_p95_ms": float(np.percentile(skews, 95) * 1000) if skews else None,
            "pair_skew_max_ms": float(max(skews) * 1000) if skews else None,
            "timestamp_basis": "host complete-frame arrival; not exposure time or end-to-end latency",
            "black_check": "every tenth received frame", "cleanup_errors": cleanup_errors,
            "cameras": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-options", action="store_true", help="只读驱动支持规格，不启动视频采集")
    parser.add_argument("--fps", type=int, help="两路共同取帧频率；不改变正式采集配置")
    parser.add_argument("--desktop-fps", type=int)
    parser.add_argument("--wrist-fps", type=int)
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 3 <= args.seconds <= 60 or not 1 <= args.warmup <= 15:
        parser.error("测试时长应为 3–60 秒，预热为 1–15 秒")
    specs = {key: dict(spec) for key, spec in DEVICES.items()}
    for key, override in [("front", args.desktop_fps), ("wrist", args.wrist_fps)]:
        specs[key]["fps"] = override if override is not None else args.fps if args.fps is not None else specs[key]["fps"]
        if not 5 <= specs[key]["fps"] <= 30:
            parser.error("取帧频率应为 5–30 FPS，并使用驱动支持的模式")
    report = list_options() if args.list_options else check_capture(specs, args.seconds, args.warmup)
    passed = all(value["status"] == "listed" for value in report.values()) if args.list_options else report["passed"]
    report = {"checked_at": datetime.now().astimezone().isoformat(), **report}
    path = args.report or ROOT / "运行记录" / ("相机规格检查_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
