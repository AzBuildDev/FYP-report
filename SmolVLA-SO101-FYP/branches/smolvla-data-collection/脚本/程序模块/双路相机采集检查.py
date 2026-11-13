"""Check simultaneous local capture from the desktop and gripper cameras."""
import json
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
import cv2

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--backend", choices=["dshow", "msmf"], default="dshow")
parser.add_argument("--front-backend", choices=["dshow", "msmf"])
parser.add_argument("--front-fps", type=int, default=15)
parser.add_argument("--reverse", action="store_true")
args = parser.parse_args()
captures = {}
results = {}
report = {"passed": False, "backend": args.backend, "requested_fps": args.fps, "cameras": results}
try:
    order = [("overhead", 1), ("front", 2)]
    if args.reverse:
        order.reverse()
    for role, index in order:
        backend = (args.front_backend or args.backend) if role == "front" else args.backend
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_MSMF)
        captures[role] = cap
        results[role] = {"index": index, "backend": backend, "opened": cap.isOpened(), "successful_frames": 0, "failed_frames": 0}
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {role} camera {index}")
        # Device capabilities: 1080P USB Camera supports MJPG at 25/30 FPS;
        # icspring camera supports YUY2 only, at 5-30 FPS.
        fourcc = "MJPG" if role == "overhead" else "YUY2"
        format_accepted = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, args.front_fps if role == "front" else args.fps)
        results[role].update(requested_fourcc=fourcc, format_accepted=format_accepted, fps_reported=cap.get(cv2.CAP_PROP_FPS))
        # Start the first stream before opening the second device.
        ok, _ = cap.read()
        results[role]["warmup_ok"] = ok
    start = time.perf_counter()
    frames = {}
    while time.perf_counter() - start < 3:
        for role, cap in captures.items():
            ok, frame = cap.read()
            results[role]["successful_frames" if ok else "failed_frames"] += 1
            if ok:
                frames[role] = frame
    for role, frame in frames.items():
        path = root / "运行记录/相机画面" / f"{role}_dual.jpg"
        cv2.imwrite(str(path), frame)
        results[role].update(shape=list(frame.shape), image_mean=float(frame.mean()), image_std=float(frame.std()), saved_frame=str(path))
    passed = all(result["successful_frames"] >= 5 and result.get("shape") == [480, 640, 3] and result.get("image_std", 0) > 1 for result in results.values())
    report.update(passed=passed, both_open_simultaneously=True, duration_s=time.perf_counter() - start)
    if not passed:
        raise RuntimeError("Dual-camera capture check failed")
except Exception as exc:
    report["error"] = str(exc)
    raise
finally:
    for cap in captures.values():
        cap.release()
    (root / "运行记录/双路相机检查.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
