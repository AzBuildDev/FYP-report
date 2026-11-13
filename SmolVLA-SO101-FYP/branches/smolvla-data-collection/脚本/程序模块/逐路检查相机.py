"""Capture one frame per accessible camera and record actual stream properties."""
import json
import argparse
import time
from pathlib import Path

import cv2

root = Path(__file__).resolve().parents[2]
output = root / "运行记录/相机画面"
output.mkdir(parents=True, exist_ok=True)
results = []
parser = argparse.ArgumentParser()
parser.add_argument("--indices", type=int, nargs="+", default=[1, 2])
args = parser.parse_args()
for index in args.indices:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    try:
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        image = None
        for _ in range(6):
            ok, frame = cap.read()
            if ok:
                image = frame
            time.sleep(0.04)
        result = {"index": index, "backend": "dshow", "frame_read_ok": image is not None, "fps_reported": cap.get(cv2.CAP_PROP_FPS)}
        if image is not None:
            path = output / f"相机_{index}.jpg"
            if not cv2.imwrite(str(path), image):
                raise OSError(f"Failed to save camera frame: {path}")
            result.update(shape=list(image.shape), image_std=float(image.std()), saved_frame=str(path))
        print(result, flush=True)
        results.append(result)
    finally:
        cap.release()
(root / "运行记录/单路相机检查.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("CAMERAS_READING:", sum(r["frame_read_ok"] for r in results))
raise SystemExit(0 if any(r["frame_read_ok"] for r in results) else 2)
