"""检查本地演示的完整性；不连接机械臂、不上传数据。"""
import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT.parent / "训练数据采集/机械臂控制项目/历史演示/前视抓取_首批_20260613"


def inspect_dataset(dataset, report_path=None):
    dataset = Path(dataset).resolve()
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    tasks = [json.loads(x) for x in (dataset / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    task_ids = {x["task_index"] for x in tasks}
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    errors, episodes, tables, warnings = [], [], [], []
    previews = []
    for path in sorted(dataset.glob("data/chunk-*/*.parquet")):
        data = pq.read_table(path).to_pydict()
        n = len(data["timestamp"])
        ep = int(data["episode_index"][0])
        state = np.asarray(data["observation.state"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        if np.ptp(state, axis=0).max() < .3:
            warnings.append(f"episode {ep}: negligible joint motion; manually review before task training")
        timestamp = np.asarray(data["timestamp"], dtype=np.float64)
        if state.shape != (n, 6) or action.shape != (n, 6) or not np.isfinite(state).all() or not np.isfinite(action).all():
            errors.append(f"episode {ep}: invalid state/action")
        if not np.allclose(timestamp, np.arange(n) / info["fps"], atol=1e-4):
            errors.append(f"episode {ep}: timestamps do not match FPS")
        if data["frame_index"] != list(range(n)) or set(data["episode_index"]) != {ep}:
            errors.append(f"episode {ep}: inconsistent indices")
        if not set(data["task_index"]).issubset(task_ids):
            errors.append(f"episode {ep}: unknown task")
        counts, black_counts = {}, {}
        for key in video_keys:
            video = dataset / info["video_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep, video_key=key)
            count, black = 0, 0
            try:
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    expected_shape = info["features"][key]["shape"]
                    if [stream.height, stream.width, 3] != expected_shape:
                        errors.append(f"episode {ep}, {key}: video dimensions disagree with metadata")
                    if stream.average_rate is None or abs(float(stream.average_rate) - info["fps"]) > .01:
                        errors.append(f"episode {ep}, {key}: video FPS disagrees with dataset FPS")
                    for i, frame in enumerate(container.decode(video=0)):
                        rgb = frame.to_ndarray(format="rgb24")
                        black += int(rgb.mean() < 3 and rgb.std() < 1)
                        if i in {0, n // 2, n - 1}:
                            preview = Image.fromarray(rgb)
                            preview.thumbnail((320, 240))
                            previews.append((ep, key, i, preview))
                        count += 1
                if count != n:
                    errors.append(f"episode {ep}, {key}: {count} video frames != {n} rows")
                if black:
                    errors.append(f"episode {ep}, {key}: {black} black frames")
            except Exception as exc:
                errors.append(f"episode {ep}, {key}: {exc}")
            counts[key], black_counts[key] = count, black
        episodes.append({"episode": ep, "frames": n, "duration_s": n / info["fps"], "video_frames": counts, "black_frames": black_counts, "state_range": np.ptp(state, axis=0).tolist()})
        tables.append(data)
    if len(episodes) != info["total_episodes"] or sum(x["frames"] for x in episodes) != info["total_frames"]:
        errors.append("Episode/frame totals do not match metadata")
    if [x["episode"] for x in episodes] != list(range(info["total_episodes"])):
        errors.append("Episode indices are not consecutive")
    if not tables:
        errors.append("No episodes found")
    report = {"status": "passed" if not errors else "failed", "dataset_root": str(dataset), "fps": info["fps"], "total_episodes": len(episodes), "total_frames": sum(x["frames"] for x in episodes), "tasks": tasks, "video_keys": video_keys, "episodes": episodes, "errors": errors, "warnings": warnings, "task_success_manually_verified": False, "robot_commands_sent": 0}
    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if previews:
            # 分页，避免200段双摄生成的长图超过JPEG的65500像素限制。
            page_size = 20 * max(1, len(video_keys)) * 3
            for start in range(0, len(previews), page_size):
                page = previews[start:start + page_size]
                sheet = Image.new("RGB", (960, ((len(page) + 2) // 3) * 262), "white")
                draw = ImageDraw.Draw(sheet)
                for i, (ep, key, frame, image) in enumerate(page):
                    image.thumbnail((320, 240))
                    x, y = i % 3 * 320, i // 3 * 262
                    sheet.paste(image, (x, y + 22))
                    draw.text((x + 5, y + 4), f"Episode {ep + 1} / {key.rsplit('.', 1)[-1]} / frame {frame}", fill="black")
                sheet.save(report_path.with_name(f"{report_path.stem}_预览{start // page_size + 1:02d}.jpg"))
    return report, info, tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-report", type=Path, help="结果报告路径，已有文件不会覆盖")
    args = parser.parse_args()
    if args.output_report and args.output_report.exists():
        parser.error("报告已存在，请指定新文件")
    result, _, _ = inspect_dataset(args.dataset_root, args.output_report or ROOT / "运行记录/演示数据检查.json")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)
