"""从未完成保存的原始图片恢复观看视频；不伪造机械臂动作数据。"""
import argparse
import json
from pathlib import Path

from lerobot.datasets.video_utils import encode_video_frames

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.dataset_root.resolve()
    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    output = ROOT.parent / "训练数据采集/机械臂控制项目/恢复视频" / source.name
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for directory in sorted((source / "images").glob("*/episode_*")):
        images = sorted(directory.glob("frame_*.png"))
        if not images:
            continue
        if [p.stem for p in images] != [f"frame_{i:06d}" for i in range(len(images))]:
            raise ValueError(f"图片序列有缺失：{directory}")
        number = int(directory.name.rsplit("_", 1)[1]) + 1
        video = output / f"桌面相机_第{number:03d}段.mp4"
        if not video.exists():
            # 上游函数以 overwrite 决定是否允许父目录已存在；上方已确认视频不存在。
            encode_video_frames(directory, video, info["fps"], vcodec="h264", overwrite=True)
        results.append({"source_images": str(directory), "frames": len(images), "seconds": len(images) / info["fps"], "recovered_video": str(video)})
    report = {"source_dataset": str(source), "videos": results, "action_parquet_files": [str(p) for p in source.glob("data/chunk-*/*.parquet")], "note": "Recovered videos preserve visual observations only. Missing action/state data cannot be reconstructed from images; this is not a completed training dataset."}
    (output / "视频恢复记录.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
