"""独立尝试归档与输入校验；拒绝清除未归档的原始图片。"""
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def joint_values(observation):
    values = [float(observation[name]) for name in JOINTS]
    if any(not math.isfinite(v) or not (-100 if i < 5 else 0) <= v <= 100 for i, v in enumerate(values)):
        raise RuntimeError("关节位置非有限数或超出标定单位范围，停止录制")
    return values


class CollectionAttempt:
    def __init__(self, root, episode, labels, task):
        self.root = Path(root).resolve()
        self.identifier = "尝试_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:6]
        self.path = self.root / "采集尝试" / self.identifier
        self.path.mkdir(parents=True, exist_ok=False)
        self.journal = self.path / "动作与状态.jsonl"
        self.metadata = {"schema_version": 1, "attempt_id": self.identifier, "episode_index": episode,
                         "labels": labels, "task": task, "fps": 20, "status": "recording",
                         "archive_complete": False, "frames": 0, "human_success": False}
        self.set_status("recording")

    def set_status(self, status, **fields):
        self.metadata.update(status=status, **fields)
        write_json(self.path / "尝试状态.json", self.metadata)

    def archive(self, dataset):
        """图片全部复制并核对动作行数后才允许重置上游缓冲区。归档路径不可覆盖。"""
        dataset._wait_image_writer()
        rows = [json.loads(line) for line in self.journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        frames = [row for row in rows if row["type"] == "frame"]
        count = dataset.episode_buffer.get("size")
        if count != len(frames):
            raise RuntimeError(f"动作备份 {len(frames)} 行与缓冲 {count} 帧不一致；保留原目录并封存批次")
        temporary = self.path / "原始帧归档中"
        final = self.path / "原始帧"
        if final.exists():
            raise RuntimeError("本次尝试已归档，禁止覆盖")
        temporary.mkdir(exist_ok=False)
        for i, row in enumerate(frames):
            if row["frame_index"] != i:
                raise RuntimeError("动作备份帧号不连续，禁止清理原始帧")
            source = (self.root / row["image_path"]).resolve()
            if not source.is_relative_to(self.root) or not source.is_file():
                raise RuntimeError("原始图片缺失或路径越界，禁止清理")
            target = temporary / f"frame_{i:06d}.png"
            shutil.copy2(source, target)
            if source.stat().st_size != target.stat().st_size:
                raise RuntimeError("原始图片复制不完整，禁止清理")
        temporary.rename(final)
        self.set_status(self.metadata["status"], archive_complete=True, frames=count,
                        journal_sha256=hashlib.sha256(self.journal.read_bytes()).hexdigest(),
                        archived_image_pattern="原始帧/frame_{frame_index:06d}.png")

    def archive_and_reset(self, dataset, status, reason):
        self.archive(dataset)
        self.set_status(status, reason=reason)
        dataset.clear_episode_buffer()
