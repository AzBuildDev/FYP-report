"""核对指定100段批次的末20段；显式 --apply 才按用户确认修订任务，先备份。"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import av
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from 红蓝抓取任务 import TASKS

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def run(dataset, apply=False):
    dataset = dataset.resolve()
    read = lambda name: json.loads((dataset / name).read_text(encoding="utf-8"))
    lines = lambda name: [json.loads(x) for x in (dataset / name).read_text(encoding="utf-8").splitlines() if x.strip()]
    info = read("meta/info.json")
    tasks = lines("meta/tasks.jsonl")
    episodes = lines("meta/episodes.jsonl")
    stats = lines("meta/episodes_stats.jsonl")
    assert info["total_episodes"] == len(episodes) == len(stats) == 100
    assert not (dataset / "meta/scene_annotations.json").exists(), "已有场景标签，需同时核对后再修订"
    assert not read("采集配置.json").get("collection_label_schema")
    task_names = {row["task_index"]: row["task"] for row in tasks}
    task_ids = {obj: next(i for i, text in task_names.items() if text == task) for obj, task in TASKS.items()}
    tables = []
    paths = []
    changes = []
    for ep in range(100):
        path = dataset / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        table = pq.read_table(path)
        assert set(table["episode_index"].to_pylist()) == {ep}
        current_ids = set(table["task_index"].to_pylist())
        assert len(current_ids) == 1
        old_id = current_ids.pop()
        assert episodes[ep]["episode_index"] == stats[ep]["episode_index"] == ep
        assert episodes[ep]["length"] == len(table)
        assert episodes[ep]["tasks"] == [task_names[old_id]]
        tables.append(table)
        paths.append(path)
        if ep >= 80:
            obj = "RC" if ep < 90 else "BC"
            changes.append({"display_episode": ep + 1, "video_episode_index": ep,
                            "old_target": next(obj for obj, text in TASKS.items() if text == task_names[old_id]),
                            "target": obj, "changed": old_id != task_ids[obj],
                            "scene": "single_red_cube" if obj == "RC" else "single_blue_cube"})
    report_dir = ROOT / "运行记录" / ("百段末尾标签核对_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    report_dir.mkdir()
    for start in (80, 90):
        sheet = Image.new("RGB", (1600, 560), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, ep in enumerate(range(start, start + 10)):
            path = dataset / info["video_path"].format(episode_chunk=0, episode_index=ep,
                                                       video_key="observation.images.front")
            with av.open(str(path)) as container:
                frame = next(container.decode(video=0)).to_image()
            frame.thumbnail((320, 240))
            x, y = (offset % 5) * 320, (offset // 5) * 280
            sheet.paste(frame, (x, y))
            row = changes[ep - 80]
            draw.text((x + 5, y + 242), f"Video {ep:06d} / Display {ep + 1}", fill="black")
            draw.text((x + 5, y + 258), f"Label: {row['old_target']} -> {row['target']}", fill="black")
        sheet.save(report_dir / f"视频{start:06d}至{start + 9:06d}_首帧.jpg")
    report = {"dataset_root": str(dataset), "total_episodes": 100, "changes": changes,
              "incorrect_count": sum(row["changed"] for row in changes),
              "numbering": "视频和meta内部episode_index从0开始；第NNN段动作备份和界面计数从1开始",
              "basis": "用户确认第81至90条为单红，第91至100条为单蓝；首帧用于场景核对，不判定抓取成功",
              "applied": False, "robot_commands_sent": 0}
    if apply:
        mutations = {}
        preserved = {}
        journal_bodies = {}
        for row in changes:
            ep, obj = row["video_episode_index"], row["target"]
            journal = dataset / "逐帧动作备份" / f"第{ep + 1:03d}段动作备份.jsonl"
            header_bytes, rest = journal.read_bytes().split(b"\n", 1)
            header = json.loads(header_bytes)
            assert header["episode"] == ep
            assert header["task"] == TASKS[row["old_target"]]
            assert header["target_object_id"] == row["old_target"]
            if not row["changed"]:
                continue
            old = tables[ep]
            field = old.schema.field("task_index")
            new = old.set_column(old.schema.get_field_index("task_index"), field,
                                 pa.array([task_ids[obj]] * len(old), type=field.type))
            mutations[paths[ep]] = new
            preserved[paths[ep]] = old.drop(["task_index"])
            episodes[ep]["tasks"] = [TASKS[obj]]
            tid = task_ids[obj]
            stats[ep]["stats"]["task_index"] = {"min": [tid], "max": [tid], "mean": [float(tid)],
                                                  "std": [0.0], "count": [len(old)]}
            header["task"], header["target_object_id"] = TASKS[obj], obj
            mutations[journal] = json.dumps(header, ensure_ascii=False).encode("utf-8") + b"\n" + rest
            journal_bodies[journal] = hashlib.sha256(rest).hexdigest()
        for name, rows in (("meta/episodes.jsonl", episodes), ("meta/episodes_stats.jsonl", stats)):
            mutations[dataset / name] = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode("utf-8")
        before = {str(path.relative_to(dataset)): digest(path) for path in dataset.rglob("*") if path.is_file()}
        backup = ROOT / "运行环境/备份" / report_dir.name / dataset.name
        backup.mkdir(parents=True)
        for path in mutations:
            target = backup / path.relative_to(dataset)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            assert digest(target) == before[str(path.relative_to(dataset))]
        (backup / "原始清单.json").write_bytes(encoded({"dataset_root": str(dataset), "files": before,
                                                       "changes": changes}))
        for path, value in mutations.items():
            temp = path.with_name(path.name + ".标签修订临时")
            assert not temp.exists()
            if isinstance(value, pa.Table):
                pq.write_table(value, temp)
            else:
                temp.write_bytes(value)
            temp.replace(path)
        for path, expected in preserved.items():
            assert pq.read_table(path).drop(["task_index"]).equals(expected), "动作状态等非任务列发生变化"
        for path, expected in journal_bodies.items():
            assert hashlib.sha256(path.read_bytes().split(b"\n", 1)[1]).hexdigest() == expected
        for rel, expected in before.items():
            if dataset / rel not in mutations:
                assert digest(dataset / rel) == expected, f"未计划修改的文件发生变化：{rel}"
        for ep in range(100):
            table = pq.read_table(paths[ep])
            expected = task_ids["RC" if ep < 90 else "BC"] if ep >= 80 else tables[ep]["task_index"][0].as_py()
            assert set(table["task_index"].to_pylist()) == {expected}
        report.update(applied=True, backup=str(backup),
                      verification="前80段不变；全部200个视频哈希不变；任务列以外所有逐帧数据不变；动作备份正文不变",
                      final_task_counts=dict(Counter(ep["tasks"][0] for ep in episodes)))
        (backup / "修订结果.json").write_bytes(encoded(report))
    (report_dir / "核对修订报告.json").write_bytes(encoded(report))
    print(json.dumps({"report_directory": str(report_dir), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run(args.dataset_root, args.apply)
