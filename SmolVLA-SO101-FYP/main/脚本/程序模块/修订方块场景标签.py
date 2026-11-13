"""离线按人工确认补充场景并修订逐段任务；先备份，保留视频和全部动作状态。"""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from 四物体采集计划 import task_text
from 管理场景标注 import ANNOTATION_FILE, SCENES, read_scene_annotations

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def jsonl_bytes(values):
    return ("\n".join(json.dumps(x, ensure_ascii=False, allow_nan=False) for x in values) + "\n").encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revise(dataset, scene, targets):
    dataset = Path(dataset).resolve()
    info_path, tasks_path = dataset / "meta/info.json", dataset / "meta/tasks.jsonl"
    config_path, episodes_path = dataset / "采集配置.json", dataset / "meta/episodes.jsonl"
    stats_path = dataset / "meta/episodes_stats.jsonl"
    info, config = read_json(info_path), read_json(config_path)
    episodes, stats = read_jsonl(episodes_path), read_jsonl(stats_path)
    if (dataset / ANNOTATION_FILE).exists() or config.get("collection_label_schema"):
        raise ValueError("已有场景/计划标签，拒绝覆盖；请先核对原标注")
    if len(targets) != info["total_episodes"] or len(episodes) != len(targets) or len(stats) != len(targets):
        raise ValueError("逐段目标数量与数据不一致")
    if any(target not in SCENES[scene] for target in targets):
        raise ValueError("抓取目标不在场景中")
    if [e["episode_index"] for e in episodes] != list(range(len(targets))):
        raise ValueError("演示段号不连续")
    # 此工具不重新编号任务；仅支持现有全红批次补充 RC/BC 区别。
    old_tasks = read_jsonl(tasks_path)
    if old_tasks != [{"task_index": 0, "task": task_text("RC")}]:
        raise ValueError("任务表不是预期的单红任务，请核对后再处理")
    new_tasks = [{"task_index": i, "task": task_text(obj)} for i, obj in enumerate(("RC", "BC"))
                 if obj in targets or i == 0]
    labels, tables, mutations = [], [], {}
    now = datetime.now().astimezone().isoformat()
    parquet_before, preserved_columns = {}, {}
    for ep, target in enumerate(targets):
        path = dataset / info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        table = pq.read_table(path)
        data = table.to_pydict()
        if (set(data["episode_index"]) != {ep} or set(data["task_index"]) != {0} or
                len(data["task_index"]) != episodes[ep]["length"] or stats[ep]["episode_index"] != ep):
            raise ValueError(f"第 {ep + 1} 段原始任务/段号/帧数不一致")
        task_id = 0 if target == "RC" else 1
        parquet_before[str(path.relative_to(dataset))] = sha256(path)
        if task_id:
            field = table.schema.field("task_index")
            changed = table.set_column(table.schema.get_field_index("task_index"), field,
                                       pa.array([task_id] * len(table), type=field.type))
            mutations[path] = changed
            preserved_columns[path] = table.drop(["task_index"])
            data["task_index"] = [task_id] * len(table)
        tables.append(data)
        label = {"episode_id": ep, "scene_type": scene, "present_objects": SCENES[scene],
                 "target_object_id": target, "destination": "target_area", "task_text": task_text(target),
                 "annotation_source": "user_confirmation", "annotated_at": now}
        labels.append(label)
        episodes[ep]["tasks"] = [task_text(target)]
        if task_id:
            stats[ep]["stats"]["task_index"] = {"min": [task_id], "max": [task_id],
                                                  "mean": [float(task_id)], "std": [0.0], "count": [len(table)]}
        journal_path = dataset / "逐帧动作备份" / f"第{ep + 1:03d}段动作备份.jsonl"
        journal = journal_path.read_bytes()
        first, rest = journal.split(b"\n", 1)
        header = json.loads(first)
        if header.get("episode") != ep or header.get("task") != task_text("RC"):
            raise ValueError(f"第 {ep + 1} 段动作备份头不一致")
        if task_id:
            header["task"] = task_text(target)
            mutations[journal_path] = json.dumps(header, ensure_ascii=False).encode("utf-8") + b"\n" + rest
    config["scene_annotation_schema"] = 1
    config["scene_annotation_status"] = "completed"
    config.pop("scene_annotation_note", None)
    config["task_scope"] = "per_episode"
    config["scene_definition"] = {"scene_type": scene, "present_objects": SCENES[scene]}
    # task 是初始指令；多任务批次以 meta/tasks.jsonl 和逐帧 task_index 为准。
    config["task"] = task_text(targets[0])
    config["tasks"] = [t["task"] for t in new_tasks]
    info["total_tasks"] = len(new_tasks)
    mutations.update({config_path: json_bytes(config), info_path: json_bytes(info),
                      tasks_path: jsonl_bytes(new_tasks), episodes_path: jsonl_bytes(episodes),
                      stats_path: jsonl_bytes(stats), dataset / ANNOTATION_FILE: json_bytes({
                          "schema_version": 1, "dataset_id": dataset.name, "episodes": labels})})
    videos = {str(p.relative_to(dataset)): sha256(p) for p in dataset.glob("videos/**/*.mp4")}
    backup = ROOT / "运行环境/备份" / ("场景标签修订_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")) / dataset.name
    backup.mkdir(parents=True, exist_ok=False)
    originals = {}
    for path in mutations:
        rel = path.relative_to(dataset)
        originals[str(rel)] = sha256(path) if path.exists() else None
        if path.exists():
            saved = backup / rel
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
    (backup / "原始清单.json").write_bytes(json_bytes({"dataset_root": str(dataset), "files": originals,
                                                   "videos": videos, "parquet": parquet_before}))
    for path, value in mutations.items():
        # 每个文件先完整写入同目录临时文件，再替换；失败时保留备份供恢复。
        temporary = path.with_name(path.name + ".场景修订临时")
        if temporary.exists():
            raise FileExistsError(temporary)
        if isinstance(value, pa.Table):
            pq.write_table(value, temporary)
        else:
            temporary.write_bytes(value)
        temporary.replace(path)
    verified = read_scene_annotations(dataset, tables, new_tasks)
    for path, expected in preserved_columns.items():
        if not pq.read_table(path).drop(["task_index"]).equals(expected):
            raise RuntimeError("任务列以外的数据发生变化")
    for rel, expected in parquet_before.items():
        if dataset / rel not in preserved_columns and sha256(dataset / rel) != expected:
            raise RuntimeError("未修改任务的 parquet 发生变化")
    if videos != {str(p.relative_to(dataset)): sha256(p) for p in dataset.glob("videos/**/*.mp4")}:
        raise RuntimeError("视频发生变化")
    report = {"dataset_root": str(dataset), "episodes": len(verified), "scene_type": scene,
              "targets": dict(Counter(targets)), "backup": str(backup),
              "verification": "场景与任务一致；视频哈希不变；动作、状态及其他逐帧列不变",
              "robot_commands_sent": 0}
    (backup / "修订结果.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--targets", nargs="+", choices=("RC", "BC"), required=True,
                        help="按第1段到最后一段，明确提供每一段的目标")
    args = parser.parse_args()
    print(json.dumps(revise(args.dataset_root, args.scene, args.targets), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
