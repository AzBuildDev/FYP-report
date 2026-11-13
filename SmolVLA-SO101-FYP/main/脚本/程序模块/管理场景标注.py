"""读取人工确认的场景与抓取目标；独立于四物体采集配额，不推填位置或划分。"""
import json
from pathlib import Path

from 四物体采集计划 import task_text


ANNOTATION_FILE = "meta/scene_annotations.json"
SCENES = {"single_red_cube": ["RC"], "red_blue_cubes": ["RC", "BC"]}


def read_scene_annotations(root, tables, tasks, check_all=True):
    root = Path(root)
    if (root / "废片归档.json").exists():
        raise ValueError("该批次已由操作者判定为废片并归档，禁止用于训练或评估")
    path = root / ANNOTATION_FILE
    config_path = root / "采集配置.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    if config.get("scene_annotation_status") == "pending_targets":
        raise ValueError("该批红蓝场景尚未确认逐段抓取目标，禁止按原全红任务训练")
    version = config.get("scene_annotation_schema")
    if not path.exists() and version is None:
        return {}
    if version != 1:
        raise ValueError("场景标注缺少支持的采集配置版本")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("dataset_id") != root.name:
        raise ValueError("场景标注版本或数据集身份不匹配")
    labels = {}
    for label in payload["episodes"]:
        ep = label["episode_id"]
        if type(ep) is not int or ep in labels:
            raise ValueError("场景标注段号无效或重复")
        present = SCENES.get(label.get("scene_type"))
        target = label.get("target_object_id")
        if (present is None or label.get("present_objects") != present or target not in present or
                label.get("destination") != "target_area" or
                label.get("task_text") != task_text(target) or
                label.get("annotation_source") != "user_confirmation"):
            raise ValueError(f"第 {ep + 1} 段场景或目标标注无效")
        labels[ep] = label
    expected = {int(t["episode_index"][0]) for t in tables}
    if (set(labels) != expected if check_all else not expected.issubset(labels)):
        raise ValueError("逐段场景标注缺失或多余")
    task_names = {t["task_index"]: t["task"] for t in tasks}
    for table in tables:
        ep = int(table["episode_index"][0])
        if {task_names[i] for i in table["task_index"]} != {labels[ep]["task_text"]}:
            raise ValueError(f"第 {ep + 1} 段训练任务与场景目标不一致")
    return {ep: labels[ep] for ep in sorted(expected)}
