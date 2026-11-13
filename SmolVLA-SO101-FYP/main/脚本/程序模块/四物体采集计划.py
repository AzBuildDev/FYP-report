"""四物体单区/双区任务目录、布局与逐段标签；导出计划不连接硬件。"""
import argparse
import csv
import json
from itertools import permutations
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
OBJECTS = {"RC": "red cube", "BC": "blue cube", "GCy": "green cylinder", "RCy": "red cylinder"}
STAGES = ("I-A", "I-B", "II")
POSITIONS = tuple(f"P{i}" for i in range(1, 6))
ORDERS = tuple(permutations(OBJECTS))[:10]


def task_text(object_id, destination="target_area"):
    if object_id not in OBJECTS or destination not in ("target_area", "A", "B"):
        raise ValueError("物体或目的地无效")
    target = "the target area" if destination == "target_area" else f"zone {destination}"
    return f"Pick up the {OBJECTS[object_id]} and place it in {target}."


def layout(round_id, layout_id):
    if type(round_id) is not int or not 1 <= round_id <= 10 or type(layout_id) is not int or not 1 <= layout_id <= 5:
        raise ValueError("轮次应为 1–10，布局应为 1–5")
    values = (*ORDERS[round_id - 1], "empty")
    shift = layout_id - 1
    values = values[-shift:] + values[:-shift] if shift else values
    return dict(zip(POSITIONS, values))


def plan_row(stage, object_id, round_id, position="P1", layout_id=1, destination="target_area"):
    if stage not in STAGES or object_id not in OBJECTS:
        raise ValueError("请选择阶段和物体")
    if stage == "II" and destination not in ("A", "B"):
        raise ValueError("双区阶段必须明确选择 A 或 B")
    if stage != "II" and destination != "target_area":
        raise ValueError("单区阶段使用原 target_area")
    placement = layout(round_id, layout_id)
    if stage == "I-A":
        if position not in POSITIONS:
            raise ValueError("起点应为 P1–P5")
        layout_id = POSITIONS.index(position) + 1
        placement = {p: object_id if p == position else "empty" for p in POSITIONS}
        group = f"I-A_{object_id}_{position}_R{round_id:02d}"
    else:
        position = next(p for p, obj in placement.items() if obj == object_id)
        # I-B 与 II 共享组身份，所有对象和目的地都继承同一 split。
        group = f"R{round_id:02d}-K{layout_id}"
    return {"plan_id": f"{stage}_{group}_{object_id}_{destination}", "stage": stage,
            "scene_type": "single" if stage == "I-A" else "four_objects",
            "scene_group_id": group, "round_id": round_id, "layout_id": layout_id,
            "object_id": object_id, "destination": destination, "position": position,
            "split": "train" if round_id <= 8 else "dev", "task_text": task_text(object_id, destination),
            "placement": placement}


def collection_plan():
    rows = []
    for stage in STAGES:
        for r in range(1, 11):
            for k in range(1, 6):
                for obj in OBJECTS:
                    for dest in (("A", "B") if stage == "II" else ("target_area",)):
                        rows.append(plan_row(stage, obj, r, POSITIONS[k - 1], k, dest))
    return rows


def episode_label(row, dataset_id, episode_id, session, operator, layout_config):
    if not all(str(v).strip() for v in (session, operator, layout_config)):
        raise ValueError("请填写会话、操作者和布局/硬件配置版本")
    return {"schema_version": 1, **row, "dataset_id": dataset_id, "episode_id": episode_id,
            "session": session.strip(), "operator": operator.strip(), "layout_config": layout_config.strip(),
            "quality": "PASS", "failure_code": "", "created_at": datetime.now().astimezone().isoformat()}


def label_path(root, episode_id):
    return Path(root) / "meta/collection_labels" / f"episode_{episode_id:06d}.json"


def write_label(root, label):
    path = label_path(root, label["episode_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    # 不覆盖标签；保存中断时交给数据检查发现缺失标签。
    with path.open("x", encoding="utf-8") as stream:
        json.dump(label, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read_labels(root, tables, tasks, check_all=True):
    root = Path(root)
    config_path = root / "采集配置.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    paths = list((root / "meta/collection_labels").glob("*.json"))
    if not paths and not config.get("collection_label_schema"):
        return {}  # 旧数据不推填标签或重新分组。
    if config.get("collection_label_schema") != 1:
        raise ValueError("四物体标签缺少支持的采集配置版本")
    task_names = {x["task_index"]: x["task"] for x in tasks}
    by_ep = {int(t["episode_index"][0]): t for t in tables}
    expected = {label_path(root, ep).name for ep in by_ep}
    actual = {p.name for p in paths}
    if (actual != expected if check_all else not expected.issubset(actual)):
        raise ValueError("四物体逐段标签缺失或多余，禁止静默当作旧数据")
    labels, slots = {}, set()
    for ep, table in by_ep.items():
        label = json.loads(label_path(root, ep).read_text(encoding="utf-8"))
        row = plan_row(label["stage"], label["object_id"], label["round_id"], label["position"],
                       label["layout_id"], label["destination"])
        if (label.get("schema_version") != 1 or label.get("episode_id") != ep or
                label.get("dataset_id") != root.name or label.get("quality") != "PASS" or label.get("failure_code") or
                any(label.get(k) != v for k, v in row.items()) or
                any(not str(label.get(k, "")).strip() for k in ("session", "operator", "layout_config"))):
            raise ValueError(f"第 {ep + 1} 段标签与采集计划不一致")
        if {task_names[i] for i in table["task_index"]} != {row["task_text"]}:
            raise ValueError(f"第 {ep + 1} 段任务文字与标签不一致")
        slot = (label["layout_config"], label["plan_id"])
        if slot in slots:
            raise ValueError("同一计划配额重复保存，请单独归档重录而非重复计数")
        slots.add(slot)
        labels[ep] = label
    return labels


def require_training_split(source):
    invalid = [ep["episode"] for ep in source["episodes"]
               if ep.get("collection_label", {}).get("split", "train") != "train"]
    if invalid:
        raise ValueError(f"开发段禁止参与训练，请仅选择 train 段（从 0 编号）：{invalid}")


def export_plan(output):
    rows = collection_plan()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], "dataset_id", "episode_id", "quality", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "placement": json.dumps(row["placement"], ensure_ascii=False)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / f"运行记录/四物体采集计划_{datetime.now():%Y%m%d_%H%M%S_%f}.csv")
    args = parser.parse_args()
    rows = export_plan(args.output)
    print(f"已导出 {len(rows)} 条待采集计划（train 640 / dev 160）：{args.output}")
    print("这是空白计划，不代表已采集。旧红方块 50 段须先逐段核对原始归属才能抵扣。")


if __name__ == "__main__":
    main()
