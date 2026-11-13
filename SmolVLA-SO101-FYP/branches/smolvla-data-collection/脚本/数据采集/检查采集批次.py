"""检查新采集批次完整性、人工标签与失败归档；只读数据，不连接硬件。"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "脚本/程序模块"))
from 检查演示数据 import inspect_dataset
from 管理采集尝试 import JOINTS, write_json


def inspect_collection(path, report_path=None):
    path = Path(path).resolve()
    result, info, tables = inspect_dataset(path, report_path)
    errors, warnings = list(result["errors"]), list(result["warnings"])
    if info["fps"] != 20 or any(info["features"][key]["names"] != JOINTS for key in ("action", "observation.state")):
        errors.append("数据需为 20 FPS、固定顺序六关节")
    if info["features"].get("observation.images.front", {}).get("shape") != [480, 640, 3]:
        errors.append("桌面图像规格需为 480×640×3")
    config_path = path / "采集配置.json"
    if not config_path.exists():
        warnings.append("历史批次没有新的场景/用途标签；不能据此自动分配训练与验证")
        config = {}
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema_version") == 2 and config.get("split") not in {"train", "held_out"}:
            errors.append("批次缺少训练/独立验证用途")
    attempts, saved = [], {}
    for status_path in sorted((path / "采集尝试").glob("*/尝试状态.json")):
        record = json.loads(status_path.read_text(encoding="utf-8"))
        attempts.append(record)
        if record["status"] in {"recording", "pending", "save_failed"} or not record.get("archive_complete"):
            errors.append(f"{record['attempt_id']}：尝试未完成归档/确认，请先处理")
        if record["status"] != "saved":
            warnings.append(f"{record['attempt_id']}：{record['status']}，不纳入已保存演示")
            continue
        episode = record["episode_index"]
        if episode in saved:
            errors.append(f"第 {episode} 段有重复成功记录")
        saved[episode] = record
        if not record.get("human_success") or not record.get("archive_complete"):
            errors.append(f"第 {episode} 段缺少人工确认或完整归档")
        if not record.get("labels", {}).get("scene") or record.get("labels", {}).get("kind") not in {"ordinary", "recovery"}:
            errors.append(f"第 {episode} 段缺少场景/演示类型")
        directory = status_path.parent
        journal = directory / "动作与状态.jsonl"
        rows = [json.loads(x) for x in journal.read_text(encoding="utf-8").splitlines() if x.strip()]
        frames = [x for x in rows if x["type"] == "frame"]
        if hashlib.sha256(journal.read_bytes()).hexdigest() != record.get("journal_sha256"):
            errors.append(f"第 {episode} 段动作备份内容发生变化")
        table = next((x for x in tables if x["episode_index"] and x["episode_index"][0] == episode), None)
        if table is None or len(frames) != record["frames"] or len(table["timestamp"]) != record["frames"]:
            errors.append(f"第 {episode} 段动作备份与表格帧数不一致")
        else:
            import numpy as np
            for source, key in (("state", "observation.state"), ("action", "action")):
                if not np.allclose([x[source] for x in frames], table[key], atol=1e-5):
                    errors.append(f"第 {episode} 段 {source} 备份与表格不一致")
        if len(list((directory / "原始帧").glob("frame_*.png"))) != record["frames"]:
            errors.append(f"第 {episode} 段原始帧归档数量不一致")
    if config.get("schema_version") == 2 and set(saved) != set(range(info["total_episodes"])):
        errors.append("已保存段与采集尝试记录不一一对应")
    sealed = path / "批次封存状态.json"
    if sealed.exists():
        errors.append("批次因保存/清理异常封存，请先人工检查，不直接用于训练")
    result.update(status="passed" if not errors else "failed", errors=errors, warnings=warnings,
                  split=config.get("split", "unknown"), attempts=attempts,
                  task_success_manually_verified=bool(saved) and all(r.get("human_success") for r in saved.values()),
                  note="人工确认的是示范完成，不能据此计算自主抓取成功率")
    if report_path:
        write_json(report_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()
    if args.dataset_root is None:
        value = input("请粘贴完整采集批次目录：").strip().strip('"')
        if not value:
            parser.error("批次目录不能为空")
        args.dataset_root = Path(value)
    if args.dataset_root is None:
        value = input("请粘贴完整采集批次目录：").strip().strip('"')
        if not value:
            parser.error("批次目录不能为空")
        args.dataset_root = Path(value)
    if args.dataset_root is None:
        value = input("请粘贴完整采集批次目录：").strip().strip('"')
        if not value:
            parser.error("批次目录不能为空")
        args.dataset_root = Path(value)
    if args.dataset_root is None:
        value = input("请粘贴完整采集批次目录：").strip().strip('"')
        if not value:
            parser.error("批次目录不能为空")
        args.dataset_root = Path(value)
    report = args.output_report or ROOT / "运行记录" / f"采集批次检查_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    if report.exists():
        parser.error("报告已存在，禁止覆盖")
    try:
        result = inspect_collection(args.dataset_root, report)
    except Exception as exc:
        result = {"status": "failed", "errors": [str(exc)], "robot_commands_sent": 0}
        write_json(report, result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"episodes", "attempts"}}, ensure_ascii=False, indent=2))
    print("报告：" + str(report))
    raise SystemExit(0 if result["status"] == "passed" else 1)
