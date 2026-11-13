"""复用预测缓存，按延迟与最多执行步数分析原始目标误差；不模拟闭环物理运动。"""
import json
import math
from pathlib import Path

import numpy as np

from 管理离线实验对比 import compare_reports, load_tables, vector
from 管理训练数据来源 import file_sha256


def window_indices(chunk_size, delay_s, hz=20, execute_limit=20):
    if (type(chunk_size) is not int or chunk_size < 1 or type(execute_limit) is not int or execute_limit < 1 or
        not math.isfinite(delay_s) or delay_s < 0 or not math.isfinite(hz) or hz <= 0):
        raise ValueError("动作块、执行步数、延迟或频率无效")
    # 与控制入口首次 offset 的取整相同；这里假设后续每个控制节拍恰好推进一步。
    first = int(delay_s * hz)
    return list(range(first, min(chunk_size, first + execute_limit)))


def summarize_case(name, samples, scale, delays, hz=20, execute_limit=20):
    if len(samples) != len(delays) or not samples:
        raise ValueError("每个采样窗口必须有一个延迟")
    scale = vector(scale, "统一动作尺度", positive=True)
    errors = {method: [] for method in ("模型预测", "保持当前位置", "训练动作均值")}
    windows = []
    for sample, delay in zip(samples, delays):
        expected = np.asarray(sample["expected"], dtype=np.float64)
        predicted = np.asarray(sample["predicted"], dtype=np.float64)
        if (expected.ndim != 2 or expected.shape[1] != 6 or not len(expected) or expected.shape != predicted.shape or
            not np.isfinite(expected).all() or not np.isfinite(predicted).all()):
            raise ValueError("示范或预测动作块无效")
        indices = window_indices(len(expected), delay, hz, execute_limit)
        windows.append({"episode": sample["episode"], "frame": sample["frame"], "delay_s": float(delay),
                        "selected_indices": indices, "scored_targets": len(indices), "expired": not indices})
        if not indices:
            continue
        for method, targets in (("模型预测", predicted), ("保持当前位置", sample["state"]), ("训练动作均值", sample["train_mean"])):
            targets = np.asarray(targets, dtype=np.float64)
            if targets.shape not in ((6,), expected.shape) or not np.isfinite(targets).all():
                raise ValueError("对照动作无效")
            selected = targets[indices] if targets.ndim == 2 else targets
            errors[method].append(np.abs(selected - expected[indices]))
    rows = []
    for method, blocks in errors.items():
        joint_mae = np.concatenate(blocks).mean(axis=0) if blocks else None
        rows.append({"name": method, "mae_per_joint": joint_mae.tolist() if joint_mae is not None else None,
                     "common_scale_mae": float((joint_mae / scale).mean()) if joint_mae is not None else None})
    return {"name": name, "scored_targets": sum(row["scored_targets"] for row in windows),
            "valid_windows": sum(not row["expired"] for row in windows),
            "expired_windows": sum(row["expired"] for row in windows), "windows": windows, "rows": rows}


def load_predictions(report, report_path):
    cache = report.get("prediction_cache")
    if not cache or cache.get("local_only") is not True or cache.get("format") != "npz_no_pickle":
        raise ValueError("报告没有本地动作预测缓存，不能从平均误差反推执行窗口")
    parent = Path(report_path).resolve().parent
    path = (parent / cache["filename"]).resolve()
    if path.parent != parent or path.suffix != ".npz" or file_sha256(path) != cache["sha256"]:
        raise ValueError("缓存路径或内容指纹不一致")
    shape = (report["samples"], report["chunk_size"], 6)
    with np.load(path, allow_pickle=False) as archive:
        predictions = archive["predicted_actions"].copy()
    if predictions.shape != shape or list(shape) != cache["shape"] or not np.isfinite(predictions).all():
        raise ValueError("预测缓存形状或数值无效")
    return predictions


def analyze_report(root, report_path, fixed_delays=(0., .4, .8, 1.2, 2.5), execute_limit=20):
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 复用已测试的模型、来源、数据内容与归一化检查。任何旧/重叠/计划报告均不放宽。
    validated = compare_reports(root, [report_path], report["model_dir"])
    predictions = load_predictions(report, report_path)
    _, _, tables = load_tables(report["evaluation_data"]["dataset_root"])
    by_id = {int(table["episode_index"][0]): table for table in tables}
    stats = json.loads((Path(report["model_dir"]) / "dataset_stats.json").read_text(encoding="utf-8"))
    samples = []
    for prediction, row in zip(predictions, report["samples_detail"]):
        ep, frame = row["episode"], row["frame"]
        table = by_id[ep]
        expected = np.asarray(table["action"][frame:frame + report["chunk_size"]], dtype=np.float32)
        raw_mae = np.abs(prediction - expected).mean(axis=0)
        if not np.allclose(raw_mae, row["mae_per_joint"], rtol=2e-5, atol=1e-5):
            raise ValueError("缓存计算的误差与原始评估明细不一致")
        samples.append({"episode": ep, "frame": frame, "predicted": prediction, "expected": expected,
                        "state": table["observation.state"][frame], "train_mean": stats["action"]["mean"]})
    hz = report["evaluation_data"]["fps"]
    cases = [summarize_case(f"固定延迟 {delay:g} 秒", samples, validated["common_action_std"], [delay] * len(samples), hz, execute_limit)
             for delay in fixed_delays]
    latencies = [row["inference_s"] for row in report["samples_detail"]]
    cases.append(summarize_case("本次 CUDA 推理耗时假设", samples, validated["common_action_std"], latencies, hz, execute_limit))
    return {"status": "passed", "scope": "teacher_forced_execution_window_sensitivity", "source_report": report_path.name,
            "source_report_sha256": file_sha256(report_path), "checkpoint_sha256": report["checkpoint_sha256"],
            "model": Path(report["model_dir"]).relative_to(root).as_posix(), "source_evaluation_date": report["created_at"],
            "evaluation_data": validated["protocol"]["source"], "source_windows": validated["protocol"]["windows"],
            "seed": report["seed"], "chunk_size": report["chunk_size"], "hz": hz, "execute_limit": execute_limit,
            "common_action_std": validated["common_action_std"], "cases": cases,
            "reused_prediction_cache": True, "new_model_inference_executed": False, "new_training_executed": False,
            "hardware_tested": False, "robot_commands_sent": 0, "task_success_evaluated": False,
            "notes": ["按示范状态与未来示范动作比较原始目标，未模拟反馈、限幅、接触或物理轨迹。",
                      "真实入口发现整块过期会停止；这里独立分析各采样窗口，不代表过期后继续真实控制。",
                      "每种情形假设控制节拍不丢步，最多执行指定步数，且期间没有新结果替换动作块。",
                      "CUDA 推理耗时只计模型调用，不含相机帧龄、预处理、线程等待或主线程接收延迟。",
                      "过期窗口不记作零误差；统计按有效目标步数加权，窗口可能重叠，非独立任务试验。",
                      "模型与对照使用同一索引和尺度，不能把误差降低解释为实机任务成功。"]}
