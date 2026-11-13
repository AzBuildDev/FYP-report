"""核对已完成的留出评估，用相同窗口与统一尺度比较模型及简单对照；不运行模型或硬件。"""
import json
from pathlib import Path

import numpy as np

from 管理训练数据来源 import (assess_sources, dataset_source, file_sha256, load_training_source,
                        select_tables, training_stats, verify_checkpoint_stats)

UNITS = "body RANGE_M100_100; gripper RANGE_0_100; not degrees"


def vector(value, label, positive=False):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (6,) or not np.isfinite(result).all() or np.any(result <= 0 if positive else result < 0):
        raise ValueError(f"{label}必须是六个有效{'正数' if positive else '非负数'}")
    return result


def load_tables(root):
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    tasks = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tables = [pq.read_table(path).to_pydict() for path in sorted(root.glob("data/chunk-*/*.parquet"))]
    return info, tasks, tables


def source_identity(source):
    # 路径不能改变内容身份；对比仍要求相同段号、帧数、任务/动作与视频内容。
    return {key: source[key] for key in ("fps", "joint_names", "state_names", "episodes", "total_frames")}


def sample_plan(report):
    chunk = report.get("chunk_size")
    if type(chunk) is not int or chunk < 1 or type(report.get("seed")) is not int:
        raise ValueError("评估缺少有效动作块长度或随机种子")
    lengths = {row["episode"]: row["frames"] for row in report["evaluation_data"]["episodes"]}
    details = report.get("samples_detail", [])
    if not details or len(details) != report.get("samples"):
        raise ValueError("采样明细数量与报告不一致")
    pairs = []
    for row in details:
        ep, frame = row["episode"], row["frame"]
        if type(ep) is not int or type(frame) is not int or ep not in lengths or not 0 <= frame <= lengths[ep] - chunk:
            raise ValueError("采样位置无效或动作窗口超出演示")
        vector(row["mae_per_joint"], "采样误差")
        pairs.append((ep, frame))
    if len(pairs) != len(set(pairs)) or pairs != sorted(pairs) or set(lengths) != {ep for ep, _ in pairs}:
        raise ValueError("采样窗口重复、无序或遗漏整段")
    return {"seed": report["seed"], "chunk_size": chunk, "windows": pairs,
            "source": source_identity(report["evaluation_data"]), "units": report["action_units"], "names": report["joint_names"]}


def validate_report(report, model, current_source, training_source, stats):
    if (report.get("status") != "passed" or report.get("held_out_validation") is not True or
        report.get("scope") != "held_out_demonstration_actions" or report.get("robot_commands_sent") != 0 or
        report.get("task_success_evaluated") is not False or report.get("model_inference_executed") is False):
        raise ValueError("只接受已完成、无硬件命令的留出动作评估；不能使用计划或训练拟合报告")
    if Path(report["model_dir"]).resolve() != model or report.get("checkpoint_sha256") != file_sha256(model / "model.safetensors"):
        raise ValueError("报告的模型位置或检查点内容与当前模型不一致")
    if source_identity(report["evaluation_data"]) != source_identity(current_source):
        raise ValueError("评估数据内容已变化，不能复用旧误差")
    assessment = assess_sources(training_source, current_source)
    if not assessment["held_out_validation"]:
        raise ValueError("当前训练来源无法证明验证数据留出")
    if report["action_units"] != UNITS or report["joint_names"] != current_source["joint_names"]:
        raise ValueError("动作单位或关节顺序不一致")
    protocol = sample_plan(report)
    raw = vector(report["mae_per_joint"], "模型平均误差")
    detail_mean = np.asarray([row["mae_per_joint"] for row in report["samples_detail"]]).mean(axis=0)
    std = vector(stats["action"]["std"], "模型动作标准差", positive=True)
    if not np.allclose(raw, detail_mean, rtol=2e-5, atol=1e-6):
        raise ValueError("平均误差与采样明细不一致")
    if not np.isclose(float(report["normalized_mae"]), (raw / std).mean(), rtol=2e-5, atol=1e-6):
        raise ValueError("归一化误差与当前模型统计不一致")
    # 新报告绑定推理配置与统计文件。旧报告未记录这两种文件指纹，保留这一限制。
    has_protocol = report.get("metric_protocol_version") == 1
    if has_protocol:
        for field, filename in (("model_config_sha256", "config.json"), ("dataset_stats_sha256", "dataset_stats.json")):
            if report.get(field) != file_sha256(model / filename):
                raise ValueError("评估时的配置或统计文件与当前模型不同")
    return protocol, assessment, has_protocol


def summarize(name, kind, errors, scale, first=None):
    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("对比误差应为有效采样数×六关节")
    mean = values.mean(axis=0)
    normalized_samples = (values / vector(scale, "统一动作尺度", positive=True)).mean(axis=1)
    result = {"name": name, "kind": kind, "samples": len(values), "mae_per_joint": mean.tolist(),
              "raw_joint_mean_mae": float(mean.mean()), "common_scale_mae": float(normalized_samples.mean()),
              "common_scale_sample_mae": normalized_samples.tolist()}
    if first is not None:
        result["first_action_mae_per_joint"] = vector(first, "首步平均误差").tolist()
    return result


def baseline_rows(tables, windows, chunk, action_mean, scale):
    mean = np.asarray(action_mean, dtype=np.float64)
    if mean.shape != (6,) or not np.isfinite(mean).all():
        raise ValueError("训练动作均值无效")
    by_id = {int(table["episode_index"][0]): table for table in tables}
    errors = {"保持当前位置": [], "训练动作均值": []}
    first = {name: [] for name in errors}
    for ep, frame in windows:
        table = by_id[ep]
        expected = np.asarray(table["action"][frame:frame + chunk], dtype=np.float64)
        state = np.asarray(table["observation.state"][frame], dtype=np.float64)
        if expected.shape != (chunk, 6) or state.shape != (6,) or not np.isfinite(expected).all() or not np.isfinite(state).all():
            raise ValueError("真实动作或状态无效")
        # 两种对照均只用当前观测或训练统计；不能读取验证段未来动作来构造预测。
        for name, predicted in (("保持当前位置", state), ("训练动作均值", mean)):
            error = np.abs(expected - predicted)
            errors[name].append(error.mean(axis=0))
            first[name].append(error[0])
    return [summarize(name, "non_model_baseline", errors[name], scale, np.asarray(first[name]).mean(axis=0)) for name in errors]


def compare_reports(root, report_paths, reference_model):
    root, reference_model = Path(root).resolve(), Path(reference_model).resolve()
    data_root, model_root = root.parent / "训练数据采集/机械臂控制项目", root / "模型"
    if not reference_model.is_relative_to(model_root.resolve()):
        raise ValueError("统一尺度模型必须在本项目模型目录")
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths]
    if not reports:
        raise ValueError("至少需要一份真实留出评估报告")
    cache = {}
    def current_training(model):
        if model in cache:
            return cache[model]
        if not model.is_relative_to(model_root.resolve()):
            raise ValueError("评估模型必须属于本项目")
        source = load_training_source(model)
        if source is None:
            raise ValueError("模型训练来源未知，不能用于留出对比")
        dataset = Path(source["dataset_root"]).resolve()
        if not dataset.is_relative_to(data_root.resolve()):
            raise ValueError("训练数据必须属于本毕业设计")
        info, tasks, tables = load_tables(dataset)
        selected = select_tables(tables, [row["episode"] for row in source["episodes"]])
        if dataset_source(dataset, info, selected, tasks) != source:
            raise ValueError("训练数据已变化，不能复用来源统计")
        stats = training_stats(selected)
        verify_checkpoint_stats(model, stats)
        saved = json.loads((model / "dataset_stats.json").read_text(encoding="utf-8"))
        for feature in ("action", "observation.state"):
            for field in ("mean", "std"):
                if not np.allclose(saved[feature][field], stats[feature][field], rtol=1e-5, atol=1e-6):
                    raise ValueError("模型旁的统计文件与实际训练段不一致")
        cache[model] = source, saved
        return cache[model]
    reference_source, reference_stats = current_training(reference_model)
    scale = vector(reference_stats["action"]["std"], "统一动作尺度", positive=True)
    rows, first_protocol, current_eval_tables = [], None, None
    seen = set()
    for path, report in zip(report_paths, reports):
        model = Path(report["model_dir"]).resolve()
        source, stats = current_training(model)
        evaluation = report["evaluation_data"]
        dataset = Path(evaluation["dataset_root"]).resolve()
        if not dataset.is_relative_to(data_root.resolve()):
            raise ValueError("验证数据必须属于本毕业设计")
        info, tasks, tables = load_tables(dataset)
        selected = select_tables(tables, [row["episode"] for row in evaluation["episodes"]], report["chunk_size"])
        current_source = dataset_source(dataset, info, selected, tasks)
        protocol, assessment, has_protocol = validate_report(report, model, current_source, source, stats)
        if not assess_sources(reference_source, current_source)["held_out_validation"]:
            raise ValueError("统一尺度或均值对照来源包含验证数据")
        if first_protocol is None:
            first_protocol, current_eval_tables = protocol, selected
        elif protocol != first_protocol:
            raise ValueError("报告的验证内容、采样帧、种子、动作块或关节协议不同，不能直接对比")
        identity = report["checkpoint_sha256"]
        if identity in seen:
            raise ValueError("同一检查点的重复报告不能充当不同模型")
        seen.add(identity)
        row = summarize(model.name, "smolvla_model", [item["mae_per_joint"] for item in report["samples_detail"]], scale,
                        report["first_action_mae_per_joint"])
        row.update(model=model.relative_to(root).as_posix(), checkpoint_sha256=identity,
                   source_report=Path(path).name, source_report_sha256=file_sha256(path),
                   evaluation_date=report["created_at"], own_scale_mae=report["normalized_mae"],
                   inference_files_fingerprinted=has_protocol, same_recording_batch=assessment["same_recording_batch"])
        rows.append(row)
    baseline = baseline_rows(current_eval_tables, first_protocol["windows"], first_protocol["chunk_size"],
                             reference_stats["action"]["mean"], scale)
    for row in rows:
        row["relative_to_hold_position"] = (row["common_scale_mae"] / baseline[0]["common_scale_mae"]
                                              if baseline[0]["common_scale_mae"] > 0 else None)
    return {"status": "passed", "scope": "descriptive_held_out_action_comparison", "protocol": first_protocol,
            "reference_model": reference_model.relative_to(root).as_posix(),
            "reference_checkpoint_sha256": file_sha256(reference_model / "model.safetensors"),
            "common_action_std": scale.tolist(), "rows": rows + baseline,
            "statistical_significance_evaluated": False, "task_success_evaluated": False,
            "new_model_inference_executed": False, "new_training_executed": False,
            "hardware_tested": False, "robot_commands_sent": 0,
            "limitations": ["窗口可能重叠且来自同一演示，不能把窗口当作独立任务试验。",
                            "当前只复用已保存模型误差，不重新执行模型或真实机械臂。",
                            "旧评估未记录推理配置和统计文件指纹；各模型自己的标准差缩放指标不能直接跨模型比较。",
                            "比较完整预测动作块，不能代表异步控制跳过过期步骤后的实际执行窗口。",
                            "保持当前位置对照只预测动作；既不执行动作，也不证明静止可完成抓取。"]}
