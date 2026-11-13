"""选择完整演示段并记录内容指纹；防止训练演示复制后被误当作留出验证。"""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_tables(tables, episodes=None, chunk_size=50):
    by_id = {int(table["episode_index"][0]): table for table in tables}
    if len(by_id) != len(tables):
        raise ValueError("演示编号重复")
    selected = sorted(by_id) if episodes is None else list(episodes)
    if not selected or len(selected) != len(set(selected)) or any(ep not in by_id for ep in selected):
        raise ValueError("演示编号为空、重复或不存在（编号从 0 开始）")
    result = [by_id[ep] for ep in sorted(selected)]
    if any(len(table["action"]) < chunk_size for table in result):
        raise ValueError(f"所选每段演示必须至少有 {chunk_size} 帧，不能静默忽略短段")
    return result


def dataset_source(dataset_root, info, tables, tasks):
    root = Path(dataset_root).resolve()
    if info["fps"] != 20:
        raise ValueError("当前控制流程按 20 Hz 执行，只支持 20 FPS 演示")
    names = info["features"]["action"]["names"]
    if len(names) != 6 or names != info["features"]["observation.state"]["names"]:
        raise ValueError("演示必须有六个关节，动作与状态的关节顺序必须一致")
    videos = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    task_names = {task["task_index"]: task["task"] for task in tasks}
    records = []
    for table in tables:
        ep = int(table["episode_index"][0])
        # 排除全局行号、段号；重新编号和另存 parquet 不改变动作/状态身份。
        content = {key: table[key] for key in ("observation.state", "action", "timestamp")}
        content["tasks"] = [task_names[index] for index in table["task_index"]]
        action_hash = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False).encode("utf-8")).hexdigest()
        video_hashes = {}
        for key in videos:
            path = (root / info["video_path"].format(episode_chunk=ep // info["chunks_size"],
                                                    episode_index=ep, video_key=key)).resolve()
            if not path.is_relative_to(root):
                raise ValueError("视频路径超出所选演示目录")
            video_hashes[key] = file_sha256(path)
        records.append({"episode": ep, "frames": len(table["action"]),
                        "action_state_sha256": action_hash, "video_sha256": video_hashes})
    return {"dataset_root": str(root), "fps": info["fps"],
            "joint_names": names,
            "state_names": info["features"]["observation.state"]["names"],
            "episodes": records, "total_frames": sum(record["frames"] for record in records)}


def training_stats(tables):
    import numpy as np
    stats = {}
    for key in ("observation.state", "action"):
        values = np.concatenate([np.asarray(table[key], dtype=np.float32) for table in tables])
        stats[key] = {"mean": values.mean(axis=0), "std": np.maximum(values.std(axis=0), 1e-6)}
    return stats


def verify_checkpoint_stats(model_dir, stats):
    import numpy as np
    from safetensors import safe_open
    with safe_open(str(Path(model_dir) / "model.safetensors"), framework="np") as checkpoint:
        for prefix, feature in (("normalize_inputs", "observation.state"),
                                ("normalize_targets", "action"), ("unnormalize_outputs", "action")):
            for name in ("mean", "std"):
                key = f"{prefix}.buffer_{feature.replace('.', '_')}.{name}"
                actual = checkpoint.get_tensor(key)
                expected = np.asarray(stats[feature][name], dtype=np.float32)
                if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
                    raise ValueError(f"检查点的归一化参数与训练统计不一致：{key}")
    return True


def assess_sources(training, evaluation):
    if training is None:
        return {"scope": "training_provenance_unknown", "held_out_validation": False,
                "overlapping_episodes": [], "reason": "模型缺少绑定检查点的训练来源记录，无法证明数据独立"}
    for key in ("fps", "joint_names", "state_names"):
        if training[key] != evaluation[key]:
            raise ValueError(f"训练与评估的 {key} 不一致")
    action_hashes = {ep["action_state_sha256"] for ep in training["episodes"]}
    video_hashes = {value for ep in training["episodes"] for value in ep["video_sha256"].values()}
    same_batch = training["dataset_root"] == evaluation["dataset_root"]
    training_ids = {ep["episode"] for ep in training["episodes"]}
    # 任一动作/状态内容或视频相同，即保守视为重叠；路径不同不能证明独立。
    overlaps = [ep["episode"] for ep in evaluation["episodes"]
                if (same_batch and ep["episode"] in training_ids) or ep["action_state_sha256"] in action_hashes or
                video_hashes.intersection(ep["video_sha256"].values())]
    return {"scope": "training_demonstration_fit_only" if overlaps else "held_out_demonstration_actions",
            "held_out_validation": not overlaps, "overlapping_episodes": overlaps,
            "same_recording_batch": same_batch,
            "reason": "存在训练内容重叠" if overlaps else "整段留出，未发现相同动作/状态或相同视频；仍非实机成功率"}


def load_training_source(model_dir):
    model_dir = Path(model_dir)
    manifests = [model_dir / "training_provenance.json", model_dir / "训练数据来源.json"]
    manifest = next((path for path in manifests if path.exists()), None)
    if manifest is None:
        return None
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("status") != "completed":
        raise ValueError("训练来源记录格式或完成状态无效")
    if file_sha256(model_dir / "model.safetensors") != record["checkpoint_sha256"]:
        raise ValueError("模型检查点与训练来源记录不匹配")
    return record["training_data"]


def load_training_record(model_dir):
    model_dir = Path(model_dir)
    manifests = [model_dir / "training_provenance.json", model_dir / "训练数据来源.json"]
    manifest = next((path for path in manifests if path.exists()), None)
    if manifest is None:
        return None
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("status") != "completed":
        raise ValueError("训练来源记录格式或完成状态无效")
    if file_sha256(model_dir / "model.safetensors") != record["checkpoint_sha256"]:
        raise ValueError("模型检查点与训练来源记录不匹配")
    return record
