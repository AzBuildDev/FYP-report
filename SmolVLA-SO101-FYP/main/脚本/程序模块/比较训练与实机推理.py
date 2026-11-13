"""只读比较演示、检查点归一化参数与实机日志，不连接设备或生成动作。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from 管理训练数据来源 import dataset_source, training_stats, verify_checkpoint_stats
from 选择推理模型 import ROOT


def summary(values):
    values = np.asarray(values)
    return {"min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
            "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist()}


def compare(log_path):
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    identity = next(event for event in events if event["type"] == "model_identity")
    model, dataset = Path(identity["model_path"]), Path(identity["dataset_path"])
    provenance = json.loads((model / "training_provenance.json").read_text(encoding="utf-8"))
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    tasks = [json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    all_tables = [pq.read_table(path).to_pydict() for path in sorted(dataset.glob("data/chunk-*/*.parquet"))]
    expected = {ep["episode"]: ep for ep in provenance["training_data"]["episodes"]}
    tables = [table for table in all_tables if int(table["episode_index"][0]) in expected]
    source = dataset_source(dataset, info, tables, tasks)
    fingerprints_match = all(record["action_state_sha256"] == expected[record["episode"]]["action_state_sha256"]
                             and record["video_sha256"] == expected[record["episode"]]["video_sha256"]
                             for record in source["episodes"])
    stats = training_stats(tables)
    models = {}
    for path in sorted(model.parent.iterdir()):
        if (path / "model.safetensors").is_file():
            exported = json.loads((path / "dataset_stats.json").read_text(encoding="utf-8"))
            models[path.name] = {"checkpoint_matches_recomputed_training_stats": verify_checkpoint_stats(path, stats),
                                 "exported_stats_match": all(np.allclose(exported[key][field], values[field], rtol=1e-5, atol=1e-6)
                                                              for key, values in stats.items() for field in ("mean", "std"))}
    states = np.concatenate([np.asarray(t["observation.state"], dtype=np.float32) for t in tables])
    actions = np.concatenate([np.asarray(t["action"], dtype=np.float32) for t in tables])
    starts = np.asarray([t["observation.state"][0] for t in tables], dtype=np.float32)
    initial = np.asarray(next(event["initial_state"] for event in events if event["type"] == "start"), dtype=np.float32)
    gripper = info["features"]["action"]["names"].index("gripper.pos")
    means, std = stats["observation.state"]["mean"], stats["observation.state"]["std"]
    chunks = [event for event in events if event["type"] == "async_chunk"]
    return {"source_log": str(log_path), "hardware_accessed": False,
            "training_episodes": len(tables), "training_frames": len(states), "joint_names": info["features"]["action"]["names"],
            "training_content_and_videos_match_provenance": fingerprints_match,
            "models": models, "training_state": summary(states), "training_action": summary(actions),
            "training_start_pose": summary(starts), "inference_initial_state": initial.tolist(),
            "initial_standardized_state": ((initial - means) / std).tolist(),
            "initial_outside_training_state_range": ((initial < states.min(axis=0)) | (initial > states.max(axis=0))).tolist(),
            "gripper": {"training_state": summary(states[:, gripper]), "training_action": summary(actions[:, gripper]),
                        "training_first_frame_state": summary(starts[:, gripper]),
                        "training_first_frame_action": summary(np.asarray([t["action"][0][gripper] for t in tables])),
                        "inference_initial": float(initial[gripper]),
                        "logged_predicted_targets": [{"t": c["t"], "value": c["first_action"][gripper]} for c in chunks]},
            "end": next(event for event in reversed(events) if event["type"] == "end"),
            "limitations": "原日志未保存逐步关节反馈、实际发送目标、启用时原始位置或实时画面；不能由块级预测还原实际运动，不能确定过载的物理原因。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "运行记录/训练与实机夹爪数据对比.json")
    args = parser.parse_args()
    result = compare(args.log)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
