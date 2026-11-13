"""在演示上抽样比较预测动作与示范动作，不连接相机或机械臂。"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.update(HF_HOME=str(ROOT / "运行环境/缓存/huggingface"), HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

import av
import numpy as np
import torch
from 选择推理模型 import load_local_policy
from 检查演示数据 import inspect_dataset
from 管理训练数据来源 import assess_sources, dataset_source, file_sha256, load_training_record, select_tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parent / "训练数据采集/机械臂控制项目/任务演示/单相机演示_20261005_205636_902311")
    parser.add_argument("--samples-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "模型/红色方块微调模型")
    parser.add_argument("--processor-dir", type=Path, help="可选：本机视觉语言处理器目录；默认自动查找")
    parser.add_argument("--episodes", type=int, nargs="+", help="只评估指定完整演示段，编号从 0 开始")
    parser.add_argument("--require-held-out", action="store_true", help="缺少训练来源或数据重叠时拒绝评估")
    parser.add_argument("--check-only", action="store_true", help="只检查数据与训练来源，不运行 GPU 推理")
    parser.add_argument("--output-report", type=Path, help="报告路径；默认保存到运行记录中的独立时间文件")
    parser.add_argument("--save-predictions", action="store_true", help="保存仅在本机使用的动作预测缓存，供执行窗口分析复用")
    args = parser.parse_args()
    if args.samples_per_episode < 1:
        parser.error("每段抽样数量必须大于零")
    if args.output_report and args.output_report.exists():
        parser.error("报告已存在，请指定新文件，保留历史评估记录")
    report, info, tables = inspect_dataset(args.dataset_root, args.output_report.with_name("评估输入数据检查.json")
                                         if args.output_report else ROOT / "运行记录/评估数据检查.json")
    if report["status"] != "passed":
        raise ValueError(report["errors"])
    model_dir = args.model_dir
    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    chunk_size = model_config["chunk_size"]
    tables = select_tables(tables, args.episodes, chunk_size)
    source = dataset_source(args.dataset_root, info, tables, report["tasks"])
    training_record = load_training_record(model_dir)
    training_source = training_record["training_data"] if training_record is not None else None
    assessment = assess_sources(training_source, source)
    checkpoint_hash = (training_record["checkpoint_sha256"] if training_record is not None
                       else file_sha256(model_dir / "model.safetensors"))
    file_identity = {"model_config_sha256": file_sha256(model_dir / "config.json"),
                     "dataset_stats_sha256": file_sha256(model_dir / "dataset_stats.json")}
    if args.require_held_out and not assessment["held_out_validation"]:
        parser.error("拒绝作为独立验证：" + assessment["reason"])
    output = args.output_report or ROOT / "运行记录" / f"模型离线评估_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error("报告已存在，请指定新文件，保留历史评估记录")
    prediction_cache = output.with_name(output.stem + "_动作预测缓存.npz")
    if args.save_predictions and prediction_cache.exists():
        parser.error("动作缓存已存在，请指定新的报告路径")
    if args.check_only:
        result = {"status": "passed", "scope": "evaluation_plan_only", "data_assessment": assessment,
                  "evaluation_data": source, "model_dir": str(model_dir.resolve()), "checkpoint_sha256": checkpoint_hash,
                  "model_inference_executed": False, "robot_commands_sent": 0, "task_success_evaluated": False,
                  "metric_protocol_version": 1, **file_identity}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"评估计划检查通过：{output}；{assessment['reason']}。没有运行模型推理。", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("请使用已配置 CUDA 的项目环境")
    policy = load_local_policy(model_dir, args.processor_dir)
    image_keys = list(policy.config.image_features)
    if image_keys != report["video_keys"]:
        raise ValueError(f"模型图像输入 {image_keys} 与数据视频字段 {report['video_keys']} 不一致")
    chunk_size = policy.config.chunk_size
    names = info["features"]["action"]["names"]
    tasks = {t["task_index"]: t["task"] for t in report["tasks"]}
    stats = json.loads((model_dir / "dataset_stats.json").read_text(encoding="utf-8"))
    std = np.maximum(np.asarray(stats["action"]["std"], dtype=np.float32), 1e-6)
    records, errors, first_errors, predictions = [], [], [], []
    for table in tables:
        ep = int(table["episode_index"][0])
        n = len(table["action"])
        if n < chunk_size:
            raise ValueError(f"第 {ep + 1} 段不足 {chunk_size} 帧")
        indices = sorted(set(np.linspace(0, n - chunk_size, args.samples_per_episode, dtype=int).tolist()))
        frames = {key: {} for key in image_keys}
        for key in image_keys:
            video = args.dataset_root / info["video_path"].format(
                episode_chunk=ep // info["chunks_size"], episode_index=ep, video_key=key)
            with av.open(str(video)) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    if i in indices:
                        frames[key][i] = frame.to_ndarray(format="rgb24")
        for i in indices:
            torch.manual_seed(args.seed + ep * 100000 + i)
            batch = {"observation.state": torch.tensor([table["observation.state"][i]], dtype=torch.float32, device="cuda"),
                     "task": [tasks[table["task_index"][i]]]}
            batch.update({key: torch.from_numpy(frames[key][i].copy()).permute(2, 0, 1)[None].to("cuda", torch.float32) / 255
                          for key in image_keys})
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                action = policy.predict_action_chunk(batch)
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            predicted = action[0].float().cpu().numpy()
            expected = np.asarray(table["action"][i:i + chunk_size], dtype=np.float32)
            if predicted.shape != expected.shape or not np.isfinite(predicted).all():
                raise ValueError("模型动作块形状或数值无效")
            error = np.abs(predicted - expected)
            if args.save_predictions:
                predictions.append(predicted.copy())
            errors.append(error)
            first_errors.append(error[0])
            records.append({"episode": ep, "frame": i, "inference_s": latency,
                            "mae_per_joint": error.mean(axis=0).tolist(),
                            "normalized_mae": float((error / std).mean())})
            print(f"第 {ep + 1} 段 / 第 {i} 帧：标准化误差 {records[-1]['normalized_mae']:.3f}，推理 {latency:.3f} 秒", flush=True)
    combined = np.concatenate(errors)
    steady = [r["inference_s"] for r in records[1:]] or [records[0]["inference_s"]]
    result = {"status": "passed", "created_at": datetime.now().isoformat(timespec="seconds"),
              **assessment, "task_success_evaluated": False, "robot_commands_sent": 0,
              "evaluation_data": source, "model_dir": str(model_dir.resolve()), "checkpoint_sha256": checkpoint_hash,
              "model_inference_executed": True, "metric_protocol_version": 1, **file_identity,
              "episodes": len(tables), "frames": source["total_frames"], "samples": len(records),
              "seed": args.seed, "action_units": "body RANGE_M100_100; gripper RANGE_0_100; not degrees",
              "joint_names": names, "chunk_size": chunk_size,
              "mae_per_joint": combined.mean(axis=0).tolist(),
              "first_action_mae_per_joint": np.asarray(first_errors).mean(axis=0).tolist(),
              "normalized_mae": float((combined / std).mean()),
              "steady_inference_median_s": float(np.median(steady)),
              "steady_inference_p95_s": float(np.percentile(steady, 95)),
              "execution_window_s": 20 / 20,
              "blocking_loop_effective_hz_estimate": 20 / (float(np.median(steady)) + 1),
              "samples_detail": records}
    if args.save_predictions:
        np.savez_compressed(prediction_cache, predicted_actions=np.stack(predictions))
        result["prediction_cache"] = {"filename": prediction_cache.name, "sha256": file_sha256(prediction_cache),
                                      "shape": [len(records), chunk_size, 6], "format": "npz_no_pickle",
                                      "local_only": True}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"评估已保存：{output}。{assessment['reason']}；动作误差不是实机成功率。", flush=True)


if __name__ == "__main__":
    main()
