"""核对训练、评估与当前检查点，生成可提交的同批次留出实验摘要。"""
import argparse
import json
from pathlib import Path

from 管理训练数据来源 import assess_sources, dataset_source, load_training_source, select_tables, training_stats, verify_checkpoint_stats

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    args = parser.parse_args()
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    checks = json.loads((ROOT / "运行记录/训练评估流程自检.json").read_text(encoding="utf-8"))
    if any(record["status"] != "passed" for record in (training, evaluation, checks)):
        raise ValueError("训练、评估或流程检查尚未通过")
    model = Path(training["output_dir"]).resolve()
    if not model.is_relative_to((ROOT / "模型").resolve()) or model != Path(evaluation["model_dir"]).resolve():
        raise ValueError("训练与评估不是同一项目模型")
    if training["checkpoint_sha256"] != evaluation["checkpoint_sha256"]:
        raise ValueError("训练与评估不是同一检查点")
    source = load_training_source(model)
    if source != training["training_data"]:
        raise ValueError("训练报告与模型训练来源不一致")
    import pyarrow.parquet as pq
    dataset = Path(source["dataset_root"])
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    tasks = [json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tables = [pq.read_table(path).to_pydict() for path in sorted(dataset.glob("data/chunk-*/*.parquet"))]
    selected = select_tables(tables, training["episodes"])
    if dataset_source(dataset, info, selected, tasks) != source:
        raise ValueError("当前训练演示内容与训练时记录不同")
    verify_checkpoint_stats(model, training_stats(selected))
    assessment = assess_sources(source, evaluation["evaluation_data"])
    if not assessment["held_out_validation"] or not evaluation["held_out_validation"]:
        raise ValueError("评估数据未通过整段留出检查")
    result = {"status": "passed", "scope": "held_out_demonstration_actions", **assessment,
              "model": str(model.relative_to(ROOT)), "checkpoint_sha256": training["checkpoint_sha256"],
              "training": {key: training[key] for key in ("steps", "batch_size", "episodes", "total_frames",
                           "first_50_avg_loss", "last_50_avg_loss", "seconds", "peak_gpu_memory_mib")},
              "evaluation": {key: evaluation[key] for key in ("created_at", "episodes", "frames", "samples", "seed",
                             "joint_names", "mae_per_joint", "normalized_mae", "steady_inference_median_s", "samples_detail")},
              "validation_episode_ids": [ep["episode"] for ep in evaluation["evaluation_data"]["episodes"]],
              "checkpoint_training_statistics_verified": True,
              "pipeline_checks": checks, "hardware_tested": False, "robot_commands_sent": 0,
              "task_success_evaluated": False,
              "notes": "同批次完整段留出，未参与本次训练；不证明独立新场景泛化或实机抓取成功率。"}
    output = ROOT / "文档/项目进展/整段留出实验.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已核对并生成实验摘要：{output}")


if __name__ == "__main__":
    main()
