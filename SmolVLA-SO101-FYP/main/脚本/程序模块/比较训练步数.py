"""比较同一数据划分上的微调步数，用整段留出误差选模型；不连接硬件。"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
MODULES = Path(__file__).resolve().parent


def choose_candidate(records):
    """误差距最优不超过 1% 时，优先较少步数；拒绝无效或不同协议的报告。"""
    if not records:
        raise ValueError("没有可比较的结果")
    reference = records[0]["evaluation"]
    for record in records:
        report = record["evaluation"]
        if (report.get("status") != "passed" or not report.get("held_out_validation")
                or not report.get("model_inference_executed")):
            raise ValueError("模型未通过整段留出推理评估")
        score = report["normalized_mae"]
        if not math.isfinite(score) or score < 0:
            raise ValueError("无效评估误差")
        for field in ("evaluation_data", "seed", "metric_protocol_version", "joint_names", "chunk_size"):
            if report[field] != reference[field]:
                raise ValueError(f"评估协议不同：{field}")
        if [(r["episode"], r["frame"]) for r in report["samples_detail"]] != [
                (r["episode"], r["frame"]) for r in reference["samples_detail"]]:
            raise ValueError("评估采样位置不同")
    best = min(r["evaluation"]["normalized_mae"] for r in records)
    close = [r for r in records if r["evaluation"]["normalized_mae"] <= best * 1.01]
    return min(close, key=lambda r: (r["steps"], r["evaluation"]["normalized_mae"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--validation-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[10000])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-episode", type=int, default=5)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "运行记录" / f"步数对比_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    if set(args.train_episodes) & set(args.validation_episodes):
        parser.error("训练段与验证段重叠")
    if min(args.steps + [args.batch_size, args.samples_per_episode]) < 1:
        parser.error("步数、批量及抽样数量必须大于零")
    baseline = args.baseline_dir.resolve()
    source = json.loads((baseline / "training_provenance.json").read_text(encoding="utf-8"))
    if (source["seed"] != args.seed or source["batch_size"] != args.batch_size
            or sorted(r["episode"] for r in source["training_data"]["episodes"]) != sorted(args.train_episodes)
            or Path(source["training_data"]["dataset_root"]).resolve() != args.dataset_root.resolve()):
        parser.error("基线的训练数据、种子或批量与对照计划不同")
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "上游框架/src"), PYTHONIOENCODING="utf-8",
               PYTHONPYCACHEPREFIX=str(ROOT / "运行环境/缓存/字节码"))
    state = {"status": "running", "phase": "starting", "started_at": datetime.now().isoformat(),
             "train_episodes": args.train_episodes, "validation_episodes": args.validation_episodes,
             "seed": args.seed, "batch_size": args.batch_size, "samples_per_episode": args.samples_per_episode,
             "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
             "selection_rule": "minimum held-out normalized MAE; prefer fewer steps within 1% of best",
             "candidates": [], "robot_commands_sent": 0, "task_success_evaluated": False}

    def save_state():
        target = args.run_dir / "对比状态.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(target)

    def run(name, module, argv):
        state["phase"] = name
        save_state()
        print(name, flush=True)
        with (args.run_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-u", str(MODULES / module), *map(str, argv)],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def evaluate(steps, model):
        report_path = args.run_dir / f"{steps}步评估.json"
        run(f"评估{steps}步", "评估微调模型.py", ["--dataset-root", args.dataset_root,
            "--model-dir", model, "--episodes", *args.validation_episodes, "--require-held-out",
            "--samples-per-episode", args.samples_per_episode, "--seed", args.seed,
            "--save-predictions", "--output-report", report_path])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        record = {"steps": steps, "model_dir": str(model), "evaluation_report": str(report_path),
                  "evaluation": report}
        state["candidates"].append(record)
        save_state()

    save_state()
    try:
        evaluate(source["steps"], baseline)
        for steps in sorted(set(args.steps)):
            model = ROOT / "模型" / f"双摄步数对照_{args.run_dir.name}_{steps}步"
            run(f"训练{steps}步", "微调预训练模型.py", ["--dataset-root", args.dataset_root,
                "--episodes", *args.train_episodes, "--steps", steps, "--batch-size", args.batch_size,
                "--seed", args.seed, "--output-dir", model,
                "--output-report", args.run_dir / f"{steps}步训练.json"])
            new_source = json.loads((model / "training_provenance.json").read_text(encoding="utf-8"))
            if new_source["training_data"] != source["training_data"]:
                raise ValueError("对照期间训练数据发生改变，拒绝自动选型")
            evaluate(steps, model)
        selected = choose_candidate(state["candidates"])
        state.update(status="completed", phase="completed", selected_steps=selected["steps"],
                     selected_model_dir=selected["model_dir"], completed_at=datetime.now().isoformat())
        save_state()
        lines = ["# 训练步数对比结果", "", f"选择：{selected['steps']} 步，模型目录：{selected['model_dir']}", "",
                 "| 步数 | 留出标准化动作误差 | 首动作各关节平均绝对误差 |", "|---|---|---|"]
        for record in state["candidates"]:
            r = record["evaluation"]
            lines.append(f"| {record['steps']} | {r['normalized_mae']:.6f} | "
                         + ", ".join(f"{v:.3f}" for v in r["first_action_mae_per_joint"]) + " |")
        lines += ["", "按留出标准化动作误差选型，距最低误差 1% 以内优先较少步数。",
                  "留出段用于选型后属于验证集，不是独立最终测试集。此结果不能证明实机抓取成功。",
                  "原模型保留；没有自动替换实机模型或驱动机械臂。"]
        (args.run_dir / "选型结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已选择 {selected['steps']} 步：{selected['model_dir']}", flush=True)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save_state()
        raise


if __name__ == "__main__":
    main()
