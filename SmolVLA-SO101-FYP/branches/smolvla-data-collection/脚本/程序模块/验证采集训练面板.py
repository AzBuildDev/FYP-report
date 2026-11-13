"""用本地真实批次与模型运行训练/评估计划和重叠拒绝检查；不训练、不推理、不连接硬件。"""
import argparse
import json
import queue
from datetime import datetime
from pathlib import Path

from 管理采集训练任务 import ProjectTasks, SoftwareJob

ROOT = Path(__file__).resolve().parents[2]


def verify(project, dataset, model):
    specifications = [
        (project.build("train_plan", dataset, [0, 1], steps=1000, name="面板训练计划验证"), "succeeded"),
        (project.build("eval_plan", dataset, [2], model=model, require_held_out=True), "succeeded"),
        (project.build("eval_plan", dataset, [0], model=model, require_held_out=True), "failed"),
    ]
    records = []
    for spec, expected in specifications:
        print(f"开始 {spec.kind}，段号 {spec.episodes}", flush=True)
        job = SoftwareJob(project, spec).start()
        while job.thread.is_alive():
            try:
                print(job.messages.get(timeout=0.2), end="", flush=True)
            except queue.Empty:
                pass
        while not job.messages.empty():
            print(job.messages.get_nowait(), end="", flush=True)
        record = {"kind": spec.kind, "episodes": spec.episodes, "expected": expected, "actual": job.status,
                  "job_directory": str(spec.directory), "report": str(spec.report), "error": job.error}
        if job.status != expected:
            raise RuntimeError(f"任务结果与预期不符：{record}")
        if expected == "failed":
            text = (spec.directory / "控制台输出.log").read_text(encoding="utf-8")
            if "拒绝作为独立验证：存在训练内容重叠" not in text or spec.report.exists():
                raise RuntimeError("重叠任务未按数据来源检查拒绝")
            record["overlap_rejected"] = True
        else:
            if job.result.get("model_inference_executed", False) or job.result.get("checkpoint_saved", False):
                raise RuntimeError("计划验证不应训练或推理")
        records.append(record)
    return {"status": "passed", "created_at": datetime.now().isoformat(timespec="seconds"),
            "scope": "local_data_and_model_plan_integration", "dataset_root": str(Path(dataset).resolve()),
            "model_dir": str(Path(model).resolve()), "checks": records,
            "model_inference_executed": False, "new_training_executed": False,
            "robot_commands_sent": 0, "hardware_tested": False, "task_success_evaluated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="有三段完整演示的批次")
    parser.add_argument("--model-dir", type=Path, required=True, help="只使用第 1、2 段训练并记录来源的模型")
    args = parser.parse_args()
    result = verify(ProjectTasks(ROOT), args.dataset_root, args.model_dir)
    (ROOT / "运行记录/采集训练面板本地计划验证.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("本地批次与模型的三项计划验证通过；未训练、未推理、未连接硬件。", flush=True)
