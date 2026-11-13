"""只读分析模型预测、实际下发与反馈运动，不连接设备、不调整控制限制。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def analyze(path):
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    identity = next(e for e in events if e["type"] == "model_identity")
    dataset, model = Path(identity["dataset_path"]), Path(identity["model_path"])
    provenance = json.loads((model / "training_provenance.json").read_text(encoding="utf-8"))
    training_ids = {int(e["episode"]) for e in provenance["training_data"]["episodes"]}
    tables = [pq.read_table(p).to_pydict() for p in sorted(dataset.glob("data/chunk-*/*.parquet"))]
    tables = [t for t in tables if int(t["episode_index"][0]) in training_ids]
    states = np.concatenate([np.asarray(t["observation.state"]) for t in tables])
    demo_actions = np.concatenate([np.asarray(t["action"]) for t in tables])
    starts = np.asarray([t["observation.state"][0] for t in tables])
    actions = [e for e in events if e["type"] == "action"]
    observed = np.asarray([e["observed_state"] for e in actions])
    sent = np.asarray([e["sent_target"] for e in actions])
    start = next(e["initial_state"] for e in events if e["type"] == "start")
    chunks = [e for e in events if e["type"] in ("async_chunk", "chunk")]
    names = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))["features"]["action"]["names"]
    joints = []
    for i, name in enumerate(names):
        joints.append(dict(name=name, start=start[i], last_observed=observed[-1, i],
                           observed_min=observed[:, i].min(), observed_max=observed[:, i].max(),
                           observed_excursion=np.ptp(observed[:, i]),
                           command_feedback_abs_mean=np.abs(sent[:, i] - observed[:, i]).mean(),
                           command_at_one_unit_fraction=np.mean(np.abs(sent[:, i] - observed[:, i]) >= .98),
                           model_chunk_target_min=min(e["first_action"][i] for e in chunks),
                           model_chunk_target_max=max(e["first_action"][i] for e in chunks),
                           demo_start_min=starts[:, i].min(), demo_start_max=starts[:, i].max(),
                           outside_demo_start=bool(start[i] < starts[:, i].min() or start[i] > starts[:, i].max()),
                           training_tracking_error_p50=np.quantile(np.abs(demo_actions[:, i] - states[:, i]), .5),
                           training_tracking_error_p95=np.quantile(np.abs(demo_actions[:, i] - states[:, i]), .95)))
    return dict(source_log=str(path), model=str(model), hardware_accessed=False,
                settings=events[0], end=next(e for e in reversed(events) if e["type"] == "end"), joints=joints,
                mean_inference_s=float(np.mean([e["inference_s"] for e in chunks])),
                dataset_tasks=[json.loads(s) for s in (dataset / "meta/tasks.jsonl").read_text(encoding="utf-8").splitlines()],
                limitations="历史日志仅记录每块被选中的首个模型目标，无法精确还原其余步的模型目标或判断画面中的物体距离；命令是下发值，不等于实际到达。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=Path, required=True)
    args = parser.parse_args()
    reports = [analyze(path) for path in args.log]
    output = ROOT / "运行记录/推理动作与训练分布排查.json"
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2, default=lambda v: v.item()) + "\n", encoding="utf-8")
    for report in reports:
        print(Path(report["source_log"]).name, Path(report["model"]).name)
        print("训练任务：", report["dataset_tasks"])
        for joint in report["joints"]:
            print(joint["name"], "实际行程", round(joint["observed_excursion"], 2), "命令差达到1比例", round(joint["command_at_one_unit_fraction"], 3),
                  "训练动作反馈差P95", round(joint["training_tracking_error_p95"], 2),
                  "起始", round(joint["start"], 2), "演示起始", [round(joint["demo_start_min"], 2), round(joint["demo_start_max"], 2)])
    print("报告：", output)


if __name__ == "__main__":
    main()
