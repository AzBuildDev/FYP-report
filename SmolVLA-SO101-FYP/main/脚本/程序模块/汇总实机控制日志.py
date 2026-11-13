"""只读汇总声明为实机的完整控制日志，保存节拍摘要；不判定任务结果、不访问硬件。"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from statistics import median

from 管理实机评测记录 import EvaluationStore

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    run = EvaluationStore(ROOT / "运行记录")._checked_run(args.log)
    if run["run_kind"] != "real_hardware" or not run["started"] or not run["sent_commands"]:
        raise ValueError("需要完整的、声明为实机模式且发送过动作的控制日志")
    raw = args.log.read_bytes()
    if hashlib.sha256(raw).hexdigest() != run["log_sha256"]:
        raise ValueError("提取期间日志内容改变，请等待记录结束后重试")
    events = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    latencies = [event["inference_s"] for event in events if event.get("type") == "async_chunk"]
    result = {"scope": "runtime_log_telemetry_pending_human_review", "created_at": datetime.now().isoformat(timespec="seconds"),
              "source_log": run["filename"], "source_log_sha256": run["log_sha256"], "declared_run_kind": run["run_kind"],
              "checkpoint_sha256": run["checkpoint_sha256"], "scene": run["scene"], "mode": run["mode"],
              "control_settings": run["control_settings"],
              "recorded_commands": run["sent_commands"], "recorded_inferences": run["inferences"],
              "active_control_hz": run["active_control_hz"], "missed_ticks": run["missed_ticks"],
              "seconds": run["seconds"], "stop_reason": run["stop_reason"], "logged_errors": run["errors"],
              "steady_inference_median_s": median(latencies[1:] or latencies) if latencies else None,
              "verification_read_only": True, "robot_commands_sent_by_verifier": 0,
              "task_success_evaluated": False,
              "notes": "指标来自日志声明和记录，只反映控制程序；任务结果需本人观察后人工标记，不能证明物理跟随或抓取成功。"}
    output = ROOT / "文档/项目进展/实机控制日志待确认摘要.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已保存只读控制日志摘要：{output}；未判定任务结果。")


if __name__ == "__main__":
    main()
