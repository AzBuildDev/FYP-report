"""只读统合已有运行记录和当前连接检查，不启动模型、相机或机械臂。"""
import json
from datetime import datetime
from pathlib import Path
from 管理实机评测记录 import read_run
from 管理训练数据来源 import file_sha256
ROOT = Path(__file__).resolve().parents[2]
logs, output = ROOT / "运行记录", ROOT / "文档/项目进展"
recovery = logs / "机械臂只读检查_20261007_181141_223993.json"
arms = json.loads(recovery.read_text(encoding="utf-8"))
recovery_ok = len(arms) == 2 and all(len(arm["motors"]) == 6 and all(m["error"] == 0 and m["fault_status"] == 0 and not m.get("read_errors") for m in arm["motors"]) for arm in arms)
follower = next(arm for arm in arms if arm["role_from_prior_usb_mapping"] == "follower")
assert recovery_ok and all(m["torque_enabled"] == 0 for m in follower["motors"])
fields = ("filename","log_sha256","checkpoint_sha256","scene","mode","run_kind","sent_commands","inferences","active_control_hz","missed_ticks","seconds","stop_reason","errors")
names = ["实机抓取测试_20261007_181235_825879.jsonl","实机抓取测试_20261007_182008_938057.jsonl","实机抓取测试_20261007_182329_105923.jsonl","实机抓取测试_20261007_182637_236203.jsonl"]
runs=[]
for name in names:
    data=read_run(logs/name)
    events=[json.loads(line) for line in (logs/name).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    runs.append({k:data[k] for k in fields} | {"configured_test_seconds":events[0]["test_seconds"]})
current=logs/"机械臂只读检查_20261007_202955_835486.json"
devices=json.loads(current.read_text(encoding="utf-8"))
state={"scope":"consolidated_runtime_history_without_new_motion","created_at":datetime.now().isoformat(timespec="seconds"),
       "operator_reset_confirmation":"已断电检查然后已经重新上电",
       "recovery_check":{"source":recovery.name,"sha256":file_sha256(recovery),"both_arms_fault_free":True,"follower_torque_disabled":True},
       "subsequent_runs":runs,"current_connection_check":{"source":current.name,"sha256":file_sha256(current),"devices_detected":len(devices),
       "motors_torque_off_confirmed_now":False if devices else None},
       "robot_commands_sent_by_consolidator":0,"task_success_evaluated":False,"success_rate_evaluated":False,
       "notes":"后续运行指标来自完整日志声明，未单独证明物体结果；当前未检出设备不等于故障或电机扭矩已关闭。"}
(output/"项目运行记录统合摘要.json").write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
precheck=output/"再测前故障检查摘要.json"
data=json.loads(precheck.read_text(encoding="utf-8-sig"))
data.update(operator_power_off_confirmation_pending=False,operator_power_cycle_confirmed=True,
            operator_confirmation=state["operator_reset_confirmation"],recovery_check=state["recovery_check"],
            note="本文件最初检查失败状态保留；后续断电复查及新日志见项目运行记录统合摘要.json。")
precheck.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"runs":len(runs),"latest_commands":runs[-1]["sent_commands"],"latest_errors":runs[-1]["errors"],"current_devices":len(devices)},ensure_ascii=False))
