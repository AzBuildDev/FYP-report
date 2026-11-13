"""从本地验证结果生成可提交 GitHub 的进展摘要，不复制原始视频或模型。"""
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
output = ROOT / "文档/项目进展"
output.mkdir(parents=True, exist_ok=True)
logs = ROOT / "运行记录"
training = json.loads((logs / "模型微调结果.json").read_text(encoding="utf-8"))
evaluation = json.loads((logs / "微调模型离线评估.json").read_text(encoding="utf-8"))
asynchronous = json.loads((logs / "异步模型验证.json").read_text(encoding="utf-8")) if (logs / "异步模型验证.json").exists() else None
loading = json.loads((ROOT.parents[1] / "公共工具/整理记录/模型重定位检查.json").read_text(encoding="utf-8"))
if evaluation["status"] != "passed" or loading["status"] != "passed":
    raise RuntimeError("当前离线评估或模型加载未通过，停止生成完成说明")
summary = {key: training[key] for key in ["status", "steps", "batch_size", "precision", "episodes", "total_frames", "first_50_avg_loss", "last_50_avg_loss", "seconds", "peak_gpu_memory_mib", "gpu", "checkpoint_saved", "robot_commands_sent", "task_success_evaluated"]}
summary["dataset"] = "../训练数据采集/机械臂控制项目/任务演示/单相机演示_20261005_205636_902311"
summary["model"] = "模型/红色方块微调模型"
for name, body in [("微调训练摘要.json", summary), ("离线动作评估.json", evaluation)]:
    (output / name).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
async_section = ""
if asynchronous:
    if asynchronous["status"] != "passed":
        raise RuntimeError("异步模型验证尚未通过")
    (output / "异步模型验证.json").write_text(json.dumps(asynchronous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks = json.loads((logs / "控制流程完整自检.json").read_text(encoding="utf-8"))
    if checks["status"] != "passed":
        raise RuntimeError("控制流程完整自检尚未通过")
    (output / "控制流程自检.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stop_text = (f"{asynchronous['stop_response_s']*1000:.1f} 毫秒" if asynchronous['stop_response_s'] > 0
                 else "当前时钟分辨率内")
    async_section = f"""## 异步控制验证

实机入口现默认采用异步模式，模型工作线程只处理输入快照；串口读取与动作发送由主线程执行。动作按时间跳过过期前缀，控制节拍不等待 GPU。每步检查相机时效、有限数值、实时位置限幅与运行范围，每个动作块最多发送 20 步。当前入口从确认时的姿态开始，运行范围合并当前姿态与演示范围，裁剪到标定行程；仍需手动确认后才启用扭矩。

本地微调模型加模拟关节、已录制图像运行 8 秒，模拟发送 {asynchronous['control']['sent_commands']} 次动作、完成 {asynchronous['control']['inferences']} 次推理，开始发送动作后的平均频率 **{asynchronous['control']['active_control_hz']:.2f} Hz**，推理中位耗时 **{asynchronous['chunk_inference_median_s']:.3f} 秒**。推理进行期间设置停止事件，控制线程在 **{stop_text}**返回，没有发送停止测试中的模拟动作。该测量受本机时钟分辨率影响，不代表真实机械臂释放扭矩耗时。

以上只验证软件并发、限幅、停止与调度；模拟关节即时跟随命令，不能代表真实负载、接触、串口时延或抓取成功率。完整报告：[异步模型验证](异步模型验证.json)。输入与清理检查、异步检查及完整入口模拟检查共 {checks['tests_run']} 项通过；[自检报告](控制流程自检.json)由 `控制流程完整自检.bat` 生成，测试替换所有硬件接口。

保留 `实机抓取测试.bat --mode serial` 作为旧串行对照，默认入口使用 `--mode async`。等待按回车期间也能通过停止按钮、关闭监视窗口、Esc 或 Ctrl+C 取消，并记录未开始、零动作的日志。退出先释放机械臂资源，再等待后台推理结束。

"""
names = ["肩部左右", "肩部升降", "肘部", "腕部俯仰", "腕部旋转", "夹爪"]
experiment_section = ""
experiment_path = output / "整段留出实验.json"
if experiment_path.exists():
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment["status"] != "passed" or not experiment["held_out_validation"] or not experiment["checkpoint_training_statistics_verified"]:
        raise RuntimeError("整段留出实验尚未通过核对")
    et, ev = experiment["training"], experiment["evaluation"]
    experiment_section = f"""## 整段留出实验

从基础模型重新训练单独的 `{experiment['model']}`，只用第 1、2 段的 {et['total_frames']} 帧训练 {et['steps']} 步，batch={et['batch_size']}，耗时 {et['seconds']/60:.2f} 分钟。第 3 段的 {ev['frames']} 帧完全留出，未进入训练或归一化统计；检查点内参数与训练段统计核对一致。原有三段、3000 步模型保留供实机入口使用。

对留出段抽样 {ev['samples']} 个动作块，按本次训练动作标准差缩放的平均绝对误差 **{ev['normalized_mae']:.4f}**。这是同一采集批次的一段留出结果，样本间可能相关，不能证明独立新场景泛化或实机成功率，也不能与三段训练基线直接比较模型优劣。完整指标：[整段留出实验](整段留出实验.json)。

训练和评估支持整段选择、模型目录选择及数据重叠检查。模型来源文件绑定检查点内容，识别同段号及复制的动作/状态或视频；未知来源不能通过强制留出检查。训练评估流程 {experiment['pipeline_checks']['tests_run']} 项软件检查通过。使用方法见[训练与独立验证流程](../训练与独立验证流程.md)。

"""
rows = "\n".join(f"| {name} | {mae:.3f} |" for name, mae in zip(names, evaluation["mae_per_joint"]))
recording_section = ""
recording_path = logs / "实机评测记录自检.json"
if recording_path.exists():
    recording = json.loads(recording_path.read_text(encoding="utf-8"))
    if recording["status"] != "passed":
        raise RuntimeError("实机评测记录自检尚未通过")
    (output / "实机评测记录自检.json").write_text(json.dumps(recording, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    recording_section = f"""## 实机结果记录

新增 `实机结果记录.bat`：只读取完整日志，人工选择真实结果、确认本人观察、填写场景和备注。标记绑定日志内容，修正保留历史；同一日志副本不重复计数，模拟日志不能计入实机结果。按场景、检查点、模式与任务分组，有待标记的尝试时不显示成功率，人工中断和程序异常保留在尝试分母中，零动作另列。

新控制日志包含模型内容指纹、场景、停止原因与准备阶段结束记录；Ctrl+C 在异步控制内部也归为用户停止。{recording['tests_run']} 项评测记录软件检查通过，包含隐藏窗口中的选择、保存和导出流程；报告：[实机评测记录自检](实机评测记录自检.json)。这些软件自检没有启动真实硬件，也未产生真实任务成功率。使用方法见[实机结果记录说明](../实机结果记录说明.md)。

"""
    telemetry_path = output / "实机控制日志待确认摘要.json"
    if telemetry_path.exists():
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        if telemetry["scope"] != "runtime_log_telemetry_pending_human_review" or telemetry["task_success_evaluated"]:
            raise RuntimeError("控制日志摘要状态无效")
        frequency = f"{telemetry['active_control_hz']:.2f} Hz" if telemetry['active_control_hz'] is not None else "未记录"
        recorded_delta = telemetry.get("control_settings", {}).get("max_step_delta", "未记录")
        recording_section += f"日志 `{telemetry['source_log']}` 声明为实机异步模式，该次每步限幅为 {recorded_delta} 个归一化单位，记录发送 {telemetry['recorded_commands']} 次动作、{telemetry['recorded_inferences']} 次推理，活动发送频率 {frequency}，停止原因为 `{telemetry['stop_reason']}`。这次只读提取没有启动硬件；摘要只记录控制程序指标，不判断任务结果或抓取成功率，任务结果通过记录窗口人工标记。摘要：[实机控制日志待确认摘要](实机控制日志待确认摘要.json)。\n\n"
        if telemetry.get("logged_errors"):
            recording_section += "本次摘要含程序异常，不能据此确认扭矩已关闭或任务成功。历史夹爪过载及后续检查分别保留在[实机过载排查记录](实机过载排查记录.md)，不能用后续正常测试覆盖原有失败。\n\n"
    retry_path = output / "断电检查后实机复测摘要.json"
    if retry_path.exists():
        retry = json.loads(retry_path.read_text(encoding="utf-8"))
        if (not telemetry_path.exists() or retry["source_log_sha256"] != telemetry["source_log_sha256"]
                or not retry["operator_power_cycle_confirmed"] or not retry["precheck"]["both_arms_fault_free"]
                or not retry["postcheck"]["both_arms_fault_free"] or not retry["postcheck"]["follower_torque_disabled"]
                or telemetry["logged_errors"] or telemetry["stop_reason"] != "time_limit"):
            raise RuntimeError("复测摘要与最新日志或电机检查不一致，请保留记录检查")
        recording_section += f"用户已确认断电检查、夹爪没有卡住并重新上电。修复后按授权完成 {retry['requested_seconds']} 秒复测；日志未记录程序异常，结束后独立读取两臂 12 个电机，故障状态均为 0，从臂六个电机扭矩均为 0。监视窗口与串口正常退出。这些程序与只读检查指标不判断物体任务结果，也不表示长期过载原因已查明。报告：[复测摘要](断电检查后实机复测摘要.json)。原 60 秒过载记录继续保留：[历史控制摘要](实机过载控制日志摘要.json)、[排查记录](实机过载排查记录.md)。\n\n"
        human_path = output / "实机复测人工结果.json"
        if human_path.exists():
            human = json.loads(human_path.read_text(encoding="utf-8"))
            annotation = human["annotation"]
            if (human["scope"] != "human_observed_single_hardware_retry" or not annotation["observed_real"]
                    or annotation["log_sha256"] != retry["source_log_sha256"]
                    or annotation["log_filename"] != retry["source_log"] or human["success_rate_evaluated"]):
                raise RuntimeError("人工结果与复测日志不一致")
            recording_section += f"用户现场确认本次结果为 **{annotation['outcome']}**，由助手代录并绑定日志内容指纹，保留修订历史。这是单次人工确认结果，没有进行多场景成功率估计。[人工结果](实机复测人工结果.json)。\n\n"
latest_hardware_notice = ""
video_retry_path = output / "录像复测运行摘要.json"
if video_retry_path.exists():
    video_retry = json.loads(video_retry_path.read_text(encoding="utf-8"))
    if (video_retry["scope"] != "user_requested_video_retries_runtime_logs"
            or video_retry["task_success_evaluated"] or video_retry["success_rate_evaluated"]):
        raise RuntimeError("录像复测摘要不能替代人工任务结果")
    from 管理实机评测记录 import read_run
    for trial in video_retry["trials"]:
        source = (logs / trial["filename"]).resolve()
        if source.parent != logs.resolve() or read_run(source)["log_sha256"] != trial["log_sha256"]:
            raise RuntimeError("录像复测摘要与原日志不一致")
    latest = video_retry["trials"][-1]
    recording_section += f"## 后续录像复测\n\n另完成 {len(video_retry['trials'])} 次用户要求的 {video_retry['test_seconds']} 秒录像复测，保留原有模型、功率及限幅。最后一次 `{latest['filename']}` 发送 {latest['sent_commands']} 次动作、{latest['inferences']} 次推理。用户明确要求直接开始，因此仅该次调用跳过回车等待，准备检查和停止按钮仍保留；普通批处理入口仍要求回车。\n\n"
    if latest["errors"]:
        latest_hardware_notice = "**最新实机状态：录像复测结束时肩部左右电机停扭矩读回为 1，全部停扭矩未确认。已要求托住从臂并切断从臂电源，暂停动作测试；后续须确认断电并重新检查，不能沿用此前的单次成功状态。**\n\n"
        recording_section += f"**最新测试退出异常：{'；'.join(latest['errors'])}。不能确认从臂全部扭矩已关闭，已要求托住从臂并切断从臂电源；不再自动启用。** 操作者断电确认状态见[录像复测摘要](录像复测运行摘要.json)。旧的单次成功记录仍保留，不能用于证明本次正常退出。\n\n"
    recording_section += "录像由用户自行拍摄，程序未自动保存视频；这几次物体结果尚未人工标记，不计算任务成功率。完整指标：[录像复测摘要](录像复测运行摘要.json)。\n\n"
failed_precheck_path = output / "再测前故障检查摘要.json"
if failed_precheck_path.exists():
    failed_precheck = json.loads(failed_precheck_path.read_text(encoding="utf-8"))
    from hashlib import sha256
    source = (logs / failed_precheck["source_report"]).resolve()
    if (failed_precheck["scope"] != "failed_read_only_precheck_no_motion"
            or failed_precheck["robot_commands_sent"] != 0 or failed_precheck["model_test_started"]
            or source.parent != logs.resolve()
            or sha256(source.read_bytes()).hexdigest() != failed_precheck["source_report_sha256"]):
        raise RuntimeError("再测前故障检查摘要与原始记录不一致")
    latest_hardware_notice = "**最新实机状态：再测前只读检查失败，肩部左右电机扭矩仍为 1，夹爪故障状态为 32（本机 SDK 定义为过载）。本次没有启动控制、没有发送动作。已要求托住从臂并断电检查，处理和复查通过前不启用测试。**\n\n"
    recording_section += "## 再次请求测试的只读检查\n\n用户再次要求直接开始，先独立只读检查两臂。肩部左右电机扭矩为 1，夹爪故障寄存器与返回包错误均为 32；根据本机 SDK 的常量，该位表示过载。夹爪的扭矩读回也带错误，不能把数值 0 当作已确认关闭。此次未加载模型或发送运动目标，不是一次已开始的任务尝试。已要求托住从臂、切断从臂电源并检查夹爪，断电状态待用户确认；不通过位置目标复位故障。[故障检查摘要](再测前故障检查摘要.json)。\n\n"
# 本轮统合保留历史失败，并用随后独立检查及完整日志更新当前状态。
state_path = output / "项目运行记录统合摘要.json"
if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state["scope"] != "consolidated_runtime_history_without_new_motion" or state["robot_commands_sent_by_consolidator"] != 0:
        raise RuntimeError("项目统合摘要状态无效")
    from 管理训练数据来源 import file_sha256
    from 管理实机评测记录 import read_run
    for evidence in (state["recovery_check"], state["current_connection_check"]):
        source = (logs / evidence["source"]).resolve()
        if source.parent != logs.resolve() or file_sha256(source) != evidence["sha256"]:
            raise RuntimeError("统合摘要与原始检查不一致")
    for trial in state["subsequent_runs"]:
        if read_run(logs / trial["filename"])["log_sha256"] != trial["log_sha256"]:
            raise RuntimeError("统合摘要与后续日志不一致")
    latest = state["subsequent_runs"][-1]
    latest_hardware_notice = ("用户随后确认断电检查并重新上电，独立复查两臂无故障、从臂扭矩全部关闭。"
        f"后续保留 {len(state['subsequent_runs'])} 条完整运行日志；最后一条发送 {latest['sent_commands']} 次动作，"
        f"清理错误 {len(latest['errors'])} 条。最新连接检查检出 {state['current_connection_check']['devices_detected']} 个设备，"
        "本轮目录整理没有启用硬件。后续物体结果仍需现场标记，历史过载与停扭矩异常保留。[统合摘要](项目运行记录统合摘要.json)。\n\n")
    recording_section += "## 后续恢复与本轮整理\n\n" + latest_hardware_notice
    layout_path = output / "脚本入口整理报告.json"
    if layout_path.exists():
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        if layout["status"] != "passed":
            raise RuntimeError("脚本整理未通过")
        recording_section += "原来分散的 89 个脚本已按用途分类。顶层仅保留项目控制台批处理、控制台程序、入口清单和使用说明；日常操作集中为六项，另外区分离线实验、检查工具、环境配置、历史工具和内部程序模块。旧脚本保留原始快照，兼容别名不再维护重复实现。新增入口未自动启动硬件；隐藏窗口、Windows 带空格路径调用、主要流程软件检查及迁移后的真实模型离线启动验证均通过。[整理报告](脚本入口整理报告.json)、[操作说明](../../脚本/使用说明.md)。\n\n"

panel_section = ""
panel_path = logs / "采集训练面板自检.json"
integration_path = logs / "采集训练面板本地计划验证.json"
if panel_path.exists() and integration_path.exists():
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    integration = json.loads(integration_path.read_text(encoding="utf-8"))
    expected_checks = [("train_plan", [0, 1], "succeeded"), ("eval_plan", [2], "succeeded"), ("eval_plan", [0], "failed")]
    if (panel["status"] != "passed" or integration["status"] != "passed" or
        panel["robot_commands_sent"] != 0 or integration["robot_commands_sent"] != 0 or
        integration["new_training_executed"] or integration["model_inference_executed"] or
        [(row["kind"], row["episodes"], row["actual"]) for row in integration["checks"]] != expected_checks or
        not integration["checks"][2].get("overlap_rejected")):
        raise RuntimeError("采集训练面板尚未通过软件与本地计划验证")
    (output / "采集训练面板自检.json").write_text(json.dumps(panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    portable = {key: integration[key] for key in ("status", "created_at", "scope", "model_inference_executed",
                "new_training_executed", "robot_commands_sent", "hardware_tested", "task_success_evaluated")}
    portable["dataset"] = "../训练数据采集/机械臂控制项目/任务演示/" + Path(integration["dataset_root"]).name
    portable["model"] = "模型/" + Path(integration["model_dir"]).name
    portable["checks"] = [{key: row[key] for key in ("kind", "episodes", "expected", "actual")}
                          | ({"overlap_rejected": True} if row.get("overlap_rejected") else {}) for row in integration["checks"]]
    (output / "采集训练面板本地计划验证.json").write_text(json.dumps(portable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    panel_section = f"""## 采集与训练面板

新增 `采集训练面板.bat`，在窗口选择训练与评估批次、完整演示段和微调模型，查看软件任务输出、停止任务及打开记录目录。训练模型另存，报告使用独立任务目录；默认强制检查评估数据留出。没有有效报告时不会因退出码为零而显示完成，训练计划与评估计划分别明确显示尚未训练、尚未推理。

{panel['tests_run']} 项软件检查通过，包含隐藏窗口操作、中断自己的任务、失败输出保留及报告与参数/模型对应检查。用本地真实三段演示及整段留出模型完成 3 项计划验证：第 1、2 段训练计划通过，第 3 段评估计划通过，第 1 段因训练内容重叠被拒绝。该项验证只检查计划，没有再次训练、运行模型推理或发送真实机械臂命令。报告：[面板自检](采集训练面板自检.json)、[本地计划验证](采集训练面板本地计划验证.json)。

录制与实机结果记录窗口仅由操作者点击打开，面板不会自动开始录制或抓取，也不会将新模型自动用于实机。关闭面板会停止自己的软件任务，独立录制窗口需自行停止。详细步骤见[采集与训练面板说明](../采集与训练面板说明.md)。

"""
comparison_section = ""
comparison_pointer = logs / "离线实验对比最新结果.json"
if comparison_pointer.exists():
    pointer = json.loads(comparison_pointer.read_text(encoding="utf-8"))
    source_dir = Path(pointer["result_dir"]).resolve()
    if not source_dir.is_relative_to((logs / "离线实验对比").resolve()):
        raise RuntimeError("离线对比结果不在本项目运行记录内")
    comparison = json.loads((source_dir / "对比结果.json").read_text(encoding="utf-8"))
    comparison_checks = json.loads((logs / "离线实验对比自检.json").read_text(encoding="utf-8"))
    if (pointer["status"] != "passed" or comparison["status"] != "passed" or comparison_checks["status"] != "passed" or
        comparison["scope"] != "descriptive_held_out_action_comparison" or comparison["robot_commands_sent"] != 0 or
        comparison["new_model_inference_executed"] or comparison["new_training_executed"] or comparison["task_success_evaluated"]):
        raise RuntimeError("离线对比或软件检查未通过，停止发布")
    compare_output = output / "离线实验对比"
    compare_output.mkdir(exist_ok=True)
    for filename in ("对比结果.json", "对比指标.csv", "对比说明.md", "动作误差对比.svg"):
        artifact = (source_dir / filename).read_text(encoding="utf-8-sig")
        if filename.endswith(".svg"):
            artifact = "\n".join(line.rstrip() for line in artifact.splitlines()) + "\n"
        (compare_output / filename).write_text(artifact, encoding="utf-8")
    (output / "离线实验对比自检.json").write_text(json.dumps(comparison_checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compare_rows = "\n".join(f"| {row['name']} | {row['common_scale_mae']:.4f} |" for row in comparison["rows"])
    hold = next(row for row in comparison["rows"] if row["name"] == "保持当前位置" and row["kind"] == "non_model_baseline")
    findings = []
    for row in comparison["rows"]:
        if row["kind"] != "smolvla_model":
            continue
        relation = "低于" if row["common_scale_mae"] < hold["common_scale_mae"] else "不低于"
        findings.append(f"{row['name']}总体误差{relation}保持当前位置对照。")
        if row["mae_per_joint"][4] > hold["mae_per_joint"][4]:
            findings.append(f"{row['name']}的腕部旋转误差高于该对照，应在新增演示与实际控制窗口评估中继续检查。")
    comparison_findings = "".join(findings)
    comparison_section = f"""## 留出动作与简单对照

复用已完成的留出推理结果，核对当前数据与检查点后，增加保持当前位置和训练动作均值两种对照。所有行使用相同 {len(comparison['protocol']['windows'])} 个采样窗口、{comparison['protocol']['chunk_size']} 步动作块与统一训练动作标准差，避免不同模型的各自尺度混用。完整 50 步覆盖 2.5 秒预测，不能代表异步控制跳过过期步骤后的实际执行窗口。

| 方法 | 统一尺度平均动作误差 ↓ |
|---|---:|
{compare_rows}

这是已保存留出报告的描述性结果，窗口可能重叠；没有重新训练或执行模型推理，未运行真实机械臂，未测量统计显著性或任务成功率。{comparison_findings}

{comparison_checks['tests_run']} 项对比软件检查通过，涵盖统一尺度、报告/检查点绑定、训练和验证内容变化、重复窗口、采样协议差异与图表导出。[图表与完整指标](离线实验对比/对比说明.md)、[软件检查](离线实验对比自检.json)、[操作说明](../离线实验对比说明.md)。新评估报告增加推理配置和统计文件指纹；旧报告保留未记录这些指纹的限制，不能声称完整复现历史推理环境。

"""
window_section = ""
window_pointer = logs / "执行窗口分析最新结果.json"
if window_pointer.exists():
    pointer = json.loads(window_pointer.read_text(encoding="utf-8"))
    source_dir = Path(pointer["result_dir"]).resolve()
    source_report_path = Path(pointer["evaluation_report"]).resolve()
    if (not source_dir.is_relative_to((logs / "执行窗口分析").resolve()) or
        not source_report_path.is_relative_to(logs.resolve())):
        raise RuntimeError("执行窗口结果或原始报告不在本项目运行记录内")
    window = json.loads((source_dir / "执行窗口结果.json").read_text(encoding="utf-8"))
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    window_checks = json.loads((logs / "执行窗口误差自检.json").read_text(encoding="utf-8"))
    if (pointer["status"] != "passed" or window["status"] != "passed" or window_checks["status"] != "passed" or
        window["scope"] != "teacher_forced_execution_window_sensitivity" or window["robot_commands_sent"] != 0 or
        window["task_success_evaluated"] or window["new_model_inference_executed"] or window["new_training_executed"] or
        source_report["model_inference_executed"] is not True or source_report["robot_commands_sent"] != 0):
        raise RuntimeError("执行窗口分析或原始离线推理状态无效")
    from 管理训练数据来源 import file_sha256
    if file_sha256(source_report_path) != window["source_report_sha256"]:
        raise RuntimeError("原始评估报告与窗口结果不一致")
    window_output = output / "执行窗口分析"
    window_output.mkdir(exist_ok=True)
    for filename in ("执行窗口结果.json", "执行窗口指标.csv", "执行窗口说明.md", "执行窗口误差.svg"):
        (window_output / filename).write_text((source_dir / filename).read_text(encoding="utf-8-sig"), encoding="utf-8")
    (output / "执行窗口误差自检.json").write_text(json.dumps(window_checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    case_rows = []
    for case in window["cases"]:
        values = [f"{row['common_scale_mae']:.4f}" if row["common_scale_mae"] is not None else "不计算" for row in case["rows"][:2]]
        case_rows.append(f"| {case['name']} | {case['scored_targets']} | {case['expired_windows']} | " + " | ".join(values) + " |")
    window_table = "\n".join(case_rows)
    first_latency = source_report["samples_detail"][0]["inference_s"]
    window_section = f"""## 执行窗口假设下的离线误差

在 {window['source_evaluation_date']} 另存 {source_report['samples']} 个留出动作块的离线预测，记录逐步动作缓存，模型与数据保留原有版本，没有训练新模型。窗口分析复用该缓存，按延迟后的索引、最多 {window['execute_limit']} 步分析原始目标，未启动真实机械臂。

| 情形 | 有效目标步数 | 过期窗口 | 模型误差 | 保持当前位置 |
|---|---:|---:|---:|---:|
{window_table}

每种情形内部使用相同索引与训练尺度；不同情形选择的未来示范动作不同，不能推断增加延迟会改善控制。首次 GPU 模型调用耗时 **{first_latency:.3f} 秒**，完整动作块有效期为 {window['chunk_size']/window['hz']:.2f} 秒；CUDA 耗时假设下，首次窗口过期。真实入口发现整块过期会停止，此表逐窗口独立分析，不能当作过期后持续运行的实机记录。

未模拟实时反馈、限幅、相机变化或物体接触。腕部旋转误差仍需继续检查。{window_checks['tests_run']} 项软件检查通过，覆盖索引、过期不计零误差、按有效步数加权、缓存与报告/数据绑定及导出。[完整指标和图表](执行窗口分析/执行窗口说明.md)、[软件检查](执行窗口误差自检.json)、[操作说明](../执行窗口误差分析说明.md)。预测缓存仅留本机，不上传原始动作或视频。

"""
warmup_section = ""
warmup_pointer = logs / "模型启动离线验证最新结果.json"
if warmup_pointer.exists():
    from 管理训练数据来源 import file_sha256
    pointer = json.loads(warmup_pointer.read_text(encoding="utf-8"))
    source_path = Path(pointer["report"]).resolve()
    if not source_path.is_relative_to(logs.resolve()) or file_sha256(source_path) != pointer["report_sha256"]:
        raise RuntimeError("启动验证报告路径或内容不匹配")
    startup = json.loads(source_path.read_text(encoding="utf-8"))
    checks = json.loads((logs / "控制流程完整自检.json").read_text(encoding="utf-8"))
    if checks["status"] != "passed" or "模型预热自检" not in checks["modules"]:
        raise RuntimeError("启动控制软件检查尚未通过")
    if (pointer["status"] != "passed" or startup["status"] != "passed" or
            startup["scope"] != "offline_model_startup" or startup["hardware_tested"] or
            startup["robot_commands_sent"] != 0 or startup["task_success_evaluated"] or
            startup["warmup"]["status"] != "ready" or startup["warmup"]["prediction_reused"] or
            startup["cancel_prediction_reused"]):
        raise RuntimeError("启动离线验证状态无效")
    model = (ROOT / startup["model"]).resolve()
    if not model.is_relative_to((ROOT / "模型").resolve()):
        raise RuntimeError("启动模型不在本机模型目录")
    for filename, key in (("model.safetensors", "checkpoint_sha256"), ("config.json", "config_sha256")):
        if file_sha256(model / filename) != startup[key]:
            raise RuntimeError("启动验证模型已变化")
    expected_scripts = {"预热模型与检查时效.py", "实机抓取测试.py", "异步动作控制.py", "模型启动离线验证.py",
                        "检查与释放电机.py", "实机测试监视窗口.py"}
    if set(startup["software_sha256"]) != expected_scripts:
        raise RuntimeError("启动验证缺少软件版本")
    for filename, digest in startup["software_sha256"].items():
        if file_sha256(ROOT / "脚本/程序模块" / filename) != digest:
            raise RuntimeError("启动软件已变化，请重新验证")
    samples = startup["warmup"]["samples"]
    if len(samples) < 2 or any(item["ready_latency_s"] >= startup["warmup"]["ready_budget_s"] for item in samples[1:]):
        raise RuntimeError("启动时效未通过")
    startup["model"] = model.relative_to(ROOT).as_posix()
    (output / "模型启动离线验证.json").write_text(json.dumps(startup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    warm_cancel_text = (f"{startup['warmup_cancel_response_s']*1000:.1f} 毫秒"
                        if startup['warmup_cancel_response_s'] > 0 else "当前时钟分辨率内")
    warmup_section = f"""## 模型启动预热与确认检查

实机入口已在配置扭矩前增加两轮模型预热，丢弃全部预热输出、重置模型队列，再等待操作者按回车。后续轮次结果准备须低于 **{startup['warmup']['ready_budget_s']:.2f} 秒**，总预热限时 30 秒；失败、超时或取消均不允许开始控制。确认后重新读取画面和关节状态，配置扭矩前再次检查画面时效和停止事件。原有功率、20 Hz、每步限幅 6 与最长 60 秒保留。

在 {startup['created_at']} 用三段训练基线模型和录制画面验证：首轮 GPU 推理 **{samples[0]['inference_s']:.3f} 秒**，第二轮 **{samples[1]['inference_s']:.3f} 秒**，第二轮含调度的准备耗时 **{samples[1]['ready_latency_s']:.3f} 秒**。随后从新快照重新推理、模拟关节运行 4 秒，发送 {startup['simulated_control']['sent_commands']} 次模拟目标，活动频率 {startup['simulated_control']['active_control_hz']:.2f} Hz；GPU 预热中取消，调用在{warm_cancel_text}返回。取消返回后才等待后台 GPU 退出，测量不代表实际扭矩释放耗时。

这是一次本机启动验证，与前文整段留出模型的冷启动耗时使用不同检查点，不能直接比较速度。未连接实时相机或真实机械臂，预热也不保证后续负载下的时效；异步运行仍保留超时及过期停止。完整软件检查共 {checks['tests_run']} 项通过，包含超预算/无效输出/超时、预热取消以及确认后停止和过期画面时不配置扭矩、不写控制目标。[离线报告](模型启动离线验证.json)、[启动说明](../模型预热与启动检查说明.md)。

"""
body = f"""# SmolVLA 项目进展

更新日期：{datetime.now().date().isoformat()}。当前主线：SO-101 主从双臂、桌面单相机、红色方块抓取并放置到目标区域。

{latest_hardware_notice}## 已完成

- 本地 GPU 环境、基础模型和视觉语言处理器下载；固定模型版本见 `脚本/程序模块/下载预训练模型.py`。
- 主从双臂全部六关节同步，用户已确认同步测试成功。
- 演示录制、停止、保存与逐帧动作备份；后台窗口模拟录制保存通过。
- 当前有效演示 3 段，共 1,114 帧，20 FPS；视频、动作和状态完整对应。
- SmolVLA 微调 3,000 步、batch=4，耗时 {training['seconds']/60:.2f} 分钟。近 50 步平均损失从 {training['first_50_avg_loss']:.5f} 降至 {training['last_50_avg_loss']:.5f}。
- 目录合并后微调检查点可离线重新加载，输出形状 `[1, 50, 6]`，未发出真实机械臂命令。
- 三段演示各抽样三处，共 9 个动作块；数值和形状全部有效。
- 实机入口新增窗口停止按钮、逐步相机时效检查、按实时位置限幅、当前姿态运行范围与关节顺序检查，以及准备失败后的资源释放。

## 离线动作评估

这是原有三段、3000 步模型在训练演示上的拟合检查。三段全部用于该基线训练，不能从中给这个模型划出独立验证集，未测量实机抓取成功率。新模型的整段留出实验另列下文。`status=passed` 表示模型正常输出，不表示任务已通过实机验收。

| 关节 | 动作块平均绝对误差 |
|---|---:|
{rows}

误差单位为 LeRobot 归一化位置单位，身体关节 RANGE_M100_100、夹爪 RANGE_0_100；不是角度。按训练动作标准差缩放后的平均绝对误差为 **{evaluation['normalized_mae']:.4f}**。

排除首次预热后，动作块推理中位耗时 **{evaluation['steady_inference_median_s']:.3f} 秒**，P95 为 {evaluation['steady_inference_p95_s']:.3f} 秒。改动前的串行模式每次推理后按 20 Hz 执行 20 步，平均动作频率估算为 **{evaluation['blocking_loop_effective_hz_estimate']:.1f} Hz**，未计串口等开销。新的异步模式测量见下文。

完整指标：[离线动作评估](离线动作评估.json)、[微调训练摘要](微调训练摘要.json)。

{async_section}{experiment_section}{recording_section}{panel_section}{comparison_section}{window_section}{warmup_section}## 下一步

1. 最新日志存在停扭矩异常时，先托住从臂并断电，确认处理状态后再做只读检查，不自动继续动作。用 `实机结果记录.bat` 按本人现场观察标记已完成日志的真实任务结果，异常和未确认记录保留；新的测试按[实机评测流程](../实机评测流程.md)进行。
2. 用采集与训练面板增加不同方块位置的演示；保留独立批次做验证，不全部放进训练集。
3. 结合已记录的实机节拍和人工任务结果决定是否调整动作块长度、增加数据或优化推理精度；短测正常不替代更长时间验证，也没有测量故障发生时的实际停扭矩耗时。

双路相机的同时取帧问题暂缓，当前训练与测试只用桌面相机。实机入口需用户按回车才启用扭矩；异步模式在控制节拍处理窗口停止请求，串行对照仍等待当前推理结束。

## GitHub 保存范围

保存中文脚本、上游代码、历史交接说明、环境依赖和进展摘要。本机虚拟环境、缓存、模型权重、原始采集视频、标定及原始运行日志不随此次代码提交上传。训练数据和论文仍属于同一个本地毕业设计项目。
"""
collection_report = output / "采集流程完善报告.json"
if collection_report.exists():
    collection = json.loads(collection_report.read_text(encoding="utf-8"))
    if collection["status"] != "passed" or collection["robot_commands_sent"] != 0:
        raise RuntimeError("采集流程完善报告尚未通过")
    latest = """## 最新采集流程与文档更新（2026-10-07）

采集指引与录制、归档、检查、自检统一在 §脚本/数据采集/§。新尝试独立编号，保存/失败/放弃均归档原始帧及动作；保存失败封存批次；独立验证批次禁止训练。数据保护与模拟双臂真实视频编码验证通过，本轮没有启动真实设备，也未新增真实演示。

最新开发维护见 [开发文档](../开发文档.md)，结构与数据链路见 [项目说明](../项目说明.md)，现场步骤和 3 段试录、30 段训练、10 段独立验证计划见 [数据采集指引](../../脚本/数据采集/数据采集指引.md)。本轮在 §smolvla-data-collection§ 新分支维护。以下保留已有训练、实机和离线实验的证据与指标。

""".replace("§", chr(96))
    body = body.replace(latest_hardware_notice, latest + latest_hardware_notice, 1)
(output / "当前进展.md").write_text(body, encoding="utf-8")
print("已生成可提交的项目进展和离线验证摘要。")
