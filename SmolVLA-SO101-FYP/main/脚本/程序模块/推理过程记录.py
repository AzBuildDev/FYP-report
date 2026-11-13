"""生成实时关节状态与可分享日志；不读取或控制硬件。"""
import json
import time
import threading
from pathlib import Path

import numpy as np

PREFIX = "实时状态JSON："
JOINT_LABELS = ("底座", "肩关节", "肘关节", "腕俯仰", "腕旋转", "夹爪")


def format_progress(event):
    def vector(key):
        return "[" + ", ".join(f"{v:.2f}" for v in event[key]) + "]"
    result = (f"运行 {event['t']:.1f}s | 推理 {event['inferences']} 次 / {event['inference_s']:.3f}s | "
            f"动作 {event['sent_commands']} 步 | 限幅 {event['limited_joints']} 轴\n"
            f"  实际反馈 {vector('observed_state')}\n"
            f"  模型预测 {vector('model_target')}\n"
            f"  发送目标 {vector('sent_target')}\n")
    contact = event.get("gripper_control")
    if contact and contact["mode"] == "contact_hold":
        limit = contact.get("max_hold_seconds")
        duration = "不限时" if limit is None else f"{limit:g}s"
        result += (f"  夹爪：疑似接触，限幅保持 {contact['held_seconds']:.1f}s / "
                   f"{duration}；闭合目标下限 {contact['hold_floor']:.2f}\n")
    return result


class ProgressReporter:
    def __init__(self, log, names, *, interval=0.5, emit=print):
        self.log, self.names, self.interval, self.emit = log, names, interval, emit
        self.last_emitted = float("-inf")
        self.count = 0
        self.limited_counts = np.zeros(6, dtype=int)

    def action(self, event, *, inference_s=0., inferences=0, fault_status=None):
        event = dict(event, type="action")
        observed, model, sent = [np.asarray(event[key], dtype=float)
                                  for key in ("observed_state", "model_target", "sent_target")]
        limited = np.abs(model - sent) > 1e-4
        event.update(joint_names=self.names, limited=limited.tolist(),
                     command_error=(sent - observed).tolist())
        self.log.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.count += 1
        self.limited_counts += limited
        if event["t"] - self.last_emitted < self.interval:
            return
        self.last_emitted = event["t"]
        progress = dict(event, type="telemetry", sent_commands=self.count,
                        inferences=inferences, inference_s=inference_s,
                        limited_joints=int(limited.sum()),
                        fault_status=fault_status, units="归一化开度/位置，非角度")
        self.log.write(json.dumps(progress, ensure_ascii=False) + "\n")
        self.emit(PREFIX + json.dumps(progress, ensure_ascii=False), flush=True)

    def summary(self):
        return dict(action_records=self.count, limited_fraction=(self.limited_counts / max(1, self.count)).tolist())


class InputArchive:
    """在模型线程保存其实际输入，每秒最多一组；不在串口控制线程写图。"""
    def __init__(self, directory, predict, *, interval=1.):
        self.directory, self.predict, self.interval = Path(directory), predict, interval
        self.last_saved = float("-inf")
        self.counter = 0
        self.phase = "warmup"
        self.errors = []
        self.lock = threading.Lock()

    def __call__(self, state, images):
        record = None
        if time.monotonic() - self.last_saved >= self.interval:
            with self.lock:
                self.last_saved = time.monotonic()
                self.counter += 1
                index = self.counter
            try:
                import cv2
                self.directory.mkdir(parents=True, exist_ok=True)
                record = dict(sample=index, phase=self.phase, model_input_time_monotonic=time.monotonic(),
                              state=np.asarray(state).tolist(), images={})
                for key, frame in images.items():
                    filename = f"{index:04d}_{key.split('.')[-1]}.jpg"
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if not ok:
                        raise OSError("输入画面保存失败")
                    (self.directory / filename).write_bytes(encoded.tobytes())
                    record["images"][key] = filename
            except (OSError, ValueError) as exc:
                self.errors.append(str(exc))
        actions = self.predict(state, images)
        if record is not None:
            # 保留同一份实际推理的完整轨迹，才能判断未执行的后段是否包含闭爪动作。
            record["prediction_return_time_monotonic"] = time.monotonic()
            recorded_actions = np.asarray(actions)
            record["first_action"] = recorded_actions[0].tolist()
            record["actions"] = recorded_actions.tolist()
            try:
                with (self.directory / "模型输入索引.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                self.errors.append(str(exc))
        return actions


def export_run_report(path, window_log=None):
    """合并窗口输出与逐步 JSONL，生成便于复制给排查人员的文本。"""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    sections = [f"实机运行报告\n原始逐步日志：{path}\n"]
    if window_log and Path(window_log).is_file():
        sections.append("\n=== 窗口输出 ===\n" + Path(window_log).read_text(encoding="utf-8", errors="replace"))
    sections.append("\n=== 逐步模型预测、限幅、目标与实测状态（JSONL）===\n")
    sections.append(raw)
    return "".join(sections)
