"""读取完整运行日志、人工标记结果、保存修订历史；不导入机械臂或模型接口。"""
import csv
import hashlib
import json
import math
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

OUTCOMES = ("成功抓取并放置", "抓取失败", "放置失败", "人工中断", "程序异常", "开始前取消")
SUCCESS = OUTCOMES[0]


def read_run(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    if not raw:
        raise ValueError("日志为空，尚不能确认运行状态")
    try:
        events = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    except (ValueError, UnicodeError) as exc:
        raise ValueError("日志格式损坏或尚未写完") from exc
    if not events or any(not isinstance(event, dict) for event in events):
        raise ValueError("日志事件格式无效")
    configs = [event for event in events if event.get("type") in ("config", "配置")]
    terminals = [event for event in events if event.get("type") in ("end", "cancelled_before_start", "preparation_end")]
    if len(configs) != 1 or events[0] is not configs[0] or len(terminals) != 1 or events[-1] is not terminals[0]:
        raise ValueError("日志没有唯一完整结束记录，不能标记仍在运行的测试")
    config, terminal = configs[0], terminals[0]
    commands = terminal.get("sent_commands", 0)
    if type(commands) is not int or commands < 0:
        raise ValueError("日志中的动作次数无效")
    starts = [event for event in events if event.get("type") == "start"]
    if len(starts) > 1:
        raise ValueError("单次日志不能包含多次测试开始")
    started = bool(starts)
    if commands and not started:
        raise ValueError("日志有动作次数但缺少开始事件")
    identities = [event for event in events if event.get("type") == "model_identity"]
    if len(identities) > 1:
        raise ValueError("日志中模型身份不唯一")
    identity = identities[-1] if identities else {}
    if identities and (not isinstance(identity.get("checkpoint_sha256"), str) or
                       len(identity["checkpoint_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in identity["checkpoint_sha256"])):
        raise ValueError("模型内容指纹无效")
    errors = [event.get("error", "程序异常") for event in events if event.get("type") == "error"]
    errors.extend(terminal.get("cleanup_errors", []))
    if terminal.get("stop_reason") == "error" and not errors:
        errors.append("控制以程序异常结束")
    if terminal.get("error"):
        errors.append(terminal["error"])
    if terminal["type"] == "cancelled_before_start" and (started or commands):
        raise ValueError("开始前取消日志不能包含已开始或动作次数")
    control_events = [event for event in events if event.get("type") == "async_end"]
    control = control_events[-1] if control_events else {}
    if len(control_events) > 1 or (control_events and control.get("sent_commands") != commands):
        raise ValueError("控制与结束日志的动作次数不一致")
    active_hz = control.get("active_control_hz")
    if active_hz is not None and (not isinstance(active_hz, (int, float)) or not math.isfinite(active_hz) or active_hz < 0):
        raise ValueError("日志中的控制频率无效")
    return {"path": str(path), "filename": path.name, "log_sha256": hashlib.sha256(raw).hexdigest(),
            "run_kind": config.get("run_kind", "legacy_unverified"), "task": config.get("task", "未记录任务"),
            "scene": config.get("scene", "演示起始位置"), "mode": config.get("mode", "未记录模式"),
            "checkpoint_sha256": identity.get("checkpoint_sha256"), "model_path": identity.get("model_path"),
            "started": started, "sent_commands": commands, "seconds": terminal.get("seconds"),
            "stop_reason": terminal.get("stop_reason", "未记录停止原因"), "errors": errors,
            "cancelled_before_start": terminal["type"] == "cancelled_before_start",
            "active_control_hz": active_hz, "inferences": terminal.get("inferences"),
            "missed_ticks": control.get("missed_ticks"),
            "control_settings": {key: config.get(key) for key in ("control_hz", "max_step_delta", "range_margin", "chunk_execute", "test_seconds")}}


@contextmanager
def write_lock(path):
    """Windows 字节锁随进程退出释放；锁文件本身可保留，不依赖删除解除锁。"""
    import msvcrt
    with Path(path).open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic() + 3
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise ValueError("另一个窗口正在保存，请稍后重试")
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


class EvaluationStore:
    def __init__(self, log_directory):
        self.logs = Path(log_directory).resolve()
        self.directory = self.logs / "实机评测"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "人工结果标记.json"

    def _load(self):
        if not self.path.exists():
            return {"schema_version": 1, "records": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or not isinstance(data.get("records"), dict):
            raise ValueError("结果标记文件格式无效，请保留原文件检查")
        for key, history in data["records"].items():
            if not isinstance(history, list) or not history:
                raise ValueError("结果标记修订历史无效")
            for index, revision in enumerate(history, 1):
                if (not isinstance(revision, dict) or revision.get("revision") != index or
                    revision.get("observed_real") is not True or revision.get("log_sha256") != key or
                    revision.get("outcome") not in OUTCOMES or
                    any(not isinstance(revision.get(field), str) for field in ("scene", "reviewer", "notes", "log_filename", "created_at"))):
                    raise ValueError("结果标记内容无效，请保留原文件检查")
        return data

    def _checked_run(self, path):
        path = Path(path).resolve()
        if path.parent != self.logs or not path.name.startswith("实机抓取测试_") or path.suffix != ".jsonl":
            raise ValueError("只能关联本项目运行记录中的实机抓取日志")
        return read_run(path)

    def record(self, path, outcome, scene, reviewer, notes="", observed_real=False):
        if outcome not in OUTCOMES or not scene.strip() or not reviewer.strip():
            raise ValueError("请选择结果并填写场景和记录者")
        if observed_real is not True:
            raise ValueError("请确认本人观察了这次真实硬件测试")
        with write_lock(self.directory / "结果标记写入锁"):
            run = self._checked_run(path)
            if run["run_kind"] not in ("real_hardware", "legacy_unverified"):
                raise ValueError("模拟或离线日志不能登记为实机结果")
            if outcome == "开始前取消" and run["started"]:
                raise ValueError("测试已经开始，不能标记为开始前取消")
            if outcome in (SUCCESS, "抓取失败", "放置失败") and (not run["started"] or not run["sent_commands"]):
                raise ValueError("没有开始或没有发送动作，不能标记抓取/放置结果")
            if outcome == SUCCESS and run["errors"]:
                raise ValueError("日志包含程序或清理异常，请先按程序异常记录并在备注说明实际物体结果")
            data = self._load()
            previous = data["records"].get(run["log_sha256"], [])
            revision = {"revision": len(previous) + 1, "created_at": datetime.now().isoformat(timespec="microseconds"),
                        "outcome": outcome, "scene": scene.strip(), "reviewer": reviewer.strip(), "notes": notes.strip(),
                        "observed_real": True, "log_sha256": run["log_sha256"], "log_filename": run["filename"]}
            data["records"][run["log_sha256"]] = previous + [revision]
            temp = self.directory / f"结果标记暂存_{uuid.uuid4().hex}.json"
            try:
                temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                os.replace(temp, self.path)
            finally:
                if temp.exists():
                    temp.unlink()
            return revision

    def scan(self):
        data = self._load()
        runs, excluded = [], []
        seen = set()
        for path in sorted(self.logs.glob("实机抓取测试_*.jsonl")):
            try:
                run = self._checked_run(path)
                if run["log_sha256"] in seen:
                    excluded.append({"filename": path.name, "reason": "同一内容的日志副本，不重复计数"})
                    continue
                seen.add(run["log_sha256"])
                revisions = data["records"].get(run["log_sha256"], [])
                run["annotation"] = revisions[-1] if revisions else None
                run["revision_count"] = len(revisions)
                runs.append(run)
            except (OSError, ValueError, KeyError) as exc:
                excluded.append({"filename": path.name, "reason": str(exc)})
        orphaned = [history[-1]["log_filename"] for key, history in data["records"].items() if key not in seen and history]
        return runs, excluded, orphaned

    def summary(self):
        runs, excluded, orphaned = self.scan()
        groups = {}
        simulation = 0
        unverified = 0
        for run in runs:
            label = run["annotation"]
            if run["run_kind"] not in ("real_hardware", "legacy_unverified"):
                simulation += 1
                continue
            if run["run_kind"] == "legacy_unverified" and not label:
                unverified += 1
                continue
            scene = label["scene"] if label else run["scene"]
            # 未知模型身份的旧日志逐次列出，不能与新模型合并比较。
            model = run["checkpoint_sha256"] or "未知检查点_" + run["log_sha256"]
            key = (scene, model, run["mode"], run["task"])
            group = groups.setdefault(key, {"scene": scene, "checkpoint": model, "mode": run["mode"], "task": run["task"],
                                           "attempts": 0, "reviewed_attempts": 0, "pending_attempts": 0,
                                           "successes": 0, "non_attempts": 0, "outcomes": dict.fromkeys(OUTCOMES, 0)})
            if label:
                group["outcomes"][label["outcome"]] += 1
            if run["started"] and run["sent_commands"] > 0:
                group["attempts"] += 1
                if label:
                    group["reviewed_attempts"] += 1
                    group["successes"] += int(label["outcome"] == SUCCESS)
                else:
                    group["pending_attempts"] += 1
            else:
                group["non_attempts"] += 1
        for group in groups.values():
            group["success_rate"] = (group["successes"] / group["attempts"]
                                     if group["attempts"] and not group["pending_attempts"] else None)
        return {"created_at": datetime.now().isoformat(timespec="seconds"), "scope": "human_observed_hardware_trials",
                "groups": list(groups.values()), "unverified_legacy_runs": unverified,
                "excluded_simulation_runs": simulation, "unreadable_or_duplicate_logs": excluded,
                "annotations_without_matching_log": orphaned,
                "definition": "仅真实控制开始且发送动作的尝试计入分母；人工中断与程序异常不从分母移除。待标记时不显示成功率。零动作和开始前取消单列。"}

    def export(self):
        report = self.summary()
        stem = self.directory / f"评测汇总_{datetime.now():%Y%m%d_%H%M%S_%f}"
        json_path, csv_path = stem.with_suffix(".json"), stem.with_suffix(".csv")
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["场景", "检查点", "模式", "任务", "尝试数", "已标记", "待标记", "成功数", "成功率", "非尝试数"])
            for group in report["groups"]:
                writer.writerow([group[key] for key in ("scene", "checkpoint", "mode", "task", "attempts", "reviewed_attempts",
                                                        "pending_attempts", "successes", "success_rate", "non_attempts")])
        return json_path, csv_path
