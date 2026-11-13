"""运行目标立即切换；每次有效请求使旧动作和旧推理代次失效。"""
import json
from pathlib import Path
from 红蓝抓取任务 import TASKS


class TaskSwitch:
    def __init__(self, task, directory=None, emit=None):
        self.current = next(key for key, value in TASKS.items() if value == task)
        self.pending = None
        self.generation = 0
        self.last_request = None
        self.request_details = {}
        self.directory = Path(directory) if directory else None
        self.emit = emit or (lambda event: None)

    @property
    def task(self):
        return TASKS[self.current]

    def update(self, state, now, guard=None):
        if self.directory is None:
            return False
        path = self.directory / "目标请求.json"
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False
        except (ValueError, OSError):
            return False  # 窗口使用原子替换；无效请求不改变任务。
        if request.get("id") == self.last_request:
            return False
        self.last_request = request.get("id")
        self.request_details = {key: request.get(key) for key in ("id", "source", "transcript")}
        target = request.get("target")
        if target not in TASKS:
            return False
        self.current = target
        self.generation += 1
        self.report("已切换，旧动作已清空，等待新目标推理")
        return True

    def report(self, message):
        self.emit({"type": "task_switch", "current": self.current,
                   "pending": self.pending, "generation": self.generation, "message": message,
                   "request": self.request_details})
