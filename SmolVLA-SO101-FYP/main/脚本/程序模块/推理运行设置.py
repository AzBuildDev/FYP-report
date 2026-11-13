"""区分短行程检查与实际任务推理，供窗口、控制入口和离线检查共用。"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ControlSettings:
    seconds: float | None
    chunk_execute: int
    max_total_travel: float | None
    profile: str = "limited_test"
    target_strategy: str = "feedback"
    max_step_delta: float = 1.0

    def __post_init__(self):
        if self.target_strategy not in ("feedback", "trajectory") or not math.isfinite(self.max_step_delta) or self.max_step_delta <= 0:
            raise ValueError("目标策略或相邻指令限速无效")
        if self.seconds is not None and (not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError("推理时长必须为正数；持续运行使用不设时限模式")
        if type(self.chunk_execute) is not int or not 1 <= self.chunk_execute <= 50:
            raise ValueError("动作块执行长度必须在 1～50 之间")
        if self.max_total_travel is not None and (not math.isfinite(self.max_total_travel) or self.max_total_travel <= 0):
            raise ValueError("总行程限制必须为正数")


def task_inference_settings(seconds=30):
    # 20 Hz 下最多使用每块 1 秒动作，覆盖本机约0.62秒的推理周期。
    # 新结果到达即替换，过期步骤仍由调度器跳过；不提高单步目标变化上限。
    seconds = float(seconds)
    return ControlSettings(None if seconds == 0 else seconds, 20, None, "task_inference", "trajectory", 0.5)


def trajectory_trial_settings():
    # 独立首轮：最多3秒，相对起点最多8单位，每步指令最多0.5。
    return ControlSettings(3., 10, 8., "trajectory_trial", "trajectory", 0.5)
