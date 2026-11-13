"""在串口所属线程读取故障状态，逐电机尝试停扭矩并关闭连接；不清除保护标志。"""
import time


class MotorHealthGuard:
    def __init__(self, bus, interval=1, clock=time.monotonic):
        if interval <= 0:
            raise ValueError("电机检查间隔必须为正")
        self.bus, self.interval, self.clock = bus, interval, clock
        self.next_check = float("-inf")
        self.last_statuses = {}

    def check(self, force=False):
        now = self.clock()
        if not force and now < self.next_check:
            return False
        statuses = {}
        for motor in self.bus.motors:
            try:
                status = self.bus.read("Status", motor, normalize=False, num_retry=0)
            except Exception as exc:
                raise RuntimeError(f"电机 {motor} 故障状态读取失败，停止控制：{exc}") from exc
            if not isinstance(status, int) or status != 0:
                raise RuntimeError(f"电机 {motor} 报告故障状态 {status!r}，停止控制")
            statuses[motor] = status
        self.last_statuses = statuses
        self.next_check = self.clock() + self.interval
        return True


def release_bus(bus):
    """每台均尝试关闭并读回扭矩；任何失败保留为错误，仍关闭串口。"""
    errors = []
    if not bus.is_connected:
        return errors
    for motor in bus.motors:
        try:
            bus.write("Torque_Enable", motor, 0, normalize=False, num_retry=1)
        except Exception as exc:
            errors.append(f"{motor} 停扭矩写入未确认：{exc}")
        try:
            enabled = bus.read("Torque_Enable", motor, normalize=False, num_retry=0)
            if enabled != 0:
                errors.append(f"{motor} 扭矩未关闭，读回 {enabled!r}")
        except Exception as exc:
            errors.append(f"{motor} 停扭矩读回未确认：{exc}")
    try:
        # 默认 disconnect 再次 disable_torque，故障时会阻断 closePort。
        # 已逐个尝试停扭矩，关闭端口不再重复写入，也不宣称故障电机已断电。
        bus.disconnect(disable_torque=False)
    except Exception as exc:
        errors.append(f"关闭串口失败：{exc}")
    return errors
