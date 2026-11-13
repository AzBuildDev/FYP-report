"""闭合受阻时限制位置目标；仅凭位置判为疑似接触，不认证夹持成功或夹持力。"""
import math


class GripperContactGuard:
    def __init__(self, lower, upper, tracking_limit, *, closing_offset=1., detect_seconds=.2,
                 release_margin=.5, on_event=None):
        values = (lower, upper, tracking_limit, closing_offset, detect_seconds, release_margin)
        if (not all(math.isfinite(value) for value in values) or lower >= upper
                or not 0 < closing_offset < tracking_limit or detect_seconds <= 0
                or release_margin <= 0):
            raise ValueError("夹爪接触保护参数无效")
        self.lower, self.upper, self.tracking_limit = lower, upper, tracking_limit
        self.closing_offset, self.detect_seconds = closing_offset, detect_seconds
        self.release_margin = release_margin
        self.on_event = on_event or (lambda event: None)
        self.started = self.contact_position = self.hold_floor = self.now = None

    @property
    def active(self):
        return self.started is not None

    def settings(self):
        return {"detection": "closing_position_stall", "closing_offset": self.closing_offset,
                "detect_seconds": self.detect_seconds, "max_hold_seconds": None,
                "release_margin": self.release_margin, "force_measured": False}

    def snapshot(self):
        return {"mode": "contact_hold" if self.active else "position",
                "contact_position": self.contact_position, "hold_floor": self.hold_floor,
                "held_seconds": max(0., self.now - self.started) if self.active else 0.,
                "max_hold_seconds": None, "force_measured": False}

    def observe(self, current, previous, requested, now, blocked_since):
        self.now = now
        if self.active:
            if abs(current - self.contact_position) > self.tracking_limit:
                raise RuntimeError("夹爪接触后位置变化超限，停止控制并检查物体或机械阻挡")
            return False
        # 只有继续闭合的目标受阻才进入；张开受阻仍由普通卡滞保护停止。
        if (blocked_since is not None and now - blocked_since >= self.detect_seconds
                and previous < current - .5 and requested < current - .5):
            self.started, self.contact_position = now, current
            # 固定在首次接触点附近，后续不随反馈向闭合方向追赶或继续加压。
            self.hold_floor = max(self.lower, min(self.upper, current - self.closing_offset))
            self.on_event({"type": "gripper_contact", "event": "limited_hold",
                           "observed_position": current, "previous_target": previous,
                           "model_target": requested, **self.snapshot()})
            return True
        return False

    def target(self, requested):
        if not self.active:
            return requested
        if requested >= self.contact_position + self.release_margin:
            self.on_event({"type": "gripper_contact", "event": "model_opening",
                           "model_target": requested, **self.snapshot()})
            self.started = self.contact_position = self.hold_floor = None
            return requested
        return max(requested, self.hold_floor)
