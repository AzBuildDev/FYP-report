"""检查闭环相机输入并按实时关节位置限制目标；不访问硬件。"""
import numpy as np
from 夹爪接触保护 import GripperContactGuard


class SlewTrackingGuard:
    """相邻指令限速，独立限制反馈误差；关节滞后时不无限累积目标。"""
    def __init__(self, initial, lo, hi, max_delta, tracking_limits, *, names=None, stall_timeout=0.6,
                 gripper_index=None, on_gripper_event=None):
        self.previous, self.lo, self.hi, self.tracking = [np.asarray(v, dtype=np.float32).copy()
                                                        for v in (initial, lo, hi, tracking_limits)]
        if any(v.shape != (6,) or not np.isfinite(v).all() for v in (self.previous, self.lo, self.hi, self.tracking)):
            raise ValueError("轨迹保护参数必须是有限六维数值")
        if (not np.isfinite(max_delta) or max_delta <= 0 or not np.isfinite(stall_timeout) or stall_timeout <= 0
                or np.any(self.tracking <= max_delta) or np.any(self.lo > self.hi)
                or np.any(self.previous < self.lo) or np.any(self.previous > self.hi)):
            raise ValueError("轨迹限速、跟随上限或初始范围无效")
        self.max_delta, self.stall_timeout = max_delta, stall_timeout
        self.names = names or [str(i) for i in range(6)]
        self.stalled_since = [None] * 6
        self.anchor = self.previous.copy()
        self.last_requested = self.previous.copy()
        self.gripper_index, self.gripper_contact = gripper_index, None
        if gripper_index is not None:
            if (type(gripper_index) is not int or not 0 <= gripper_index < 6
                    or self.names[gripper_index] != "gripper.pos"):
                raise ValueError("接触保护只能用于明确命名的夹爪关节")
            self.gripper_contact = GripperContactGuard(
                float(self.lo[gripper_index]), float(self.hi[gripper_index]),
                float(self.tracking[gripper_index]), closing_offset=min(1., float(self.tracking[gripper_index]) / 2),
                detect_seconds=min(.2, stall_timeout / 2), on_event=on_gripper_event)

    @property
    def gripper_status(self):
        return self.gripper_contact.snapshot() if self.gripper_contact is not None else None

    def observe(self, current, now):
        current = np.asarray(current, dtype=np.float32)
        if current.shape != (6,) or not np.isfinite(current).all() or not np.isfinite(now):
            raise RuntimeError("轨迹跟随反馈无效")
        error = np.abs(self.previous - current)
        if np.any(error > self.tracking + 0.05):
            index = int(np.argmax(error - self.tracking))
            raise RuntimeError(f"关节 {self.names[index]} 跟随误差超限：{error[index]:.2f} > {self.tracking[index]:.2f}，停止控制")
        for i in range(6):
            if error[i] < max(1.5, 0.8 * self.tracking[i]):
                self.stalled_since[i] = None
            elif self.stalled_since[i] is None or abs(float(current[i] - self.anchor[i])) >= 0.25:
                self.stalled_since[i], self.anchor[i] = now, current[i]
            if self.gripper_contact is not None and i == self.gripper_index:
                entered = self.gripper_contact.observe(float(current[i]), float(self.previous[i]),
                                                       float(self.last_requested[i]), now, self.stalled_since[i])
                if entered:
                    # 仅在转换时给限速卸载目标一个窗口；后续跟随误差/卡滞检查不豁免。
                    self.stalled_since[i] = now
            if self.stalled_since[i] is not None and now - self.stalled_since[i] >= self.stall_timeout:
                raise RuntimeError(f"关节 {self.names[i]} 跟随停滞：误差 {error[i]:.2f}，{self.stall_timeout:g} 秒无有效进展，停止控制")

    def target(self, model_target, current):
        model_target, current = [np.asarray(v, dtype=np.float32) for v in (model_target, current)]
        if any(v.shape != (6,) or not np.isfinite(v).all() for v in (model_target, current)):
            raise RuntimeError("轨迹目标或反馈无效")
        self.last_requested = model_target.copy()
        if self.gripper_contact is not None:
            model_target = model_target.copy()
            was_active = self.gripper_contact.active
            model_target[self.gripper_index] = self.gripper_contact.target(float(model_target[self.gripper_index]))
            if was_active and not self.gripper_contact.active:
                self.stalled_since[self.gripper_index] = None
                self.anchor[self.gripper_index] = current[self.gripper_index]
        lower = np.maximum.reduce([self.lo, self.previous - self.max_delta, current - self.tracking])
        upper = np.minimum.reduce([self.hi, self.previous + self.max_delta, current + self.tracking])
        if np.any(lower > upper):
            raise RuntimeError("相邻指令限速与跟随范围无交集，停止控制")
        return np.clip(model_target, lower, upper)

    def commit(self, sent):
        sent = np.asarray(sent, dtype=np.float32)
        if sent.shape != (6,) or not np.isfinite(sent).all() or np.any(np.abs(sent - self.previous) > self.max_delta + 1e-4):
            raise RuntimeError("实际发送指令违反相邻指令限速，停止控制")
        self.previous = sent.copy()


def validate_frame(stamp, image, now, max_age=0.5):
    if isinstance(image, dict):
        if not image:
            raise RuntimeError("没有相机画面，停止控制。")
        timestamps = getattr(image, "timestamps", {})
        diagnostics = getattr(image, "diagnostics", {})
        failures, messages = {}, []
        for name, frame in image.items():
            camera_stamp = timestamps.get(name, stamp)
            age = now - camera_stamp
            reasons = []
            if frame is None:
                reasons.append("未收到画面")
            if not np.isfinite(camera_stamp):
                reasons.append("到帧时间戳无效")
            elif camera_stamp > now:
                reasons.append(f"到帧时间戳在检查时间之后 {camera_stamp-now:.3f} 秒")
            elif age > max_age:
                reasons.append(f"画面过期 {age:.3f} 秒，允许上限 {max_age:.3f} 秒")
            try:
                if frame is not None:
                    _validate_image(frame)
            except RuntimeError as exc:
                reasons.append(str(exc))
            if reasons:
                state = diagnostics.get(name, {})
                failures[name] = dict(reasons=reasons, frame_monotonic=float(camera_stamp) if np.isfinite(camera_stamp) else None,
                                      host_frame_age_s=float(age) if np.isfinite(age) else None,
                                      max_age_s=max_age, capture=state)
                label = state.get("label", name)
                messages.append(f"{label}（{name}）：{'；'.join(reasons)}")
        # 保留旧调用者的合并时间戳校验；逐路快照使用各自时间戳。
        if not failures and (not np.isfinite(stamp) or stamp > now or now-stamp > max_age):
            raise RuntimeError("相机合并时间戳无效或过期，停止控制。")
        if failures:
            error = RuntimeError("相机输入异常，停止控制。" + " | ".join(messages))
            error.camera_details = dict(failed_cameras=failures, cameras=diagnostics,
                                        checked_monotonic=now, max_age_s=max_age)
            raise error
        return image
    if image is None or not np.isfinite(stamp) or now - stamp > max_age or stamp > now:
        raise RuntimeError("相机画面中断，停止控制。")
    _validate_image(image)
    return image


def _validate_image(image):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise RuntimeError("相机画面格式无效，停止控制。")
    if image.mean() < 3 and image.std() < 1:
        raise RuntimeError("相机黑帧，停止控制。")
    return image


def bounds_including_start(current, lo, hi, margin=8):
    """接纳当前姿态，将运行范围扩至当前姿态周围；裁剪到本机 SO-101 标定单位行程。"""
    current, lo, hi = [np.asarray(value, dtype=np.float32) for value in (current, lo, hi)]
    if any(value.shape != (6,) or not np.isfinite(value).all() for value in (current, lo, hi)):
        raise RuntimeError("关节状态或运行范围包含无效数值。")
    if not np.isfinite(margin) or margin < 0 or np.any(lo > hi):
        raise RuntimeError("运行范围余量无效。")
    calibration_lo = np.array([-100, -100, -100, -100, -100, 0], dtype=np.float32)
    calibration_hi = np.full(6, 100, dtype=np.float32)
    return (np.maximum(calibration_lo, np.minimum(lo, current - margin)),
            np.minimum(calibration_hi, np.maximum(hi, current + margin)))


def bounded_target(target, current, lo, hi, max_delta):
    values = [np.asarray(x, dtype=np.float32) for x in (target, current, lo, hi)]
    if any(x.shape != (6,) or not np.isfinite(x).all() for x in values):
        raise RuntimeError("关节目标、状态或限制包含无效数值。")
    target, current, lo, hi = values
    if not np.isfinite(max_delta) or max_delta <= 0 or np.any(lo > hi):
        raise RuntimeError("控制限制无效。")
    lower, upper = np.maximum(lo, current - max_delta), np.minimum(hi, current + max_delta)
    if np.any(lower > upper):
        raise RuntimeError("当前关节位置与运行边界没有可达交集。")
    return np.clip(target, lower, upper)


def restrict_gripper_to_demo(current, lo, hi, gripper_index, demo_lower, demo_upper):
    """夹爪不随任意起点扩展到全开或全闭；超出示范范围时拒绝启用。"""
    if not (np.isfinite(demo_lower) and np.isfinite(demo_upper) and 0 <= demo_lower < demo_upper <= 100):
        raise RuntimeError("演示夹爪范围无效，未启动控制")
    position = float(current[gripper_index])
    if not np.isfinite(position) or not demo_lower <= position <= demo_upper:
        raise RuntimeError(f"夹爪起点超出演示范围：当前 {position:.2f}，允许 {demo_lower:.2f}～{demo_upper:.2f}。"
                           "已拒绝启用；演示范围与机械最大开度不同。托住从臂并确认扭矩已关闭后，"
                           "手动将夹爪起始开度调入上述范围，再重新启动；不会自动强行回位。")
    lo, hi = lo.copy(), hi.copy()
    lo[gripper_index], hi[gripper_index] = demo_lower, demo_upper
    return lo, hi


def validate_gripper_raw_position(position, range_min, range_max):
    if not np.isfinite(position) or not range_min <= position <= range_max:
        if np.isfinite(position) and 0 <= position <= 4095:
            distance = range_min - position if position < range_min else position - range_max
            side = "低于下限" if position < range_min else "高于上限"
            raise RuntimeError(f"夹爪原始位置异常：当前位置 {position}，标定范围 {range_min}～{range_max}，"
                               f"{side} {distance:g} 个计数。这是原始位置越界，本次未执行保持验证。"
                               "已停止控制；托住从臂并确认扭矩已关闭后，将夹爪调整到标定与演示范围内，"
                               "不要顶住机械限位。若实际开度正常却持续越界，再核对坐标与标定；不会自动修改标定。")
        raise RuntimeError(f"夹爪原始位置异常：有符号位置 {position}，标定范围 {range_min}～{range_max}。"
                           "已拒绝启用；这不是可靠的开度读数。请先完全断电并按录制时的开度摆放，重新上电后只读复查；"
                           "若仍超出范围，先核对位置坐标与标定，不要自动取模、改标定或强行回位。")
