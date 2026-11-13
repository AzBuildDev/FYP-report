"""检查闭环相机输入并按实时关节位置限制目标；不访问硬件。"""
import numpy as np


def validate_frame(stamp, image, now, max_age=0.5):
    if image is None or not np.isfinite(stamp) or now - stamp > max_age or stamp > now:
        raise RuntimeError("相机画面中断，停止控制。")
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
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
