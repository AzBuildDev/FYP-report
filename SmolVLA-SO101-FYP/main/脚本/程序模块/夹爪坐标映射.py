"""按本机端点核对记录转换夹爪坐标；训练坐标与设备坐标必须双向一致。"""
import json
from pathlib import Path

from 检查控制输入 import validate_gripper_raw_position


def closed_endpoint_tolerance(calibration):
    """仅适用正向开度、单圈合法标定；不移动零位、不扩大目标行程。"""
    lower, upper = calibration.range_min, calibration.range_max
    if calibration.drive_mode != 0 or not 0 <= lower < upper <= 4095:
        return 0
    return min(16, int((upper - lower) * .0125))


def native_gripper_feedback_position(position, calibration):
    lower, upper = calibration.range_min, calibration.range_max
    if 0 <= position < lower and lower - position <= closed_endpoint_tolerance(calibration):
        return lower
    validate_gripper_raw_position(position, lower, upper)
    return position


def prepare_initial_gripper_hold(positions, calibration):
    """预装目标始终在硬件限位内；只修正已接受的闭合端小偏差。"""
    result = positions.copy()
    result["gripper"] = native_gripper_feedback_position(result["gripper"], calibration)
    validate_gripper_raw_position(result["gripper"], calibration.range_min, calibration.range_max)
    return result


def _guard_native_coordinates(bus, calibration):
    motor_id = bus.motors["gripper"].id
    original_normalize, original_unnormalize = bus._normalize, bus._unnormalize

    def normalize(values):
        values = values.copy()
        if motor_id in values:
            values[motor_id] = native_gripper_feedback_position(values[motor_id], calibration)
        return original_normalize(values)

    def unnormalize(values):
        result = original_unnormalize(values)
        if motor_id in result:
            # 输出目标仍严格限位，反馈容差不用于放宽目标。
            validate_gripper_raw_position(result[motor_id], calibration.range_min, calibration.range_max)
        return result

    bus._normalize, bus._unnormalize = normalize, unnormalize


def attach_gripper_coordinates(bus, root, serial, *, require_hold=False, on_diagnostic=None):
    calibration = bus.calibration["gripper"]
    native = int(bus.read("Present_Position", "gripper", normalize=False))
    lower, upper = calibration.range_min, calibration.range_max
    tolerance = closed_endpoint_tolerance(calibration)
    near_closed = 0 <= native < lower and lower - native <= tolerance
    # 在任何启动拒绝之前保存实际读数，避免只留下历史失败提示。
    if on_diagnostic is not None:
        on_diagnostic({"type": "gripper_raw_diagnostic", "phase": "connection",
                       "signed_position": native, "calibration_min": lower,
                       "calibration_max": upper, "in_calibration_range": lower <= native <= upper,
                       "closed_endpoint_tolerance": tolerance, "near_closed_endpoint": near_closed})
    repair_path = Path(root) / "运行环境/夹爪坐标修复.json"
    if require_hold and repair_path.is_file():
        repair = json.loads(repair_path.read_text(encoding="utf-8-sig"))
        if repair.get("serial") == serial and (repair.get("inference_allowed") is not True
                                                or repair.get("endpoints_verified") is not True
                                                or repair.get("hold_verified") is not True):
            raise RuntimeError("夹爪零位已调整，端点与保持验证尚未全部通过，推理继续锁定")
    if lower <= native <= upper or near_closed:
        effective = native_gripper_feedback_position(native, calibration)
        _guard_native_coordinates(bus, calibration)
        result = {"shift": 0, "native_position": native, "training_position": effective}
        if near_closed:
            result.update(closed_endpoint_correction=effective - native,
                          closed_endpoint_tolerance=tolerance)
        return result

    # 普通端点越界不属于整圈偏移，不能引用旧偏移方案的失败记录。
    if not any(lower <= native + shift <= upper for shift in (-4096, 4096)):
        validate_gripper_raw_position(native, lower, upper)
    record_path = Path(root) / "运行环境/夹爪端点核对.json"
    if not record_path.is_file():
        validate_gripper_raw_position(native, lower, upper)
    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    motor = bus.motors["gripper"]
    identity = {"serial": serial, "motor_id": motor.id, "model": motor.model,
                "homing_offset": calibration.homing_offset, "drive_mode": calibration.drive_mode,
                "range_min": lower, "range_max": upper}
    if record.get("schema_version") != 1 or record.get("endpoints_verified") is not True or record.get("identity") != identity:
        raise RuntimeError("夹爪端点核对记录与当前设备/标定不匹配，未启用控制")
    shift = record.get("position_shift")
    if type(shift) is not int or abs(shift) != 4096 or not 0 < upper - lower < 4096:
        raise RuntimeError("夹爪坐标偏移记录无效，未启用控制")
    validate_gripper_raw_position(native + shift, lower, upper)
    if require_hold and record.get("hardware_command_rejected") is True:
        raise RuntimeError(f"当前夹爪位置 {native} 需要偏移 {shift} 才能对应标定范围 {lower}～{upper}。"
                           "该偏移方案此前实机保持验证失败，仍禁止推理；本次未重新执行保持验证，未启用控制。"
                           "请停止启动并核对设备坐标，不要删除失败记录或强行发送偏移目标。")
    if require_hold and record.get("hold_verified") is not True:
        raise RuntimeError("夹爪端点坐标已核对，但实机保持验证尚未通过，暂不运行模型。请先完成独立夹爪保持验证。")

    original_normalize, original_unnormalize = bus._normalize, bus._unnormalize

    def normalize(values):
        values = values.copy()
        if motor.id in values:
            corrected = values[motor.id] + shift
            validate_gripper_raw_position(corrected, lower, upper)
            values[motor.id] = corrected
        return original_normalize(values)

    def unnormalize(values):
        if motor.id in values and record.get("hardware_command_rejected") is True:
            raise RuntimeError("夹爪负坐标目标未通过设备验证，禁止生成发送目标")
        values = original_unnormalize(values)
        if motor.id in values:
            validate_gripper_raw_position(values[motor.id], lower, upper)
            values[motor.id] -= shift
        return values

    bus._normalize, bus._unnormalize = normalize, unnormalize
    return {"shift": shift, "native_position": native, "training_position": native + shift,
            "verification_record": str(record_path)}
