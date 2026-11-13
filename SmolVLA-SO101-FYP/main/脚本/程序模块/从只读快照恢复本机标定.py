"""从本机完整只读报告恢复文件；不连接串口、不修改电机、不覆盖已有标定。"""
import argparse
import json
from pathlib import Path

from lerobot.utils.encoding_utils import decode_sign_magnitude

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# 仓库历史表中的身体五关节，以及 2026-10-19 回滚后的原生夹爪坐标。
EXPECTED_FOLLOWER = ((-173, 708, 3314), (-17, 2041, 4425), (112, 0, 4095),
                     (-80, 39, 2319), (-78, 131, 3971), (1925, 2046, 3492))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    arms = json.loads(args.report.read_text(encoding="utf-8"))
    roles = {"5C82108242": ("robots/so101_follower/my_awesome_follower_arm.json", "follower"),
             "5C82108314": ("teleoperators/so101_leader/my_awesome_leader_arm.json", "leader")}
    if len(arms) != 2 or {a["serial"] for a in arms} != set(roles):
        raise RuntimeError("报告不包含指定的两套硬件")
    outputs = []
    for arm in arms:
        relative, role = roles[arm["serial"]]
        motors = sorted(arm["motors"], key=lambda m: m["id"])
        if arm.get("connection_error") or [m["id"] for m in motors] != list(range(1, 7)):
            raise RuntimeError("报告不完整")
        calibration = {}
        for name, motor in zip(NAMES, motors):
            if (motor["model"] != 777 or motor["ping_status"] or motor["error"]
                    or motor.get("fault_status") != 0 or motor.get("torque_enabled") != 0
                    or motor.get("read_errors") or not motor.get("read_statuses")
                    or any(motor["read_statuses"].values())):
                raise RuntimeError(f"{role} {name} 的读取、故障或扭矩检查未通过")
            offset = decode_sign_magnitude(motor["homing_offset_raw"], 11)
            values = (offset, motor["range_min"], motor["range_max"])
            if role == "follower" and values != EXPECTED_FOLLOWER[motor["id"] - 1]:
                raise RuntimeError(f"从臂 {name} 与历史标定不符：{values}")
            calibration[name] = dict(id=motor["id"], drive_mode=0, homing_offset=offset,
                                     range_min=values[1], range_max=values[2])
        path = ROOT / "标定备份" / relative
        if path.exists():
            raise FileExistsError(f"不覆盖已有标定：{path}")
        outputs.append((path, calibration))
    for path, calibration in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            json.dump(calibration, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(path, flush=True)
    evidence = {"source_report": str(args.report.resolve()), "motor_writes": 0,
                "follower_matches_documented_calibration": True,
                "drive_mode_basis": "FeetechMotorsBus.read_calibration uses drive_mode=0",
                "files": [str(path) for path, _ in outputs]}
    (ROOT / "运行记录/本机标定恢复来源_20261024.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
