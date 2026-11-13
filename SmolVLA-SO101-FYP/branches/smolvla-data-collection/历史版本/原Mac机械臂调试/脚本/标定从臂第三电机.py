#!/usr/bin/env python3
"""Recalibrate only the follower elbow_flex motor, ID 3."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


FOLLOWER_PORT = "/dev/tty.usbmodem5C821082421"
FOLLOWER_ID = "my_awesome_follower_arm"
MOTOR_NAME = "elbow_flex"

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATHS = [
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower" / f"{FOLLOWER_ID}.json",
    ROOT / "标定备份/robots/so101_follower" / f"{FOLLOWER_ID}.json",
]


def update_json_files(calibration: MotorCalibration) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in CALIBRATION_PATHS:
        backup_path = path.with_suffix(path.suffix + f".bak_{timestamp}")
        shutil.copy2(path, backup_path)
        data = json.loads(path.read_text())
        data[MOTOR_NAME] = {
            "id": calibration.id,
            "drive_mode": calibration.drive_mode,
            "homing_offset": calibration.homing_offset,
            "range_min": calibration.range_min,
            "range_max": calibration.range_max,
        }
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Updated {path}")
        print(f"Backup  {backup_path}")


def main() -> None:
    bus = FeetechMotorsBus(
        port=FOLLOWER_PORT,
        motors={MOTOR_NAME: Motor(3, "sts3215", MotorNormMode.RANGE_M100_100)},
    )
    try:
        bus.connect(handshake=False)
        print("This recalibrates only follower motor ID 3 / elbow_flex.")
        print("Other motors are not changed.")
        print()

        bus.disable_torque(MOTOR_NAME, num_retry=5)
        input("Move follower ID3/elbow_flex to the middle of its usable range, then press ENTER...")
        homing_offsets = bus.set_half_turn_homings(MOTOR_NAME)

        print()
        print("Now slowly move only ID3/elbow_flex through its safe usable range.")
        print("Do not force it into a hard mechanical stop.")
        input("Press ENTER to start recording min/max, then press ENTER again when finished...")
        range_mins, range_maxes = bus.record_ranges_of_motion(MOTOR_NAME)

        calibration = MotorCalibration(
            id=3,
            drive_mode=0,
            homing_offset=homing_offsets[MOTOR_NAME],
            range_min=range_mins[MOTOR_NAME],
            range_max=range_maxes[MOTOR_NAME],
        )

        print()
        print(f"New {MOTOR_NAME} calibration: {calibration}")
        answer = input("Write this calibration to motor and JSON files? Type YES to continue: ")
        if answer.strip() != "YES":
            print("Cancelled. Nothing was written.")
            return

        bus.write_calibration({MOTOR_NAME: calibration})
        update_json_files(calibration)

        print()
        print("Done. Power-cycle the arm once, then run a short teleoperation test.")
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
