#!/usr/bin/env python3
"""Restore follower ID3/elbow_flex calibration from the latest backup."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


FOLLOWER_PORT = "/dev/tty.usbmodem5C821082421"
FOLLOWER_ID = "my_awesome_follower_arm"
MOTOR_NAME = "elbow_flex"
FULL_RANGE_MIN = 0
FULL_RANGE_MAX = 4095

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PATHS = [
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower" / f"{FOLLOWER_ID}.json",
    ROOT / "标定备份/robots/so101_follower" / f"{FOLLOWER_ID}.json",
]


def load_motor_calibration(path: Path) -> MotorCalibration:
    data = json.loads(path.read_text())[MOTOR_NAME]
    return MotorCalibration(
        id=data["id"],
        drive_mode=data["drive_mode"],
        homing_offset=data["homing_offset"],
        range_min=data["range_min"],
        range_max=data["range_max"],
    )


def find_backup_candidates() -> list[Path]:
    candidates: set[Path] = set()
    for directory in {path.parent for path in ACTIVE_PATHS}:
        candidates.update(directory.glob(f"{FOLLOWER_ID}.json.*"))
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def pick_backup_path(cli_path: str | None) -> Path:
    if cli_path:
        chosen = Path(cli_path).expanduser().resolve()
        if not chosen.exists():
            raise FileNotFoundError(f"Backup file not found: {chosen}")
        return chosen

    candidates = find_backup_candidates()
    if not candidates:
        raise FileNotFoundError("No backup calibration files were found for follower ID3.")

    for path in candidates:
        calibration = load_motor_calibration(path)
        if (calibration.range_min, calibration.range_max) != (FULL_RANGE_MIN, FULL_RANGE_MAX):
            return path

    return candidates[0]


def update_json_files(calibration: MotorCalibration) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in ACTIVE_PATHS:
        backup_path = path.with_suffix(path.suffix + f".before_restore_{timestamp}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore follower elbow_flex calibration from a backup JSON file."
    )
    parser.add_argument(
        "--source",
        help="Optional explicit backup JSON path. If omitted, the script prefers the newest non-full-range backup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = pick_backup_path(args.source)
    calibration = load_motor_calibration(source_path)
    print(f"Restoring {MOTOR_NAME} from {source_path}")
    print(f"Calibration: {calibration}")

    bus = FeetechMotorsBus(
        port=FOLLOWER_PORT,
        motors={MOTOR_NAME: Motor(3, "sts3215", MotorNormMode.RANGE_M100_100)},
        calibration={MOTOR_NAME: calibration},
    )
    try:
        bus.connect(handshake=False)
        bus.disable_torque(MOTOR_NAME, num_retry=5)
        bus.write_calibration({MOTOR_NAME: calibration})
        update_json_files(calibration)
        print("Done. Power-cycle the follower arm before testing.")
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
