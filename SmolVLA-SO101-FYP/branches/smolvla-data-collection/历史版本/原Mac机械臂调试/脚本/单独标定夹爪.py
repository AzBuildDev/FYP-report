#!/usr/bin/env python3
"""Recalibrate only the SO-101 leader/follower gripper open/close range."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


FOLLOWER_PORT = "/dev/tty.usbmodem5C821082421"
LEADER_PORT = "/dev/tty.usbmodem5C821083141"
FOLLOWER_ID = "my_awesome_follower_arm"
LEADER_ID = "my_awesome_leader_arm"

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATHS = {
    "follower": [
        Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower" / f"{FOLLOWER_ID}.json",
        ROOT / "标定备份/robots/so101_follower" / f"{FOLLOWER_ID}.json",
    ],
    "leader": [
        Path.home()
        / ".cache/huggingface/lerobot/calibration/teleoperators/so101_leader"
        / f"{LEADER_ID}.json",
        ROOT / "标定备份/teleoperators/so101_leader" / f"{LEADER_ID}.json",
    ],
}


def read_gripper_raw(device) -> int:
    return int(device.bus.read("Present_Position", "gripper", normalize=False))


def gripper_calibration(closed_raw: int, open_raw: int, current):
    if closed_raw == open_raw:
        raise ValueError("Closed and open gripper raw positions are identical; cannot calibrate.")

    if closed_raw < open_raw:
        return replace(current, drive_mode=0, range_min=closed_raw, range_max=open_raw)

    return replace(current, drive_mode=1, range_min=open_raw, range_max=closed_raw)


def write_json_files(device_name: str, calibration) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gripper = calibration["gripper"]
    for path in CALIBRATION_PATHS[device_name]:
        backup_path = path.with_suffix(path.suffix + f".bak_{timestamp}")
        shutil.copy2(path, backup_path)
        data = json.loads(path.read_text())
        data["gripper"]["drive_mode"] = gripper.drive_mode
        data["gripper"]["range_min"] = gripper.range_min
        data["gripper"]["range_max"] = gripper.range_max
        path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Updated {path}")
        print(f"Backup  {backup_path}")


def print_norms(leader: SO101Leader, follower: SO101Follower) -> None:
    leader_value = leader.get_action()["gripper.pos"]
    follower_value = follower.get_observation()["gripper.pos"]
    print(f"leader gripper.pos:   {leader_value:.2f}")
    print(f"follower gripper.pos: {follower_value:.2f}")


def main() -> None:
    follower = SO101Follower(
        SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, max_relative_target=5)
    )
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))

    try:
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)
        follower.bus.disable_torque()

        print("This recalibrates only the gripper ranges.")
        print("Follower torque is disabled so you can move its gripper by hand.")
        print()

        input("Set BOTH grippers physically CLOSED, then press ENTER...")
        leader_closed = read_gripper_raw(leader)
        follower_closed = read_gripper_raw(follower)
        print(f"leader closed raw:   {leader_closed}")
        print(f"follower closed raw: {follower_closed}")
        print()

        input("Set BOTH grippers physically OPEN, then press ENTER...")
        leader_open = read_gripper_raw(leader)
        follower_open = read_gripper_raw(follower)
        print(f"leader open raw:   {leader_open}")
        print(f"follower open raw: {follower_open}")
        print()

        leader.calibration["gripper"] = gripper_calibration(
            leader_closed, leader_open, leader.calibration["gripper"]
        )
        follower.calibration["gripper"] = gripper_calibration(
            follower_closed, follower_open, follower.calibration["gripper"]
        )

        print("New gripper calibration:")
        print(f"leader:   {leader.calibration['gripper']}")
        print(f"follower: {follower.calibration['gripper']}")
        print()

        answer = input("Write this calibration to motors and JSON files? Type YES to continue: ")
        if answer.strip() != "YES":
            print("Cancelled. Nothing was written.")
            return

        leader.bus.write_calibration(leader.calibration)
        follower.bus.write_calibration(follower.calibration)
        write_json_files("leader", leader.calibration)
        write_json_files("follower", follower.calibration)

        print()
        print("Verification readings with new calibration:")
        print_norms(leader, follower)
    finally:
        try:
            leader.disconnect()
        except Exception:
            pass
        try:
            follower.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
