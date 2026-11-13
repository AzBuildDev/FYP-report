#!/usr/bin/env python3
"""Probe SO-101 leader/follower gripper mapping without commanding movement."""

from __future__ import annotations

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


FOLLOWER_PORT = "/dev/tty.usbmodem5C821082421"
LEADER_PORT = "/dev/tty.usbmodem5C821083141"
FOLLOWER_ID = "my_awesome_follower_arm"
LEADER_ID = "my_awesome_leader_arm"


def read_grippers(leader: SO101Leader, follower: SO101Follower) -> None:
    leader_action = leader.get_action()
    follower_obs = follower.get_observation()
    print(f"leader gripper.pos:   {leader_action['gripper.pos']:.2f}")
    print(f"follower gripper.pos: {follower_obs['gripper.pos']:.2f}")


def main() -> None:
    follower = SO101Follower(
        SO101FollowerConfig(port=FOLLOWER_PORT, id=FOLLOWER_ID, max_relative_target=5)
    )
    leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID))

    try:
        leader.connect(calibrate=False)
        follower.connect(calibrate=False)
        follower.bus.disable_torque()

        print("Torque is disabled on the follower so you can move its gripper by hand.")
        print("Only gripper positions are read; this script does not send motion commands.")
        print()

        for label in ("both physically OPEN", "both physically CLOSED", "leader CLOSED, follower OPEN"):
            input(f"Set {label}, then press ENTER...")
            read_grippers(leader, follower)
            print()
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
