import argparse
import csv
import time
from pathlib import Path

from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader
from lerobot.utils.robot_utils import busy_wait


MOTOR_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def display_value(value):
    value = scalar(value)
    return float(value) if value != "" else 0.0


def write_motor_values(row, prefix, values):
    for key in MOTOR_KEYS:
        row[f"{prefix}.{key}"] = scalar(values.get(key))


def main():
    parser = argparse.ArgumentParser(description="Teleoperate SO101 and record leader/follower motor traces.")
    parser.add_argument("--follower_port", default="COM11")
    parser.add_argument("--leader_port", default="COM12")
    parser.add_argument("--follower_id", default="my_awesome_follower_arm")
    parser.add_argument("--leader_id", default="my_awesome_leader_arm")
    parser.add_argument("--output_dir", default="F:/FYP/Project/so101_mac_bringup/inference_traces")
    parser.add_argument("--trial_id", default="")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--duration_s", type=float, default=30.0)
    parser.add_argument("--max_relative_target", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trial_id = args.trial_id or f"teleop_home_demo_{timestamp}"
    output_path = output_dir / f"{trial_id}_{int(time.time())}.csv"

    fieldnames = ["trial_id", "event", "wall_time", "elapsed_s", "timestep", "loop_ms"]
    for prefix in ["leader", "command", "performed", "obs_before", "obs_after"]:
        fieldnames += [f"{prefix}.{key}" for key in MOTOR_KEYS]

    robot = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            max_relative_target=args.max_relative_target,
            cameras={},
        )
    )
    teleop = SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))

    print(f"Writing trace to: {output_path}")
    print("Connects leader and follower in one process, so do not run lerobot-teleoperate at the same time.")
    print("If calibration prompt appears, press ENTER to use the existing calibration.")

    teleop.connect()
    robot.connect()

    try:
        input("Press ENTER to start recording teleop trace...")
        start = time.perf_counter()
        timestep = 0
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            while True:
                loop_start = time.perf_counter()
                elapsed = loop_start - start
                if args.duration_s > 0 and elapsed >= args.duration_s:
                    break

                obs_before = robot.get_observation()
                command = teleop.get_action()
                performed = robot.send_action(command)
                obs_after = robot.get_observation()

                loop_ms = (time.perf_counter() - loop_start) * 1000.0
                row = {
                    "trial_id": trial_id,
                    "event": "teleop_step",
                    "wall_time": time.time(),
                    "elapsed_s": elapsed,
                    "timestep": timestep,
                    "loop_ms": loop_ms,
                }
                write_motor_values(row, "leader", command)
                write_motor_values(row, "command", command)
                write_motor_values(row, "performed", performed)
                write_motor_values(row, "obs_before", obs_before)
                write_motor_values(row, "obs_after", obs_after)
                writer.writerow(row)
                file.flush()

                timestep += 1
                busy_wait((1.0 / args.fps) - (time.perf_counter() - loop_start))

                print(
                    f"\rrecording {elapsed:6.2f}s | "
                    f"lift={display_value(obs_after.get('shoulder_lift.pos')):7.2f} "
                    f"elbow={display_value(obs_after.get('elbow_flex.pos')):7.2f} "
                    f"wrist={display_value(obs_after.get('wrist_flex.pos')):7.2f} "
                    f"gripper={display_value(obs_after.get('gripper.pos')):7.2f}",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\nStopped by Ctrl+C.")
    finally:
        print(f"\nSaved trace: {output_path}")
        try:
            if teleop.is_connected:
                teleop.disconnect()
        finally:
            if robot.is_connected:
                robot.disconnect()


if __name__ == "__main__":
    main()
