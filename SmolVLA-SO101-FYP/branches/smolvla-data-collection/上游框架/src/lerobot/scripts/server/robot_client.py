# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command:
```shell
python src/lerobot/scripts/server/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import pickle  # nosec
import csv
import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.scripts.server.configs import RobotClientConfig
from lerobot.scripts.server.constants import SUPPORTED_ROBOTS
from lerobot.scripts.server.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    validate_robot_cameras_for_policy,
    visualize_action_queue_size,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


DEFAULT_POST_RELEASE_HOME = {
    "shoulder_pan.pos": 5.49,
    "shoulder_lift.pos": -97.74,
    "elbow_flex.pos": 0.02,
    "wrist_flex.pos": 60.88,
    "wrist_roll.pos": 1.33,
    "gripper.pos": 1.45,
}


class TrialTraceLogger:
    """Write per-trial robot observations/actions without changing control behavior."""

    def __init__(
        self,
        output_dir: str,
        trial_id: str,
        observation_features: list[str],
        action_features: list[str],
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        safe_trial_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", trial_id.strip()) if trial_id else ""
        if not safe_trial_id:
            safe_trial_id = time.strftime("trial_%Y%m%d_%H%M%S")

        self.trial_id = safe_trial_id
        self.start_wall_time = time.time()
        self.path = self.output_dir / f"{safe_trial_id}_{int(self.start_wall_time)}.csv"

        self.observation_features = observation_features
        self.action_features = action_features
        fieldnames = [
            "trial_id",
            "event",
            "wall_time",
            "elapsed_s",
            "timestep",
            "queue_size",
            "must_go",
        ]
        fieldnames += [f"obs.{name}" for name in observation_features]
        fieldnames += [f"action_cmd.{name}" for name in action_features]
        fieldnames += [f"action_performed.{name}" for name in action_features]

        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()
        self.file.flush()

    @staticmethod
    def _as_scalar(value: Any) -> float | int | str | None:
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            return value.detach().cpu().reshape(-1)[0].item()

        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                return None
            return TrialTraceLogger._as_scalar(value[0])

        if hasattr(value, "shape") and getattr(value, "size", 1) != 1:
            return None

        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return None

        if isinstance(value, (int, float, str, bool)):
            return value

        return None

    def _base_row(self, event: str, timestep: int | None, queue_size: int | None, must_go: bool | None):
        wall_time = time.time()
        return {
            "trial_id": self.trial_id,
            "event": event,
            "wall_time": f"{wall_time:.6f}",
            "elapsed_s": f"{wall_time - self.start_wall_time:.6f}",
            "timestep": "" if timestep is None else timestep,
            "queue_size": "" if queue_size is None else queue_size,
            "must_go": "" if must_go is None else must_go,
        }

    def record_observation(
        self,
        raw_observation: RawObservation,
        timestep: int,
        queue_size: int,
        must_go: bool,
    ) -> None:
        row = self._base_row("observation", timestep, queue_size, must_go)
        for name in self.observation_features:
            value = self._as_scalar(raw_observation.get(name))
            if value is not None:
                row[f"obs.{name}"] = value
        self.writer.writerow(row)
        self.file.flush()

    def record_state(
        self,
        event: str,
        state: dict[str, Any],
        timestep: int | None,
        queue_size: int | None,
    ) -> None:
        row = self._base_row(event, timestep, queue_size, None)
        for name in self.observation_features:
            value = self._as_scalar(state.get(name))
            if value is not None:
                row[f"obs.{name}"] = value
        self.writer.writerow(row)
        self.file.flush()

    def record_action(
        self,
        timestep: int,
        queue_size: int,
        command_action: dict[str, Any],
        performed_action: dict[str, Any] | None,
    ) -> None:
        row = self._base_row("action", timestep, queue_size, None)
        for name in self.action_features:
            command_value = self._as_scalar(command_action.get(name))
            if command_value is not None:
                row[f"action_cmd.{name}"] = command_value

            performed_value = None
            if performed_action is not None:
                performed_value = self._as_scalar(performed_action.get(name))
            if performed_value is not None:
                row[f"action_performed.{name}"] = performed_value

        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        if config.verify_robot_cameras:
            # Load policy config for validation
            policy_config = PreTrainedConfig.from_pretrained(config.pretrained_name_or_path)
            policy_image_features = policy_config.image_features

            # The cameras specified for inference must match the one supported by the policy chosen
            validate_robot_cameras_for_policy(lerobot_features, policy_image_features)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

        self.post_release_home_started_at = time.time()
        self.post_release_home_closed_seen = False
        self.post_release_home_max_gripper = None
        self.post_release_home_open_count = 0
        self.post_release_home_release_started_at = None
        self.post_release_home_target_started_at = None
        self.post_release_home_initial_state = None
        self.post_release_home_manual_requested = threading.Event()
        self.post_release_home_in_progress = threading.Event()

        self.trace_logger = None
        if config.trace_enabled:
            observation_features = list(self.robot.observation_features)
            action_features = list(self.robot.action_features)
            self.trace_logger = TrialTraceLogger(
                config.trace_output_dir,
                config.trace_trial_id,
                observation_features,
                action_features,
            )
            self.logger.info(f"Writing trial trace to {self.trace_logger.path}")

    def start_post_release_home_manual_listener(self) -> None:
        if not self.config.post_release_home_manual_trigger_enabled:
            return

        def wait_for_enter():
            input(
                "\n[manual home] Press ENTER after the object is placed and the gripper is open to return home...\n"
            )
            self.post_release_home_manual_requested.set()

        listener = threading.Thread(target=wait_for_enter, daemon=True)
        listener.start()

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if isinstance(value, str):
                if value.strip() == "":
                    return None
                return float(value)

            scalar = TrialTraceLogger._as_scalar(value)
            if scalar is None:
                return None
            return float(scalar)
        except Exception:
            return None

    def _clear_action_queue(self) -> None:
        with self.action_queue_lock:
            self.action_queue = Queue()

    def _read_motor_state(self) -> dict[str, float]:
        state = self.robot.bus.sync_read("Present_Position")
        return {f"{motor}.pos": float(value) for motor, value in state.items()}

    @staticmethod
    def _interpolate_action(
        start: dict[str, float], target: dict[str, float], alpha: float
    ) -> dict[str, float]:
        return {key: start[key] + (target[key] - start[key]) * alpha for key in target}

    def _send_supervisor_action(self, action: dict[str, float], timestep: int) -> None:
        performed_action = self.robot.send_action(action)
        if self.trace_logger is not None:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()
            self.trace_logger.record_action(
                timestep=timestep,
                queue_size=current_queue_size,
                command_action=action,
                performed_action=performed_action,
            )

    def _send_interpolated_supervisor_steps(
        self, target: dict[str, float], steps: int, sleep_s: float, start_timestep: int
    ) -> int:
        current = self._read_motor_state()
        action_keys = list(self.robot.action_features)
        start = {key: current.get(key, target.get(key, 0.0)) for key in action_keys}
        target = {key: target.get(key, start[key]) for key in action_keys}

        timestep = start_timestep
        for i in range(1, steps + 1):
            action = self._interpolate_action(start, target, i / steps)
            self._send_supervisor_action(action, timestep)
            timestep += 1
            time.sleep(sleep_s)
        return timestep

    def _run_closed_loop_prelift(self, current: dict[str, float], start_timestep: int) -> tuple[dict[str, float], int]:
        if "shoulder_lift.pos" not in current:
            return current, start_timestep

        target_lift = current["shoulder_lift.pos"] + self.config.post_release_home_prelift_delta
        direction = -1.0 if target_lift < current["shoulder_lift.pos"] else 1.0
        max_step = 3.0
        tolerance = 1.5
        timestep = start_timestep

        self.logger.info(
            "Post-release home closed-loop pre-lift: "
            f"shoulder_lift {current['shoulder_lift.pos']:.2f} -> {target_lift:.2f}"
        )

        for _ in range(self.config.post_release_home_prelift_steps):
            actual = self._read_motor_state()
            actual_lift = actual.get("shoulder_lift.pos")
            if actual_lift is None:
                break

            remaining = target_lift - actual_lift
            if abs(remaining) <= tolerance or remaining * direction <= 0:
                current = actual
                break

            action = actual.copy()
            action["shoulder_lift.pos"] = actual_lift + direction * min(max_step, abs(remaining))
            self._send_supervisor_action(action, timestep)
            timestep += 1
            time.sleep(max(self.config.post_release_home_sleep_s, 0.08))
            current = actual

        current = self._read_motor_state()

        return current, timestep

    def _send_repeated_supervisor_steps(
        self, target: dict[str, float], steps: int, sleep_s: float, start_timestep: int
    ) -> int:
        timestep = start_timestep
        for _ in range(steps):
            self._send_supervisor_action(target, timestep)
            timestep += 1
            time.sleep(sleep_s)
        return timestep

    def _load_post_release_replay_waypoints(self) -> list[dict[str, float]]:
        if not self.config.post_release_home_replay_trace_path:
            return []

        path = Path(self.config.post_release_home_replay_trace_path)
        if not path.exists():
            raise FileNotFoundError(f"Replay trace does not exist: {path}")

        waypoints = []
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("event") != "observation":
                    continue

                elapsed_s = self._safe_float(row.get("elapsed_s"))
                if elapsed_s is None:
                    continue

                if elapsed_s < self.config.post_release_home_replay_start_s:
                    continue

                if (
                    self.config.post_release_home_replay_end_s
                    > self.config.post_release_home_replay_start_s
                    and elapsed_s > self.config.post_release_home_replay_end_s
                ):
                    continue

                waypoint = {}
                for key in self.robot.action_features:
                    value = self._safe_float(row.get(f"obs.{key}"))
                    if value is None:
                        value = self._safe_float(row.get(f"obs_after.{key}"))
                    if value is not None:
                        waypoint[key] = value

                if set(waypoint) == set(self.robot.action_features):
                    waypoints.append(waypoint)

        if not waypoints:
            raise RuntimeError(f"No replay waypoints found in {path}")

        return waypoints

    def _run_post_release_replay_home(self, start_timestep: int) -> int:
        waypoints = self._load_post_release_replay_waypoints()
        timestep = start_timestep

        self.logger.info(
            "Post-release replay home: "
            f"{len(waypoints)} waypoints from {self.config.post_release_home_replay_trace_path}"
        )
        replay_waypoints = waypoints
        if self.config.post_release_home_replay_bridge_steps > 0 and waypoints:
            first_waypoint = waypoints[0]
            self.logger.info(
                "Post-release replay bridge to first waypoint: "
                f"{self.config.post_release_home_replay_bridge_steps} steps"
            )
            timestep = self._send_interpolated_supervisor_steps(
                first_waypoint,
                self.config.post_release_home_replay_bridge_steps,
                self.config.post_release_home_replay_bridge_sleep_s,
                timestep,
            )
            replay_waypoints = waypoints[1:]

        for waypoint in replay_waypoints:
            timestep = self._send_interpolated_supervisor_steps(
                waypoint,
                self.config.post_release_home_replay_steps_per_waypoint,
                self.config.post_release_home_replay_sleep_s,
                timestep,
            )

        return timestep

    def _record_post_release_home_trigger(self, event: str, state: dict[str, float], timestep: int) -> None:
        if self.trace_logger is None:
            return

        with self.action_queue_lock:
            current_queue_size = self.action_queue.qsize()
        self.trace_logger.record_state(
            event=event,
            state=state,
            timestep=timestep,
            queue_size=current_queue_size,
        )

    def run_post_release_home_sequence(self, trigger_event: str = "post_release_home_trigger") -> None:
        self.post_release_home_in_progress.set()
        self._clear_action_queue()

        with self.latest_action_lock:
            timestep = max(self.latest_action + 1, 0)

        self.logger.info("Post-release home triggered: policy released gripper; returning home")
        try:
            current = self._read_motor_state()
            self._record_post_release_home_trigger(trigger_event, current, timestep)

            if self.config.post_release_home_prelift_enabled and "shoulder_lift.pos" in current:
                current, timestep = self._run_closed_loop_prelift(current, timestep)

            if self.config.post_release_home_replay_trace_path:
                timestep = self._run_post_release_replay_home(timestep)

                with self.latest_action_lock:
                    self.latest_action = timestep

                self.logger.info("Post-release replay home complete; stopping client")
                return

            home = (
                self.post_release_home_initial_state.copy()
                if self.post_release_home_initial_state is not None
                else DEFAULT_POST_RELEASE_HOME.copy()
            )
            if "gripper.pos" in current:
                home["gripper.pos"] = current["gripper.pos"]

            timestep = self._send_interpolated_supervisor_steps(
                home,
                self.config.post_release_home_steps,
                self.config.post_release_home_sleep_s,
                timestep,
            )
            if self.config.post_release_home_final_close_enabled:
                close_target = home.copy()
                close_target["gripper.pos"] = self.config.post_release_home_final_gripper
                self.logger.info(
                    "Post-release home final gripper close: "
                    f"{home.get('gripper.pos', 0.0):.2f} -> {close_target['gripper.pos']:.2f}"
                )
                timestep = self._send_interpolated_supervisor_steps(
                    close_target,
                    self.config.post_release_home_final_gripper_steps,
                    self.config.post_release_home_sleep_s,
                    timestep,
                )
                home = close_target

            timestep = self._send_repeated_supervisor_steps(
                home,
                self.config.post_release_home_final_hold_steps,
                self.config.post_release_home_sleep_s,
                timestep,
            )

            with self.latest_action_lock:
                self.latest_action = timestep

            self.logger.info("Post-release home complete; stopping client")
        except Exception as e:
            self.logger.error(f"Post-release home failed: {e}")
        finally:
            self.shutdown_event.set()

    def record_assisted_post_release_home_demo(self) -> None:
        self.post_release_home_in_progress.set()
        self._clear_action_queue()

        with self.latest_action_lock:
            timestep = max(self.latest_action + 1, 0)

        self.logger.info(
            "Manual assisted home recording started: move the arm to the desired home pose now"
        )
        try:
            start = time.time()
            while time.time() - start <= self.config.post_release_home_record_demo_s:
                state = self._read_motor_state()
                event = "manual_home_demo_start" if timestep == max(self.latest_action + 1, 0) else "manual_home_demo"
                self._record_post_release_home_trigger(event, state, timestep)
                timestep += 1
                time.sleep(self.config.post_release_home_record_demo_dt_s)

            with self.latest_action_lock:
                self.latest_action = timestep

            self.logger.info("Manual assisted home recording complete; stopping client")
        except Exception as e:
            self.logger.error(f"Manual assisted home recording failed: {e}")
        finally:
            self.shutdown_event.set()

    def update_post_release_home_state(self, raw_observation: RawObservation) -> None:
        self._maybe_capture_post_release_initial_state(raw_observation)
        self._update_post_release_home_from_state(raw_observation, "observation")

    def _maybe_capture_post_release_initial_state(self, raw_observation: RawObservation) -> None:
        if not self.config.post_release_home_enabled:
            return

        elapsed = time.time() - self.post_release_home_started_at
        if elapsed > self.config.post_release_home_initial_capture_s or self.post_release_home_closed_seen:
            return

        lift = self._safe_float(raw_observation.get("shoulder_lift.pos"))
        elbow = self._safe_float(raw_observation.get("elbow_flex.pos"))
        if lift is None or elbow is None:
            return

        if lift > -75 or abs(elbow) > 15:
            return

        state = {}
        for key in self.robot.action_features:
            value = self._safe_float(raw_observation.get(key))
            if value is not None:
                state[key] = value

        if set(state) == set(self.robot.action_features):
            self.post_release_home_initial_state = state

    def _is_post_release_target_pose(self, state: dict[str, Any]) -> bool:
        pan = self._safe_float(state.get("shoulder_pan.pos"))
        lift = self._safe_float(state.get("shoulder_lift.pos"))
        elbow = self._safe_float(state.get("elbow_flex.pos"))
        wrist = self._safe_float(state.get("wrist_flex.pos"))

        if pan is None or lift is None or elbow is None or wrist is None:
            return False

        return (
            self.config.post_release_home_target_pan_min
            <= pan
            <= self.config.post_release_home_target_pan_max
            and self.config.post_release_home_target_lift_min
            <= lift
            <= self.config.post_release_home_target_lift_max
            and self.config.post_release_home_target_elbow_min
            <= elbow
            <= self.config.post_release_home_target_elbow_max
            and self.config.post_release_home_target_wrist_min
            <= wrist
            <= self.config.post_release_home_target_wrist_max
        )

    def _post_release_open_threshold(self) -> float:
        threshold = self.config.post_release_home_open_threshold
        if self.post_release_home_initial_state is None:
            return threshold

        initial_gripper = self._safe_float(self.post_release_home_initial_state.get("gripper.pos"))
        if initial_gripper is None:
            return threshold

        return min(threshold, initial_gripper + self.config.post_release_home_open_margin)

    def _is_post_release_gripper_open(self, gripper_pos: float) -> tuple[bool, str]:
        if (
            self.config.post_release_home_release_gripper_min >= 0
            and self.config.post_release_home_release_gripper_max
            >= self.config.post_release_home_release_gripper_min
        ):
            if (
                self.config.post_release_home_release_gripper_min
                <= gripper_pos
                <= self.config.post_release_home_release_gripper_max
            ):
                return True, (
                    "release_range("
                    f"{self.config.post_release_home_release_gripper_min:.2f}<="
                    f"{gripper_pos:.2f}<="
                    f"{self.config.post_release_home_release_gripper_max:.2f})"
                )

            return False, (
                "outside_release_range("
                f"{self.config.post_release_home_release_gripper_min:.2f}<="
                f"{gripper_pos:.2f}<="
                f"{self.config.post_release_home_release_gripper_max:.2f})"
            )

        open_threshold = self._post_release_open_threshold()
        if gripper_pos <= open_threshold:
            return True, f"below_open_threshold({gripper_pos:.2f}<={open_threshold:.2f})"

        if self.post_release_home_max_gripper is None:
            return False, f"not_open({gripper_pos:.2f}>{open_threshold:.2f})"

        gripper_drop = self.post_release_home_max_gripper - gripper_pos
        if (
            gripper_pos <= self.config.post_release_home_open_threshold
            and gripper_drop >= self.config.post_release_home_release_drop_threshold
        ):
            return True, (
                f"large_drop(max={self.post_release_home_max_gripper:.2f}, "
                f"current={gripper_pos:.2f}, drop={gripper_drop:.2f})"
            )

        return False, (
            f"not_open(current={gripper_pos:.2f}, threshold={open_threshold:.2f}, "
            f"drop={gripper_drop:.2f})"
        )

    def _update_post_release_home_from_state(self, state: dict[str, Any], source: str) -> None:
        if not self.config.post_release_home_enabled:
            return

        if not self.config.post_release_home_auto_trigger_enabled:
            return

        if self.post_release_home_in_progress.is_set():
            return

        elapsed = time.time() - self.post_release_home_started_at
        if elapsed < self.config.post_release_home_min_elapsed_s:
            return

        gripper_pos = self._safe_float(state.get("gripper.pos"))
        if gripper_pos is None:
            return

        explicit_release_range_enabled = (
            self.config.post_release_home_release_gripper_min >= 0
            and self.config.post_release_home_release_gripper_max
            >= self.config.post_release_home_release_gripper_min
        )
        in_explicit_release_range = (
            explicit_release_range_enabled
            and self.config.post_release_home_release_gripper_min
            <= gripper_pos
            <= self.config.post_release_home_release_gripper_max
        )

        if gripper_pos >= self.config.post_release_home_closed_threshold and not (
            self.post_release_home_closed_seen and in_explicit_release_range
        ):
            if self.post_release_home_max_gripper is None:
                self.post_release_home_max_gripper = gripper_pos
            else:
                self.post_release_home_max_gripper = max(self.post_release_home_max_gripper, gripper_pos)
            if not self.post_release_home_closed_seen:
                self.logger.info(
                    f"Post-release home armed after closed gripper from {source}: {gripper_pos:.2f}"
                )
            self.post_release_home_closed_seen = True
            self.post_release_home_open_count = 0
            return

        if not self.post_release_home_closed_seen:
            return

        in_target_pose = self._is_post_release_target_pose(state)
        if in_target_pose:
            if self.post_release_home_target_started_at is None:
                self.post_release_home_target_started_at = time.time()
            target_dwell = time.time() - self.post_release_home_target_started_at
            if (
                self.config.post_release_home_target_dwell_s > 0
                and target_dwell >= self.config.post_release_home_target_dwell_s
            ):
                self.logger.info(
                    "Post-release home target dwell reached: "
                    f"{target_dwell:.2f}s >= {self.config.post_release_home_target_dwell_s:.2f}s"
                )
                self.run_post_release_home_sequence()
                return
        else:
            self.post_release_home_target_started_at = None

        is_open, open_reason = self._is_post_release_gripper_open(gripper_pos)
        if is_open:
            if in_target_pose:
                self.post_release_home_open_count += 1
                if self.post_release_home_release_started_at is None:
                    self.post_release_home_release_started_at = time.time()
                release_dwell = time.time() - self.post_release_home_release_started_at
                self.logger.info(
                    "Post-release home target release observation "
                    f"{self.post_release_home_open_count}/{self.config.post_release_home_min_open_observations}: "
                    f"{open_reason}, "
                    f"dwell={release_dwell:.2f}s from {source}"
                )
            else:
                if self.post_release_home_open_count:
                    self.logger.info(
                        f"Post-release home open count reset; gripper is open outside target pose from {source}"
                    )
                self.post_release_home_open_count = 0
                self.post_release_home_release_started_at = None
        else:
            self.post_release_home_open_count = 0
            self.post_release_home_release_started_at = None

        release_dwell = (
            time.time() - self.post_release_home_release_started_at
            if self.post_release_home_release_started_at is not None
            else 0
        )
        if (
            self.post_release_home_open_count >= self.config.post_release_home_min_open_observations
            and release_dwell >= self.config.post_release_home_release_dwell_s
        ):
            self.run_post_release_home_sequence()

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        self.shutdown_event.set()

        if self.trace_logger is not None:
            self.trace_logger.close()
            self.logger.info(f"Trial trace saved to {self.trace_logger.path}")

        try:
            self.robot.disconnect()
            self.logger.debug("Robot disconnected")
        except Exception as e:
            self.logger.error(f"Robot disconnect failed: {e}")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.info(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                # Calculate network latency if we have matching observations
                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    # Get queue state before changes
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]  # queue was empty

                    # Log incoming actions
                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    # Get queue state after changes
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        command_action = self._action_tensor_to_action_dict(timed_action.get_action())
        _performed_action = self.robot.send_action(command_action)
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        with self.action_queue_lock:
            current_queue_size = self.action_queue.qsize()

        if self.trace_logger is not None:
            self.trace_logger.record_action(
                timestep=timed_action.get_timestep(),
                queue_size=current_queue_size,
                command_action=command_action,
                performed_action=_performed_action,
            )

        if verbose:
            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if self.trace_logger is not None:
                self.trace_logger.record_observation(
                    raw_observation=raw_observation,
                    timestep=observation.get_timestep(),
                    queue_size=current_queue_size,
                    must_go=observation.must_go,
                )

            self.update_post_release_home_state(raw_observation)

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()

            if (
                self.post_release_home_manual_requested.is_set()
                and not self.post_release_home_in_progress.is_set()
            ):
                if self.config.post_release_home_record_demo_enabled:
                    self.record_assisted_post_release_home_demo()
                else:
                    self.run_post_release_home_sequence(trigger_event="manual_home_trigger")
                break

            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.info(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.start_post_release_home_manual_listener()
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    async_client()  # run the client
