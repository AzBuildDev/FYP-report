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

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from lerobot.robots.config import RobotConfig
from lerobot.scripts.server.constants import (
    DEFAULT_FPS,
    DEFAULT_INFERENCE_LATENCY,
    DEFAULT_OBS_QUEUE_TIMEOUT,
)

# Aggregate function registry for CLI usage
AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


def get_aggregate_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get aggregate function by name from registry."""
    if name not in AGGREGATE_FUNCTIONS:
        available = list(AGGREGATE_FUNCTIONS.keys())
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


@dataclass
class PolicyServerConfig:
    """Configuration for PolicyServer.

    This class defines all configurable parameters for the PolicyServer,
    including networking settings and action chunking specifications.
    """

    # Networking configuration
    host: str = field(default="localhost", metadata={"help": "Host address to bind the server to"})
    port: int = field(default=8080, metadata={"help": "Port number to bind the server to"})

    # Timing configuration
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})
    inference_latency: float = field(
        default=DEFAULT_INFERENCE_LATENCY, metadata={"help": "Target inference latency in seconds"}
    )

    obs_queue_timeout: float = field(
        default=DEFAULT_OBS_QUEUE_TIMEOUT, metadata={"help": "Timeout for observation queue in seconds"}
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {self.port}")

        if self.environment_dt <= 0:
            raise ValueError(f"environment_dt must be positive, got {self.environment_dt}")

        if self.inference_latency < 0:
            raise ValueError(f"inference_latency must be non-negative, got {self.inference_latency}")

        if self.obs_queue_timeout < 0:
            raise ValueError(f"obs_queue_timeout must be non-negative, got {self.obs_queue_timeout}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PolicyServerConfig":
        """Create a PolicyServerConfig from a dictionary."""
        return cls(**config_dict)

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "fps": self.fps,
            "environment_dt": self.environment_dt,
            "inference_latency": self.inference_latency,
        }


@dataclass
class RobotClientConfig:
    """Configuration for RobotClient.

    This class defines all configurable parameters for the RobotClient,
    including network connection, policy settings, and control behavior.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration (for CLI usage - robot instance will be created from this)
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Policies typically output K actions at max, but we can use less to avoid wasting bandwidth (as actions
    # would be aggregated on the client side anyway, depending on the value of `chunk_size_threshold`)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk"})

    # Task instruction for the robot to execute (e.g., 'fold my tshirt')
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})

    # Control behavior configuration
    chunk_size_threshold: float = field(default=0.5, metadata={"help": "Threshold for chunk size control"})
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Frames per second"})

    # Aggregate function configuration (CLI-compatible)
    aggregate_fn_name: str = field(
        default="weighted_average",
        metadata={"help": f"Name of aggregate function to use. Options: {list(AGGREGATE_FUNCTIONS.keys())}"},
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size"}
    )

    # Verification configuration
    verify_robot_cameras: bool = field(
        default=True, metadata={"help": "Verify that the robot cameras match the policy cameras"}
    )

    # Optional per-trial trace logging. This only records observations/actions for later analysis and does
    # not change the robot control path.
    trace_enabled: bool = field(default=False, metadata={"help": "Write per-trial robot trace CSV"})
    trace_output_dir: str = field(
        default="inference_traces", metadata={"help": "Directory where trace CSV files are written"}
    )
    trace_trial_id: str = field(default="", metadata={"help": "Human-readable trial id for trace naming"})

    # Optional 200k-oriented supervisor. It never opens the gripper by itself; it only returns home after
    # observing that the policy has already closed and then reopened the gripper.
    post_release_home_enabled: bool = field(
        default=False, metadata={"help": "Return home after detecting policy-driven gripper release"}
    )
    post_release_home_auto_trigger_enabled: bool = field(
        default=True, metadata={"help": "Enable automatic post-release home trigger logic"}
    )
    post_release_home_manual_trigger_enabled: bool = field(
        default=False, metadata={"help": "Enable ENTER-triggered post-release home"}
    )
    post_release_home_record_demo_enabled: bool = field(
        default=False, metadata={"help": "After ENTER, record assisted home trajectory instead of commanding home"}
    )
    post_release_home_record_demo_s: float = field(
        default=15.0, metadata={"help": "Seconds to record assisted home trajectory after ENTER"}
    )
    post_release_home_record_demo_dt_s: float = field(
        default=0.1, metadata={"help": "Sampling interval for assisted home trajectory recording"}
    )
    post_release_home_closed_threshold: float = field(
        default=25.0, metadata={"help": "Gripper position treated as closed/holding"}
    )
    post_release_home_open_threshold: float = field(
        default=20.0, metadata={"help": "Gripper position treated as released/open"}
    )
    post_release_home_open_margin: float = field(
        default=5.0, metadata={"help": "Allowed margin above the captured startup gripper-open position"}
    )
    post_release_home_release_drop_threshold: float = field(
        default=35.0, metadata={"help": "Minimum gripper drop from observed closed max to count as released"}
    )
    post_release_home_release_gripper_min: float = field(
        default=-1.0, metadata={"help": "Minimum gripper position for explicit release range; negative disables"}
    )
    post_release_home_release_gripper_max: float = field(
        default=-1.0, metadata={"help": "Maximum gripper position for explicit release range; negative disables"}
    )
    post_release_home_min_open_observations: int = field(
        default=3, metadata={"help": "Consecutive open observations required before returning home"}
    )
    post_release_home_min_elapsed_s: float = field(
        default=20.0, metadata={"help": "Minimum runtime before post-release home can trigger"}
    )
    post_release_home_initial_capture_s: float = field(
        default=8.0, metadata={"help": "Capture the startup pose during this initial time window"}
    )
    post_release_home_release_dwell_s: float = field(
        default=3.0, metadata={"help": "Seconds to wait after target-area release before returning home"}
    )
    post_release_home_target_dwell_s: float = field(
        default=0.0,
        metadata={"help": "Seconds to wait in target pose before returning home, ignoring gripper state"},
    )
    post_release_home_target_pan_min: float = field(
        default=-180.0, metadata={"help": "Minimum shoulder_pan position for target-area release"}
    )
    post_release_home_target_pan_max: float = field(
        default=180.0, metadata={"help": "Maximum shoulder_pan position for target-area release"}
    )
    post_release_home_target_lift_min: float = field(
        default=-180.0, metadata={"help": "Minimum shoulder_lift position for target-area release"}
    )
    post_release_home_target_lift_max: float = field(
        default=180.0, metadata={"help": "Maximum shoulder_lift position for target-area release"}
    )
    post_release_home_target_elbow_min: float = field(
        default=-180.0, metadata={"help": "Minimum elbow_flex position for target-area release"}
    )
    post_release_home_target_elbow_max: float = field(
        default=180.0, metadata={"help": "Maximum elbow_flex position for target-area release"}
    )
    post_release_home_target_wrist_min: float = field(
        default=-180.0, metadata={"help": "Minimum wrist_flex position for target-area release"}
    )
    post_release_home_target_wrist_max: float = field(
        default=180.0, metadata={"help": "Maximum wrist_flex position for target-area release"}
    )
    post_release_home_steps: int = field(default=80, metadata={"help": "Number of interpolation steps to home"})
    post_release_home_sleep_s: float = field(default=0.05, metadata={"help": "Sleep between home steps"})
    post_release_home_prelift_enabled: bool = field(
        default=False, metadata={"help": "Move only shoulder_lift before returning home"}
    )
    post_release_home_prelift_delta: float = field(
        default=-25.0, metadata={"help": "Relative shoulder_lift movement before returning home"}
    )
    post_release_home_prelift_steps: int = field(
        default=30, metadata={"help": "Number of interpolation steps for shoulder_lift pre-lift"}
    )
    post_release_home_final_close_enabled: bool = field(
        default=False, metadata={"help": "Close gripper after returning home"}
    )
    post_release_home_final_gripper: float = field(
        default=55.0, metadata={"help": "Final gripper position after returning home"}
    )
    post_release_home_final_gripper_steps: int = field(
        default=25, metadata={"help": "Number of interpolation steps for final gripper close"}
    )
    post_release_home_replay_trace_path: str = field(
        default="", metadata={"help": "Trace CSV to replay for post-release home"}
    )
    post_release_home_replay_start_s: float = field(
        default=0.0, metadata={"help": "Start elapsed_s in replay trace"}
    )
    post_release_home_replay_end_s: float = field(
        default=0.0, metadata={"help": "End elapsed_s in replay trace; <= start means until end"}
    )
    post_release_home_replay_steps_per_waypoint: int = field(
        default=4, metadata={"help": "Interpolation steps per replay waypoint"}
    )
    post_release_home_replay_bridge_steps: int = field(
        default=0, metadata={"help": "Interpolation steps from current pose to first replay waypoint"}
    )
    post_release_home_replay_bridge_sleep_s: float = field(
        default=0.05, metadata={"help": "Sleep between replay bridge interpolation steps"}
    )
    post_release_home_replay_sleep_s: float = field(
        default=0.05, metadata={"help": "Sleep between replay interpolation steps"}
    )
    post_release_home_final_hold_steps: int = field(
        default=15, metadata={"help": "Number of final repeated home commands"}
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step, in seconds"""
        return 1 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")

        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")

        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")

        if not self.policy_device:
            raise ValueError("policy_device cannot be empty")

        if self.chunk_size_threshold < 0 or self.chunk_size_threshold > 1:
            raise ValueError(f"chunk_size_threshold must be between 0 and 1, got {self.chunk_size_threshold}")

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")

        if self.post_release_home_min_open_observations <= 0:
            raise ValueError("post_release_home_min_open_observations must be positive")

        if self.post_release_home_steps <= 0:
            raise ValueError("post_release_home_steps must be positive")

        if self.post_release_home_prelift_steps <= 0:
            raise ValueError("post_release_home_prelift_steps must be positive")

        if self.post_release_home_final_gripper_steps <= 0:
            raise ValueError("post_release_home_final_gripper_steps must be positive")

        if self.post_release_home_replay_steps_per_waypoint <= 0:
            raise ValueError("post_release_home_replay_steps_per_waypoint must be positive")

        if self.post_release_home_replay_bridge_steps < 0:
            raise ValueError("post_release_home_replay_bridge_steps must be non-negative")

        if self.post_release_home_replay_bridge_sleep_s < 0:
            raise ValueError("post_release_home_replay_bridge_sleep_s must be non-negative")

        if self.post_release_home_replay_sleep_s < 0:
            raise ValueError("post_release_home_replay_sleep_s must be non-negative")

        if self.post_release_home_sleep_s < 0:
            raise ValueError("post_release_home_sleep_s must be non-negative")

        if self.post_release_home_release_dwell_s < 0:
            raise ValueError("post_release_home_release_dwell_s must be non-negative")

        if self.post_release_home_final_hold_steps < 0:
            raise ValueError("post_release_home_final_hold_steps must be non-negative")

        if self.post_release_home_record_demo_s <= 0:
            raise ValueError("post_release_home_record_demo_s must be positive")

        if self.post_release_home_record_demo_dt_s <= 0:
            raise ValueError("post_release_home_record_demo_dt_s must be positive")

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RobotClientConfig":
        """Create a RobotClientConfig from a dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        return {
            "server_address": self.server_address,
            "policy_type": self.policy_type,
            "pretrained_name_or_path": self.pretrained_name_or_path,
            "policy_device": self.policy_device,
            "chunk_size_threshold": self.chunk_size_threshold,
            "fps": self.fps,
            "actions_per_chunk": self.actions_per_chunk,
            "task": self.task,
            "debug_visualize_queue_size": self.debug_visualize_queue_size,
            "aggregate_fn_name": self.aggregate_fn_name,
            "trace_enabled": self.trace_enabled,
            "trace_output_dir": self.trace_output_dir,
            "trace_trial_id": self.trace_trial_id,
            "post_release_home_enabled": self.post_release_home_enabled,
            "post_release_home_auto_trigger_enabled": self.post_release_home_auto_trigger_enabled,
            "post_release_home_manual_trigger_enabled": self.post_release_home_manual_trigger_enabled,
            "post_release_home_record_demo_enabled": self.post_release_home_record_demo_enabled,
            "post_release_home_record_demo_s": self.post_release_home_record_demo_s,
            "post_release_home_record_demo_dt_s": self.post_release_home_record_demo_dt_s,
            "post_release_home_closed_threshold": self.post_release_home_closed_threshold,
            "post_release_home_open_threshold": self.post_release_home_open_threshold,
            "post_release_home_open_margin": self.post_release_home_open_margin,
            "post_release_home_release_drop_threshold": self.post_release_home_release_drop_threshold,
            "post_release_home_release_gripper_min": self.post_release_home_release_gripper_min,
            "post_release_home_release_gripper_max": self.post_release_home_release_gripper_max,
            "post_release_home_min_open_observations": self.post_release_home_min_open_observations,
            "post_release_home_min_elapsed_s": self.post_release_home_min_elapsed_s,
            "post_release_home_initial_capture_s": self.post_release_home_initial_capture_s,
            "post_release_home_release_dwell_s": self.post_release_home_release_dwell_s,
            "post_release_home_target_dwell_s": self.post_release_home_target_dwell_s,
            "post_release_home_target_pan_min": self.post_release_home_target_pan_min,
            "post_release_home_target_pan_max": self.post_release_home_target_pan_max,
            "post_release_home_target_lift_min": self.post_release_home_target_lift_min,
            "post_release_home_target_lift_max": self.post_release_home_target_lift_max,
            "post_release_home_target_elbow_min": self.post_release_home_target_elbow_min,
            "post_release_home_target_elbow_max": self.post_release_home_target_elbow_max,
            "post_release_home_target_wrist_min": self.post_release_home_target_wrist_min,
            "post_release_home_target_wrist_max": self.post_release_home_target_wrist_max,
            "post_release_home_steps": self.post_release_home_steps,
            "post_release_home_sleep_s": self.post_release_home_sleep_s,
            "post_release_home_prelift_enabled": self.post_release_home_prelift_enabled,
            "post_release_home_prelift_delta": self.post_release_home_prelift_delta,
            "post_release_home_prelift_steps": self.post_release_home_prelift_steps,
            "post_release_home_final_close_enabled": self.post_release_home_final_close_enabled,
            "post_release_home_final_gripper": self.post_release_home_final_gripper,
            "post_release_home_final_gripper_steps": self.post_release_home_final_gripper_steps,
            "post_release_home_replay_trace_path": self.post_release_home_replay_trace_path,
            "post_release_home_replay_start_s": self.post_release_home_replay_start_s,
            "post_release_home_replay_end_s": self.post_release_home_replay_end_s,
            "post_release_home_replay_steps_per_waypoint": self.post_release_home_replay_steps_per_waypoint,
            "post_release_home_replay_bridge_steps": self.post_release_home_replay_bridge_steps,
            "post_release_home_replay_bridge_sleep_s": self.post_release_home_replay_bridge_sleep_s,
            "post_release_home_replay_sleep_s": self.post_release_home_replay_sleep_s,
            "post_release_home_final_hold_steps": self.post_release_home_final_hold_steps,
        }
