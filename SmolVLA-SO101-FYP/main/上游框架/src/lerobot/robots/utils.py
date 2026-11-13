# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import logging
from pprint import pformat

from lerobot.robots import RobotConfig

from .robot import Robot


def make_robot_from_config(config: RobotConfig) -> Robot:
    if config.type == "koch_follower":
        from .koch_follower import KochFollower

        return KochFollower(config)
    elif config.type == "so100_follower":
        from .so100_follower import SO100Follower

        return SO100Follower(config)
    elif config.type == "so100_follower_end_effector":
        from .so100_follower import SO100FollowerEndEffector

        return SO100FollowerEndEffector(config)
    elif config.type == "so101_follower":
        from .so101_follower import SO101Follower

        return SO101Follower(config)
    elif config.type == "lekiwi":
        from .lekiwi import LeKiwi

        return LeKiwi(config)
    elif config.type == "stretch3":
        from .stretch3 import Stretch3Robot

        return Stretch3Robot(config)
    elif config.type == "viperx":
        from .viperx import ViperX

        return ViperX(config)
    elif config.type == "hope_jr_hand":
        from .hope_jr import HopeJrHand

        return HopeJrHand(config)
    elif config.type == "hope_jr_arm":
        from .hope_jr import HopeJrArm

        return HopeJrArm(config)
    elif config.type == "bi_so100_follower":
        from .bi_so100_follower import BiSO100Follower

        return BiSO100Follower(config)
    elif config.type == "mock_robot":
        from tests.mocks.mock_robot import MockRobot

        return MockRobot(config)
    else:
        raise ValueError(config.type)


def ensure_safe_goal_position(
    goal_present_pos: dict[str, tuple[float, float]], max_relative_target: int | float | dict[float]
) -> dict[str, float]:
    """Caps relative action target magnitude for safety."""

    if isinstance(max_relative_target, (int, float)):
        diff_cap = dict.fromkeys(goal_present_pos, max_relative_target)
    elif isinstance(max_relative_target, dict):
        if not set(goal_present_pos) == set(max_relative_target):
            raise ValueError("max_relative_target keys must match those of goal_present_pos.")
        diff_cap = max_relative_target
    else:
        raise TypeError(max_relative_target)

    warnings_dict = {}
    safe_goal_positions = {}
    for key, (goal_pos, present_pos) in goal_present_pos.items():
        diff = goal_pos - present_pos
        max_diff = diff_cap[key]
        safe_diff = min(diff, max_diff)
        safe_diff = max(safe_diff, -max_diff)
        safe_goal_pos = present_pos + safe_diff
        safe_goal_positions[key] = safe_goal_pos
        if abs(safe_goal_pos - goal_pos) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": goal_pos,
                "safe goal_pos": safe_goal_pos,
            }

    if warnings_dict:
        logging.debug(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"{pformat(warnings_dict, indent=4)}"
        )

    return safe_goal_positions


def clamp_goal_position(
    goal_pos: dict[str, float], position_limits: dict[str, list[float] | tuple[float, float]]
) -> dict[str, float]:
    """Clamps absolute joint targets to software-defined safe limits."""

    warnings_dict = {}
    safe_goal_positions = goal_pos.copy()

    for key, limits in position_limits.items():
        if key not in safe_goal_positions:
            continue

        if len(limits) != 2:
            raise ValueError(f"Position limits for '{key}' must have exactly 2 values.")

        min_pos, max_pos = limits
        if min_pos > max_pos:
            raise ValueError(f"Position limits for '{key}' are invalid: min > max.")

        original_goal = safe_goal_positions[key]
        safe_goal = min(max(original_goal, min_pos), max_pos)
        safe_goal_positions[key] = safe_goal

        if abs(safe_goal - original_goal) > 1e-4:
            warnings_dict[key] = {
                "original goal_pos": original_goal,
                "safe goal_pos": safe_goal,
                "min limit": min_pos,
                "max limit": max_pos,
            }

    if warnings_dict:
        logging.warning(
            "Absolute goal position had to be clamped to stay inside software safety limits.\n"
            f"{pformat(warnings_dict, indent=4)}"
        )

    return safe_goal_positions


def hold_position_if_outside_limits(
    goal_pos: dict[str, float],
    present_pos: dict[str, float],
    position_limits: dict[str, list[float] | tuple[float, float]],
    requested_goal_pos: dict[str, float] | None = None,
    hold_positions: dict[str, float] | None = None,
) -> dict[str, float]:
    """Holds joints at their current position when the requested goal goes outside software safety limits."""

    if requested_goal_pos is None:
        requested_goal_pos = goal_pos
    if hold_positions is None:
        hold_positions = {}

    warnings_dict = {}
    safe_goal_positions = goal_pos.copy()

    for key, limits in position_limits.items():
        if key not in safe_goal_positions or key not in present_pos or key not in requested_goal_pos:
            continue

        if len(limits) != 2:
            raise ValueError(f"Position limits for '{key}' must have exactly 2 values.")

        min_pos, max_pos = limits
        if min_pos > max_pos:
            raise ValueError(f"Position limits for '{key}' are invalid: min > max.")

        requested_goal = requested_goal_pos[key]
        if min_pos <= requested_goal <= max_pos:
            hold_positions.pop(key, None)
            continue

        is_new_hold = key not in hold_positions
        held_goal = hold_positions.setdefault(key, present_pos[key])
        safe_goal_positions[key] = held_goal
        if is_new_hold:
            warnings_dict[key] = {
                "requested goal_pos": requested_goal,
                "limited goal_pos": goal_pos[key],
                "held goal_pos": held_goal,
                "min limit": min_pos,
                "max limit": max_pos,
            }

    if warnings_dict:
        logging.warning(
            "Goal position moved outside software safety limits, holding current position instead.\n"
            f"{pformat(warnings_dict, indent=4)}"
        )

    return safe_goal_positions
