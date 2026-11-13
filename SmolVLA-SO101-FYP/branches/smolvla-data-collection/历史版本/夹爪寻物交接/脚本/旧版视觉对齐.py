#!/usr/bin/env python
"""YOLO-guided SO101 overhead alignment.

Stage 1 only:
- detect red_cube / green_cylinder from the fixed overhead camera;
- map target mask center pixels to a calibrated safe-plane joint target;
- move the follower to hover above the target at the recorded safe pose;
- do not descend, grip, release, or run ACT.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from 旧版夹爪视觉伺服 import (
        broyden_update,
        damped_least_squares_step,
        ensure_minimum_effective_step,
        estimate_image_jacobian,
        jacobian_is_usable,
        robust_reference_cluster,
    )
except ModuleNotFoundError:
    from .旧版夹爪视觉伺服 import (
        broyden_update,
        damped_least_squares_step,
        ensure_minimum_effective_step,
        estimate_image_jacobian,
        jacobian_is_usable,
        robust_reference_cluster,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "vision" / "models" / "overhead_yolo_seg.pt"
DEFAULT_GRIPPER_MODEL = PROJECT_ROOT / "vision" / "models" / "gripper_yolo_seg.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "vision" / "config" / "桌面相机安全平面标定.json"
DEFAULT_GRIPPER_MASK_SHAPE_PROFILE = PROJECT_ROOT / "vision" / "config" / "夹爪目标轮廓配置.json"
WINDOW_NAME = "SO101 YOLO guided align"
MOTOR_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
DEFAULT_ORIENTATION_JOINTS = [
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]
DEFAULT_POSE_JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]
_MODEL_PREDICT_LOCKS: dict[int, threading.RLock] = {}
_MODEL_PREDICT_LOCKS_GUARD = threading.Lock()
DEFAULT_ORIENTATION_SOURCE_CSV = PROJECT_ROOT / "vision" / "data" / "metadata" / "captures.csv"
TOP_FACE_METHOD = "contour_apex_parallelogram_v1"
MASK_SHAPE_FEATURE_KEYS = (
    "mask_area_px",
    "mask_sqrt_area_px",
    "mask_bbox_w_px",
    "mask_bbox_h_px",
    "mask_bbox_aspect",
    "mask_extent",
    "mask_solidity",
    "mask_circularity",
    "mask_rect_aspect",
    "mask_angle_deg",
    "mask_major_axis_px",
    "mask_minor_axis_px",
    "mask_axis_ratio",
)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (40, 160, 255),
        (40, 220, 80),
        (255, 190, 40),
        (230, 80, 220),
        (80, 220, 220),
        (220, 80, 80),
    ]
    return palette[class_id % len(palette)]


def draw_text(image, lines: list[str], origin=(12, 26), line_height=24) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
        y += line_height


def normalize_key(key: int) -> int:
    if ord("A") <= key <= ord("Z"):
        return key + 32
    return key


def handle_target_nudge(key: int, args) -> str | None:
    step = float(args.target_nudge_px)
    if key == ord("j"):
        args.target_offset_u -= step
    elif key == ord("l"):
        args.target_offset_u += step
    elif key == ord("i"):
        args.target_offset_v -= step
    elif key == ord("k"):
        args.target_offset_v += step
    elif key == ord("o"):
        args.target_offset_u = 0.0
        args.target_offset_v = 0.0
    else:
        return None
    return f"target offset u={args.target_offset_u:.1f}, v={args.target_offset_v:.1f}"


class PreviewWindow:
    def __init__(self, title: str):
        self.title = title
        self.backend = "opencv"
        self.root = None
        self.label = None
        self.photo = None
        self.tk = None
        self.closed = False
        self.last_key = -1

        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, np.zeros((1, 1, 3), dtype=np.uint8))
            cv2.waitKey(1)
        except Exception as exc:  # noqa: BLE001
            self.backend = "tkinter"
            try:
                cv2.destroyWindow(title)
            except Exception:
                pass
            print(f"OpenCV preview unavailable, falling back to Tkinter: {exc}")
            self._init_tkinter()

    def _init_tkinter(self) -> None:
        try:
            import tkinter as tk
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Neither OpenCV HighGUI nor Tkinter preview is available.") from exc
        self.tk = tk
        self.root = tk.Tk()
        self.root.title(self.title)
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.bind("<Key>", self._on_key)

    def _close(self) -> None:
        self.closed = True
        self.last_key = ord("q")

    def _on_key(self, event) -> None:
        if event.keysym == "space":
            self.last_key = 32
        elif event.keysym == "BackSpace":
            self.last_key = 8
        elif event.char:
            self.last_key = ord(event.char.lower())

    def show(self, image) -> int:
        if self.backend == "opencv":
            cv2.imshow(self.title, image)
            return cv2.waitKey(1) & 0xFF

        if self.closed:
            return ord("q")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        ppm = f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes()
        self.photo = self.tk.PhotoImage(data=ppm, format="PPM")
        self.label.configure(image=self.photo)
        self.root.update_idletasks()
        self.root.update()
        key = self.last_key
        self.last_key = -1
        return key

    def close(self) -> None:
        if self.backend == "opencv":
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass
            return
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass


@dataclass
class TopFaceEstimate:
    center_uv: tuple[float, float]
    quad_uv: list[tuple[float, float]] | None = None
    method: str = TOP_FACE_METHOD


@dataclass
class TargetDetection:
    class_id: int
    class_name: str
    confidence: float
    center_uv: tuple[float, float]
    xyxy: list[float]
    point_kind: str = "mask_centroid"
    mask_centroid_uv: tuple[float, float] | None = None
    mask: Any | None = None
    top_face_quad_uv: list[tuple[float, float]] | None = None
    top_face_method: str | None = None


@dataclass
class CalibrationState:
    points: list[dict[str, Any]] = field(default_factory=list)
    safe_pose: dict[str, float] | None = None
    homography: list[list[float]] | None = None
    pose_model: dict[str, list[float]] = field(default_factory=dict)
    pose_model_features: list[str] = field(default_factory=list)
    pose_model_point_kind: str | None = None
    pose_model_target_class: str | None = None
    pose_model_target_point_params: dict[str, Any] | None = None
    config_target_class: str | None = None
    config_active_target_point_kind: str | None = None
    config_active_target_point_params: dict[str, Any] | None = None
    pose_limits: dict[str, list[float]] = field(default_factory=dict)
    orientation_model: dict[str, list[float]] = field(default_factory=dict)
    joint_limits: dict[str, list[float]] = field(default_factory=dict)
    orientation_limits: dict[str, list[float]] = field(default_factory=dict)
    hover_height_samples: list[dict[str, Any]] = field(default_factory=list)
    hover_height_model: dict[str, list[float]] = field(default_factory=dict)
    hover_height_model_features: list[str] = field(default_factory=list)
    hover_height_joints: list[str] = field(default_factory=list)
    hover_height_input_joints: list[str] = field(default_factory=list)
    hover_height_limits: dict[str, list[float]] = field(default_factory=dict)
    hover_height_summary: dict[str, Any] = field(default_factory=dict)
    gripper_correction_samples: list[dict[str, Any]] = field(default_factory=list)
    gripper_motion_samples: list[dict[str, Any]] = field(default_factory=list)
    gripper_primitive_samples: list[dict[str, Any]] = field(default_factory=list)
    gripper_learned_param_model: dict[str, Any] = field(default_factory=dict)
    coarse_pose_bias_model: dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinePoseAnchor:
    xy: dict[str, float]
    pose: dict[str, float]
    hover_model_pose: dict[str, float]
    orientation_model_pose: dict[str, float]


@dataclass
class GripperServoObservation:
    detection: TargetDetection | None
    ref_uv: tuple[float, float] | None
    error_uv: tuple[float, float]
    dist_px: float
    ref_source: str
    sample_count: int
    jitter_px: float
    status: str
    target_area_px: float = 0.0
    mask_features: dict[str, float] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "align"), required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--gripper-model", "--gripper_model", default=str(DEFAULT_GRIPPER_MODEL))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--target-class", "--target_class", default="red_cube")
    parser.add_argument("--camera-index", "--camera_index", type=int, default=2)
    parser.add_argument("--gripper-camera-index", "--gripper_camera_index", type=int, default=0)
    parser.add_argument("--gripper-rotate", "--gripper_rotate", choices=("none", "90", "180", "270"), default="none")
    parser.add_argument("--gripper-flip", "--gripper_flip", choices=("none", "h", "v", "both"), default="none")
    parser.add_argument("--camera-width", "--camera_width", type=int, default=640)
    parser.add_argument("--camera-height", "--camera_height", type=int, default=480)
    parser.add_argument("--camera-fps", "--camera_fps", type=int, default=30)
    parser.add_argument("--camera-backend", "--camera_backend", choices=("dshow", "msmf", "any"), default="dshow")
    parser.add_argument("--duplicate-camera-check", "--duplicate_camera_check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duplicate-camera-threshold", "--duplicate_camera_threshold", type=float, default=2.0)
    parser.add_argument("--duplicate-camera-frames", "--duplicate_camera_frames", type=int, default=3)
    parser.add_argument(
        "--duplicate-camera-auto-reopen",
        "--duplicate_camera_auto_reopen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically reopen both cameras when gripper frames duplicate overhead frames. Disabled by default because Windows camera indexes can bind to the wrong device after a quick reopen.",
    )
    parser.add_argument("--follower-port", "--follower_port", default="COM11")
    parser.add_argument("--leader-port", "--leader_port", default="COM12")
    parser.add_argument("--follower-id", "--follower_id", default="my_awesome_follower_arm")
    parser.add_argument("--leader-id", "--leader_id", default="my_awesome_leader_arm")
    parser.add_argument("--map-joints", "--map_joints", nargs=2, default=["shoulder_pan.pos", "shoulder_lift.pos"])
    parser.add_argument("--pose-joints", "--pose_joints", nargs="*", default=DEFAULT_POSE_JOINTS)
    parser.add_argument("--pose-margin", "--pose_margin", type=float, default=5.0)
    parser.add_argument("--max-relative-target", "--max_relative_target", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--target-point", "--target_point", choices=("auto", "mask_centroid", "top_face"), default="auto")
    parser.add_argument(
        "--share-calibration-across-classes",
        "--share_calibration_across_classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse mask-centroid correction points across object classes because the pixel-to-hover-pose map is workspace geometry.",
    )
    parser.add_argument("--cube-top-face-fraction", "--cube_top_face_fraction", type=float, default=0.40)
    parser.add_argument("--cube-top-face-center-y-frac", "--cube_top_face_center_y_frac", type=float, default=0.32)
    parser.add_argument("--cube-top-face-band-frac", "--cube_top_face_band_frac", type=float, default=0.18)
    parser.add_argument("--cube-top-contour-epsilon-frac", "--cube_top_contour_epsilon_frac", type=float, default=0.018)
    parser.add_argument("--cube-top-min-edge-px", "--cube_top_min_edge_px", type=float, default=12.0)
    parser.add_argument("--top-face-min-pixels", "--top_face_min_pixels", type=int, default=24)
    parser.add_argument("--target-offset-u", "--target_offset_u", type=float, default=0.0)
    parser.add_argument("--target-offset-v", "--target_offset_v", type=float, default=0.0)
    parser.add_argument("--target-nudge-px", "--target_nudge_px", type=float, default=3.0)
    parser.add_argument("--allow-legacy-top-face-points", "--allow_legacy_top_face_points", action="store_true")
    parser.add_argument("--workspace-margin", "--workspace_margin", type=float, default=4.0)
    parser.add_argument("--orientation-joints", "--orientation_joints", nargs="*", default=DEFAULT_ORIENTATION_JOINTS)
    parser.add_argument("--orientation-margin", "--orientation_margin", type=float, default=6.0)
    parser.add_argument("--orientation-source-csv", "--orientation_source_csv", default=str(DEFAULT_ORIENTATION_SOURCE_CSV))
    parser.add_argument("--disable-csv-orientation", "--disable_csv_orientation", action="store_true")
    parser.add_argument("--move-steps", "--move_steps", type=int, default=90)
    parser.add_argument("--min-move-steps", "--min_move_steps", type=int, default=12)
    parser.add_argument("--max-joint-step", "--max_joint_step", type=float, default=3.0)
    parser.add_argument("--move-sleep-s", "--move_sleep_s", type=float, default=0.015)
    parser.add_argument("--joint-speed-deg-s", "--joint_speed_deg_s", type=float, default=24.0)
    parser.add_argument("--max-move-sleep-s", "--max_move_sleep_s", type=float, default=0.10)
    parser.add_argument("--gripper-refine-joint-speed-deg-s", "--gripper_refine_joint_speed_deg_s", type=float, default=75.0)
    parser.add_argument("--gripper-refine-min-move-steps", "--gripper_refine_min_move_steps", type=int, default=4)
    parser.add_argument("--gripper-refine-wait", "--gripper_refine_wait", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-refine-position-threshold", "--gripper_refine_position_threshold", type=float, default=0.80)
    parser.add_argument("--gripper-refine-wait-timeout-s", "--gripper_refine_wait_timeout_s", type=float, default=4.0)
    parser.add_argument("--gripper-refine-continue-threshold", "--gripper_refine_continue_threshold", type=float, default=1.2)
    parser.add_argument("--gripper-refine-hold-command-interval-s", "--gripper_refine_hold_command_interval_s", type=float, default=0.05)
    parser.add_argument("--gripper-refine-wait-joints", "--gripper_refine_wait_joints", nargs="*", default=None)
    parser.add_argument("--gripper-refine-stable-epsilon-deg", "--gripper_refine_stable_epsilon_deg", type=float, default=0.12)
    parser.add_argument("--gripper-refine-stable-samples", "--gripper_refine_stable_samples", type=int, default=4)
    parser.add_argument("--xy-hover-wait-timeout-s", "--xy_hover_wait_timeout_s", type=float, default=8.0)
    parser.add_argument("--xy-hover-stable-offset-deg", "--xy_hover_stable_offset_deg", type=float, default=6.0)
    parser.add_argument("--safe-settle-s", "--safe_settle_s", type=float, default=0.25)
    parser.add_argument("--camera-flush-frames", "--camera_flush_frames", type=int, default=3)
    parser.add_argument("--safe-position-threshold", "--safe_position_threshold", type=float, default=3.0)
    parser.add_argument("--safe-lift-position-threshold", "--safe_lift_position_threshold", type=float, default=5.0)
    parser.add_argument("--safe-lift-continue-threshold", "--safe_lift_continue_threshold", type=float, default=8.0)
    parser.add_argument("--full-safe-position-threshold", "--full_safe_position_threshold", type=float, default=5.0)
    parser.add_argument("--full-safe-continue-threshold", "--full_safe_continue_threshold", type=float, default=8.0)
    parser.add_argument("--safe-wait-timeout-s", "--safe_wait_timeout_s", type=float, default=5.0)
    parser.add_argument("--safe-progress-extra-timeout-s", "--safe_progress_extra_timeout_s", type=float, default=4.0)
    parser.add_argument("--safe-progress-grace-s", "--safe_progress_grace_s", type=float, default=1.0)
    parser.add_argument("--safe-progress-epsilon", "--safe_progress_epsilon", type=float, default=0.20)
    parser.add_argument("--safe-hold-command-interval-s", "--safe_hold_command_interval_s", type=float, default=0.05)
    parser.add_argument(
        "--safe-lift-joints",
        "--safe_lift_joints",
        nargs="*",
        default=["shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"],
    )
    parser.add_argument(
        "--safe-lift-wait-joints",
        "--safe_lift_wait_joints",
        nargs="*",
        default=["shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"],
    )
    parser.add_argument(
        "--full-safe-wait-joints",
        "--full_safe_wait_joints",
        nargs="*",
        default=["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"],
    )
    parser.add_argument(
        "--full-safe-must-settle-joints",
        "--full_safe_must_settle_joints",
        nargs="*",
        default=["shoulder_pan.pos"],
        help="Joints that must reach the full safe_pose threshold before the script may leave stage 1b.",
    )
    parser.add_argument("--safe-retry-count", "--safe_retry_count", type=int, default=1)
    parser.add_argument("--closed-loop-iters", "--closed_loop_iters", type=int, default=1)
    parser.add_argument("--joint-error-threshold", "--joint_error_threshold", type=float, default=1.5)
    parser.add_argument("--continue-if-safe-timeout", "--continue_if_safe_timeout", action="store_true")
    parser.add_argument("--pose-predictor", "--pose_predictor", choices=("local", "linear"), default="local")
    parser.add_argument("--alignment-pose-mode", "--alignment_pose_mode", choices=("auto", "full_pose", "xy_hover"), default="auto")
    parser.add_argument(
        "--hover-height-joints",
        "--hover_height_joints",
        nargs="*",
        default=["elbow_flex.pos", "wrist_flex.pos"],
    )
    parser.add_argument(
        "--hover-height-input-joints",
        "--hover_height_input_joints",
        nargs="*",
        default=["shoulder_lift.pos"],
        help="Joint inputs used to keep end-effector hover height. Default uses motor 2 only, then predicts motors 3/4.",
    )
    parser.add_argument("--hover-height-margin", "--hover_height_margin", type=float, default=6.0)
    parser.add_argument("--hover-height-record-hz", "--hover_height_record_hz", type=float, default=10.0)
    parser.add_argument("--hover-height-min-samples", "--hover_height_min_samples", type=int, default=5)
    parser.add_argument("--local-pose-k", "--local_pose_k", type=int, default=4)
    parser.add_argument("--local-pose-power", "--local_pose_power", type=float, default=2.0)
    parser.add_argument("--local-snap-px", "--local_snap_px", type=float, default=18.0)
    parser.add_argument("--local-radius-px", "--local_radius_px", type=float, default=180.0)
    parser.add_argument("--preview-inference-interval-s", "--preview_inference_interval_s", type=float, default=0.08)
    parser.add_argument("--manual-inference-interval-s", "--manual_inference_interval_s", type=float, default=0.08)
    parser.add_argument("--manual-teleop-fps", "--manual_teleop_fps", type=float, default=20.0)
    parser.add_argument("--enable-gripper-final-align", "--enable_gripper_final_align", action="store_true")
    parser.add_argument(
        "--gripper-final-iters",
        "--gripper_final_iters",
        type=int,
        default=24,
        help="Maximum gripper final-refine observations. 0 keeps refining until tolerance; a finite limit is safer.",
    )
    parser.add_argument("--gripper-center-tolerance-px", "--gripper_center_tolerance_px", type=float, default=8.0)
    parser.add_argument(
        "--gripper-reference-mode",
        "--gripper_reference_mode",
        choices=("recorded_robust", "recorded_median", "recorded_local", "frame_center"),
        default="recorded_robust",
        help="Final gripper-camera XY target. recorded_* uses gripper pixels saved with C; frame_center is the old behavior.",
    )
    parser.add_argument(
        "--require-gripper-detection-on-correction",
        "--require_gripper_detection_on_correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a gripper-camera detection when pressing C so final alignment has a recorded gripper reference.",
    )
    parser.add_argument("--gripper-reference-min-conf", "--gripper_reference_min_conf", type=float, default=0.85)
    parser.add_argument("--gripper-reference-max-outlier-px", "--gripper_reference_max_outlier_px", type=float, default=30.0)
    parser.add_argument("--gripper-reference-mad-scale", "--gripper_reference_mad_scale", type=float, default=2.5)
    parser.add_argument("--gripper-reference-min-radius-px", "--gripper_reference_min_radius_px", type=float, default=8.0)
    parser.add_argument("--gripper-reference-min-count", "--gripper_reference_min_count", type=int, default=1)
    parser.add_argument("--gripper-u-to-pan", "--gripper_u_to_pan", type=float, default=0.010)
    parser.add_argument("--gripper-v-to-lift", "--gripper_v_to_lift", type=float, default=0.010)
    parser.add_argument(
        "--gripper-servo-mode",
        "--gripper_servo_mode",
        choices=("verify_only", "debug_ibvs", "ibvs", "learned", "search", "jacobian", "fixed_gain"),
        default="ibvs",
    )
    parser.add_argument(
        "--allow-legacy-gripper-servo",
        "--allow_legacy_gripper_servo",
        action="store_true",
        help="Diagnostic escape hatch for the old search/jacobian/fixed_gain controllers.",
    )
    parser.add_argument("--gripper-servo-gain", "--gripper_servo_gain", type=float, default=0.35)
    parser.add_argument("--gripper-jacobian-probe-deg", "--gripper_jacobian_probe_deg", type=float, default=0.4)
    parser.add_argument("--gripper-jacobian-min-px-per-deg", "--gripper_jacobian_min_px_per_deg", type=float, default=1.0)
    parser.add_argument("--gripper-jacobian-max-condition", "--gripper_jacobian_max_condition", type=float, default=40.0)
    parser.add_argument(
        "--gripper-jacobian-refresh-iters",
        "--gripper_jacobian_refresh_iters",
        type=int,
        default=0,
        help="Refresh Jacobian every N refine iterations. 0 disables periodic refresh and only refreshes after divergence.",
    )
    parser.add_argument("--gripper-max-step-deg", "--gripper_max_step_deg", type=float, default=2.3)
    parser.add_argument("--gripper-min-step-deg", "--gripper_min_step_deg", type=float, default=0.75)
    parser.add_argument("--gripper-full-step-error-px", "--gripper_full_step_error_px", type=float, default=45.0)
    parser.add_argument("--gripper-refine-if-visible-after-full-pose-error", "--gripper_refine_if_visible_after_full_pose_error", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-refine-visible-max-joint-error-deg", "--gripper_refine_visible_max_joint_error_deg", type=float, default=2.5)
    parser.add_argument("--gripper-settle-s", "--gripper_settle_s", type=float, default=0.25)
    parser.add_argument("--gripper-observation-samples", "--gripper_observation_samples", type=int, default=5)
    parser.add_argument("--gripper-observation-interval-s", "--gripper_observation_interval_s", type=float, default=0.035)
    parser.add_argument("--gripper-observation-max-jitter-px", "--gripper_observation_max_jitter_px", type=float, default=6.0)
    parser.add_argument("--gripper-observation-retries", "--gripper_observation_retries", type=int, default=2)
    parser.add_argument("--gripper-mask-area-guard", "--gripper_mask_area_guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-max-area-growth", "--gripper_max_area_growth", type=float, default=0.30)
    parser.add_argument("--gripper-area-guard-min-px", "--gripper_area_guard_min_px", type=float, default=200.0)
    parser.add_argument(
        "--gripper-mask-shape-guard",
        "--gripper_mask_shape_guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reject gripper refine moves that improve center error by corrupting mask area/shape.",
    )
    parser.add_argument(
        "--gripper-mask-shape-profile",
        "--gripper_mask_shape_profile",
        default=str(DEFAULT_GRIPPER_MASK_SHAPE_PROFILE),
        help="Optional JSON profile from mask-shape demos; used as the target gripper mask shape.",
    )
    parser.add_argument("--gripper-mask-area-tolerance", "--gripper_mask_area_tolerance", type=float, default=0.28)
    parser.add_argument("--gripper-mask-axis-tolerance", "--gripper_mask_axis_tolerance", type=float, default=0.35)
    parser.add_argument("--gripper-mask-circularity-drop", "--gripper_mask_circularity_drop", type=float, default=0.12)
    parser.add_argument("--gripper-mask-solidity-drop", "--gripper_mask_solidity_drop", type=float, default=0.06)
    parser.add_argument("--gripper-mask-shape-worse-tolerance", "--gripper_mask_shape_worse_tolerance", type=float, default=0.20)
    parser.add_argument("--gripper-max-cumulative-lift-drift-deg", "--gripper_max_cumulative_lift_drift_deg", type=float, default=4.0)
    parser.add_argument(
        "--gripper-cumulative-lift-margin-deg",
        "--gripper_cumulative_lift_margin_deg",
        type=float,
        default=0.45,
        help="Keep this much shoulder_lift drift budget unused before sending gripper IBVS correction/probe commands.",
    )
    parser.add_argument(
        "--gripper-height-guard-action",
        "--gripper_height_guard_action",
        choices=("recover", "stop"),
        default="recover",
        help="Recover hover height and continue, or stop, when final gripper alignment exceeds cumulative lift drift.",
    )
    parser.add_argument("--gripper-hover-input-margin-deg", "--gripper_hover_input_margin_deg", type=float, default=0.5)
    parser.add_argument("--gripper-auto-flip-gains", "--gripper_auto_flip_gains", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-diverge-epsilon-px", "--gripper_diverge_epsilon_px", type=float, default=6.0)
    parser.add_argument("--gripper-camera-flush-frames", "--gripper_camera_flush_frames", type=int, default=5)
    parser.add_argument("--gripper-verify-move", "--gripper_verify_move", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-worse-epsilon-px", "--gripper_worse_epsilon_px", type=float, default=4.0)
    parser.add_argument("--gripper-rollback-on-worse", "--gripper_rollback_on_worse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-adaptive-min-scale", "--gripper_adaptive_min_scale", type=float, default=0.25)
    parser.add_argument("--gripper-adaptive-recover", "--gripper_adaptive_recover", type=float, default=1.12)
    parser.add_argument("--gripper-search-step-deg", "--gripper_search_step_deg", type=float, default=1.35)
    parser.add_argument("--gripper-search-min-step-deg", "--gripper_search_min_step_deg", type=float, default=0.25)
    parser.add_argument("--gripper-search-improve-px", "--gripper_search_improve_px", type=float, default=0.20)
    parser.add_argument("--gripper-search-diagonal", "--gripper_search_diagonal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-ibvs-probe-deg", "--gripper_ibvs_probe_deg", type=float, default=1.20)
    parser.add_argument("--gripper-ibvs-gain", "--gripper_ibvs_gain", type=float, default=0.72)
    parser.add_argument("--gripper-ibvs-damping", "--gripper_ibvs_damping", type=float, default=2.0)
    parser.add_argument("--gripper-debug-ibvs-probe-deg", "--gripper_debug_ibvs_probe_deg", type=float, default=1.2)
    parser.add_argument("--gripper-debug-ibvs-max-step-deg", "--gripper_debug_ibvs_max_step_deg", type=float, default=0.5)
    parser.add_argument("--gripper-debug-ibvs-min-step-deg", "--gripper_debug_ibvs_min_step_deg", type=float, default=0.2)
    parser.add_argument("--gripper-debug-ibvs-restore-threshold-deg", "--gripper_debug_ibvs_restore_threshold_deg", type=float, default=0.8)
    parser.add_argument("--gripper-debug-ibvs-restore-timeout-s", "--gripper_debug_ibvs_restore_timeout_s", type=float, default=8.0)
    parser.add_argument("--gripper-debug-ibvs-restore-resend-interval-s", "--gripper_debug_ibvs_restore_resend_interval_s", type=float, default=0.05)
    parser.add_argument("--gripper-debug-ibvs-min-actual-controlled-deg", "--gripper_debug_ibvs_min_actual_controlled_deg", type=float, default=0.25)
    parser.add_argument("--gripper-debug-ibvs-min-pixel-motion-px", "--gripper_debug_ibvs_min_pixel_motion_px", type=float, default=3.0)
    parser.add_argument("--gripper-debug-ibvs-max-jitter-px", "--gripper_debug_ibvs_max_jitter_px", type=float, default=6.0)
    parser.add_argument("--gripper-debug-ibvs-uncontrolled-warn-deg", "--gripper_debug_ibvs_uncontrolled_warn_deg", type=float, default=0.5)
    parser.add_argument("--gripper-debug-ibvs-max-uncontrolled-drift-deg", "--gripper_debug_ibvs_max_uncontrolled_drift_deg", type=float, default=1.0)
    parser.add_argument("--gripper-ibvs-max-condition", "--gripper_ibvs_max_condition", type=float, default=30.0)
    parser.add_argument("--gripper-ibvs-min-px-per-deg", "--gripper_ibvs_min_px_per_deg", type=float, default=1.0)
    parser.add_argument("--gripper-ibvs-max-rmse-px", "--gripper_ibvs_max_rmse_px", type=float, default=8.0)
    parser.add_argument("--gripper-ibvs-max-probes", "--gripper_ibvs_max_probes", type=int, default=6)
    parser.add_argument("--gripper-ibvs-max-worse-px", "--gripper_ibvs_max_worse_px", type=float, default=18.0)
    parser.add_argument("--gripper-ibvs-stagnation-px", "--gripper_ibvs_stagnation_px", type=float, default=1.0)
    parser.add_argument("--gripper-ibvs-max-stagnant", "--gripper_ibvs_max_stagnant", type=int, default=3)
    parser.add_argument("--gripper-ibvs-broyden-blend", "--gripper_ibvs_broyden_blend", type=float, default=0.65)
    parser.add_argument(
        "--gripper-ibvs-fresh-start",
        "--gripper_ibvs_fresh_start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start each final align by measuring a fresh local gripper-camera Jacobian instead of trusting saved history.",
    )
    parser.add_argument("--gripper-motion-local-joint-radius-deg", "--gripper_motion_local_joint_radius_deg", type=float, default=12.0)
    parser.add_argument("--gripper-motion-local-overhead-radius-px", "--gripper_motion_local_overhead_radius_px", type=float, default=140.0)
    parser.add_argument("--gripper-motion-max-px-per-deg", "--gripper_motion_max_px_per_deg", type=float, default=35.0)
    parser.add_argument("--gripper-motion-extra-joint-max-deg", "--gripper_motion_extra_joint_max_deg", type=float, default=0.85)
    parser.add_argument("--gripper-motion-extra-joint-ratio", "--gripper_motion_extra_joint_ratio", type=float, default=0.80)
    parser.add_argument("--gripper-motion-max-samples", "--gripper_motion_max_samples", type=int, default=400)
    parser.add_argument("--gripper-primitive-local-overhead-radius-px", "--gripper_primitive_local_overhead_radius_px", type=float, default=140.0)
    parser.add_argument("--gripper-primitive-max-px-per-unit", "--gripper_primitive_max_px_per_unit", type=float, default=80.0)
    parser.add_argument("--gripper-primitive-min-action", "--gripper_primitive_min_action", type=float, default=0.10)
    parser.add_argument("--gripper-primitive-max-samples", "--gripper_primitive_max_samples", type=int, default=400)
    parser.add_argument(
        "--gripper-min-actual-motion-deg",
        "--gripper_min_actual_motion_deg",
        type=float,
        default=0.12,
        help="Reject refine moves whose actual encoder delta is too small to trust as visual-servo feedback.",
    )
    parser.add_argument(
        "--gripper-min-actual-motion-fraction",
        "--gripper_min_actual_motion_fraction",
        type=float,
        default=0.18,
        help="Reject refine moves that execute too little of the commanded pan/lift direction.",
    )
    parser.add_argument(
        "--gripper-ibvs-probe-restore-worse-px",
        "--gripper_ibvs_probe_restore_worse_px",
        type=float,
        default=4.0,
        help="Restore the best pose after a probe worsens the gripper-camera error by more than this many pixels.",
    )
    parser.add_argument("--gripper-learned-k", "--gripper_learned_k", type=int, default=4)
    parser.add_argument("--gripper-learned-power", "--gripper_learned_power", type=float, default=2.0)
    parser.add_argument("--gripper-learned-min-error-px", "--gripper_learned_min_error_px", type=float, default=8.0)
    parser.add_argument("--gripper-learned-min-improve-px", "--gripper_learned_min_improve_px", type=float, default=1.0)
    parser.add_argument("--gripper-learned-accept-improve-px", "--gripper_learned_accept_improve_px", type=float, default=0.20)
    parser.add_argument("--gripper-learned-neutral-band-px", "--gripper_learned_neutral_band_px", type=float, default=1.5)
    parser.add_argument("--gripper-learned-reverse-worse-px", "--gripper_learned_reverse_worse_px", type=float, default=2.5)
    parser.add_argument("--gripper-learned-fallback-after-worse-px", "--gripper_learned_fallback_after_worse_px", type=float, default=0.5)
    parser.add_argument("--gripper-learned-rollback-worse-px", "--gripper_learned_rollback_worse_px", type=float, default=1.0)
    parser.add_argument("--gripper-learned-local-overhead-radius-px", "--gripper_learned_local_overhead_radius_px", type=float, default=70.0)
    parser.add_argument("--gripper-learned-local-joint-radius-deg", "--gripper_learned_local_joint_radius_deg", type=float, default=8.0)
    parser.add_argument("--gripper-learned-direction-gate-error-px", "--gripper_learned_direction_gate_error_px", type=float, default=25.0)
    parser.add_argument("--gripper-learned-direction-min-cos", "--gripper_learned_direction_min_cos", type=float, default=0.65)
    parser.add_argument("--gripper-learned-control-space", "--gripper_learned_control_space", choices=("xy", "full_pose"), default="full_pose")
    parser.add_argument("--gripper-learned-full-min-step-deg", "--gripper_learned_full_min_step_deg", type=float, default=0.50)
    parser.add_argument("--gripper-learned-full-max-step-deg", "--gripper_learned_full_max_step_deg", type=float, default=6.00)
    parser.add_argument("--gripper-learned-full-max-pan-deg", "--gripper_learned_full_max_pan_deg", type=float, default=2.50)
    parser.add_argument("--gripper-learned-full-max-lift-deg", "--gripper_learned_full_max_lift_deg", type=float, default=4.00)
    parser.add_argument("--gripper-learned-full-max-elbow-deg", "--gripper_learned_full_max_elbow_deg", type=float, default=4.00)
    parser.add_argument("--gripper-learned-full-max-wrist-flex-deg", "--gripper_learned_full_max_wrist_flex_deg", type=float, default=2.50)
    parser.add_argument("--gripper-learned-full-max-wrist-roll-deg", "--gripper_learned_full_max_wrist_roll_deg", type=float, default=2.00)
    parser.add_argument("--gripper-learned-full-min-local-samples", "--gripper_learned_full_min_local_samples", type=int, default=2)
    parser.add_argument("--gripper-learned-full-require-both-local", "--gripper_learned_full_require_both_local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-learned-full-allow-global-sample", "--gripper_learned_full_allow_global_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-learned-sample-roles", "--gripper_learned_sample_roles", nargs="*", default=[])
    parser.add_argument("--gripper-learned-autonomous-search", "--gripper_learned_autonomous_search", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-learned-search-step-deg", "--gripper_learned_search_step_deg", type=float, default=0.45)
    parser.add_argument("--gripper-learned-search-axis-step-deg", "--gripper_learned_search_axis_step_deg", type=float, default=0.25)
    parser.add_argument("--gripper-learned-search-max-candidates", "--gripper_learned_search_max_candidates", type=int, default=8)
    parser.add_argument("--gripper-learned-search-axis-first", "--gripper_learned_search_axis_first", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-learned-allow-axis-probes", "--gripper_learned_allow_axis_probes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-learned-rollback-outer-correction", "--gripper_learned_rollback_outer_correction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gripper-learned-search-axis-joints",
        "--gripper_learned_search_axis_joints",
        nargs="*",
        default=["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"],
    )
    parser.add_argument("--gripper-learned-clearance", "--gripper_learned_clearance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-learned-clearance-max-deg", "--gripper_learned_clearance_max_deg", type=float, default=2.00)
    parser.add_argument(
        "--gripper-learned-clearance-joints",
        "--gripper_learned_clearance_joints",
        nargs="*",
        default=["shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos"],
    )
    parser.add_argument("--gripper-learned-near-done-px", "--gripper_learned_near_done_px", type=float, default=0.0)
    parser.add_argument("--gripper-learned-refine-outer-correction", "--gripper_learned_refine_outer_correction", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-learned-settle-threshold-deg", "--gripper_learned_settle_threshold_deg", type=float, default=0.35)
    parser.add_argument("--gripper-learned-stop-after-accepted-move", "--gripper_learned_stop_after_accepted_move", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-learned-require-linked-sample", "--gripper_learned_require_linked_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gripper-learned-use-param-model", "--gripper_learned_use_param_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-learned-auto-update-params", "--gripper_learned_auto_update_params", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-learned-param-history", "--gripper_learned_param_history", type=int, default=3)
    parser.add_argument("--gripper-learned-param-only", "--gripper_learned_param_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coarse-bias-enable", "--coarse_bias_enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coarse-bias-auto-update", "--coarse_bias_auto_update", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coarse-bias-history", "--coarse_bias_history", type=int, default=3)
    parser.add_argument("--coarse-bias-max-joint-deg", "--coarse_bias_max_joint_deg", type=float, default=4.0)
    parser.add_argument("--coarse-bias-min-improve-px", "--coarse_bias_min_improve_px", type=float, default=1.0)
    parser.add_argument("--gripper-correction-max-deg-per-px", "--gripper_correction_max_deg_per_px", type=float, default=0.25)
    parser.add_argument("--gripper-replay-nearest-pose", "--gripper_replay_nearest_pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-replay-snap-px", "--gripper_replay_snap_px", type=float, default=50.0)
    parser.add_argument("--gripper-replay-tie-px", "--gripper_replay_tie_px", type=float, default=2.0)
    parser.add_argument("--gripper-replay-max-fail-count", "--gripper_replay_max_fail_count", type=int, default=1)
    parser.add_argument("--use-legacy-auto-calibration-points", "--use_legacy_auto_calibration_points", action="store_true")
    parser.add_argument("--manual-save-min-gripper-conf", "--manual_save_min_gripper_conf", type=float, default=0.85)
    parser.add_argument("--manual-save-max-ref-error-px", "--manual_save_max_ref_error_px", type=float, default=25.0)
    parser.add_argument("--manual-save-stable-timeout-s", "--manual_save_stable_timeout_s", type=float, default=3.0)
    parser.add_argument(
        "--direct-replay-point-id",
        "--direct_replay_point_id",
        type=int,
        default=0,
        help="Point id used by the P key direct replay test. 0 means latest saved manual/verified point.",
    )
    parser.add_argument(
        "--direct-replay-include-gripper",
        "--direct_replay_include_gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include gripper.pos when directly replaying a saved point with the P key.",
    )
    parser.add_argument(
        "--direct-replay-outer-correction",
        "--direct_replay_outer_correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After raw P-key replay, nudge the commanded joints to make observed encoders match the saved joints.",
    )
    parser.add_argument("--direct-replay-outer-iters", "--direct_replay_outer_iters", type=int, default=3)
    parser.add_argument("--direct-replay-outer-gain", "--direct_replay_outer_gain", type=float, default=1.0)
    parser.add_argument("--direct-replay-outer-max-offset", "--direct_replay_outer_max_offset", type=float, default=6.0)
    parser.add_argument(
        "--leader-sync-joints",
        "--leader_sync_joints",
        nargs="*",
        default=["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos"],
        help="Follower joints copied to the leader when pressing N. Gripper is excluded by default.",
    )
    parser.add_argument("--leader-sync-steps", "--leader_sync_steps", type=int, default=100)
    parser.add_argument("--leader-sync-sleep-s", "--leader_sync_sleep_s", type=float, default=0.025)
    parser.add_argument(
        "--leader-sync-hold-torque",
        "--leader_sync_hold_torque",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep leader torque enabled after N sync so the leader stays at the follower pose until G releases it.",
    )
    parser.add_argument(
        "--leader-sync-include-gripper",
        "--leader_sync_include_gripper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also sync gripper.pos to the leader. Disabled by default to avoid unnecessary gripper torque.",
    )
    parser.add_argument(
        "--handoff-open-gripper",
        "--handoff_open_gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the follower gripper before N leader-sync handoff.",
    )
    parser.add_argument("--handoff-open-gripper-pos", "--handoff_open_gripper_pos", type=float, default=60.0)
    parser.add_argument("--handoff-open-gripper-steps", "--handoff_open_gripper_steps", type=int, default=24)
    parser.add_argument("--handoff-open-gripper-sleep-s", "--handoff_open_gripper_sleep_s", type=float, default=0.025)
    parser.add_argument(
        "--act-record-root",
        "--act_record_root",
        default=str(PROJECT_ROOT / "datasets" / "so101_local_grasp_act_handoff_test"),
        help="Local LeRobotDataset root for D-key local grasp ACT recordings.",
    )
    parser.add_argument(
        "--act-record-repo-id",
        "--act_record_repo_id",
        default="local/so101_local_grasp_act_handoff_test",
        help="Repo id stored in metadata for D-key local grasp ACT recordings.",
    )
    parser.add_argument(
        "--act-record-task",
        "--act_record_task",
        default="Local grasp the aligned object",
        help="Task string attached to each D-key ACT episode.",
    )
    parser.add_argument("--act-record-fps", "--act_record_fps", type=int, default=20)
    parser.add_argument(
        "--act-record-video",
        "--act_record_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store D-key ACT camera streams as videos, matching the existing dual-camera datasets.",
    )
    parser.add_argument("--act-record-image-writer-threads", "--act_record_image_writer_threads", type=int, default=0)
    parser.add_argument(
        "--manual-correction-trace",
        "--manual_correction_trace",
        default=str(PROJECT_ROOT / "vision" / "outputs" / "yolo_guided_align_traces" / "manual_corrections.csv"),
    )
    parser.add_argument(
        "--auto-learn-overhead-after-gripper",
        "--auto_learn_overhead_after_gripper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After successful gripper final alignment, add the refined final pose as an overhead calibration point.",
    )
    parser.add_argument(
        "--auto-learn-min-xy-delta-deg",
        "--auto_learn_min_xy_delta_deg",
        type=float,
        default=0.25,
        help="Only auto-save an overhead correction point if gripper refine changed pan/lift by at least this much.",
    )
    parser.add_argument("--gripper-trace", "--gripper_trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gripper-trace-dir",
        "--gripper_trace_dir",
        default=str(PROJECT_ROOT / "vision" / "outputs" / "yolo_guided_align_traces"),
    )
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--no-preview", "--no_preview", action="store_true")
    parser.add_argument(
        "--enable-manual-correction",
        "--enable_manual_correction",
        action="store_true",
        help="In align mode, connect the leader so you can manually correct the follower and press c to save the corrected pose.",
    )
    parser.add_argument("--correction-max-target-age-s", "--correction_max_target_age_s", type=float, default=15.0)
    return parser.parse_args()


def require_deps() -> None:
    if cv2 is None or np is None:
        raise RuntimeError("cv2 and numpy are required in the lerobot conda environment.")


def camera_backend_flag(args) -> int:
    if args.camera_backend == "dshow":
        return cv2.CAP_DSHOW
    if args.camera_backend == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def open_camera(camera_index: int, args=None, label: str = "camera"):
    backend = camera_backend_flag(args) if args is not None else cv2.CAP_DSHOW
    capture = cv2.VideoCapture(camera_index, backend)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {label} camera index {camera_index}")
    if args is not None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.camera_width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.camera_height))
        capture.set(cv2.CAP_PROP_FPS, int(args.camera_fps))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(3):
        capture.read()
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"{label} camera index {camera_index} opened but did not return a frame")
    h, w = frame.shape[:2]
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    print(f"{label} camera {camera_index}: {w}x{h}, reported FPS={fps:.2f}, backend={args.camera_backend if args is not None else 'dshow'}")
    return capture


def transform_gripper_frame(frame, args):
    image = frame
    rotate = getattr(args, "gripper_rotate", "none")
    flip = getattr(args, "gripper_flip", "none")
    if rotate == "90":
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == "180":
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif rotate == "270":
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if flip == "h":
        image = cv2.flip(image, 1)
    elif flip == "v":
        image = cv2.flip(image, 0)
    elif flip == "both":
        image = cv2.flip(image, -1)
    return image


def frames_look_duplicate(left, right, threshold: float) -> bool:
    if left is None or right is None:
        return False
    lh, lw = left.shape[:2]
    rh, rw = right.shape[:2]
    h = min(lh, rh, 120)
    if h <= 0:
        return False
    left_small = resize_to_height(left, h)
    right_small = resize_to_height(right, h)
    h2 = min(left_small.shape[0], right_small.shape[0])
    w2 = min(left_small.shape[1], right_small.shape[1], 160)
    if h2 <= 0 or w2 <= 0:
        return False
    left_crop = left_small[:h2, :w2].astype(np.float32)
    right_crop = right_small[:h2, :w2].astype(np.float32)
    diff = float(np.mean(np.abs(left_crop - right_crop)))
    return diff <= float(threshold)


def create_robot(args):
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower

    return SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id=args.follower_id,
            max_relative_target=args.max_relative_target,
            cameras={},
        )
    )


def create_leader(args):
    from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
    from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader

    return SO101Leader(SO101LeaderConfig(port=args.leader_port, id=args.leader_id))


def manual_teleop_worker(
    robot,
    leader,
    robot_lock: threading.RLock,
    enabled_event: threading.Event,
    stop_event: threading.Event,
    shared: dict[str, Any],
    args,
) -> None:
    target_dt = 1.0 / max(1.0, float(args.manual_teleop_fps))
    frames = 0
    started_at = time.perf_counter()
    while not stop_event.is_set():
        loop_start = time.perf_counter()
        if not enabled_event.is_set():
            time.sleep(0.02)
            continue
        try:
            action = leader.get_action()
            with robot_lock:
                performed = robot.send_action(action)
            shared["latest_action"] = scalar_motor_state(action)
            shared["latest_performed"] = scalar_motor_state(performed)
            shared["latest_obs"] = scalar_motor_state(performed)
            shared["latest_command_time"] = time.perf_counter()
            frames += 1
            elapsed = time.perf_counter() - started_at
            if elapsed >= 1.0:
                shared["teleop_fps"] = frames / elapsed
                frames = 0
                started_at = time.perf_counter()
        except Exception as exc:  # noqa: BLE001
            shared["error"] = f"manual teleop error: {exc}"
            enabled_event.clear()
        sleep_s = target_dt - (time.perf_counter() - loop_start)
        if sleep_s > 0:
            time.sleep(sleep_s)


def leader_motor_name(joint_key: str) -> str:
    return joint_key.removesuffix(".pos")


def sync_leader_to_follower_pose(
    leader,
    follower_pose: dict[str, float],
    args,
    *,
    stop_event: threading.Event | None = None,
) -> str:
    if leader is None:
        return "leader sync unavailable: leader is not connected in this run"
    if not getattr(leader, "is_connected", False):
        return "leader sync unavailable: leader is not connected"
    if not hasattr(leader, "bus") or not hasattr(leader.bus, "sync_write"):
        return "leader sync unavailable: leader bus cannot write Goal_Position"

    requested_joints = list(getattr(args, "leader_sync_joints", []) or [])
    if bool(getattr(args, "leader_sync_include_gripper", False)) and "gripper.pos" not in requested_joints:
        requested_joints.append("gripper.pos")
    joints = [
        joint
        for joint in requested_joints
        if joint.endswith(".pos")
        and joint in follower_pose
        and leader_motor_name(joint) in leader.bus.motors
    ]
    if not joints:
        return "leader sync skipped: no matching follower/leader joints"

    leader_start = leader.get_action()
    missing = [joint for joint in joints if joint not in leader_start]
    if missing:
        return "leader sync skipped: leader missing " + ",".join(missing)

    start = {joint: float(leader_start[joint]) for joint in joints}
    target = {joint: float(follower_pose[joint]) for joint in joints}
    motors = [leader_motor_name(joint) for joint in joints]
    steps = max(1, int(getattr(args, "leader_sync_steps", 100)))
    sleep_s = max(0.0, float(getattr(args, "leader_sync_sleep_s", 0.025)))
    keep_torque = False

    try:
        leader.bus.sync_write(
            "Goal_Position",
            {leader_motor_name(joint): start[joint] for joint in joints},
        )
        leader.bus.enable_torque(motors)
        for step in range(1, steps + 1):
            if stop_event is not None and stop_event.is_set():
                return "leader sync cancelled"
            alpha = step / steps
            goal = {
                leader_motor_name(joint): start[joint] + (target[joint] - start[joint]) * alpha
                for joint in joints
            }
            leader.bus.sync_write("Goal_Position", goal)
            if sleep_s > 0:
                time.sleep(sleep_s)
        final = leader.get_action()
        errors = {joint: float(target[joint]) - float(final.get(joint, target[joint])) for joint in joints}
        max_error = max((abs(value) for value in errors.values()), default=0.0)
        detail = " ".join(
            f"{joint.split('.')[0]}={float(final.get(joint, float('nan'))):.1f}/{target[joint]:.1f}"
            for joint in joints
        )
        keep_torque = bool(getattr(args, "leader_sync_hold_torque", True))
        hold_text = "; leader torque held, press G to release/start teleop" if keep_torque else "; leader torque released"
        return f"leader sync done max_error={max_error:.2f}; {detail}{hold_text}"
    except Exception as exc:  # noqa: BLE001
        return f"leader sync failed: {exc}"
    finally:
        if not keep_torque:
            try:
                leader.bus.disable_torque(motors)
            except Exception:
                pass


def release_leader_sync_hold(leader, hold_state: dict[str, Any]) -> str:
    if not bool(hold_state.get("active")):
        return "leader sync hold is not active"
    motors = list(hold_state.get("motors") or [])
    hold_state["active"] = False
    hold_state["motors"] = []
    if leader is None or not getattr(leader, "is_connected", False):
        return "leader sync hold cleared; leader is not connected"
    try:
        leader.bus.disable_torque(motors or None)
    except Exception as exc:  # noqa: BLE001
        return f"leader sync hold release failed: {exc}"
    return "leader sync hold released"


def open_follower_gripper_for_handoff(
    robot,
    args,
    robot_lock: threading.RLock | None = None,
    *,
    stop_event: threading.Event | None = None,
) -> str:
    if not bool(getattr(args, "handoff_open_gripper", True)):
        return "handoff gripper open disabled"
    target = float(getattr(args, "handoff_open_gripper_pos", 60.0))
    steps = max(1, int(getattr(args, "handoff_open_gripper_steps", 24)))
    sleep_s = max(0.0, float(getattr(args, "handoff_open_gripper_sleep_s", 0.025)))
    current = locked_observation(robot, robot_lock)
    if "gripper.pos" not in current:
        return "handoff gripper open skipped: follower gripper.pos missing"
    start = float(current["gripper.pos"])
    pose = {
        joint: float(current[joint])
        for joint in MOTOR_KEYS
        if joint in current
    }
    try:
        for step in range(1, steps + 1):
            if stop_event is not None and stop_event.is_set():
                return "handoff gripper open cancelled"
            alpha = step / steps
            pose["gripper.pos"] = start + (target - start) * alpha
            send_action_once(robot, pose, robot_lock)
            if sleep_s > 0:
                time.sleep(sleep_s)
        final = locked_observation(robot, robot_lock)
        final_gripper = float(final.get("gripper.pos", target))
        return f"handoff gripper opened {start:.1f}->{final_gripper:.1f} target={target:.1f}"
    except Exception as exc:  # noqa: BLE001
        return f"handoff gripper open failed: {exc}"


def act_record_features(args) -> dict[str, dict]:
    from lerobot.datasets.utils import hw_to_dataset_features

    image_shape = (int(args.camera_height), int(args.camera_width), 3)
    motor_features = {key: float for key in MOTOR_KEYS}
    obs_features = {
        **motor_features,
        "overhead": image_shape,
        "front": image_shape,
    }
    features = {}
    features.update(hw_to_dataset_features(motor_features, "action", use_video=bool(args.act_record_video)))
    features.update(hw_to_dataset_features(obs_features, "observation", use_video=bool(args.act_record_video)))
    return features


def load_or_create_act_record_dataset(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.act_record_root)
    repo_id = str(args.act_record_repo_id)
    if (root / "meta" / "info.json").exists():
        dataset = LeRobotDataset(repo_id, root=root, batch_encoding_size=1)
        dataset.episode_buffer = dataset.create_episode_buffer()
        if int(getattr(args, "act_record_image_writer_threads", 0) or 0) > 0:
            dataset.start_image_writer(num_threads=int(args.act_record_image_writer_threads))
        return dataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(args.act_record_fps),
        features=act_record_features(args),
        root=root,
        robot_type="so101_follower",
        use_videos=bool(args.act_record_video),
        image_writer_processes=0,
        image_writer_threads=max(0, int(getattr(args, "act_record_image_writer_threads", 0) or 0)),
        batch_encoding_size=1,
    )


def act_record_dataset_episode_size(dataset) -> int:
    episode_buffer = getattr(dataset, "episode_buffer", None)
    if not isinstance(episode_buffer, dict):
        return 0
    return int(episode_buffer.get("size") or 0)


def prepare_act_record_image(image_bgr, args):
    if image_bgr is None:
        return None
    expected_w = int(args.camera_width)
    expected_h = int(args.camera_height)
    image = image_bgr
    if image.shape[1] != expected_w or image.shape[0] != expected_h:
        image = cv2.resize(image, (expected_w, expected_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def add_act_record_frame(
    dataset,
    overhead_bgr,
    front_bgr,
    observation: dict[str, float],
    action: dict[str, float],
    args,
) -> tuple[bool, str]:
    from lerobot.datasets.utils import build_dataset_frame

    if not all(key in observation for key in MOTOR_KEYS):
        missing = [key for key in MOTOR_KEYS if key not in observation]
        return False, "ACT frame skipped: observation missing " + ",".join(missing)
    if not all(key in action for key in MOTOR_KEYS):
        missing = [key for key in MOTOR_KEYS if key not in action]
        return False, "ACT frame skipped: action missing " + ",".join(missing)

    overhead_rgb = prepare_act_record_image(overhead_bgr, args)
    front_rgb = prepare_act_record_image(front_bgr, args)
    if overhead_rgb is None or front_rgb is None:
        return False, "ACT frame skipped: overhead/front frame missing"

    obs_values = {key: float(observation[key]) for key in MOTOR_KEYS}
    obs_values["overhead"] = overhead_rgb
    obs_values["front"] = front_rgb
    action_values = {key: float(action[key]) for key in MOTOR_KEYS}
    frame = {
        **build_dataset_frame(dataset.features, obs_values, prefix="observation"),
        **build_dataset_frame(dataset.features, action_values, prefix="action"),
    }
    dataset.add_frame(frame, task=str(args.act_record_task))
    return True, f"ACT frame recorded: frames={act_record_dataset_episode_size(dataset)}"


def save_act_record_episode(dataset, args) -> str:
    frame_count = act_record_dataset_episode_size(dataset)
    if frame_count <= 0:
        dataset.clear_episode_buffer()
        return "ACT recording stopped: no frames captured"
    dataset.save_episode()
    return f"ACT episode saved: frames={frame_count} root={Path(args.act_record_root)}"


def scalar_motor_state(state: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key in MOTOR_KEYS:
        try:
            result[key] = float(state[key])
        except (KeyError, TypeError, ValueError):
            pass
    return result


def locked_observation(robot, robot_lock: threading.RLock | None = None) -> dict[str, float]:
    if robot_lock is None:
        return scalar_motor_state(robot.get_observation())
    with robot_lock:
        return scalar_motor_state(robot.get_observation())


def load_yolo(model_path: Path):
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Install ultralytics in the lerobot conda environment: python -m pip install ultralytics") from exc
    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print(f"Loaded YOLO model: {model_path}")
    print(f"classes: {names}")
    return model, names


def active_target_point_kind(class_name: str, args) -> str:
    if args.target_point != "auto":
        return args.target_point
    return "mask_centroid"


def active_target_point_params(args) -> dict[str, Any]:
    kind = active_target_point_kind(args.target_class, args)
    if kind != "top_face":
        return {}
    return {
        "cube_top_method": TOP_FACE_METHOD,
        "cube_top_face_fraction": float(args.cube_top_face_fraction),
        "cube_top_face_center_y_frac": float(args.cube_top_face_center_y_frac),
        "cube_top_face_band_frac": float(args.cube_top_face_band_frac),
        "cube_top_contour_epsilon_frac": float(args.cube_top_contour_epsilon_frac),
        "cube_top_min_edge_px": float(args.cube_top_min_edge_px),
        "top_face_min_pixels": int(args.top_face_min_pixels),
        "target_offset_u": float(args.target_offset_u),
        "target_offset_v": float(args.target_offset_v),
    }


def cli_option_was_supplied(*option_names: str) -> bool:
    prefixes = tuple(f"{name}=" for name in option_names)
    for token in sys.argv[1:]:
        if token in option_names or token.startswith(prefixes):
            return True
    return False


def apply_saved_target_point_params(state: CalibrationState, args) -> str:
    if active_target_point_kind(args.target_class, args) != "top_face":
        return "target params: mask/centroid mode"
    if state.config_active_target_point_kind not in (None, "top_face"):
        return "target params: using command/default values"
    if state.config_target_class not in (None, args.target_class):
        return "target params: using command/default values"
    params = state.config_active_target_point_params or state.pose_model_target_point_params
    if not isinstance(params, dict):
        return "target params: using command/default values"

    restored: list[str] = []
    option_map = {
        "cube_top_face_fraction": ("cube_top_face_fraction", "--cube-top-face-fraction", "--cube_top_face_fraction"),
        "cube_top_face_center_y_frac": (
            "cube_top_face_center_y_frac",
            "--cube-top-face-center-y-frac",
            "--cube_top_face_center_y_frac",
        ),
        "cube_top_face_band_frac": ("cube_top_face_band_frac", "--cube-top-face-band-frac", "--cube_top_face_band_frac"),
        "cube_top_contour_epsilon_frac": (
            "cube_top_contour_epsilon_frac",
            "--cube-top-contour-epsilon-frac",
            "--cube_top_contour_epsilon_frac",
        ),
        "cube_top_min_edge_px": ("cube_top_min_edge_px", "--cube-top-min-edge-px", "--cube_top_min_edge_px"),
        "top_face_min_pixels": ("top_face_min_pixels", "--top-face-min-pixels", "--top_face_min_pixels"),
        "target_offset_u": ("target_offset_u", "--target-offset-u", "--target_offset_u"),
        "target_offset_v": ("target_offset_v", "--target-offset-v", "--target_offset_v"),
    }
    for param_key, (attr_name, *option_names) in option_map.items():
        if param_key not in params or cli_option_was_supplied(*option_names):
            continue
        current = getattr(args, attr_name)
        value = params[param_key]
        if isinstance(current, int) and not isinstance(current, bool):
            setattr(args, attr_name, int(value))
        else:
            setattr(args, attr_name, float(value))
        restored.append(param_key)

    if not restored:
        return "target params: using command/default values"
    offset = f"offset=({float(args.target_offset_u):.1f},{float(args.target_offset_v):.1f})"
    return f"target params: restored {','.join(restored)} from config {offset}"


def compatible_target_point_params(point: dict[str, Any], args) -> bool:
    if active_target_point_kind(args.target_class, args) != "top_face":
        return True
    params = point.get("target_point_params")
    if params is None:
        return bool(args.allow_legacy_top_face_points)
    try:
        required_keys = {
            "cube_top_method",
            "cube_top_face_fraction",
            "cube_top_face_center_y_frac",
            "cube_top_face_band_frac",
            "cube_top_contour_epsilon_frac",
            "cube_top_min_edge_px",
            "top_face_min_pixels",
            "target_offset_u",
            "target_offset_v",
        }
        if not required_keys.issubset(set(params.keys())):
            return bool(args.allow_legacy_top_face_points)
        return (
            str(params.get("cube_top_method")) == TOP_FACE_METHOD
            and abs(float(params.get("cube_top_face_fraction")) - float(args.cube_top_face_fraction)) < 1e-6
            and abs(float(params.get("cube_top_face_center_y_frac")) - float(args.cube_top_face_center_y_frac)) < 1e-6
            and abs(float(params.get("cube_top_face_band_frac")) - float(args.cube_top_face_band_frac)) < 1e-6
            and abs(float(params.get("cube_top_contour_epsilon_frac")) - float(args.cube_top_contour_epsilon_frac)) < 1e-6
            and abs(float(params.get("cube_top_min_edge_px")) - float(args.cube_top_min_edge_px)) < 1e-6
            and int(params.get("top_face_min_pixels")) == int(args.top_face_min_pixels)
            and abs(float(params.get("target_offset_u")) - float(args.target_offset_u)) < 1e-6
            and abs(float(params.get("target_offset_v")) - float(args.target_offset_v)) < 1e-6
        )
    except (TypeError, ValueError):
        return False


def clamp_image_point(u: float, v: float, shape: tuple[int, ...]) -> tuple[float, float]:
    return float(np.clip(u, 0, shape[1] - 1)), float(np.clip(v, 0, shape[0] - 1))


def with_target_offset(center: tuple[float, float], shape: tuple[int, ...], args) -> tuple[float, float]:
    return clamp_image_point(
        float(center[0]) + float(args.target_offset_u),
        float(center[1]) + float(args.target_offset_v),
        shape,
    )


def largest_mask_contour(mask) -> Any | None:
    mask_u8 = (mask.astype("uint8") * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def mask_shape_features(mask, xyxy: list[float] | tuple[float, float, float, float] | None = None) -> dict[str, float]:
    """Return compact geometric features for a YOLO segmentation mask.

    These values are intentionally simple and image-space only. They are useful
    for logging whether the gripper is moving closer/farther from the object
    (area/sqrt_area) and whether pose/approach changed (axis ratio/angle).
    """
    empty = {
        "mask_area_px": 0.0,
        "mask_sqrt_area_px": 0.0,
        "mask_bbox_w_px": 0.0,
        "mask_bbox_h_px": 0.0,
        "mask_bbox_aspect": 0.0,
        "mask_extent": 0.0,
        "mask_contour_area_px": 0.0,
        "mask_hull_area_px": 0.0,
        "mask_solidity": 0.0,
        "mask_perimeter_px": 0.0,
        "mask_circularity": 0.0,
        "mask_rect_w_px": 0.0,
        "mask_rect_h_px": 0.0,
        "mask_rect_aspect": 0.0,
        "mask_angle_deg": 0.0,
        "mask_major_axis_px": 0.0,
        "mask_minor_axis_px": 0.0,
        "mask_axis_ratio": 0.0,
    }
    if mask is None:
        return empty
    mask_bool = np.asarray(mask, dtype=bool)
    area = float(np.count_nonzero(mask_bool))
    if area <= 0.0:
        return empty

    features = dict(empty)
    features["mask_area_px"] = area
    features["mask_sqrt_area_px"] = math.sqrt(area)

    ys, xs = np.nonzero(mask_bool)
    if len(xs):
        bbox_w = float(xs.max() - xs.min() + 1)
        bbox_h = float(ys.max() - ys.min() + 1)
    elif xyxy is not None:
        bbox_w = max(0.0, float(xyxy[2]) - float(xyxy[0]))
        bbox_h = max(0.0, float(xyxy[3]) - float(xyxy[1]))
    else:
        bbox_w = bbox_h = 0.0
    features["mask_bbox_w_px"] = bbox_w
    features["mask_bbox_h_px"] = bbox_h
    features["mask_bbox_aspect"] = bbox_w / max(1e-6, bbox_h)
    features["mask_extent"] = area / max(1e-6, bbox_w * bbox_h)

    contour = largest_mask_contour(mask_bool)
    if contour is None:
        return features
    contour_area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    rect = cv2.minAreaRect(contour)
    (rect_w, rect_h) = rect[1]
    rect_w = float(rect_w)
    rect_h = float(rect_h)
    major_axis = max(rect_w, rect_h)
    minor_axis = min(rect_w, rect_h)

    features["mask_contour_area_px"] = contour_area
    features["mask_hull_area_px"] = hull_area
    features["mask_solidity"] = contour_area / max(1e-6, hull_area)
    features["mask_perimeter_px"] = perimeter
    features["mask_circularity"] = (4.0 * math.pi * contour_area) / max(1e-6, perimeter * perimeter)
    features["mask_rect_w_px"] = rect_w
    features["mask_rect_h_px"] = rect_h
    features["mask_rect_aspect"] = major_axis / max(1e-6, minor_axis)
    features["mask_angle_deg"] = float(rect[2])
    features["mask_major_axis_px"] = major_axis
    features["mask_minor_axis_px"] = minor_axis
    features["mask_axis_ratio"] = major_axis / max(1e-6, minor_axis)
    return features


def approx_cube_polygon(contour, args) -> np.ndarray | None:
    hull = cv2.convexHull(contour)
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter <= 0:
        return None

    base_eps = min(0.08, max(0.002, float(args.cube_top_contour_epsilon_frac)))
    best: np.ndarray | None = None
    for factor in (0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.5, 7.0):
        approx = cv2.approxPolyDP(hull, perimeter * base_eps * factor, True).reshape(-1, 2).astype(np.float64)
        if len(approx) >= 4:
            best = approx
        if 5 <= len(approx) <= 8:
            return approx
        if 4 <= len(approx) <= 8 and best is not None:
            return approx
    return best if best is not None and len(best) >= 4 else None


def point_in_mask(mask, point: tuple[float, float]) -> bool:
    u, v = int(round(point[0])), int(round(point[1]))
    if v < 0 or v >= mask.shape[0] or u < 0 or u >= mask.shape[1]:
        return False
    return bool(mask[v, u])


def estimate_top_face_from_polygon(mask, polygon: np.ndarray, args) -> TopFaceEstimate | None:
    if polygon is None or len(polygon) < 4:
        return None

    y_min = float(np.min(polygon[:, 1]))
    y_max = float(np.max(polygon[:, 1]))
    height = max(1.0, y_max - y_min)
    min_edge = max(2.0, float(args.cube_top_min_edge_px))
    y_gate = y_min + height * 0.45
    candidates: list[tuple[float, TopFaceEstimate]] = []

    for i, vertex in enumerate(polygon):
        if float(vertex[1]) > y_gate:
            continue
        prev_vertex = polygon[(i - 1) % len(polygon)]
        next_vertex = polygon[(i + 1) % len(polygon)]
        prev_len = float(np.linalg.norm(prev_vertex - vertex))
        next_len = float(np.linalg.norm(next_vertex - vertex))
        if prev_len < min_edge or next_len < min_edge:
            continue

        center = ((prev_vertex + next_vertex) * 0.5).astype(np.float64)
        if float(center[1]) < float(vertex[1]) - 2.0:
            continue

        fourth = prev_vertex + next_vertex - vertex
        quad = [
            (float(vertex[0]), float(vertex[1])),
            (float(prev_vertex[0]), float(prev_vertex[1])),
            (float(fourth[0]), float(fourth[1])),
            (float(next_vertex[0]), float(next_vertex[1])),
        ]
        center_uv = with_target_offset((float(center[0]), float(center[1])), mask.shape, args)
        inside_penalty = 0.0 if point_in_mask(mask, center_uv) else height
        balance_penalty = abs(prev_len - next_len) * 0.05
        score = float(vertex[1]) + inside_penalty + balance_penalty
        candidates.append((score, TopFaceEstimate(center_uv=center_uv, quad_uv=quad, method=TOP_FACE_METHOD)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def estimate_top_face_by_band(mask, args) -> TopFaceEstimate | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < max(1, args.top_face_min_pixels):
        return None

    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    if y_max <= y_min:
        return None

    height = max(1.0, float(y_max - y_min))
    center_frac = min(0.95, max(0.05, float(args.cube_top_face_center_y_frac)))
    band_frac = min(0.8, max(0.02, float(args.cube_top_face_band_frac)))
    center_y = y_min + height * center_frac
    half_band = max(2.0, height * band_frac * 0.5)

    row_centers: list[float] = []
    row_ys: list[float] = []
    row_weights: list[float] = []
    for y in range(max(0, int(round(center_y - half_band))), min(mask.shape[0], int(round(center_y + half_band)) + 1)):
        row_xs = np.flatnonzero(mask[y])
        if len(row_xs) < 2:
            continue
        row_centers.append((float(row_xs[0]) + float(row_xs[-1])) / 2.0)
        row_ys.append(float(y))
        row_weights.append(float(len(row_xs)))

    if len(row_centers) < 2 or sum(row_weights) < max(1, args.top_face_min_pixels):
        fraction = min(1.0, max(0.1, float(args.cube_top_face_fraction)))
        cutoff = y_min + height * fraction
        keep = ys <= cutoff
        if int(np.count_nonzero(keep)) < max(1, args.top_face_min_pixels):
            return None
        row_centers = xs[keep].astype(np.float64).tolist()
        row_ys = ys[keep].astype(np.float64).tolist()
        row_weights = [1.0] * len(row_centers)

    weights = np.array(row_weights, dtype=np.float64)
    target_u = float(np.average(np.array(row_centers, dtype=np.float64), weights=weights))
    target_v = float(np.average(np.array(row_ys, dtype=np.float64), weights=weights))
    center_uv = with_target_offset((target_u, target_v), mask.shape, args)
    return TopFaceEstimate(center_uv=center_uv, quad_uv=None, method="band_fallback")


def estimate_top_face_center(mask, xyxy: list[float], args) -> TopFaceEstimate | None:
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if len(xs) < max(1, args.top_face_min_pixels):
        return None

    contour = largest_mask_contour(mask)
    if contour is not None and cv2.contourArea(contour) >= max(1, args.top_face_min_pixels):
        polygon = approx_cube_polygon(contour, args)
        estimate = estimate_top_face_from_polygon(mask, polygon, args)
        if estimate is not None:
            return estimate

    return estimate_top_face_by_band(mask, args)


def select_detection(result, names: dict[int, str], target_class: str, conf: float, args=None) -> TargetDetection | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    masks = result.masks.data.detach().cpu().numpy() if result.masks is not None and result.masks.data is not None else None
    best: TargetDetection | None = None
    image_h, image_w = result.orig_shape

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        class_name = names.get(cls_id, str(cls_id))
        score = float(boxes.conf[i].item())
        if class_name != target_class or score < conf:
            continue

        xyxy = [float(v) for v in boxes.xyxy[i].tolist()]
        mask = None
        if masks is not None and i < len(masks):
            resized = cv2.resize(masks[i].astype("float32"), (image_w, image_h), interpolation=cv2.INTER_LINEAR)
            binary = resized > 0.5
            moments = cv2.moments(binary.astype("uint8"))
            if moments["m00"] > 0:
                mask_center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
                mask = binary
            else:
                mask_center = ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)
        else:
            mask_center = ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)

        point_kind = active_target_point_kind(class_name, args) if args is not None else "mask_centroid"
        center = mask_center
        top_face_quad = None
        top_face_method = None
        if point_kind == "top_face":
            top_estimate = estimate_top_face_center(mask, xyxy, args)
            if top_estimate is not None:
                center = top_estimate.center_uv
                top_face_quad = top_estimate.quad_uv
                top_face_method = top_estimate.method
            else:
                point_kind = "mask_centroid_fallback"

        det = TargetDetection(
            cls_id,
            class_name,
            score,
            center,
            xyxy,
            point_kind,
            mask_center,
            mask,
            top_face_quad,
            top_face_method,
        )
        if best is None or det.confidence > best.confidence:
            best = det

    return best


def run_predict(model, frame, args):
    kwargs = {"source": frame, "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou, "verbose": False}
    if args.device:
        kwargs["device"] = args.device
    model_id = id(model)
    with _MODEL_PREDICT_LOCKS_GUARD:
        predict_lock = _MODEL_PREDICT_LOCKS.setdefault(model_id, threading.RLock())
    with predict_lock:
        return model.predict(**kwargs)[0]


def draw_detection(frame, det: TargetDetection | None, status_lines: list[str]) -> Any:
    image = frame.copy()
    if det is not None:
        color = color_for_class(det.class_id)
        if det.mask is not None:
            overlay = image.copy()
            overlay[det.mask] = (
                overlay[det.mask].astype(np.float32) * 0.62 + np.array(color, dtype=np.float32) * 0.38
            ).astype(np.uint8)
            image = overlay
        x1, y1, x2, y2 = [int(round(v)) for v in det.xyxy]
        u, v = det.center_uv
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if det.mask_centroid_uv is not None and det.point_kind.startswith("top_face"):
            cu, cv = det.mask_centroid_uv
            cv2.circle(image, (int(round(cu)), int(round(cv))), 4, (255, 255, 255), 1)
        if det.top_face_quad_uv:
            quad = np.array(
                [[int(round(uq)), int(round(vq))] for uq, vq in det.top_face_quad_uv],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(image, [quad], isClosed=True, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            for uq, vq in det.top_face_quad_uv:
                cv2.circle(image, (int(round(uq)), int(round(vq))), 3, (255, 255, 0), -1)
        cv2.circle(image, (int(round(u)), int(round(v))), 6, (0, 0, 255), -1)
        method_text = f"/{det.top_face_method}" if det.top_face_method else ""
        label = f"{det.class_name} {det.confidence:.2f} target={det.point_kind}{method_text} ({u:.1f},{v:.1f})"
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )
    draw_text(image, status_lines)
    return image


def draw_reference_marker(
    image,
    uv: tuple[float, float] | None,
    label: str = "ref",
    radius_px: float | None = None,
) -> None:
    if uv is None:
        return
    u, v = int(round(uv[0])), int(round(uv[1]))
    if radius_px is not None and math.isfinite(float(radius_px)) and float(radius_px) > 0:
        cv2.circle(image, (u, v), int(round(float(radius_px))), (0, 200, 255), 1, lineType=cv2.LINE_AA)
    cv2.drawMarker(image, (u, v), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
    cv2.circle(image, (u, v), 9, (0, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(image, label, (u + 10, max(18, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, label, (u + 10, max(18, v - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)


def resize_to_height(image, height: int):
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, int(round(w * (height / h))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def compose_side_by_side(left, right):
    h = min(left.shape[0], right.shape[0])
    return np.concatenate([resize_to_height(left, h), resize_to_height(right, h)], axis=1)


def load_calibration(path: Path) -> CalibrationState:
    state = CalibrationState()
    if not path.exists():
        return state
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    state.points = data.get("points", [])
    state.safe_pose = data.get("safe_pose")
    state.homography = data.get("homography")
    state.pose_model = data.get("pose_model", {})
    state.pose_model_features = data.get("pose_model_features", [])
    state.pose_model_point_kind = data.get("pose_model_point_kind")
    state.pose_model_target_class = data.get("pose_model_target_class")
    state.pose_model_target_point_params = data.get("pose_model_target_point_params")
    state.config_target_class = data.get("target_class")
    state.config_active_target_point_kind = data.get("active_target_point_kind")
    state.config_active_target_point_params = data.get("active_target_point_params")
    state.pose_limits = data.get("pose_limits", {})
    state.orientation_model = data.get("orientation_model", {})
    state.joint_limits = data.get("joint_limits", {})
    state.orientation_limits = data.get("orientation_limits", {})
    state.hover_height_samples = data.get("hover_height_samples", [])
    state.hover_height_model = data.get("hover_height_model", {})
    state.hover_height_model_features = data.get("hover_height_model_features", [])
    state.hover_height_joints = data.get("hover_height_joints", [])
    state.hover_height_input_joints = data.get("hover_height_input_joints", [])
    state.hover_height_limits = data.get("hover_height_limits", {})
    state.hover_height_summary = data.get("hover_height_summary", {})
    state.gripper_correction_samples = data.get("gripper_correction_samples", [])
    state.gripper_motion_samples = data.get("gripper_motion_samples", [])
    state.gripper_primitive_samples = data.get("gripper_primitive_samples", [])
    state.gripper_learned_param_model = data.get("gripper_learned_param_model", {})
    state.coarse_pose_bias_model = data.get("coarse_pose_bias_model", {})
    return state


def save_calibration(path: Path, state: CalibrationState, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": 2,
        "updated_at": now_iso(),
        "camera_index": args.camera_index,
        "model": str(Path(args.model)),
        "map_joints": args.map_joints,
        "pose_joints": args.pose_joints,
        "pose_predictor": args.pose_predictor,
        "target_class": args.target_class,
        "target_point": args.target_point,
        "active_target_point_kind": active_target_point_kind(args.target_class, args),
        "active_target_point_params": active_target_point_params(args),
        "share_calibration_across_classes": args.share_calibration_across_classes,
        "workspace_margin": args.workspace_margin,
        "pose_margin": args.pose_margin,
        "orientation_joints": args.orientation_joints,
        "orientation_margin": args.orientation_margin,
        "orientation_source_csv": args.orientation_source_csv,
        "safe_pose": state.safe_pose,
        "points": state.points,
        "homography": state.homography,
        "pose_model": state.pose_model,
        "pose_model_features": state.pose_model_features,
        "pose_model_point_kind": state.pose_model_point_kind,
        "pose_model_target_class": state.pose_model_target_class,
        "pose_model_target_point_params": state.pose_model_target_point_params,
        "pose_limits": state.pose_limits,
        "orientation_model": state.orientation_model,
        "joint_limits": state.joint_limits,
        "orientation_limits": state.orientation_limits,
        "hover_height_samples": state.hover_height_samples,
        "hover_height_model": state.hover_height_model,
        "hover_height_model_features": state.hover_height_model_features,
        "hover_height_joints": state.hover_height_joints,
        "hover_height_input_joints": state.hover_height_input_joints,
        "hover_height_limits": state.hover_height_limits,
        "hover_height_summary": state.hover_height_summary,
        "gripper_correction_samples": state.gripper_correction_samples,
        "gripper_motion_samples": state.gripper_motion_samples,
        "gripper_primitive_samples": state.gripper_primitive_samples,
        "gripper_learned_param_model": state.gripper_learned_param_model,
        "coarse_pose_bias_model": state.coarse_pose_bias_model,
        "notes": (
            "IBVS v2 judges final alignment by a fixed robust gripper reference. "
            "gripper_motion_samples contain settled actual encoder/image transitions; "
            "gripper_primitive_samples contain virtual full-pose primitive/image transitions; "
            "unverified legacy auto_gripper_refine points are excluded unless explicitly enabled."
        ),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_calibration_point(
    state: CalibrationState,
    det: TargetDetection,
    joints: dict[str, float],
    args,
    source: str,
    gripper_det: TargetDetection | None = None,
) -> tuple[bool, str]:
    if not all(joint in joints for joint in args.pose_joints):
        return False, "cannot capture: pose joints missing from observation"

    point_id = len(state.points) + 1
    point = {
        "id": point_id,
        "captured_at": now_iso(),
        "source": source,
        "target_class": det.class_name,
        "confidence": det.confidence,
        "pixel": [float(det.center_uv[0]), float(det.center_uv[1])],
        "point_kind": det.point_kind,
        "target_point_params": active_target_point_params(args),
        "joints": {joint: float(joints[joint]) for joint in MOTOR_KEYS if joint in joints},
    }
    if gripper_det is not None:
        point["gripper_pixel"] = [float(gripper_det.center_uv[0]), float(gripper_det.center_uv[1])]
        point["gripper_confidence"] = float(gripper_det.confidence)
        point["gripper_point_kind"] = gripper_det.point_kind
        point["gripper_mask_features"] = mask_shape_features(gripper_det.mask, gripper_det.xyxy)
    state.points.append(point)
    recompute_homography(state, args)
    gripper_text = ""
    if gripper_det is not None:
        gripper_text = f" gripper=({gripper_det.center_uv[0]:.1f},{gripper_det.center_uv[1]:.1f})"
    return (
        True,
        f"captured {source} point {point_id}: point={det.point_kind} pixel=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) "
        f"joints=({joints[args.map_joints[0]]:.2f},{joints[args.map_joints[1]]:.2f}){gripper_text}",
    )


def append_auto_gripper_refine_point(
    state: CalibrationState,
    det: TargetDetection,
    initial_xy: dict[str, float],
    final_joints: dict[str, float],
    args,
    final_error_px: float | None = None,
) -> tuple[bool, str]:
    if not bool(args.auto_learn_overhead_after_gripper):
        return False, "auto overhead learning disabled"
    if not all(joint in initial_xy and joint in final_joints for joint in args.map_joints):
        return False, "auto overhead learning skipped: XY joints missing"
    if not all(joint in final_joints for joint in args.pose_joints):
        return False, "auto overhead learning skipped: final pose joints missing"
    if final_error_px is None or not math.isfinite(float(final_error_px)):
        return False, "auto overhead learning skipped: final gripper error was not re-observed"
    acceptance = gripper_ibvs_tolerance_px(state, args)
    if float(final_error_px) > acceptance:
        return False, (
            f"auto overhead learning skipped: re-observed gripper error {float(final_error_px):.1f}px "
            f"> acceptance {acceptance:.1f}px"
        )

    max_xy_delta = max(
        abs(float(final_joints[joint]) - float(initial_xy[joint]))
        for joint in args.map_joints
    )
    if max_xy_delta < max(0.0, float(args.auto_learn_min_xy_delta_deg)):
        return False, f"auto overhead learning skipped: refine delta {max_xy_delta:.2f}deg below threshold"

    ok, status = append_calibration_point(
        state,
        det,
        final_joints,
        args,
        source="auto_gripper_refine",
    )
    if not ok:
        return False, status
    state.points[-1]["verified"] = True
    state.points[-1]["controller_version"] = "ibvs_v2"
    state.points[-1]["final_gripper_error_px"] = float(final_error_px)
    recompute_homography(state, args)
    return True, f"{status}; auto_delta={max_xy_delta:.2f}deg"


def append_gripper_correction_sample(
    state: CalibrationState,
    before: dict[str, Any],
    after_det: TargetDetection,
    after_gripper_det: TargetDetection,
    after_joints: dict[str, float],
    args,
    *,
    attempted_refine: dict[str, Any] | None = None,
    sample_role: str = "manual_correction",
) -> tuple[bool, str]:
    before_joints = before.get("joints", {})
    before_gripper = before.get("gripper_pixel")
    if before_gripper is None or len(before_gripper) != 2:
        return False, "cannot save gripper correction sample: before gripper pixel missing"
    if after_gripper_det is None:
        return False, "cannot save gripper correction sample: after gripper detection missing"
    if not all(joint in before_joints and joint in after_joints for joint in args.map_joints):
        return False, "cannot save gripper correction sample: before/after XY joints missing"

    before_u = float(before_gripper[0])
    before_v = float(before_gripper[1])
    after_u = float(after_gripper_det.center_uv[0])
    after_v = float(after_gripper_det.center_uv[1])
    err_u = before_u - after_u
    err_v = before_v - after_v
    err_dist = math.hypot(err_u, err_v)
    if err_dist < float(args.gripper_learned_min_error_px):
        return False, f"gripper correction sample ignored: before/after error too small ({err_dist:.1f}px)"

    delta_xy = {
        joint: float(after_joints[joint]) - float(before_joints[joint])
        for joint in args.map_joints
    }
    delta_norm = math.hypot(*(float(delta_xy[joint]) for joint in args.map_joints))
    deg_per_px = delta_norm / max(1e-6, err_dist)
    sample = {
        "id": len(state.gripper_correction_samples) + 1,
        "captured_at": now_iso(),
        "target_class": after_det.class_name,
        "point_kind": after_det.point_kind,
        "before_overhead_pixel": before.get("overhead_pixel"),
        "after_overhead_pixel": [float(after_det.center_uv[0]), float(after_det.center_uv[1])],
        "before_gripper_pixel": [before_u, before_v],
        "after_gripper_pixel": [after_u, after_v],
        "error_uv": [err_u, err_v],
        "error_dist_px": err_dist,
        "before_xy": {joint: float(before_joints[joint]) for joint in args.map_joints},
        "after_xy": {joint: float(after_joints[joint]) for joint in args.map_joints},
        "before_joints": {joint: float(before_joints[joint]) for joint in MOTOR_KEYS if joint in before_joints},
        "after_joints": {joint: float(after_joints[joint]) for joint in MOTOR_KEYS if joint in after_joints},
        "delta_xy": delta_xy,
        "delta_norm_deg": delta_norm,
        "deg_per_px": deg_per_px,
        "delta_pose": {
            joint: float(after_joints[joint]) - float(before_joints[joint])
            for joint in args.pose_joints
            if joint in before_joints and joint in after_joints
        },
        "usable_for_xy_servo": True,
        "sample_role": sample_role,
        "controller_note": (
            "manual before/after correction is trusted as local correction teaching data; "
            "delta_xy supports legacy pan/lift refinement and delta_pose supports full-pose refinement"
        ),
    }
    if attempted_refine is not None:
        attempted_joints = attempted_refine.get("joints", {})
        attempted_gripper = attempted_refine.get("gripper_pixel")
        attempted_payload: dict[str, Any] = {
            "captured_at": attempted_refine.get("captured_at"),
            "label": attempted_refine.get("label"),
            "status": attempted_refine.get("status"),
            "overhead_pixel": attempted_refine.get("overhead_pixel"),
            "gripper_pixel": attempted_gripper,
            "gripper_confidence": attempted_refine.get("gripper_confidence"),
            "joints": {
                joint: float(attempted_joints[joint])
                for joint in MOTOR_KEYS
                if isinstance(attempted_joints, dict) and joint in attempted_joints
            },
        }
        if attempted_gripper is not None and len(attempted_gripper) == 2:
            attempted_payload["error_uv_to_corrected"] = [
                float(attempted_gripper[0]) - after_u,
                float(attempted_gripper[1]) - after_v,
            ]
            attempted_payload["error_dist_px_to_corrected"] = float(
                math.hypot(float(attempted_gripper[0]) - after_u, float(attempted_gripper[1]) - after_v)
            )
        if isinstance(attempted_joints, dict):
            attempted_payload["delta_pose_from_before"] = {
                joint: float(attempted_joints[joint]) - float(before_joints[joint])
                for joint in args.pose_joints
                if joint in attempted_joints and joint in before_joints
            }
            attempted_payload["delta_pose_to_corrected"] = {
                joint: float(after_joints[joint]) - float(attempted_joints[joint])
                for joint in args.pose_joints
                if joint in attempted_joints and joint in after_joints
            }
        sample["attempted_refine"] = attempted_payload
    if sample["delta_pose"]:
        sample["delta_pose_norm_deg"] = float(
            math.sqrt(sum(float(value) * float(value) for value in sample["delta_pose"].values()))
        )
    ok_quality, quality_reason = gripper_correction_sample_quality(sample, args)
    if not ok_quality:
        return False, "gripper correction sample ignored: " + quality_reason
    state.gripper_correction_samples.append(sample)
    delta_text = " ".join(f"{joint.split('.')[0]}={value:.2f}" for joint, value in delta_xy.items())
    return True, (
        f"saved gripper correction sample {sample['id']} role={sample_role}: "
        f"err=({err_u:.1f},{err_v:.1f})px delta {delta_text}"
    )


def gripper_correction_sample_quality(sample: dict[str, Any], args) -> tuple[bool, str]:
    error_uv = sample.get("error_uv")
    delta_xy = sample.get("delta_xy")
    if error_uv is None or len(error_uv) != 2 or not isinstance(delta_xy, dict):
        return False, "missing error_uv/delta_xy"
    if not all(joint in delta_xy for joint in args.map_joints):
        return False, "missing map-joint delta"
    err_dist = float(sample.get("error_dist_px") or math.hypot(float(error_uv[0]), float(error_uv[1])))
    if err_dist < float(args.gripper_learned_min_error_px):
        return False, f"error too small ({err_dist:.1f}px)"
    delta_norm = float(sample.get("delta_norm_deg") or math.hypot(*(float(delta_xy[joint]) for joint in args.map_joints)))
    deg_per_px = float(sample.get("deg_per_px") or (delta_norm / max(1e-6, err_dist)))
    max_deg_per_px = max(0.0, float(args.gripper_correction_max_deg_per_px))
    if max_deg_per_px > 0 and deg_per_px > max_deg_per_px:
        return False, f"outlier deg_per_px={deg_per_px:.3f} > {max_deg_per_px:.3f}"
    return True, f"quality ok deg_per_px={deg_per_px:.3f}"


def append_manual_correction_trace(
    args,
    before: dict[str, Any] | None,
    after_det: TargetDetection,
    after_gripper_det: TargetDetection | None,
    after_joints: dict[str, float],
    status: str,
) -> None:
    trace_path = Path(args.manual_correction_trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    exists = trace_path.exists()
    before_joints = before.get("joints", {}) if before else {}
    before_gripper = before.get("gripper_pixel") if before else None
    before_overhead = before.get("overhead_pixel") if before else None
    after_gripper = after_gripper_det.center_uv if after_gripper_det is not None else (None, None)
    fieldnames = [
        "timestamp",
        "target_class",
        "point_kind",
        "before_overhead_u",
        "before_overhead_v",
        "after_overhead_u",
        "after_overhead_v",
        "before_gripper_u",
        "before_gripper_v",
        "after_gripper_u",
        "after_gripper_v",
        "before_pan",
        "before_lift",
        "after_pan",
        "after_lift",
        "delta_pan",
        "delta_lift",
        "status",
    ]
    row = {
        "timestamp": now_iso(),
        "target_class": after_det.class_name,
        "point_kind": after_det.point_kind,
        "before_overhead_u": f"{float(before_overhead[0]):.3f}" if before_overhead is not None else "",
        "before_overhead_v": f"{float(before_overhead[1]):.3f}" if before_overhead is not None else "",
        "after_overhead_u": f"{float(after_det.center_uv[0]):.3f}",
        "after_overhead_v": f"{float(after_det.center_uv[1]):.3f}",
        "before_gripper_u": f"{float(before_gripper[0]):.3f}" if before_gripper is not None else "",
        "before_gripper_v": f"{float(before_gripper[1]):.3f}" if before_gripper is not None else "",
        "after_gripper_u": f"{float(after_gripper[0]):.3f}" if after_gripper[0] is not None else "",
        "after_gripper_v": f"{float(after_gripper[1]):.3f}" if after_gripper[1] is not None else "",
        "before_pan": f"{float(before_joints.get(args.map_joints[0])):.4f}" if args.map_joints[0] in before_joints else "",
        "before_lift": f"{float(before_joints.get(args.map_joints[1])):.4f}" if args.map_joints[1] in before_joints else "",
        "after_pan": f"{float(after_joints.get(args.map_joints[0])):.4f}" if args.map_joints[0] in after_joints else "",
        "after_lift": f"{float(after_joints.get(args.map_joints[1])):.4f}" if args.map_joints[1] in after_joints else "",
        "delta_pan": (
            f"{float(after_joints[args.map_joints[0]]) - float(before_joints[args.map_joints[0]]):.4f}"
            if args.map_joints[0] in before_joints and args.map_joints[0] in after_joints
            else ""
        ),
        "delta_lift": (
            f"{float(after_joints[args.map_joints[1]]) - float(before_joints[args.map_joints[1]]):.4f}"
            if args.map_joints[1] in before_joints and args.map_joints[1] in after_joints
            else ""
        ),
        "status": status,
    }
    with trace_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    jsonl_path = trace_path.with_suffix(".jsonl")
    jsonl_row = dict(row)
    jsonl_row["before_joints"] = {joint: float(value) for joint, value in before_joints.items()}
    jsonl_row["after_joints"] = {joint: float(value) for joint, value in after_joints.items()}
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")


def point_kind_of(point: dict[str, Any]) -> str:
    return str(point.get("point_kind") or "mask_centroid")


def use_class_agnostic_points(args) -> bool:
    return bool(args.share_calibration_across_classes) and active_target_point_kind(args.target_class, args) == "mask_centroid"


def calibration_point_is_trusted(point: dict[str, Any], args) -> bool:
    source = str(point.get("source") or "")
    if source.startswith("auto_gripper_refine"):
        if bool(point.get("verified")):
            return True
        return bool(args.use_legacy_auto_calibration_points)
    return True


def matching_calibration_points(points: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    expected_kind = active_target_point_kind(args.target_class, args)
    class_agnostic = use_class_agnostic_points(args)
    matches = []
    for point in points:
        if not calibration_point_is_trusted(point, args):
            continue
        if not class_agnostic and point.get("target_class") not in (None, args.target_class):
            continue
        if point_kind_of(point) != expected_kind:
            continue
        if not compatible_target_point_params(point, args):
            continue
        matches.append(point)

    manual_refs = [
        (point, [float(point["gripper_pixel"][0]), float(point["gripper_pixel"][1])])
        for point in matches
        if str(point.get("source") or "") == "manual_correction"
        and point.get("gripper_pixel") is not None
        and len(point.get("gripper_pixel")) == 2
    ]
    if len(manual_refs) >= 5:
        cluster = robust_reference_cluster(
            [pixel for _point, pixel in manual_refs],
            hard_limit_px=max(0.0, float(getattr(args, "gripper_reference_max_outlier_px", 30.0))),
            mad_scale=max(1.0, float(getattr(args, "gripper_reference_mad_scale", 2.5))),
            min_radius_px=max(1.0, float(getattr(args, "gripper_reference_min_radius_px", 8.0))),
        )
        if cluster is not None:
            inlier_objects = {id(manual_refs[index][0]) for index in cluster.inlier_indices}
            matches = [
                point
                for point in matches
                if str(point.get("source") or "") != "manual_correction"
                or point.get("gripper_pixel") is None
                or id(point) in inlier_objects
            ]
    return matches


def recompute_homography(state: CalibrationState, args) -> None:
    active_points = matching_calibration_points(state.points, args)
    if len(active_points) < 4:
        state.homography = None
        state.joint_limits = {}
        state.pose_model = {}
        state.pose_model_features = []
        state.pose_model_point_kind = None
        state.pose_model_target_class = None
        state.pose_model_target_point_params = None
        state.pose_limits = {}
        state.orientation_model = {}
        state.orientation_limits = {}
        return

    src = np.array([[p["pixel"][0], p["pixel"][1]] for p in active_points], dtype=np.float32)
    dst = np.array([[p["joints"][args.map_joints[0]], p["joints"][args.map_joints[1]]] for p in active_points], dtype=np.float32)
    homography, _ = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise RuntimeError("cv2.findHomography failed. Need at least 4 non-degenerate calibration points.")
    state.homography = homography.astype(float).tolist()

    state.joint_limits = {}
    for joint in args.map_joints:
        values = [float(p["joints"][joint]) for p in active_points]
        state.joint_limits[joint] = [min(values) - args.workspace_margin, max(values) + args.workspace_margin]
    state.pose_model, state.pose_model_features, state.pose_limits = fit_pose_model(active_points, args)
    state.pose_model_point_kind = active_target_point_kind(args.target_class, args) if state.pose_model else None
    state.pose_model_target_class = None if state.pose_model and use_class_agnostic_points(args) else args.target_class if state.pose_model else None
    state.pose_model_target_point_params = active_target_point_params(args) if state.pose_model else None
    samples = [p["joints"] for p in active_points if "joints" in p]
    state.orientation_model, state.orientation_limits = fit_orientation_model(samples, args)


def pixel_features(u: float, v: float, feature_names: list[str]) -> list[float]:
    values = {
        "u": float(u),
        "v": float(v),
        "u2": float(u) * float(u),
        "uv": float(u) * float(v),
        "v2": float(v) * float(v),
        "1": 1.0,
    }
    return [values[name] for name in feature_names]


def fit_pose_model(points: list[dict[str, Any]], args) -> tuple[dict[str, list[float]], list[str], dict[str, list[float]]]:
    usable = []
    required = list(args.pose_joints)
    for point in points:
        joints = point.get("joints", {})
        pixel = point.get("pixel")
        if pixel is None or len(pixel) != 2:
            continue
        if all(joint in joints for joint in required):
            usable.append(point)
    if len(usable) < 4:
        return {}, [], {}

    feature_names = ["u", "v", "1"]
    if len(usable) >= 6:
        feature_names = ["u", "v", "u2", "uv", "v2", "1"]

    x = np.array([pixel_features(point["pixel"][0], point["pixel"][1], feature_names) for point in usable], dtype=np.float64)
    model: dict[str, list[float]] = {}
    limits: dict[str, list[float]] = {}
    for joint in args.pose_joints:
        y = np.array([float(point["joints"][joint]) for point in usable], dtype=np.float64)
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        model[joint] = [float(v) for v in coeffs]
        limits[joint] = [float(y.min() - args.pose_margin), float(y.max() + args.pose_margin)]
    return model, feature_names, limits


def sample_joints(sample: dict[str, Any]) -> dict[str, float]:
    joints = sample.get("joints", sample)
    return {key: float(value) for key, value in joints.items() if isinstance(value, (int, float))}


def hover_height_outputs(args) -> list[str]:
    input_set = set(args.hover_height_input_joints)
    return [joint for joint in args.hover_height_joints if joint not in input_set]


def hover_feature_names(input_count: int, sample_count: int) -> list[str]:
    if input_count <= 0:
        return ["1"]
    if input_count == 1:
        return ["h0", "h0_2", "1"] if sample_count >= 3 else ["h0", "1"]
    names = [f"h{i}" for i in range(input_count)]
    if sample_count >= input_count + 4:
        names.extend(f"h{i}_h{j}" for i in range(input_count) for j in range(i, input_count))
    names.append("1")
    return names


def hover_features(input_values: list[float], feature_names: list[str]) -> list[float]:
    values: dict[str, float] = {"1": 1.0}
    for index, value in enumerate(input_values):
        value_f = float(value)
        values[f"h{index}"] = value_f
        values[f"j{index}"] = value_f
    for i, value_i in enumerate(input_values):
        for j, value_j in enumerate(input_values[i:], start=i):
            product = float(value_i) * float(value_j)
            values[f"h{i}_h{j}"] = product
            if i == j:
                values[f"h{i}_2"] = product
                values[f"j{i}_2"] = product
            else:
                values[f"j{i}_j{j}"] = product
    return [values[name] for name in feature_names]


def fit_hover_height_model(samples: list[dict[str, Any]], args) -> tuple[dict[str, list[float]], list[str], dict[str, list[float]], dict[str, Any]]:
    input_joints = list(args.hover_height_input_joints)
    output_joints = hover_height_outputs(args)
    required = input_joints + output_joints
    usable = []
    for sample in samples:
        joints = sample_joints(sample)
        if all(joint in joints for joint in required):
            usable.append(joints)
    if len(usable) < max(1, int(args.hover_height_min_samples)):
        return {}, [], {}, {"usable_samples": len(usable), "required_samples": int(args.hover_height_min_samples)}

    feature_names = hover_feature_names(len(input_joints), len(usable))

    x = np.array(
        [hover_features([joints[joint] for joint in input_joints], feature_names) for joints in usable],
        dtype=np.float64,
    )
    model: dict[str, list[float]] = {}
    limits: dict[str, list[float]] = {}
    medians: dict[str, float] = {}
    ranges: dict[str, list[float]] = {}
    for joint in output_joints:
        y = np.array([float(joints[joint]) for joints in usable], dtype=np.float64)
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        model[joint] = [float(value) for value in coeffs]
        limits[joint] = [float(y.min() - args.hover_height_margin), float(y.max() + args.hover_height_margin)]
        medians[joint] = float(np.median(y))
        ranges[joint] = [float(y.min()), float(y.max())]
    for joint in input_joints:
        y = np.array([float(joints[joint]) for joints in usable], dtype=np.float64)
        ranges[joint] = [float(y.min()), float(y.max())]
        medians[joint] = float(np.median(y))
    summary = {
        "captured_at": now_iso(),
        "usable_samples": len(usable),
        "input_joints": input_joints,
        "output_joints": output_joints,
        "medians": medians,
        "ranges": ranges,
    }
    return model, feature_names, limits, summary


def recompute_hover_height_model(state: CalibrationState, args) -> str:
    model, features, limits, summary = fit_hover_height_model(state.hover_height_samples, args)
    state.hover_height_model = model
    state.hover_height_model_features = features
    state.hover_height_limits = limits
    state.hover_height_summary = summary
    state.hover_height_joints = hover_height_outputs(args)
    state.hover_height_input_joints = list(args.hover_height_input_joints)
    if model:
        inputs = ",".join(joint.split(".")[0] for joint in state.hover_height_input_joints)
        outputs = ",".join(joint.split(".")[0] for joint in state.hover_height_joints)
        return f"hover height: fitted {len(model)} joints from {summary.get('usable_samples', 0)} samples, inputs={inputs}, outputs={outputs}"
    return (
        f"hover height: missing; record at least {int(args.hover_height_min_samples)} samples "
        f"(usable={summary.get('usable_samples', 0)})"
    )


def ensure_hover_height_model(state: CalibrationState, args) -> str:
    expected_outputs = hover_height_outputs(args)
    expected_inputs = list(args.hover_height_input_joints)
    if (
        state.hover_height_model
        and state.hover_height_model_features
        and state.hover_height_joints == expected_outputs
        and state.hover_height_input_joints == expected_inputs
    ):
        return (
            f"hover height: {len(state.hover_height_samples)} samples, "
            f"inputs={','.join(joint.split('.')[0] for joint in expected_inputs)} "
            f"outputs={','.join(joint.split('.')[0] for joint in expected_outputs)}"
        )
    return recompute_hover_height_model(state, args)


def predict_hover_height_joints(xy_target: dict[str, float], state: CalibrationState, args) -> dict[str, float]:
    if not state.hover_height_model or not state.hover_height_model_features:
        return {}
    input_joints = state.hover_height_input_joints or list(args.hover_height_input_joints)
    if not all(joint in xy_target for joint in input_joints):
        return {}
    x = hover_features([xy_target[joint] for joint in input_joints], state.hover_height_model_features)
    target: dict[str, float] = {}
    for joint, coeffs in state.hover_height_model.items():
        if len(coeffs) != len(x):
            continue
        target[joint] = float(sum(coeff * value for coeff, value in zip(coeffs, x)))
    return target


def predict_xy_hover_orientation_joints(
    xy_target: dict[str, float],
    hover_target: dict[str, float],
    state: CalibrationState,
    args,
) -> dict[str, float]:
    raw = predict_orientation_joints(xy_target, state.orientation_model, args)
    owned_joints = set(args.map_joints) | set(hover_target.keys())
    return {joint: float(value) for joint, value in raw.items() if joint not in owned_joints}


def alignment_pose_mode(state: CalibrationState, args) -> str:
    if args.alignment_pose_mode != "auto":
        return args.alignment_pose_mode
    return "xy_hover" if state.hover_height_model else "full_pose"


def ensure_pose_model(calib: CalibrationState, args) -> str:
    expected_kind = active_target_point_kind(args.target_class, args)
    expected_params = active_target_point_params(args)
    params_ok = True
    if expected_kind == "top_face" and not args.allow_legacy_top_face_points:
        params_ok = calib.pose_model_target_point_params == expected_params
    class_ok = calib.pose_model_target_class in (None, args.target_class) or use_class_agnostic_points(args)
    if (
        calib.pose_model
        and calib.pose_model_point_kind == expected_kind
        and class_ok
        and params_ok
    ):
        scope = "class-agnostic" if use_class_agnostic_points(args) else args.target_class
        return f"pose model: {expected_kind}/{args.pose_predictor}/{scope} {len(calib.pose_model_features) or '?'} features"

    active_points = matching_calibration_points(calib.points, args)
    model, features, limits = fit_pose_model(active_points, args)
    if model:
        calib.pose_model = model
        calib.pose_model_features = features
        calib.pose_model_point_kind = expected_kind
        calib.pose_model_target_class = None if use_class_agnostic_points(args) else args.target_class
        calib.pose_model_target_point_params = expected_params
        calib.pose_limits = limits
        scope = "class-agnostic" if use_class_agnostic_points(args) else args.target_class
        return f"pose model: fitted from {len(active_points)} {expected_kind} points, predictor={args.pose_predictor}, scope={scope}"

    calib.pose_model = {}
    calib.pose_model_features = []
    calib.pose_model_point_kind = None
    calib.pose_model_target_class = None
    calib.pose_model_target_point_params = None
    calib.pose_limits = {}
    calib.homography = None
    calib.joint_limits = {}
    return f"pose model: missing for {args.target_class}/{expected_kind}; collect at least 4 matching points"


def predict_pose_from_pixel(u: float, v: float, calib: CalibrationState) -> dict[str, float]:
    if not calib.pose_model or not calib.pose_model_features:
        return {}
    x = pixel_features(u, v, calib.pose_model_features)
    target: dict[str, float] = {}
    for joint, coeffs in calib.pose_model.items():
        if len(coeffs) != len(x):
            continue
        target[joint] = float(sum(c * value for c, value in zip(coeffs, x)))
    return target


def predict_pose_local(u: float, v: float, calib: CalibrationState, args) -> dict[str, float]:
    points = matching_calibration_points(calib.points, args)
    usable = []
    for point in points:
        joints = point.get("joints", {})
        pixel = point.get("pixel")
        if pixel is None or len(pixel) != 2:
            continue
        if all(joint in joints for joint in args.pose_joints):
            du = float(pixel[0]) - float(u)
            dv = float(pixel[1]) - float(v)
            dist = math.hypot(du, dv)
            usable.append((dist, point))

    if not usable:
        return {}
    usable.sort(key=lambda item: item[0])
    if usable[0][0] <= max(0.0, float(args.local_snap_px)):
        joints = usable[0][1]["joints"]
        return {joint: float(joints[joint]) for joint in args.pose_joints if joint in joints}

    radius = max(1.0, float(args.local_radius_px))
    radius_neighbors = [item for item in usable if item[0] <= radius]
    pool = radius_neighbors if radius_neighbors else usable
    k = min(max(1, int(args.local_pose_k)), len(pool))
    neighbors = pool[:k]
    power = max(0.1, float(args.local_pose_power))
    weights = [1.0 / ((dist + 1e-6) ** power) for dist, _ in neighbors]
    total_weight = sum(weights)
    if total_weight <= 0:
        return {}

    target: dict[str, float] = {}
    for joint in args.pose_joints:
        value = 0.0
        ok = True
        for weight, (_, point) in zip(weights, neighbors):
            joints = point.get("joints", {})
            if joint not in joints:
                ok = False
                break
            value += (weight / total_weight) * float(joints[joint])
        if ok:
            target[joint] = float(value)
    return target


def nearest_matching_point_distance(u: float, v: float, calib: CalibrationState, args) -> float | None:
    distances = []
    for point in matching_calibration_points(calib.points, args):
        pixel = point.get("pixel")
        if pixel is None or len(pixel) != 2:
            continue
        distances.append(math.hypot(float(pixel[0]) - float(u), float(pixel[1]) - float(v)))
    if not distances:
        return None
    return float(min(distances))


def robust_gripper_reference_data(
    calib: CalibrationState,
    args,
) -> tuple[tuple[float, float] | None, float, list[tuple[float, float, dict[str, Any]]], int, int]:
    raw_refs: list[tuple[float, float, dict[str, Any]]] = []
    for point in matching_calibration_points(calib.points, args):
        gripper_pixel = point.get("gripper_pixel")
        if gripper_pixel is None or len(gripper_pixel) != 2:
            continue
        raw_refs.append((float(gripper_pixel[0]), float(gripper_pixel[1]), point))

    min_conf = max(float(args.gripper_reference_min_conf), float(args.manual_save_min_gripper_conf))
    confidence_refs = [
        item
        for item in raw_refs
        if float(item[2].get("gripper_confidence", 1.0)) >= min_conf
    ]
    min_count = max(1, int(getattr(args, "gripper_reference_min_count", 1)))
    if len(confidence_refs) < min_count:
        return None, 0.0, [], len(raw_refs), len(raw_refs) - len(confidence_refs)
    cluster = robust_reference_cluster(
        [(item[0], item[1]) for item in confidence_refs],
        hard_limit_px=max(0.0, float(args.gripper_reference_max_outlier_px)),
        mad_scale=max(1.0, float(args.gripper_reference_mad_scale)),
        min_radius_px=max(1.0, float(args.gripper_reference_min_radius_px)),
    )
    if cluster is None:
        return None, 0.0, [], len(raw_refs), len(raw_refs) - len(confidence_refs)
    inliers = [confidence_refs[index] for index in cluster.inlier_indices]
    filtered = len(raw_refs) - len(inliers)
    return cluster.center_uv, float(cluster.radius_px), inliers, len(raw_refs), filtered


def nearest_replay_pose_point(
    u: float,
    v: float,
    calib: CalibrationState,
    args,
) -> tuple[dict[str, Any] | None, float | None]:
    if not bool(args.gripper_replay_nearest_pose):
        return None, None
    reference_uv, reference_radius, _refs, _raw_count, _filtered = robust_gripper_reference_data(calib, args)
    candidates: list[tuple[float, float, int, dict[str, Any]]] = []
    for point in matching_calibration_points(calib.points, args):
        source = str(point.get("source") or "")
        if (
            source not in ("manual_correction", "human_verified_alignment")
            and not (source.startswith("auto_gripper_refine") and bool(point.get("verified")))
        ):
            continue
        replay_fail_count = int(point.get("replay_fail_count") or 0)
        replay_fail_limit = max(1, int(getattr(args, "gripper_replay_max_fail_count", 1)))
        if replay_fail_count >= replay_fail_limit and not replay_failure_was_refine_only(point):
            continue
        pixel = point.get("pixel")
        joints = point.get("joints")
        if pixel is None or len(pixel) != 2 or not isinstance(joints, dict):
            continue
        if not all(joint in joints for joint in args.pose_joints):
            continue
        dist = math.hypot(float(pixel[0]) - float(u), float(pixel[1]) - float(v))
        if dist > max(0.0, float(args.gripper_replay_snap_px)):
            continue
        gripper_pixel = point.get("gripper_pixel")
        if gripper_pixel is not None and len(gripper_pixel) == 2:
            confidence = float(point.get("gripper_confidence", 1.0))
            if confidence < float(args.manual_save_min_gripper_conf):
                continue
            if reference_uv is None:
                gripper_error = 0.0
            else:
                gripper_error = math.hypot(
                    float(gripper_pixel[0]) - float(reference_uv[0]),
                    float(gripper_pixel[1]) - float(reference_uv[1]),
                )
                max_ref_error = max(float(args.manual_save_max_ref_error_px), 1.5 * reference_radius)
                if gripper_error > max_ref_error:
                    continue
        else:
            gripper_error = max(1.0, reference_radius) + 5.0
        candidates.append((dist, gripper_error, int(point.get("id") or 0), point))

    if not candidates:
        return None, None
    nearest_dist = min(item[0] for item in candidates)
    nearby_radius = max(1.0, float(getattr(args, "gripper_replay_tie_px", 2.0)))
    nearby = [item for item in candidates if item[0] <= nearest_dist + nearby_radius]
    best_dist, _gripper_error, _point_id, best_point = min(
        nearby,
        key=lambda item: (item[0], -item[2], item[1]),
    )
    return best_point, float(best_dist)


def replay_failure_was_refine_only(point: dict[str, Any]) -> bool:
    """Legacy Stage 3 failures must not invalidate a correctly saved replay pose."""
    reason = str(point.get("last_replay_failure") or "").lower()
    return any(
        marker in reason
        for marker in (
            "gripper_verify",
            "gripper_learned",
            "gripper_final",
            "gripper_ibvs",
            "gripper_stepwise",
            "gripper_greedy",
            "stage 3",
        )
    )


def mark_replay_pose_failure(
    point: dict[str, Any] | None,
    reason: str,
) -> str:
    if point is None:
        return "replay failure not marked: no replay point"
    point["replay_fail_count"] = int(point.get("replay_fail_count") or 0) + 1
    point["last_replay_failed_at"] = now_iso()
    point["last_replay_failure"] = str(reason)
    return f"marked replay point {point.get('id')} failed count={point['replay_fail_count']}"


def direct_replay_pose_points(
    calib: CalibrationState,
    args,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    required = list(args.pose_joints)
    for point in calib.points:
        source = str(point.get("source") or "")
        if source not in ("manual_correction", "human_verified_alignment"):
            continue
        joints = point.get("joints")
        if not isinstance(joints, dict):
            continue
        if not all(joint in joints for joint in required):
            continue
        points.append(point)
    return points


def select_direct_replay_pose_point(
    calib: CalibrationState,
    args,
) -> tuple[dict[str, Any] | None, str]:
    requested_id = int(getattr(args, "direct_replay_point_id", 0) or 0)
    points = direct_replay_pose_points(calib, args)
    if requested_id > 0:
        for point in points:
            if int(point.get("id") or 0) == requested_id:
                return point, f"id={requested_id}"
        return None, f"direct replay point id={requested_id} not found or missing pose joints"
    if not points:
        return None, "no manual/verified saved point with full pose joints"
    point = max(points, key=lambda item: int(item.get("id") or 0))
    return point, f"latest id={point.get('id')}"


def set_direct_replay_point_to_saved_point(
    point: dict[str, Any] | None,
    args,
) -> str:
    if point is None:
        return ""
    point_id = int(point.get("id") or 0)
    if point_id <= 0:
        return ""
    setattr(args, "direct_replay_point_id", point_id)
    return f"; P will replay saved point={point_id}"


def direct_replay_target_from_point(
    point: dict[str, Any],
    args,
) -> dict[str, float]:
    joints = point.get("joints")
    if not isinstance(joints, dict):
        return {}
    replay_joints = list(args.pose_joints)
    if bool(getattr(args, "direct_replay_include_gripper", True)) and "gripper.pos" in joints:
        replay_joints.append("gripper.pos")
    return {
        joint: float(joints[joint])
        for joint in replay_joints
        if joint in joints
    }


def send_pose_with_encoder_outer_correction(
    robot,
    target: dict[str, float],
    args,
    robot_lock: threading.RLock | None,
    stop_event: threading.Event | None,
    label: str,
    *,
    speed_deg_s: float | None = None,
    min_steps: int | None = None,
    wait_timeout_s: float | None = None,
    settle_threshold: float | None = None,
    outer_correction_enabled: bool | None = None,
) -> dict[str, Any]:
    send_kwargs: dict[str, Any] = {}
    if speed_deg_s is not None:
        send_kwargs["speed_deg_s"] = float(speed_deg_s)
    if min_steps is not None:
        send_kwargs["min_steps"] = int(min_steps)
    send_interpolated(
        robot,
        target,
        args,
        robot_lock,
        stop_event=stop_event,
        **send_kwargs,
    )
    if stop_event is not None and stop_event.is_set():
        return {"cancelled": True, "stage": "motion"}

    wait_joints = [joint for joint in target if joint != "gripper.pos"]
    threshold = float(args.joint_error_threshold if settle_threshold is None else settle_threshold)
    timeout_s = float(args.gripper_refine_wait_timeout_s if wait_timeout_s is None else wait_timeout_s)
    stable_ok, _stable_obs, stable_status = wait_until_observation_stable(
        robot,
        wait_joints,
        timeout_s,
        float(args.gripper_refine_stable_epsilon_deg),
        int(args.gripper_refine_stable_samples),
        robot_lock,
    )
    max_error, details = current_pose_error_details(robot, target, wait_joints, robot_lock)
    if stable_ok and max_error <= threshold:
        ok_wait = True
        wait_status = (
            f"encoder_reached max_error={max_error:.2f} threshold={threshold:.2f}; "
            f"{stable_status}"
        )
    elif stable_ok:
        ok_wait = False
        wait_status = (
            f"encoder_stable_offset max_error={max_error:.2f} threshold={threshold:.2f}; "
            f"{stable_status}"
        )
    else:
        ok_wait = False
        wait_status = (
            f"encoder_stable_timeout max_error={max_error:.2f} threshold={threshold:.2f}; "
            f"{stable_status}"
        )
    if stop_event is not None and stop_event.is_set():
        return {"cancelled": True, "stage": "wait"}
    raw_error = float(max_error)
    correction_notes: list[str] = []
    command_target = dict(target)

    use_outer_correction = (
        bool(getattr(args, "direct_replay_outer_correction", True))
        if outer_correction_enabled is None
        else bool(outer_correction_enabled)
    )
    if use_outer_correction and max_error > threshold:
        outer_iters = max(0, int(getattr(args, "direct_replay_outer_iters", 3)))
        outer_gain = max(0.0, float(getattr(args, "direct_replay_outer_gain", 1.0)))
        max_offset = max(0.0, float(getattr(args, "direct_replay_outer_max_offset", 6.0)))
        for outer_index in range(1, outer_iters + 1):
            observed = locked_observation(robot, robot_lock)
            changed = False
            offsets: list[str] = []
            for joint in wait_joints:
                if joint not in observed or joint not in target:
                    continue
                correction = float(target[joint]) - float(observed[joint])
                if abs(correction) <= threshold:
                    continue
                proposed = float(command_target[joint]) + outer_gain * correction
                if max_offset > 0:
                    proposed = min(max(proposed, float(target[joint]) - max_offset), float(target[joint]) + max_offset)
                if abs(proposed - float(command_target[joint])) > 1e-6:
                    command_target[joint] = proposed
                    changed = True
                    offsets.append(f"{joint.split('.')[0]}_cmd={proposed:.2f}")
            if not changed:
                correction_notes.append(f"outer{outer_index}=no_command_change")
                break
            print(
                f"{label} outer correction {outer_index}/{outer_iters}: "
                + " ".join(offsets),
                flush=True,
            )
            send_interpolated(
                robot,
                command_target,
                args,
                robot_lock,
                speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
                min_steps=int(args.gripper_refine_min_move_steps),
                stop_event=stop_event,
            )
            if stop_event is not None and stop_event.is_set():
                return {"cancelled": True, "stage": "outer correction"}
            stable_ok, _stable_obs, stable_status = wait_until_observation_stable(
                robot,
                wait_joints,
                float(args.gripper_refine_wait_timeout_s),
                float(args.gripper_refine_stable_epsilon_deg),
                int(args.gripper_refine_stable_samples),
                robot_lock,
            )
            max_error, details = current_pose_error_details(robot, target, wait_joints, robot_lock)
            wait_status = f"outer{outer_index} {'stable' if stable_ok else 'unstable'}; {stable_status}"
            command_error = max(
                (abs(float(command_target[joint]) - float(target[joint])) for joint in wait_joints if joint in command_target),
                default=0.0,
            )
            correction_notes.append(
                f"outer{outer_index}: max_joint_error={max_error:.2f} max_cmd_offset={command_error:.2f}"
            )
            if max_error <= threshold:
                ok_wait = True
                break
            ok_wait = False

    return {
        "cancelled": False,
        "ok_wait": bool(ok_wait),
        "wait_status": wait_status,
        "max_error": float(max_error),
        "details": details,
        "raw_error": raw_error,
        "correction_notes": correction_notes,
        "command_target": command_target,
        "wait_joints": wait_joints,
    }

def run_direct_pose_replay_test(
    robot,
    calib: CalibrationState,
    args,
    robot_lock: threading.RLock | None = None,
    stop_event: threading.Event | None = None,
    gripper_capture=None,
    gripper_model=None,
    gripper_names: dict[int, str] | None = None,
    gripper_camera_lock: threading.RLock | None = None,
) -> str:
    point, point_status = select_direct_replay_pose_point(calib, args)
    if point is None:
        return "direct_replay=skipped " + point_status
    target = direct_replay_target_from_point(point, args)
    if not target:
        return f"direct_replay=skipped point {point.get('id')} has no replayable joints"

    pixel = point.get("pixel")
    pixel_text = "pixel=n/a"
    if pixel is not None and len(pixel) == 2:
        pixel_text = f"pixel=({float(pixel[0]):.1f},{float(pixel[1]):.1f})"
    joint_text = " ".join(f"{joint.split('.')[0]}={value:.2f}" for joint, value in target.items())
    print(
        f"DIRECT_REPLAY start point={point.get('id')} {point_status} source={point.get('source')} "
        f"{pixel_text}; target {joint_text}",
        flush=True,
    )

    replay_result = send_pose_with_encoder_outer_correction(
        robot,
        target,
        args,
        robot_lock,
        stop_event=stop_event,
        label="DIRECT_REPLAY",
        speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
        min_steps=int(args.gripper_refine_min_move_steps),
        wait_timeout_s=float(args.gripper_refine_wait_timeout_s),
        settle_threshold=float(args.joint_error_threshold),
    )
    if replay_result.get("cancelled"):
        return f"STOP: direct_replay point={point.get('id')} cancelled during {replay_result.get('stage')}"
    ok_wait = bool(replay_result["ok_wait"])
    wait_status = str(replay_result["wait_status"])
    max_error = float(replay_result["max_error"])
    details = replay_result["details"]
    raw_error = float(replay_result["raw_error"])
    correction_notes = list(replay_result["correction_notes"])
    wait_joints = list(replay_result["wait_joints"])
    gripper_error = None
    if "gripper.pos" in target:
        obs = locked_observation(robot, robot_lock)
        if "gripper.pos" in obs:
            gripper_error = abs(float(target["gripper.pos"]) - float(obs["gripper.pos"]))
    detail_text = format_pose_error_details(details, limit=6)
    gripper_text = "" if gripper_error is None else f"; gripper_err={gripper_error:.3f}"
    raw_text = f"; raw_max_joint_error={raw_error:.2f}"
    correction_text = ""
    if correction_notes:
        correction_text = "; outer_correction " + " | ".join(correction_notes)
    visual_text = ""
    saved_error_value: float | None = None
    saved_gripper = point.get("gripper_pixel")
    if (
        gripper_capture is not None
        and gripper_model is not None
        and saved_gripper is not None
        and len(saved_gripper) == 2
    ):
        overhead_uv = None
        pixel = point.get("pixel")
        if pixel is not None and len(pixel) == 2:
            overhead_uv = (float(pixel[0]), float(pixel[1]))
        observation = observe_gripper_error_median(
            gripper_capture,
            gripper_model,
            gripper_names or {},
            calib,
            args,
            gripper_camera_lock,
            overhead_uv,
        )
        if observation.detection is None:
            visual_text = f"; gripper_visual=missing {observation.status}"
        else:
            center = observation.detection.center_uv
            saved_error = math.hypot(
                float(center[0]) - float(saved_gripper[0]),
                float(center[1]) - float(saved_gripper[1]),
            )
            saved_error_value = float(saved_error)
            visual_text = (
                f"; saved_gripper_error_px={saved_error:.1f} "
                f"saved=({float(saved_gripper[0]):.1f},{float(saved_gripper[1]):.1f}) "
                f"now=({float(center[0]):.1f},{float(center[1]):.1f}) "
                f"ref_dist={observation.dist_px:.1f}px"
            )
    joint_ok = bool(ok_wait and max_error <= float(args.joint_error_threshold))
    visual_ok = saved_error_value is not None and gripper_error_within_tolerance(saved_error_value, args)
    visual_measured = saved_error_value is not None
    if joint_ok and visual_measured and not visual_ok:
        prefix = "DONE_JOINTS_VISUAL_MISMATCH"
    elif joint_ok:
        prefix = "DONE"
    elif visual_ok:
        prefix = "DONE_VISUAL"
    else:
        prefix = "STOP"
    return (
        f"{prefix}: direct_replay point={point.get('id')} {pixel_text} "
        f"max_joint_error={max_error:.2f} threshold={float(args.joint_error_threshold):.2f}"
        f"{raw_text}{gripper_text}{visual_text}{correction_text}; {wait_status}; {detail_text}"
    )


def gripper_reference_pixel(
    calib: CalibrationState,
    args,
    frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    overhead_uv: tuple[float, float] | None = None,
) -> tuple[tuple[float, float] | None, str]:
    if args.gripper_reference_mode == "frame_center":
        if frame_shape is None:
            return None, "frame_center unavailable"
        height, width = frame_shape[:2]
        return (width / 2.0, height / 2.0), "frame_center"

    robust_center, robust_radius, refs, raw_count, filtered_count = robust_gripper_reference_data(calib, args)
    if not refs or robust_center is None:
        return None, "recorded reference missing; press C again with gripper detection visible"
    filter_text = f" inliers={len(refs)}/{raw_count} radius={robust_radius:.1f}px"

    if args.gripper_reference_mode == "recorded_local" and overhead_uv is not None:
        usable = []
        for ref_u, ref_v, point in refs:
            pixel = point.get("pixel")
            if pixel is None or len(pixel) != 2:
                continue
            dist = math.hypot(float(pixel[0]) - float(overhead_uv[0]), float(pixel[1]) - float(overhead_uv[1]))
            usable.append((dist, ref_u, ref_v))
        if usable:
            usable.sort(key=lambda item: item[0])
            if usable[0][0] <= max(0.0, float(args.local_snap_px)):
                snap_limit = max(0.0, float(args.local_snap_px))
                snap_neighbors = [item for item in usable if item[0] <= snap_limit]
                k_snap = min(max(1, int(args.local_pose_k)), len(snap_neighbors))
                snap_neighbors = snap_neighbors[:k_snap]
                ref_u = float(np.median([item[1] for item in snap_neighbors]))
                ref_v = float(np.median([item[2] for item in snap_neighbors]))
                return (
                    (ref_u, ref_v),
                    f"recorded_local snap k={k_snap} nearest={usable[0][0]:.1f}px{filter_text}",
                )
            k = min(max(1, int(args.local_pose_k)), len(usable))
            neighbors = usable[:k]
            power = max(0.1, float(args.local_pose_power))
            weights = [1.0 / ((dist + 1e-6) ** power) for dist, _, _ in neighbors]
            total_weight = sum(weights)
            if total_weight > 0:
                ref_u = sum((weight / total_weight) * item[1] for weight, item in zip(weights, neighbors))
                ref_v = sum((weight / total_weight) * item[2] for weight, item in zip(weights, neighbors))
                return (float(ref_u), float(ref_v)), f"recorded_local k={k}{filter_text}"

    return robust_center, f"recorded_robust{filter_text}"


def gripper_reference_circle(
    calib: CalibrationState,
    args,
    frame_shape: tuple[int, int, int] | tuple[int, int] | None = None,
    overhead_uv: tuple[float, float] | None = None,
) -> tuple[tuple[float, float] | None, float, str]:
    if args.gripper_reference_mode == "frame_center":
        if frame_shape is None:
            return None, 0.0, "frame_center unavailable"
        height, width = frame_shape[:2]
        radius = max(1.0, float(args.gripper_center_tolerance_px))
        return (width / 2.0, height / 2.0), radius, f"frame_center radius={radius:.1f}px"
    center, source = gripper_reference_pixel(calib, args, frame_shape, overhead_uv)
    _robust_center, robust_radius, refs, raw_count, filtered = robust_gripper_reference_data(calib, args)
    if center is None:
        return None, 0.0, source
    radius = max(
        1.0,
        float(args.gripper_center_tolerance_px),
        float(args.gripper_reference_min_radius_px),
        float(robust_radius),
    )
    return center, radius, f"{source}; circle_radius={radius:.1f}px raw={raw_count} filtered={filtered}"


def count_gripper_reference_points(calib: CalibrationState, args) -> int:
    count = 0
    for point in matching_calibration_points(calib.points, args):
        gripper_pixel = point.get("gripper_pixel")
        if gripper_pixel is not None and len(gripper_pixel) == 2:
            count += 1
    return count


def matching_gripper_correction_samples(calib: CalibrationState, args) -> list[dict[str, Any]]:
    samples = []
    class_agnostic = use_class_agnostic_points(args)
    allowed_roles = set(str(role) for role in (getattr(args, "gripper_learned_sample_roles", []) or []))
    for sample in calib.gripper_correction_samples:
        if sample.get("usable_for_xy_servo") is not True:
            continue
        if allowed_roles and str(sample.get("sample_role") or "manual_correction") not in allowed_roles:
            continue
        if bool(getattr(args, "gripper_learned_require_linked_sample", False)) and int(sample.get("after_point_id") or 0) <= 0:
            continue
        if not class_agnostic and sample.get("target_class") not in (None, args.target_class):
            continue
        if sample.get("point_kind") not in (None, active_target_point_kind(args.target_class, args)):
            continue
        error_uv = sample.get("error_uv")
        delta_xy = sample.get("delta_xy")
        if error_uv is None or len(error_uv) != 2 or not isinstance(delta_xy, dict):
            continue
        if all(joint in delta_xy for joint in args.map_joints):
            ok_quality, _quality_reason = gripper_correction_sample_quality(sample, args)
            if not ok_quality:
                continue
            samples.append(sample)
    return samples


def predict_gripper_learned_delta(
    err_u: float,
    err_v: float,
    calib: CalibrationState,
    args,
    *,
    current_xy: dict[str, float] | None = None,
    overhead_uv: tuple[float, float] | None = None,
) -> tuple[dict[str, float], str]:
    samples = matching_gripper_correction_samples(calib, args)
    if not samples:
        return {}, "learned gripper: no correction samples; press B before manual correction, then C after alignment"

    error = np.array([float(err_u), float(err_v)], dtype=np.float64)
    candidates = []
    overhead_radius = max(0.0, float(getattr(args, "gripper_learned_local_overhead_radius_px", 70.0)))
    joint_radius = max(0.0, float(getattr(args, "gripper_learned_local_joint_radius_deg", 8.0)))
    for sample in samples:
        sample_error = np.array([float(sample["error_uv"][0]), float(sample["error_uv"][1])], dtype=np.float64)
        norm2 = float(np.dot(sample_error, sample_error))
        if norm2 <= 1e-6:
            continue
        distance = float(np.linalg.norm(error - sample_error))
        projection = float(np.dot(error, sample_error) / norm2)
        delta_xy = sample.get("delta_xy", {})
        sample_delta = np.array([float(delta_xy[joint]) for joint in args.map_joints], dtype=np.float64)
        overhead_distance = float("inf")
        if overhead_uv is not None:
            sample_overhead = sample.get("before_overhead_pixel") or sample.get("after_overhead_pixel")
            if sample_overhead is not None and len(sample_overhead) == 2:
                overhead_distance = math.hypot(
                    float(sample_overhead[0]) - float(overhead_uv[0]),
                    float(sample_overhead[1]) - float(overhead_uv[1]),
                )
        joint_distance = float("inf")
        if current_xy is not None:
            sample_xy = sample.get("before_xy") or sample.get("after_xy")
            if isinstance(sample_xy, dict) and all(joint in sample_xy and joint in current_xy for joint in args.map_joints):
                joint_distance = math.hypot(
                    *(float(sample_xy[joint]) - float(current_xy[joint]) for joint in args.map_joints)
                )
        overhead_local = overhead_uv is not None and overhead_distance <= overhead_radius
        joint_local = current_xy is not None and joint_distance <= joint_radius
        candidates.append(
            {
                "error_distance": distance,
                "overhead_distance": overhead_distance,
                "joint_distance": joint_distance,
                "delta": projection * sample_delta,
                "id": sample.get("id"),
                "overhead_local": overhead_local,
                "joint_local": joint_local,
            }
        )
    if not candidates:
        return {}, "learned gripper: correction samples unusable"

    local_mode = "global"
    usable = candidates
    both_local = [item for item in candidates if item["overhead_local"] and item["joint_local"]]
    overhead_local = [item for item in candidates if item["overhead_local"]]
    joint_local = [item for item in candidates if item["joint_local"]]
    if both_local:
        usable = both_local
        local_mode = "overhead+joint"
    elif overhead_local:
        usable = overhead_local
        local_mode = "overhead"
    elif joint_local:
        usable = joint_local
        local_mode = "joint"

    usable.sort(key=lambda item: (float(item["error_distance"]), float(item["overhead_distance"]), float(item["joint_distance"])))
    k = min(max(1, int(args.gripper_learned_k)), len(usable))
    neighbors = usable[:k]
    power = max(0.1, float(args.gripper_learned_power))
    weights = [1.0 / ((float(item["error_distance"]) + 1e-6) ** power) for item in neighbors]
    total_weight = sum(weights)
    if total_weight <= 0:
        return {}, "learned gripper: invalid sample weights"
    delta = sum((weight / total_weight) * item["delta"] for weight, item in zip(weights, neighbors))
    step_limit = adaptive_gripper_step_limit(float(np.linalg.norm(error)), args)
    delta = scale_joint_delta(delta, step_limit)
    result = {joint: float(delta[index]) for index, joint in enumerate(args.map_joints)}
    ids = ",".join(str(item["id"]) for item in neighbors)
    return result, (
        f"learned gripper: k={k}/{len(samples)} samples ids={ids} local={local_mode} "
        f"pool={len(usable)} step_limit={step_limit:.2f}deg"
    )


def adaptive_gripper_full_pose_step_limit(dist_px: float, args) -> float:
    max_step = abs(float(getattr(args, "gripper_learned_full_max_step_deg", 6.0)))
    min_step = min(max_step, abs(float(getattr(args, "gripper_learned_full_min_step_deg", 0.5))))
    if max_step <= 0:
        return 0.0
    tolerance = max(1.0, float(args.gripper_center_tolerance_px))
    full_error = max(tolerance + 1.0, float(args.gripper_full_step_error_px))
    alpha = (float(dist_px) - tolerance) / (full_error - tolerance)
    alpha = min(1.0, max(0.0, alpha))
    return min_step + (max_step - min_step) * alpha


def clamp_scalar(value: float, limit: float) -> float:
    limit_abs = abs(float(limit))
    if limit_abs <= 0:
        return 0.0
    return min(max(float(value), -limit_abs), limit_abs)


def learned_full_pose_joint_limits(args) -> dict[str, float]:
    return {
        "shoulder_pan.pos": float(getattr(args, "gripper_learned_full_max_pan_deg", 2.5)),
        "shoulder_lift.pos": float(getattr(args, "gripper_learned_full_max_lift_deg", 4.0)),
        "elbow_flex.pos": float(getattr(args, "gripper_learned_full_max_elbow_deg", 4.0)),
        "wrist_flex.pos": float(getattr(args, "gripper_learned_full_max_wrist_flex_deg", 2.5)),
        "wrist_roll.pos": float(getattr(args, "gripper_learned_full_max_wrist_roll_deg", 2.0)),
    }


def clamp_learned_full_pose_delta(delta: dict[str, float], args) -> dict[str, float]:
    limits = learned_full_pose_joint_limits(args)
    return {
        joint: clamp_scalar(float(value), limits.get(joint, float(getattr(args, "gripper_learned_full_max_step_deg", 6.0))))
        for joint, value in delta.items()
    }


def full_pose_delta_max_abs(delta: dict[str, float]) -> float:
    return max((abs(float(value)) for value in delta.values()), default=0.0)


def full_pose_actual_motion_quality(
    commanded_delta: dict[str, float],
    actual_delta: dict[str, float],
    args,
) -> tuple[bool, str]:
    joints = [joint for joint in args.pose_joints if joint in commanded_delta]
    command_vec = np.array([float(commanded_delta.get(joint, 0.0)) for joint in joints], dtype=np.float64)
    actual_vec = np.array([float(actual_delta.get(joint, 0.0)) for joint in joints], dtype=np.float64)
    command_norm = float(np.linalg.norm(command_vec))
    actual_norm = float(np.linalg.norm(actual_vec))
    if command_norm <= 1e-9:
        return True, "actual_motion skipped: zero command"
    min_actual = max(0.0, float(getattr(args, "gripper_min_actual_motion_deg", 0.12)))
    if actual_norm < min_actual:
        return False, f"actual_motion rejected: actual_norm={actual_norm:.3f}deg<{min_actual:.3f}deg"
    projected = float(np.dot(actual_vec, command_vec) / command_norm)
    min_fraction = max(0.0, float(getattr(args, "gripper_min_actual_motion_fraction", 0.18)))
    required_projected = max(min_actual, command_norm * min_fraction)
    lateral_sq = max(0.0, actual_norm * actual_norm - projected * projected)
    lateral = math.sqrt(lateral_sq)
    if projected < required_projected:
        return (
            False,
            f"actual_motion rejected: projected={projected:.3f}deg<{required_projected:.3f}deg "
            f"actual_norm={actual_norm:.3f}deg lateral={lateral:.3f}deg",
        )
    return (
        True,
        f"actual_motion ok actual_norm={actual_norm:.3f}deg projected={projected:.3f}deg lateral={lateral:.3f}deg",
    )


def classify_gripper_visual_step(
    before_dist: float,
    after_dist: float,
    accept_improve_px: float,
    farther_px: float,
) -> tuple[str, float]:
    """Classify a move from image distance only; encoder direction is diagnostic."""
    if not math.isfinite(float(before_dist)) or not math.isfinite(float(after_dist)):
        return "invalid", float("-inf")
    improved_by = float(before_dist) - float(after_dist)
    if improved_by >= max(0.0, float(accept_improve_px)):
        return "closer", improved_by
    if -improved_by > max(0.0, float(farther_px)):
        return "farther", improved_by
    return "neutral", improved_by


def scale_full_pose_delta_to_limit(delta: dict[str, float], limit: float) -> dict[str, float]:
    max_abs = full_pose_delta_max_abs(delta)
    limit_abs = abs(float(limit))
    if max_abs <= 1e-9 or limit_abs <= 0.0:
        return {joint: 0.0 for joint in delta}
    scale = min(1.0, limit_abs / max_abs)
    return {joint: float(value) * scale for joint, value in delta.items()}


def learned_full_pose_search_candidates(
    primary_delta: dict[str, float],
    args,
    preferred_delta: dict[str, float] | None = None,
) -> list[tuple[str, dict[str, float]]]:
    step = max(0.0, float(getattr(args, "gripper_learned_search_step_deg", 0.45)))
    axis_step = max(0.0, float(getattr(args, "gripper_learned_search_axis_step_deg", 0.25)))
    max_candidates = max(1, int(getattr(args, "gripper_learned_search_max_candidates", 8)))
    candidates: list[tuple[str, dict[str, float]]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()

    def add(label: str, delta: dict[str, float]) -> None:
        if len(candidates) >= max_candidates:
            return
        full_delta = {joint: float(delta.get(joint, 0.0)) for joint in args.pose_joints}
        full_delta = clamp_learned_full_pose_delta(full_delta, args)
        max_abs = full_pose_delta_max_abs(full_delta)
        if max_abs <= 1e-6:
            return
        if not label.startswith("axis_"):
            min_step = min(
                step,
                max(0.0, float(getattr(args, "gripper_learned_full_min_step_deg", 0.5))),
            )
            if 0.0 < max_abs < min_step:
                full_delta = {
                    joint: float(value) * min_step / max_abs
                    for joint, value in full_delta.items()
                }
                full_delta = clamp_learned_full_pose_delta(full_delta, args)
        key = tuple((joint, round(float(full_delta[joint]), 4)) for joint in args.pose_joints)
        if key in seen:
            return
        seen.add(key)
        candidates.append((label, full_delta))

    if preferred_delta:
        preferred = scale_full_pose_delta_to_limit(
            {joint: float(preferred_delta.get(joint, 0.0)) for joint in args.pose_joints},
            step,
        )
        add("continue_last", preferred)
        add("reverse_last", {joint: -float(value) for joint, value in preferred.items()})

    axis_joints = (
        list(getattr(args, "gripper_learned_search_axis_joints", []) or [])
        if bool(getattr(args, "gripper_learned_allow_axis_probes", True))
        else []
    )
    axis_first = bool(getattr(args, "gripper_learned_search_axis_first", False))

    def add_axis_candidates() -> None:
        for joint in axis_joints:
            if joint not in args.pose_joints:
                continue
            add(f"axis_{joint.split('.')[0]}_pos", {joint: axis_step})
            add(f"axis_{joint.split('.')[0]}_neg", {joint: -axis_step})

    if axis_first:
        add_axis_candidates()

    if primary_delta:
        capped = scale_full_pose_delta_to_limit(
            {joint: float(primary_delta.get(joint, 0.0)) for joint in args.pose_joints},
            step,
        )
        add("learned_small", capped)
        add("learned_reverse", {joint: -float(value) for joint, value in capped.items()})
        add("learned_half", {joint: 0.5 * float(value) for joint, value in capped.items()})
        add("learned_reverse_half", {joint: -0.5 * float(value) for joint, value in capped.items()})

    if not axis_first:
        add_axis_candidates()
    return candidates


def learned_clearance_pose(
    current_pose: dict[str, float],
    safe_pose: dict[str, float] | None,
    args,
) -> tuple[dict[str, float], str]:
    if not bool(getattr(args, "gripper_learned_clearance", True)):
        return dict(current_pose), "clearance disabled"
    if not safe_pose:
        return dict(current_pose), "clearance skipped: safe_pose missing"
    max_deg = max(0.0, float(getattr(args, "gripper_learned_clearance_max_deg", 2.0)))
    target = dict(current_pose)
    offsets: list[str] = []
    for joint in getattr(args, "gripper_learned_clearance_joints", []) or []:
        if joint not in current_pose or joint not in safe_pose:
            continue
        delta = float(safe_pose[joint]) - float(current_pose[joint])
        step = clamp_scalar(delta, max_deg)
        if abs(step) <= 1e-6:
            continue
        target[joint] = float(current_pose[joint]) + step
        offsets.append(f"{joint.split('.')[0]}={step:.2f}")
    if not offsets:
        return target, "clearance skipped: no usable offset"
    return target, "clearance " + " ".join(offsets)


def sample_delta_pose(sample: dict[str, Any], args) -> dict[str, float]:
    delta_pose = sample.get("delta_pose")
    if isinstance(delta_pose, dict) and all(joint in delta_pose for joint in args.pose_joints):
        return {joint: float(delta_pose[joint]) for joint in args.pose_joints}
    before_joints = sample.get("before_joints")
    after_joints = sample.get("after_joints")
    if not isinstance(before_joints, dict) or not isinstance(after_joints, dict):
        return {}
    if not all(joint in before_joints and joint in after_joints for joint in args.pose_joints):
        return {}
    return {
        joint: float(after_joints[joint]) - float(before_joints[joint])
        for joint in args.pose_joints
    }


def gripper_param_gain_from_error_delta(
    error_uv: tuple[float, float] | list[float],
    delta_pose: dict[str, float],
    args,
) -> dict[str, list[float]]:
    if len(error_uv) != 2:
        return {}
    err_u = float(error_uv[0])
    err_v = float(error_uv[1])
    norm2 = err_u * err_u + err_v * err_v
    if norm2 <= 1e-6:
        return {}
    gain: dict[str, list[float]] = {}
    for joint in args.pose_joints:
        if joint not in delta_pose:
            return {}
        delta = float(delta_pose[joint])
        gain[joint] = [float(delta * err_u / norm2), float(delta * err_v / norm2)]
    return gain


def gripper_param_predict_delta(
    model: dict[str, Any],
    err_u: float,
    err_v: float,
    args,
) -> tuple[dict[str, float], str]:
    gain = model.get("gain") if isinstance(model, dict) else None
    if not isinstance(gain, dict):
        return {}, "param_model missing gain"
    delta: dict[str, float] = {}
    for joint in args.pose_joints:
        values = gain.get(joint)
        if not isinstance(values, list) or len(values) != 2:
            return {}, f"param_model missing {joint}"
        delta[joint] = float(values[0]) * float(err_u) + float(values[1]) * float(err_v)
    step_limit = adaptive_gripper_full_pose_step_limit(math.hypot(float(err_u), float(err_v)), args)
    max_abs = max((abs(value) for value in delta.values()), default=0.0)
    if max_abs > step_limit > 0:
        scale = step_limit / max_abs
        delta = {joint: float(value) * scale for joint, value in delta.items()}
    delta = clamp_learned_full_pose_delta(delta, args)
    count = int(model.get("sample_count") or 0)
    return delta, f"param_model n={count} step_limit={step_limit:.2f}deg"


def build_gripper_learned_param_model_from_samples(
    calib: CalibrationState,
    args,
) -> tuple[dict[str, Any], str]:
    gains: list[dict[str, list[float]]] = []
    sample_ids: list[int] = []
    for sample in matching_gripper_correction_samples(calib, args):
        error_uv = sample.get("error_uv")
        delta_pose = sample_delta_pose(sample, args)
        if not error_uv or len(error_uv) != 2 or not delta_pose:
            continue
        gain = gripper_param_gain_from_error_delta(error_uv, delta_pose, args)
        if not gain:
            continue
        gains.append(gain)
        try:
            sample_ids.append(int(sample.get("id") or 0))
        except (TypeError, ValueError):
            pass
    if not gains:
        return {}, "param_model seed skipped: no usable correction samples"
    averaged: dict[str, list[float]] = {}
    for joint in args.pose_joints:
        averaged[joint] = [
            float(sum(gain[joint][0] for gain in gains) / len(gains)),
            float(sum(gain[joint][1] for gain in gains) / len(gains)),
        ]
    model = {
        "version": 1,
        "updated_at": now_iso(),
        "sample_count": len(gains),
        "gain": averaged,
        "history": [],
        "seed_source": "gripper_correction_samples",
        "seed_sample_ids": sample_ids[-25:],
        "last_update": {
            "source": "seed_from_existing_samples",
            "usable_samples": len(gains),
        },
    }
    return model, f"param_model seeded from {len(gains)} correction samples"


def ensure_gripper_learned_param_model(calib: CalibrationState, args) -> str:
    model = calib.gripper_learned_param_model
    if isinstance(model, dict) and isinstance(model.get("gain"), dict) and int(model.get("sample_count") or 0) > 0:
        return f"param_model loaded n={int(model.get('sample_count') or 0)}"
    seeded, status = build_gripper_learned_param_model_from_samples(calib, args)
    if seeded:
        calib.gripper_learned_param_model = seeded
    return status


def update_gripper_learned_param_model(
    calib: CalibrationState,
    args,
    *,
    error_uv: tuple[float, float],
    delta_pose: dict[str, float],
    before_dist_px: float,
    after_dist_px: float,
    source: str,
    overhead_uv: tuple[float, float] | None = None,
) -> tuple[bool, str]:
    if not bool(getattr(args, "gripper_learned_auto_update_params", True)):
        return False, "param_model update disabled"
    if not math.isfinite(float(before_dist_px)) or not math.isfinite(float(after_dist_px)):
        return False, "param_model update skipped: invalid distance"
    if float(after_dist_px) > gripper_ibvs_tolerance_px(calib, args):
        return False, (
            f"param_model update skipped: after_dist {float(after_dist_px):.1f}px "
            f"> tolerance {gripper_ibvs_tolerance_px(calib, args):.1f}px"
        )
    if float(before_dist_px) - float(after_dist_px) < max(0.0, float(getattr(args, "gripper_learned_min_improve_px", 1.0))):
        return False, "param_model update skipped: move did not improve enough"
    new_gain = gripper_param_gain_from_error_delta(error_uv, delta_pose, args)
    if not new_gain:
        return False, "param_model update skipped: cannot compute gain"
    ensure_gripper_learned_param_model(calib, args)
    model = calib.gripper_learned_param_model if isinstance(calib.gripper_learned_param_model, dict) else {}
    old_gain = model.get("gain") if isinstance(model.get("gain"), dict) else {}
    old_count = max(0, int(model.get("sample_count") or 0))
    if not old_gain or old_count <= 0:
        averaged = new_gain
        new_count = 1
    else:
        new_count = old_count + 1
        averaged = {}
        for joint in args.pose_joints:
            old_values = old_gain.get(joint, [0.0, 0.0])
            averaged[joint] = [
                float((float(old_values[0]) * old_count + float(new_gain[joint][0])) / new_count),
                float((float(old_values[1]) * old_count + float(new_gain[joint][1])) / new_count),
            ]
    history_limit = max(0, int(getattr(args, "gripper_learned_param_history", 3)))
    snapshot = {
        "updated_at": model.get("updated_at"),
        "sample_count": old_count,
        "gain": old_gain,
        "last_update": model.get("last_update"),
    }
    history = list(model.get("history") or [])
    if old_gain and history_limit > 0:
        history = [snapshot] + history
    history = history[:history_limit]
    calib.gripper_learned_param_model = {
        "version": 1,
        "updated_at": now_iso(),
        "sample_count": new_count,
        "gain": averaged,
        "history": history,
        "last_update": {
            "source": source,
            "before_dist_px": float(before_dist_px),
            "after_dist_px": float(after_dist_px),
            "improved_by_px": float(before_dist_px) - float(after_dist_px),
            "error_uv": [float(error_uv[0]), float(error_uv[1])],
            "delta_pose": {joint: float(delta_pose.get(joint, 0.0)) for joint in args.pose_joints},
            "overhead_uv": [float(overhead_uv[0]), float(overhead_uv[1])] if overhead_uv is not None else None,
        },
    }
    return True, f"param_model updated n={old_count}->{new_count} source={source}"


def apply_coarse_pose_bias(
    pose_target: dict[str, float],
    calib: CalibrationState,
    args,
    *,
    source: str,
) -> tuple[dict[str, float], str]:
    if not bool(getattr(args, "coarse_bias_enable", True)):
        return dict(pose_target), "coarse_bias disabled"
    model = calib.coarse_pose_bias_model
    if not isinstance(model, dict) or not isinstance(model.get("bias"), dict):
        return dict(pose_target), "coarse_bias unavailable"
    bias = model.get("bias") or {}
    max_deg = max(0.0, float(getattr(args, "coarse_bias_max_joint_deg", 4.0)))
    adjusted = dict(pose_target)
    applied: list[str] = []
    for joint in args.pose_joints:
        if joint not in adjusted or joint not in bias:
            continue
        delta = clamp_scalar(float(bias[joint]), max_deg)
        if abs(delta) <= 1e-6:
            continue
        adjusted[joint] = float(adjusted[joint]) + delta
        applied.append(f"{joint.split('.')[0]}={delta:+.2f}")
    count = int(model.get("sample_count") or 0)
    if not applied:
        return adjusted, f"coarse_bias n={count} no usable joint for {source}"
    return adjusted, f"coarse_bias n={count} applied({source}) " + " ".join(applied)


def update_coarse_pose_bias_model(
    calib: CalibrationState,
    args,
    *,
    before_joints: dict[str, float],
    after_joints: dict[str, float],
    before_dist_px: float | None,
    after_dist_px: float | None,
    source: str,
    overhead_uv: tuple[float, float] | None = None,
) -> tuple[bool, str]:
    if not bool(getattr(args, "coarse_bias_auto_update", True)):
        return False, "coarse_bias update disabled"
    if not all(joint in before_joints and joint in after_joints for joint in args.pose_joints):
        return False, "coarse_bias update skipped: pose joints missing"
    if after_dist_px is not None and math.isfinite(float(after_dist_px)):
        if float(after_dist_px) > gripper_ibvs_tolerance_px(calib, args):
            return False, (
                f"coarse_bias update skipped: after_dist {float(after_dist_px):.1f}px "
                f"> tolerance {gripper_ibvs_tolerance_px(calib, args):.1f}px"
            )
    if (
        before_dist_px is not None
        and after_dist_px is not None
        and math.isfinite(float(before_dist_px))
        and math.isfinite(float(after_dist_px))
    ):
        min_improve = max(0.0, float(getattr(args, "coarse_bias_min_improve_px", 1.0)))
        if float(before_dist_px) - float(after_dist_px) < min_improve:
            return False, "coarse_bias update skipped: refine did not improve enough"
    measured = {
        joint: float(after_joints[joint]) - float(before_joints[joint])
        for joint in args.pose_joints
    }
    max_deg = max(0.0, float(getattr(args, "coarse_bias_max_joint_deg", 4.0)))
    measured = {joint: clamp_scalar(value, max_deg) for joint, value in measured.items()}
    model = calib.coarse_pose_bias_model if isinstance(calib.coarse_pose_bias_model, dict) else {}
    old_bias = model.get("bias") if isinstance(model.get("bias"), dict) else {}
    old_count = max(0, int(model.get("sample_count") or 0))
    if old_bias and old_count > 0:
        new_count = old_count + 1
        averaged = {
            joint: float((float(old_bias.get(joint, 0.0)) * old_count + measured[joint]) / new_count)
            for joint in args.pose_joints
        }
    else:
        new_count = 1
        averaged = measured
    history_limit = max(0, int(getattr(args, "coarse_bias_history", 3)))
    history = list(model.get("history") or [])
    if old_bias and history_limit > 0:
        history = [
            {
                "updated_at": model.get("updated_at"),
                "sample_count": old_count,
                "bias": old_bias,
                "last_update": model.get("last_update"),
            }
        ] + history
    history = history[:history_limit]
    calib.coarse_pose_bias_model = {
        "version": 1,
        "updated_at": now_iso(),
        "sample_count": new_count,
        "bias": averaged,
        "history": history,
        "last_update": {
            "source": source,
            "before_dist_px": float(before_dist_px) if before_dist_px is not None and math.isfinite(float(before_dist_px)) else None,
            "after_dist_px": float(after_dist_px) if after_dist_px is not None and math.isfinite(float(after_dist_px)) else None,
            "measured_delta_pose": measured,
            "overhead_uv": [float(overhead_uv[0]), float(overhead_uv[1])] if overhead_uv is not None else None,
        },
    }
    return True, f"coarse_bias updated n={old_count}->{new_count} source={source}"


def predict_gripper_learned_full_pose_delta(
    err_u: float,
    err_v: float,
    calib: CalibrationState,
    args,
    *,
    current_pose: dict[str, float] | None = None,
    overhead_uv: tuple[float, float] | None = None,
) -> tuple[dict[str, float], str]:
    if bool(getattr(args, "gripper_learned_use_param_model", True)):
        seed_status = ensure_gripper_learned_param_model(calib, args)
        param_delta, param_status = gripper_param_predict_delta(
            calib.gripper_learned_param_model,
            err_u,
            err_v,
            args,
        )
        if param_delta:
            return param_delta, f"learned full_pose: {param_status}; {seed_status}"

    samples = matching_gripper_correction_samples(calib, args)
    if not samples:
        return {}, "learned full_pose: no correction samples; press B before manual correction, then C after alignment"

    error = np.array([float(err_u), float(err_v)], dtype=np.float64)
    overhead_radius = max(0.0, float(getattr(args, "gripper_learned_local_overhead_radius_px", 70.0)))
    joint_radius = max(0.0, float(getattr(args, "gripper_learned_local_joint_radius_deg", 8.0)))
    err_norm = float(np.linalg.norm(error))
    direction_gate_error = max(0.0, float(getattr(args, "gripper_learned_direction_gate_error_px", 40.0)))
    direction_filter_threshold = min(direction_gate_error, max(0.0, float(getattr(args, "gripper_center_tolerance_px", direction_gate_error))))
    min_direction_cos = float(getattr(args, "gripper_learned_direction_min_cos", 0.35))
    candidates: list[dict[str, Any]] = []
    for sample in samples:
        delta_pose = sample_delta_pose(sample, args)
        if not delta_pose:
            continue
        sample_error = np.array([float(sample["error_uv"][0]), float(sample["error_uv"][1])], dtype=np.float64)
        norm2 = float(np.dot(sample_error, sample_error))
        if norm2 <= 1e-6:
            continue
        sample_norm = math.sqrt(norm2)
        direction_cos = 1.0
        if err_norm > 1e-6 and sample_norm > 1e-6:
            direction_cos = float(np.dot(error, sample_error) / (err_norm * sample_norm))
        projection = float(np.dot(error, sample_error) / norm2)
        distance = float(np.linalg.norm(error - sample_error))
        overhead_distance = float("inf")
        if overhead_uv is not None:
            sample_overhead = sample.get("before_overhead_pixel") or sample.get("after_overhead_pixel")
            if sample_overhead is not None and len(sample_overhead) == 2:
                overhead_distance = math.hypot(
                    float(sample_overhead[0]) - float(overhead_uv[0]),
                    float(sample_overhead[1]) - float(overhead_uv[1]),
                )
        joint_distance = float("inf")
        if current_pose is not None:
            before_joints = sample.get("before_joints")
            if isinstance(before_joints, dict) and all(joint in before_joints and joint in current_pose for joint in args.pose_joints):
                joint_distance = math.sqrt(
                    sum((float(before_joints[joint]) - float(current_pose[joint])) ** 2 for joint in args.pose_joints)
                )
        candidates.append(
            {
                "id": sample.get("id"),
                "error_distance": distance,
                "overhead_distance": overhead_distance,
                "joint_distance": joint_distance,
                "overhead_local": overhead_uv is not None and overhead_distance <= overhead_radius,
                "joint_local": current_pose is not None and joint_distance <= joint_radius,
                "direction_cos": direction_cos,
                "direction_ok": err_norm < direction_filter_threshold or direction_cos >= min_direction_cos,
                "delta": {
                    joint: float(projection) * float(delta_pose[joint])
                    for joint in args.pose_joints
                },
            }
        )
    if not candidates:
        return {}, "learned full_pose: correction samples lack full before/after pose"

    local_mode = "global"
    usable = candidates
    both_local = [item for item in candidates if item["overhead_local"] and item["joint_local"]]
    overhead_local = [item for item in candidates if item["overhead_local"]]
    joint_local = [item for item in candidates if item["joint_local"]]
    require_both_local = bool(getattr(args, "gripper_learned_full_require_both_local", True))
    if both_local:
        usable = both_local
        local_mode = "overhead+joint"
    elif overhead_local:
        usable = overhead_local
        local_mode = "overhead"
    elif joint_local:
        usable = joint_local
        local_mode = "joint"
    min_local_samples = max(1, int(getattr(args, "gripper_learned_full_min_local_samples", 2)))
    allow_global = bool(getattr(args, "gripper_learned_full_allow_global_sample", False))
    if local_mode == "global" and not allow_global:
        return {}, (
            "learned full_pose: no local sample near this overhead/pose; "
            f"global pool={len(usable)} disabled, using visual probes instead"
        )
    if require_both_local and not both_local:
        return {}, (
            "learned full_pose: no sample is local in both overhead and full pose; "
            f"overhead_pool={len(overhead_local)} joint_pool={len(joint_local)}; using visual probes instead"
        )
    if err_norm >= direction_filter_threshold:
        directional = [item for item in usable if item["direction_ok"]]
        if directional:
            usable = directional
            local_mode += "+dir"
        else:
            best_cos = max((float(item["direction_cos"]) for item in usable), default=float("-inf"))
            return {}, (
                "learned full_pose: no direction-compatible local sample; "
                f"err={err_norm:.1f}px gate>={direction_filter_threshold:.1f}px min_cos={min_direction_cos:.2f} "
                f"best_cos={best_cos:.2f} local={local_mode} pool={len(usable)}; press B then C here"
            )
    if len(usable) < min_local_samples:
        best_cos = max((float(item["direction_cos"]) for item in usable), default=float("-inf"))
        return {}, (
            "learned full_pose: local evidence too thin for direct learned move; "
            f"local={local_mode} pool={len(usable)} min={min_local_samples} best_cos={best_cos:.2f}; "
            "using visual probes instead"
        )

    usable.sort(key=lambda item: (float(item["error_distance"]), float(item["overhead_distance"]), float(item["joint_distance"])))
    k = min(max(1, int(args.gripper_learned_k)), len(usable))
    neighbors = usable[:k]
    power = max(0.1, float(args.gripper_learned_power))
    weights = [1.0 / ((float(item["error_distance"]) + 1e-6) ** power) for item in neighbors]
    total_weight = sum(weights)
    if total_weight <= 0:
        return {}, "learned full_pose: invalid sample weights"
    delta = {
        joint: sum((weight / total_weight) * float(item["delta"][joint]) for weight, item in zip(weights, neighbors))
        for joint in args.pose_joints
    }
    step_limit = adaptive_gripper_full_pose_step_limit(float(np.linalg.norm(error)), args)
    max_abs = max((abs(value) for value in delta.values()), default=0.0)
    if max_abs > step_limit > 0:
        scale = step_limit / max_abs
        delta = {joint: float(value) * scale for joint, value in delta.items()}
    delta = clamp_learned_full_pose_delta(delta, args)
    ids = ",".join(str(item["id"]) for item in neighbors)
    cos_text = ",".join(f"{float(item['direction_cos']):.2f}" for item in neighbors)
    return delta, (
        f"learned full_pose: k={k}/{len(samples)} samples ids={ids} local={local_mode} "
        f"pool={len(usable)} cos={cos_text} step_limit={step_limit:.2f}deg"
    )


def predict_pose_target(u: float, v: float, calib: CalibrationState, args) -> dict[str, float]:
    if args.pose_predictor == "local":
        return predict_pose_local(u, v, calib, args)
    return predict_pose_from_pixel(u, v, calib)


def fit_orientation_model(samples: list[dict[str, float]], args) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    usable = []
    required = list(args.map_joints) + list(args.orientation_joints)
    for sample in samples:
        if all(joint in sample for joint in required):
            usable.append(sample)
    if len(usable) < 3:
        return {}, {}

    x = np.array(
        [[float(sample[args.map_joints[0]]), float(sample[args.map_joints[1]]), 1.0] for sample in usable],
        dtype=np.float64,
    )
    model: dict[str, list[float]] = {}
    limits: dict[str, list[float]] = {}
    for joint in args.orientation_joints:
        y = np.array([float(sample[joint]) for sample in usable], dtype=np.float64)
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        model[joint] = [float(v) for v in coeffs]
        limits[joint] = [float(y.min() - args.orientation_margin), float(y.max() + args.orientation_margin)]
    return model, limits


def load_orientation_samples_from_csv(path: Path, args) -> list[dict[str, float]]:
    if not path.exists():
        return []
    samples: list[dict[str, float]] = []
    required = list(args.map_joints) + list(args.orientation_joints)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample: dict[str, float] = {}
            ok = True
            for joint in required:
                raw = row.get(f"performed.{joint}")
                if raw in (None, ""):
                    ok = False
                    break
                try:
                    sample[joint] = float(raw)
                except ValueError:
                    ok = False
                    break
            if ok:
                samples.append(sample)
    return samples


def ensure_orientation_model(calib: CalibrationState, args) -> str:
    if calib.orientation_model:
        return "orientation model: calibration"
    if args.disable_csv_orientation:
        return "orientation model: missing"
    samples = load_orientation_samples_from_csv(Path(args.orientation_source_csv), args)
    model, limits = fit_orientation_model(samples, args)
    if model:
        calib.orientation_model = model
        calib.orientation_limits = limits
        return f"orientation model: csv {len(samples)} samples"
    return "orientation model: missing"


def pixel_to_joint_target(homography: list[list[float]], u: float, v: float) -> tuple[float, float]:
    h = np.array(homography, dtype=np.float64)
    src = np.array([[[u, v]]], dtype=np.float64)
    dst = cv2.perspectiveTransform(src, h)[0, 0]
    return float(dst[0]), float(dst[1])


def within_limits(target: dict[str, float], joint_limits: dict[str, list[float]]) -> tuple[bool, str]:
    for joint, value in target.items():
        if joint not in joint_limits:
            return False, f"missing joint limit for {joint}"
        low, high = joint_limits[joint]
        if value < low or value > high:
            return False, f"{joint} target {value:.2f} outside [{low:.2f}, {high:.2f}]"
    return True, "ok"


def clamp_to_limits(target: dict[str, float], joint_limits: dict[str, list[float]]) -> tuple[dict[str, float], bool, str]:
    clamped: dict[str, float] = {}
    notes: list[str] = []
    changed = False
    for joint, value in target.items():
        if joint not in joint_limits:
            return {}, False, f"missing joint limit for {joint}"
        low, high = joint_limits[joint]
        value_f = float(value)
        clipped = min(max(value_f, float(low)), float(high))
        clamped[joint] = clipped
        if abs(clipped - value_f) > 1e-6:
            changed = True
            notes.append(f"{joint} {value_f:.2f}->{clipped:.2f} within [{float(low):.2f},{float(high):.2f}]")
    if changed:
        return clamped, True, "clamped " + "; ".join(notes)
    return clamped, False, "ok"


def interpolate_pose(start: dict[str, float], target: dict[str, float], steps: int) -> list[dict[str, float]]:
    steps = max(1, int(steps))
    poses = []
    keys = [k for k in MOTOR_KEYS if k in start and k in target]
    for i in range(1, steps + 1):
        alpha = i / steps
        poses.append({key: start[key] + (target[key] - start[key]) * alpha for key in keys})
    return poses


def planned_motion_steps(
    current: dict[str, float],
    target: dict[str, float],
    args,
    speed_deg_s: float | None = None,
    min_steps: int | None = None,
) -> tuple[int, float, float, float]:
    keys = [key for key in MOTOR_KEYS if key in current and key in target]
    if not keys:
        return 0, 0.0, 0.0, 0.0
    max_delta = max(abs(float(target[key]) - float(current[key])) for key in keys)
    if max_delta < 0.25:
        return 0, max_delta, 0.0, 0.0
    by_step = math.ceil(max_delta / max(0.1, float(args.max_joint_step)))
    effective_min_steps = int(args.min_move_steps) if min_steps is None else int(min_steps)
    steps = max(max(1, effective_min_steps), by_step)
    steps = min(int(args.move_steps), steps)
    effective_speed = float(args.joint_speed_deg_s) if speed_deg_s is None else float(speed_deg_s)
    desired_duration = max_delta / max(1.0, effective_speed)
    sleep_s = desired_duration / max(1, steps)
    sleep_s = max(float(args.move_sleep_s), min(float(args.max_move_sleep_s), sleep_s))
    return steps, max_delta, sleep_s, effective_speed


def send_interpolated(
    robot,
    target: dict[str, float],
    args,
    robot_lock: threading.RLock | None = None,
    speed_deg_s: float | None = None,
    min_steps: int | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, float]:
    current = locked_observation(robot, robot_lock)
    full_target = current.copy()
    full_target.update(target)
    steps, max_delta, sleep_s, effective_speed = planned_motion_steps(current, full_target, args, speed_deg_s, min_steps)
    if steps <= 0:
        return current
    last_sent = current
    print(
        f"motion plan: max_delta={max_delta:.2f}, steps={steps}, sleep={sleep_s:.3f}s, "
        f"speed~{effective_speed:.1f}deg/s",
        flush=True,
    )
    for pose in interpolate_pose(current, full_target, steps):
        if stop_event is not None and stop_event.is_set():
            break
        if robot_lock is None:
            last_sent = robot.send_action(pose)
        else:
            with robot_lock:
                last_sent = robot.send_action(pose)
        time.sleep(max(0.0, sleep_s))
    return scalar_motor_state(last_sent)


def predict_orientation_joints(mapped_target: dict[str, float], orientation_model: dict[str, list[float]], args) -> dict[str, float]:
    if not orientation_model:
        return {}
    j0 = float(mapped_target[args.map_joints[0]])
    j1 = float(mapped_target[args.map_joints[1]])
    predicted: dict[str, float] = {}
    for joint, coeffs in orientation_model.items():
        if len(coeffs) != 3 or joint in args.map_joints:
            continue
        predicted[joint] = float(coeffs[0] * j0 + coeffs[1] * j1 + coeffs[2])
    return predicted


def within_orientation_limits(target: dict[str, float], orientation_limits: dict[str, list[float]]) -> tuple[bool, str]:
    for joint, value in target.items():
        if joint not in orientation_limits:
            return False, f"missing orientation limit for {joint}"
        low, high = orientation_limits[joint]
        if value < low or value > high:
            return False, f"{joint} target {value:.2f} outside [{low:.2f}, {high:.2f}]"
    return True, "ok"


def build_safe_plane_target(
    current: dict[str, float],
    safe_pose: dict[str, float],
    orientation_model: dict[str, list[float]],
    map_joints: list[str],
    mapped_target: dict[str, float],
    args,
) -> dict[str, float]:
    target = current.copy()
    for key in MOTOR_KEYS:
        if key in safe_pose:
            target[key] = float(safe_pose[key])
    target.update(predict_orientation_joints(mapped_target, orientation_model, args))
    for joint in map_joints:
        target[joint] = mapped_target[joint]
    return target


def build_full_pose_target(current: dict[str, float], safe_pose: dict[str, float], pose_target: dict[str, float]) -> dict[str, float]:
    target = current.copy()
    for key in MOTOR_KEYS:
        if key in safe_pose:
            target[key] = float(safe_pose[key])
    for joint, value in pose_target.items():
            target[joint] = float(value)
    return target


def build_xy_hover_pose_target(
    current: dict[str, float],
    safe_pose: dict[str, float],
    xy_target: dict[str, float],
    hover_target: dict[str, float],
    orientation_target: dict[str, float],
) -> dict[str, float]:
    target = current.copy()
    for key in MOTOR_KEYS:
        if key in safe_pose:
            target[key] = float(safe_pose[key])
    for joint, value in orientation_target.items():
        target[joint] = float(value)
    for joint, value in hover_target.items():
        target[joint] = float(value)
    for joint, value in xy_target.items():
        target[joint] = float(value)
    return target


def build_safe_pose_target(current: dict[str, float], safe_pose: dict[str, float]) -> dict[str, float]:
    target = current.copy()
    for key in MOTOR_KEYS:
        if key in safe_pose:
            target[key] = float(safe_pose[key])
    return target


def build_safe_lift_in_place_target(current: dict[str, float], safe_pose: dict[str, float], args) -> dict[str, float]:
    target = current.copy()
    for joint in args.safe_lift_joints:
        if joint in safe_pose and joint in current:
            target[joint] = float(safe_pose[joint])
    return target


def wait_until_pose(
    robot,
    target: dict[str, float],
    joints: list[str],
    args,
    label: str,
    robot_lock: threading.RLock | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[bool, str]:
    started_at = time.perf_counter()
    base_deadline = started_at + max(0.0, float(args.safe_wait_timeout_s))
    max_deadline = base_deadline + max(0.0, float(args.safe_progress_extra_timeout_s))
    deadline = base_deadline
    last_error = 0.0
    last_details: list[tuple[float, str, float, float]] = []
    best_error = float("inf")
    last_hold_sent_at = started_at
    active_joints = [joint for joint in joints if joint in target]
    if not active_joints:
        return True, f"{label}: no joints to wait for"
    threshold = pose_wait_threshold(label, args)
    while time.perf_counter() <= deadline:
        if stop_event is not None and stop_event.is_set():
            return False, f"{label}: cancelled"
        now = time.perf_counter()
        observed = locked_observation(robot, robot_lock)
        last_details = []
        for joint in active_joints:
            target_value = float(target[joint])
            observed_value = float(observed.get(joint, target[joint]))
            last_details.append((abs(target_value - observed_value), joint, target_value, observed_value))
        last_details.sort(reverse=True, key=lambda item: item[0])
        last_error = last_details[0][0] if last_details else 0.0
        if last_error <= threshold:
            return True, (
                f"{label}: reached max_error={last_error:.2f}; threshold={threshold:.2f}; "
                f"best_error={min(best_error, last_error):.2f}; "
                f"{format_pose_error_details(last_details)}"
            )
        if last_error < best_error - max(0.0, float(args.safe_progress_epsilon)):
            best_error = last_error
            deadline = max(deadline, min(now + max(0.0, float(args.safe_progress_grace_s)), max_deadline))
        hold_interval = max(0.0, float(args.safe_hold_command_interval_s))
        if hold_interval > 0 and now - last_hold_sent_at >= hold_interval:
            send_action_once(robot, target, robot_lock)
            last_hold_sent_at = now
        time.sleep(0.05)
    return False, (
        f"{label}: timeout max_error={last_error:.2f}; threshold={threshold:.2f}; best_error={best_error:.2f}; "
        f"{format_pose_error_details(last_details)}"
    )


def pose_wait_threshold(label: str, args) -> float:
    if label == "safe lift":
        return float(args.safe_lift_position_threshold)
    if label == "full safe_pose":
        return float(args.full_safe_position_threshold)
    return float(args.safe_position_threshold)


def send_action_once(robot, target: dict[str, float], robot_lock: threading.RLock | None = None) -> dict[str, float]:
    if robot_lock is None:
        return scalar_motor_state(robot.send_action(target))
    with robot_lock:
        return scalar_motor_state(robot.send_action(target))


def format_pose_error_details(details: list[tuple[float, str, float, float]], limit: int = 3) -> str:
    if not details:
        return "no joint errors"
    chunks = []
    for error, joint, target_value, observed_value in details[: max(1, limit)]:
        chunks.append(f"{joint} err={error:.2f} target={target_value:.2f} obs={observed_value:.2f}")
    return "worst " + "; ".join(chunks)


def filtered_pose_error_details(
    details: list[tuple[float, str, float, float]],
    joints: list[str],
) -> tuple[float, list[tuple[float, str, float, float]]]:
    joint_set = set(joints)
    filtered = [item for item in details if item[1] in joint_set]
    filtered.sort(reverse=True, key=lambda item: item[0])
    return (filtered[0][0] if filtered else 0.0), filtered


def current_pose_error_details(
    robot,
    target: dict[str, float],
    joints: list[str],
    robot_lock: threading.RLock | None = None,
) -> tuple[float, list[tuple[float, str, float, float]]]:
    active_joints = [joint for joint in joints if joint in target]
    if not active_joints:
        return 0.0, []
    observed = locked_observation(robot, robot_lock)
    details = []
    for joint in active_joints:
        target_value = float(target[joint])
        observed_value = float(observed.get(joint, target[joint]))
        details.append((abs(target_value - observed_value), joint, target_value, observed_value))
    details.sort(reverse=True, key=lambda item: item[0])
    return (details[0][0] if details else 0.0), details


def send_and_wait_pose(
    robot,
    target: dict[str, float],
    wait_joints: list[str],
    args,
    label: str,
    robot_lock: threading.RLock | None = None,
    retry_should_stop: Callable[[float, list[tuple[float, str, float, float]]], tuple[bool, str]] | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[bool, str]:
    attempts = max(1, int(args.safe_retry_count) + 1)
    last_status = ""
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            print(f"{label}: retry {attempt - 1}/{attempts - 1} sending the same target again", flush=True)
        send_interpolated(robot, target, args, robot_lock, stop_event=stop_event)
        ok_wait, last_status = wait_until_pose(
            robot,
            target,
            wait_joints,
            args,
            label,
            robot_lock,
            stop_event,
        )
        print(last_status, flush=True)
        if ok_wait:
            return True, last_status
        if stop_event is not None and stop_event.is_set():
            return False, last_status
        if attempt < attempts and retry_should_stop is not None:
            current_error, current_details = current_pose_error_details(robot, target, wait_joints, robot_lock)
            stop_retry, reason = retry_should_stop(current_error, current_details)
            if stop_retry:
                print(f"{label}: skip retry; {reason}", flush=True)
                return False, last_status
    return False, last_status


def detect_target_from_camera(capture, model, names, args, flush_frames: int = 0) -> TargetDetection | None:
    frame = None
    ok = False
    for _ in range(max(0, flush_frames)):
        capture.read()
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return select_detection(run_predict(model, frame, args), names, args.target_class, args.conf, args)


def detect_target_from_camera_locked(
    capture,
    model,
    names,
    args,
    camera_lock: threading.RLock | None = None,
    flush_frames: int = 0,
) -> TargetDetection | None:
    if camera_lock is None:
        return detect_target_from_camera(capture, model, names, args, flush_frames)
    with camera_lock:
        frame = None
        ok = False
        for _ in range(max(0, flush_frames)):
            capture.read()
        ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return select_detection(run_predict(model, frame, args), names, args.target_class, args.conf, args)


def read_camera_frame_locked(capture, camera_lock: threading.RLock | None = None, flush_frames: int = 0):
    if camera_lock is None:
        for _ in range(max(0, flush_frames)):
            capture.read()
        return capture.read()
    with camera_lock:
        for _ in range(max(0, flush_frames)):
            capture.read()
        return capture.read()


def read_gripper_frame_locked(capture, args, camera_lock: threading.RLock | None = None, flush_frames: int = 0):
    ok, frame = read_camera_frame_locked(capture, camera_lock, flush_frames)
    if not ok or frame is None:
        return ok, frame
    return ok, transform_gripper_frame(frame, args)


def detect_from_gripper_frame(
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    args,
    gripper_camera_lock: threading.RLock | None = None,
    flush_frames: int = 0,
) -> tuple[TargetDetection | None, Any | None, str]:
    ok, frame = read_gripper_frame_locked(gripper_capture, args, gripper_camera_lock, flush_frames)
    if not ok or frame is None:
        return None, None, "camera_read_failed"
    result = run_predict(gripper_model, frame, args)
    det = select_detection(result, gripper_names, args.target_class, args.conf, args)
    if det is None:
        return None, frame, f"target_not_detected conf={args.conf}"
    return det, frame, "ok"


def clamp_delta(value: float, max_abs: float) -> float:
    limit = abs(float(max_abs))
    return min(max(float(value), -limit), limit)


def create_refine_pose_anchor(
    robot,
    calib: CalibrationState,
    args,
    robot_lock: threading.RLock | None = None,
) -> RefinePoseAnchor:
    pose = locked_observation(robot, robot_lock)
    xy = {joint: float(pose[joint]) for joint in args.map_joints if joint in pose}
    hover_model_pose = predict_hover_height_joints(xy, calib, args)
    orientation_model_pose = predict_xy_hover_orientation_joints(xy, hover_model_pose, calib, args)
    return RefinePoseAnchor(
        xy=xy,
        pose={joint: float(value) for joint, value in pose.items()},
        hover_model_pose=hover_model_pose,
        orientation_model_pose=orientation_model_pose,
    )


def prepare_xy_hover_refine_target(
    robot,
    calib: CalibrationState,
    args,
    current_xy: dict[str, float],
    robot_lock: threading.RLock | None = None,
    anchor: RefinePoseAnchor | None = None,
) -> tuple[bool, dict[str, float], str]:
    if calib.pose_limits:
        xy_limits = {joint: calib.pose_limits[joint] for joint in current_xy if joint in calib.pose_limits}
        ok_xy, xy_reason = within_limits(current_xy, xy_limits)
        if not ok_xy:
            return False, {}, f"xy_limit {xy_reason}"

    hover_target = predict_hover_height_joints(current_xy, calib, args)
    if not hover_target:
        current = locked_observation(robot, robot_lock)
        target_pose = current.copy()
        for joint, value in current_xy.items():
            target_pose[joint] = float(value)
        return True, target_pose, "simple_xy_no_hover_model"
    hover_target, hover_clamped, hover_reason = clamp_to_limits(hover_target, calib.hover_height_limits)
    if not hover_target:
        return False, {}, f"hover_limit {hover_reason}"

    orientation_target = predict_xy_hover_orientation_joints(current_xy, hover_target, calib, args)
    ok_orientation, orientation_reason = within_orientation_limits(orientation_target, calib.orientation_limits)
    if not ok_orientation:
        return False, {}, f"orientation_limit {orientation_reason}"

    current = locked_observation(robot, robot_lock)
    if anchor is None:
        target_pose = build_xy_hover_pose_target(current, calib.safe_pose or {}, current_xy, hover_target, orientation_target)
    else:
        target_pose = current.copy()
        for joint, value in current_xy.items():
            target_pose[joint] = float(value)
        for joint, value in hover_target.items():
            if joint in anchor.pose and joint in anchor.hover_model_pose:
                target_pose[joint] = float(anchor.pose[joint]) + float(value) - float(anchor.hover_model_pose[joint])
            else:
                target_pose[joint] = float(value)
        for joint, value in orientation_target.items():
            if joint in anchor.pose and joint in anchor.orientation_model_pose:
                target_pose[joint] = float(anchor.pose[joint]) + float(value) - float(anchor.orientation_model_pose[joint])
            else:
                target_pose[joint] = float(value)
    note = hover_reason
    if hover_clamped:
        note = "hover_clamped " + hover_reason
    if anchor is not None:
        note = f"{note}; anchored_pose"
    return True, target_pose, note


def wait_until_gripper_refine_pose(
    robot,
    target_pose: dict[str, float],
    args,
    robot_lock: threading.RLock | None = None,
    timeout_s: float | None = None,
    stable_offset_deg: float | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[bool, str]:
    wait_joints = args.gripper_refine_wait_joints or list(args.pose_joints)
    active_joints = [joint for joint in wait_joints if joint in target_pose]
    if not active_joints:
        return True, "refine_wait=no_joints"
    threshold = max(0.0, float(args.gripper_refine_position_threshold))
    effective_timeout = float(args.gripper_refine_wait_timeout_s) if timeout_s is None else float(timeout_s)
    deadline = time.perf_counter() + max(0.0, effective_timeout)
    hold_interval = max(0.0, float(args.gripper_refine_hold_command_interval_s))
    last_hold_sent_at = 0.0
    last_error = float("inf")
    last_details: list[tuple[float, str, float, float]] = []
    previous: dict[str, float] | None = None
    stable_count = 0
    max_motion = float("inf")
    stable_epsilon = max(0.0, float(args.gripper_refine_stable_epsilon_deg))
    required_stable = max(1, int(args.gripper_refine_stable_samples))
    continue_threshold = max(
        threshold,
        float(args.gripper_refine_continue_threshold) if stable_offset_deg is None else float(stable_offset_deg),
    )
    while time.perf_counter() <= deadline:
        if stop_event is not None and stop_event.is_set():
            return False, "refine_wait cancelled"
        observed = locked_observation(robot, robot_lock)
        last_details = []
        for joint in active_joints:
            target_value = float(target_pose[joint])
            observed_value = float(observed.get(joint, target_value))
            last_details.append((abs(target_value - observed_value), joint, target_value, observed_value))
        last_details.sort(reverse=True, key=lambda item: item[0])
        last_error = last_details[0][0] if last_details else 0.0
        if previous is not None:
            max_motion = max(abs(float(observed[joint]) - float(previous[joint])) for joint in active_joints)
            stable_count = stable_count + 1 if max_motion <= stable_epsilon else 0
        previous = {joint: float(observed[joint]) for joint in active_joints}
        if stable_count >= required_stable and last_error <= continue_threshold:
            mode = "reached" if last_error <= threshold else "stable_offset"
            return True, (
                f"refine_wait {mode} max_error={last_error:.2f} threshold={threshold:.2f} "
                f"continue<={continue_threshold:.2f} motion={max_motion:.3f}; "
                f"{format_pose_error_details(last_details)}"
            )
        now = time.perf_counter()
        if hold_interval > 0 and now - last_hold_sent_at >= hold_interval:
            send_action_once(robot, target_pose, robot_lock)
            last_hold_sent_at = now
        time.sleep(0.02)
    return False, (
        f"refine_wait timeout max_error={last_error:.2f} threshold={threshold:.2f} "
        f"continue<={continue_threshold:.2f} stable={stable_count}/{required_stable} motion={max_motion:.3f}; "
        f"{format_pose_error_details(last_details)}"
    )


def wait_until_observation_stable(
    robot,
    joints: list[str],
    timeout_s: float,
    epsilon_deg: float,
    stable_samples: int,
    robot_lock: threading.RLock | None = None,
) -> tuple[bool, dict[str, float], str]:
    """Wait for encoder motion to stop without changing the last commanded pose."""
    deadline = time.perf_counter() + max(0.0, float(timeout_s))
    required = max(1, int(stable_samples))
    epsilon = max(0.0, float(epsilon_deg))
    previous: dict[str, float] | None = None
    stable_count = 0
    last_motion = float("inf")
    observed: dict[str, float] = {}
    while time.perf_counter() <= deadline:
        observed = locked_observation(robot, robot_lock)
        active = [joint for joint in joints if joint in observed]
        if not active:
            return False, observed, "pose_stable failed: no requested joints in observation"
        if previous is not None:
            last_motion = max(abs(float(observed[joint]) - float(previous[joint])) for joint in active)
            stable_count = stable_count + 1 if last_motion <= epsilon else 0
            if stable_count >= required:
                return True, observed, (
                    f"pose_stable motion={last_motion:.3f}deg samples={stable_count}/{required}"
                )
        previous = {joint: float(observed[joint]) for joint in active}
        time.sleep(0.02)
    return False, observed, (
        f"pose_stable timeout motion={last_motion:.3f}deg samples={stable_count}/{required}"
    )


def send_xy_hover_refine_target(
    robot,
    calib: CalibrationState,
    args,
    current_xy: dict[str, float],
    robot_lock: threading.RLock | None = None,
    anchor: RefinePoseAnchor | None = None,
    stop_event: threading.Event | None = None,
    outer_correction_enabled: bool | None = None,
) -> tuple[bool, str]:
    ok_target, target_pose, reason = prepare_xy_hover_refine_target(robot, calib, args, current_xy, robot_lock, anchor)
    if not ok_target:
        return False, reason
    move_result = send_pose_with_encoder_outer_correction(
        robot,
        target_pose,
        args,
        robot_lock,
        stop_event=stop_event,
        label="GRIPPER_XY",
        speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
        min_steps=int(args.gripper_refine_min_move_steps),
        wait_timeout_s=float(args.gripper_refine_wait_timeout_s),
        settle_threshold=float(args.gripper_refine_continue_threshold),
        outer_correction_enabled=outer_correction_enabled,
    )
    if move_result.get("cancelled"):
        return False, reason + f"; cancelled during {move_result.get('stage')}"
    correction_notes = list(move_result.get("correction_notes") or [])
    correction_text = ""
    if correction_notes:
        correction_text = "; outer_correction " + " | ".join(correction_notes)
    motion_status = (
        f"GRIPPER_XY max_joint_error={float(move_result.get('max_error', float('inf'))):.2f}; "
        f"raw_max_joint_error={float(move_result.get('raw_error', float('inf'))):.2f}"
        f"{correction_text}"
    )
    if bool(args.gripper_refine_wait):
        reason = f"{reason}; {motion_status}; {move_result.get('wait_status', 'encoder wait failed')}"
        if not bool(move_result.get("ok_wait")):
            return False, reason
    else:
        reason = f"{reason}; {motion_status}"
    return True, reason


def observed_map_xy(
    robot,
    args,
    robot_lock: threading.RLock | None = None,
) -> dict[str, float]:
    obs = locked_observation(robot, robot_lock)
    return {joint: float(obs[joint]) for joint in args.map_joints if joint in obs}


def scale_joint_delta(delta: np.ndarray, max_abs: float) -> np.ndarray:
    limit = abs(float(max_abs))
    if limit <= 0:
        return np.zeros_like(delta)
    max_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
    if max_delta > limit:
        return delta * (limit / max_delta)
    return delta


def gripper_actual_motion_quality(
    command_step: np.ndarray,
    actual_step: np.ndarray,
    args,
) -> tuple[bool, str]:
    command = np.asarray(command_step, dtype=np.float64).reshape(2)
    actual = np.asarray(actual_step, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(command)) or not np.all(np.isfinite(actual)):
        return False, "actual_motion rejected: non-finite command/actual step"
    command_norm = float(np.linalg.norm(command))
    actual_norm = float(np.linalg.norm(actual))
    if command_norm <= 1e-6:
        return True, "actual_motion skipped: zero command"

    min_actual = max(0.0, float(getattr(args, "gripper_min_actual_motion_deg", 0.12)))
    if actual_norm < min_actual:
        return False, (
            f"actual_motion rejected: actual move {actual_norm:.3f}deg<{min_actual:.3f}deg "
            f"for command_norm={command_norm:.3f}deg"
        )

    command_unit = command / command_norm
    projected = float(actual @ command_unit)
    min_fraction = max(0.0, float(getattr(args, "gripper_min_actual_motion_fraction", 0.18)))
    required_projection = min_fraction * command_norm
    if projected < required_projection:
        return False, (
            f"actual_motion rejected: projected move {projected:.3f}deg<"
            f"{required_projection:.3f}deg ({min_fraction:.2f} of command)"
        )

    lateral = float(np.linalg.norm(actual - projected * command_unit))
    return True, (
        f"actual_motion ok actual_norm={actual_norm:.3f}deg projected={projected:.3f}deg "
        f"lateral={lateral:.3f}deg"
    )


def limit_gripper_step_by_lift_budget(
    current_pose: dict[str, float],
    command_step: np.ndarray,
    args,
    baseline_lift: float | None,
) -> tuple[np.ndarray, str]:
    step = np.asarray(command_step, dtype=np.float64).reshape(2).copy()
    if getattr(args, "gripper_height_guard_action", "recover") == "recover":
        return step, "lift_budget skipped: height recovery enabled"
    if baseline_lift is None or len(args.map_joints) < 2:
        return step, "lift_budget skipped"
    lift_joint = args.map_joints[1]
    if lift_joint not in current_pose:
        return step, "lift_budget skipped: lift joint missing"
    max_drift = max(0.0, float(getattr(args, "gripper_max_cumulative_lift_drift_deg", 0.0)))
    if max_drift <= 0.0:
        return step, "lift_budget disabled"
    margin = max(0.0, float(getattr(args, "gripper_cumulative_lift_margin_deg", 0.45)))
    usable_drift = max(0.0, max_drift - margin)
    current_lift = float(current_pose[lift_joint])
    low = float(baseline_lift) - usable_drift
    high = float(baseline_lift) + usable_drift
    requested_lift = float(step[1])
    target_lift = current_lift + requested_lift
    clipped_lift = min(max(target_lift, low), high)
    step[1] = clipped_lift - current_lift
    if abs(step[1] - requested_lift) <= 1e-6:
        return step, (
            f"lift_budget ok {lift_joint} target={target_lift:.2f} "
            f"range=[{low:.2f},{high:.2f}]"
        )
    return step, (
        f"lift_budget clipped {lift_joint} delta {requested_lift:.3f}->{step[1]:.3f} "
        f"target {target_lift:.2f}->{clipped_lift:.2f} range=[{low:.2f},{high:.2f}]"
    )


def adaptive_gripper_step_limit(dist_px: float, args) -> float:
    max_step = abs(float(args.gripper_max_step_deg))
    min_step = min(max_step, abs(float(args.gripper_min_step_deg)))
    if max_step <= 0:
        return 0.0
    tolerance = max(1.0, float(args.gripper_center_tolerance_px))
    full_error = max(tolerance + 1.0, float(args.gripper_full_step_error_px))
    alpha = (float(dist_px) - tolerance) / (full_error - tolerance)
    alpha = min(1.0, max(0.0, alpha))
    return min_step + (max_step - min_step) * alpha


def gripper_error_within_tolerance(dist_px: float, args) -> bool:
    tolerance = float(args.gripper_center_tolerance_px)
    # Status logs round pixel error to 0.1px, so accept the tiny boundary band
    # that still prints as exactly the configured tolerance.
    return float(dist_px) <= tolerance + 0.05


def gripper_required_improve_px(dist_px: float, args) -> float:
    base = max(0.0, float(args.gripper_search_improve_px))
    tolerance = max(1.0, float(args.gripper_center_tolerance_px))
    return max(base, 0.08 * max(0.0, float(dist_px) - tolerance))


def open_gripper_trace(args) -> tuple[Any | None, csv.DictWriter | None, Path | None]:
    if not bool(args.gripper_trace):
        return None, None, None
    trace_dir = Path(args.gripper_trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"gripper_stage3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    handle = path.open("w", encoding="utf-8", newline="", buffering=1)
    fieldnames = [
        "timestamp",
        "event",
        "iter",
        "servo_mode",
        "center_u",
        "center_v",
        "ref_u",
        "ref_v",
        "err_u",
        "err_v",
        "dist_px",
        "delta_pan",
        "delta_lift",
        "actual_delta_pan",
        "actual_delta_lift",
        "target_pan",
        "target_lift",
        "actual_pan",
        "actual_lift",
        "actual_elbow",
        "actual_wrist_flex",
        "actual_wrist_roll",
        "actual_gripper",
        "baseline_pan",
        "baseline_lift",
        "baseline_elbow",
        "baseline_wrist_flex",
        "baseline_wrist_roll",
        "baseline_gripper",
        "target_elbow",
        "target_wrist_flex",
        "target_wrist_roll",
        "target_gripper",
        "actual_delta_elbow",
        "actual_delta_wrist_flex",
        "actual_delta_wrist_roll",
        "actual_delta_gripper",
        "restore_ok",
        "target_area_px",
        "area_ratio",
        "mask_sqrt_area_px",
        "mask_bbox_w_px",
        "mask_bbox_h_px",
        "mask_bbox_aspect",
        "mask_extent",
        "mask_solidity",
        "mask_circularity",
        "mask_rect_aspect",
        "mask_angle_deg",
        "mask_major_axis_px",
        "mask_minor_axis_px",
        "mask_axis_ratio",
        "j00_du_dpan",
        "j01_du_dlift",
        "j10_dv_dpan",
        "j11_dv_dlift",
        "condition",
        "jacobian_rmse_px",
        "observation_samples",
        "observation_jitter_px",
        "ref_source",
        "status",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    print(f"gripper_final trace: {path}", flush=True)
    return handle, writer, path


def write_gripper_trace(writer: csv.DictWriter | None, **row) -> None:
    if writer is None:
        return
    base = {key: "" for key in writer.fieldnames or []}
    base.update(row)
    base["timestamp"] = now_iso()
    writer.writerow(base)


def estimate_gripper_image_jacobian(
    robot,
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    current_xy: dict[str, float],
    base_center: tuple[float, float],
    robot_lock: threading.RLock | None = None,
    gripper_camera_lock: threading.RLock | None = None,
    trace_writer: csv.DictWriter | None = None,
    iter_index: int = 0,
) -> tuple[np.ndarray | None, float | None, str]:
    probe = abs(float(args.gripper_jacobian_probe_deg))
    if probe <= 0:
        return None, None, "jacobian probe disabled"

    columns: list[list[float]] = []
    base_xy = {joint: float(current_xy[joint]) for joint in args.map_joints}
    for joint in args.map_joints:
        chosen_step = None
        chosen_xy = None
        for sign in (1.0, -1.0):
            step = sign * probe
            candidate_xy = dict(base_xy)
            candidate_xy[joint] = float(candidate_xy[joint]) + step
            ok_target, _target_pose, _reason = prepare_xy_hover_refine_target(robot, calib, args, candidate_xy, robot_lock)
            if ok_target:
                chosen_step = step
                chosen_xy = candidate_xy
                break
        if chosen_step is None or chosen_xy is None:
            return None, None, f"jacobian probe for {joint} outside safe limits"

        ok_move, move_reason = send_xy_hover_refine_target(robot, calib, args, chosen_xy, robot_lock)
        if not ok_move:
            return None, None, f"jacobian probe move failed for {joint}: {move_reason}"
        time.sleep(max(0.0, float(args.gripper_settle_s)))

        det, _frame, det_status = detect_from_gripper_frame(
            gripper_capture,
            gripper_model,
            gripper_names,
            args,
            gripper_camera_lock,
            flush_frames=0 if args.dry_run else int(args.gripper_camera_flush_frames),
        )
        if det is None:
            send_xy_hover_refine_target(robot, calib, args, base_xy, robot_lock)
            return None, None, f"jacobian probe lost target for {joint}: {det_status}"

        du = (float(det.center_uv[0]) - float(base_center[0])) / float(chosen_step)
        dv = (float(det.center_uv[1]) - float(base_center[1])) / float(chosen_step)
        columns.append([du, dv])
        write_gripper_trace(
            trace_writer,
            event="jacobian_probe",
            iter=iter_index,
            servo_mode="jacobian",
            center_u=f"{det.center_uv[0]:.3f}",
            center_v=f"{det.center_uv[1]:.3f}",
            delta_pan=f"{chosen_step:.4f}" if joint == args.map_joints[0] else "0.0000",
            delta_lift=f"{chosen_step:.4f}" if joint == args.map_joints[1] else "0.0000",
            target_pan=f"{chosen_xy[args.map_joints[0]]:.4f}",
            target_lift=f"{chosen_xy[args.map_joints[1]]:.4f}",
            status=f"{joint} {move_reason}",
        )

        ok_return, return_reason = send_xy_hover_refine_target(robot, calib, args, base_xy, robot_lock)
        if not ok_return:
            return None, None, f"jacobian return failed after {joint}: {return_reason}"
        time.sleep(max(0.0, float(args.gripper_settle_s)))

    if len(columns) != 2:
        return None, None, "jacobian incomplete"
    jacobian = np.array([[columns[0][0], columns[1][0]], [columns[0][1], columns[1][1]]], dtype=np.float64)
    column_norms = [float(np.linalg.norm(jacobian[:, index])) for index in range(2)]
    min_norm = min(column_norms)
    if min_norm < float(args.gripper_jacobian_min_px_per_deg):
        return None, None, f"jacobian too weak min_col_norm={min_norm:.2f}px/deg"
    try:
        condition = float(np.linalg.cond(jacobian))
    except Exception:  # noqa: BLE001
        return None, None, "jacobian condition failed"
    if not math.isfinite(condition) or condition > float(args.gripper_jacobian_max_condition):
        return None, condition, f"jacobian ill-conditioned cond={condition:.1f}"
    status = (
        f"jacobian ok J=[[{jacobian[0, 0]:.2f},{jacobian[0, 1]:.2f}],"
        f"[{jacobian[1, 0]:.2f},{jacobian[1, 1]:.2f}]] cond={condition:.1f}"
    )
    print("gripper_final " + status, flush=True)
    return jacobian, condition, status


def gripper_error_from_detection(det: TargetDetection, ref_uv: tuple[float, float]) -> tuple[float, float, float]:
    err_u = float(det.center_uv[0]) - float(ref_uv[0])
    err_v = float(det.center_uv[1]) - float(ref_uv[1])
    return err_u, err_v, math.hypot(err_u, err_v)


def detection_area_px(det: TargetDetection | None) -> float:
    if det is None:
        return 0.0
    if det.mask is not None:
        try:
            return float(np.count_nonzero(det.mask))
        except Exception:  # noqa: BLE001
            pass
    try:
        x1, y1, x2, y2 = [float(value) for value in det.xyxy]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    except (TypeError, ValueError):
        return 0.0


def median_mask_shape_features(detections: list[TargetDetection]) -> dict[str, float]:
    if not detections:
        return {key: 0.0 for key in MASK_SHAPE_FEATURE_KEYS}
    rows = [mask_shape_features(det.mask, det.xyxy) for det in detections]
    result: dict[str, float] = {}
    for key in MASK_SHAPE_FEATURE_KEYS:
        values = [float(row.get(key, 0.0)) for row in rows if math.isfinite(float(row.get(key, 0.0)))]
        result[key] = float(np.median(values)) if values else 0.0
    return result


def load_gripper_mask_shape_profile(args) -> tuple[dict[str, float] | None, str]:
    path_text = str(getattr(args, "gripper_mask_shape_profile", "") or "").strip()
    if not path_text:
        return None, "mask_shape_profile disabled"
    path = Path(path_text)
    if not path.exists():
        return None, f"mask_shape_profile missing: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"mask_shape_profile unreadable: {exc}"

    target_class = data.get("target_class")
    if target_class not in (None, "", "all", getattr(args, "target_class", None)):
        return None, f"mask_shape_profile ignored: target_class={target_class} != {args.target_class}"

    profile: dict[str, float] = {}
    stats = data.get("stats") if isinstance(data.get("stats"), dict) else {}
    for key in MASK_SHAPE_FEATURE_KEYS:
        value = None
        if isinstance(stats.get(key), dict) and stats[key].get("median") is not None:
            value = stats[key].get("median")
        elif data.get(key) is not None:
            value = data.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            profile[key] = number
    if float(profile.get("mask_area_px", 0.0)) <= 0.0:
        return None, f"mask_shape_profile has no usable area: {path}"
    return profile, f"mask_shape_profile loaded: {path}"


def gripper_mask_shape_reference_from_calibration(
    calib: CalibrationState,
    args,
) -> tuple[dict[str, float] | None, str]:
    class_agnostic = use_class_agnostic_points(args)
    rows: list[dict[str, float]] = []
    for point in calib.points:
        features = point.get("gripper_mask_features")
        if not isinstance(features, dict):
            continue
        if not class_agnostic and point.get("target_class") not in (None, args.target_class):
            continue
        try:
            confidence = float(point.get("gripper_confidence", point.get("confidence", 0.0)) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < float(args.gripper_reference_min_conf):
            continue
        numeric: dict[str, float] = {}
        for key in MASK_SHAPE_FEATURE_KEYS:
            try:
                value = float(features.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if math.isfinite(value):
                numeric[key] = value
        if float(numeric.get("mask_area_px", 0.0)) > 0.0:
            rows.append(numeric)
    if not rows:
        return None, "mask_shape calib reference unavailable"
    reference = {
        key: float(np.median([row.get(key, 0.0) for row in rows]))
        for key in MASK_SHAPE_FEATURE_KEYS
    }
    return reference, f"mask_shape calib reference n={len(rows)}"


def gripper_mask_shape_penalty(
    features: dict[str, float],
    reference: dict[str, float],
    args,
) -> tuple[float, str]:
    if not features or not reference:
        return 0.0, "shape n/a"
    area = float(features.get("mask_area_px", 0.0) or 0.0)
    ref_area = float(reference.get("mask_area_px", 0.0) or 0.0)
    min_area = max(0.0, float(args.gripper_area_guard_min_px))
    if area < min_area or ref_area < min_area:
        return 0.0, "shape area below guard"

    penalty = 0.0
    notes: list[str] = []
    area_tol = max(0.01, float(args.gripper_mask_area_tolerance))
    lower = 1.0 / (1.0 + area_tol)
    upper = 1.0 + area_tol
    area_ratio = area / max(ref_area, 1e-6)
    if area_ratio < lower:
        excess = (lower - area_ratio) / max(area_tol, 1e-6)
        penalty += excess
        notes.append(f"area_ratio={area_ratio:.2f}<{lower:.2f}")
    elif area_ratio > upper:
        excess = (area_ratio - upper) / max(area_tol, 1e-6)
        penalty += excess
        notes.append(f"area_ratio={area_ratio:.2f}>{upper:.2f}")

    axis = float(features.get("mask_axis_ratio", 0.0) or features.get("mask_rect_aspect", 0.0) or 0.0)
    ref_axis = float(reference.get("mask_axis_ratio", 0.0) or reference.get("mask_rect_aspect", 0.0) or 0.0)
    axis_tol = max(0.01, float(args.gripper_mask_axis_tolerance))
    if axis > 0.0 and ref_axis > 0.0:
        axis_rel = abs(axis - ref_axis) / max(ref_axis, 1e-6)
        if axis_rel > axis_tol:
            penalty += (axis_rel - axis_tol) / axis_tol
            notes.append(f"axis={axis:.2f} ref={ref_axis:.2f}")

    circ = float(features.get("mask_circularity", 0.0) or 0.0)
    ref_circ = float(reference.get("mask_circularity", 0.0) or 0.0)
    circ_drop = max(0.0, float(args.gripper_mask_circularity_drop))
    if circ > 0.0 and ref_circ > 0.0 and circ < ref_circ - circ_drop:
        penalty += (ref_circ - circ - circ_drop) / max(circ_drop, 0.05)
        notes.append(f"circ={circ:.2f}<ref-{circ_drop:.2f}")

    solidity = float(features.get("mask_solidity", 0.0) or 0.0)
    ref_solidity = float(reference.get("mask_solidity", 0.0) or 0.0)
    solidity_drop = max(0.0, float(args.gripper_mask_solidity_drop))
    if solidity > 0.0 and ref_solidity > 0.0 and solidity < ref_solidity - solidity_drop:
        penalty += (ref_solidity - solidity - solidity_drop) / max(solidity_drop, 0.03)
        notes.append(f"sol={solidity:.2f}<ref-{solidity_drop:.2f}")

    if not notes:
        notes.append(f"area_ratio={area_ratio:.2f}")
    return penalty, "; ".join(notes)


def gripper_mask_shape_guard(
    start_observation: GripperServoObservation,
    end_observation: GripperServoObservation,
    reference: dict[str, float] | None,
    args,
) -> tuple[bool, str]:
    if not bool(args.gripper_mask_shape_guard):
        return True, "mask_shape_guard disabled"
    if reference is None:
        reference = start_observation.mask_features
    if not reference:
        return True, "mask_shape_guard no reference"
    start_penalty, start_note = gripper_mask_shape_penalty(start_observation.mask_features, reference, args)
    end_penalty, end_note = gripper_mask_shape_penalty(end_observation.mask_features, reference, args)
    tolerance = max(0.0, float(getattr(args, "gripper_mask_shape_worse_tolerance", 0.20)))
    if end_penalty <= 0.0:
        return True, f"mask_shape_guard ok; {end_note}"
    if start_penalty > 0.0 and end_penalty <= start_penalty + tolerance:
        return True, (
            f"mask_shape_guard not_worse end_penalty={end_penalty:.2f} "
            f"start_penalty={start_penalty:.2f}; {end_note}"
        )
    return False, (
        f"mask_shape_guard rejected end_penalty={end_penalty:.2f} "
        f"start_penalty={start_penalty:.2f}; start={start_note}; end={end_note}"
    )


def observe_gripper_error(
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
) -> tuple[TargetDetection | None, tuple[float, float] | None, float, float, float, str, str]:
    det, frame, det_status = detect_from_gripper_frame(
        gripper_capture,
        gripper_model,
        gripper_names,
        args,
        gripper_camera_lock,
        flush_frames=0 if args.dry_run else int(args.gripper_camera_flush_frames),
    )
    if frame is None:
        return None, None, 0.0, 0.0, float("inf"), "reference n/a", "camera_read_failed"
    if det is None:
        return None, None, 0.0, 0.0, float("inf"), "reference n/a", det_status
    ref_uv, ref_source = gripper_reference_pixel(calib, args, frame.shape, overhead_uv)
    if ref_uv is None:
        return det, None, 0.0, 0.0, float("inf"), ref_source, "reference_unavailable"
    err_u, err_v, dist = gripper_error_from_detection(det, ref_uv)
    return det, ref_uv, err_u, err_v, dist, ref_source, "ok"


def observe_gripper_error_median(
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
) -> GripperServoObservation:
    """Measure a stable image error from several fresh camera frames."""
    requested = 1 if bool(args.dry_run) else max(1, int(args.gripper_observation_samples))
    retries = max(0, int(args.gripper_observation_retries))
    minimum = 1 if requested == 1 else max(2, int(math.ceil(0.6 * requested)))
    last_status = "target_not_detected"

    for attempt in range(retries + 1):
        detections: list[TargetDetection] = []
        frame_shape = None
        statuses: list[str] = []
        for sample_index in range(requested):
            det, frame, det_status = detect_from_gripper_frame(
                gripper_capture,
                gripper_model,
                gripper_names,
                args,
                gripper_camera_lock,
                flush_frames=(
                    0
                    if args.dry_run or sample_index > 0
                    else max(0, int(args.gripper_camera_flush_frames))
                ),
            )
            statuses.append(det_status)
            if frame is not None:
                frame_shape = frame.shape
            if det is not None:
                detections.append(det)
            if sample_index + 1 < requested:
                time.sleep(max(0.0, float(args.gripper_observation_interval_s)))

        if len(detections) < minimum or frame_shape is None:
            last_status = (
                f"insufficient_detections n={len(detections)}/{requested} "
                f"attempt={attempt + 1}/{retries + 1} last={statuses[-1] if statuses else 'camera_read_failed'}"
            )
            continue

        centers = np.asarray([det.center_uv for det in detections], dtype=np.float64)
        areas = np.asarray([detection_area_px(det) for det in detections], dtype=np.float64)
        shape_features = median_mask_shape_features(detections)
        center = np.median(centers, axis=0)
        radial = np.linalg.norm(centers - center, axis=1)
        jitter = float(np.percentile(radial, 90)) if len(radial) > 1 else 0.0
        max_jitter = max(0.0, float(args.gripper_observation_max_jitter_px))
        if jitter > max_jitter:
            last_status = (
                f"unstable_detection jitter={jitter:.2f}px>{max_jitter:.2f}px "
                f"n={len(detections)}/{requested} attempt={attempt + 1}/{retries + 1}"
            )
            continue

        representative = max(detections, key=lambda item: float(item.confidence))
        detection = replace(
            representative,
            center_uv=(float(center[0]), float(center[1])),
            confidence=float(np.median([det.confidence for det in detections])),
        )
        ref_uv, ref_source = gripper_reference_pixel(calib, args, frame_shape, overhead_uv)
        if ref_uv is None:
            return GripperServoObservation(
                detection=detection,
                ref_uv=None,
                error_uv=(0.0, 0.0),
                dist_px=float("inf"),
                ref_source=ref_source,
                sample_count=len(detections),
                jitter_px=jitter,
                status="reference_unavailable",
                target_area_px=float(np.median(areas)) if len(areas) else 0.0,
                mask_features=shape_features,
            )
        err_u, err_v, dist = gripper_error_from_detection(detection, ref_uv)
        return GripperServoObservation(
            detection=detection,
            ref_uv=ref_uv,
            error_uv=(err_u, err_v),
            dist_px=dist,
            ref_source=ref_source,
            sample_count=len(detections),
            jitter_px=jitter,
            status=f"ok median={len(detections)}/{requested} jitter={jitter:.2f}px attempt={attempt + 1}",
            target_area_px=float(np.median(areas)) if len(areas) else 0.0,
            mask_features=shape_features,
        )

    return GripperServoObservation(
        detection=None,
        ref_uv=None,
        error_uv=(0.0, 0.0),
        dist_px=float("inf"),
        ref_source="reference n/a",
        sample_count=0,
        jitter_px=float("inf"),
        status=last_status,
        target_area_px=0.0,
        mask_features={key: 0.0 for key in MASK_SHAPE_FEATURE_KEYS},
    )


def observe_gripper_error_stable(
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
) -> tuple[TargetDetection | None, tuple[float, float] | None, float, float, float, str, str]:
    """Adapt the robust multi-frame observation to the legacy tuple interface."""
    observation = observe_gripper_error_median(
        gripper_capture,
        gripper_model,
        gripper_names,
        calib,
        args,
        gripper_camera_lock,
        overhead_uv,
    )
    detail = f"{observation.ref_source}; {observation.status}"
    if observation.detection is None:
        return None, observation.ref_uv, 0.0, 0.0, float("inf"), detail, observation.status
    if observation.ref_uv is None:
        return observation.detection, None, 0.0, 0.0, float("inf"), detail, observation.status
    return (
        observation.detection,
        observation.ref_uv,
        float(observation.error_uv[0]),
        float(observation.error_uv[1]),
        float(observation.dist_px),
        detail,
        "ok" if str(observation.status).startswith("ok") else str(observation.status),
    )


def gripper_ibvs_tolerance_px(calib: CalibrationState, args) -> float:
    configured = max(1.0, float(args.gripper_center_tolerance_px))
    _center, learned_radius, refs, _raw, _filtered = robust_gripper_reference_data(calib, args)
    if not refs:
        return configured
    return max(configured, max(1.0, float(learned_radius)))


def gripper_motion_sample_quality(
    sample: dict[str, Any],
    args,
) -> tuple[bool, str]:
    delta_joint = sample.get("delta_joint")
    delta_pixel = sample.get("delta_pixel")
    if delta_joint is None or delta_pixel is None or len(delta_joint) != 2 or len(delta_pixel) != 2:
        return False, "motion sample rejected: malformed delta"
    try:
        dq = np.asarray([float(delta_joint[0]), float(delta_joint[1])], dtype=np.float64)
        dp = np.asarray([float(delta_pixel[0]), float(delta_pixel[1])], dtype=np.float64)
    except (TypeError, ValueError):
        return False, "motion sample rejected: non-numeric delta"
    if not np.all(np.isfinite(dq)) or not np.all(np.isfinite(dp)):
        return False, "motion sample rejected: non-finite delta"

    joint_norm = float(np.linalg.norm(dq))
    pixel_norm = float(np.linalg.norm(dp))
    min_actual = max(0.05, min(0.5, float(args.gripper_ibvs_probe_deg) * 0.10))
    if joint_norm < min_actual:
        return False, f"motion sample rejected: actual move {joint_norm:.3f}deg<{min_actual:.3f}deg"

    px_per_deg = pixel_norm / max(joint_norm, 1e-6)
    max_px_per_deg = max(1.0, float(args.gripper_motion_max_px_per_deg))
    if px_per_deg > max_px_per_deg:
        return False, (
            f"motion sample rejected: response {px_per_deg:.1f}px/deg exceeds "
            f"{max_px_per_deg:.1f}px/deg"
        )

    start_joints = sample.get("start_joints")
    end_joints = sample.get("end_joints")
    if isinstance(start_joints, dict) and isinstance(end_joints, dict):
        extra_deltas: list[float] = []
        for joint in MOTOR_KEYS:
            if joint in args.map_joints or joint == "gripper.pos":
                continue
            if joint not in start_joints or joint not in end_joints:
                continue
            try:
                extra_deltas.append(abs(float(end_joints[joint]) - float(start_joints[joint])))
            except (TypeError, ValueError):
                continue
        if extra_deltas:
            extra_norm = float(np.linalg.norm(extra_deltas))
            extra_limit = max(
                max(0.0, float(args.gripper_motion_extra_joint_max_deg)),
                max(0.0, float(args.gripper_motion_extra_joint_ratio)) * joint_norm,
            )
            if extra_norm > extra_limit:
                return False, (
                    f"motion sample rejected: extra joint motion {extra_norm:.3f}deg>"
                    f"{extra_limit:.3f}deg while map move {joint_norm:.3f}deg"
                )

    return True, f"motion sample quality ok px_per_deg={px_per_deg:.1f}"


def gripper_hover_input_range(calib: CalibrationState, args) -> tuple[str, float, float] | None:
    joint = args.map_joints[1] if len(args.map_joints) > 1 else "shoulder_lift.pos"
    ranges = (calib.hover_height_summary or {}).get("ranges", {})
    joint_range = ranges.get(joint) if isinstance(ranges, dict) else None
    if not isinstance(joint_range, list) or len(joint_range) != 2:
        return None
    try:
        low = float(joint_range[0]) - max(0.0, float(args.gripper_hover_input_margin_deg))
        high = float(joint_range[1]) + max(0.0, float(args.gripper_hover_input_margin_deg))
    except (TypeError, ValueError):
        return None
    return joint, low, high


def gripper_pose_height_guard(
    pose: dict[str, float],
    calib: CalibrationState,
    args,
    baseline_lift: float | None = None,
) -> tuple[bool, str]:
    joint = args.map_joints[1] if len(args.map_joints) > 1 else "shoulder_lift.pos"
    if joint not in pose:
        return True, "height_guard skipped: lift joint missing"
    lift = float(pose[joint])
    range_info = gripper_hover_input_range(calib, args)
    if range_info is not None:
        _range_joint, low, high = range_info
        if lift < low or lift > high:
            return False, f"height_guard {joint}={lift:.2f} outside hover range [{low:.2f},{high:.2f}]"
    if baseline_lift is not None:
        drift = abs(lift - float(baseline_lift))
        max_drift = max(0.0, float(args.gripper_max_cumulative_lift_drift_deg))
        if max_drift > 0 and drift > max_drift:
            return False, f"height_guard cumulative {joint} drift {drift:.2f}deg>{max_drift:.2f}deg"
    return True, "height_guard ok"


def append_gripper_motion_sample(
    state: CalibrationState,
    start_pose: dict[str, float],
    end_pose: dict[str, float],
    start_observation: GripperServoObservation,
    end_observation: GripperServoObservation,
    args,
    overhead_uv: tuple[float, float] | None,
    event: str,
) -> tuple[bool, str]:
    if start_observation.detection is None or end_observation.detection is None:
        return False, "motion sample rejected: detection missing"
    if not all(joint in start_pose and joint in end_pose for joint in args.map_joints):
        return False, "motion sample rejected: actual map joints missing"

    delta_joint = np.asarray(
        [float(end_pose[joint]) - float(start_pose[joint]) for joint in args.map_joints],
        dtype=np.float64,
    )
    start_pixel = np.asarray(start_observation.detection.center_uv, dtype=np.float64)
    end_pixel = np.asarray(end_observation.detection.center_uv, dtype=np.float64)
    delta_pixel = end_pixel - start_pixel
    joint_norm = float(np.linalg.norm(delta_joint))
    pixel_norm = float(np.linalg.norm(delta_pixel))
    px_per_deg = pixel_norm / max(joint_norm, 1e-6)

    next_id = 1 + max((int(sample.get("id") or 0) for sample in state.gripper_motion_samples), default=0)
    sample = {
        "id": next_id,
        "captured_at": now_iso(),
        "controller_version": "ibvs_v2",
        "event": str(event),
        "settled": True,
        "target_class": start_observation.detection.class_name,
        "point_kind": start_observation.detection.point_kind,
        "overhead_pixel": [float(overhead_uv[0]), float(overhead_uv[1])] if overhead_uv is not None else None,
        "start_xy": {joint: float(start_pose[joint]) for joint in args.map_joints},
        "end_xy": {joint: float(end_pose[joint]) for joint in args.map_joints},
        "start_joints": {joint: float(start_pose[joint]) for joint in MOTOR_KEYS if joint in start_pose},
        "end_joints": {joint: float(end_pose[joint]) for joint in MOTOR_KEYS if joint in end_pose},
        "delta_joint": [float(value) for value in delta_joint],
        "start_gripper_pixel": [float(value) for value in start_pixel],
        "end_gripper_pixel": [float(value) for value in end_pixel],
        "delta_pixel": [float(value) for value in delta_pixel],
        "start_error_uv": [float(value) for value in start_observation.error_uv],
        "end_error_uv": [float(value) for value in end_observation.error_uv],
        "start_jitter_px": float(start_observation.jitter_px),
        "end_jitter_px": float(end_observation.jitter_px),
        "start_mask_features": dict(start_observation.mask_features),
        "end_mask_features": dict(end_observation.mask_features),
        "px_per_deg": px_per_deg,
    }
    sample_quality_ok, sample_quality_status = gripper_motion_sample_quality(sample, args)
    if not sample_quality_ok:
        return False, sample_quality_status
    state.gripper_motion_samples.append(sample)
    maximum = max(20, int(args.gripper_motion_max_samples))
    if len(state.gripper_motion_samples) > maximum:
        state.gripper_motion_samples = state.gripper_motion_samples[-maximum:]
    return True, (
        f"motion sample {next_id} actual_dq=({delta_joint[0]:.3f},{delta_joint[1]:.3f})deg "
        f"dp=({delta_pixel[0]:.2f},{delta_pixel[1]:.2f})px"
    )


def local_gripper_motion_transitions(
    state: CalibrationState,
    current_xy: dict[str, float],
    overhead_uv: tuple[float, float] | None,
    args,
    minimum_sample_id: int = 0,
) -> list[tuple[list[float], list[float], int]]:
    current = np.asarray([float(current_xy[joint]) for joint in args.map_joints], dtype=np.float64)
    candidates: list[tuple[float, int, list[float], list[float]]] = []
    class_agnostic = use_class_agnostic_points(args)
    for sample in state.gripper_motion_samples:
        if sample.get("controller_version") != "ibvs_v2" or not bool(sample.get("settled")):
            continue
        sample_id = int(sample.get("id") or 0)
        if sample_id <= int(minimum_sample_id):
            continue
        if not class_agnostic and sample.get("target_class") not in (None, args.target_class):
            continue
        sample_quality_ok, _sample_quality_status = gripper_motion_sample_quality(sample, args)
        if not sample_quality_ok:
            continue
        start_joints = sample.get("start_joints")
        end_joints = sample.get("end_joints")
        if isinstance(start_joints, dict):
            ok_height, _height_status = gripper_pose_height_guard(start_joints, state, args)
            if not ok_height:
                continue
        if isinstance(end_joints, dict):
            ok_height, _height_status = gripper_pose_height_guard(end_joints, state, args)
            if not ok_height:
                continue
        delta_joint = sample.get("delta_joint")
        delta_pixel = sample.get("delta_pixel")
        start_xy = sample.get("start_xy")
        end_xy = sample.get("end_xy")
        if (
            delta_joint is None
            or delta_pixel is None
            or len(delta_joint) != 2
            or len(delta_pixel) != 2
            or not isinstance(start_xy, dict)
            or not isinstance(end_xy, dict)
            or not all(joint in start_xy and joint in end_xy for joint in args.map_joints)
        ):
            continue
        midpoint = np.asarray(
            [0.5 * (float(start_xy[joint]) + float(end_xy[joint])) for joint in args.map_joints],
            dtype=np.float64,
        )
        joint_distance = float(np.linalg.norm(midpoint - current))
        if joint_distance > max(0.0, float(args.gripper_motion_local_joint_radius_deg)):
            continue
        sample_overhead = sample.get("overhead_pixel")
        if overhead_uv is not None and sample_overhead is not None and len(sample_overhead) == 2:
            overhead_distance = math.hypot(
                float(sample_overhead[0]) - float(overhead_uv[0]),
                float(sample_overhead[1]) - float(overhead_uv[1]),
            )
            if overhead_distance > max(0.0, float(args.gripper_motion_local_overhead_radius_px)):
                continue
        candidates.append(
            (
                joint_distance,
                sample_id,
                [float(delta_joint[0]), float(delta_joint[1])],
                [float(delta_pixel[0]), float(delta_pixel[1])],
            )
        )
    candidates.sort(key=lambda item: (item[0], -item[1]))
    return [(dq, dp, sample_id) for _distance, sample_id, dq, dp in candidates[:60]]


def estimate_local_gripper_jacobian(
    state: CalibrationState,
    current_xy: dict[str, float],
    overhead_uv: tuple[float, float] | None,
    args,
    minimum_sample_id: int = 0,
):
    transitions = local_gripper_motion_transitions(
        state,
        current_xy,
        overhead_uv,
        args,
        minimum_sample_id,
    )
    estimate = estimate_image_jacobian(
        [(delta_joint, delta_pixel) for delta_joint, delta_pixel, _sample_id in transitions],
        min_joint_delta_deg=max(0.05, min(0.5, float(args.gripper_ibvs_probe_deg) * 0.10)),
        min_samples=4,
        require_symmetric_axes=True,
    )
    usable = jacobian_is_usable(
        estimate,
        max_condition=float(args.gripper_ibvs_max_condition),
        min_px_per_deg=float(args.gripper_ibvs_min_px_per_deg),
    )
    if estimate is not None and estimate.rmse_px > float(args.gripper_ibvs_max_rmse_px):
        usable = False
    ids = ",".join(str(sample_id) for _dq, _dp, sample_id in transitions[:8]) or "none"
    if estimate is None:
        return None, (
            f"local J unavailable samples={len(transitions)} ids={ids} "
            f"minimum_id>{int(minimum_sample_id)}"
        )
    status = (
        f"local J samples={estimate.sample_count} ids={ids} "
        f"cond={estimate.condition:.2f} min_sv={estimate.min_singular_value:.2f} "
        f"rmse={estimate.rmse_px:.2f}px"
    )
    return (estimate if usable else None), status


def gripper_primitive_sample_quality(
    sample: dict[str, Any],
    args,
) -> tuple[bool, str]:
    delta_action = sample.get("delta_action")
    delta_pixel = sample.get("delta_pixel")
    if delta_action is None or delta_pixel is None or len(delta_action) != 2 or len(delta_pixel) != 2:
        return False, "primitive sample rejected: malformed delta"
    try:
        da = np.asarray([float(delta_action[0]), float(delta_action[1])], dtype=np.float64)
        dp = np.asarray([float(delta_pixel[0]), float(delta_pixel[1])], dtype=np.float64)
    except (TypeError, ValueError):
        return False, "primitive sample rejected: non-numeric delta"
    if not np.all(np.isfinite(da)) or not np.all(np.isfinite(dp)):
        return False, "primitive sample rejected: non-finite delta"

    action_norm = float(np.linalg.norm(da))
    min_action = max(1e-4, float(getattr(args, "gripper_primitive_min_action", 0.10)))
    if action_norm < min_action:
        return False, f"primitive sample rejected: action {action_norm:.3f}<{min_action:.3f}"

    pixel_norm = float(np.linalg.norm(dp))
    px_per_unit = pixel_norm / max(action_norm, 1e-6)
    max_px_per_unit = max(1.0, float(getattr(args, "gripper_primitive_max_px_per_unit", 80.0)))
    if px_per_unit > max_px_per_unit:
        return False, (
            f"primitive sample rejected: response {px_per_unit:.1f}px/unit exceeds "
            f"{max_px_per_unit:.1f}px/unit"
        )

    return True, f"primitive sample quality ok px_per_unit={px_per_unit:.1f}"


def append_gripper_primitive_sample(
    state: CalibrationState,
    delta_action: list[float] | tuple[float, float],
    start_observation: GripperServoObservation,
    end_observation: GripperServoObservation,
    args,
    overhead_uv: tuple[float, float] | None,
    event: str,
    start_pose: dict[str, float] | None = None,
    end_pose: dict[str, float] | None = None,
) -> tuple[bool, str]:
    if start_observation.detection is None or end_observation.detection is None:
        return False, "primitive sample rejected: detection missing"
    start_pixel = np.asarray(start_observation.detection.center_uv, dtype=np.float64)
    end_pixel = np.asarray(end_observation.detection.center_uv, dtype=np.float64)
    delta_pixel = end_pixel - start_pixel
    action = np.asarray(delta_action, dtype=np.float64).reshape(2)

    next_id = 1 + max((int(sample.get("id") or 0) for sample in state.gripper_primitive_samples), default=0)
    sample = {
        "id": next_id,
        "captured_at": now_iso(),
        "controller_version": "primitive_ibvs_v1",
        "event": str(event),
        "settled": True,
        "target_class": start_observation.detection.class_name,
        "point_kind": start_observation.detection.point_kind,
        "overhead_pixel": [float(overhead_uv[0]), float(overhead_uv[1])] if overhead_uv is not None else None,
        "delta_action": [float(value) for value in action],
        "start_gripper_pixel": [float(value) for value in start_pixel],
        "end_gripper_pixel": [float(value) for value in end_pixel],
        "delta_pixel": [float(value) for value in delta_pixel],
        "start_error_uv": [float(value) for value in start_observation.error_uv],
        "end_error_uv": [float(value) for value in end_observation.error_uv],
        "start_jitter_px": float(start_observation.jitter_px),
        "end_jitter_px": float(end_observation.jitter_px),
    }
    if start_pose is not None:
        sample["start_joints"] = {joint: float(start_pose[joint]) for joint in MOTOR_KEYS if joint in start_pose}
    if end_pose is not None:
        sample["end_joints"] = {joint: float(end_pose[joint]) for joint in MOTOR_KEYS if joint in end_pose}

    ok, reason = gripper_primitive_sample_quality(sample, args)
    if not ok:
        return False, reason
    state.gripper_primitive_samples.append(sample)
    maximum = max(20, int(getattr(args, "gripper_primitive_max_samples", 400)))
    if len(state.gripper_primitive_samples) > maximum:
        state.gripper_primitive_samples = state.gripper_primitive_samples[-maximum:]
    return True, (
        f"primitive sample {next_id} da=({action[0]:.3f},{action[1]:.3f}) "
        f"dp=({delta_pixel[0]:.2f},{delta_pixel[1]:.2f})px"
    )


def local_gripper_primitive_transitions(
    state: CalibrationState,
    overhead_uv: tuple[float, float] | None,
    args,
    minimum_sample_id: int = 0,
) -> list[tuple[list[float], list[float], int]]:
    candidates: list[tuple[float, int, list[float], list[float]]] = []
    class_agnostic = use_class_agnostic_points(args)
    for sample in state.gripper_primitive_samples:
        if sample.get("controller_version") != "primitive_ibvs_v1" or not bool(sample.get("settled")):
            continue
        sample_id = int(sample.get("id") or 0)
        if sample_id <= int(minimum_sample_id):
            continue
        if not class_agnostic and sample.get("target_class") not in (None, args.target_class):
            continue
        ok, _reason = gripper_primitive_sample_quality(sample, args)
        if not ok:
            continue
        sample_overhead = sample.get("overhead_pixel")
        overhead_distance = 0.0
        if overhead_uv is not None and sample_overhead is not None and len(sample_overhead) == 2:
            overhead_distance = math.hypot(
                float(sample_overhead[0]) - float(overhead_uv[0]),
                float(sample_overhead[1]) - float(overhead_uv[1]),
            )
            radius = max(0.0, float(getattr(args, "gripper_primitive_local_overhead_radius_px", 140.0)))
            if overhead_distance > radius:
                continue
        delta_action = sample.get("delta_action")
        delta_pixel = sample.get("delta_pixel")
        candidates.append(
            (
                overhead_distance,
                sample_id,
                [float(delta_action[0]), float(delta_action[1])],
                [float(delta_pixel[0]), float(delta_pixel[1])],
            )
        )
    candidates.sort(key=lambda item: (item[0], -item[1]))
    return [(da, dp, sample_id) for _distance, sample_id, da, dp in candidates[:60]]


def estimate_local_gripper_primitive_jacobian(
    state: CalibrationState,
    overhead_uv: tuple[float, float] | None,
    args,
    minimum_sample_id: int = 0,
):
    transitions = local_gripper_primitive_transitions(
        state,
        overhead_uv,
        args,
        minimum_sample_id,
    )
    estimate = estimate_image_jacobian(
        [(delta_action, delta_pixel) for delta_action, delta_pixel, _sample_id in transitions],
        min_joint_delta_deg=max(1e-4, float(getattr(args, "gripper_primitive_min_action", 0.10))),
        min_samples=4,
        require_symmetric_axes=True,
    )
    usable = jacobian_is_usable(
        estimate,
        max_condition=float(args.gripper_ibvs_max_condition),
        min_px_per_deg=float(args.gripper_ibvs_min_px_per_deg),
    )
    if estimate is not None and estimate.rmse_px > float(args.gripper_ibvs_max_rmse_px):
        usable = False
    ids = ",".join(str(sample_id) for _da, _dp, sample_id in transitions[:8]) or "none"
    if estimate is None:
        return None, (
            f"primitive J unavailable samples={len(transitions)} ids={ids} "
            f"minimum_id>{int(minimum_sample_id)}"
        )
    status = (
        f"primitive J samples={estimate.sample_count} ids={ids} "
        f"cond={estimate.condition:.2f} min_sv={estimate.min_singular_value:.2f} "
        f"rmse={estimate.rmse_px:.2f}px"
    )
    return (estimate if usable else None), status


def pose_for_debug_pan_lift_target(
    baseline_pose: dict[str, float],
    args,
    delta_xy: np.ndarray | None = None,
) -> dict[str, float]:
    target = {joint: float(baseline_pose[joint]) for joint in MOTOR_KEYS if joint in baseline_pose}
    if delta_xy is not None:
        delta = np.asarray(delta_xy, dtype=np.float64).reshape(2)
        for index, joint in enumerate(args.map_joints[:2]):
            if joint in target:
                target[joint] = float(baseline_pose[joint]) + float(delta[index])
    return target


def pose_delta_from_baseline(
    pose: dict[str, float],
    baseline_pose: dict[str, float],
) -> dict[str, float]:
    return {
        joint: float(pose[joint]) - float(baseline_pose[joint])
        for joint in MOTOR_KEYS
        if joint in pose and joint in baseline_pose
    }


def max_pose_error(
    pose: dict[str, float],
    target: dict[str, float],
    joints: list[str],
) -> tuple[float, str]:
    errors: list[tuple[float, str, float, float]] = []
    for joint in joints:
        if joint not in pose or joint not in target:
            continue
        target_value = float(target[joint])
        observed_value = float(pose[joint])
        errors.append((abs(observed_value - target_value), joint, target_value, observed_value))
    if not errors:
        return 0.0, "no comparable joints"
    errors.sort(reverse=True, key=lambda item: item[0])
    err, joint, target_value, observed_value = errors[0]
    return float(err), f"worst {joint} err={err:.2f} target={target_value:.2f} obs={observed_value:.2f}"


def debug_noncontrolled_drift_status(
    pose: dict[str, float],
    baseline_pose: dict[str, float],
    args,
) -> tuple[bool, str]:
    controlled = set(args.map_joints[:2])
    warn = max(0.0, float(getattr(args, "gripper_debug_ibvs_uncontrolled_warn_deg", 0.5)))
    limit = max(warn, float(args.gripper_debug_ibvs_max_uncontrolled_drift_deg))
    drift_rows: list[tuple[float, str]] = []
    for joint in MOTOR_KEYS:
        if joint in controlled:
            continue
        if joint in pose and joint in baseline_pose:
            drift_rows.append((abs(float(pose[joint]) - float(baseline_pose[joint])), joint))
    if not drift_rows:
        return True, "uncontrolled_drift no joints"
    drift_rows.sort(reverse=True)
    max_drift, joint = drift_rows[0]
    if max_drift > limit:
        return False, f"uncontrolled_drift {joint}={max_drift:.3f}deg>{limit:.3f}deg"
    if max_drift > warn:
        return True, f"uncontrolled_drift warn max={max_drift:.3f}deg {joint}>{warn:.3f}deg stop>{limit:.3f}deg"
    return True, f"uncontrolled_drift ok max={max_drift:.3f}deg {joint} warn<={warn:.3f}deg stop>{limit:.3f}deg"


def wait_until_debug_pose(
    robot,
    target_pose: dict[str, float],
    args,
    robot_lock: threading.RLock | None = None,
    stop_event: threading.Event | None = None,
    timeout_s: float | None = None,
    resend_interval_s: float | None = None,
) -> tuple[bool, dict[str, float], str]:
    threshold = max(0.0, float(args.gripper_debug_ibvs_restore_threshold_deg))
    effective_timeout = (
        float(getattr(args, "gripper_debug_ibvs_restore_timeout_s", 8.0))
        if timeout_s is None
        else float(timeout_s)
    )
    deadline = time.perf_counter() + max(0.0, effective_timeout)
    if resend_interval_s is None:
        hold_interval = max(0.0, float(getattr(args, "gripper_debug_ibvs_restore_resend_interval_s", 0.05)))
    else:
        hold_interval = max(0.0, float(resend_interval_s))
    last_hold_sent_at = 0.0
    joints = [joint for joint in MOTOR_KEYS if joint in target_pose]
    last_pose = locked_observation(robot, robot_lock)
    last_status = "debug_pose_wait not started"
    previous: dict[str, float] | None = None
    stable_count = 0
    max_motion = float("inf")
    stable_epsilon = max(0.0, float(args.gripper_refine_stable_epsilon_deg))
    required_stable = max(1, int(args.gripper_refine_stable_samples))
    while time.perf_counter() <= deadline:
        if stop_event is not None and stop_event.is_set():
            return False, locked_observation(robot, robot_lock), "debug_pose_wait cancelled"
        pose = locked_observation(robot, robot_lock)
        last_pose = pose
        err, detail = max_pose_error(pose, target_pose, joints)
        active = [joint for joint in joints if joint in pose]
        if not active:
            return False, pose, "debug_pose_wait failed: no target joints in observation"
        if previous is not None:
            max_motion = max(abs(float(pose[joint]) - float(previous[joint])) for joint in active)
            stable_count = stable_count + 1 if max_motion <= stable_epsilon else 0
        previous = {joint: float(pose[joint]) for joint in active}
        last_status = (
            f"debug_pose_wait max_error={err:.2f} threshold={threshold:.2f} "
            f"stable={stable_count}/{required_stable} motion={max_motion:.3f}; {detail}"
        )
        if err <= threshold and stable_count >= required_stable:
            return True, pose, last_status
        now = time.perf_counter()
        if hold_interval > 0 and now - last_hold_sent_at >= hold_interval:
            send_action_once(robot, target_pose, robot_lock)
            last_hold_sent_at = now
        time.sleep(0.02)
    return False, last_pose, "debug_pose_wait timeout; " + last_status


def wait_until_debug_stable(
    robot,
    args,
    robot_lock: threading.RLock | None = None,
    stop_event: threading.Event | None = None,
    timeout_s: float | None = None,
) -> tuple[bool, dict[str, float], str]:
    joints = [joint for joint in MOTOR_KEYS if joint]
    effective_timeout = float(args.gripper_refine_wait_timeout_s) if timeout_s is None else float(timeout_s)
    deadline = time.perf_counter() + max(0.0, effective_timeout)
    stable_epsilon = max(0.0, float(args.gripper_refine_stable_epsilon_deg))
    required_stable = max(1, int(args.gripper_refine_stable_samples))
    previous: dict[str, float] | None = None
    stable_count = 0
    max_motion = float("inf")
    last_pose = locked_observation(robot, robot_lock)
    while time.perf_counter() <= deadline:
        if stop_event is not None and stop_event.is_set():
            return False, locked_observation(robot, robot_lock), "debug_stable_wait cancelled"
        pose = locked_observation(robot, robot_lock)
        last_pose = pose
        active = [joint for joint in joints if joint in pose]
        if not active:
            return False, pose, "debug_stable_wait failed: no motor joints in observation"
        if previous is not None:
            max_motion = max(abs(float(pose[joint]) - float(previous[joint])) for joint in active)
            stable_count = stable_count + 1 if max_motion <= stable_epsilon else 0
            if stable_count >= required_stable:
                return True, pose, (
                    f"debug_stable_wait ok motion={max_motion:.3f}deg "
                    f"samples={stable_count}/{required_stable}"
                )
        previous = {joint: float(pose[joint]) for joint in active}
        time.sleep(0.02)
    return False, last_pose, (
        f"debug_stable_wait timeout motion={max_motion:.3f}deg "
        f"samples={stable_count}/{required_stable}"
    )


def run_gripper_debug_ibvs_xy_refine(
    robot,
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    xy_target: dict[str, float],
    robot_lock: threading.RLock | None = None,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
    stop_event: threading.Event | None = None,
) -> str:
    """One-shot debug IBVS in local command space with symmetric pan/lift primitives."""
    del xy_target
    if len(args.map_joints) < 2:
        return "INCOMPLETE: debug_ibvs needs two map_joints"

    trace_handle, trace_writer, _trace_path = open_gripper_trace(args)
    controlled = list(args.map_joints[:2])
    probe = abs(float(args.gripper_debug_ibvs_probe_deg))
    max_step = abs(float(args.gripper_debug_ibvs_max_step_deg))
    min_step = min(max_step, abs(float(args.gripper_debug_ibvs_min_step_deg)))
    min_controlled = max(0.0, float(args.gripper_debug_ibvs_min_actual_controlled_deg))
    min_pixel_motion = max(0.0, float(args.gripper_debug_ibvs_min_pixel_motion_px))
    max_jitter = max(0.0, float(args.gripper_debug_ibvs_max_jitter_px))
    tolerance = gripper_ibvs_tolerance_px(calib, args)

    def cancelled() -> bool:
        return stop_event is not None and stop_event.is_set()

    def debug_trace(
        event: str,
        iter_index: int,
        observation: GripperServoObservation,
        pose: dict[str, float],
        status: str,
        command_step: np.ndarray | None = None,
        target_pose: dict[str, float] | None = None,
        baseline_pose: dict[str, float] | None = None,
        jacobian: np.ndarray | None = None,
        condition: float | None = None,
        restore_ok: bool | None = None,
    ) -> None:
        det = observation.detection
        ref_uv = observation.ref_uv
        baseline = baseline_pose or {}
        target = target_pose or {}
        deltas = pose_delta_from_baseline(pose, baseline) if baseline else {}
        row: dict[str, Any] = {
            "event": event,
            "iter": iter_index,
            "servo_mode": "debug_ibvs",
            "center_u": f"{det.center_uv[0]:.3f}" if det is not None else "",
            "center_v": f"{det.center_uv[1]:.3f}" if det is not None else "",
            "ref_u": f"{ref_uv[0]:.3f}" if ref_uv is not None else "",
            "ref_v": f"{ref_uv[1]:.3f}" if ref_uv is not None else "",
            "err_u": f"{observation.error_uv[0]:.3f}",
            "err_v": f"{observation.error_uv[1]:.3f}",
            "dist_px": f"{observation.dist_px:.3f}" if math.isfinite(observation.dist_px) else "",
            "delta_pan": f"{command_step[0]:.4f}" if command_step is not None else "",
            "delta_lift": f"{command_step[1]:.4f}" if command_step is not None else "",
            "target_pan": f"{target.get(controlled[0], float('nan')):.4f}" if target else "",
            "target_lift": f"{target.get(controlled[1], float('nan')):.4f}" if target else "",
            "actual_pan": f"{pose.get(controlled[0], float('nan')):.4f}",
            "actual_lift": f"{pose.get(controlled[1], float('nan')):.4f}",
            "actual_elbow": f"{pose.get('elbow_flex.pos', float('nan')):.4f}",
            "actual_wrist_flex": f"{pose.get('wrist_flex.pos', float('nan')):.4f}",
            "actual_wrist_roll": f"{pose.get('wrist_roll.pos', float('nan')):.4f}",
            "actual_gripper": f"{pose.get('gripper.pos', float('nan')):.4f}",
            "baseline_pan": f"{baseline.get(controlled[0], float('nan')):.4f}" if baseline else "",
            "baseline_lift": f"{baseline.get(controlled[1], float('nan')):.4f}" if baseline else "",
            "baseline_elbow": f"{baseline.get('elbow_flex.pos', float('nan')):.4f}" if baseline else "",
            "baseline_wrist_flex": f"{baseline.get('wrist_flex.pos', float('nan')):.4f}" if baseline else "",
            "baseline_wrist_roll": f"{baseline.get('wrist_roll.pos', float('nan')):.4f}" if baseline else "",
            "baseline_gripper": f"{baseline.get('gripper.pos', float('nan')):.4f}" if baseline else "",
            "target_elbow": f"{target.get('elbow_flex.pos', float('nan')):.4f}" if target else "",
            "target_wrist_flex": f"{target.get('wrist_flex.pos', float('nan')):.4f}" if target else "",
            "target_wrist_roll": f"{target.get('wrist_roll.pos', float('nan')):.4f}" if target else "",
            "target_gripper": f"{target.get('gripper.pos', float('nan')):.4f}" if target else "",
            "actual_delta_pan": f"{deltas.get(controlled[0], float('nan')):.4f}" if deltas else "",
            "actual_delta_lift": f"{deltas.get(controlled[1], float('nan')):.4f}" if deltas else "",
            "actual_delta_elbow": f"{deltas.get('elbow_flex.pos', float('nan')):.4f}" if deltas else "",
            "actual_delta_wrist_flex": f"{deltas.get('wrist_flex.pos', float('nan')):.4f}" if deltas else "",
            "actual_delta_wrist_roll": f"{deltas.get('wrist_roll.pos', float('nan')):.4f}" if deltas else "",
            "actual_delta_gripper": f"{deltas.get('gripper.pos', float('nan')):.4f}" if deltas else "",
            "target_area_px": f"{observation.target_area_px:.2f}" if observation.target_area_px > 0 else "",
            "j00_du_dpan": f"{jacobian[0, 0]:.5f}" if jacobian is not None else "",
            "j01_du_dlift": f"{jacobian[0, 1]:.5f}" if jacobian is not None else "",
            "j10_dv_dpan": f"{jacobian[1, 0]:.5f}" if jacobian is not None else "",
            "j11_dv_dlift": f"{jacobian[1, 1]:.5f}" if jacobian is not None else "",
            "condition": f"{condition:.4f}" if condition is not None else "",
            "observation_samples": observation.sample_count,
            "observation_jitter_px": f"{observation.jitter_px:.4f}" if math.isfinite(observation.jitter_px) else "",
            "ref_source": observation.ref_source,
            "restore_ok": "" if restore_ok is None else int(bool(restore_ok)),
            "status": status,
        }
        for key in MASK_SHAPE_FEATURE_KEYS:
            if key == "mask_area_px":
                continue
            row[key] = (
                f"{float(observation.mask_features.get(key, 0.0)):.4f}"
                if key in observation.mask_features
                else ""
            )
        write_gripper_trace(trace_writer, **row)

    def observe() -> GripperServoObservation:
        return observe_gripper_error_median(
            gripper_capture,
            gripper_model,
            gripper_names,
            calib,
            args,
            gripper_camera_lock,
            overhead_uv,
        )

    def restore_baseline(
        baseline_pose: dict[str, float],
        iter_index: int,
        reason: str,
    ) -> tuple[bool, dict[str, float], GripperServoObservation, str]:
        target = pose_for_debug_pan_lift_target(baseline_pose, args)
        send_interpolated(
            robot,
            target,
            args,
            robot_lock,
            speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
            min_steps=int(args.gripper_refine_min_move_steps),
            stop_event=stop_event,
        )
        if cancelled():
            pose = locked_observation(robot, robot_lock)
            observation = observe()
            status = f"{reason}; strict_restore cancelled"
            debug_trace("debug_restore", iter_index, observation, pose, status, target_pose=target, baseline_pose=baseline_pose, restore_ok=False)
            return False, pose, observation, status
        ok_wait, pose, wait_status = wait_until_debug_pose(
            robot,
            target,
            args,
            robot_lock,
            stop_event,
            timeout_s=float(getattr(args, "gripper_debug_ibvs_restore_timeout_s", 8.0)),
            resend_interval_s=float(getattr(args, "gripper_debug_ibvs_restore_resend_interval_s", 0.05)),
        )
        observation = observe()
        max_err, detail = max_pose_error(pose, target, [joint for joint in MOTOR_KEYS if joint in target])
        ok = bool(ok_wait and max_err <= float(args.gripper_debug_ibvs_restore_threshold_deg))
        status = f"{reason}; strict_restore {'ok' if ok else 'failed'}; {wait_status}; {detail}"
        debug_trace("debug_restore", iter_index, observation, pose, status, target_pose=target, baseline_pose=baseline_pose, restore_ok=ok)
        return ok, pose, observation, status

    def restore_specific_pose(
        target_pose: dict[str, float],
        iter_index: int,
        reason: str,
    ) -> tuple[bool, dict[str, float], GripperServoObservation, str]:
        target = {joint: float(target_pose[joint]) for joint in MOTOR_KEYS if joint in target_pose}
        send_interpolated(
            robot,
            target,
            args,
            robot_lock,
            speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
            min_steps=int(args.gripper_refine_min_move_steps),
            stop_event=stop_event,
        )
        if cancelled():
            pose = locked_observation(robot, robot_lock)
            observation = observe()
            status = f"{reason}; strict_restore_pose cancelled"
            debug_trace("debug_restore_pose", iter_index, observation, pose, status, target_pose=target, baseline_pose=target, restore_ok=False)
            return False, pose, observation, status
        ok_wait, pose, wait_status = wait_until_debug_pose(
            robot,
            target,
            args,
            robot_lock,
            stop_event,
            timeout_s=float(getattr(args, "gripper_debug_ibvs_restore_timeout_s", 8.0)),
            resend_interval_s=float(getattr(args, "gripper_debug_ibvs_restore_resend_interval_s", 0.05)),
        )
        observation = observe()
        max_err, detail = max_pose_error(pose, target, [joint for joint in MOTOR_KEYS if joint in target])
        ok = bool(ok_wait and max_err <= float(args.gripper_debug_ibvs_restore_threshold_deg))
        status = f"{reason}; strict_restore_pose {'ok' if ok else 'failed'}; {wait_status}; {detail}"
        debug_trace("debug_restore_pose", iter_index, observation, pose, status, target_pose=target, baseline_pose=target, restore_ok=ok)
        return ok, pose, observation, status

    def validate_probe(
        label: str,
        command_step: np.ndarray,
        pre_pose: dict[str, float],
        post_pose: dict[str, float],
        pre_observation: GripperServoObservation,
        post_observation: GripperServoObservation,
    ) -> tuple[bool, str, np.ndarray, np.ndarray]:
        deltas = pose_delta_from_baseline(post_pose, pre_pose)
        actual_xy = np.asarray([deltas.get(controlled[0], 0.0), deltas.get(controlled[1], 0.0)], dtype=np.float64)
        axis = 0 if abs(float(command_step[0])) >= abs(float(command_step[1])) else 1
        expected_sign = 1.0 if float(command_step[axis]) >= 0 else -1.0
        actual_axis = expected_sign * float(actual_xy[axis])
        if actual_axis < min_controlled:
            return False, f"{label} invalid: controlled axis actual {actual_axis:.3f}deg<{min_controlled:.3f}deg actual=({actual_xy[0]:.3f},{actual_xy[1]:.3f})", actual_xy, np.zeros(2, dtype=np.float64)
        if (
            pre_observation.detection is None
            or post_observation.detection is None
            or pre_observation.ref_uv is None
            or post_observation.ref_uv is None
        ):
            return False, f"{label} invalid: missing pre/post detection", actual_xy, np.zeros(2, dtype=np.float64)
        pre_center = np.asarray(pre_observation.detection.center_uv, dtype=np.float64)
        post_center = np.asarray(post_observation.detection.center_uv, dtype=np.float64)
        delta_pixel = post_center - pre_center
        pixel_motion = float(np.linalg.norm(delta_pixel))
        if pixel_motion < min_pixel_motion:
            return False, f"{label} invalid: pixel motion {pixel_motion:.2f}px<{min_pixel_motion:.2f}px", actual_xy, delta_pixel
        pre_jitter = float(pre_observation.jitter_px)
        post_jitter = float(post_observation.jitter_px)
        if (
            not math.isfinite(pre_jitter)
            or not math.isfinite(post_jitter)
            or max(pre_jitter, post_jitter) > max_jitter
        ):
            return False, f"{label} invalid: jitter pre={pre_jitter:.2f}px post={post_jitter:.2f}px max={max_jitter:.2f}px", actual_xy, delta_pixel
        ok_drift, drift_status = debug_noncontrolled_drift_status(post_pose, pre_pose, args)
        if not ok_drift:
            return False, f"{label} invalid: {drift_status}", actual_xy, delta_pixel
        return True, (
            f"{label} valid command=({command_step[0]:.3f},{command_step[1]:.3f})deg "
            f"actual=({actual_xy[0]:.3f},{actual_xy[1]:.3f})deg "
            f"dp=({delta_pixel[0]:.2f},{delta_pixel[1]:.2f})px "
            f"motion={pixel_motion:.2f}px jitter=({pre_jitter:.2f},{post_jitter:.2f})px; {drift_status}"
        ), actual_xy, delta_pixel

    try:
        if cancelled():
            return "STOP: debug_ibvs cancelled before baseline"
        baseline_pose = locked_observation(robot, robot_lock)
        if not all(joint in baseline_pose for joint in MOTOR_KEYS):
            missing = [joint for joint in MOTOR_KEYS if joint not in baseline_pose]
            return f"INCOMPLETE: debug_ibvs baseline missing joints: {missing}"
        baseline_observation = observe()
        debug_trace("debug_baseline", 0, baseline_observation, baseline_pose, "strict debug baseline", baseline_pose=baseline_pose)
        if baseline_observation.detection is None or baseline_observation.ref_uv is None:
            return f"INCOMPLETE: debug_ibvs initial observation failed: {baseline_observation.status}"
        if baseline_observation.dist_px <= tolerance:
            return (
                f"DONE: debug_ibvs already aligned dist={baseline_observation.dist_px:.1f}px "
                f"accept<={tolerance:.1f}px"
            )

        probe_specs = [
            ("pan_pos", np.asarray([probe, 0.0], dtype=np.float64)),
            ("pan_neg", np.asarray([-probe, 0.0], dtype=np.float64)),
            ("lift_pos", np.asarray([0.0, probe], dtype=np.float64)),
            ("lift_neg", np.asarray([0.0, -probe], dtype=np.float64)),
        ]
        probe_results: dict[str, dict[str, Any]] = {}
        for index, (label, command_step) in enumerate(probe_specs, start=1):
            ok_restore, restore_pose, _restore_obs, restore_status = restore_baseline(
                baseline_pose,
                index,
                f"before {label}",
            )
            if not ok_restore:
                return f"STOP: debug_ibvs strict restore failed before probe {label}; {restore_status}"
            pre_probe_pose = locked_observation(robot, robot_lock)
            pre_probe_observation = observe()
            debug_trace(
                "debug_pre_probe",
                index,
                pre_probe_observation,
                pre_probe_pose,
                f"pre_probe baseline after restore {label}; {restore_status}",
                baseline_pose=pre_probe_pose,
                restore_ok=True,
            )
            if pre_probe_observation.detection is None or pre_probe_observation.ref_uv is None:
                return f"STOP: debug_ibvs pre_probe observation failed {label}: {pre_probe_observation.status}"
            target_pose = pose_for_debug_pan_lift_target(pre_probe_pose, args, command_step)
            send_interpolated(
                robot,
                target_pose,
                args,
                robot_lock,
                speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
                min_steps=int(args.gripper_refine_min_move_steps),
                stop_event=stop_event,
            )
            if cancelled():
                return f"STOP: debug_ibvs cancelled during probe {label}"
            ok_wait, probe_pose, wait_status = wait_until_debug_stable(
                robot,
                args,
                robot_lock,
                stop_event,
                timeout_s=float(args.gripper_refine_wait_timeout_s),
            )
            probe_observation = observe()
            valid_probe, probe_status, actual_xy, delta_pixel = validate_probe(
                label,
                command_step,
                pre_probe_pose,
                probe_pose,
                pre_probe_observation,
                probe_observation,
            )
            status = f"{wait_status}; {probe_status}"
            debug_trace(
                "debug_probe",
                index,
                probe_observation,
                probe_pose,
                status,
                command_step=command_step,
                target_pose=target_pose,
                baseline_pose=pre_probe_pose,
                restore_ok=ok_wait,
            )
            if not ok_wait:
                ok_restore_after, _pose_after, _obs_after, restore_after_status = restore_baseline(
                    baseline_pose,
                    index,
                    f"after failed target wait {label}",
                )
                return f"STOP: debug_ibvs probe {label} target not reached; {status}; restore={ok_restore_after} {restore_after_status}"
            if not valid_probe or probe_observation.detection is None or probe_observation.ref_uv is None:
                ok_restore_after, _pose_after, _obs_after, restore_after_status = restore_baseline(
                    baseline_pose,
                    index,
                    f"after invalid {label}",
                )
                return f"STOP: debug_ibvs invalid probe {label}; {status}; restore={ok_restore_after} {restore_after_status}"
            probe_results[label] = {
                "center": np.asarray(probe_observation.detection.center_uv, dtype=np.float64),
                "pre_center": np.asarray(pre_probe_observation.detection.center_uv, dtype=np.float64),
                "delta_pixel": delta_pixel,
                "actual_xy": actual_xy,
                "observation": probe_observation,
                "pre_observation": pre_probe_observation,
                "pre_pose": pre_probe_pose,
                "post_pose": probe_pose,
            }
            ok_restore_after, _pose_after, _obs_after, restore_after_status = restore_baseline(
                baseline_pose,
                index,
                f"after {label}",
            )
            if not ok_restore_after:
                return f"STOP: debug_ibvs strict restore failed after probe {label}; {restore_after_status}"

        if len(probe_results) < 4:
            return f"STOP: debug_ibvs requires 4 valid probes, got {len(probe_results)}"

        command_den = 2.0 * probe
        if command_den <= 1e-6:
            return "STOP: debug_ibvs probe_deg must be non-zero for command-space central diff"
        pan_column = (probe_results["pan_pos"]["delta_pixel"] - probe_results["pan_neg"]["delta_pixel"]) / command_den
        lift_column = (probe_results["lift_pos"]["delta_pixel"] - probe_results["lift_neg"]["delta_pixel"]) / command_den
        jacobian = np.asarray(
            [[float(pan_column[0]), float(lift_column[0])], [float(pan_column[1]), float(lift_column[1])]],
            dtype=np.float64,
        )
        singular = np.linalg.svd(jacobian, compute_uv=False)
        condition = float(singular[0] / max(singular[-1], 1e-9))
        debug_trace(
            "debug_jacobian",
            5,
            baseline_observation,
            baseline_pose,
            f"command_space central_diff ok command_den={command_den:.3f}deg cond={condition:.2f}",
            baseline_pose=baseline_pose,
            jacobian=jacobian,
            condition=condition,
        )
        if (
            not np.all(np.isfinite(jacobian))
            or condition > float(args.gripper_ibvs_max_condition)
            or float(singular[-1]) < float(args.gripper_ibvs_min_px_per_deg)
        ):
            return f"STOP: debug_ibvs unusable central diff J cond={condition:.2f} min_sv={float(singular[-1]):.2f}"

        ok_restore, correction_pre_pose, correction_pre_observation, restore_status = restore_baseline(
            baseline_pose,
            6,
            "before correction",
        )
        if not ok_restore:
            return f"STOP: debug_ibvs strict restore failed before correction; {restore_status}"
        if correction_pre_observation.detection is None or correction_pre_observation.ref_uv is None:
            return f"STOP: debug_ibvs correction pre observation failed: {correction_pre_observation.status}"
        debug_trace(
            "debug_pre_correction",
            6,
            correction_pre_observation,
            correction_pre_pose,
            f"pre_correction baseline after restore; {restore_status}",
            baseline_pose=correction_pre_pose,
            restore_ok=True,
        )

        command_step = damped_least_squares_step(
            jacobian,
            correction_pre_observation.error_uv,
            gain=float(args.gripper_ibvs_gain),
            damping=float(args.gripper_ibvs_damping),
            max_abs_step_deg=max_step,
        )
        command_step = ensure_minimum_effective_step(
            command_step,
            minimum_abs_step_deg=min_step,
            maximum_abs_step_deg=max_step,
        )
        predicted_error = np.asarray(correction_pre_observation.error_uv, dtype=np.float64) + jacobian @ command_step
        if float(np.linalg.norm(predicted_error)) >= correction_pre_observation.dist_px - 0.05:
            return (
                f"STOP: debug_ibvs predicted correction does not improve "
                f"{correction_pre_observation.dist_px:.2f}->{float(np.linalg.norm(predicted_error)):.2f}px"
            )

        correction_target = pose_for_debug_pan_lift_target(correction_pre_pose, args, command_step)
        send_interpolated(
            robot,
            correction_target,
            args,
            robot_lock,
            speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
            min_steps=int(args.gripper_refine_min_move_steps),
            stop_event=stop_event,
        )
        if cancelled():
            return "STOP: debug_ibvs cancelled during correction"
        ok_wait, correction_pose, wait_status = wait_until_debug_stable(
            robot,
            args,
            robot_lock,
            stop_event,
            timeout_s=float(args.gripper_refine_wait_timeout_s),
        )
        correction_observation = observe()
        ok_drift, drift_status = debug_noncontrolled_drift_status(correction_pose, correction_pre_pose, args)
        correction_status = f"{wait_status}; {drift_status}"
        debug_trace(
            "debug_correction",
            7,
            correction_observation,
            correction_pose,
            correction_status,
            command_step=command_step,
            target_pose=correction_target,
            baseline_pose=correction_pre_pose,
            jacobian=jacobian,
            condition=condition,
            restore_ok=ok_wait,
        )
        if not ok_wait or not ok_drift or correction_observation.detection is None or correction_observation.ref_uv is None:
            ok_restore_after, _pose_after, _obs_after, restore_after_status = restore_specific_pose(
                correction_pre_pose,
                8,
                "after invalid correction restore pre-correction",
            )
            return f"STOP: debug_ibvs correction invalid; {correction_status}; restore={ok_restore_after} {restore_after_status}"

        if correction_observation.dist_px >= correction_pre_observation.dist_px:
            ok_restore_after, _pose_after, _obs_after, restore_after_status = restore_specific_pose(
                correction_pre_pose,
                8,
                f"correction did not improve {correction_pre_observation.dist_px:.1f}->{correction_observation.dist_px:.1f}px",
            )
            return (
                f"STOP: debug_ibvs correction worsened/not improved "
                f"{correction_pre_observation.dist_px:.1f}->{correction_observation.dist_px:.1f}px; "
                f"restore={ok_restore_after} {restore_after_status}"
            )

        if correction_observation.dist_px <= tolerance:
            return (
                f"DONE: debug_ibvs one correction aligned "
                f"{correction_pre_observation.dist_px:.1f}->{correction_observation.dist_px:.1f}px "
                f"accept<={tolerance:.1f}px"
            )
        return (
            f"INCOMPLETE: debug_ibvs one correction improved "
            f"{correction_pre_observation.dist_px:.1f}->{correction_observation.dist_px:.1f}px; "
            "stopping after one debug correction by design"
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()


def gripper_search_directions(args, preferred: tuple[float, float] | None) -> list[tuple[float, float]]:
    directions = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
    if bool(args.gripper_search_diagonal):
        directions.extend([(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)])
    if preferred is None:
        return directions
    ordered = [preferred]
    ordered.extend(direction for direction in directions if direction != preferred)
    return ordered


def run_gripper_search_xy_refine(
    robot,
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    xy_target: dict[str, float],
    robot_lock: threading.RLock | None = None,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
    stop_event: threading.Event | None = None,
) -> str:
    current_xy = {joint: float(xy_target[joint]) for joint in args.map_joints if joint in xy_target}
    if len(current_xy) != len(args.map_joints):
        return "gripper_search=skipped_missing_xy_target"

    trace_handle, trace_writer, _trace_path = open_gripper_trace(args)
    min_step = abs(float(args.gripper_search_min_step_deg))
    improve_px = max(0.0, float(args.gripper_search_improve_px))
    max_iters = int(args.gripper_final_iters)
    preferred_direction: tuple[float, float] | None = None
    step_scale = 1.0
    status = "gripper_search=no_iterations"
    i = 0
    try:
        while max_iters <= 0 or i < max_iters:
            if stop_event is not None and stop_event.is_set():
                return "STOP: gripper_search cancelled by user before reaching tolerance"
            i += 1
            actual_xy = observed_map_xy(robot, args, robot_lock)
            if len(actual_xy) == len(args.map_joints):
                current_xy = actual_xy
            det, ref_uv, err_u, err_v, dist, ref_source, obs_status = observe_gripper_error_stable(
                gripper_capture,
                gripper_model,
                gripper_names,
                calib,
                args,
                gripper_camera_lock,
                overhead_uv,
            )
            if obs_status != "ok" or det is None or ref_uv is None:
                return f"gripper_search=stop_{obs_status} {ref_source}"
            adaptive_step = min(
                abs(float(args.gripper_search_step_deg)),
                adaptive_gripper_step_limit(dist, args),
            )
            step = adaptive_step * step_scale
            status = (
                f"gripper_search iter={i} step={step:.3f} adaptive={adaptive_step:.3f} scale={step_scale:.2f} "
                f"err=({err_u:.1f},{err_v:.1f}) "
                f"dist={dist:.1f}px center=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) "
                f"ref=({ref_uv[0]:.1f},{ref_uv[1]:.1f}) {ref_source}"
            )
            print(status, flush=True)
            write_gripper_trace(
                trace_writer,
                event="search_observe",
                iter=i,
                servo_mode="search",
                center_u=f"{det.center_uv[0]:.3f}",
                center_v=f"{det.center_uv[1]:.3f}",
                ref_u=f"{ref_uv[0]:.3f}",
                ref_v=f"{ref_uv[1]:.3f}",
                err_u=f"{err_u:.3f}",
                err_v=f"{err_v:.3f}",
                dist_px=f"{dist:.3f}",
                target_pan=f"{current_xy[args.map_joints[0]]:.4f}",
                target_lift=f"{current_xy[args.map_joints[1]]:.4f}",
                ref_source=ref_source,
                status=f"step={step:.4f}; adaptive={adaptive_step:.4f}; scale={step_scale:.3f}",
            )
            if gripper_error_within_tolerance(dist, args):
                return "DONE: " + status
            if step < min_step:
                return (
                    "INCOMPLETE: gripper_search local minimum; no improving move above min step. "
                    f"min_step={min_step:.3f}; {status}"
                )

            accepted = False
            base_xy = dict(current_xy)
            best_candidate_xy: dict[str, float] | None = None
            best_direction: tuple[float, float] | None = None
            best_improved_by = -math.inf
            best_dist = float("inf")
            restart_after_failed_move = False
            for direction in gripper_search_directions(args, preferred_direction):
                if stop_event is not None and stop_event.is_set():
                    return "STOP: gripper_search cancelled by user during direction search"
                candidate_xy = dict(base_xy)
                candidate_xy[args.map_joints[0]] = float(candidate_xy[args.map_joints[0]]) + direction[0] * step
                candidate_xy[args.map_joints[1]] = float(candidate_xy[args.map_joints[1]]) + direction[1] * step
                ok_move, move_reason = send_xy_hover_refine_target(robot, calib, args, candidate_xy, robot_lock)
                if not ok_move:
                    actual_after_failed = observed_map_xy(robot, args, robot_lock)
                    if len(actual_after_failed) == len(args.map_joints):
                        current_xy = actual_after_failed
                    time.sleep(max(0.0, float(args.gripper_settle_s)))
                    cand_det, cand_ref_uv, cand_err_u, cand_err_v, cand_dist, cand_ref_source, cand_status = observe_gripper_error(
                        gripper_capture,
                        gripper_model,
                        gripper_names,
                        calib,
                        args,
                        gripper_camera_lock,
                        overhead_uv,
                    )
                    improved_by = dist - cand_dist
                    write_gripper_trace(
                        trace_writer,
                        event="search_try_partial",
                        iter=i,
                        servo_mode="search",
                        center_u=f"{cand_det.center_uv[0]:.3f}" if cand_det is not None else "",
                        center_v=f"{cand_det.center_uv[1]:.3f}" if cand_det is not None else "",
                        ref_u=f"{cand_ref_uv[0]:.3f}" if cand_ref_uv is not None else "",
                        ref_v=f"{cand_ref_uv[1]:.3f}" if cand_ref_uv is not None else "",
                        err_u=f"{cand_err_u:.3f}",
                        err_v=f"{cand_err_v:.3f}",
                        dist_px=f"{cand_dist:.3f}" if math.isfinite(cand_dist) else "",
                        delta_pan=f"{direction[0] * step:.4f}",
                        delta_lift=f"{direction[1] * step:.4f}",
                        target_pan=f"{candidate_xy[args.map_joints[0]]:.4f}",
                        target_lift=f"{candidate_xy[args.map_joints[1]]:.4f}",
                        ref_source=cand_ref_source,
                        status=f"partial_move; {cand_status}; improved_by={improved_by:.3f}px; {move_reason}",
                    )
                    if cand_status == "ok" and improved_by >= improve_px:
                        preferred_direction = direction
                        accepted = True
                        print(
                            f"gripper_search accept partial move dir=({direction[0]:.0f},{direction[1]:.0f}) "
                            f"dist {dist:.1f}->{cand_dist:.1f}px improved_by={improved_by:.3f}px; {move_reason}",
                            flush=True,
                        )
                    else:
                        ok_rollback, rollback_reason = send_xy_hover_refine_target(robot, calib, args, base_xy, robot_lock)
                        if not ok_rollback:
                            actual_after_rollback = observed_map_xy(robot, args, robot_lock)
                            if len(actual_after_rollback) == len(args.map_joints):
                                current_xy = actual_after_rollback
                        print(
                            "gripper_search partial/incomplete move did not improve; refreshing actual XY before next observe. "
                            f"{move_reason}",
                            flush=True,
                        )
                        restart_after_failed_move = True
                    break
                time.sleep(max(0.0, float(args.gripper_settle_s)))

                cand_det, cand_ref_uv, cand_err_u, cand_err_v, cand_dist, cand_ref_source, cand_status = observe_gripper_error(
                    gripper_capture,
                    gripper_model,
                    gripper_names,
                    calib,
                    args,
                    gripper_camera_lock,
                    overhead_uv,
                )
                improved_by = dist - cand_dist
                write_gripper_trace(
                    trace_writer,
                    event="search_try",
                    iter=i,
                    servo_mode="search",
                    center_u=f"{cand_det.center_uv[0]:.3f}" if cand_det is not None else "",
                    center_v=f"{cand_det.center_uv[1]:.3f}" if cand_det is not None else "",
                    ref_u=f"{cand_ref_uv[0]:.3f}" if cand_ref_uv is not None else "",
                    ref_v=f"{cand_ref_uv[1]:.3f}" if cand_ref_uv is not None else "",
                    err_u=f"{cand_err_u:.3f}",
                    err_v=f"{cand_err_v:.3f}",
                    dist_px=f"{cand_dist:.3f}" if math.isfinite(cand_dist) else "",
                    delta_pan=f"{direction[0] * step:.4f}",
                    delta_lift=f"{direction[1] * step:.4f}",
                    target_pan=f"{candidate_xy[args.map_joints[0]]:.4f}",
                    target_lift=f"{candidate_xy[args.map_joints[1]]:.4f}",
                    ref_source=cand_ref_source,
                    status=f"{cand_status}; improved_by={improved_by:.3f}px; {move_reason}",
                )
                if cand_status == "ok" and improved_by > best_improved_by:
                    best_candidate_xy = candidate_xy
                    best_direction = direction
                    best_improved_by = improved_by
                    best_dist = cand_dist

                ok_rollback, rollback_reason = send_xy_hover_refine_target(robot, calib, args, base_xy, robot_lock)
                if not ok_rollback:
                    return f"gripper_search=stop_rollback_failed {rollback_reason}"
                time.sleep(max(0.0, float(args.gripper_settle_s)))

            if restart_after_failed_move:
                preferred_direction = None
                step_scale = max(float(args.gripper_adaptive_min_scale), step_scale * 0.5)
                continue

            if best_candidate_xy is not None and best_direction is not None and best_improved_by >= improve_px:
                ok_commit, commit_reason = send_xy_hover_refine_target(robot, calib, args, best_candidate_xy, robot_lock)
                if not ok_commit:
                    return f"gripper_search=stop_commit_failed {commit_reason}"
                time.sleep(max(0.0, float(args.gripper_settle_s)))
                commit_det, commit_ref_uv, commit_err_u, commit_err_v, commit_dist, commit_ref_source, commit_status = observe_gripper_error(
                    gripper_capture,
                    gripper_model,
                    gripper_names,
                    calib,
                    args,
                    gripper_camera_lock,
                    overhead_uv,
                )
                commit_improved_by = dist - commit_dist
                write_gripper_trace(
                    trace_writer,
                    event="search_commit",
                    iter=i,
                    servo_mode="search",
                    center_u=f"{commit_det.center_uv[0]:.3f}" if commit_det is not None else "",
                    center_v=f"{commit_det.center_uv[1]:.3f}" if commit_det is not None else "",
                    ref_u=f"{commit_ref_uv[0]:.3f}" if commit_ref_uv is not None else "",
                    ref_v=f"{commit_ref_uv[1]:.3f}" if commit_ref_uv is not None else "",
                    err_u=f"{commit_err_u:.3f}",
                    err_v=f"{commit_err_v:.3f}",
                    dist_px=f"{commit_dist:.3f}" if math.isfinite(commit_dist) else "",
                    delta_pan=f"{best_direction[0] * step:.4f}",
                    delta_lift=f"{best_direction[1] * step:.4f}",
                    target_pan=f"{best_candidate_xy[args.map_joints[0]]:.4f}",
                    target_lift=f"{best_candidate_xy[args.map_joints[1]]:.4f}",
                    ref_source=commit_ref_source,
                    status=(
                        f"{commit_status}; {commit_reason}; try_improved_by={best_improved_by:.3f}px; "
                        f"commit_improved_by={commit_improved_by:.3f}px"
                    ),
                )
                if commit_status == "ok" and commit_improved_by >= improve_px:
                    current_xy = best_candidate_xy
                    preferred_direction = best_direction
                    accepted = True
                    print(
                        f"gripper_search accept best dir=({best_direction[0]:.0f},{best_direction[1]:.0f}) "
                        f"dist {dist:.1f}->{commit_dist:.1f}px commit_improved_by={commit_improved_by:.3f}px "
                        f"try_improved_by={best_improved_by:.3f}px step={step:.3f}",
                        flush=True,
                    )
                else:
                    ok_rollback, rollback_reason = send_xy_hover_refine_target(robot, calib, args, base_xy, robot_lock)
                    if not ok_rollback:
                        return f"gripper_search=stop_commit_rollback_failed {rollback_reason}"
                    time.sleep(max(0.0, float(args.gripper_settle_s)))
                    print(
                        f"gripper_search reject commit dir=({best_direction[0]:.0f},{best_direction[1]:.0f}); "
                        f"try_improved_by={best_improved_by:.3f}px commit_improved_by={commit_improved_by:.3f}px; "
                        f"rollback {rollback_reason}",
                        flush=True,
                    )

            if accepted:
                step_scale = min(1.0, step_scale * float(args.gripper_adaptive_recover))
            else:
                preferred_direction = None
                step_scale *= 0.5
                print(f"gripper_search no improving direction; shrinking step scale to {step_scale:.2f}", flush=True)

        return (
            "INCOMPLETE: gripper_search max iterations reached before tolerance; "
            f"tolerance={float(args.gripper_center_tolerance_px):.1f}px; {status}"
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()


def run_gripper_greedy_xy_refine(
    robot,
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    xy_target: dict[str, float],
    robot_lock: threading.RLock | None = None,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
    stop_event: threading.Event | None = None,
) -> str:
    current_xy = {joint: float(xy_target[joint]) for joint in args.map_joints if joint in xy_target}
    if len(current_xy) != len(args.map_joints):
        return "gripper_search=skipped_missing_xy_target"

    trace_handle, trace_writer, _trace_path = open_gripper_trace(args)
    min_step = abs(float(args.gripper_search_min_step_deg))
    improve_px = max(0.0, float(args.gripper_search_improve_px))
    max_iters = int(args.gripper_final_iters)
    preferred_direction: tuple[float, float] | None = None
    step_scale = 1.0
    status = "gripper_greedy=no_iterations"
    i = 0
    try:
        while max_iters <= 0 or i < max_iters:
            if stop_event is not None and stop_event.is_set():
                return "STOP: gripper_greedy cancelled by user before reaching tolerance"
            i += 1
            actual_xy = observed_map_xy(robot, args, robot_lock)
            if len(actual_xy) == len(args.map_joints):
                current_xy = actual_xy
            det, ref_uv, err_u, err_v, dist, ref_source, obs_status = observe_gripper_error(
                gripper_capture,
                gripper_model,
                gripper_names,
                calib,
                args,
                gripper_camera_lock,
                overhead_uv,
            )
            if obs_status != "ok" or det is None or ref_uv is None:
                return f"gripper_greedy=stop_{obs_status} {ref_source}"
            adaptive_step = min(abs(float(args.gripper_search_step_deg)), adaptive_gripper_step_limit(dist, args))
            step = adaptive_step * step_scale
            status = (
                f"gripper_greedy iter={i} step={step:.3f} adaptive={adaptive_step:.3f} scale={step_scale:.2f} "
                f"err=({err_u:.1f},{err_v:.1f}) dist={dist:.1f}px "
                f"center=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) "
                f"ref=({ref_uv[0]:.1f},{ref_uv[1]:.1f}) {ref_source}"
            )
            print(status, flush=True)
            write_gripper_trace(
                trace_writer,
                event="greedy_observe",
                iter=i,
                servo_mode="greedy",
                center_u=f"{det.center_uv[0]:.3f}",
                center_v=f"{det.center_uv[1]:.3f}",
                ref_u=f"{ref_uv[0]:.3f}",
                ref_v=f"{ref_uv[1]:.3f}",
                err_u=f"{err_u:.3f}",
                err_v=f"{err_v:.3f}",
                dist_px=f"{dist:.3f}",
                target_pan=f"{current_xy[args.map_joints[0]]:.4f}",
                target_lift=f"{current_xy[args.map_joints[1]]:.4f}",
                ref_source=ref_source,
                status=f"step={step:.4f}; adaptive={adaptive_step:.4f}; scale={step_scale:.3f}",
            )
            if gripper_error_within_tolerance(dist, args):
                return "DONE: " + status
            if step < min_step:
                return (
                    "INCOMPLETE: gripper_greedy local minimum; no improving move above min step. "
                    f"min_step={min_step:.3f}; {status}"
                )

            accepted = False
            for direction in gripper_search_directions(args, preferred_direction):
                if stop_event is not None and stop_event.is_set():
                    return "STOP: gripper_greedy cancelled by user during direction search"

                base_xy = observed_map_xy(robot, args, robot_lock)
                if len(base_xy) != len(args.map_joints):
                    base_xy = dict(current_xy)
                base_det, base_ref_uv, base_err_u, base_err_v, base_dist, base_ref_source, base_status = observe_gripper_error(
                    gripper_capture,
                    gripper_model,
                    gripper_names,
                    calib,
                    args,
                    gripper_camera_lock,
                    overhead_uv,
                )
                if base_status != "ok" or base_det is None or base_ref_uv is None:
                    return f"gripper_greedy=stop_{base_status} {base_ref_source}"
                if gripper_error_within_tolerance(base_dist, args):
                    return "DONE: " + status

                candidate_xy = dict(base_xy)
                candidate_xy[args.map_joints[0]] = float(candidate_xy[args.map_joints[0]]) + direction[0] * step
                candidate_xy[args.map_joints[1]] = float(candidate_xy[args.map_joints[1]]) + direction[1] * step
                ok_move, move_reason = send_xy_hover_refine_target(robot, calib, args, candidate_xy, robot_lock)
                time.sleep(max(0.0, float(args.gripper_settle_s)))
                cand_det, cand_ref_uv, cand_err_u, cand_err_v, cand_dist, cand_ref_source, cand_status = observe_gripper_error(
                    gripper_capture,
                    gripper_model,
                    gripper_names,
                    calib,
                    args,
                    gripper_camera_lock,
                    overhead_uv,
                )
                improved_by = base_dist - cand_dist
                event_name = "greedy_try" if ok_move else "greedy_try_partial"
                write_gripper_trace(
                    trace_writer,
                    event=event_name,
                    iter=i,
                    servo_mode="greedy",
                    center_u=f"{cand_det.center_uv[0]:.3f}" if cand_det is not None else "",
                    center_v=f"{cand_det.center_uv[1]:.3f}" if cand_det is not None else "",
                    ref_u=f"{cand_ref_uv[0]:.3f}" if cand_ref_uv is not None else "",
                    ref_v=f"{cand_ref_uv[1]:.3f}" if cand_ref_uv is not None else "",
                    err_u=f"{cand_err_u:.3f}",
                    err_v=f"{cand_err_v:.3f}",
                    dist_px=f"{cand_dist:.3f}" if math.isfinite(cand_dist) else "",
                    delta_pan=f"{direction[0] * step:.4f}",
                    delta_lift=f"{direction[1] * step:.4f}",
                    target_pan=f"{candidate_xy[args.map_joints[0]]:.4f}",
                    target_lift=f"{candidate_xy[args.map_joints[1]]:.4f}",
                    ref_source=cand_ref_source,
                    status=f"{cand_status}; improved_by={improved_by:.3f}px; {move_reason}",
                )
                if cand_status == "ok" and improved_by >= improve_px:
                    actual_after = observed_map_xy(robot, args, robot_lock)
                    if len(actual_after) == len(args.map_joints):
                        current_xy = actual_after
                    else:
                        current_xy = candidate_xy
                    preferred_direction = direction
                    accepted = True
                    print(
                        f"gripper_greedy accept dir=({direction[0]:.0f},{direction[1]:.0f}) "
                        f"dist {base_dist:.1f}->{cand_dist:.1f}px improved_by={improved_by:.3f}px; {move_reason}",
                        flush=True,
                    )
                    break

                ok_rollback, rollback_reason = send_xy_hover_refine_target(robot, calib, args, base_xy, robot_lock)
                time.sleep(max(0.0, float(args.gripper_settle_s)))
                actual_after_rollback = observed_map_xy(robot, args, robot_lock)
                if len(actual_after_rollback) == len(args.map_joints):
                    current_xy = actual_after_rollback
                write_gripper_trace(
                    trace_writer,
                    event="greedy_rollback",
                    iter=i,
                    servo_mode="greedy",
                    delta_pan=f"{-direction[0] * step:.4f}",
                    delta_lift=f"{-direction[1] * step:.4f}",
                    target_pan=f"{base_xy[args.map_joints[0]]:.4f}",
                    target_lift=f"{base_xy[args.map_joints[1]]:.4f}",
                    status=f"try_improved_by={improved_by:.3f}px; rollback_ok={ok_rollback}; {rollback_reason}",
                )
                if not ok_rollback:
                    print(
                        "gripper_greedy rollback did not fully settle; continuing from actual observed XY",
                        flush=True,
                    )

            if accepted:
                step_scale = min(1.0, step_scale * float(args.gripper_adaptive_recover))
            else:
                preferred_direction = None
                step_scale *= 0.5
                print(f"gripper_greedy no improving direction; shrinking step scale to {step_scale:.2f}", flush=True)

        return (
            "INCOMPLETE: gripper_greedy max iterations reached before tolerance; "
            f"tolerance={float(args.gripper_center_tolerance_px):.1f}px; {status}"
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()


def run_gripper_stepwise_xy_refine(
    robot,
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    xy_target: dict[str, float],
    robot_lock: threading.RLock | None = None,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
    stop_event: threading.Event | None = None,
) -> str:
    current_xy = {joint: float(xy_target[joint]) for joint in args.map_joints if joint in xy_target}
    if len(current_xy) != len(args.map_joints):
        return "gripper_stepwise=skipped_missing_xy_target"

    trace_handle, trace_writer, _trace_path = open_gripper_trace(args)
    min_step = abs(float(args.gripper_search_min_step_deg))
    improve_px = max(0.0, float(args.gripper_search_improve_px))
    max_iters = int(args.gripper_final_iters)
    directions = gripper_search_directions(args, None)
    direction_index = 0
    preferred_direction: tuple[float, float] | None = None
    failures_at_scale = 0
    step_scale = 1.0
    status = "gripper_stepwise=no_iterations"
    i = 0
    try:
        while max_iters <= 0 or i < max_iters:
            if stop_event is not None and stop_event.is_set():
                return "STOP: gripper_stepwise cancelled by user before reaching tolerance"
            i += 1
            actual_xy = observed_map_xy(robot, args, robot_lock)
            if len(actual_xy) == len(args.map_joints):
                current_xy = actual_xy
            det, ref_uv, err_u, err_v, dist, ref_source, obs_status = observe_gripper_error(
                gripper_capture,
                gripper_model,
                gripper_names,
                calib,
                args,
                gripper_camera_lock,
                overhead_uv,
            )
            if obs_status != "ok" or det is None or ref_uv is None:
                return f"gripper_stepwise=stop_{obs_status} {ref_source}"
            adaptive_step = min(abs(float(args.gripper_search_step_deg)), adaptive_gripper_step_limit(dist, args))
            step = adaptive_step * step_scale
            status = (
                f"gripper_stepwise iter={i} step={step:.3f} adaptive={adaptive_step:.3f} scale={step_scale:.2f} "
                f"err=({err_u:.1f},{err_v:.1f}) dist={dist:.1f}px "
                f"center=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) "
                f"ref=({ref_uv[0]:.1f},{ref_uv[1]:.1f}) {ref_source}"
            )
            print(status, flush=True)
            write_gripper_trace(
                trace_writer,
                event="stepwise_observe",
                iter=i,
                servo_mode="stepwise",
                center_u=f"{det.center_uv[0]:.3f}",
                center_v=f"{det.center_uv[1]:.3f}",
                ref_u=f"{ref_uv[0]:.3f}",
                ref_v=f"{ref_uv[1]:.3f}",
                err_u=f"{err_u:.3f}",
                err_v=f"{err_v:.3f}",
                dist_px=f"{dist:.3f}",
                target_pan=f"{current_xy[args.map_joints[0]]:.4f}",
                target_lift=f"{current_xy[args.map_joints[1]]:.4f}",
                ref_source=ref_source,
                status=f"step={step:.4f}; adaptive={adaptive_step:.4f}; scale={step_scale:.3f}",
            )
            if gripper_error_within_tolerance(dist, args):
                return "DONE: " + status
            if step < min_step:
                return (
                    "INCOMPLETE: gripper_stepwise local minimum; no improving move above min step. "
                    f"min_step={min_step:.3f}; {status}"
                )

            direction = preferred_direction if preferred_direction is not None else directions[direction_index % len(directions)]
            base_xy = dict(current_xy)
            candidate_xy = dict(base_xy)
            candidate_xy[args.map_joints[0]] = float(candidate_xy[args.map_joints[0]]) + direction[0] * step
            candidate_xy[args.map_joints[1]] = float(candidate_xy[args.map_joints[1]]) + direction[1] * step

            ok_move, move_reason = send_xy_hover_refine_target(robot, calib, args, candidate_xy, robot_lock)
            time.sleep(max(0.0, float(args.gripper_settle_s)))
            cand_det, cand_ref_uv, cand_err_u, cand_err_v, cand_dist, cand_ref_source, cand_status = observe_gripper_error(
                gripper_capture,
                gripper_model,
                gripper_names,
                calib,
                args,
                gripper_camera_lock,
                overhead_uv,
            )
            improved_by = dist - cand_dist
            write_gripper_trace(
                trace_writer,
                event="stepwise_try" if ok_move else "stepwise_try_partial",
                iter=i,
                servo_mode="stepwise",
                center_u=f"{cand_det.center_uv[0]:.3f}" if cand_det is not None else "",
                center_v=f"{cand_det.center_uv[1]:.3f}" if cand_det is not None else "",
                ref_u=f"{cand_ref_uv[0]:.3f}" if cand_ref_uv is not None else "",
                ref_v=f"{cand_ref_uv[1]:.3f}" if cand_ref_uv is not None else "",
                err_u=f"{cand_err_u:.3f}",
                err_v=f"{cand_err_v:.3f}",
                dist_px=f"{cand_dist:.3f}" if math.isfinite(cand_dist) else "",
                delta_pan=f"{direction[0] * step:.4f}",
                delta_lift=f"{direction[1] * step:.4f}",
                target_pan=f"{candidate_xy[args.map_joints[0]]:.4f}",
                target_lift=f"{candidate_xy[args.map_joints[1]]:.4f}",
                ref_source=cand_ref_source,
                status=f"{cand_status}; improved_by={improved_by:.3f}px; {move_reason}",
            )

            required_improve = gripper_required_improve_px(dist, args)
            if cand_status == "ok" and improved_by >= required_improve:
                actual_after = observed_map_xy(robot, args, robot_lock)
                if len(actual_after) == len(args.map_joints):
                    current_xy = actual_after
                else:
                    current_xy = candidate_xy
                preferred_direction = direction
                failures_at_scale = 0
                step_scale = min(1.0, step_scale * float(args.gripper_adaptive_recover))
                print(
                    f"gripper_stepwise accept dir=({direction[0]:.0f},{direction[1]:.0f}) "
                    f"dist {dist:.1f}->{cand_dist:.1f}px improved_by={improved_by:.3f}px "
                    f"required={required_improve:.3f}px; {move_reason}",
                    flush=True,
                )
                continue

            ok_rollback, rollback_reason = send_xy_hover_refine_target(robot, calib, args, base_xy, robot_lock)
            time.sleep(max(0.0, float(args.gripper_settle_s)))
            rb_det, rb_ref_uv, rb_err_u, rb_err_v, rb_dist, rb_ref_source, rb_status = observe_gripper_error(
                gripper_capture,
                gripper_model,
                gripper_names,
                calib,
                args,
                gripper_camera_lock,
                overhead_uv,
            )
            actual_after_rollback = observed_map_xy(robot, args, robot_lock)
            if len(actual_after_rollback) == len(args.map_joints):
                current_xy = actual_after_rollback
            write_gripper_trace(
                trace_writer,
                event="stepwise_rollback",
                iter=i,
                servo_mode="stepwise",
                center_u=f"{rb_det.center_uv[0]:.3f}" if rb_det is not None else "",
                center_v=f"{rb_det.center_uv[1]:.3f}" if rb_det is not None else "",
                ref_u=f"{rb_ref_uv[0]:.3f}" if rb_ref_uv is not None else "",
                ref_v=f"{rb_ref_uv[1]:.3f}" if rb_ref_uv is not None else "",
                err_u=f"{rb_err_u:.3f}",
                err_v=f"{rb_err_v:.3f}",
                dist_px=f"{rb_dist:.3f}" if math.isfinite(rb_dist) else "",
                delta_pan=f"{-direction[0] * step:.4f}",
                delta_lift=f"{-direction[1] * step:.4f}",
                target_pan=f"{base_xy[args.map_joints[0]]:.4f}",
                target_lift=f"{base_xy[args.map_joints[1]]:.4f}",
                ref_source=rb_ref_source,
                status=f"try_improved_by={improved_by:.3f}px; rollback_ok={ok_rollback}; {rollback_reason}",
            )
            if not ok_rollback:
                print("gripper_stepwise rollback did not fully settle; continuing from actual observed XY", flush=True)
            if rb_status == "ok" and rb_dist > dist + max(2.0, float(args.gripper_worse_epsilon_px)):
                return (
                    "STOP: gripper_stepwise rollback failed to restore visual state; "
                    f"before={dist:.1f}px after_rollback={rb_dist:.1f}px. "
                    "Refusing to keep moving farther; use manual correction or inspect trace."
                )
            preferred_direction = None
            direction_index += 1
            failures_at_scale += 1
            if failures_at_scale >= len(directions):
                failures_at_scale = 0
                step_scale *= 0.5
                print(f"gripper_stepwise no improving direction set; shrinking step scale to {step_scale:.2f}", flush=True)

        return (
            "INCOMPLETE: gripper_stepwise max iterations reached before tolerance; "
            f"tolerance={float(args.gripper_center_tolerance_px):.1f}px; {status}"
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()


def run_gripper_learned_xy_refine(
    robot,
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    xy_target: dict[str, float],
    robot_lock: threading.RLock | None = None,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
    stop_event: threading.Event | None = None,
) -> str:
    current_xy = {joint: float(xy_target[joint]) for joint in args.map_joints if joint in xy_target}
    if len(current_xy) != len(args.map_joints):
        return "gripper_learned=skipped_missing_xy_target"
    if not matching_gripper_correction_samples(calib, args):
        return "gripper_learned=missing_samples; press B before manual correction, then C after alignment"

    trace_handle, trace_writer, _trace_path = open_gripper_trace(args)
    max_iters = int(args.gripper_final_iters)
    i = 0
    status = "gripper_learned=no_iterations"
    try:
        while max_iters <= 0 or i < max_iters:
            if stop_event is not None and stop_event.is_set():
                return "STOP: gripper_learned cancelled by user before reaching tolerance"
            i += 1
            actual_xy = observed_map_xy(robot, args, robot_lock)
            if len(actual_xy) == len(args.map_joints):
                current_xy = actual_xy
            det, ref_uv, err_u, err_v, dist, ref_source, obs_status = observe_gripper_error(
                gripper_capture,
                gripper_model,
                gripper_names,
                calib,
                args,
                gripper_camera_lock,
                overhead_uv,
            )
            if obs_status != "ok" or det is None or ref_uv is None:
                return f"gripper_learned=stop_{obs_status} {ref_source}"
            delta_xy, delta_source = predict_gripper_learned_delta(
                err_u,
                err_v,
                calib,
                args,
                current_xy=current_xy,
                overhead_uv=overhead_uv,
            )
            status = (
                f"gripper_learned iter={i} err=({err_u:.1f},{err_v:.1f}) dist={dist:.1f}px "
                f"center=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) ref=({ref_uv[0]:.1f},{ref_uv[1]:.1f}) "
                f"{ref_source}; {delta_source}"
            )
            print(status, flush=True)
            write_gripper_trace(
                trace_writer,
                event="learned_observe",
                iter=i,
                servo_mode="learned",
                center_u=f"{det.center_uv[0]:.3f}",
                center_v=f"{det.center_uv[1]:.3f}",
                ref_u=f"{ref_uv[0]:.3f}",
                ref_v=f"{ref_uv[1]:.3f}",
                err_u=f"{err_u:.3f}",
                err_v=f"{err_v:.3f}",
                dist_px=f"{dist:.3f}",
                target_pan=f"{current_xy[args.map_joints[0]]:.4f}",
                target_lift=f"{current_xy[args.map_joints[1]]:.4f}",
                ref_source=ref_source,
                status=delta_source,
            )
            if gripper_error_within_tolerance(dist, args):
                return "DONE: " + status
            if not delta_xy:
                return f"gripper_learned=stop_no_delta {delta_source}"

            previous_xy = dict(current_xy)
            accepted = False
            best_improve = -math.inf
            last_attempt_status = ""
            accept_improve = max(0.0, float(args.gripper_learned_accept_improve_px))
            min_improve = max(accept_improve, float(args.gripper_learned_min_improve_px))
            fallback_after_worse = max(0.0, float(args.gripper_learned_fallback_after_worse_px))
            rollback_worse_px = max(0.0, float(args.gripper_learned_rollback_worse_px))
            force_search = False
            for attempt_index, scale in enumerate((1.0, 0.5, 0.25), start=1):
                if stop_event is not None and stop_event.is_set():
                    return "STOP: gripper_learned cancelled by user during backtracking"
                next_xy = dict(previous_xy)
                scaled_delta = {joint: float(delta) * float(scale) for joint, delta in delta_xy.items()}
                for joint, delta in scaled_delta.items():
                    next_xy[joint] = float(next_xy[joint]) + float(delta)
                ok_move, move_reason = send_xy_hover_refine_target(robot, calib, args, next_xy, robot_lock)
                if not ok_move:
                    time.sleep(max(0.0, float(args.gripper_settle_s)))
                    verify_det, verify_ref_uv, verify_err_u, verify_err_v, verify_dist, verify_ref_source, verify_status = observe_gripper_error(
                        gripper_capture,
                        gripper_model,
                        gripper_names,
                        calib,
                        args,
                        gripper_camera_lock,
                        overhead_uv,
                    )
                    improved_by = dist - verify_dist
                    best_improve = max(best_improve, improved_by)
                    actual_after_failed = observed_map_xy(robot, args, robot_lock)
                    if len(actual_after_failed) == len(args.map_joints):
                        current_xy = actual_after_failed
                    accepted = verify_status == "ok" and improved_by >= accept_improve
                    last_attempt_status = (
                        f"scale={scale:.2f} partial_move {verify_status}; improved_by={improved_by:.3f}px; {move_reason}"
                    )
                    write_gripper_trace(
                        trace_writer,
                        event="learned_verify_partial",
                        iter=i,
                        servo_mode="learned",
                        center_u=f"{verify_det.center_uv[0]:.3f}" if verify_det is not None else "",
                        center_v=f"{verify_det.center_uv[1]:.3f}" if verify_det is not None else "",
                        ref_u=f"{verify_ref_uv[0]:.3f}" if verify_ref_uv is not None else "",
                        ref_v=f"{verify_ref_uv[1]:.3f}" if verify_ref_uv is not None else "",
                        err_u=f"{verify_err_u:.3f}",
                        err_v=f"{verify_err_v:.3f}",
                        dist_px=f"{verify_dist:.3f}" if math.isfinite(verify_dist) else "",
                        delta_pan=f"{scaled_delta[args.map_joints[0]]:.4f}",
                        delta_lift=f"{scaled_delta[args.map_joints[1]]:.4f}",
                        target_pan=f"{next_xy[args.map_joints[0]]:.4f}",
                        target_lift=f"{next_xy[args.map_joints[1]]:.4f}",
                        ref_source=verify_ref_source,
                        status=last_attempt_status,
                    )
                    print(f"gripper_learned attempt {attempt_index}: {last_attempt_status}", flush=True)
                    if accepted:
                        print(
                            f"gripper_learned accepted partial/incomplete move improved_by={improved_by:.2f}px",
                            flush=True,
                        )
                        break
                    force_search = True
                    break
                current_xy = next_xy
                print(
                    f"gripper_learned move scale={scale:.2f} delta_pan={scaled_delta[args.map_joints[0]]:.2f} "
                    f"delta_lift={scaled_delta[args.map_joints[1]]:.2f}; xy=({current_xy[args.map_joints[0]]:.2f},"
                    f"{current_xy[args.map_joints[1]]:.2f}); {move_reason}",
                    flush=True,
                )
                time.sleep(max(0.0, float(args.gripper_settle_s)))

                verify_det, verify_ref_uv, verify_err_u, verify_err_v, verify_dist, verify_ref_source, verify_status = observe_gripper_error_stable(
                    gripper_capture,
                    gripper_model,
                    gripper_names,
                    calib,
                    args,
                    gripper_camera_lock,
                    overhead_uv,
                )
                improved_by = dist - verify_dist
                best_improve = max(best_improve, improved_by)
                accepted = verify_status == "ok" and improved_by >= accept_improve
                enough = verify_status == "ok" and improved_by >= min_improve
                attempt_status = (
                    f"{verify_status}; scale={scale:.2f}; improved_by={improved_by:.3f}px; "
                    f"{'accepted' if accepted else 'rollback'}"
                )
                write_gripper_trace(
                    trace_writer,
                    event="learned_verify",
                    iter=i,
                    servo_mode="learned",
                    center_u=f"{verify_det.center_uv[0]:.3f}" if verify_det is not None else "",
                    center_v=f"{verify_det.center_uv[1]:.3f}" if verify_det is not None else "",
                    ref_u=f"{verify_ref_uv[0]:.3f}" if verify_ref_uv is not None else "",
                    ref_v=f"{verify_ref_uv[1]:.3f}" if verify_ref_uv is not None else "",
                    err_u=f"{verify_err_u:.3f}",
                    err_v=f"{verify_err_v:.3f}",
                    dist_px=f"{verify_dist:.3f}" if math.isfinite(verify_dist) else "",
                    delta_pan=f"{scaled_delta[args.map_joints[0]]:.4f}",
                    delta_lift=f"{scaled_delta[args.map_joints[1]]:.4f}",
                    target_pan=f"{current_xy[args.map_joints[0]]:.4f}",
                    target_lift=f"{current_xy[args.map_joints[1]]:.4f}",
                    ref_source=verify_ref_source,
                    status=attempt_status,
                )
                last_attempt_status = attempt_status
                if attempt_index == 1 and verify_status == "ok" and improved_by < -fallback_after_worse:
                    force_search = True
                if accepted:
                    if not enough:
                        print(
                            f"gripper_learned accepted small improvement {improved_by:.2f}px; continuing instead of stopping",
                            flush=True,
                        )
                    break
                if verify_status == "ok" and improved_by >= -rollback_worse_px:
                    current_xy = next_xy
                    force_search = True
                    last_attempt_status = (
                        f"{attempt_status}; kept near-neutral attempt without rollback "
                        f"(rollback_worse_px={rollback_worse_px:.2f})"
                    )
                    print(
                        "gripper_learned near-neutral attempt; keeping pose and switching to search instead of rollback. "
                        f"improved_by={improved_by:.3f}px threshold=-{rollback_worse_px:.2f}px",
                        flush=True,
                    )
                    break
                ok_rollback, rollback_reason = send_xy_hover_refine_target(
                    robot,
                    calib,
                    args,
                    previous_xy,
                    robot_lock,
                    outer_correction_enabled=False,
                )
                time.sleep(max(0.0, float(args.gripper_settle_s)))
                rb_det, rb_ref_uv, rb_err_u, rb_err_v, rb_dist, rb_ref_source, rb_status = observe_gripper_error(
                    gripper_capture,
                    gripper_model,
                    gripper_names,
                    calib,
                    args,
                    gripper_camera_lock,
                    overhead_uv,
                )
                actual_after_rollback = observed_map_xy(robot, args, robot_lock)
                if len(actual_after_rollback) == len(args.map_joints):
                    current_xy = actual_after_rollback
                elif ok_rollback:
                    current_xy = previous_xy
                rollback_visual_ok = rb_status == "ok" and rb_dist <= dist + max(2.0, rollback_worse_px)
                last_attempt_status = (
                    f"{attempt_status}; rollback {'visual_restored' if rollback_visual_ok else 'visual_not_restored'} "
                    f"rb_dist={rb_dist:.1f}px before={dist:.1f}px; {rollback_reason}"
                )
                write_gripper_trace(
                    trace_writer,
                    event="learned_rollback",
                    iter=i,
                    servo_mode="learned",
                    center_u=f"{rb_det.center_uv[0]:.3f}" if rb_det is not None else "",
                    center_v=f"{rb_det.center_uv[1]:.3f}" if rb_det is not None else "",
                    ref_u=f"{rb_ref_uv[0]:.3f}" if rb_ref_uv is not None else "",
                    ref_v=f"{rb_ref_uv[1]:.3f}" if rb_ref_uv is not None else "",
                    err_u=f"{rb_err_u:.3f}",
                    err_v=f"{rb_err_v:.3f}",
                    dist_px=f"{rb_dist:.3f}" if math.isfinite(rb_dist) else "",
                    target_pan=f"{previous_xy[args.map_joints[0]]:.4f}",
                    target_lift=f"{previous_xy[args.map_joints[1]]:.4f}",
                    ref_source=rb_ref_source,
                    status=last_attempt_status,
                )
                print(
                    f"gripper_learned rollback scale={scale:.2f}; "
                    f"{'visual restored' if rollback_visual_ok else 'visual not restored'}; {rollback_reason}",
                    flush=True,
                )
                force_search = True
                break

            if not accepted:
                fallback_status = (
                    "gripper_learned prediction did not improve; "
                    f"best_improved_by={best_improve:.1f}px; {last_attempt_status}; "
                    "stopped for manual correction instead of trying blind search"
                )
                print("WARNING: " + fallback_status, flush=True)
                return "STOP: " + fallback_status

        return (
            "INCOMPLETE: gripper_learned max iterations reached before tolerance; "
            f"tolerance={float(args.gripper_center_tolerance_px):.1f}px; {status}"
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()


def run_gripper_learned_full_pose_refine(
    robot,
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    xy_target: dict[str, float],
    robot_lock: threading.RLock | None = None,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
    stop_event: threading.Event | None = None,
) -> str:
    param_status = ensure_gripper_learned_param_model(calib, args)
    has_param_model = (
        isinstance(calib.gripper_learned_param_model, dict)
        and isinstance(calib.gripper_learned_param_model.get("gain"), dict)
        and int(calib.gripper_learned_param_model.get("sample_count") or 0) > 0
    )
    if not has_param_model and not matching_gripper_correction_samples(calib, args):
        return "gripper_learned_full=missing_params; press B before manual correction, then C after alignment"

    trace_handle, trace_writer, _trace_path = open_gripper_trace(args)
    max_iters = int(args.gripper_final_iters)
    status = "gripper_learned_full=no_iterations"
    i = 0
    last_accepted_delta: dict[str, float] | None = None
    refine_outer_correction = bool(getattr(args, "gripper_learned_refine_outer_correction", False))
    learned_settle_threshold = max(
        0.0,
        float(getattr(args, "gripper_learned_settle_threshold_deg", 0.35)),
    )
    if param_status:
        print(param_status, flush=True)

    def send_learned_full_target_with_clearance(
        start_pose: dict[str, float],
        target_pose: dict[str, float],
        label: str,
    ) -> dict[str, Any]:
        if not bool(getattr(args, "gripper_learned_clearance", True)):
            result = send_pose_with_encoder_outer_correction(
                robot,
                target_pose,
                args,
                robot_lock,
                stop_event=stop_event,
                label=label,
                speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
                min_steps=int(args.gripper_refine_min_move_steps),
                wait_timeout_s=float(args.gripper_refine_wait_timeout_s),
                settle_threshold=learned_settle_threshold,
                outer_correction_enabled=refine_outer_correction,
            )
            result["clearance_status"] = "clearance disabled"
            return result

        clearance_pose, clearance_plan = learned_clearance_pose(start_pose, calib.safe_pose, args)
        clearance_result = send_pose_with_encoder_outer_correction(
            robot,
            clearance_pose,
            args,
            robot_lock,
            stop_event=stop_event,
            label=label + "_CLEARANCE",
            speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
            min_steps=int(args.gripper_refine_min_move_steps),
            wait_timeout_s=float(args.gripper_refine_wait_timeout_s),
            settle_threshold=learned_settle_threshold,
            outer_correction_enabled=refine_outer_correction,
        )
        if clearance_result.get("cancelled"):
            clearance_result["clearance_status"] = clearance_plan
            return clearance_result

        actual_clearance = locked_observation(robot, robot_lock)
        actual_offset = {
            joint: float(actual_clearance[joint]) - float(start_pose[joint])
            for joint in args.pose_joints
            if joint in actual_clearance and joint in start_pose
        }
        lifted_target = {
            joint: float(target_pose[joint]) + float(actual_offset.get(joint, 0.0))
            for joint in target_pose
        }
        offset_text = " ".join(
            f"{joint.split('.')[0]}={float(actual_offset.get(joint, 0.0)):.2f}"
            for joint in args.pose_joints
            if joint in actual_offset
        )
        lifted_result = send_pose_with_encoder_outer_correction(
            robot,
            lifted_target,
            args,
            robot_lock,
            stop_event=stop_event,
            label=label + "_LIFTED",
            speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
            min_steps=int(args.gripper_refine_min_move_steps),
            wait_timeout_s=float(args.gripper_refine_wait_timeout_s),
            settle_threshold=learned_settle_threshold,
            outer_correction_enabled=refine_outer_correction,
        )
        if lifted_result.get("cancelled"):
            lifted_result["clearance_status"] = f"{clearance_plan}; actual_offset {offset_text}; cancelled during lifted move"
            return lifted_result

        descend_result = send_pose_with_encoder_outer_correction(
            robot,
            target_pose,
            args,
            robot_lock,
            stop_event=stop_event,
            label=label + "_DESCEND",
            speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
            min_steps=int(args.gripper_refine_min_move_steps),
            wait_timeout_s=float(args.gripper_refine_wait_timeout_s),
            settle_threshold=learned_settle_threshold,
            outer_correction_enabled=refine_outer_correction,
        )
        descend_result["clearance_status"] = (
            f"{clearance_plan}; actual_offset {offset_text}; "
            f"clearance={clearance_result.get('wait_status', '')}; "
            f"lifted={lifted_result.get('wait_status', '')}"
        )
        return descend_result

    try:
        while max_iters <= 0 or i < max_iters:
            if stop_event is not None and stop_event.is_set():
                return "STOP: gripper_learned_full cancelled by user before reaching tolerance"
            i += 1
            current_pose = locked_observation(robot, robot_lock)
            det, ref_uv, err_u, err_v, dist, ref_source, obs_status = observe_gripper_error_stable(
                gripper_capture,
                gripper_model,
                gripper_names,
                calib,
                args,
                gripper_camera_lock,
                overhead_uv,
            )
            if obs_status != "ok" or det is None or ref_uv is None:
                return f"gripper_learned_full=stop_{obs_status} {ref_source}"
            delta_pose, delta_source = predict_gripper_learned_full_pose_delta(
                err_u,
                err_v,
                calib,
                args,
                current_pose=current_pose,
                overhead_uv=overhead_uv,
            )
            status = (
                f"gripper_learned_full iter={i} err=({err_u:.1f},{err_v:.1f}) dist={dist:.1f}px "
                f"center=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) ref=({ref_uv[0]:.1f},{ref_uv[1]:.1f}) "
                f"{ref_source}; {delta_source}"
            )
            print(status, flush=True)
            write_gripper_trace(
                trace_writer,
                event="learned_full_observe",
                iter=i,
                servo_mode="learned_full_pose",
                center_u=f"{det.center_uv[0]:.3f}",
                center_v=f"{det.center_uv[1]:.3f}",
                ref_u=f"{ref_uv[0]:.3f}",
                ref_v=f"{ref_uv[1]:.3f}",
                err_u=f"{err_u:.3f}",
                err_v=f"{err_v:.3f}",
                dist_px=f"{dist:.3f}",
                actual_pan=f"{current_pose.get(args.map_joints[0], float('nan')):.4f}",
                actual_lift=f"{current_pose.get(args.map_joints[1], float('nan')):.4f}",
                actual_elbow=f"{current_pose.get('elbow_flex.pos', float('nan')):.4f}",
                actual_wrist_flex=f"{current_pose.get('wrist_flex.pos', float('nan')):.4f}",
                actual_wrist_roll=f"{current_pose.get('wrist_roll.pos', float('nan')):.4f}",
                ref_source=ref_source,
                status=delta_source,
            )
            if gripper_error_within_tolerance(dist, args):
                return "DONE: " + status
            near_done_px = max(0.0, float(getattr(args, "gripper_learned_near_done_px", 0.0)))
            if near_done_px > 0.0 and dist <= near_done_px:
                return (
                    "DONE: "
                    + status
                    + f"; near_done dist={dist:.1f}px <= {near_done_px:.1f}px, not risking extra micro-move"
                )
            search_mode = bool(getattr(args, "gripper_learned_autonomous_search", False))
            if not delta_pose and not search_mode:
                return f"gripper_learned_full=stop_no_delta {delta_source}"

            previous_pose = {
                joint: float(current_pose[joint])
                for joint in args.pose_joints
                if joint in current_pose
            }
            if len(previous_pose) != len(args.pose_joints):
                return "gripper_learned_full=stop_missing_current_pose_joints"
            accept_improve = max(0.0, float(args.gripper_learned_accept_improve_px))
            min_improve = max(accept_improve, float(args.gripper_learned_min_improve_px))
            rollback_worse_px = max(0.0, float(args.gripper_learned_rollback_worse_px))
            neutral_band_px = max(0.0, float(getattr(args, "gripper_learned_neutral_band_px", 1.5)))
            reverse_worse_px = max(neutral_band_px, float(getattr(args, "gripper_learned_reverse_worse_px", 2.5)))
            rollback_outer_correction = bool(getattr(args, "gripper_learned_rollback_outer_correction", True))
            best_improve = -math.inf
            last_attempt_status = ""
            accepted = False
            if search_mode:
                candidates = learned_full_pose_search_candidates(delta_pose, args, preferred_delta=last_accepted_delta)
                candidate_source = "autonomous_search"
            else:
                candidates = [
                    (
                        f"scale={scale:.2f}",
                        {joint: float(delta_pose[joint]) * float(scale) for joint in args.pose_joints},
                    )
                    for scale in (1.0, 0.5, 0.25)
                ]
                candidate_source = "backtracking"
            if not candidates:
                return f"gripper_learned_full=stop_no_candidate {delta_source}"
            for attempt_index, (candidate_label, scaled_delta) in enumerate(candidates, start=1):
                if stop_event is not None and stop_event.is_set():
                    return "STOP: gripper_learned_full cancelled by user during backtracking"
                target_pose = dict(previous_pose)
                for joint, delta in scaled_delta.items():
                    target_pose[joint] = float(target_pose[joint]) + float(delta)
                if calib.pose_limits:
                    ok_limits, limit_reason = within_limits(target_pose, calib.pose_limits)
                    if not ok_limits:
                        last_attempt_status = f"{candidate_label} target outside pose limits: {limit_reason}"
                        continue
                move_result = send_learned_full_target_with_clearance(
                    previous_pose,
                    target_pose,
                    "GRIPPER_FULL_SEARCH" if search_mode else "GRIPPER_FULL",
                )
                if move_result.get("cancelled"):
                    return f"STOP: gripper_learned_full cancelled during {move_result.get('stage')}"
                correction_notes = list(move_result.get("correction_notes") or [])
                correction_text = ""
                if correction_notes:
                    correction_text = "; outer_correction " + " | ".join(correction_notes)
                clearance_text = str(move_result.get("clearance_status") or "")
                if clearance_text:
                    clearance_text = "; " + clearance_text
                move_reason = (
                    f"GRIPPER_FULL max_joint_error={float(move_result.get('max_error', float('inf'))):.2f}; "
                    f"raw_max_joint_error={float(move_result.get('raw_error', float('inf'))):.2f}"
                    f"{correction_text}; {move_result.get('wait_status', 'encoder wait failed')}"
                    f"{clearance_text}"
                )
                actual_pose_after_move = locked_observation(robot, robot_lock)
                actual_delta_pose = {
                    joint: float(actual_pose_after_move[joint]) - float(previous_pose[joint])
                    for joint in args.pose_joints
                    if joint in actual_pose_after_move and joint in previous_pose
                }
                actual_move_max = full_pose_delta_max_abs(actual_delta_pose)
                commanded_move_max = full_pose_delta_max_abs(scaled_delta)
                min_actual_motion = max(0.0, float(getattr(args, "gripper_min_actual_motion_deg", 0.12)))
                actual_motion_ok = commanded_move_max <= 1e-6 or actual_move_max >= min_actual_motion
                actual_delta_text = " ".join(
                    f"{joint.split('.')[0]}={float(actual_delta_pose.get(joint, 0.0)):+.3f}"
                    for joint in args.pose_joints
                )
                actual_motion_ok, actual_motion_status = full_pose_actual_motion_quality(
                    scaled_delta,
                    actual_delta_pose,
                    args,
                )
                time.sleep(max(0.0, float(args.gripper_settle_s)))
                verify_det, verify_ref_uv, verify_err_u, verify_err_v, verify_dist, verify_ref_source, verify_status = observe_gripper_error_stable(
                    gripper_capture,
                    gripper_model,
                    gripper_names,
                    calib,
                    args,
                    gripper_camera_lock,
                    overhead_uv,
                )
                distance_delta = verify_dist - dist
                meaningful_accept_px = max(accept_improve, neutral_band_px)
                visual_outcome, improved_by = classify_gripper_visual_step(
                    dist,
                    verify_dist,
                    meaningful_accept_px,
                    reverse_worse_px,
                )
                best_improve = max(best_improve, improved_by)
                accepted = verify_status == "ok" and visual_outcome == "closer"
                enough = verify_status == "ok" and improved_by >= min_improve
                clearly_worse = verify_status == "ok" and visual_outcome == "farther"
                neutral_or_jitter = verify_status == "ok" and visual_outcome == "neutral"
                result_label = (
                    "accepted_closer"
                    if accepted
                    else (
                        "worse_try_reverse"
                        if clearly_worse
                        else ("invalid_observation" if visual_outcome == "invalid" else "neutral_try_other_side")
                    )
                )
                attempt_status = (
                    f"{verify_status}; {candidate_source} candidate={candidate_label}; "
                    f"dist {dist:.2f}->{verify_dist:.2f}px delta={distance_delta:+.3f}px; "
                    f"improved_by={improved_by:.3f}px need>={meaningful_accept_px:.2f}px; "
                    f"{result_label}; actual_max={actual_move_max:.3f}deg need>={min_actual_motion:.3f}deg "
                    f"actual_delta {actual_delta_text}; {actual_motion_status}; {move_reason}"
                )
                write_gripper_trace(
                    trace_writer,
                    event="learned_full_verify",
                    iter=i,
                    servo_mode="learned_full_pose",
                    center_u=f"{verify_det.center_uv[0]:.3f}" if verify_det is not None else "",
                    center_v=f"{verify_det.center_uv[1]:.3f}" if verify_det is not None else "",
                    ref_u=f"{verify_ref_uv[0]:.3f}" if verify_ref_uv is not None else "",
                    ref_v=f"{verify_ref_uv[1]:.3f}" if verify_ref_uv is not None else "",
                    err_u=f"{verify_err_u:.3f}",
                    err_v=f"{verify_err_v:.3f}",
                    dist_px=f"{verify_dist:.3f}" if math.isfinite(verify_dist) else "",
                    delta_pan=f"{scaled_delta.get(args.map_joints[0], 0.0):.4f}",
                    delta_lift=f"{scaled_delta.get(args.map_joints[1], 0.0):.4f}",
                    actual_delta_pan=f"{actual_delta_pose.get(args.map_joints[0], 0.0):.4f}",
                    actual_delta_lift=f"{actual_delta_pose.get(args.map_joints[1], 0.0):.4f}",
                    target_pan=f"{target_pose.get(args.map_joints[0], float('nan')):.4f}",
                    target_lift=f"{target_pose.get(args.map_joints[1], float('nan')):.4f}",
                    actual_pan=f"{actual_pose_after_move.get(args.map_joints[0], float('nan')):.4f}",
                    actual_lift=f"{actual_pose_after_move.get(args.map_joints[1], float('nan')):.4f}",
                    actual_elbow=f"{actual_pose_after_move.get('elbow_flex.pos', float('nan')):.4f}",
                    actual_wrist_flex=f"{actual_pose_after_move.get('wrist_flex.pos', float('nan')):.4f}",
                    actual_wrist_roll=f"{actual_pose_after_move.get('wrist_roll.pos', float('nan')):.4f}",
                    actual_delta_elbow=f"{actual_delta_pose.get('elbow_flex.pos', 0.0):.4f}",
                    actual_delta_wrist_flex=f"{actual_delta_pose.get('wrist_flex.pos', 0.0):.4f}",
                    actual_delta_wrist_roll=f"{actual_delta_pose.get('wrist_roll.pos', 0.0):.4f}",
                    ref_source=verify_ref_source,
                    status=attempt_status,
                )
                print(
                    "gripper_learned_full move "
                    f"candidate={candidate_label} delta="
                    + " ".join(f"{joint.split('.')[0]}={scaled_delta[joint]:.2f}" for joint in args.pose_joints)
                    + f"; {attempt_status}",
                    flush=True,
                )
                last_attempt_status = attempt_status
                if accepted:
                    accepted_direction = actual_delta_pose if actual_move_max >= min_actual_motion else scaled_delta
                    last_accepted_delta = {
                        joint: float(accepted_direction.get(joint, 0.0))
                        for joint in args.pose_joints
                    }
                    if gripper_error_within_tolerance(verify_dist, args):
                        ok_param, param_update_status = update_gripper_learned_param_model(
                            calib,
                            args,
                            error_uv=(float(err_u), float(err_v)),
                            delta_pose=last_accepted_delta,
                            before_dist_px=float(dist),
                            after_dist_px=float(verify_dist),
                            source=f"auto_done_{candidate_label}",
                            overhead_uv=overhead_uv,
                        )
                        done_status = (
                            f"gripper_learned_full accepted and aligned dist {dist:.1f}->{verify_dist:.1f}px "
                            f"improved_by={improved_by:.2f}px; {attempt_status}; {param_update_status}"
                        )
                        print(done_status, flush=True)
                        return "DONE: " + done_status
                    if candidate_label == "continue_last":
                        print(
                            "gripper_learned_full greedy: last direction still improved; continuing this direction next",
                            flush=True,
                        )
                    elif candidate_label == "reverse_last":
                        print(
                            "gripper_learned_full greedy: reverse of last direction improved; using reverse direction next",
                            flush=True,
                        )
                    if not enough:
                        print(
                            f"gripper_learned_full accepted small improvement {improved_by:.2f}px; continuing",
                            flush=True,
                        )
                    if bool(getattr(args, "gripper_learned_stop_after_accepted_move", True)) and not search_mode:
                        stop_status = (
                            f"gripper_learned_full accepted one clearance correction dist {dist:.1f}->{verify_dist:.1f}px "
                            f"improved_by={improved_by:.2f}px; stopped before a second learned move"
                        )
                        print(stop_status, flush=True)
                        return "INCOMPLETE: " + stop_status
                    break

                has_next_candidate = attempt_index < len(candidates)
                if search_mode and verify_status == "ok":
                    if clearly_worse:
                        reverse_source_delta = (
                            actual_delta_pose
                            if actual_move_max >= min_actual_motion
                            else scaled_delta
                        )
                        reverse_delta = {
                            joint: -float(reverse_source_delta.get(joint, 0.0))
                            for joint in args.pose_joints
                        }
                        reverse_delta = clamp_learned_full_pose_delta(reverse_delta, args)
                        reverse_label = (
                            "comp_reverse_actual"
                            if actual_move_max >= min_actual_motion
                            else "comp_reverse_command"
                        )
                        reverse_target = dict(previous_pose)
                        for joint, delta in reverse_delta.items():
                            reverse_target[joint] = float(reverse_target[joint]) + float(delta)
                        limits_ok = True
                        limit_reason = ""
                        if calib.pose_limits:
                            limits_ok, limit_reason = within_limits(reverse_target, calib.pose_limits)
                        if limits_ok and full_pose_delta_max_abs(reverse_delta) > 1e-6:
                            print(
                                "gripper_learned_full VISUAL_DECISION=farther; "
                                f"testing compensation-aware reverse from {reverse_label}.",
                                flush=True,
                            )
                            reverse_move = send_learned_full_target_with_clearance(
                                actual_pose_after_move,
                                reverse_target,
                                "GRIPPER_FULL_COMP_REVERSE",
                            )
                            if reverse_move.get("cancelled"):
                                return f"STOP: gripper_learned_full cancelled during {reverse_move.get('stage')}"
                            time.sleep(max(0.0, float(args.gripper_settle_s)))
                            reverse_pose_after_move = locked_observation(robot, robot_lock)
                            reverse_actual_delta = {
                                joint: float(reverse_pose_after_move[joint]) - float(previous_pose[joint])
                                for joint in args.pose_joints
                                if joint in reverse_pose_after_move and joint in previous_pose
                            }
                            reverse_det, reverse_ref_uv, reverse_err_u, reverse_err_v, reverse_dist, reverse_ref_source, reverse_status = observe_gripper_error_stable(
                                gripper_capture,
                                gripper_model,
                                gripper_names,
                                calib,
                                args,
                                gripper_camera_lock,
                                overhead_uv,
                            )
                            reverse_outcome, reverse_improved_by = classify_gripper_visual_step(
                                dist,
                                reverse_dist,
                                meaningful_accept_px,
                                reverse_worse_px,
                            )
                            best_improve = max(best_improve, reverse_improved_by)
                            reverse_actual_text = " ".join(
                                f"{joint.split('.')[0]}={float(reverse_actual_delta.get(joint, 0.0)):+.3f}"
                                for joint in args.pose_joints
                            )
                            reverse_move_reason = (
                                f"GRIPPER_FULL_COMP_REVERSE max_joint_error="
                                f"{float(reverse_move.get('max_error', float('inf'))):.2f}; "
                                f"raw_max_joint_error={float(reverse_move.get('raw_error', float('inf'))):.2f}; "
                                f"{reverse_move.get('wait_status', 'encoder wait failed')}; "
                                f"{reverse_move.get('clearance_status', '')}"
                            )
                            reverse_attempt_status = (
                                f"{reverse_status}; autonomous_search candidate={reverse_label}; "
                                f"dist {dist:.2f}->{reverse_dist:.2f}px "
                                f"improved_by={reverse_improved_by:.3f}px need>={meaningful_accept_px:.2f}px; "
                                f"VISUAL_DECISION={reverse_outcome}; actual_delta {reverse_actual_text}; "
                                f"{reverse_move_reason}"
                            )
                            write_gripper_trace(
                                trace_writer,
                                event="learned_full_verify",
                                iter=i,
                                servo_mode="learned_full_pose",
                                center_u=f"{reverse_det.center_uv[0]:.3f}" if reverse_det is not None else "",
                                center_v=f"{reverse_det.center_uv[1]:.3f}" if reverse_det is not None else "",
                                ref_u=f"{reverse_ref_uv[0]:.3f}" if reverse_ref_uv is not None else "",
                                ref_v=f"{reverse_ref_uv[1]:.3f}" if reverse_ref_uv is not None else "",
                                err_u=f"{reverse_err_u:.3f}",
                                err_v=f"{reverse_err_v:.3f}",
                                dist_px=f"{reverse_dist:.3f}" if math.isfinite(reverse_dist) else "",
                                delta_pan=f"{reverse_delta.get(args.map_joints[0], 0.0):.4f}",
                                delta_lift=f"{reverse_delta.get(args.map_joints[1], 0.0):.4f}",
                                actual_delta_pan=f"{reverse_actual_delta.get(args.map_joints[0], 0.0):.4f}",
                                actual_delta_lift=f"{reverse_actual_delta.get(args.map_joints[1], 0.0):.4f}",
                                target_pan=f"{reverse_target.get(args.map_joints[0], float('nan')):.4f}",
                                target_lift=f"{reverse_target.get(args.map_joints[1], float('nan')):.4f}",
                                actual_pan=f"{reverse_pose_after_move.get(args.map_joints[0], float('nan')):.4f}",
                                actual_lift=f"{reverse_pose_after_move.get(args.map_joints[1], float('nan')):.4f}",
                                actual_elbow=f"{reverse_pose_after_move.get('elbow_flex.pos', float('nan')):.4f}",
                                actual_wrist_flex=f"{reverse_pose_after_move.get('wrist_flex.pos', float('nan')):.4f}",
                                actual_wrist_roll=f"{reverse_pose_after_move.get('wrist_roll.pos', float('nan')):.4f}",
                                actual_delta_elbow=f"{reverse_actual_delta.get('elbow_flex.pos', 0.0):.4f}",
                                actual_delta_wrist_flex=f"{reverse_actual_delta.get('wrist_flex.pos', 0.0):.4f}",
                                actual_delta_wrist_roll=f"{reverse_actual_delta.get('wrist_roll.pos', 0.0):.4f}",
                                ref_source=reverse_ref_source,
                                status=reverse_attempt_status,
                            )
                            print(
                                "gripper_learned_full compensation-aware reverse "
                                f"delta="
                                + " ".join(
                                    f"{joint.split('.')[0]}={reverse_delta[joint]:.2f}"
                                    for joint in args.pose_joints
                                )
                                + f"; {reverse_attempt_status}",
                                flush=True,
                            )
                            if reverse_status == "ok" and reverse_outcome == "closer":
                                accepted = True
                                reverse_move_max = full_pose_delta_max_abs(reverse_actual_delta)
                                accepted_direction = (
                                    reverse_actual_delta
                                    if reverse_move_max >= min_actual_motion
                                    else reverse_delta
                                )
                                last_accepted_delta = {
                                    joint: float(accepted_direction.get(joint, 0.0))
                                    for joint in args.pose_joints
                                }
                                last_attempt_status = reverse_attempt_status
                                if gripper_error_within_tolerance(reverse_dist, args):
                                    ok_param, param_update_status = update_gripper_learned_param_model(
                                        calib,
                                        args,
                                        error_uv=(float(err_u), float(err_v)),
                                        delta_pose=last_accepted_delta,
                                        before_dist_px=float(dist),
                                        after_dist_px=float(reverse_dist),
                                        source=f"auto_done_{reverse_label}",
                                        overhead_uv=overhead_uv,
                                    )
                                    done_status = (
                                        f"gripper_learned_full reverse accepted and aligned "
                                        f"dist {dist:.1f}->{reverse_dist:.1f}px "
                                        f"improved_by={reverse_improved_by:.2f}px; {reverse_attempt_status}; "
                                        f"{param_update_status}"
                                    )
                                    print(done_status, flush=True)
                                    return "DONE: " + done_status
                                break
                            last_attempt_status = reverse_attempt_status
                        else:
                            last_attempt_status = (
                                f"{attempt_status}; compensation reverse skipped "
                                f"limits_ok={limits_ok} {limit_reason}"
                            )
                        if accepted:
                            break
                        if has_next_candidate:
                            continue
                    elif has_next_candidate:
                        print(
                            "gripper_learned_full VISUAL_DECISION=neutral; "
                            "testing the next candidate from the same original visual baseline",
                            flush=True,
                        )
                        continue
                    print(
                        "gripper_learned_full visual candidates exhausted; "
                        "not issuing an encoder-only rollback that may move the image farther",
                        flush=True,
                    )
                    break

                if neutral_or_jitter:
                    rollback_result = send_pose_with_encoder_outer_correction(
                        robot,
                        previous_pose,
                        args,
                        robot_lock,
                        stop_event=stop_event,
                        label="GRIPPER_FULL_NEUTRAL_ROLLBACK",
                        speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
                        min_steps=int(args.gripper_refine_min_move_steps),
                        wait_timeout_s=float(args.gripper_refine_wait_timeout_s),
                        settle_threshold=learned_settle_threshold,
                        outer_correction_enabled=rollback_outer_correction,
                    )
                    time.sleep(max(0.0, float(args.gripper_settle_s)))
                    rb_det, rb_ref_uv, rb_err_u, rb_err_v, rb_dist, rb_ref_source, rb_status = observe_gripper_error_stable(
                        gripper_capture,
                        gripper_model,
                        gripper_names,
                        calib,
                        args,
                        gripper_camera_lock,
                        overhead_uv,
                    )
                    last_attempt_status = (
                        f"{attempt_status}; neutral band {neutral_band_px:.1f}px: not reversing; "
                        f"rollback rb_dist={rb_dist:.1f}px before={dist:.1f}px"
                    )
                    write_gripper_trace(
                        trace_writer,
                        event="learned_full_neutral_rollback",
                        iter=i,
                        servo_mode="learned_full_pose",
                        center_u=f"{rb_det.center_uv[0]:.3f}" if rb_det is not None else "",
                        center_v=f"{rb_det.center_uv[1]:.3f}" if rb_det is not None else "",
                        ref_u=f"{rb_ref_uv[0]:.3f}" if rb_ref_uv is not None else "",
                        ref_v=f"{rb_ref_uv[1]:.3f}" if rb_ref_uv is not None else "",
                        err_u=f"{rb_err_u:.3f}",
                        err_v=f"{rb_err_v:.3f}",
                        dist_px=f"{rb_dist:.3f}" if math.isfinite(rb_dist) else "",
                        ref_source=rb_ref_source,
                        status=last_attempt_status + f"; rollback_result={rollback_result.get('wait_status', '')}",
                    )
                    print(
                        "gripper_learned_full neutral/noise: movement did not clearly improve or worsen; "
                        "rolled back and trying the next candidate direction",
                        flush=True,
                    )
                    neutral_visual_ok = rb_status == "ok" and rb_dist <= dist + max(2.0, neutral_band_px)
                    if not neutral_visual_ok:
                        last_attempt_status += "; neutral rollback visual_not_restored"
                        break
                    continue

                rollback_result = send_pose_with_encoder_outer_correction(
                    robot,
                    previous_pose,
                    args,
                    robot_lock,
                    stop_event=stop_event,
                    label="GRIPPER_FULL_ROLLBACK",
                        speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
                        min_steps=int(args.gripper_refine_min_move_steps),
                        wait_timeout_s=float(args.gripper_refine_wait_timeout_s),
                        settle_threshold=learned_settle_threshold,
                        outer_correction_enabled=rollback_outer_correction,
                    )
                time.sleep(max(0.0, float(args.gripper_settle_s)))
                rb_det, rb_ref_uv, rb_err_u, rb_err_v, rb_dist, rb_ref_source, rb_status = observe_gripper_error_stable(
                    gripper_capture,
                    gripper_model,
                    gripper_names,
                    calib,
                    args,
                    gripper_camera_lock,
                    overhead_uv,
                )
                rollback_visual_ok = rb_status == "ok" and rb_dist <= dist + max(2.0, rollback_worse_px)
                last_attempt_status = (
                    f"{attempt_status}; rollback {'visual_restored' if rollback_visual_ok else 'visual_not_restored'} "
                    f"rb_dist={rb_dist:.1f}px before={dist:.1f}px"
                )
                write_gripper_trace(
                    trace_writer,
                    event="learned_full_rollback",
                    iter=i,
                    servo_mode="learned_full_pose",
                    center_u=f"{rb_det.center_uv[0]:.3f}" if rb_det is not None else "",
                    center_v=f"{rb_det.center_uv[1]:.3f}" if rb_det is not None else "",
                    ref_u=f"{rb_ref_uv[0]:.3f}" if rb_ref_uv is not None else "",
                    ref_v=f"{rb_ref_uv[1]:.3f}" if rb_ref_uv is not None else "",
                    err_u=f"{rb_err_u:.3f}",
                    err_v=f"{rb_err_v:.3f}",
                    dist_px=f"{rb_dist:.3f}" if math.isfinite(rb_dist) else "",
                    ref_source=rb_ref_source,
                    status=last_attempt_status + f"; rollback_result={rollback_result.get('wait_status', '')}",
                )
                print(
                    f"gripper_learned_full rollback candidate={candidate_label}; "
                    f"{'visual restored' if rollback_visual_ok else 'visual not restored'}",
                    flush=True,
                )
                if not rollback_visual_ok:
                    break

            if not accepted:
                fallback_status = (
                    "gripper_learned_full prediction did not improve; "
                    f"best_improved_by={best_improve:.1f}px; {last_attempt_status}; "
                    "stopped for manual correction"
                )
                print("WARNING: " + fallback_status, flush=True)
                return "STOP: " + fallback_status

        return (
            "INCOMPLETE: gripper_learned_full max iterations reached before tolerance; "
            f"tolerance={float(args.gripper_center_tolerance_px):.1f}px; {status}"
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()


def run_gripper_verify_only_refine(
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
) -> str:
    observation = observe_gripper_error_median(
        gripper_capture,
        gripper_model,
        gripper_names,
        calib,
        args,
        gripper_camera_lock,
        overhead_uv,
    )
    if observation.detection is None or observation.ref_uv is None:
        return f"STOP: gripper_verify_only cannot judge alignment; {observation.status}; {observation.ref_source}"

    det = observation.detection
    ref = observation.ref_uv
    status = (
        f"gripper_verify_only err=({observation.error_uv[0]:.1f},{observation.error_uv[1]:.1f}) "
        f"dist={observation.dist_px:.1f}px center=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) "
        f"ref=({ref[0]:.1f},{ref[1]:.1f}) {observation.ref_source}"
    )
    print(status, flush=True)
    if gripper_error_within_tolerance(observation.dist_px, args):
        return "DONE: " + status
    return (
        "STOP: gripper_verify_only outside tolerance; "
        f"manual correction required with B/C. {status}"
    )


def run_gripper_ibvs_xy_refine(
    robot,
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    xy_target: dict[str, float],
    robot_lock: threading.RLock | None = None,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
    stop_event: threading.Event | None = None,
) -> str:
    """Closed-loop image servo using only settled, measured robot/image transitions."""
    if not all(joint in xy_target for joint in args.map_joints):
        return "INCOMPLETE: gripper_ibvs missing initial XY"

    trace_handle, trace_writer, _trace_path = open_gripper_trace(args)
    tolerance = gripper_ibvs_tolerance_px(calib, args)
    max_moves = int(args.gripper_final_iters)
    move_index = 0
    probe_count = 0
    stagnant_count = 0
    last_status = ""
    fresh_sample_floor = 0
    fresh_probes_needed = 0
    if bool(getattr(args, "gripper_ibvs_fresh_start", True)):
        fresh_sample_floor = max(
            (int(sample.get("id") or 0) for sample in calib.gripper_motion_samples),
            default=0,
        )
        fresh_probes_needed = 2
        last_status = f"fresh_start: ignoring saved motion samples id<={fresh_sample_floor}"
    baseline_area_px = 0.0
    baseline_lift: float | None = None
    shape_reference: dict[str, float] | None = None
    shape_reference_status = "mask_shape_profile not initialized"

    def cancelled() -> bool:
        return stop_event is not None and stop_event.is_set()

    def trace_observation(
        event: str,
        observation: GripperServoObservation,
        pose: dict[str, float],
        status_text: str,
        command_step: np.ndarray | None = None,
        actual_step: np.ndarray | None = None,
        jacobian: np.ndarray | None = None,
        condition: float | None = None,
        rmse: float | None = None,
    ) -> None:
        det = observation.detection
        ref_uv = observation.ref_uv
        mask_values = {
            key: f"{float(observation.mask_features.get(key, 0.0)):.4f}"
            if key in observation.mask_features
            else ""
            for key in MASK_SHAPE_FEATURE_KEYS
            if key != "mask_area_px"
        }
        write_gripper_trace(
            trace_writer,
            event=event,
            iter=move_index,
            servo_mode="ibvs",
            center_u=f"{det.center_uv[0]:.3f}" if det is not None else "",
            center_v=f"{det.center_uv[1]:.3f}" if det is not None else "",
            ref_u=f"{ref_uv[0]:.3f}" if ref_uv is not None else "",
            ref_v=f"{ref_uv[1]:.3f}" if ref_uv is not None else "",
            err_u=f"{observation.error_uv[0]:.3f}",
            err_v=f"{observation.error_uv[1]:.3f}",
            dist_px=f"{observation.dist_px:.3f}" if math.isfinite(observation.dist_px) else "",
            delta_pan=f"{command_step[0]:.4f}" if command_step is not None else "",
            delta_lift=f"{command_step[1]:.4f}" if command_step is not None else "",
            actual_delta_pan=f"{actual_step[0]:.4f}" if actual_step is not None else "",
            actual_delta_lift=f"{actual_step[1]:.4f}" if actual_step is not None else "",
            actual_pan=f"{pose.get(args.map_joints[0], float('nan')):.4f}",
            actual_lift=f"{pose.get(args.map_joints[1], float('nan')):.4f}",
            actual_elbow=f"{pose.get('elbow_flex.pos', float('nan')):.4f}",
            actual_wrist_flex=f"{pose.get('wrist_flex.pos', float('nan')):.4f}",
            actual_wrist_roll=f"{pose.get('wrist_roll.pos', float('nan')):.4f}",
            target_area_px=f"{observation.target_area_px:.2f}" if observation.target_area_px > 0 else "",
            area_ratio=(
                f"{(observation.target_area_px / baseline_area_px):.4f}"
                if baseline_area_px > 0 and observation.target_area_px > 0
                else ""
            ),
            **mask_values,
            j00_du_dpan=f"{jacobian[0, 0]:.5f}" if jacobian is not None else "",
            j01_du_dlift=f"{jacobian[0, 1]:.5f}" if jacobian is not None else "",
            j10_dv_dpan=f"{jacobian[1, 0]:.5f}" if jacobian is not None else "",
            j11_dv_dlift=f"{jacobian[1, 1]:.5f}" if jacobian is not None else "",
            condition=f"{condition:.4f}" if condition is not None else "",
            jacobian_rmse_px=f"{rmse:.4f}" if rmse is not None else "",
            observation_samples=observation.sample_count,
            observation_jitter_px=(
                f"{observation.jitter_px:.4f}" if math.isfinite(observation.jitter_px) else ""
            ),
            ref_source=observation.ref_source,
            status=status_text,
        )

    def observe() -> GripperServoObservation:
        return observe_gripper_error_median(
            gripper_capture,
            gripper_model,
            gripper_names,
            calib,
            args,
            gripper_camera_lock,
            overhead_uv,
        )

    def recover_hover_height(reason: str) -> tuple[bool, dict[str, float], GripperServoObservation, str]:
        nonlocal baseline_lift
        if getattr(args, "gripper_height_guard_action", "recover") != "recover":
            return False, locked_observation(robot, robot_lock), observe(), "height_recover disabled"
        current = locked_observation(robot, robot_lock)
        notes = [f"height_recover triggered: {reason}"]
        print("gripper_ibvs " + notes[0], flush=True)

        if calib.safe_pose:
            safe_lift_target = build_safe_lift_in_place_target(current, calib.safe_pose, args)
            send_interpolated(
                robot,
                safe_lift_target,
                args,
                robot_lock,
                speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
                min_steps=int(args.gripper_refine_min_move_steps),
                stop_event=stop_event,
            )
            if cancelled():
                recovered_pose = locked_observation(robot, robot_lock)
                recovered_observation = observe()
                return False, recovered_pose, recovered_observation, "height_recover cancelled during safe lift"
            ok_safe, safe_wait = wait_until_gripper_refine_pose(
                robot,
                safe_lift_target,
                args,
                robot_lock,
                stop_event=stop_event,
            )
            notes.append(("safe_lift ok: " if ok_safe else "safe_lift soft: ") + safe_wait)

        current = locked_observation(robot, robot_lock)
        current_xy = observed_map_xy(robot, args, robot_lock)
        if len(current_xy) != len(args.map_joints):
            recovered_observation = observe()
            return False, current, recovered_observation, "; ".join(notes + ["height_recover missing current XY"])
        ok_target, hover_target, hover_reason = prepare_xy_hover_refine_target(
            robot,
            calib,
            args,
            current_xy,
            robot_lock,
            anchor=None,
        )
        if not ok_target:
            recovered_observation = observe()
            return False, current, recovered_observation, "; ".join(notes + [hover_reason])
        send_interpolated(
            robot,
            hover_target,
            args,
            robot_lock,
            speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
            min_steps=int(args.gripper_refine_min_move_steps),
            stop_event=stop_event,
        )
        if cancelled():
            recovered_pose = locked_observation(robot, robot_lock)
            recovered_observation = observe()
            return False, recovered_pose, recovered_observation, "; ".join(notes + ["height_recover cancelled during hover restore"])
        ok_hover, hover_wait = wait_until_gripper_refine_pose(
            robot,
            hover_target,
            args,
            robot_lock,
            stop_event=stop_event,
        )
        time.sleep(max(0.0, float(args.gripper_settle_s)))
        recovered_pose = locked_observation(robot, robot_lock)
        recovered_observation = observe()
        if len(args.map_joints) > 1 and args.map_joints[1] in recovered_pose:
            baseline_lift = float(recovered_pose[args.map_joints[1]])
            notes.append(f"height_recover rebaselined {args.map_joints[1]}={baseline_lift:.2f}")
        notes.append(("hover_restore ok: " if ok_hover else "hover_restore soft: ") + hover_wait)
        status = "; ".join(notes)
        trace_observation("ibvs_height_recover", recovered_observation, recovered_pose, status)
        return (
            recovered_observation.detection is not None and recovered_observation.ref_uv is not None,
            recovered_pose,
            recovered_observation,
            status,
        )

    def move_and_measure(
        start_pose: dict[str, float],
        start_observation: GripperServoObservation,
        command_step: np.ndarray,
        anchor: RefinePoseAnchor,
        event: str,
        jacobian: np.ndarray | None,
    ) -> tuple[bool, str, dict[str, float], GripperServoObservation, bool]:
        command_step, lift_budget_status = limit_gripper_step_by_lift_budget(
            start_pose,
            command_step,
            args,
            baseline_lift,
        )
        if float(np.max(np.abs(command_step))) <= 1e-6:
            return False, lift_budget_status + "; command reduced to zero", start_pose, start_observation, False
        start_xy = {joint: float(start_pose[joint]) for joint in args.map_joints}
        target_xy = {
            joint: float(start_xy[joint]) + float(command_step[index])
            for index, joint in enumerate(args.map_joints)
        }
        ok_move, move_reason = send_xy_hover_refine_target(
            robot,
            calib,
            args,
            target_xy,
            robot_lock,
            anchor=anchor,
            stop_event=stop_event,
        )
        move_reason = f"{lift_budget_status}; {move_reason}"
        if not ok_move:
            end_pose = locked_observation(robot, robot_lock)
            end_observation = observe()
            ok_height, height_status = gripper_pose_height_guard(end_pose, calib, args, baseline_lift)
            if not ok_height:
                if "cumulative" in height_status and getattr(args, "gripper_height_guard_action", "recover") == "recover":
                    recovered, recovered_pose, recovered_observation, recovery_status = recover_hover_height(height_status)
                    if recovered:
                        trace_observation(event, recovered_observation, recovered_pose, recovery_status, command_step, None, jacobian)
                        return True, recovery_status, recovered_pose, recovered_observation, False
                    trace_observation(event, recovered_observation, recovered_pose, recovery_status, command_step, None, jacobian)
                    return False, recovery_status, recovered_pose, recovered_observation, False
                trace_observation(event, end_observation, end_pose, f"{move_reason}; {height_status}", command_step, None, jacobian)
                return False, f"{move_reason}; {height_status}", end_pose, end_observation, False
            if (
                bool(args.gripper_mask_area_guard)
                and baseline_area_px >= max(0.0, float(args.gripper_area_guard_min_px))
                and end_observation.target_area_px >= max(0.0, float(args.gripper_area_guard_min_px))
            ):
                area_ratio = end_observation.target_area_px / max(baseline_area_px, 1e-6)
                max_ratio = 1.0 + max(0.0, float(args.gripper_max_area_growth))
                if area_ratio > max_ratio:
                    guard_status = (
                        f"{move_reason}; mask_area_guard area_ratio={area_ratio:.2f}>{max_ratio:.2f}; "
                        f"area={end_observation.target_area_px:.0f}px baseline={baseline_area_px:.0f}px"
                    )
                    trace_observation(event, end_observation, end_pose, guard_status, command_step, None, jacobian)
                    return False, guard_status, end_pose, end_observation, False
            ok_shape, shape_status = gripper_mask_shape_guard(start_observation, end_observation, shape_reference, args)
            if not ok_shape:
                guard_status = f"{move_reason}; {shape_status}"
                trace_observation(event, end_observation, end_pose, guard_status, command_step, None, jacobian)
                return False, guard_status, end_pose, end_observation, False
            actual_step = np.asarray(
                [float(end_pose[joint]) - float(start_pose[joint]) for joint in args.map_joints],
                dtype=np.float64,
            )
            motion_ok, motion_status = gripper_actual_motion_quality(command_step, actual_step, args)
            if not motion_ok and end_observation.dist_px > tolerance:
                hard_status = f"{move_reason}; {motion_status}; {end_observation.status}; {shape_status}"
                trace_observation(event, end_observation, end_pose, hard_status, command_step, actual_step, jacobian)
                return False, hard_status, end_pose, end_observation, False
            soft_status = (
                f"{move_reason}; soft_wait_continue_from_actual_pose; "
                f"{motion_status}; {end_observation.status}; {shape_status}"
            )
            trace_observation(event, end_observation, end_pose, soft_status, command_step, actual_step, jacobian)
            if end_observation.detection is not None and end_observation.ref_uv is not None:
                print(
                    f"gripper_ibvs {event} soft-wait command=({command_step[0]:.2f},{command_step[1]:.2f})deg "
                    f"actual=({actual_step[0]:.2f},{actual_step[1]:.2f})deg "
                    f"dist={end_observation.dist_px:.1f}px; {motion_status}; not saving sample",
                    flush=True,
                )
                return True, soft_status, end_pose, end_observation, False
            return False, soft_status, end_pose, end_observation, False
        if cancelled():
            return False, "cancelled after move", locked_observation(robot, robot_lock), start_observation, False
        time.sleep(max(0.0, float(args.gripper_settle_s)))
        end_pose = locked_observation(robot, robot_lock)
        end_observation = observe()
        ok_height, height_status = gripper_pose_height_guard(end_pose, calib, args, baseline_lift)
        if not ok_height:
            if "cumulative" in height_status and getattr(args, "gripper_height_guard_action", "recover") == "recover":
                recovered, recovered_pose, recovered_observation, recovery_status = recover_hover_height(height_status)
                if recovered:
                    trace_observation(event, recovered_observation, recovered_pose, recovery_status, command_step, None, jacobian)
                    return True, recovery_status, recovered_pose, recovered_observation, False
                trace_observation(event, recovered_observation, recovered_pose, recovery_status, command_step, None, jacobian)
                return False, recovery_status, recovered_pose, recovered_observation, False
            trace_observation(event, end_observation, end_pose, height_status, command_step, None, jacobian)
            return False, height_status, end_pose, end_observation, False
        if (
            bool(args.gripper_mask_area_guard)
            and baseline_area_px >= max(0.0, float(args.gripper_area_guard_min_px))
            and end_observation.target_area_px >= max(0.0, float(args.gripper_area_guard_min_px))
        ):
            area_ratio = end_observation.target_area_px / max(baseline_area_px, 1e-6)
            max_ratio = 1.0 + max(0.0, float(args.gripper_max_area_growth))
            if area_ratio > max_ratio:
                guard_status = (
                    f"mask_area_guard area_ratio={area_ratio:.2f}>{max_ratio:.2f}; "
                    f"area={end_observation.target_area_px:.0f}px baseline={baseline_area_px:.0f}px"
                )
                trace_observation(event, end_observation, end_pose, guard_status, command_step, None, jacobian)
                return False, guard_status, end_pose, end_observation, False
        ok_shape, shape_status = gripper_mask_shape_guard(start_observation, end_observation, shape_reference, args)
        if not ok_shape:
            trace_observation(event, end_observation, end_pose, shape_status, command_step, None, jacobian)
            return False, shape_status, end_pose, end_observation, False
        actual_step = np.asarray(
            [float(end_pose[joint]) - float(start_pose[joint]) for joint in args.map_joints],
            dtype=np.float64,
        )
        motion_ok, motion_status = gripper_actual_motion_quality(command_step, actual_step, args)
        if not motion_ok and end_observation.dist_px > tolerance:
            hard_status = f"{motion_status}; {end_observation.status}; {shape_status}"
            trace_observation(event, end_observation, end_pose, hard_status, command_step, actual_step, jacobian)
            return False, hard_status, end_pose, end_observation, False
        sample_ok = False
        sample_status = "motion sample skipped: post-move observation unavailable"
        if end_observation.detection is not None and end_observation.ref_uv is not None:
            sample_ok, sample_status = append_gripper_motion_sample(
                calib,
                start_pose,
                end_pose,
                start_observation,
                end_observation,
                args,
                overhead_uv,
                event,
            )
            sample_status = f"{motion_status}; {sample_status}"
        condition = None
        if jacobian is not None:
            try:
                condition = float(np.linalg.cond(jacobian))
            except Exception:  # noqa: BLE001
                condition = None
        combined_status = f"{move_reason}; {end_observation.status}; {shape_status}; {sample_status}"
        trace_observation(
            event,
            end_observation,
            end_pose,
            combined_status,
            command_step,
            actual_step,
            jacobian,
            condition,
        )
        print(
            f"gripper_ibvs {event} command=({command_step[0]:.2f},{command_step[1]:.2f})deg "
            f"actual=({actual_step[0]:.2f},{actual_step[1]:.2f})deg "
            f"dist={end_observation.dist_px:.1f}px; {sample_status}",
            flush=True,
        )
        if end_observation.detection is None or end_observation.ref_uv is None:
            return False, combined_status, end_pose, end_observation, sample_ok
        return True, combined_status, end_pose, end_observation, sample_ok

    def restore_pose(
        best_pose: dict[str, float],
        reason: str,
    ) -> tuple[bool, dict[str, float], GripperServoObservation, str]:
        print(f"gripper_ibvs restoring best measured full pose: {reason}", flush=True)
        restore_result = send_pose_with_encoder_outer_correction(
            robot,
            best_pose,
            args,
            robot_lock,
            stop_event=stop_event,
            label="GRIPPER_RESTORE",
            speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
            min_steps=int(args.gripper_refine_min_move_steps),
            wait_timeout_s=float(args.gripper_refine_wait_timeout_s),
            settle_threshold=float(args.joint_error_threshold),
        )
        if cancelled() or restore_result.get("cancelled"):
            return False, locked_observation(robot, robot_lock), observe(), "restore cancelled"
        time.sleep(max(0.0, float(args.gripper_settle_s)))
        restored_pose = locked_observation(robot, robot_lock)
        restored_observation = observe()
        ok_height, height_status = gripper_pose_height_guard(restored_pose, calib, args, baseline_lift)
        correction_notes = list(restore_result.get("correction_notes") or [])
        correction_text = ""
        if correction_notes:
            correction_text = "; outer_correction " + " | ".join(correction_notes)
        wait_reason = (
            f"restore max_joint_error={float(restore_result.get('max_error', float('inf'))):.2f}; "
            f"raw_max_joint_error={float(restore_result.get('raw_error', float('inf'))):.2f}"
            f"{correction_text}; {restore_result.get('wait_status', '')}"
        )
        ok_wait = bool(restore_result.get("ok_wait"))
        if not ok_height:
            if "cumulative" in height_status and getattr(args, "gripper_height_guard_action", "recover") == "recover":
                recovered, recovered_pose, recovered_observation, recovery_status = recover_hover_height(height_status)
                if recovered:
                    return True, recovered_pose, recovered_observation, f"{reason}; {wait_reason}; {recovery_status}"
                trace_observation("ibvs_restore", recovered_observation, recovered_pose, f"{reason}; {wait_reason}; {recovery_status}")
                return False, recovered_pose, recovered_observation, f"{wait_reason}; {recovery_status}"
            trace_observation("ibvs_restore", restored_observation, restored_pose, f"{reason}; {wait_reason}; {height_status}")
            return False, restored_pose, restored_observation, f"{wait_reason}; {height_status}"
        restore_status = f"{reason}; {wait_reason}"
        if not ok_wait:
            restore_status += "; soft_restore_continue_from_actual_pose"
        restore_band = max(0.0, float(getattr(args, "gripper_ibvs_probe_restore_worse_px", 4.0)))
        if (
            restored_observation.detection is not None
            and restored_observation.ref_uv is not None
            and restored_observation.dist_px > best_observation.dist_px + restore_band
        ):
            restore_status += (
                f"; visual_restore_mismatch restored={restored_observation.dist_px:.1f}px "
                f"best={best_observation.dist_px:.1f}px band={restore_band:.1f}px"
            )
            trace_observation("ibvs_restore", restored_observation, restored_pose, restore_status)
            return False, restored_pose, restored_observation, restore_status
        trace_observation("ibvs_restore", restored_observation, restored_pose, restore_status)
        return (
            restored_observation.detection is not None and restored_observation.ref_uv is not None,
            restored_pose,
            restored_observation,
            restore_status,
        )

    try:
        if cancelled():
            return "STOP: gripper_ibvs cancelled before first observation"
        current_pose = locked_observation(robot, robot_lock)
        if not all(joint in current_pose for joint in args.map_joints):
            return "INCOMPLETE: gripper_ibvs cannot read actual pan/lift"
        baseline_lift = float(current_pose[args.map_joints[1]]) if len(args.map_joints) > 1 else None
        ok_start_height, start_height_status = gripper_pose_height_guard(current_pose, calib, args, baseline_lift)
        if not ok_start_height:
            return f"INCOMPLETE: gripper_ibvs unsafe start height; {start_height_status}"
        anchor = create_refine_pose_anchor(robot, calib, args, robot_lock)
        current_observation = observe()
        if current_observation.detection is None or current_observation.ref_uv is None:
            return f"INCOMPLETE: gripper_ibvs initial observation failed: {current_observation.status}"
        if current_observation.target_area_px > 0:
            baseline_area_px = float(current_observation.target_area_px)
        profile_reference, profile_status = load_gripper_mask_shape_profile(args)
        if profile_reference is not None:
            shape_reference = profile_reference
            shape_reference_status = profile_status
        else:
            calib_reference, calib_shape_status = gripper_mask_shape_reference_from_calibration(calib, args)
            if calib_reference is not None:
                shape_reference = calib_reference
                shape_reference_status = profile_status + "; " + calib_shape_status
            else:
                shape_reference = dict(current_observation.mask_features)
                shape_reference_status = (
                    profile_status + "; " + calib_shape_status + "; using initial observation as shape reference"
                )
        print(shape_reference_status, flush=True)
        best_pose = dict(current_pose)
        best_observation = current_observation
        jacobian: np.ndarray | None = None

        while max_moves <= 0 or move_index < max_moves:
            if cancelled():
                return "STOP: gripper_ibvs cancelled; no further motion sent"

            err_u, err_v = current_observation.error_uv
            current_xy = {joint: float(current_pose[joint]) for joint in args.map_joints}
            estimate_rmse = None
            condition = None
            if jacobian is None and fresh_probes_needed <= 0:
                estimate, estimate_status = estimate_local_gripper_jacobian(
                    calib,
                    current_xy,
                    overhead_uv,
                    args,
                    fresh_sample_floor,
                )
                last_status = estimate_status
                if estimate is not None:
                    jacobian = estimate.matrix.copy()
                    fresh_sample_floor = 0
                    condition = estimate.condition
                    estimate_rmse = estimate.rmse_px
                    print(
                        "gripper_ibvs " + estimate_status + " "
                        f"J=[[{jacobian[0, 0]:.2f},{jacobian[0, 1]:.2f}],"
                        f"[{jacobian[1, 0]:.2f},{jacobian[1, 1]:.2f}]]",
                        flush=True,
                    )
            elif jacobian is not None:
                condition = float(np.linalg.cond(jacobian))

            trace_observation(
                "ibvs_observe",
                current_observation,
                current_pose,
                last_status or current_observation.status,
                jacobian=jacobian,
                condition=condition,
                rmse=estimate_rmse,
            )
            print(
                f"gripper_ibvs iter={move_index + 1} err=({err_u:.1f},{err_v:.1f}) "
                f"dist={current_observation.dist_px:.1f}px accept<={tolerance:.1f}px "
                f"actual_xy=({current_xy[args.map_joints[0]]:.2f},{current_xy[args.map_joints[1]]:.2f}) "
                f"area={current_observation.target_area_px:.0f}px {current_observation.status}",
                flush=True,
            )
            if current_observation.dist_px <= tolerance:
                return (
                    f"DONE: gripper_ibvs aligned dist={current_observation.dist_px:.1f}px "
                    f"accept<={tolerance:.1f}px; {current_observation.ref_source}"
                )

            if jacobian is None:
                if probe_count >= max(2, int(args.gripper_ibvs_max_probes)):
                    return (
                        "INCOMPLETE: gripper_ibvs could not identify a usable local image Jacobian; "
                        f"probes={probe_count}; {last_status}"
                    )
                probe = abs(float(args.gripper_ibvs_probe_deg))
                directions = [
                    np.asarray([probe, 0.0]),
                    np.asarray([0.0, probe]),
                    np.asarray([-probe, 0.0]),
                    np.asarray([0.0, -probe]),
                    np.asarray([probe, probe]),
                    np.asarray([-probe, probe]),
                ]
                command_step = directions[probe_count % len(directions)]
                probe_count += 1
                move_index += 1
                ok_move, last_status, next_pose, next_observation, probe_sample_ok = move_and_measure(
                    current_pose,
                    current_observation,
                    command_step,
                    anchor,
                    "ibvs_probe",
                    None,
                )
                if not ok_move:
                    if cancelled():
                        return "STOP: gripper_ibvs cancelled during probe"
                    restored, current_pose, current_observation, restore_status = restore_pose(
                        best_pose,
                        "probe did not produce a valid settled observation",
                    )
                    if not restored:
                        return f"INCOMPLETE: gripper_ibvs probe failed; {last_status}; restore={restore_status}"
                    anchor = create_refine_pose_anchor(robot, calib, args, robot_lock)
                    continue
                if probe_sample_ok and fresh_probes_needed > 0:
                    fresh_probes_needed -= 1
                current_pose = next_pose
                current_observation = next_observation
                anchor = create_refine_pose_anchor(robot, calib, args, robot_lock)
                should_restore_probe = False
                should_restore_guard_band = False
                if current_observation.dist_px < best_observation.dist_px:
                    best_pose = dict(current_pose)
                    best_observation = current_observation
                else:
                    probe_restore_worse_px = max(
                        0.0,
                        float(getattr(args, "gripper_ibvs_probe_restore_worse_px", 4.0)),
                    )
                    should_restore_probe = (
                        current_observation.dist_px > best_observation.dist_px + probe_restore_worse_px
                    )
                    should_restore_guard_band = (
                        current_observation.dist_px
                        > best_observation.dist_px + 2.0 * float(args.gripper_ibvs_max_worse_px)
                    )
                if should_restore_probe or should_restore_guard_band:
                    restore_reason = (
                        f"probe worsened best by {current_observation.dist_px - best_observation.dist_px:.1f}px"
                        if should_restore_probe
                        else "probe drift exceeded guard band"
                    )
                    restored, current_pose, current_observation, restore_status = restore_pose(
                        best_pose,
                        restore_reason,
                    )
                    if not restored:
                        return f"INCOMPLETE: gripper_ibvs could not restore after probe drift; {restore_status}"
                    anchor = create_refine_pose_anchor(robot, calib, args, robot_lock)
                continue

            step_limit = adaptive_gripper_step_limit(current_observation.dist_px, args)
            command_step = damped_least_squares_step(
                jacobian,
                current_observation.error_uv,
                gain=float(args.gripper_ibvs_gain),
                damping=float(args.gripper_ibvs_damping),
                max_abs_step_deg=step_limit,
            )
            command_step = ensure_minimum_effective_step(
                command_step,
                minimum_abs_step_deg=min(step_limit, abs(float(args.gripper_min_step_deg))),
                maximum_abs_step_deg=step_limit,
            )
            predicted_error = np.asarray(current_observation.error_uv, dtype=np.float64) + jacobian @ command_step
            predicted_dist = float(np.linalg.norm(predicted_error))
            if not np.all(np.isfinite(command_step)) or predicted_dist >= current_observation.dist_px - 0.05:
                last_status = (
                    f"local J rejected: predicted dist {predicted_dist:.2f}px does not improve "
                    f"{current_observation.dist_px:.2f}px"
                )
                jacobian = None
                fresh_sample_floor = max(
                    (int(sample.get("id") or 0) for sample in calib.gripper_motion_samples),
                    default=0,
                )
                fresh_probes_needed = 2
                continue

            previous_pose = current_pose
            previous_observation = current_observation
            move_index += 1
            ok_move, last_status, next_pose, next_observation, sample_ok = move_and_measure(
                previous_pose,
                previous_observation,
                command_step,
                anchor,
                "ibvs_move",
                jacobian,
            )
            if not ok_move:
                if cancelled():
                    return "STOP: gripper_ibvs cancelled during correction"
                restored, current_pose, current_observation, restore_status = restore_pose(
                    best_pose,
                    "correction did not produce a valid settled observation",
                )
                if not restored:
                    return f"INCOMPLETE: gripper_ibvs correction failed; {last_status}; restore={restore_status}"
                anchor = create_refine_pose_anchor(robot, calib, args, robot_lock)
                jacobian = None
                fresh_sample_floor = max(
                    (int(sample.get("id") or 0) for sample in calib.gripper_motion_samples),
                    default=0,
                )
                fresh_probes_needed = 2
                continue

            current_pose = next_pose
            current_observation = next_observation
            anchor = create_refine_pose_anchor(robot, calib, args, robot_lock)
            actual_step = np.asarray(
                [float(current_pose[joint]) - float(previous_pose[joint]) for joint in args.map_joints],
                dtype=np.float64,
            )
            actual_pixel = np.asarray(current_observation.detection.center_uv) - np.asarray(
                previous_observation.detection.center_uv
            )
            if sample_ok:
                jacobian = broyden_update(
                    jacobian,
                    actual_step,
                    actual_pixel,
                    blend=float(args.gripper_ibvs_broyden_blend),
                )
                singular = np.linalg.svd(jacobian, compute_uv=False)
                condition = float(singular[0] / max(singular[-1], 1e-9))
                if (
                    not np.all(np.isfinite(jacobian))
                    or condition > float(args.gripper_ibvs_max_condition)
                    or float(singular[-1]) < float(args.gripper_ibvs_min_px_per_deg)
                ):
                    last_status += f"; updated J invalid cond={condition:.2f}; re-probing"
                    jacobian = None
            else:
                jacobian = None
            if jacobian is None:
                fresh_sample_floor = max(
                    (int(sample.get("id") or 0) for sample in calib.gripper_motion_samples),
                    default=0,
                )
                fresh_probes_needed = 2

            improvement = previous_observation.dist_px - current_observation.dist_px
            if current_observation.dist_px < best_observation.dist_px:
                best_pose = dict(current_pose)
                best_observation = current_observation
            if current_observation.dist_px > best_observation.dist_px + float(args.gripper_ibvs_max_worse_px):
                restored, current_pose, current_observation, restore_status = restore_pose(
                    best_pose,
                    f"measured divergence {current_observation.dist_px - best_observation.dist_px:.1f}px",
                )
                if not restored:
                    return f"INCOMPLETE: gripper_ibvs divergence restore failed; {restore_status}"
                anchor = create_refine_pose_anchor(robot, calib, args, robot_lock)
                jacobian = None
                fresh_sample_floor = max(
                    (int(sample.get("id") or 0) for sample in calib.gripper_motion_samples),
                    default=0,
                )
                fresh_probes_needed = 2
                stagnant_count = 0
                continue

            if improvement < float(args.gripper_ibvs_stagnation_px):
                stagnant_count += 1
            else:
                stagnant_count = 0
            if stagnant_count >= max(1, int(args.gripper_ibvs_max_stagnant)):
                last_status += f"; stagnant={stagnant_count}; re-identifying local J"
                jacobian = None
                fresh_sample_floor = max(
                    (int(sample.get("id") or 0) for sample in calib.gripper_motion_samples),
                    default=0,
                )
                fresh_probes_needed = 2
                stagnant_count = 0

        return (
            "INCOMPLETE: gripper_ibvs max moves reached before tolerance; "
            f"best={best_observation.dist_px:.1f}px current={current_observation.dist_px:.1f}px "
            f"accept<={tolerance:.1f}px; {last_status}"
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()


def run_gripper_final_xy_refine(
    robot,
    gripper_capture,
    gripper_model,
    gripper_names: dict[int, str],
    calib: CalibrationState,
    args,
    xy_target: dict[str, float],
    robot_lock: threading.RLock | None = None,
    gripper_camera_lock: threading.RLock | None = None,
    overhead_uv: tuple[float, float] | None = None,
    stop_event: threading.Event | None = None,
) -> str:
    if not args.enable_gripper_final_align:
        return "gripper_final=off"
    if gripper_capture is None or gripper_model is None:
        return "gripper_final=missing_camera_or_model"

    current_xy = {joint: float(xy_target[joint]) for joint in args.map_joints if joint in xy_target}
    if len(current_xy) != len(args.map_joints):
        return "gripper_final=skipped_missing_xy_target"
    if args.gripper_servo_mode == "verify_only":
        return run_gripper_verify_only_refine(
            gripper_capture,
            gripper_model,
            gripper_names,
            calib,
            args,
            gripper_camera_lock,
            overhead_uv,
        )
    if args.gripper_servo_mode == "debug_ibvs":
        return run_gripper_debug_ibvs_xy_refine(
            robot,
            gripper_capture,
            gripper_model,
            gripper_names,
            calib,
            args,
            current_xy,
            robot_lock,
            gripper_camera_lock,
            overhead_uv,
            stop_event,
        )
    if args.gripper_servo_mode == "learned":
        if getattr(args, "gripper_learned_control_space", "full_pose") == "full_pose":
            return run_gripper_learned_full_pose_refine(
                robot,
                gripper_capture,
                gripper_model,
                gripper_names,
                calib,
                args,
                current_xy,
                robot_lock,
                gripper_camera_lock,
                overhead_uv,
                stop_event,
            )
        return run_gripper_learned_xy_refine(
            robot,
            gripper_capture,
            gripper_model,
            gripper_names,
            calib,
            args,
            current_xy,
            robot_lock,
            gripper_camera_lock,
            overhead_uv,
            stop_event,
        )
    use_ibvs = (
        args.gripper_servo_mode == "ibvs"
        or not bool(args.allow_legacy_gripper_servo)
    )
    if use_ibvs:
        if args.gripper_servo_mode != "ibvs":
            print(
                f"WARNING: --gripper-servo-mode={args.gripper_servo_mode} is routed to measured-transition IBVS. "
                "Use --allow-legacy-gripper-servo only for offline/diagnostic comparison.",
                flush=True,
            )
        return run_gripper_ibvs_xy_refine(
            robot,
            gripper_capture,
            gripper_model,
            gripper_names,
            calib,
            args,
            current_xy,
            robot_lock,
            gripper_camera_lock,
            overhead_uv,
            stop_event,
        )
    if args.gripper_servo_mode == "search":
        return run_gripper_stepwise_xy_refine(
            robot,
            gripper_capture,
            gripper_model,
            gripper_names,
            calib,
            args,
            current_xy,
            robot_lock,
            gripper_camera_lock,
            overhead_uv,
            stop_event,
        )

    trace_handle, trace_writer, _trace_path = open_gripper_trace(args)
    status = "gripper_final=no_iterations"
    u_gain = float(args.gripper_u_to_pan)
    v_gain = float(args.gripper_v_to_lift)
    prev_err_u: float | None = None
    prev_err_v: float | None = None
    prev_dist: float | None = None
    prev_delta_pan = 0.0
    prev_delta_lift = 0.0
    jacobian: np.ndarray | None = None
    jacobian_condition: float | None = None
    jacobian_status = "jacobian not estimated"
    step_scale = 1.0
    max_iters = int(args.gripper_final_iters)
    i = 0
    try:
        while max_iters <= 0 or i < max_iters:
            if stop_event is not None and stop_event.is_set():
                return "STOP: gripper_final cancelled by user before reaching tolerance"
            i += 1
            det, frame, det_status = detect_from_gripper_frame(
                gripper_capture,
                gripper_model,
                gripper_names,
                args,
                gripper_camera_lock,
                flush_frames=0 if args.dry_run else int(args.gripper_camera_flush_frames),
            )
            if frame is None:
                return "gripper_final=stop_camera_read_failed"
            if det is None:
                return f"gripper_final=stop_{det_status}"

            ref_uv, ref_source = gripper_reference_pixel(calib, args, frame.shape, overhead_uv)
            if ref_uv is None:
                return f"gripper_final=stop_reference_unavailable {ref_source}"
            err_u = float(det.center_uv[0]) - float(ref_uv[0])
            err_v = float(det.center_uv[1]) - float(ref_uv[1])
            dist = math.hypot(err_u, err_v)
            if gripper_error_within_tolerance(dist, args):
                status = (
                    f"gripper_final iter={i} mode=already_aligned err=({err_u:.1f},{err_v:.1f}) "
                    f"dist={dist:.1f}px center=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) "
                    f"ref=({ref_uv[0]:.1f},{ref_uv[1]:.1f}) {ref_source}"
                )
                print(status, flush=True)
                write_gripper_trace(
                    trace_writer,
                    event="observe",
                    iter=i,
                    servo_mode="already_aligned",
                    center_u=f"{det.center_uv[0]:.3f}",
                    center_v=f"{det.center_uv[1]:.3f}",
                    ref_u=f"{ref_uv[0]:.3f}",
                    ref_v=f"{ref_uv[1]:.3f}",
                    err_u=f"{err_u:.3f}",
                    err_v=f"{err_v:.3f}",
                    dist_px=f"{dist:.3f}",
                    target_pan=f"{current_xy[args.map_joints[0]]:.4f}",
                    target_lift=f"{current_xy[args.map_joints[1]]:.4f}",
                    ref_source=ref_source,
                    status="within circular tolerance before probe",
                )
                return "DONE: " + status
            if (
                args.gripper_servo_mode == "jacobian"
                and prev_dist is not None
                and dist > prev_dist + float(args.gripper_diverge_epsilon_px)
            ):
                jacobian = None
                jacobian_status = f"jacobian refresh requested after divergence {prev_dist:.1f}->{dist:.1f}px"

            active_servo_mode = args.gripper_servo_mode
            if active_servo_mode == "jacobian":
                refresh_iters = int(args.gripper_jacobian_refresh_iters)
                needs_refresh = jacobian is None or (refresh_iters > 0 and i > 1 and (i - 1) % refresh_iters == 0)
                if needs_refresh:
                    jacobian, jacobian_condition, jacobian_status = estimate_gripper_image_jacobian(
                        robot,
                        gripper_capture,
                        gripper_model,
                        gripper_names,
                        calib,
                        args,
                        current_xy,
                        (float(det.center_uv[0]), float(det.center_uv[1])),
                        robot_lock,
                        gripper_camera_lock,
                        trace_writer,
                        i,
                    )
                    if jacobian is None:
                        active_servo_mode = "fixed_gain"
                        print(f"WARNING: gripper_final {jacobian_status}; falling back to fixed_gain", flush=True)

            if args.gripper_auto_flip_gains and active_servo_mode == "fixed_gain":
                if prev_err_u is not None and abs(prev_delta_pan) > 1e-6:
                    if abs(err_u) > abs(prev_err_u) + float(args.gripper_diverge_epsilon_px):
                        u_gain *= -1.0
                        print(
                            f"gripper_final auto-flip u gain; |err_u| {abs(prev_err_u):.1f}->{abs(err_u):.1f}, "
                            f"new gripper_u_to_pan={u_gain:.4f}",
                            flush=True,
                        )
                if prev_err_v is not None and abs(prev_delta_lift) > 1e-6:
                    if abs(err_v) > abs(prev_err_v) + float(args.gripper_diverge_epsilon_px):
                        v_gain *= -1.0
                        print(
                            f"gripper_final auto-flip v gain; |err_v| {abs(prev_err_v):.1f}->{abs(err_v):.1f}, "
                            f"new gripper_v_to_lift={v_gain:.4f}",
                            flush=True,
                        )

            status = (
                f"gripper_final iter={i} mode={active_servo_mode} err=({err_u:.1f},{err_v:.1f}) "
                f"dist={dist:.1f}px center=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) "
                f"ref=({ref_uv[0]:.1f},{ref_uv[1]:.1f}) {ref_source} {jacobian_status}"
            )
            print(status, flush=True)
            write_gripper_trace(
                trace_writer,
                event="observe",
                iter=i,
                servo_mode=active_servo_mode,
                center_u=f"{det.center_uv[0]:.3f}",
                center_v=f"{det.center_uv[1]:.3f}",
                ref_u=f"{ref_uv[0]:.3f}",
                ref_v=f"{ref_uv[1]:.3f}",
                err_u=f"{err_u:.3f}",
                err_v=f"{err_v:.3f}",
                dist_px=f"{dist:.3f}",
                target_pan=f"{current_xy[args.map_joints[0]]:.4f}",
                target_lift=f"{current_xy[args.map_joints[1]]:.4f}",
                j00_du_dpan=f"{jacobian[0, 0]:.6f}" if jacobian is not None else "",
                j01_du_dlift=f"{jacobian[0, 1]:.6f}" if jacobian is not None else "",
                j10_dv_dpan=f"{jacobian[1, 0]:.6f}" if jacobian is not None else "",
                j11_dv_dlift=f"{jacobian[1, 1]:.6f}" if jacobian is not None else "",
                condition=f"{jacobian_condition:.6f}" if jacobian_condition is not None else "",
                ref_source=ref_source,
                status=jacobian_status,
            )

            if active_servo_mode == "jacobian" and jacobian is not None:
                error_vec = np.array([err_u, err_v], dtype=np.float64)
                raw_delta = -float(args.gripper_servo_gain) * np.linalg.pinv(jacobian).dot(error_vec)
                delta = scale_joint_delta(raw_delta, float(args.gripper_max_step_deg) * step_scale)
                delta_pan = float(delta[0])
                delta_lift = float(delta[1])
            else:
                delta_pan = clamp_delta(u_gain * err_u, float(args.gripper_max_step_deg) * step_scale)
                delta_lift = clamp_delta(v_gain * err_v, float(args.gripper_max_step_deg) * step_scale)

            previous_xy = dict(current_xy)
            next_xy = dict(current_xy)
            next_xy[args.map_joints[0]] = float(next_xy[args.map_joints[0]]) + delta_pan
            next_xy[args.map_joints[1]] = float(next_xy[args.map_joints[1]]) + delta_lift
            ok_move, move_reason = send_xy_hover_refine_target(robot, calib, args, next_xy, robot_lock)
            if not ok_move:
                return f"gripper_final=stop_{move_reason}"
            current_xy = next_xy
            print(
                f"gripper_final move mode={active_servo_mode} delta_pan={delta_pan:.2f} delta_lift={delta_lift:.2f}; "
                f"xy=({current_xy[args.map_joints[0]]:.2f},{current_xy[args.map_joints[1]]:.2f}); {move_reason}",
                flush=True,
            )
            write_gripper_trace(
                trace_writer,
                event="move",
                iter=i,
                servo_mode=active_servo_mode,
                err_u=f"{err_u:.3f}",
                err_v=f"{err_v:.3f}",
                dist_px=f"{dist:.3f}",
                delta_pan=f"{delta_pan:.4f}",
                delta_lift=f"{delta_lift:.4f}",
                target_pan=f"{current_xy[args.map_joints[0]]:.4f}",
                target_lift=f"{current_xy[args.map_joints[1]]:.4f}",
                status=f"{move_reason}; step_scale={step_scale:.3f}",
            )
            time.sleep(max(0.0, float(args.gripper_settle_s)))
            accepted_move = True
            if bool(args.gripper_verify_move):
                verify_det, verify_frame, verify_status = detect_from_gripper_frame(
                    gripper_capture,
                    gripper_model,
                    gripper_names,
                    args,
                    gripper_camera_lock,
                    flush_frames=0 if args.dry_run else int(args.gripper_camera_flush_frames),
                )
                if verify_det is not None and verify_frame is not None:
                    verify_ref_uv, verify_ref_source = gripper_reference_pixel(calib, args, verify_frame.shape, overhead_uv)
                    if verify_ref_uv is not None:
                        verify_err_u = float(verify_det.center_uv[0]) - float(verify_ref_uv[0])
                        verify_err_v = float(verify_det.center_uv[1]) - float(verify_ref_uv[1])
                        verify_dist = math.hypot(verify_err_u, verify_err_v)
                        worse_by = verify_dist - dist
                        write_gripper_trace(
                            trace_writer,
                            event="verify",
                            iter=i,
                            servo_mode=active_servo_mode,
                            center_u=f"{verify_det.center_uv[0]:.3f}",
                            center_v=f"{verify_det.center_uv[1]:.3f}",
                            ref_u=f"{verify_ref_uv[0]:.3f}",
                            ref_v=f"{verify_ref_uv[1]:.3f}",
                            err_u=f"{verify_err_u:.3f}",
                            err_v=f"{verify_err_v:.3f}",
                            dist_px=f"{verify_dist:.3f}",
                            target_pan=f"{current_xy[args.map_joints[0]]:.4f}",
                            target_lift=f"{current_xy[args.map_joints[1]]:.4f}",
                            ref_source=verify_ref_source,
                            status=f"worse_by={worse_by:.3f}px",
                        )
                        if worse_by > float(args.gripper_worse_epsilon_px):
                            accepted_move = False
                            jacobian = None
                            jacobian_status = f"last move worsened {dist:.1f}->{verify_dist:.1f}px; Jacobian invalidated"
                            step_scale = max(float(args.gripper_adaptive_min_scale), step_scale * 0.5)
                            print(
                                f"WARNING: gripper_final {jacobian_status}; "
                                f"step_scale={step_scale:.2f}",
                                flush=True,
                            )
                            if bool(args.gripper_rollback_on_worse):
                                ok_rollback, rollback_reason = send_xy_hover_refine_target(
                                    robot,
                                    calib,
                                    args,
                                    previous_xy,
                                    robot_lock,
                                )
                                if ok_rollback:
                                    current_xy = previous_xy
                                    print("gripper_final rollback to previous XY; " + rollback_reason, flush=True)
                                    time.sleep(max(0.0, float(args.gripper_settle_s)))
                                    write_gripper_trace(
                                        trace_writer,
                                        event="rollback",
                                        iter=i,
                                        servo_mode=active_servo_mode,
                                        delta_pan=f"{-delta_pan:.4f}",
                                        delta_lift=f"{-delta_lift:.4f}",
                                        target_pan=f"{current_xy[args.map_joints[0]]:.4f}",
                                        target_lift=f"{current_xy[args.map_joints[1]]:.4f}",
                                        status=rollback_reason,
                                    )
                                else:
                                    return f"gripper_final=stop_rollback_failed {rollback_reason}"
                        else:
                            step_scale = min(1.0, step_scale * float(args.gripper_adaptive_recover))
                    else:
                        print(f"WARNING: gripper_final verify reference unavailable: {verify_ref_source}", flush=True)
                else:
                    print(f"WARNING: gripper_final verify skipped: {verify_status}", flush=True)
            if accepted_move:
                prev_err_u = err_u
                prev_err_v = err_v
                prev_dist = dist
                prev_delta_pan = delta_pan
                prev_delta_lift = delta_lift

        return (
            "INCOMPLETE: gripper_final max iterations reached before tolerance; "
            f"tolerance={float(args.gripper_center_tolerance_px):.1f}px; {status}"
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()


def calibration_help() -> None:
    print(
        """
Calibration keys:
  s = save current follower pose as safe hover pose
  c = capture calibration point: YOLO target center + current follower joint pose
  i/j/k/l = nudge the red target dot up/left/down/right before saving points
  o = reset target dot nudge offset
  u / Backspace = undo last calibration point
  h = help
  q = quit and save

Workflow:
  1. Teleoperate follower to a safe hover pose and press s.
  2. Move gripper center above target at several positions on the table.
  3. Auto target pixel defaults to mask_centroid for every object class.
  4. Keep the gripper down with the front jaw facing forward, just like your capture logs.
  5. At each aligned position, ensure YOLO detects the selected target, then press c.
  6. Need at least 4 well-spread object-center points. More is better.
"""
    )


def align_help() -> None:
    print(
        """
Align keys:
  a = run one guarded sequence: detect/lock object-center target -> lift in place -> full safe_pose -> move above locked target
  b = stop auto-align, wait for stability, mark the misaligned state, then enable manual correction
  m = stop auto-align and toggle manual correction without recording a BEFORE sample
  n = open follower gripper, sync the leader arm to the follower's current pose, then hold leader torque
  g = release the leader sync hold and enable manual teleop from that pose
  d = start/stop a local ACT grasp recording while controlling from the leader
  t = start/stop hover-height recording while you move at cube-hover height
  c = stop teleop, wait for stability, then update compact learned parameters from the corrected pose
  v = disabled in compact-parameter mode; DONE auto-updates learned parameters
  p = direct replay test: send one saved point's raw joints only, without vision/model/nearest-point logic
  r = stop auto-align if needed, then reopen overhead/gripper cameras together on the main thread
  i/j/k/l = debug only: nudge the target dot up/left/down/right
  o = reset target dot nudge offset
  u / Backspace = undo last calibration/correction point
  h = help
  q = stop the current alignment; press again while idle to quit

The script refuses to move if homography/safe_pose is missing, target is not detected,
orientation compensation is missing, the mapped joint target is outside calibrated limits,
or the predicted wrist compensation is outside calibrated/logged limits.

Correction workflow:
  0. Simple reteach mode can use full_pose plus gripper center refine; hover-height recording is optional.
  1. Press a to try auto-align. Stage 3 records the coarse pre-refine state before center refinement.
  2. If auto-align is good, press n to open the follower gripper, then move and lock the leader at the follower pose.
  3. Press g to release the leader lock and start local grasp teleop from that pose.
  4. Press d to start local ACT recording, perform the grasp, then press d again to save the episode.
  5. If XY is off, press b. After the arm stops, manual correction opens automatically.
  6. Move the gripper center truly above the target, then press c; c updates compact learned parameters.
  7. If auto-refine reaches DONE, compact learned parameters update automatically.
  8. Save at least 4 mask_centroid correction/verified points before expecting pose=yes.
  9. By default mask_centroid points are reused across object classes.
"""
    )


def run_calibrate(args) -> int:
    model, names = load_yolo(Path(args.model))
    config_path = Path(args.config)
    state = load_calibration(config_path)
    param_source = apply_saved_target_point_params(state, args)
    ensure_pose_model(state, args)
    ensure_orientation_model(state, args)
    capture = open_camera(args.camera_index, args, "overhead")
    preview = None if args.no_preview else PreviewWindow(WINDOW_NAME)
    robot = create_robot(args)
    leader = create_leader(args)
    latest_obs: dict[str, float] = {}
    cached_det: TargetDetection | None = None
    last_predict_time = 0.0
    status = f"calibration ready; {param_source}"
    calibration_help()

    try:
        print("Connecting leader/follower. If calibration prompt appears, press ENTER to use existing calibration.")
        leader.connect()
        robot.connect()

        while True:
            loop_start = time.perf_counter()
            action = leader.get_action()
            robot.send_action(action)
            latest_obs = scalar_motor_state(robot.get_observation())

            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera read failed")
            now = time.perf_counter()
            if cached_det is None or (now - last_predict_time) >= max(0.0, float(args.preview_inference_interval_s)):
                result = run_predict(model, frame, args)
                cached_det = select_detection(result, names, args.target_class, args.conf, args)
                last_predict_time = now
            det = cached_det

            h_state = "yes" if state.homography is not None else "no"
            safe_state = "yes" if state.safe_pose is not None else "no"
            pose_state = "yes" if state.pose_model else "no"
            orient_state = "yes" if state.orientation_model else "no"
            active_kind = active_target_point_kind(args.target_class, args)
            matching_points = len(matching_calibration_points(state.points, args))
            lines = [
                f"CALIBRATE target={args.target_class} point={active_kind} points={matching_points}/{len(state.points)} H={h_state} safe={safe_state} pose={pose_state} orient={orient_state}",
                f"pose_joints={','.join(j.split('.')[0] for j in args.pose_joints)} offset=({args.target_offset_u:.1f},{args.target_offset_v:.1f})",
                status,
                "s safe | c capture | i/j/k/l nudge target | o reset | u undo | q quit",
            ]
            annotated = draw_detection(frame, det, lines)
            key = -1 if preview is None else preview.show(annotated)
            key = normalize_key(key)

            if key == ord("q"):
                break
            if key == ord("h"):
                calibration_help()
            nudge_status = handle_target_nudge(key, args)
            if nudge_status is not None:
                cached_det = None
                last_predict_time = 0.0
                ensure_pose_model(state, args)
                ensure_orientation_model(state, args)
                status = f"{nudge_status}; save points with this offset"
                print(status, flush=True)
            if key in (ord("u"), 8):
                if state.points:
                    removed = state.points.pop()
                    recompute_homography(state, args)
                    save_calibration(config_path, state, args)
                    status = f"removed point {removed.get('id')}; points={len(state.points)}"
                else:
                    status = "no point to undo"
            if key == ord("s"):
                state.safe_pose = latest_obs.copy()
                save_calibration(config_path, state, args)
                status = "saved current follower pose as safe hover pose"
                print(status)
            if key == ord("c"):
                if det is None:
                    status = f"cannot capture: target {args.target_class} not detected"
                    print(status)
                else:
                    ok_capture, status = append_calibration_point(state, det, latest_obs, args, source="calibrate")
                    if ok_capture:
                        save_calibration(config_path, state, args)
                    print(status)

            elapsed = time.perf_counter() - loop_start
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)

    finally:
        if preview is not None:
            preview.close()
        capture.release()
        if gripper_capture is not None:
            gripper_capture.release()
        try:
            if leader.is_connected:
                leader.disconnect()
        finally:
            if robot.is_connected:
                robot.disconnect()
        save_calibration(config_path, state, args)
        print(f"Calibration saved to: {config_path}")

    return 0


def run_alignment_once(
    robot,
    model,
    names,
    capture,
    calib: CalibrationState,
    args,
    robot_lock: threading.RLock | None = None,
    camera_lock: threading.RLock | None = None,
    gripper_model=None,
    gripper_names: dict[int, str] | None = None,
    gripper_capture=None,
    gripper_camera_lock: threading.RLock | None = None,
    stop_event: threading.Event | None = None,
    config_path: Path | None = None,
    locked_overhead_callback: Callable[[tuple[float, float]], None] | None = None,
    pre_refine_callback: Callable[[dict[str, Any]], None] | None = None,
    post_refine_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    if calib.safe_pose is None:
        return "REFUSE: safe_pose missing. Run calibration mode and press s at safe hover pose."
    pose_source = ensure_pose_model(calib, args)
    orientation_source = ensure_orientation_model(calib, args)
    hover_source = ensure_hover_height_model(calib, args)
    gripper_only_available = (
        args.enable_gripper_final_align
        and gripper_capture is not None
        and gripper_model is not None
        and bool(calib.hover_height_model)
    )
    gripper_only_mode = gripper_only_available and not calib.pose_model and calib.homography is None
    replay_seed_available = bool(
        args.gripper_replay_nearest_pose
        and any(
            isinstance(point.get("joints"), dict)
            and all(joint in point.get("joints", {}) for joint in args.pose_joints)
            for point in matching_calibration_points(calib.points, args)
        )
    )
    if not calib.pose_model and calib.homography is None and not gripper_only_mode and not replay_seed_available:
        return "REFUSE: pose_model/homography missing. Save at least one C/V point for replay, or 4+ points for a model."
    if not calib.pose_model and not calib.orientation_model and not gripper_only_mode and not replay_seed_available:
        return "REFUSE: orientation compensation missing. Capture correct gripper-down calibration points or enable CSV orientation."

    if gripper_only_mode:
        current = locked_observation(robot, robot_lock)
        current_xy = {joint: float(current[joint]) for joint in args.map_joints if joint in current}
        if len(current_xy) != len(args.map_joints):
            return "REFUSE: gripper-only align needs current shoulder_pan/shoulder_lift observation."
        hover_target = predict_hover_height_joints(current_xy, calib, args)
        hover_target, hover_clamped, hover_reason = clamp_to_limits(hover_target, calib.hover_height_limits)
        if not hover_target:
            return f"REFUSE: gripper-only align cannot compute fixed height: {hover_reason}"
        if hover_clamped:
            print(f"WARNING: gripper-only initial hover clamped; {hover_reason}", flush=True)
        orientation_target = predict_xy_hover_orientation_joints(current_xy, hover_target, calib, args)
        if orientation_target and calib.orientation_limits:
            ok_orientation, orientation_reason = within_orientation_limits(orientation_target, calib.orientation_limits)
            if not ok_orientation:
                return f"REFUSE: gripper-only orientation {orientation_reason}"

        if args.dry_run:
            xy_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in current_xy.items())
            hover_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in hover_target.items())
            return f"DRY RUN: gripper-only fixed-height align; {hover_source}; current {xy_short}; hover {hover_short}"

        print("stage 1/2: gripper-only mode keeps current XY and moves to fixed hover/contact height", flush=True)
        target_pose = build_xy_hover_pose_target(current, calib.safe_pose, current_xy, hover_target, orientation_target)
        send_interpolated(robot, target_pose, args, robot_lock, stop_event=stop_event)
        ok_hover, hover_wait = wait_until_gripper_refine_pose(
            robot,
            target_pose,
            args,
            robot_lock,
            timeout_s=float(args.xy_hover_wait_timeout_s),
            stable_offset_deg=float(args.xy_hover_stable_offset_deg),
            stop_event=stop_event,
        )
        print("gripper-only hover: " + hover_wait, flush=True)
        if not ok_hover:
            return "STOP: gripper-only hover pose did not become stable; " + hover_wait
        print("stage 2/2: gripper-camera XY refine; success is judged by gripper-camera center error", flush=True)
        gripper_status = run_gripper_final_xy_refine(
            robot,
            gripper_capture,
            gripper_model,
            gripper_names or {},
            calib,
            args,
            current_xy,
            robot_lock,
            gripper_camera_lock,
            stop_event=stop_event,
        )
        if config_path is not None and not args.dry_run:
            save_calibration(config_path, calib, args)
        return gripper_status

    det = detect_target_from_camera_locked(
        capture,
        model,
        names,
        args,
        camera_lock,
        flush_frames=0 if args.dry_run else args.camera_flush_frames,
    )
    if det is None:
        return f"REFUSE: target {args.target_class} not detected above conf={args.conf}. No motion sent."
    print(
        f"locked target before safe_pose: pixel=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}); "
        "using this target even if the arm later occludes the camera",
        flush=True,
    )
    if locked_overhead_callback is not None:
        locked_overhead_callback((float(det.center_uv[0]), float(det.center_uv[1])))

    def capture_refine_checkpoint(label: str) -> tuple[dict[str, Any] | None, str]:
        if gripper_capture is None or gripper_model is None:
            return None, "refine checkpoint not recorded"
        checkpoint_pose = locked_observation(robot, robot_lock)
        gripper_observation = observe_gripper_error_median(
            gripper_capture,
            gripper_model,
            gripper_names or {},
            calib,
            args,
            gripper_camera_lock,
            (float(det.center_uv[0]), float(det.center_uv[1])),
        )
        gripper_det = gripper_observation.detection
        if gripper_det is None:
            return None, f"refine checkpoint not recorded {label}: gripper target missing; {gripper_observation.status}"
        checkpoint = {
            "captured_at": now_iso(),
            "label": label,
            "target_class": det.class_name,
            "overhead_detection": det,
            "overhead_pixel": [float(det.center_uv[0]), float(det.center_uv[1])],
            "gripper_pixel": [
                float(gripper_det.center_uv[0]),
                float(gripper_det.center_uv[1]),
            ],
            "gripper_confidence": float(gripper_det.confidence),
            "joints": {joint: float(checkpoint_pose[joint]) for joint in MOTOR_KEYS if joint in checkpoint_pose},
            "status": gripper_observation.status,
        }
        return (
            checkpoint,
            f"refine checkpoint recorded {label}: "
            f"gripper=({gripper_det.center_uv[0]:.1f},{gripper_det.center_uv[1]:.1f})",
        )

    last_pre_refine_checkpoint: dict[str, Any] | None = None
    last_post_refine_checkpoint: dict[str, Any] | None = None

    def remember_pre_refine(label: str) -> str:
        nonlocal last_pre_refine_checkpoint
        if pre_refine_callback is None:
            return "pre_refine not recorded"
        checkpoint, checkpoint_status = capture_refine_checkpoint(label)
        if checkpoint is None:
            return checkpoint_status.replace("refine checkpoint", "pre_refine")
        last_pre_refine_checkpoint = checkpoint
        pre_refine_callback(checkpoint)
        return checkpoint_status.replace("refine checkpoint", "pre_refine")

    def remember_post_refine(label: str) -> str:
        nonlocal last_post_refine_checkpoint
        if post_refine_callback is None:
            return "post_refine not recorded"
        checkpoint, checkpoint_status = capture_refine_checkpoint(label)
        if checkpoint is None:
            return checkpoint_status.replace("refine checkpoint", "post_refine")
        last_post_refine_checkpoint = checkpoint
        post_refine_callback(checkpoint)
        return checkpoint_status.replace("refine checkpoint", "post_refine")

    def update_coarse_bias_from_last_refine(source: str) -> str:
        before = last_pre_refine_checkpoint
        after = last_post_refine_checkpoint
        if not isinstance(before, dict) or not isinstance(after, dict):
            return "coarse_bias not updated: missing refine checkpoints"
        before_joints = before.get("joints", {})
        after_joints = after.get("joints", {})
        before_gripper = before.get("gripper_pixel")
        after_gripper = after.get("gripper_pixel")
        overhead_pixel = before.get("overhead_pixel") or after.get("overhead_pixel")
        ref_uv, _ref_source = gripper_reference_pixel(
            calib,
            args,
            None,
            (float(overhead_pixel[0]), float(overhead_pixel[1]))
            if overhead_pixel is not None and len(overhead_pixel) == 2
            else None,
        )
        before_dist_px = None
        after_dist_px = None
        if ref_uv is not None:
            if before_gripper is not None and len(before_gripper) == 2:
                before_dist_px = math.hypot(float(before_gripper[0]) - float(ref_uv[0]), float(before_gripper[1]) - float(ref_uv[1]))
            if after_gripper is not None and len(after_gripper) == 2:
                after_dist_px = math.hypot(float(after_gripper[0]) - float(ref_uv[0]), float(after_gripper[1]) - float(ref_uv[1]))
        ok_bias, bias_status = update_coarse_pose_bias_model(
            calib,
            args,
            before_joints=before_joints,
            after_joints=after_joints,
            before_dist_px=before_dist_px,
            after_dist_px=after_dist_px,
            source=source,
            overhead_uv=(
                (float(overhead_pixel[0]), float(overhead_pixel[1]))
                if overhead_pixel is not None and len(overhead_pixel) == 2
                else None
            ),
        )
        return bias_status

    current = locked_observation(robot, robot_lock)
    safe_lift_target = build_safe_lift_in_place_target(current, calib.safe_pose, args)
    safe_pose_target = build_safe_pose_target(current, calib.safe_pose)

    if args.dry_run:
        print("DRY RUN: would lock target, lift in place, move to full safe_pose, then align to locked target.", flush=True)
    else:
        print("stage 1a/2: lifting in place to safe height/orientation; shoulder_pan is intentionally held", flush=True)
        ok_wait, wait_status = send_and_wait_pose(
            robot,
            safe_lift_target,
            args.safe_lift_wait_joints,
            args,
            "safe lift",
            robot_lock,
            stop_event=stop_event,
        )
        if stop_event is not None and stop_event.is_set():
            return "STOP: alignment cancelled during safe lift"
        if not ok_wait:
            safe_lift_error, safe_lift_details = current_pose_error_details(
                robot,
                safe_lift_target,
                args.safe_lift_wait_joints,
                robot_lock,
            )
            if safe_lift_error <= float(args.safe_lift_continue_threshold):
                print(
                    "WARNING: safe lift is approximate but close enough to continue to full safe_pose; "
                    f"max_error={safe_lift_error:.2f} continue_threshold={float(args.safe_lift_continue_threshold):.2f}; "
                    f"{format_pose_error_details(safe_lift_details)}",
                    flush=True,
                )
            elif not args.continue_if_safe_timeout:
                return "STOP: safe lift did not settle; not moving toward target from a low/unsafe pose."
            else:
                print("WARNING: safe lift did not fully settle; continuing because --continue-if-safe-timeout is set.", flush=True)

        current = locked_observation(robot, robot_lock)
        safe_pose_target = build_safe_pose_target(current, calib.safe_pose)
        must_settle = [joint for joint in args.full_safe_must_settle_joints if joint in safe_pose_target]
        must_text = ",".join(must_settle) if must_settle else "none"
        print(
            "stage 1b/2: moving to full safe_pose after target lock; "
            f"must settle: {must_text}",
            flush=True,
        )

        def full_safe_retry_should_stop(
            current_error: float,
            current_details: list[tuple[float, str, float, float]],
        ) -> tuple[bool, str]:
            must_error, _ = filtered_pose_error_details(current_details, args.full_safe_must_settle_joints)
            if must_error > float(args.full_safe_position_threshold):
                return False, "required full-safe joint is still outside threshold"
            if current_error <= float(args.full_safe_continue_threshold):
                return True, (
                    "required joint settled and remaining non-critical error is within continue threshold; "
                    f"max_error={current_error:.2f} continue_threshold={float(args.full_safe_continue_threshold):.2f}; "
                    f"{format_pose_error_details(current_details)}"
                )
            return False, "full safe error is still above continue threshold"

        ok_wait, wait_status = send_and_wait_pose(
            robot,
            safe_pose_target,
            args.full_safe_wait_joints,
            args,
            "full safe_pose",
            robot_lock,
            full_safe_retry_should_stop,
            stop_event,
        )
        if stop_event is not None and stop_event.is_set():
            return "STOP: alignment cancelled during full safe_pose"
        if not ok_wait:
            full_safe_error, full_safe_details = current_pose_error_details(
                robot,
                safe_pose_target,
                args.full_safe_wait_joints,
                robot_lock,
            )
            full_safe_must_error, full_safe_must_details = filtered_pose_error_details(
                full_safe_details,
                args.full_safe_must_settle_joints,
            )
            if full_safe_must_error > float(args.full_safe_position_threshold):
                if not args.continue_if_safe_timeout:
                    return (
                        "STOP: full safe_pose required joint did not settle; "
                        f"must_error={full_safe_must_error:.2f} "
                        f"threshold={float(args.full_safe_position_threshold):.2f}; "
                        f"{format_pose_error_details(full_safe_must_details)}"
                    )
                print(
                    "WARNING: full safe_pose required joint did not settle; continuing because "
                    f"--continue-if-safe-timeout is set. must_error={full_safe_must_error:.2f}; "
                    f"{format_pose_error_details(full_safe_must_details)}",
                    flush=True,
                )
            elif full_safe_error <= float(args.full_safe_continue_threshold):
                print(
                    "WARNING: full safe_pose is approximate but close enough to continue toward the locked target; "
                    f"max_error={full_safe_error:.2f} continue_threshold={float(args.full_safe_continue_threshold):.2f}; "
                    f"{format_pose_error_details(full_safe_details)}",
                    flush=True,
                )
            elif not args.continue_if_safe_timeout:
                return "STOP: full safe_pose did not settle; not moving toward target from a low/unsafe pose."
            else:
                print("WARNING: full safe_pose did not fully settle; continuing because --continue-if-safe-timeout is set.", flush=True)
        time.sleep(max(0.0, args.safe_settle_s))

    current = locked_observation(robot, robot_lock)

    if not calib.pose_model and calib.homography is None and replay_seed_available:
        replay_point, replay_dist = nearest_replay_pose_point(det.center_uv[0], det.center_uv[1], calib, args)
        if replay_point is None:
            return (
                "REFUSE: only replay points are available, but no saved point is near this overhead target; "
                f"move manually and press C/V, or collect 4+ points for pose_model. snap<={float(args.gripper_replay_snap_px):.1f}px"
            )
        replay_joints = replay_point.get("joints", {})
        if not isinstance(replay_joints, dict) or not all(joint in replay_joints for joint in args.pose_joints):
            return "REFUSE: nearest replay point is missing full pose joints"
        target_pose = {
            joint: float(replay_joints[joint])
            for joint in MOTOR_KEYS
            if joint in replay_joints
        }
        if args.dry_run:
            pose_short = " ".join(f"{joint.split('.')[0]}={target_pose[joint]:.1f}" for joint in args.pose_joints if joint in target_pose)
            return (
                f"DRY RUN: replay point {replay_point.get('id')} "
                f"dist={float(replay_dist or 0.0):.1f}px would move saved full pose {pose_short}"
            )
        print(
            f"stage 2/3: replaying saved full pose point {replay_point.get('id')} "
            f"source={replay_point.get('source')} dist={float(replay_dist or 0.0):.1f}px",
            flush=True,
        )
        replay_result = send_pose_with_encoder_outer_correction(
            robot,
            target_pose,
            args,
            robot_lock,
            stop_event=stop_event,
            label="SAVED_POSE_REPLAY",
            speed_deg_s=float(args.gripper_refine_joint_speed_deg_s),
            min_steps=int(args.gripper_refine_min_move_steps),
            wait_timeout_s=float(args.xy_hover_wait_timeout_s),
            settle_threshold=float(args.xy_hover_stable_offset_deg),
        )
        if replay_result.get("cancelled"):
            return f"STOP: alignment cancelled during saved-pose replay {replay_result.get('stage')}"
        after = locked_observation(robot, robot_lock)
        max_error = float(replay_result["max_error"])
        detail = format_pose_error_details(replay_result["details"])
        raw_error = float(replay_result["raw_error"])
        correction_notes = list(replay_result["correction_notes"])
        correction_text = ""
        if correction_notes:
            correction_text = "; outer_correction " + " | ".join(correction_notes)
        status = (
            f"replay point {replay_point.get('id')} dist={float(replay_dist or 0.0):.1f}px "
            f"max_joint_error={max_error:.2f}; raw_max_joint_error={raw_error:.2f}"
            f"{correction_text}; {replay_result['wait_status']}; {detail}"
        )
        print(status, flush=True)
        if not bool(replay_result["ok_wait"]) or max_error > float(args.joint_error_threshold):
            return "STOP: saved-pose replay did not become stable; " + status
        if args.enable_gripper_final_align:
            refine_xy = {
                joint: float(after[joint])
                for joint in args.map_joints
                if joint in after
            }
            if len(refine_xy) != len(args.map_joints):
                return "STOP: gripper final align cannot read actual pan/lift after saved-pose replay"
            pre_refine_status = remember_pre_refine("replay_before_stage3")
            print("stage 3/3: gripper-camera final XY refine from replayed pose", flush=True)
            gripper_status = run_gripper_final_xy_refine(
                robot,
                gripper_capture,
                gripper_model,
                gripper_names or {},
                calib,
                args,
                refine_xy,
                robot_lock,
                gripper_camera_lock,
                overhead_uv=(float(det.center_uv[0]), float(det.center_uv[1])),
                stop_event=stop_event,
            )
            post_refine_status = remember_post_refine("replay_after_stage3")
            if config_path is not None and not args.dry_run:
                save_calibration(config_path, calib, args)
            if gripper_status.startswith("DONE:"):
                coarse_bias_status = update_coarse_bias_from_last_refine("replay_stage3_done")
                if config_path is not None and not args.dry_run:
                    save_calibration(config_path, calib, args)
                return "DONE: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status + "; " + coarse_bias_status
            if gripper_status.startswith("STOP:"):
                return "STOP: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status
            return "INCOMPLETE: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status
        return "DONE: " + status

    if calib.pose_model:
        pose_target = predict_pose_target(det.center_uv[0], det.center_uv[1], calib, args)
        if not pose_target:
            return "REFUSE: pose_model produced no joint targets; recalibrate or delete stale config fields."
        nearest_dist = nearest_matching_point_distance(det.center_uv[0], det.center_uv[1], calib, args)
        snap_text = "nearest=n/a"
        if nearest_dist is not None:
            snap_text = f"nearest={nearest_dist:.1f}px snap<={float(args.local_snap_px):.1f}px"
        mode = alignment_pose_mode(calib, args)
        if mode == "xy_hover":
            xy_target = {joint: float(pose_target[joint]) for joint in args.map_joints if joint in pose_target}
            if len(xy_target) != len(args.map_joints):
                return "REFUSE: pose model did not produce both XY map joints; recalibrate."
            replay_point, replay_dist = nearest_replay_pose_point(det.center_uv[0], det.center_uv[1], calib, args)
            replay_joints = replay_point.get("joints", {}) if replay_point is not None else {}
            replay_text = ""
            if replay_point is not None:
                xy_target = {joint: float(replay_joints[joint]) for joint in args.map_joints}
                replay_text = (
                    f" replay_point={replay_point.get('id')} source={replay_point.get('source')} "
                    f"dist={float(replay_dist):.1f}px"
                )
            xy_limits = {joint: calib.pose_limits[joint] for joint in xy_target if joint in calib.pose_limits}
            ok_xy, xy_reason = within_limits(xy_target, xy_limits)
            if not ok_xy:
                return f"REFUSE: xy target {xy_reason}"
            hover_target = (
                {joint: float(replay_joints[joint]) for joint in hover_height_outputs(args) if joint in replay_joints}
                if replay_point is not None
                else predict_hover_height_joints(xy_target, calib, args)
            )
            if not hover_target:
                return f"REFUSE: xy_hover requested but hover height model is missing. Press T to record hover height samples. {hover_source}"
            hover_target, hover_clamped, hover_reason = (
                (hover_target, False, "saved replay hover")
                if replay_point is not None
                else clamp_to_limits(hover_target, calib.hover_height_limits)
            )
            if not hover_target:
                return f"REFUSE: hover target {hover_reason}"
            if hover_clamped:
                print(f"WARNING: hover target outside learned height range; {hover_reason}", flush=True)
            orientation_target = (
                {joint: float(replay_joints[joint]) for joint in args.orientation_joints if joint in replay_joints}
                if replay_point is not None
                else predict_xy_hover_orientation_joints(xy_target, hover_target, calib, args)
            )
            ok_orientation, orientation_reason = (
                within_orientation_limits(orientation_target, calib.orientation_limits)
                if orientation_target
                else (True, "orientation kept from safe_pose")
            )
            if not ok_orientation:
                return f"REFUSE: xy_hover orientation {orientation_reason}"

            if args.dry_run:
                xy_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in xy_target.items())
                hover_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in hover_target.items())
                orientation_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in orientation_target.items())
                return f"DRY RUN: {pose_source}; {hover_source}; {orientation_source}; {snap_text}{replay_text}; would move xy_hover {xy_short}; {hover_short}; {orientation_short}"

            print("stage 2/2: moving down/over to xy_hover target above object", flush=True)
            status = ""
            target_check = {**orientation_target, **hover_target, **xy_target}
            for i in range(max(1, args.closed_loop_iters)):
                current = locked_observation(robot, robot_lock)
                target_pose = build_xy_hover_pose_target(current, calib.safe_pose, xy_target, hover_target, orientation_target)
                sent = send_interpolated(robot, target_pose, args, robot_lock, stop_event=stop_event)
                if stop_event is not None and stop_event.is_set():
                    return "STOP: alignment cancelled while moving to xy_hover"
                ok_stable, stable_status = wait_until_gripper_refine_pose(
                    robot,
                    target_pose,
                    args,
                    robot_lock,
                    timeout_s=float(args.xy_hover_wait_timeout_s),
                    stable_offset_deg=float(args.xy_hover_stable_offset_deg),
                    stop_event=stop_event,
                )
                after = locked_observation(robot, robot_lock)
                errors = {
                    joint: target_check[joint] - after.get(joint, sent.get(joint, 0.0))
                    for joint in target_check
                    if joint in after or joint in sent
                }
                max_error = max((abs(value) for value in errors.values()), default=0.0)
                status = (
                    f"iter {i + 1}: xy_hover pixel=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) "
                    f"{snap_text}{replay_text} max_joint_error={max_error:.2f}; {stable_status}"
                )
                print(status, flush=True)
                if ok_stable:
                    break
                if i + 1 >= max(1, args.closed_loop_iters):
                    return (
                        "STOP: xy_hover camera pose never became stable, so gripper image servo was not started; "
                        + status
                    )
            if args.enable_gripper_final_align:
                refine_obs = locked_observation(robot, robot_lock)
                refine_xy = {
                    joint: float(refine_obs[joint])
                    for joint in args.map_joints
                    if joint in refine_obs
                }
                if len(refine_xy) != len(args.map_joints):
                    return "STOP: gripper final align cannot read actual pan/lift after xy_hover"
                if max_error > float(args.joint_error_threshold):
                    print(
                        "WARNING: xy_hover did not settle exactly; starting gripper refine from actual observed XY "
                        f"instead of predicted XY. max_joint_error={max_error:.2f} "
                        f"actual=({refine_xy[args.map_joints[0]]:.2f},{refine_xy[args.map_joints[1]]:.2f}) "
                        f"predicted=({xy_target[args.map_joints[0]]:.2f},{xy_target[args.map_joints[1]]:.2f})",
                        flush=True,
                    )
                pre_refine_status = remember_pre_refine("xy_hover_before_stage3")
                print("stage 3/3: gripper-camera final XY refine from current coarse pose", flush=True)
                gripper_status = run_gripper_final_xy_refine(
                    robot,
                    gripper_capture,
                    gripper_model,
                    gripper_names or {},
                    calib,
                    args,
                    refine_xy,
                    robot_lock,
                    gripper_camera_lock,
                    overhead_uv=(float(det.center_uv[0]), float(det.center_uv[1])),
                    stop_event=stop_event,
                )
                post_refine_status = remember_post_refine("xy_hover_after_stage3")
                if config_path is not None and not args.dry_run:
                    save_calibration(config_path, calib, args)
                if gripper_status.startswith("DONE:"):
                    coarse_bias_status = update_coarse_bias_from_last_refine("xy_hover_stage3_done")
                    learn_status = ""
                    if not args.dry_run:
                        final_obs = locked_observation(robot, robot_lock)
                        final_gripper_observation = observe_gripper_error_median(
                            gripper_capture,
                            gripper_model,
                            gripper_names or {},
                            calib,
                            args,
                            gripper_camera_lock,
                            (float(det.center_uv[0]), float(det.center_uv[1])),
                        )
                        final_error_px = (
                            final_gripper_observation.dist_px
                            if final_gripper_observation.detection is not None
                            else None
                        )
                        ok_learn, learn_status = append_auto_gripper_refine_point(
                            calib,
                            det,
                            refine_xy,
                            final_obs,
                            args,
                            final_error_px=final_error_px,
                        )
                        if ok_learn:
                            if config_path is not None:
                                save_calibration(config_path, calib, args)
                            pose_source = ensure_pose_model(calib, args)
                            orientation_source = ensure_orientation_model(calib, args)
                            learn_status = f"auto overhead learned; {learn_status}; {pose_source}; {orientation_source}"
                        else:
                            learn_status = "auto overhead not learned; " + learn_status
                    if config_path is not None and not args.dry_run:
                        save_calibration(config_path, calib, args)
                    return "DONE: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status + "; " + coarse_bias_status + ("; " + learn_status if learn_status else "")
                if gripper_status.startswith("STOP:"):
                    return "STOP: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status
                return "INCOMPLETE: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status
            return "DONE: " + status

        ok_pose, pose_reason = within_limits(pose_target, calib.pose_limits)
        if not ok_pose:
            return f"REFUSE: {pose_reason}"
        replay_point, replay_dist = nearest_replay_pose_point(det.center_uv[0], det.center_uv[1], calib, args)
        replay_text = ""
        if replay_point is not None:
            replay_joints = replay_point.get("joints", {})
            if isinstance(replay_joints, dict) and all(joint in replay_joints for joint in args.pose_joints):
                pose_target = {
                    joint: float(replay_joints[joint])
                    for joint in args.pose_joints
                    if joint in replay_joints
                }
                if bool(getattr(args, "direct_replay_include_gripper", True)) and "gripper.pos" in replay_joints:
                    pose_target["gripper.pos"] = float(replay_joints["gripper.pos"])
                replay_text = (
                    f" replay_point={replay_point.get('id')} source={replay_point.get('source')} "
                    f"dist={float(replay_dist or 0.0):.1f}px"
                )
        bias_text = ""
        if replay_point is None:
            pose_target, bias_text = apply_coarse_pose_bias(
                pose_target,
                calib,
                args,
                source="full_pose_model",
            )
            if bias_text and not bias_text.endswith("unavailable") and "disabled" not in bias_text:
                print(bias_text, flush=True)
        target_pose = build_full_pose_target(current, calib.safe_pose, pose_target)

        if args.dry_run:
            pose_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in pose_target.items())
            return f"DRY RUN: {pose_source}; {snap_text}{replay_text}; {bias_text}; after safe_pose would move full_pose {pose_short}"

        print("stage 2/2: moving down/over to full_pose above target", flush=True)
        status = ""
        # A nearby saved point should behave like the P direct-replay test: one
        # full-pose replay plus encoder outer correction. Repeating the whole
        # full_pose approach changes the backlash path and can move the gripper
        # camera away even when the encoder error looks better.
        full_pose_attempts = (
            1
            if replay_point is not None
            else max(max(1, int(args.closed_loop_iters)), max(2, int(args.safe_retry_count) + 1))
        )
        full_pose_ok = False
        for i in range(full_pose_attempts):
            current = locked_observation(robot, robot_lock)
            target_pose = build_full_pose_target(current, calib.safe_pose, pose_target)
            replay_result = send_pose_with_encoder_outer_correction(
                robot,
                target_pose,
                args,
                robot_lock,
                stop_event=stop_event,
                label="FULL_POSE_REPLAY" if replay_point is not None else "FULL_POSE",
                speed_deg_s=float(args.gripper_refine_joint_speed_deg_s) if replay_point is not None else None,
                min_steps=int(args.gripper_refine_min_move_steps) if replay_point is not None else None,
                wait_timeout_s=float(args.xy_hover_wait_timeout_s),
                settle_threshold=float(args.gripper_refine_continue_threshold),
            )
            if replay_result.get("cancelled"):
                return f"STOP: alignment cancelled during full_pose {replay_result.get('stage')}"
            after = locked_observation(robot, robot_lock)
            errors = {joint: pose_target[joint] - after.get(joint, target_pose.get(joint, 0.0)) for joint in pose_target}
            max_error = max((abs(value) for value in errors.values()), default=0.0)
            raw_error = float(replay_result["raw_error"])
            correction_notes = list(replay_result["correction_notes"])
            correction_text = ""
            if correction_notes:
                correction_text = "; outer_correction " + " | ".join(correction_notes)
            status = (
                f"iter {i + 1}: full_pose pixel=({det.center_uv[0]:.1f},{det.center_uv[1]:.1f}) "
                f"{snap_text}{replay_text} {bias_text} max_joint_error={max_error:.2f}; raw_max_joint_error={raw_error:.2f}"
                f"{correction_text}; {replay_result['wait_status']}"
            )
            print(status, flush=True)
            if bool(replay_result["ok_wait"]) and max_error <= max(float(args.joint_error_threshold), float(args.gripper_refine_continue_threshold)):
                full_pose_ok = True
                break
        if not full_pose_ok:
            visible_refine_status = ""
            visible_joint_limit = max(0.0, float(getattr(args, "gripper_refine_visible_max_joint_error_deg", 2.5)))
            if (
                args.enable_gripper_final_align
                and bool(getattr(args, "gripper_refine_if_visible_after_full_pose_error", False))
                and max_error <= visible_joint_limit
            ):
                visible_checkpoint, visible_refine_status = capture_refine_checkpoint("full_pose_visible_before_stage3")
                if visible_checkpoint is not None:
                    if pre_refine_callback is not None:
                        pre_refine_callback(visible_checkpoint)
                    full_pose_ok = True
                    status += (
                        f"; allowing gripper refine because target is visible despite "
                        f"max_joint_error={max_error:.2f}>{float(args.joint_error_threshold):.2f}"
                    )
                    print(status, flush=True)
                else:
                    status += "; " + visible_refine_status
            if full_pose_ok:
                pass
            else:
                replay_fail_status = ""
                if replay_text:
                    replay_fail_status = "; " + mark_replay_pose_failure(replay_point, status)
                    if config_path is not None and not args.dry_run:
                        save_calibration(config_path, calib, args)
                return (
                    "STOP: full_pose/replay target did not settle close enough; "
                    + status
                    + replay_fail_status
                    + "; not starting gripper refine from the wrong pose"
                )
        if args.enable_gripper_final_align:
            refine_obs = locked_observation(robot, robot_lock)
            refine_xy = {
                joint: float(refine_obs[joint])
                for joint in args.map_joints
                if joint in refine_obs
            }
            if len(refine_xy) != len(args.map_joints):
                return "STOP: gripper final align cannot read actual pan/lift after full_pose"
            pre_refine_status = remember_pre_refine("full_pose_before_stage3")
            print("stage 3/3: gripper-camera final XY refine from full_pose coarse pose", flush=True)
            gripper_status = run_gripper_final_xy_refine(
                robot,
                gripper_capture,
                gripper_model,
                gripper_names or {},
                calib,
                args,
                refine_xy,
                robot_lock,
                gripper_camera_lock,
                overhead_uv=(float(det.center_uv[0]), float(det.center_uv[1])),
                stop_event=stop_event,
            )
            post_refine_status = remember_post_refine("full_pose_after_stage3")
            if config_path is not None and not args.dry_run:
                save_calibration(config_path, calib, args)
            if gripper_status.startswith("DONE:"):
                coarse_bias_status = update_coarse_bias_from_last_refine("full_pose_stage3_done")
                if config_path is not None and not args.dry_run:
                    save_calibration(config_path, calib, args)
                return "DONE: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status + "; " + coarse_bias_status
            if gripper_status.startswith("STOP:"):
                return "STOP: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status
            return "INCOMPLETE: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status
        return "DONE: " + status

    j0, j1 = pixel_to_joint_target(calib.homography, det.center_uv[0], det.center_uv[1])
    mapped_target = {args.map_joints[0]: j0, args.map_joints[1]: j1}
    ok_limits, reason = within_limits(mapped_target, calib.joint_limits)
    if not ok_limits:
        return f"REFUSE: {reason}"
    orientation_target = predict_orientation_joints(mapped_target, calib.orientation_model, args)
    ok_orientation, orientation_reason = within_orientation_limits(orientation_target, calib.orientation_limits)
    if not ok_orientation:
        return f"REFUSE: {orientation_reason}"

    safe_lift_target = current.copy()
    for key, value in calib.safe_pose.items():
        if key not in args.map_joints:
            safe_lift_target[key] = float(value)
    if all(joint in current for joint in args.map_joints):
        current_mapped = {joint: float(current[joint]) for joint in args.map_joints}
        safe_lift_target.update(predict_orientation_joints(current_mapped, calib.orientation_model, args))

    target_pose = build_safe_plane_target(
        current,
        calib.safe_pose,
        calib.orientation_model,
        args.map_joints,
        mapped_target,
        args,
    )

    if args.dry_run:
        return (
            f"DRY RUN: {orientation_source}; after safe_pose would move {args.map_joints[0]}={j0:.2f}, "
            f"{args.map_joints[1]}={j1:.2f}, orientation={orientation_target}"
        )

    print("stage 2/2: moving down/over to homography target above object", flush=True)
    send_interpolated(robot, safe_lift_target, args, robot_lock, stop_event=stop_event)
    if stop_event is not None and stop_event.is_set():
        return "STOP: alignment cancelled during homography safe lift"

    status = ""
    for i in range(max(1, args.closed_loop_iters)):
        current = locked_observation(robot, robot_lock)
        target_pose = build_safe_plane_target(
            current,
            calib.safe_pose,
            calib.orientation_model,
            args.map_joints,
            mapped_target,
            args,
        )
        sent = send_interpolated(robot, target_pose, args, robot_lock, stop_event=stop_event)
        if stop_event is not None and stop_event.is_set():
            return "STOP: alignment cancelled during homography target motion"
        after = locked_observation(robot, robot_lock)
        err0 = mapped_target[args.map_joints[0]] - after.get(args.map_joints[0], sent.get(args.map_joints[0], 0.0))
        err1 = mapped_target[args.map_joints[1]] - after.get(args.map_joints[1], sent.get(args.map_joints[1], 0.0))
        status = f"iter {i + 1}: target=({j0:.2f},{j1:.2f}) joint_error=({err0:.2f},{err1:.2f})"
        print(status)
        if max(abs(err0), abs(err1)) <= args.joint_error_threshold:
            break
    if args.enable_gripper_final_align:
        refine_obs = locked_observation(robot, robot_lock)
        refine_xy = {
            joint: float(refine_obs[joint])
            for joint in args.map_joints
            if joint in refine_obs
        }
        if len(refine_xy) != len(args.map_joints):
            return "STOP: gripper final align cannot read actual pan/lift after homography target"
        pre_refine_status = remember_pre_refine("homography_before_stage3")
        print("stage 3/3: gripper-camera final XY refine from homography coarse pose", flush=True)
        gripper_status = run_gripper_final_xy_refine(
            robot,
            gripper_capture,
            gripper_model,
            gripper_names or {},
            calib,
            args,
            refine_xy,
            robot_lock,
            gripper_camera_lock,
            overhead_uv=(float(det.center_uv[0]), float(det.center_uv[1])),
            stop_event=stop_event,
        )
        post_refine_status = remember_post_refine("homography_after_stage3")
        if config_path is not None and not args.dry_run:
            save_calibration(config_path, calib, args)
        if gripper_status.startswith("DONE:"):
            coarse_bias_status = update_coarse_bias_from_last_refine("homography_stage3_done")
            if config_path is not None and not args.dry_run:
                save_calibration(config_path, calib, args)
            return "DONE: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status + "; " + coarse_bias_status
        if gripper_status.startswith("STOP:"):
            return "STOP: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status
        return "INCOMPLETE: " + status + "; " + pre_refine_status + "; " + gripper_status + "; " + post_refine_status
    return "DONE: " + status


def run_align(args) -> int:
    model, names = load_yolo(Path(args.model))
    gripper_model = None
    gripper_names: dict[int, str] = {}
    if args.enable_gripper_final_align:
        gripper_model, gripper_names = load_yolo(Path(args.gripper_model))
    config_path = Path(args.config)
    calib = load_calibration(config_path)
    param_source = apply_saved_target_point_params(calib, args)
    ignored_legacy_auto = sum(
        1
        for point in calib.points
        if str(point.get("source") or "").startswith("auto_gripper_refine")
        and not bool(point.get("verified"))
        and not bool(args.use_legacy_auto_calibration_points)
    )
    trusted_point_objects = {id(point) for point in matching_calibration_points(calib.points, args)}
    ignored_manual_points = sum(
        1
        for point in calib.points
        if str(point.get("source") or "") == "manual_correction"
        and id(point) not in trusted_point_objects
    )
    if ignored_legacy_auto or ignored_manual_points:
        recompute_homography(calib, args)
        print(
            f"calibration audit: excluded {ignored_legacy_auto} unverified legacy auto-refine points and "
            f"{ignored_manual_points} low-confidence/outlier manual points; "
            f"rebuilt derived pose/orientation models from trusted points",
            flush=True,
        )
    pose_source = ensure_pose_model(calib, args)
    orientation_source = ensure_orientation_model(calib, args)
    hover_source = ensure_hover_height_model(calib, args)
    if calib.pose_model or calib.hover_height_model:
        save_calibration(config_path, calib, args)
    gripper_only_ready = bool(args.enable_gripper_final_align and calib.hover_height_model)
    if calib.safe_pose is None or (
        not gripper_only_ready and not calib.pose_model and (calib.homography is None or not calib.orientation_model)
    ):
        print(f"Calibration incomplete: {config_path}")
        print("Run calibrate mode first, press s for safe hover, or use gripper final align with a recorded hover-height model.")
    if args.enable_gripper_final_align and int(args.camera_index) == int(args.gripper_camera_index):
        raise RuntimeError("overhead camera index and gripper camera index are the same; use different indexes.")
    capture = open_camera(args.camera_index, args, "overhead")
    gripper_capture = open_camera(args.gripper_camera_index, args, "gripper") if args.enable_gripper_final_align else None
    preview = None if args.no_preview else PreviewWindow(WINDOW_NAME)
    robot = create_robot(args)
    leader = create_leader(args) if args.enable_manual_correction else None
    robot_lock = threading.RLock()
    camera_lock = threading.RLock()
    gripper_camera_lock = threading.RLock()
    alignment_cancel = threading.Event()
    manual_enabled = threading.Event()
    stop_teleop = threading.Event()
    teleop_shared: dict[str, Any] = {
        "latest_obs": {},
        "latest_action": {},
        "latest_performed": {},
        "latest_command_time": 0.0,
        "teleop_fps": 0.0,
        "error": "",
    }
    teleop_thread: threading.Thread | None = None
    alignment_state: dict[str, Any] = {
        "thread": None,
        "running": False,
        "status": "",
        "started_at": 0.0,
        "locked_overhead_uv": None,
        "pre_refine_before": None,
        "post_refine_after": None,
    }
    latest_obs: dict[str, float] = {}
    last_det: TargetDetection | None = None
    last_det_time = 0.0
    cached_det: TargetDetection | None = None
    cached_gripper_det: TargetDetection | None = None
    cached_gripper_frame = None
    last_gripper_det: TargetDetection | None = None
    last_gripper_det_time = 0.0
    last_gripper_predict_time = 0.0
    last_predict_time = 0.0
    duplicate_camera_count = 0
    gripper_camera_fault = ""
    hover_recording = False
    hover_record_samples: list[dict[str, Any]] = []
    last_hover_record_time = 0.0
    act_dataset = None
    act_recording = False
    act_record_last_time = 0.0
    act_record_status = "act_rec=off"
    manual_correction = False
    gripper_correction_before: dict[str, Any] | None = None
    pending_manual_after_alignment = False
    pending_before_after_alignment = False
    pending_camera_reopen_reason: str | None = None
    before_marker_state: dict[str, Any] = {
        "thread": None,
        "running": False,
        "status": "",
        "started_at": 0.0,
    }
    leader_sync_hold: dict[str, Any] = {
        "active": False,
        "motors": [],
    }
    status = "press a to align; q quit"
    align_help()

    def enable_manual_correction(reason: str) -> str:
        nonlocal manual_correction
        if leader is None:
            return "manual correction disabled; restart with --enable-manual-correction"
        teleop_shared["error"] = ""
        manual_enabled.set()
        manual_correction = True
        return f"manual correction ON: {reason}; camera streams unchanged"

    def request_alignment_stop_for_manual(reason: str) -> str:
        nonlocal pending_manual_after_alignment
        alignment_cancel.set()
        pending_manual_after_alignment = True
        return f"alignment stop requested for manual correction: {reason}; wait for current motion step to finish"

    def clear_gripper_cache() -> None:
        nonlocal cached_gripper_det, cached_gripper_frame, last_gripper_det, last_gripper_det_time, last_gripper_predict_time
        cached_gripper_det = None
        cached_gripper_frame = None
        last_gripper_det = None
        last_gripper_det_time = 0.0
        last_gripper_predict_time = 0.0

    def reopen_camera_pair(reason: str) -> str:
        nonlocal capture, gripper_capture, duplicate_camera_count
        if bool(alignment_state.get("running")):
            return f"camera pair reopen deferred until alignment stops: {reason}"
        clear_gripper_cache()
        with camera_lock:
            with gripper_camera_lock:
                if gripper_capture is not None:
                    try:
                        gripper_capture.release()
                    except Exception:
                        pass
                try:
                    capture.release()
                except Exception:
                    pass
                time.sleep(0.35)
                capture = open_camera(args.camera_index, args, "overhead")
                gripper_capture = (
                    open_camera(args.gripper_camera_index, args, "gripper")
                    if args.enable_gripper_final_align
                    else None
                )
        duplicate_camera_count = 0
        return f"camera pair reopened on main thread: {reason}"

    def fresh_correction_detections() -> tuple[TargetDetection | None, TargetDetection | None, str]:
        fresh_det = detect_target_from_camera_locked(
            capture,
            model,
            names,
            args,
            camera_lock,
            flush_frames=max(1, int(args.camera_flush_frames)),
        )
        fresh_gripper_det = None
        grip_status = "gripper camera disabled"
        if gripper_capture is not None and gripper_model is not None:
            gripper_observation = observe_gripper_error_median(
                gripper_capture,
                gripper_model,
                gripper_names,
                calib,
                args,
                gripper_camera_lock,
            )
            fresh_gripper_det = gripper_observation.detection
            grip_status = gripper_observation.status
        det_status = "fresh overhead ok" if fresh_det is not None else "fresh overhead missing"
        return fresh_det, fresh_gripper_det, f"{det_status}; fresh gripper {grip_status}"

    def mark_gripper_correction_before(
        before_det: TargetDetection | None,
        before_gripper_det: TargetDetection | None,
        obs: dict[str, float],
    ) -> tuple[bool, str]:
        nonlocal gripper_correction_before
        if before_det is None:
            return False, "cannot mark before: overhead target not visible"
        if before_gripper_det is None:
            return False, "cannot mark before: gripper target not visible"
        if not obs:
            return False, "cannot mark before: follower observation missing"
        gripper_correction_before = {
            "captured_at": now_iso(),
            "target_class": before_det.class_name,
            "overhead_detection": before_det,
            "overhead_pixel": [float(before_det.center_uv[0]), float(before_det.center_uv[1])],
            "gripper_pixel": [float(before_gripper_det.center_uv[0]), float(before_gripper_det.center_uv[1])],
            "gripper_confidence": float(before_gripper_det.confidence),
            "joints": {joint: float(obs[joint]) for joint in MOTOR_KEYS if joint in obs},
        }
        return (
            True,
            "marked BEFORE misaligned sample: "
            f"overhead=({before_det.center_uv[0]:.1f},{before_det.center_uv[1]:.1f}) "
            f"gripper=({before_gripper_det.center_uv[0]:.1f},{before_gripper_det.center_uv[1]:.1f}); "
            "manually align, then press C",
        )

    def start_alignment_worker() -> None:
        def remember_locked_overhead_uv(uv: tuple[float, float]) -> None:
            alignment_state["locked_overhead_uv"] = (float(uv[0]), float(uv[1]))

        def remember_pre_refine_before(sample: dict[str, Any]) -> None:
            alignment_state["pre_refine_before"] = sample

        def remember_post_refine_after(sample: dict[str, Any]) -> None:
            alignment_state["post_refine_after"] = sample

        def worker() -> None:
            try:
                result = run_alignment_once(
                    robot,
                    model,
                    names,
                    capture,
                    calib,
                    args,
                    robot_lock,
                    camera_lock,
                    gripper_model,
                    gripper_names,
                    gripper_capture,
                    gripper_camera_lock,
                    alignment_cancel,
                    config_path,
                    remember_locked_overhead_uv,
                    remember_pre_refine_before,
                    remember_post_refine_after,
                )
            except Exception as exc:  # noqa: BLE001
                result = f"ERROR: alignment worker failed: {exc}"
            alignment_state["status"] = result
            alignment_state["running"] = False
            print(result, flush=True)

        alignment_state["running"] = True
        alignment_cancel.clear()
        alignment_state["status"] = "alignment running in background; preview remains live"
        alignment_state["started_at"] = time.perf_counter()
        alignment_state["locked_overhead_uv"] = None
        alignment_state["pre_refine_before"] = None
        alignment_state["post_refine_after"] = None
        thread = threading.Thread(target=worker, name="yolo-guided-alignment", daemon=True)
        alignment_state["thread"] = thread
        thread.start()

    def start_direct_replay_worker() -> None:
        def worker() -> None:
            try:
                result = run_direct_pose_replay_test(
                    robot,
                    calib,
                    args,
                    robot_lock,
                    alignment_cancel,
                    gripper_capture,
                    gripper_model,
                    gripper_names,
                    gripper_camera_lock,
                )
            except Exception as exc:  # noqa: BLE001
                result = f"ERROR: direct replay worker failed: {exc}"
            alignment_state["status"] = result
            alignment_state["running"] = False
            print(result, flush=True)

        alignment_state["running"] = True
        alignment_cancel.clear()
        alignment_state["status"] = "direct replay running in background; preview remains live"
        alignment_state["started_at"] = time.perf_counter()
        alignment_state["locked_overhead_uv"] = None
        alignment_state["pre_refine_before"] = None
        alignment_state["post_refine_after"] = None
        thread = threading.Thread(target=worker, name="direct-pose-replay", daemon=True)
        alignment_state["thread"] = thread
        thread.start()

    def start_before_marker_worker(
        *,
        reason: str,
        fallback_det: TargetDetection | None,
        fallback_last_det: TargetDetection | None,
        fallback_gripper_det: TargetDetection | None,
        fallback_last_gripper_det: TargetDetection | None,
        locked_overhead_uv: tuple[float, float] | None,
    ) -> None:
        if before_marker_state.get("running"):
            alignment_state["status"] = "BEFORE marker is already being captured; preview remains live"
            return

        def worker() -> None:
            try:
                manual_enabled.clear()
                stable_before, marker_obs, marker_stable_status = wait_until_observation_stable(
                    robot,
                    list(args.pose_joints),
                    float(args.manual_save_stable_timeout_s),
                    float(args.gripper_refine_stable_epsilon_deg),
                    int(args.gripper_refine_stable_samples),
                    robot_lock,
                )
                if not stable_before:
                    result = "cannot mark BEFORE: " + marker_stable_status
                    return
                fresh_det, fresh_gripper_det, fresh_status = fresh_correction_detections()
                before_det = fresh_det if fresh_det is not None else (fallback_det if fallback_det is not None else fallback_last_det)
                if before_det is not None and locked_overhead_uv is not None:
                    before_det = replace(
                        before_det,
                        center_uv=(float(locked_overhead_uv[0]), float(locked_overhead_uv[1])),
                    )
                before_gripper_det = (
                    fresh_gripper_det
                    if fresh_gripper_det is not None
                    else (fallback_gripper_det if fallback_gripper_det is not None else fallback_last_gripper_det)
                )
                ok_before, result = mark_gripper_correction_before(before_det, before_gripper_det, marker_obs)
                if not ok_before:
                    result += "; " + fresh_status
                    return
                if isinstance(alignment_state.get("pre_refine_before"), dict):
                    result += "; auto pre-refine checkpoint is preserved; this B marker will be saved as post-refine/attempted state on C"
                if not manual_enabled.is_set() and leader is not None:
                    result += "; " + enable_manual_correction("manual correction after BEFORE marker")
                result += "; " + marker_stable_status + "; " + fresh_status
            except Exception as exc:  # noqa: BLE001
                result = f"ERROR: BEFORE marker worker failed: {exc}"
            finally:
                before_marker_state["status"] = result
                before_marker_state["running"] = False
                print(result, flush=True)

        before_marker_state["running"] = True
        before_marker_state["status"] = f"BEFORE marker capture running in background: {reason}; preview remains live"
        before_marker_state["started_at"] = time.perf_counter()
        thread = threading.Thread(target=worker, name="before-marker-capture", daemon=True)
        before_marker_state["thread"] = thread
        thread.start()

    try:
        if args.enable_manual_correction:
            print("Connecting follower and leader for manual correction. If calibration prompt appears, press ENTER to use existing calibration.")
        else:
            print("Connecting follower only. If calibration prompt appears, press ENTER to use existing calibration.")
        robot.connect()
        if leader is not None:
            leader.connect()
            teleop_thread = threading.Thread(
                target=manual_teleop_worker,
                name="manual-correction-teleop",
                daemon=True,
                args=(robot, leader, robot_lock, manual_enabled, stop_teleop, teleop_shared, args),
            )
            teleop_thread.start()

        while True:
            loop_start = time.perf_counter()
            manual_correction = manual_enabled.is_set()
            alignment_running = bool(alignment_state.get("running"))
            if alignment_running:
                elapsed_align = time.perf_counter() - float(alignment_state.get("started_at") or time.perf_counter())
                status = f"{alignment_state.get('status')}; t={elapsed_align:.1f}s"
            elif before_marker_state.get("running"):
                elapsed_before = time.perf_counter() - float(before_marker_state.get("started_at") or time.perf_counter())
                status = f"{before_marker_state.get('status')}; t={elapsed_before:.1f}s"
            elif alignment_state.get("status"):
                status = str(alignment_state.get("status"))
                alignment_state["status"] = ""
            elif before_marker_state.get("status"):
                status = str(before_marker_state.get("status"))
                before_marker_state["status"] = ""
            if teleop_shared.get("error"):
                status = str(teleop_shared.get("error"))
            if alignment_running:
                shared_obs = dict(teleop_shared.get("latest_obs") or {})
                if shared_obs:
                    latest_obs = shared_obs
            elif manual_correction:
                latest_obs = dict(teleop_shared.get("latest_obs") or {})
                if not latest_obs:
                    latest_obs = locked_observation(robot, robot_lock)
            else:
                latest_obs = locked_observation(robot, robot_lock)

            if pending_camera_reopen_reason is not None and not alignment_running:
                reopen_reason = pending_camera_reopen_reason
                pending_camera_reopen_reason = None
                try:
                    status = reopen_camera_pair(reopen_reason)
                    gripper_camera_fault = ""
                except Exception as exc:  # noqa: BLE001
                    gripper_capture = None
                    clear_gripper_cache()
                    gripper_camera_fault = f"ERROR: camera pair reopen failed: {exc}"
                    status = gripper_camera_fault
                print(status, flush=True)

            record_now = time.perf_counter()
            if hover_recording:
                record_dt = 1.0 / max(1.0, float(args.hover_height_record_hz))
                if record_now - last_hover_record_time >= record_dt:
                    recorded_obs = locked_observation(robot, robot_lock)
                    hover_record_samples.append(
                        {
                            "captured_at": now_iso(),
                            "measurement": "encoder_observation",
                            "joints": {
                                joint: float(recorded_obs[joint])
                                for joint in MOTOR_KEYS
                                if joint in recorded_obs
                            },
                        }
                    )
                    last_hover_record_time = record_now

            with camera_lock:
                ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Camera read failed")
            gripper_frame = None
            if gripper_capture is not None:
                with gripper_camera_lock:
                    ok_g, gripper_frame = gripper_capture.read()
                if not ok_g or gripper_frame is None:
                    gripper_frame = None
                else:
                    gripper_frame = transform_gripper_frame(gripper_frame, args)
                    if args.duplicate_camera_check and frames_look_duplicate(
                        frame,
                        gripper_frame,
                        float(args.duplicate_camera_threshold),
                    ):
                        duplicate_camera_count += 1
                        gripper_frame = None
                        clear_gripper_cache()
                        gripper_camera_fault = (
                            f"WARNING: gripper camera looks identical to overhead "
                            f"({duplicate_camera_count}/{int(args.duplicate_camera_frames)} frames)"
                        )
                        if duplicate_camera_count >= max(1, int(args.duplicate_camera_frames)):
                            if not bool(getattr(args, "duplicate_camera_auto_reopen", False)):
                                if alignment_running:
                                    alignment_cancel.set()
                                    gripper_camera_fault = (
                                        "WARNING: duplicate stream confirmed; alignment stop requested. "
                                        "Automatic camera reopen is disabled; press R when idle or restart the script."
                                    )
                                else:
                                    gripper_camera_fault = (
                                        "WARNING: duplicate stream confirmed. Automatic camera reopen is disabled; "
                                        "press R to reopen both cameras, or restart the script if Windows bound the wrong device."
                                    )
                            elif alignment_running:
                                alignment_cancel.set()
                                pending_camera_reopen_reason = "duplicate stream detected during alignment"
                                gripper_camera_fault = (
                                    "WARNING: duplicate stream confirmed; alignment stop requested before camera pair reopen"
                                )
                            else:
                                print(
                                    "WARNING: gripper camera stream duplicated overhead; reopening both cameras",
                                    flush=True,
                                )
                                try:
                                    gripper_camera_fault = reopen_camera_pair("duplicate stream detection")
                                except Exception as exc:  # noqa: BLE001
                                    gripper_capture = None
                                    clear_gripper_cache()
                                    gripper_camera_fault = f"ERROR: camera pair reopen failed: {exc}"
                                    print(gripper_camera_fault, flush=True)
                    else:
                        duplicate_camera_count = 0
                        gripper_camera_fault = ""
            now = time.perf_counter()
            inference_interval = args.manual_inference_interval_s if manual_correction else args.preview_inference_interval_s
            if cached_det is None or (now - last_predict_time) >= max(0.0, float(inference_interval)):
                result = run_predict(model, frame, args)
                cached_det = select_detection(result, names, args.target_class, args.conf, args)
                last_predict_time = now
            if (
                gripper_frame is not None
                and gripper_model is not None
                and (cached_gripper_det is None or (now - last_gripper_predict_time) >= max(0.0, float(inference_interval)))
            ):
                gripper_result = run_predict(gripper_model, gripper_frame, args)
                cached_gripper_det = select_detection(gripper_result, gripper_names, args.target_class, args.conf, args)
                cached_gripper_frame = gripper_frame
                last_gripper_predict_time = now
                if cached_gripper_det is not None:
                    last_gripper_det = cached_gripper_det
                    last_gripper_det_time = time.perf_counter()
            elif gripper_frame is not None:
                cached_gripper_frame = gripper_frame
            det = cached_det
            if det is not None:
                last_det = det
                last_det_time = time.perf_counter()

            if act_recording:
                act_dt = 1.0 / max(1.0, float(args.act_record_fps))
                if now - act_record_last_time >= act_dt:
                    record_obs = dict(teleop_shared.get("latest_performed") or teleop_shared.get("latest_obs") or {})
                    record_action = dict(teleop_shared.get("latest_action") or teleop_shared.get("latest_performed") or {})
                    ok_record, act_record_status = add_act_record_frame(
                        act_dataset,
                        frame,
                        cached_gripper_frame,
                        record_obs,
                        record_action,
                        args,
                    )
                    if ok_record:
                        act_record_last_time = now

            if pending_before_after_alignment and not alignment_running:
                pending_before_after_alignment = False
                pending_manual_after_alignment = False
                locked_before_uv = alignment_state.get("locked_overhead_uv")
                start_before_marker_worker(
                    reason="after alignment stop",
                    fallback_det=det,
                    fallback_last_det=last_det,
                    fallback_gripper_det=cached_gripper_det,
                    fallback_last_gripper_det=last_gripper_det,
                    locked_overhead_uv=(
                        (float(locked_before_uv[0]), float(locked_before_uv[1]))
                        if locked_before_uv is not None
                        else None
                    ),
                )
                status = str(before_marker_state.get("status") or "BEFORE marker capture started")
                print(status, flush=True)
            elif pending_manual_after_alignment and not alignment_running:
                pending_manual_after_alignment = False
                status = enable_manual_correction("auto-opened after alignment stop")
                print(status, flush=True)

            mapped_text = "target pose: n/a"
            if det is not None and calib.pose_model:
                pose_target = predict_pose_target(det.center_uv[0], det.center_uv[1], calib, args)
                nearest_dist = nearest_matching_point_distance(det.center_uv[0], det.center_uv[1], calib, args)
                snap_text = "nearest=n/a"
                if nearest_dist is not None:
                    snap_text = f"nearest={nearest_dist:.1f}px snap<={float(args.local_snap_px):.1f}px"
                if pose_target and alignment_pose_mode(calib, args) == "xy_hover":
                    xy_target = {joint: float(pose_target[joint]) for joint in args.map_joints if joint in pose_target}
                    raw_hover_target = predict_hover_height_joints(xy_target, calib, args) if len(xy_target) == len(args.map_joints) else {}
                    hover_target, hover_clamped, hover_reason = (
                        clamp_to_limits(raw_hover_target, calib.hover_height_limits)
                        if raw_hover_target
                        else ({}, False, "hover target n/a")
                    )
                    orientation_target = predict_xy_hover_orientation_joints(xy_target, hover_target, calib, args) if hover_target else {}
                    xy_limits = {joint: calib.pose_limits[joint] for joint in xy_target if joint in calib.pose_limits}
                    ok_xy, xy_reason = within_limits(xy_target, xy_limits) if xy_target else (False, "xy target n/a")
                    ok_hover = bool(hover_target)
                    ok_orient, orient_reason = (
                        within_orientation_limits(orientation_target, calib.orientation_limits)
                        if orientation_target
                        else (False, "orientation target n/a")
                    )
                    xy_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in xy_target.items())
                    hover_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in hover_target.items())
                    orient_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in orientation_target.items())
                    mapped_text = (
                        f"xy_hover xy={ok_xy} {xy_reason}; hover={ok_hover} {hover_reason}; "
                        f"orient={ok_orient} {orient_reason}; {snap_text}; {xy_short}; {hover_short}; {orient_short}"
                    )
                else:
                    ok_pose, pose_reason = within_limits(pose_target, calib.pose_limits) if pose_target else (False, "pose target n/a")
                    pose_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in pose_target.items())
                    mapped_text = f"full_pose limits={ok_pose} {pose_reason}; {snap_text}; {pose_short}"
            elif det is not None and calib.homography is not None:
                j0, j1 = pixel_to_joint_target(calib.homography, det.center_uv[0], det.center_uv[1])
                mapped = {args.map_joints[0]: j0, args.map_joints[1]: j1}
                ok_limits, reason = within_limits(mapped, calib.joint_limits)
                orient = predict_orientation_joints(mapped, calib.orientation_model, args)
                ok_orient, orient_reason = within_orientation_limits(orient, calib.orientation_limits) if orient else (False, "orientation n/a")
                orient_short = " ".join(f"{joint.split('.')[0]}={value:.1f}" for joint, value in orient.items())
                mapped_text = (
                    f"mapped {args.map_joints[0]}={j0:.2f} {args.map_joints[1]}={j1:.2f} "
                    f"pos_limits={ok_limits} orient_limits={ok_orient} {reason}; {orient_reason}; {orient_short}"
                )

            safe_state = "yes" if calib.safe_pose is not None else "no"
            pose_state = "yes" if calib.pose_model else "no"
            orient_state = "yes" if calib.orientation_model else "no"
            h_state = "yes" if calib.homography is not None else "no"
            active_kind = active_target_point_kind(args.target_class, args)
            matching_points = len(matching_calibration_points(calib.points, args))
            gripper_ref_count = count_gripper_reference_points(calib, args)
            manual_demo_count = len(calib.gripper_correction_samples)
            gripper_motion_count = len(calib.gripper_motion_samples)
            correction_state = "manual=on" if manual_correction else "manual=off"
            if not args.enable_manual_correction:
                correction_state = "manual=disabled"
            elif manual_correction:
                correction_state = f"manual=on teleop_fps={float(teleop_shared.get('teleop_fps') or 0.0):.1f}"
            hover_record_state = f"height_rec={'on' if hover_recording else 'off'}"
            if hover_recording:
                hover_record_state += f" samples={len(hover_record_samples)}"
            act_record_state = f"act_rec={'on' if act_recording else 'off'}"
            if act_recording and act_dataset is not None:
                act_record_state += f" frames={act_record_dataset_episode_size(act_dataset)}"
            lines = [
                f"ALIGN target={args.target_class} point={active_kind} mode={alignment_pose_mode(calib, args)} predictor={args.pose_predictor} points={matching_points}/{len(calib.points)} refs={gripper_ref_count}",
                f"controller={args.gripper_servo_mode} ibvs_samples={gripper_motion_count} manual_demos={manual_demo_count} H={h_state} safe={safe_state} pose={pose_state} hover={'yes' if calib.hover_height_model else 'no'} orient={orient_state} dry_run={args.dry_run}",
                f"{pose_source}; {hover_source}; {orientation_source}; {param_source}",
                mapped_text,
                status,
                f"{correction_state} | {hover_record_state} | {act_record_state} | A align | N handoff | G teleop | D ACT rec | C params | Q stop/quit",
            ]
            if gripper_camera_fault:
                lines.insert(3, gripper_camera_fault)
            annotated = draw_detection(frame, det, lines)
            if gripper_capture is not None:
                gripper_ref_uv = None
                gripper_ref_radius = 0.0
                gripper_ref_source = "reference n/a"
                if cached_gripper_frame is not None:
                    locked_overhead_uv = alignment_state.get("locked_overhead_uv")
                    if locked_overhead_uv is not None:
                        overhead_uv = (float(locked_overhead_uv[0]), float(locked_overhead_uv[1]))
                    else:
                        overhead_uv = (float(det.center_uv[0]), float(det.center_uv[1])) if det is not None else None
                    gripper_ref_uv, gripper_ref_radius, gripper_ref_source = gripper_reference_circle(
                        calib,
                        args,
                        cached_gripper_frame.shape,
                        overhead_uv,
                    )
                gripper_lines = [
                    f"GRIPPER cam={args.gripper_camera_index} model={Path(args.gripper_model).name}",
                    f"stage 3 final XY only; ref={gripper_ref_source}",
                ]
                if alignment_state.get("locked_overhead_uv") is not None:
                    locked_overhead_uv = alignment_state["locked_overhead_uv"]
                    gripper_lines.append(
                        f"locked overhead=({float(locked_overhead_uv[0]):.1f},{float(locked_overhead_uv[1]):.1f})"
                    )
                if gripper_camera_fault:
                    gripper_lines.append(gripper_camera_fault)
                if cached_gripper_frame is None:
                    gripper_image = np.zeros_like(frame)
                    gripper_lines.append("gripper frame: n/a")
                else:
                    gripper_image = cached_gripper_frame
                if cached_gripper_det is not None and cached_gripper_frame is not None:
                    if gripper_ref_uv is None:
                        err_u = 0.0
                        err_v = 0.0
                    else:
                        err_u = cached_gripper_det.center_uv[0] - gripper_ref_uv[0]
                        err_v = cached_gripper_det.center_uv[1] - gripper_ref_uv[1]
                    gripper_lines.append(
                        f"selected {cached_gripper_det.class_name} conf={cached_gripper_det.confidence:.2f} "
                        f"err=({err_u:.0f},{err_v:.0f})px"
                    )
                    mask_features = mask_shape_features(cached_gripper_det.mask, cached_gripper_det.xyxy)
                    gripper_lines.append(
                        f"mask area={mask_features['mask_area_px']:.0f} sqrt={mask_features['mask_sqrt_area_px']:.1f} "
                        f"axis={mask_features['mask_axis_ratio']:.2f} angle={mask_features['mask_angle_deg']:.1f}"
                    )
                    gripper_lines.append(
                        f"shape circ={mask_features['mask_circularity']:.2f} sol={mask_features['mask_solidity']:.2f} "
                        f"bbox={mask_features['mask_bbox_w_px']:.0f}x{mask_features['mask_bbox_h_px']:.0f}"
                    )
                else:
                    gripper_lines.append("selected: none")
                gripper_lines.append("A works in this guided-align window; preview-only dualcam has q/s only.")
                annotated_gripper = draw_detection(gripper_image, cached_gripper_det, gripper_lines)
                draw_reference_marker(annotated_gripper, gripper_ref_uv, radius_px=gripper_ref_radius)
                annotated = compose_side_by_side(annotated, annotated_gripper)
            key = -1 if preview is None else preview.show(annotated)
            key = normalize_key(key)
            alignment_running = bool(alignment_state.get("running"))

            if key == ord("q"):
                if alignment_running:
                    alignment_cancel.set()
                    status = "alignment stop requested; waiting for current motion/refine step to finish"
                    print(status, flush=True)
                    continue
                if act_recording:
                    status = "ACT recording is running; press d to stop/save before quitting"
                    print(status, flush=True)
                    continue
                if hover_recording:
                    status = "height recording is running; press t to stop/save before quitting"
                    print(status, flush=True)
                    continue
                break
            if key == ord("h"):
                align_help()
            if key == ord("r"):
                if not args.enable_gripper_final_align:
                    status = "cannot reopen gripper camera: final-align camera is not enabled"
                elif alignment_running:
                    alignment_cancel.set()
                    pending_camera_reopen_reason = "manual R key"
                    status = "alignment stop requested; both camera streams will reopen on the main thread"
                else:
                    try:
                        status = reopen_camera_pair("manual R key")
                    except Exception as exc:  # noqa: BLE001
                        gripper_capture = None
                        clear_gripper_cache()
                        status = f"ERROR: manual camera pair reopen failed: {exc}"
                print(status, flush=True)
            nudge_status = handle_target_nudge(key, args)
            if nudge_status is not None:
                if alignment_running:
                    status = "alignment is running; target nudge ignored"
                    print(status, flush=True)
                    continue
                cached_det = None
                last_predict_time = 0.0
                pose_source = ensure_pose_model(calib, args)
                orientation_source = ensure_orientation_model(calib, args)
                param_source = f"target params: {nudge_status}"
                status = f"{nudge_status}; save corrections with this offset"
                print(status, flush=True)
            if key == ord("t"):
                if alignment_running:
                    status = "alignment is running; height recording waits until it finishes"
                    print(status, flush=True)
                    continue
                if act_recording:
                    status = "ACT recording is running; press d to stop/save before height recording"
                    print(status, flush=True)
                    continue
                if leader is None:
                    status = "height recording needs the leader; use 视觉对齐与手动修正.bat"
                    print(status, flush=True)
                    continue
                if not hover_recording:
                    hover_record_samples = []
                    last_hover_record_time = 0.0
                    hover_recording = True
                    if not manual_enabled.is_set():
                        teleop_shared["error"] = ""
                        manual_enabled.set()
                    status = "height recording ON: move the gripper around at cube-hover height, then press t again"
                else:
                    hover_recording = False
                    if hover_record_samples:
                        calib.hover_height_samples = list(hover_record_samples)
                        hover_source = recompute_hover_height_model(calib, args)
                        save_calibration(config_path, calib, args)
                        status = f"height recording saved {len(hover_record_samples)} samples; {hover_source}"
                    else:
                        status = "height recording stopped: no samples captured"
                print(status, flush=True)
            if key == ord("d"):
                if alignment_running:
                    status = "alignment is running; ACT recording waits until it finishes"
                    print(status, flush=True)
                    continue
                if before_marker_state.get("running"):
                    status = "BEFORE marker capture is running; wait for it to finish before ACT recording"
                    print(status, flush=True)
                    continue
                if hover_recording:
                    status = "height recording is running; press t to stop/save before ACT recording"
                    print(status, flush=True)
                    continue
                if leader is None:
                    status = "ACT recording needs the leader; restart with --enable-manual-correction"
                    print(status, flush=True)
                    continue
                if act_recording:
                    act_recording = False
                    status = "ACT recording stopping; saving episode..."
                    print(status, flush=True)
                    try:
                        status = save_act_record_episode(act_dataset, args)
                    except Exception as exc:  # noqa: BLE001
                        status = f"ACT episode save failed: {exc}"
                    act_record_status = status
                    print(status, flush=True)
                    continue
                if cached_gripper_frame is None:
                    status = "ACT recording needs a live gripper/front camera frame; wait for the gripper view and press d again"
                    print(status, flush=True)
                    continue
                if bool(leader_sync_hold.get("active")):
                    release_status = release_leader_sync_hold(leader, leader_sync_hold)
                    print(release_status, flush=True)
                if not manual_enabled.is_set():
                    status = enable_manual_correction("ACT local grasp recording")
                    print(status, flush=True)
                    time.sleep(1.0 / max(1.0, float(args.manual_teleop_fps)))
                try:
                    if act_dataset is None:
                        act_dataset = load_or_create_act_record_dataset(args)
                    elif getattr(act_dataset, "episode_buffer", None) is None:
                        act_dataset.episode_buffer = act_dataset.create_episode_buffer()
                    elif act_record_dataset_episode_size(act_dataset) > 0:
                        act_dataset.clear_episode_buffer()
                    act_record_last_time = 0.0
                    act_recording = True
                    act_record_status = "ACT recording ON"
                    status = f"ACT recording ON: press d again to save; root={Path(args.act_record_root)}"
                except Exception as exc:  # noqa: BLE001
                    act_recording = False
                    status = f"ACT recording start failed: {exc}"
                    act_record_status = status
                print(status, flush=True)
            if key == ord("m"):
                if alignment_running:
                    status = request_alignment_stop_for_manual("M pressed")
                    print(status, flush=True)
                    continue
                if act_recording:
                    status = "ACT recording is running; press d to stop/save before toggling manual teleop"
                    print(status, flush=True)
                    continue
                if hover_recording:
                    status = "height recording is running; keep manual teleop on and press t to stop/save"
                    print(status, flush=True)
                    continue
                if leader is None:
                    status = "manual correction disabled; restart with --enable-manual-correction"
                    print(status, flush=True)
                else:
                    if manual_enabled.is_set():
                        manual_enabled.clear()
                        manual_correction = False
                    else:
                        status = enable_manual_correction("use leader, then press C to save")
                        print(status, flush=True)
                        continue
                    status = "manual correction OFF"
                    print(status, flush=True)
            if key == ord("g"):
                if alignment_running:
                    status = "alignment is running; leader handoff waits until it finishes"
                    print(status, flush=True)
                    continue
                if leader is None:
                    status = "leader handoff needs the leader; restart with --enable-manual-correction"
                    print(status, flush=True)
                    continue
                release_status = release_leader_sync_hold(leader, leader_sync_hold)
                status = enable_manual_correction("leader sync handoff; start local grasp teleop")
                print(release_status + "; " + status, flush=True)
                continue
            if key == ord("n"):
                if alignment_running:
                    status = "alignment is running; leader sync waits until it finishes"
                    print(status, flush=True)
                    continue
                if before_marker_state.get("running"):
                    status = "BEFORE marker capture is running; wait for it to finish before leader sync"
                    print(status, flush=True)
                    continue
                if hover_recording:
                    status = "height recording is running; press t to stop/save before leader sync"
                    print(status, flush=True)
                    continue
                if leader is None:
                    status = "leader sync needs the leader; restart with --enable-manual-correction"
                    print(status, flush=True)
                    continue
                if manual_enabled.is_set():
                    manual_enabled.clear()
                    manual_correction = False
                    time.sleep(1.0 / max(1.0, float(args.manual_teleop_fps)))
                if bool(leader_sync_hold.get("active")):
                    release_status = release_leader_sync_hold(leader, leader_sync_hold)
                    print(release_status, flush=True)
                gripper_status = open_follower_gripper_for_handoff(
                    robot,
                    args,
                    robot_lock,
                    stop_event=alignment_cancel,
                )
                print(gripper_status, flush=True)
                follower_pose = locked_observation(robot, robot_lock)
                status = "leader sync requested; moving leader to current follower pose..."
                print(status, flush=True)
                status = sync_leader_to_follower_pose(
                    leader,
                    follower_pose,
                    args,
                    stop_event=alignment_cancel,
                )
                if status.startswith("leader sync done") and bool(getattr(args, "leader_sync_hold_torque", True)):
                    held_motors = [
                        leader_motor_name(joint)
                        for joint in list(getattr(args, "leader_sync_joints", []) or [])
                        if joint.endswith(".pos")
                        and joint in follower_pose
                        and leader_motor_name(joint) in leader.bus.motors
                    ]
                    if bool(getattr(args, "leader_sync_include_gripper", False)) and "gripper" not in held_motors:
                        held_motors.append("gripper")
                    leader_sync_hold["active"] = True
                    leader_sync_hold["motors"] = held_motors
                print(status, flush=True)
            if key == ord("a"):
                if alignment_running:
                    status = "alignment already running; wait for DONE/STOP"
                    print(status, flush=True)
                    continue
                if act_recording:
                    status = "ACT recording is running; press d to stop/save before alignment"
                    print(status, flush=True)
                    continue
                if before_marker_state.get("running"):
                    status = "BEFORE marker capture is running; wait for it to finish before A"
                    print(status, flush=True)
                    continue
                if hover_recording:
                    status = "height recording is running; press t to stop/save before alignment"
                    print(status, flush=True)
                    continue
                if manual_enabled.is_set():
                    manual_enabled.clear()
                    manual_correction = False
                    time.sleep(1.0 / max(1.0, float(args.manual_teleop_fps)))
                status = "alignment requested; checking safety gates..."
                print(status, flush=True)
                start_alignment_worker()
            if key == ord("p"):
                if alignment_running:
                    status = "alignment is running; direct replay waits until it finishes"
                    print(status, flush=True)
                    continue
                if act_recording:
                    status = "ACT recording is running; press d to stop/save before direct replay"
                    print(status, flush=True)
                    continue
                if before_marker_state.get("running"):
                    status = "BEFORE marker capture is running; wait for it to finish before P"
                    print(status, flush=True)
                    continue
                if hover_recording:
                    status = "height recording is running; press t to stop/save before direct replay"
                    print(status, flush=True)
                    continue
                if manual_enabled.is_set():
                    manual_enabled.clear()
                    manual_correction = False
                    time.sleep(1.0 / max(1.0, float(args.manual_teleop_fps)))
                status = "direct replay requested; checking saved point in background..."
                print(status, flush=True)
                start_direct_replay_worker()
            if key in (ord("u"), 8):
                if alignment_running:
                    status = "alignment is running; undo waits until it finishes"
                    print(status, flush=True)
                    continue
                if act_recording:
                    status = "ACT recording is running; press d to stop/save before undo"
                    print(status, flush=True)
                    continue
                if before_marker_state.get("running"):
                    status = "BEFORE marker capture is running; wait for it to finish before undo"
                    print(status, flush=True)
                    continue
                if hover_recording:
                    status = "height recording is running; press t to stop/save before undo"
                    print(status, flush=True)
                    continue
                if calib.points:
                    removed = calib.points.pop()
                    removed_id = int(removed.get("id") or 0)
                    before_sample_count = len(calib.gripper_correction_samples)
                    if removed_id:
                        calib.gripper_correction_samples = [
                            sample
                            for sample in calib.gripper_correction_samples
                            if int(sample.get("after_point_id") or 0) != removed_id
                        ]
                    replay_reset_text = ""
                    if removed_id and int(getattr(args, "direct_replay_point_id", 0) or 0) == removed_id:
                        setattr(args, "direct_replay_point_id", 0)
                        replay_reset_text = "; P target reset to latest saved point"
                    removed_sample_count = before_sample_count - len(calib.gripper_correction_samples)
                    recompute_homography(calib, args)
                    save_calibration(config_path, calib, args)
                    pose_source = ensure_pose_model(calib, args)
                    orientation_source = ensure_orientation_model(calib, args)
                    sample_text = "" if removed_sample_count <= 0 else f"; removed linked correction_samples={removed_sample_count}"
                    status = f"removed point {removed.get('id')}; points={len(calib.points)}{sample_text}{replay_reset_text}"
                else:
                    status = "no point to undo"
                print(status, flush=True)
            if key == ord("b"):
                if alignment_running:
                    alignment_cancel.set()
                    pending_manual_after_alignment = True
                    pending_before_after_alignment = True
                    status = (
                        "alignment stop requested for BEFORE marker; "
                        "will mark current error pose after the current motion step stops"
                    )
                    print(status, flush=True)
                    continue
                if act_recording:
                    status = "ACT recording is running; press d to stop/save before before marker"
                    print(status, flush=True)
                    continue
                if before_marker_state.get("running"):
                    status = "BEFORE marker capture is already running; preview remains live"
                    print(status, flush=True)
                    continue
                if hover_recording:
                    status = "height recording is running; press t to stop/save before before marker"
                    print(status, flush=True)
                    continue
                locked_before_uv = alignment_state.get("locked_overhead_uv")
                start_before_marker_worker(
                    reason="B pressed",
                    fallback_det=det,
                    fallback_last_det=last_det,
                    fallback_gripper_det=cached_gripper_det,
                    fallback_last_gripper_det=last_gripper_det,
                    locked_overhead_uv=(
                        (float(locked_before_uv[0]), float(locked_before_uv[1]))
                        if locked_before_uv is not None
                        else None
                    ),
                )
                status = str(before_marker_state.get("status") or "BEFORE marker capture started")
                print(status, flush=True)
            if key == ord("v"):
                status = "V verified coordinate save is disabled; auto DONE now updates compact learned parameters"
                print(status, flush=True)
                continue
                if alignment_running:
                    status = "alignment is running; verified save waits until it finishes"
                    print(status, flush=True)
                    continue
                if before_marker_state.get("running"):
                    status = "BEFORE marker capture is running; wait for it to finish before V"
                    print(status, flush=True)
                    continue
                if hover_recording:
                    status = "height recording is running; press t to stop/save before verified save"
                    print(status, flush=True)
                    continue
                manual_enabled.clear()
                manual_correction = False
                stable_after, saved_obs, stable_status = wait_until_observation_stable(
                    robot,
                    list(args.pose_joints),
                    float(args.manual_save_stable_timeout_s),
                    float(args.gripper_refine_stable_epsilon_deg),
                    int(args.gripper_refine_stable_samples),
                    robot_lock,
                )
                if not stable_after:
                    status = "cannot save verified alignment: arm did not become stable; " + stable_status
                    print(status, flush=True)
                    continue
                fresh_det, fresh_gripper_det, fresh_status = fresh_correction_detections()
                verified_det = fresh_det if fresh_det is not None else det
                locked_uv = alignment_state.get("locked_overhead_uv")
                if verified_det is not None and locked_uv is not None:
                    verified_det = replace(
                        verified_det,
                        center_uv=(float(locked_uv[0]), float(locked_uv[1])),
                    )
                if verified_det is None and last_det is not None:
                    age = time.perf_counter() - last_det_time
                    if age <= args.correction_max_target_age_s:
                        verified_det = last_det
                if verified_det is None:
                    status = "cannot save verified alignment: target not visible and no recent target cached"
                    print(status, flush=True)
                    continue
                if args.enable_gripper_final_align and fresh_gripper_det is None:
                    status = "cannot save verified alignment: gripper camera target not visible"
                    print(status, flush=True)
                    continue
                ok_capture, status = append_calibration_point(
                    calib,
                    verified_det,
                    saved_obs,
                    args,
                    source="human_verified_alignment",
                    gripper_det=fresh_gripper_det,
                )
                status += "; " + stable_status + "; " + fresh_status
                if ok_capture:
                    calib.points[-1]["verified"] = True
                    calib.points[-1]["controller_version"] = str(args.gripper_servo_mode)
                    if fresh_gripper_det is not None:
                        ref_uv, _ref_source = gripper_reference_pixel(
                            calib,
                            args,
                            cached_gripper_frame.shape if cached_gripper_frame is not None else None,
                            tuple(verified_det.center_uv),
                        )
                        if ref_uv is not None:
                            calib.points[-1]["final_gripper_error_px"] = float(
                                math.hypot(
                                    float(fresh_gripper_det.center_uv[0]) - float(ref_uv[0]),
                                    float(fresh_gripper_det.center_uv[1]) - float(ref_uv[1]),
                                )
                            )
                    before_sample = alignment_state.get("pre_refine_before")
                    if isinstance(before_sample, dict) and fresh_gripper_det is not None:
                        ok_sample, sample_status = append_gripper_correction_sample(
                            calib,
                            before_sample,
                            verified_det,
                            fresh_gripper_det,
                            saved_obs,
                            args,
                        )
                        if ok_sample and calib.gripper_correction_samples:
                            calib.gripper_correction_samples[-1]["after_point_id"] = int(calib.points[-1].get("id") or 0)
                        status += "; " + sample_status
                        append_manual_correction_trace(
                            args,
                            before_sample,
                            verified_det,
                            fresh_gripper_det,
                            saved_obs,
                            sample_status,
                        )
                        alignment_state["pre_refine_before"] = None
                    save_calibration(config_path, calib, args)
                    pose_source = ensure_pose_model(calib, args)
                    orientation_source = ensure_orientation_model(calib, args)
                    status += "; verified point saved" + set_direct_replay_point_to_saved_point(calib.points[-1], args) + "; " + pose_source
                print(status, flush=True)
            if key == ord("c"):
                if alignment_running:
                    status = "alignment is running; correction save waits until it finishes"
                    print(status, flush=True)
                    continue
                if act_recording:
                    status = "ACT recording is running; press d to stop/save before correction save"
                    print(status, flush=True)
                    continue
                if before_marker_state.get("running"):
                    status = "BEFORE marker capture is running; wait for it to finish before C"
                    print(status, flush=True)
                    continue
                if hover_recording:
                    status = "height recording is running; press t to stop/save before correction save"
                    print(status, flush=True)
                    continue
                manual_enabled.clear()
                manual_correction = False
                time.sleep(1.0 / max(1.0, float(args.manual_teleop_fps)))
                stable_after, saved_obs, stable_status = wait_until_observation_stable(
                    robot,
                    list(args.pose_joints),
                    float(args.manual_save_stable_timeout_s),
                    float(args.gripper_refine_stable_epsilon_deg),
                    int(args.gripper_refine_stable_samples),
                    robot_lock,
                )
                if not stable_after:
                    status = "cannot save correction: arm did not become stable after teleop stopped; " + stable_status
                    print(status, flush=True)
                    continue
                fresh_det, fresh_gripper_det, fresh_status = fresh_correction_detections()
                correction_det = fresh_det if fresh_det is not None else det
                if correction_det is None and gripper_correction_before is not None:
                    correction_det = gripper_correction_before.get("overhead_detection")
                if correction_det is None and last_det is not None:
                    age = time.perf_counter() - last_det_time
                    if age <= args.correction_max_target_age_s:
                        correction_det = last_det
                        status = f"using last visible target from {age:.1f}s ago for correction"
                        print(status, flush=True)
                if correction_det is not None and gripper_correction_before is not None:
                    before_overhead = gripper_correction_before.get("overhead_pixel")
                    if before_overhead is not None and len(before_overhead) == 2:
                        correction_det = replace(
                            correction_det,
                            center_uv=(float(before_overhead[0]), float(before_overhead[1])),
                        )
                if correction_det is None:
                    status = "cannot save correction: target not visible and no recent target cached"
                    print(status, flush=True)
                elif not saved_obs:
                    status = "cannot save correction: follower observation missing"
                    print(status, flush=True)
                else:
                    correction_gripper_det = fresh_gripper_det
                    if (
                        args.enable_gripper_final_align
                        and args.require_gripper_detection_on_correction
                        and correction_gripper_det is None
                    ):
                        status = "cannot save correction: gripper camera target not visible; move until gripper view detects it"
                        print(status, flush=True)
                        continue
                    if (
                        correction_gripper_det is not None
                        and float(correction_gripper_det.confidence) < float(args.manual_save_min_gripper_conf)
                    ):
                        status = (
                            f"cannot save correction: fresh gripper confidence "
                            f"{float(correction_gripper_det.confidence):.2f} "
                            f"< {float(args.manual_save_min_gripper_conf):.2f}"
                        )
                        print(status, flush=True)
                        continue
                    if bool(getattr(args, "gripper_learned_param_only", False)):
                        auto_pre_sample = alignment_state.get("pre_refine_before")
                        auto_post_sample = alignment_state.get("post_refine_after")
                        manual_marker_sample = gripper_correction_before
                        before_sample = (
                            auto_pre_sample
                            if isinstance(auto_pre_sample, dict)
                            else manual_marker_sample
                        )
                        if not isinstance(before_sample, dict):
                            status = "cannot update compact params: press B before C so a BEFORE pose exists"
                            print(status, flush=True)
                            continue
                        if correction_gripper_det is None:
                            status = "cannot update compact params: gripper camera target not visible"
                            print(status, flush=True)
                            continue
                        before_joints = before_sample.get("joints", {})
                        before_gripper = before_sample.get("gripper_pixel")
                        if (
                            not isinstance(before_joints, dict)
                            or before_gripper is None
                            or len(before_gripper) != 2
                        ):
                            status = "cannot update compact params: BEFORE sample lacks joints/gripper pixel"
                            print(status, flush=True)
                            continue
                        if not all(joint in before_joints and joint in saved_obs for joint in args.pose_joints):
                            status = "cannot update compact params: before/after pose joints missing"
                            print(status, flush=True)
                            continue
                        ref_uv, ref_source = gripper_reference_pixel(
                            calib,
                            args,
                            cached_gripper_frame.shape if cached_gripper_frame is not None else None,
                            tuple(correction_det.center_uv),
                        )
                        if ref_uv is None:
                            status = "cannot update compact params: gripper reference unavailable"
                            print(status, flush=True)
                            continue
                        error_uv = (
                            float(before_gripper[0]) - float(correction_gripper_det.center_uv[0]),
                            float(before_gripper[1]) - float(correction_gripper_det.center_uv[1]),
                        )
                        before_dist_px = math.hypot(
                            float(before_gripper[0]) - float(ref_uv[0]),
                            float(before_gripper[1]) - float(ref_uv[1]),
                        )
                        after_dist_px = math.hypot(
                            float(correction_gripper_det.center_uv[0]) - float(ref_uv[0]),
                            float(correction_gripper_det.center_uv[1]) - float(ref_uv[1]),
                        )
                        delta_pose = {
                            joint: float(saved_obs[joint]) - float(before_joints[joint])
                            for joint in args.pose_joints
                        }
                        ok_param, param_status = update_gripper_learned_param_model(
                            calib,
                            args,
                            error_uv=error_uv,
                            delta_pose=delta_pose,
                            before_dist_px=before_dist_px,
                            after_dist_px=after_dist_px,
                            source="manual_correction_param_only",
                            overhead_uv=tuple(correction_det.center_uv),
                        )
                        status = (
                            f"compact correction params {'updated' if ok_param else 'not updated'}; "
                            f"before_dist={before_dist_px:.1f}px after_dist={after_dist_px:.1f}px; "
                            f"{param_status}; {stable_status}; {fresh_status}; ref={ref_source}; manual correction OFF"
                        )
                        gripper_correction_before = None
                        alignment_state["pre_refine_before"] = None
                        alignment_state["post_refine_after"] = None
                        save_calibration(config_path, calib, args)
                        print(status, flush=True)
                        continue
                    ok_capture, status = append_calibration_point(
                        calib,
                        correction_det,
                        saved_obs,
                        args,
                        source="manual_correction",
                        gripper_det=correction_gripper_det,
                    )
                    status += "; " + stable_status + "; " + fresh_status + "; manual correction OFF"
                    if ok_capture:
                        extra_status = ""
                        auto_pre_sample = alignment_state.get("pre_refine_before")
                        auto_post_sample = alignment_state.get("post_refine_after")
                        manual_marker_sample = gripper_correction_before
                        before_sample = (
                            auto_pre_sample
                            if isinstance(auto_pre_sample, dict)
                            else manual_marker_sample
                        )
                        attempted_refine_sample = None
                        if isinstance(auto_pre_sample, dict):
                            if isinstance(manual_marker_sample, dict):
                                attempted_refine_sample = manual_marker_sample
                            elif isinstance(auto_post_sample, dict):
                                attempted_refine_sample = auto_post_sample
                        if (
                            before_sample is not None
                            and correction_gripper_det is not None
                        ):
                            ok_sample, sample_status = append_gripper_correction_sample(
                                calib,
                                before_sample,
                                correction_det,
                                correction_gripper_det,
                                saved_obs,
                                args,
                                attempted_refine=attempted_refine_sample,
                                sample_role=(
                                    "pre_refine_to_manual"
                                    if isinstance(auto_pre_sample, dict)
                                    else "manual_before_to_manual"
                                ),
                            )
                            if ok_sample and calib.gripper_correction_samples:
                                calib.gripper_correction_samples[-1]["after_point_id"] = int(calib.points[-1].get("id") or 0)
                            extra_status = "; " + sample_status
                            append_manual_correction_trace(
                                args,
                                before_sample,
                                correction_det,
                                correction_gripper_det,
                                saved_obs,
                                sample_status,
                            )
                            post_before_sample = None
                            if isinstance(auto_pre_sample, dict):
                                if isinstance(manual_marker_sample, dict):
                                    post_before_sample = manual_marker_sample
                                elif isinstance(auto_post_sample, dict):
                                    post_before_sample = auto_post_sample
                            if (
                                post_before_sample is not None
                                and post_before_sample is not before_sample
                                and correction_gripper_det is not None
                            ):
                                ok_post_sample, post_sample_status = append_gripper_correction_sample(
                                    calib,
                                    post_before_sample,
                                    correction_det,
                                    correction_gripper_det,
                                    saved_obs,
                                    args,
                                    sample_role="post_refine_to_manual",
                                )
                                if ok_post_sample and calib.gripper_correction_samples:
                                    calib.gripper_correction_samples[-1]["after_point_id"] = int(calib.points[-1].get("id") or 0)
                                extra_status += "; " + post_sample_status
                            gripper_correction_before = None
                            alignment_state["pre_refine_before"] = None
                            alignment_state["post_refine_after"] = None
                        save_calibration(config_path, calib, args)
                        pose_source = ensure_pose_model(calib, args)
                        orientation_source = ensure_orientation_model(calib, args)
                        if extra_status:
                            status += extra_status
                        status += set_direct_replay_point_to_saved_point(calib.points[-1], args)
                    print(status, flush=True)

            elapsed = time.perf_counter() - loop_start
            if elapsed < 0.01:
                time.sleep(0.01 - elapsed)

    finally:
        if act_recording and act_dataset is not None:
            try:
                print(save_act_record_episode(act_dataset, args), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"ACT episode save failed during shutdown: {exc}", flush=True)
        if act_dataset is not None:
            try:
                act_dataset.stop_image_writer()
            except Exception:
                pass
        alignment_cancel.set()
        alignment_thread = alignment_state.get("thread")
        if alignment_thread is not None and alignment_thread.is_alive():
            alignment_thread.join(timeout=10.0)
        before_marker_thread = before_marker_state.get("thread")
        if before_marker_thread is not None and before_marker_thread.is_alive():
            before_marker_thread.join(timeout=5.0)
        stop_teleop.set()
        manual_enabled.clear()
        if teleop_thread is not None:
            teleop_thread.join(timeout=1.0)
        if preview is not None:
            preview.close()
        capture.release()
        if gripper_capture is not None:
            gripper_capture.release()
        try:
            if leader is not None and leader.is_connected:
                release_leader_sync_hold(leader, leader_sync_hold)
                leader.disconnect()
        finally:
            if robot.is_connected:
                robot.disconnect()

    return 0


def main() -> int:
    require_deps()
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if args.mode == "calibrate":
        return run_calibrate(args)
    return run_align(args)


if __name__ == "__main__":
    raise SystemExit(main())
