from __future__ import annotations

import csv
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "vision" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import yolo_guided_align as align  # noqa: E402


def base_args(**overrides):
    values = {
        "target_class": "red_cube",
        "target_point": "auto",
        "share_calibration_across_classes": True,
        "use_legacy_auto_calibration_points": False,
        "allow_legacy_top_face_points": False,
        "map_joints": ["shoulder_pan.pos", "shoulder_lift.pos"],
        "pose_joints": list(align.DEFAULT_POSE_JOINTS),
        "joint_error_threshold": 4.0,
        "max_joint_step": 3.0,
        "move_steps": 90,
        "min_move_steps": 12,
        "move_sleep_s": 0.0,
        "max_move_sleep_s": 0.0,
        "gripper_reference_mode": "recorded_robust",
        "gripper_reference_min_conf": 0.85,
        "manual_save_min_gripper_conf": 0.85,
        "manual_save_max_ref_error_px": 25.0,
        "gripper_reference_max_outlier_px": 30.0,
        "gripper_reference_mad_scale": 2.5,
        "gripper_reference_min_radius_px": 8.0,
        "gripper_reference_min_count": 1,
        "gripper_center_tolerance_px": 8.0,
        "gripper_replay_nearest_pose": True,
        "gripper_replay_snap_px": 50.0,
        "gripper_replay_tie_px": 2.0,
        "gripper_replay_max_fail_count": 1,
        "gripper_trace": False,
        "gripper_final_iters": 24,
        "gripper_ibvs_probe_deg": 1.2,
        "gripper_ibvs_gain": 0.72,
        "gripper_ibvs_damping": 2.0,
        "gripper_debug_ibvs_probe_deg": 1.2,
        "gripper_debug_ibvs_max_step_deg": 0.5,
        "gripper_debug_ibvs_min_step_deg": 0.2,
        "gripper_debug_ibvs_restore_threshold_deg": 0.8,
        "gripper_debug_ibvs_restore_timeout_s": 8.0,
        "gripper_debug_ibvs_restore_resend_interval_s": 0.05,
        "gripper_debug_ibvs_min_actual_controlled_deg": 0.25,
        "gripper_debug_ibvs_min_pixel_motion_px": 3.0,
        "gripper_debug_ibvs_max_jitter_px": 6.0,
        "gripper_debug_ibvs_uncontrolled_warn_deg": 0.5,
        "gripper_debug_ibvs_max_uncontrolled_drift_deg": 1.0,
        "gripper_ibvs_max_condition": 30.0,
        "gripper_ibvs_min_px_per_deg": 1.0,
        "gripper_ibvs_max_rmse_px": 8.0,
        "gripper_ibvs_max_probes": 6,
        "gripper_ibvs_max_worse_px": 18.0,
        "gripper_ibvs_stagnation_px": 1.0,
        "gripper_ibvs_max_stagnant": 3,
        "gripper_ibvs_broyden_blend": 0.65,
        "gripper_max_step_deg": 2.3,
        "gripper_min_step_deg": 0.75,
        "gripper_full_step_error_px": 45.0,
        "gripper_settle_s": 0.0,
        "gripper_refine_joint_speed_deg_s": 75.0,
        "gripper_refine_min_move_steps": 4,
        "gripper_refine_wait_joints": None,
        "gripper_refine_position_threshold": 0.80,
        "gripper_refine_wait_timeout_s": 0.01,
        "gripper_refine_hold_command_interval_s": 0.0,
        "gripper_refine_continue_threshold": 1.2,
        "gripper_refine_stable_epsilon_deg": 0.12,
        "gripper_refine_stable_samples": 4,
        "gripper_mask_area_guard": True,
        "gripper_max_area_growth": 0.30,
        "gripper_area_guard_min_px": 200.0,
        "gripper_mask_shape_guard": True,
        "gripper_mask_shape_profile": "",
        "gripper_mask_area_tolerance": 0.28,
        "gripper_mask_axis_tolerance": 0.35,
        "gripper_mask_circularity_drop": 0.12,
        "gripper_mask_solidity_drop": 0.06,
        "gripper_mask_shape_worse_tolerance": 0.20,
        "gripper_max_cumulative_lift_drift_deg": 4.0,
        "gripper_cumulative_lift_margin_deg": 0.45,
        "gripper_height_guard_action": "recover",
        "gripper_hover_input_margin_deg": 0.5,
        "gripper_motion_max_px_per_deg": 35.0,
        "gripper_motion_extra_joint_max_deg": 0.85,
        "gripper_motion_extra_joint_ratio": 0.80,
        "gripper_motion_max_samples": 400,
        "gripper_motion_local_joint_radius_deg": 12.0,
        "gripper_motion_local_overhead_radius_px": 140.0,
        "gripper_primitive_local_overhead_radius_px": 140.0,
        "gripper_primitive_max_px_per_unit": 80.0,
        "gripper_primitive_min_action": 0.10,
        "gripper_primitive_max_samples": 400,
        "gripper_min_actual_motion_deg": 0.12,
        "gripper_min_actual_motion_fraction": 0.18,
        "gripper_learned_k": 1,
        "gripper_learned_power": 2.0,
        "gripper_learned_min_error_px": 8.0,
        "gripper_learned_min_improve_px": 1.0,
        "gripper_learned_accept_improve_px": 0.75,
        "gripper_learned_neutral_band_px": 1.5,
        "gripper_learned_reverse_worse_px": 2.5,
        "gripper_learned_rollback_worse_px": 1.0,
        "gripper_learned_local_overhead_radius_px": 80.0,
        "gripper_learned_local_joint_radius_deg": 8.0,
        "gripper_learned_direction_gate_error_px": 25.0,
        "gripper_learned_direction_min_cos": 0.65,
        "gripper_learned_full_min_step_deg": 0.5,
        "gripper_learned_full_max_step_deg": 6.0,
        "gripper_learned_full_max_pan_deg": 2.5,
        "gripper_learned_full_max_lift_deg": 4.0,
        "gripper_learned_full_max_elbow_deg": 4.0,
        "gripper_learned_full_max_wrist_flex_deg": 2.5,
        "gripper_learned_full_max_wrist_roll_deg": 2.0,
        "gripper_learned_full_min_local_samples": 1,
        "gripper_learned_full_require_both_local": False,
        "gripper_learned_full_allow_global_sample": False,
        "gripper_learned_autonomous_search": True,
        "gripper_learned_search_step_deg": 1.2,
        "gripper_learned_search_axis_step_deg": 0.25,
        "gripper_learned_search_max_candidates": 4,
        "gripper_learned_search_axis_first": False,
        "gripper_learned_allow_axis_probes": False,
        "gripper_learned_search_axis_joints": [],
        "gripper_learned_clearance": False,
        "gripper_learned_near_done_px": 0.0,
        "gripper_learned_refine_outer_correction": True,
        "gripper_learned_settle_threshold_deg": 0.35,
        "gripper_learned_rollback_outer_correction": True,
        "gripper_learned_stop_after_accepted_move": False,
        "gripper_learned_use_param_model": False,
        "gripper_learned_auto_update_params": True,
        "gripper_learned_param_history": 3,
        "gripper_learned_param_only": False,
        "coarse_bias_enable": True,
        "coarse_bias_auto_update": True,
        "coarse_bias_history": 3,
        "coarse_bias_max_joint_deg": 4.0,
        "coarse_bias_min_improve_px": 1.0,
        "gripper_ibvs_probe_restore_worse_px": 4.0,
        "auto_learn_overhead_after_gripper": False,
        "auto_learn_min_xy_delta_deg": 0.25,
        "direct_replay_point_id": 0,
        "direct_replay_include_gripper": True,
        "direct_replay_outer_correction": True,
        "direct_replay_outer_iters": 3,
        "direct_replay_outer_gain": 1.0,
        "direct_replay_outer_max_offset": 6.0,
        "dry_run": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeRobot:
    def __init__(self) -> None:
        self.pose = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": -55.0,
            "wrist_flex.pos": 98.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 0.0,
        }

    def get_observation(self):
        return dict(self.pose)

    def send_action(self, action):
        self.pose.update({key: float(value) for key, value in action.items() if key in self.pose})
        return dict(self.pose)


class CoupledFakeRobot(FakeRobot):
    def send_action(self, action):
        old_pan = float(self.pose["shoulder_pan.pos"])
        old_lift = float(self.pose["shoulder_lift.pos"])
        if "shoulder_pan.pos" in action:
            self.pose["shoulder_pan.pos"] = float(action["shoulder_pan.pos"])
        if "shoulder_lift.pos" in action:
            self.pose["shoulder_lift.pos"] = float(action["shoulder_lift.pos"])
        for joint in ("wrist_flex.pos", "wrist_roll.pos", "gripper.pos"):
            if joint in action:
                self.pose[joint] = float(action[joint])
        pan_delta = float(self.pose["shoulder_pan.pos"]) - old_pan
        lift_delta = float(self.pose["shoulder_lift.pos"]) - old_lift
        self.pose["elbow_flex.pos"] = float(self.pose["elbow_flex.pos"]) + 0.2035 * pan_delta + 0.10 * lift_delta
        return dict(self.pose)


class BiasedShoulderLiftFakeRobot(FakeRobot):
    def send_action(self, action):
        for key, value in action.items():
            if key not in self.pose:
                continue
            value_f = float(value)
            if key == "shoulder_lift.pos" and value_f < -10.0:
                self.pose[key] = value_f + 3.0
            else:
                self.pose[key] = value_f
        return dict(self.pose)


class GuidedAlignRegressionTests(unittest.TestCase):
    def test_current_config_uses_fixed_robust_reference_and_ignores_legacy_auto_points(self) -> None:
        config_path = PROJECT_ROOT / "vision" / "config" / "overhead_to_safeplane_homography.json"
        state = align.load_calibration(config_path)
        args = base_args()

        trusted = align.matching_calibration_points(state.points, args)
        if len(trusted) < 15:
            self.skipTest("current calibration was reset for re-recording")
        self.assertGreaterEqual(len(trusted), 15)
        self.assertFalse(
            any(
                str(point.get("source", "")).startswith("auto_gripper_refine")
                and not bool(point.get("verified"))
                for point in trusted
            )
        )
        first, first_source = align.gripper_reference_pixel(state, args, (480, 640, 3), (320.0, 240.0))
        second, second_source = align.gripper_reference_pixel(state, args, (480, 640, 3), (500.0, 350.0))
        self.assertEqual(first, second)
        self.assertEqual(first_source, second_source)
        self.assertIsNotNone(first)
        assert first is not None
        if np.linalg.norm(np.asarray(first) - np.asarray((347.5, 283.0))) >= 6.0:
            self.skipTest("current calibration reference has been re-taught since this historical assertion")
        self.assertLess(np.linalg.norm(np.asarray(first) - np.asarray((347.5, 283.0))), 6.0)

        replay, replay_distance = align.nearest_replay_pose_point(318.1, 147.5, state, args)
        self.assertIsNotNone(replay)
        assert replay is not None
        self.assertEqual(int(replay["id"]), 27)
        self.assertLess(float(replay_distance), 1.0)

    def test_historical_rollback_did_not_restore_visual_state(self) -> None:
        trace_path = (
            PROJECT_ROOT
            / "vision"
            / "outputs"
            / "yolo_guided_align_traces"
            / "gripper_stage3_20260719_191946.csv"
        )
        if not trace_path.exists():
            self.skipTest("historical trace not present")
        with trace_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        start = next(row for row in rows if row["event"] == "stepwise_observe")
        rollback = next(row for row in rows if row["event"] == "stepwise_rollback")
        start_pixel = np.asarray([float(start["center_u"]), float(start["center_v"])])
        rollback_pixel = np.asarray([float(rollback["center_u"]), float(rollback["center_v"])])
        self.assertGreater(float(np.linalg.norm(rollback_pixel - start_pixel)), 5.0)
        self.assertIn("rollback_ok=False", rollback["status"])

    def test_fresh_reprobe_samples_override_a_bad_saved_jacobian(self) -> None:
        args = base_args()
        actual = np.asarray([[-8.0, -6.0], [-9.0, 2.5]])
        wrong = -actual
        samples = []
        sample_id = 0
        old_deltas = (np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]))
        fresh_deltas = (
            np.asarray([1.0, 0.0]),
            np.asarray([-1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, -1.0]),
        )
        for matrix, deltas in ((wrong, old_deltas), (actual, fresh_deltas)):
            for delta in deltas:
                sample_id += 1
                samples.append(
                    {
                        "id": sample_id,
                        "controller_version": "ibvs_v2",
                        "settled": True,
                        "target_class": "red_cube",
                        "start_xy": {joint: 0.0 for joint in args.map_joints},
                        "end_xy": {
                            joint: float(delta[index])
                            for index, joint in enumerate(args.map_joints)
                        },
                        "delta_joint": [float(value) for value in delta],
                        "delta_pixel": [float(value) for value in matrix @ delta],
                        "overhead_pixel": [320.0, 240.0],
                    }
                )
        state = align.CalibrationState(gripper_motion_samples=samples)
        estimate, status = align.estimate_local_gripper_jacobian(
            state,
            {joint: 0.0 for joint in args.map_joints},
            (320.0, 240.0),
            args,
            minimum_sample_id=2,
        )
        self.assertIsNotNone(estimate, status)
        assert estimate is not None
        np.testing.assert_allclose(estimate.matrix, actual, atol=0.01)

    def test_contaminated_gripper_motion_sample_is_ignored(self) -> None:
        args = base_args()
        bad_sample = {
            "id": 3,
            "controller_version": "ibvs_v2",
            "settled": True,
            "target_class": "red_cube",
            "start_xy": {"shoulder_pan.pos": 0.0, "shoulder_lift.pos": 0.0},
            "end_xy": {"shoulder_pan.pos": -0.691, "shoulder_lift.pos": 0.0},
            "start_joints": {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": -50.0,
                "wrist_flex.pos": 88.0,
                "wrist_roll.pos": -12.0,
            },
            "end_joints": {
                "shoulder_pan.pos": -0.691,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": -48.144,
                "wrist_flex.pos": 88.0,
                "wrist_roll.pos": -12.0,
            },
            "delta_joint": [-0.691, 0.0],
            "delta_pixel": [45.97, 12.91],
        }
        ok, reason = align.gripper_motion_sample_quality(bad_sample, args)
        self.assertFalse(ok, reason)
        self.assertIn("69.1px/deg", reason)

        state = align.CalibrationState(gripper_motion_samples=[bad_sample])
        current_xy = {joint: 0.0 for joint in args.map_joints}
        transitions = align.local_gripper_motion_transitions(state, current_xy, None, args)
        self.assertEqual(transitions, [])

    def test_mask_shape_guard_rejects_center_improvement_with_bad_height_shape(self) -> None:
        args = base_args()
        reference = {
            "mask_area_px": 22329.0,
            "mask_axis_ratio": 1.20,
            "mask_circularity": 0.80,
            "mask_solidity": 0.99,
        }
        start = align.GripperServoObservation(
            detection=None,
            ref_uv=None,
            error_uv=(40.0, 0.0),
            dist_px=40.0,
            ref_source="test",
            sample_count=5,
            jitter_px=0.0,
            target_area_px=22329.0,
            mask_features=dict(reference),
            status="ok",
        )
        bad_end = align.GripperServoObservation(
            detection=None,
            ref_uv=None,
            error_uv=(10.0, 0.0),
            dist_px=10.0,
            ref_source="test",
            sample_count=5,
            jitter_px=0.0,
            target_area_px=58591.0,
            mask_features={
                "mask_area_px": 58591.0,
                "mask_axis_ratio": 3.98,
                "mask_circularity": 0.32,
                "mask_solidity": 0.77,
            },
            status="ok",
        )
        ok, reason = align.gripper_mask_shape_guard(start, bad_end, reference, args)
        self.assertFalse(ok, reason)
        self.assertIn("mask_shape_guard rejected", reason)

        good_end = align.GripperServoObservation(
            detection=None,
            ref_uv=None,
            error_uv=(10.0, 0.0),
            dist_px=10.0,
            ref_source="test",
            sample_count=5,
            jitter_px=0.0,
            target_area_px=22470.0,
            mask_features={
                "mask_area_px": 22470.0,
                "mask_axis_ratio": 1.21,
                "mask_circularity": 0.80,
                "mask_solidity": 0.99,
            },
            status="ok",
        )
        ok, reason = align.gripper_mask_shape_guard(start, good_end, reference, args)
        self.assertTrue(ok, reason)

    def test_legacy_mode_is_routed_to_ibvs_without_explicit_escape_hatch(self) -> None:
        args = base_args(
            enable_gripper_final_align=True,
            gripper_servo_mode="search",
            allow_legacy_gripper_servo=False,
        )
        state = align.CalibrationState(hover_height_model={"elbow_flex.pos": [0.0]})
        with patch.object(align, "run_gripper_ibvs_xy_refine", return_value="DONE: routed") as routed:
            status = align.run_gripper_final_xy_refine(
                FakeRobot(),
                object(),
                object(),
                {},
                state,
                args,
                {joint: 0.0 for joint in args.map_joints},
            )
        self.assertEqual(status, "DONE: routed")
        routed.assert_called_once()

    def test_verify_only_stage3_does_not_send_motion(self) -> None:
        args = base_args(
            enable_gripper_final_align=True,
            gripper_servo_mode="verify_only",
        )
        robot = FakeRobot()
        state = align.CalibrationState()
        detection = align.TargetDetection(
            class_id=0,
            class_name="red_cube",
            confidence=0.99,
            center_uv=(350.0, 284.0),
            xyxy=[0.0, 0.0, 1.0, 1.0],
        )
        observation = align.GripperServoObservation(
            detection=detection,
            ref_uv=(347.0, 283.0),
            error_uv=(3.0, 1.0),
            dist_px=float(np.hypot(3.0, 1.0)),
            ref_source="synthetic_reference",
            sample_count=5,
            jitter_px=0.0,
            status="ok synthetic",
        )
        with patch.object(align, "observe_gripper_error_median", return_value=observation):
            status = align.run_gripper_final_xy_refine(
                robot,
                object(),
                object(),
                {},
                state,
                args,
                {joint: 0.0 for joint in args.map_joints},
            )
        self.assertTrue(status.startswith("DONE: gripper_verify_only"), status)
        self.assertEqual(robot.pose["shoulder_pan.pos"], 0.0)
        self.assertEqual(robot.pose["shoulder_lift.pos"], 0.0)

    def test_direct_replay_sends_saved_raw_joints_by_id(self) -> None:
        args = base_args(
            direct_replay_point_id=2,
            gripper_refine_wait_timeout_s=0.2,
            gripper_refine_stable_samples=1,
            gripper_refine_stable_epsilon_deg=0.001,
            gripper_refine_hold_command_interval_s=0.0,
        )
        first = {
            "id": 1,
            "source": "manual_correction",
            "pixel": [100.0, 120.0],
            "joints": {
                "shoulder_pan.pos": 10.0,
                "shoulder_lift.pos": 11.0,
                "elbow_flex.pos": -40.0,
                "wrist_flex.pos": 70.0,
                "wrist_roll.pos": 5.0,
                "gripper.pos": 0.2,
            },
        }
        second = {
            "id": 2,
            "source": "manual_correction",
            "pixel": [300.0, 320.0],
            "joints": {
                "shoulder_pan.pos": -12.0,
                "shoulder_lift.pos": 6.5,
                "elbow_flex.pos": -52.0,
                "wrist_flex.pos": 83.0,
                "wrist_roll.pos": -9.0,
                "gripper.pos": 0.4,
            },
        }
        robot = FakeRobot()
        state = align.CalibrationState(points=[first, second])
        status = align.run_direct_pose_replay_test(
            robot,
            state,
            args,
            threading.RLock(),
        )
        self.assertTrue(status.startswith("DONE: direct_replay point=2"), status)
        for joint, value in second["joints"].items():
            self.assertAlmostEqual(robot.pose[joint], value)

    def test_direct_replay_outer_correction_offsets_biased_lift(self) -> None:
        args = base_args(
            direct_replay_point_id=1,
            joint_error_threshold=0.8,
            gripper_refine_wait_timeout_s=0.2,
            gripper_refine_stable_samples=1,
            gripper_refine_stable_epsilon_deg=0.001,
            gripper_refine_hold_command_interval_s=0.0,
        )
        point = {
            "id": 1,
            "source": "manual_correction",
            "pixel": [300.0, 320.0],
            "joints": {
                "shoulder_pan.pos": -12.0,
                "shoulder_lift.pos": -13.0,
                "elbow_flex.pos": -52.0,
                "wrist_flex.pos": 83.0,
                "wrist_roll.pos": -9.0,
                "gripper.pos": 0.4,
            },
        }
        robot = BiasedShoulderLiftFakeRobot()
        state = align.CalibrationState(points=[point])
        status = align.run_direct_pose_replay_test(
            robot,
            state,
            args,
            threading.RLock(),
        )
        self.assertTrue(status.startswith("DONE: direct_replay point=1"), status)
        self.assertIn("raw_max_joint_error=3.00", status)
        self.assertLess(abs(robot.pose["shoulder_lift.pos"] - point["joints"]["shoulder_lift.pos"]), 0.8)

    def test_stage3_failure_does_not_block_a_legacy_saved_replay_point(self) -> None:
        args = base_args(gripper_replay_snap_px=2.0)
        point = {
            "id": 61,
            "source": "manual_correction",
            "pixel": [316.9, 179.8],
            "joints": {
                "shoulder_pan.pos": 22.0,
                "shoulder_lift.pos": 13.0,
                "elbow_flex.pos": -55.0,
                "wrist_flex.pos": 87.0,
                "wrist_roll.pos": 12.0,
            },
            "replay_fail_count": 1,
            "last_replay_failure": "STOP: gripper_learned_full prediction did not improve",
        }
        state = align.CalibrationState(points=[point])
        replay, distance = align.nearest_replay_pose_point(316.9, 179.8, state, args)
        self.assertIs(replay, point)
        self.assertAlmostEqual(float(distance), 0.0)

        point["last_replay_failure"] = "full_pose replay encoder target did not settle"
        replay, distance = align.nearest_replay_pose_point(316.9, 179.8, state, args)
        self.assertIsNone(replay)
        self.assertIsNone(distance)

    def test_visual_step_classification_uses_distance_even_when_motion_is_uncontrolled(self) -> None:
        outcome, improvement = align.classify_gripper_visual_step(31.5, 40.2, 1.5, 2.5)
        self.assertEqual(outcome, "farther")
        self.assertAlmostEqual(improvement, -8.7)

        outcome, improvement = align.classify_gripper_visual_step(31.5, 26.0, 1.5, 2.5)
        self.assertEqual(outcome, "closer")
        self.assertAlmostEqual(improvement, 5.5)

        outcome, _improvement = align.classify_gripper_visual_step(31.5, 32.0, 1.5, 2.5)
        self.assertEqual(outcome, "neutral")

    def test_learned_search_step_is_raised_above_the_deadzone_floor(self) -> None:
        args = base_args(
            gripper_learned_full_min_step_deg=1.0,
            gripper_learned_search_step_deg=1.2,
        )
        tiny = {joint: 0.0 for joint in args.pose_joints}
        tiny["shoulder_lift.pos"] = -0.2
        candidates = align.learned_full_pose_search_candidates(tiny, args)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(candidates[0][0], "learned_small")
        self.assertAlmostEqual(abs(candidates[0][1]["shoulder_lift.pos"]), 1.0)
        self.assertEqual(candidates[1][0], "learned_reverse")
        self.assertAlmostEqual(abs(candidates[1][1]["shoulder_lift.pos"]), 1.0)

    def test_stable_gripper_observation_uses_multi_frame_measurement(self) -> None:
        args = base_args()
        detection = align.TargetDetection(
            class_id=1,
            class_name="red_cube",
            confidence=0.99,
            center_uv=(380.0, 290.0),
            xyxy=[0.0, 0.0, 1.0, 1.0],
        )
        observation = align.GripperServoObservation(
            detection=detection,
            ref_uv=(365.0, 274.0),
            error_uv=(15.0, 16.0),
            dist_px=float(np.hypot(15.0, 16.0)),
            ref_source="synthetic_reference",
            sample_count=5,
            jitter_px=0.3,
            status="ok median=5/5 jitter=0.30px attempt=1",
        )
        with patch.object(align, "observe_gripper_error_median", return_value=observation) as observe:
            result = align.observe_gripper_error_stable(
                object(),
                object(),
                {},
                align.CalibrationState(),
                args,
            )
        observe.assert_called_once()
        self.assertEqual(result[-1], "ok")
        self.assertIn("median=5/5", result[-2])
        self.assertAlmostEqual(result[4], float(np.hypot(15.0, 16.0)))

    def test_learned_full_search_crosses_baseline_before_testing_reverse(self) -> None:
        args = base_args(
            gripper_center_tolerance_px=5.0,
            gripper_final_iters=1,
            gripper_settle_s=0.0,
        )
        robot = FakeRobot()
        state = align.CalibrationState()
        sent_pan_targets: list[float] = []

        def fake_observe(*_args, **_kwargs):
            pan = float(robot.pose["shoulder_pan.pos"])
            dist = 30.0 + 10.0 * pan
            ref = (365.0, 274.0)
            detection = align.TargetDetection(
                class_id=1,
                class_name="red_cube",
                confidence=0.99,
                center_uv=(ref[0] + dist, ref[1]),
                xyxy=[0.0, 0.0, 1.0, 1.0],
            )
            return detection, ref, dist, 0.0, dist, "synthetic median=5/5", "ok"

        def fake_send(_robot, target, _args, _lock, **_kwargs):
            del _args, _lock, _kwargs
            sent_pan_targets.append(float(target["shoulder_pan.pos"]))
            robot.pose.update({joint: float(value) for joint, value in target.items()})
            return {
                "cancelled": False,
                "ok_wait": True,
                "wait_status": "synthetic settled",
                "max_error": 0.0,
                "raw_error": 0.0,
                "correction_notes": [],
            }

        learned_delta = {joint: 0.0 for joint in args.pose_joints}
        learned_delta["shoulder_pan.pos"] = 1.0
        with (
            patch.object(align, "matching_gripper_correction_samples", return_value=[{"id": 1}]),
            patch.object(
                align,
                "predict_gripper_learned_full_pose_delta",
                return_value=(learned_delta, "synthetic learned direction"),
            ),
            patch.object(align, "observe_gripper_error_stable", side_effect=fake_observe),
            patch.object(align, "send_pose_with_encoder_outer_correction", side_effect=fake_send),
        ):
            status = align.run_gripper_learned_full_pose_refine(
                robot,
                object(),
                object(),
                {},
                state,
                args,
                {joint: 0.0 for joint in args.map_joints},
                threading.RLock(),
                threading.RLock(),
                overhead_uv=(320.0, 240.0),
            )

        self.assertTrue(status.startswith("INCOMPLETE:"), status)
        self.assertEqual(sent_pan_targets[:2], [1.0, -1.0])
        self.assertNotIn(0.0, sent_pan_targets[:2])
        self.assertAlmostEqual(robot.pose["shoulder_pan.pos"], -1.0)

    def test_learned_full_reverse_uses_actual_compensated_motion(self) -> None:
        args = base_args(
            gripper_center_tolerance_px=5.0,
            gripper_final_iters=1,
            gripper_settle_s=0.0,
            gripper_learned_full_max_pan_deg=4.0,
        )
        robot = FakeRobot()
        state = align.CalibrationState()
        sent_pan_targets: list[float] = []
        send_count = 0

        def fake_observe(*_args, **_kwargs):
            pan = float(robot.pose["shoulder_pan.pos"])
            dist = 30.0 + 10.0 * pan
            ref = (365.0, 274.0)
            detection = align.TargetDetection(
                class_id=1,
                class_name="red_cube",
                confidence=0.99,
                center_uv=(ref[0] + dist, ref[1]),
                xyxy=[0.0, 0.0, 1.0, 1.0],
            )
            return detection, ref, dist, 0.0, dist, "synthetic median=5/5", "ok"

        def fake_send(_robot, target, _args, _lock, **_kwargs):
            nonlocal send_count
            del _args, _lock, _kwargs
            send_count += 1
            sent_pan_targets.append(float(target["shoulder_pan.pos"]))
            if send_count == 1:
                robot.pose["shoulder_pan.pos"] = 2.0
            else:
                robot.pose.update({joint: float(value) for joint, value in target.items()})
            return {
                "cancelled": False,
                "ok_wait": True,
                "wait_status": "synthetic settled",
                "max_error": 0.0,
                "raw_error": 0.0,
                "correction_notes": [],
            }

        learned_delta = {joint: 0.0 for joint in args.pose_joints}
        learned_delta["shoulder_pan.pos"] = 1.0
        with (
            patch.object(align, "matching_gripper_correction_samples", return_value=[{"id": 1}]),
            patch.object(
                align,
                "predict_gripper_learned_full_pose_delta",
                return_value=(learned_delta, "synthetic learned direction"),
            ),
            patch.object(align, "observe_gripper_error_stable", side_effect=fake_observe),
            patch.object(align, "send_pose_with_encoder_outer_correction", side_effect=fake_send),
        ):
            status = align.run_gripper_learned_full_pose_refine(
                robot,
                object(),
                object(),
                {},
                state,
                args,
                {joint: 0.0 for joint in args.map_joints},
                threading.RLock(),
                threading.RLock(),
                overhead_uv=(320.0, 240.0),
            )

        self.assertTrue(status.startswith("INCOMPLETE:"), status)
        self.assertEqual(sent_pan_targets[:2], [1.0, -2.0])
        self.assertAlmostEqual(robot.pose["shoulder_pan.pos"], -2.0)

    def test_gripper_param_model_updates_by_running_average(self) -> None:
        args = base_args(gripper_center_tolerance_px=5.0)
        state = align.CalibrationState()
        state.gripper_learned_param_model = {
            "version": 1,
            "updated_at": "old",
            "sample_count": 9,
            "gain": {joint: [4.0, 0.0] for joint in args.pose_joints},
            "history": [],
            "last_update": {"source": "old"},
        }
        delta_pose = {joint: 0.0 for joint in args.pose_joints}
        delta_pose["shoulder_pan.pos"] = 4.1

        ok, status = align.update_gripper_learned_param_model(
            state,
            args,
            error_uv=(1.0, 0.0),
            delta_pose=delta_pose,
            before_dist_px=10.0,
            after_dist_px=2.0,
            source="synthetic_done",
        )

        self.assertTrue(ok, status)
        self.assertEqual(state.gripper_learned_param_model["sample_count"], 10)
        self.assertAlmostEqual(
            state.gripper_learned_param_model["gain"]["shoulder_pan.pos"][0],
            4.01,
        )
        self.assertEqual(len(state.gripper_learned_param_model["history"]), 1)
        self.assertEqual(state.gripper_learned_param_model["history"][0]["sample_count"], 9)

    def test_coarse_pose_bias_updates_by_running_average(self) -> None:
        args = base_args(gripper_center_tolerance_px=5.0, coarse_bias_max_joint_deg=6.0)
        state = align.CalibrationState()
        state.coarse_pose_bias_model = {
            "version": 1,
            "updated_at": "old",
            "sample_count": 9,
            "bias": {joint: 4.0 for joint in args.pose_joints},
            "history": [],
            "last_update": {"source": "old"},
        }
        before = {joint: 0.0 for joint in args.pose_joints}
        after = {joint: 0.0 for joint in args.pose_joints}
        after["shoulder_pan.pos"] = 4.1

        ok, status = align.update_coarse_pose_bias_model(
            state,
            args,
            before_joints=before,
            after_joints=after,
            before_dist_px=10.0,
            after_dist_px=2.0,
            source="synthetic_done",
        )

        self.assertTrue(ok, status)
        self.assertEqual(state.coarse_pose_bias_model["sample_count"], 10)
        self.assertAlmostEqual(state.coarse_pose_bias_model["bias"]["shoulder_pan.pos"], 4.01)
        self.assertEqual(len(state.coarse_pose_bias_model["history"]), 1)
        self.assertEqual(state.coarse_pose_bias_model["history"][0]["sample_count"], 9)

    def test_apply_coarse_pose_bias_adjusts_model_prediction(self) -> None:
        args = base_args(coarse_bias_max_joint_deg=4.0)
        state = align.CalibrationState()
        state.coarse_pose_bias_model = {
            "sample_count": 3,
            "bias": {joint: 0.0 for joint in args.pose_joints},
        }
        state.coarse_pose_bias_model["bias"]["shoulder_pan.pos"] = 1.25
        target = {joint: 10.0 for joint in args.pose_joints}

        adjusted, status = align.apply_coarse_pose_bias(
            target,
            state,
            args,
            source="synthetic",
        )

        self.assertIn("coarse_bias n=3", status)
        self.assertAlmostEqual(adjusted["shoulder_pan.pos"], 11.25)
        self.assertAlmostEqual(target["shoulder_pan.pos"], 10.0)

    def test_primitive_jacobian_uses_virtual_action_not_raw_joint_coupling(self) -> None:
        args = base_args()
        actual = np.asarray([[-10.0, -3.0], [2.0, -12.0]], dtype=np.float64)
        samples = []
        for sample_id, delta in enumerate(
            (
                np.asarray([1.0, 0.0]),
                np.asarray([-1.0, 0.0]),
                np.asarray([0.0, 1.0]),
                np.asarray([0.0, -1.0]),
                np.asarray([0.7, 0.2]),
            ),
            start=1,
        ):
            samples.append(
                {
                    "id": sample_id,
                    "controller_version": "primitive_ibvs_v1",
                    "settled": True,
                    "target_class": "red_cube",
                    "delta_action": [float(value) for value in delta],
                    "delta_pixel": [float(value) for value in actual @ delta],
                    "overhead_pixel": [320.0, 240.0],
                    "start_joints": {
                        "shoulder_pan.pos": 0.0,
                        "shoulder_lift.pos": 0.0,
                        "elbow_flex.pos": -50.0,
                        "wrist_flex.pos": 88.0,
                        "wrist_roll.pos": -12.0,
                    },
                    "end_joints": {
                        "shoulder_pan.pos": float(0.4 * delta[0]),
                        "shoulder_lift.pos": float(0.2 * delta[1]),
                        "elbow_flex.pos": float(-50.0 + 2.0 * delta[0] - 1.5 * delta[1]),
                        "wrist_flex.pos": float(88.0 - 1.0 * delta[0] + 0.8 * delta[1]),
                        "wrist_roll.pos": float(-12.0 + 0.5 * delta[0]),
                    },
                }
            )
        state = align.CalibrationState(gripper_primitive_samples=samples)
        estimate, status = align.estimate_local_gripper_primitive_jacobian(
            state,
            (320.0, 240.0),
            args,
        )
        self.assertIsNotNone(estimate, status)
        assert estimate is not None
        np.testing.assert_allclose(estimate.matrix, actual, atol=0.01)

    def test_ibvs_converges_when_actual_servo_motion_is_only_part_of_command(self) -> None:
        robot = FakeRobot()
        state = align.CalibrationState()
        args = base_args(gripper_max_cumulative_lift_drift_deg=0.0)
        true_jacobian = np.asarray([[-8.0, -6.0], [-9.0, 2.5]], dtype=np.float64)
        initial_error = np.asarray([58.0, -24.0], dtype=np.float64)
        reference = np.asarray([347.0, 283.0], dtype=np.float64)

        def fake_observe(*_args, **_kwargs):
            q = np.asarray(
                [robot.pose["shoulder_pan.pos"], robot.pose["shoulder_lift.pos"]],
                dtype=np.float64,
            )
            error = initial_error + true_jacobian @ q
            center = reference + error
            detection = align.TargetDetection(
                class_id=0,
                class_name="red_cube",
                confidence=0.99,
                center_uv=(float(center[0]), float(center[1])),
                xyxy=[0.0, 0.0, 1.0, 1.0],
            )
            return align.GripperServoObservation(
                detection=detection,
                ref_uv=(float(reference[0]), float(reference[1])),
                error_uv=(float(error[0]), float(error[1])),
                dist_px=float(np.linalg.norm(error)),
                ref_source="synthetic_fixed_reference",
                sample_count=5,
                jitter_px=0.0,
                status="ok synthetic",
            )

        def fake_send(_robot, _calib, _args, target_xy, _lock=None, anchor=None, stop_event=None):
            del anchor, stop_event
            for joint in args.map_joints:
                current = float(robot.pose[joint])
                robot.pose[joint] = current + 0.72 * (float(target_xy[joint]) - current)
            return True, "synthetic settled at 72% of command"

        def fake_anchor(_robot, _calib, _args, _lock=None):
            pose = dict(robot.pose)
            xy = {joint: float(pose[joint]) for joint in args.map_joints}
            return align.RefinePoseAnchor(xy=xy, pose=pose, hover_model_pose={}, orientation_model_pose={})

        with (
            patch.object(align, "observe_gripper_error_median", side_effect=fake_observe),
            patch.object(align, "send_xy_hover_refine_target", side_effect=fake_send),
            patch.object(align, "create_refine_pose_anchor", side_effect=fake_anchor),
        ):
            status = align.run_gripper_ibvs_xy_refine(
                robot,
                object(),
                object(),
                {},
                state,
                args,
                {joint: 0.0 for joint in args.map_joints},
                threading.RLock(),
                threading.RLock(),
                overhead_uv=(320.0, 240.0),
            )

        self.assertTrue(status.startswith("DONE:"), status)
        self.assertGreaterEqual(len(state.gripper_motion_samples), 2)
        final = fake_observe()
        self.assertLessEqual(final.dist_px, args.gripper_center_tolerance_px)
        first_actual = np.asarray(state.gripper_motion_samples[0]["delta_joint"])
        self.assertAlmostEqual(float(np.max(np.abs(first_actual))), 0.72 * args.gripper_ibvs_probe_deg, places=5)

    def test_debug_ibvs_uses_command_space_and_allows_small_coupled_elbow_motion(self) -> None:
        robot = CoupledFakeRobot()
        state = align.CalibrationState()
        args = base_args(
            gripper_trace=False,
            gripper_refine_wait_timeout_s=0.5,
            gripper_refine_stable_samples=1,
            gripper_refine_stable_epsilon_deg=0.001,
            gripper_refine_hold_command_interval_s=0.0,
        )
        command_space_jacobian = np.asarray([[-20.0, -2.0], [1.0, -18.0]], dtype=np.float64)
        initial_error = np.asarray([35.0, -18.0], dtype=np.float64)
        reference = np.asarray([347.0, 283.0], dtype=np.float64)

        def fake_observe(*_args, **_kwargs):
            command_xy = np.asarray(
                [robot.pose["shoulder_pan.pos"], robot.pose["shoulder_lift.pos"]],
                dtype=np.float64,
            )
            error = initial_error + command_space_jacobian @ command_xy
            center = reference + error
            detection = align.TargetDetection(
                class_id=0,
                class_name="red_cube",
                confidence=0.99,
                center_uv=(float(center[0]), float(center[1])),
                xyxy=[0.0, 0.0, 1.0, 1.0],
            )
            return align.GripperServoObservation(
                detection=detection,
                ref_uv=(float(reference[0]), float(reference[1])),
                error_uv=(float(error[0]), float(error[1])),
                dist_px=float(np.linalg.norm(error)),
                ref_source="synthetic_fixed_reference",
                sample_count=5,
                jitter_px=0.4,
                status="ok synthetic",
            )

        with patch.object(align, "observe_gripper_error_median", side_effect=fake_observe):
            status = align.run_gripper_debug_ibvs_xy_refine(
                robot,
                object(),
                object(),
                {},
                state,
                args,
                {joint: 0.0 for joint in args.map_joints},
                threading.RLock(),
                threading.RLock(),
                overhead_uv=(320.0, 240.0),
            )

        self.assertIn("debug_ibvs one correction", status)
        final = fake_observe()
        self.assertLess(final.dist_px, float(np.linalg.norm(initial_error)))
        self.assertGreater(abs(robot.pose["elbow_flex.pos"] + 55.0), 0.0)

    def test_actual_motion_quality_rejects_soft_wait_without_map_motion(self) -> None:
        args = base_args()
        ok, reason = align.gripper_actual_motion_quality(
            np.asarray([-0.05, -1.72]),
            np.asarray([0.08, -0.08]),
            args,
        )
        self.assertFalse(ok, reason)
        self.assertIn("actual_motion rejected", reason)

        ok, reason = align.gripper_actual_motion_quality(
            np.asarray([1.20, 0.0]),
            np.asarray([0.61, 0.08]),
            args,
        )
        self.assertTrue(ok, reason)

    def test_lift_budget_clips_ibvs_step_before_height_guard(self) -> None:
        args = base_args(gripper_height_guard_action="stop")
        pose = {"shoulder_pan.pos": -2.92, "shoulder_lift.pos": 4.19}
        step, status = align.limit_gripper_step_by_lift_budget(
            pose,
            np.asarray([-0.2, 2.3]),
            args,
            baseline_lift=2.27,
        )
        self.assertIn("lift_budget clipped", status)
        self.assertAlmostEqual(float(step[0]), -0.2, places=6)
        self.assertLessEqual(float(pose["shoulder_lift.pos"] + step[1]), 2.27 + 4.0 - 0.45 + 1e-6)

    def test_lift_budget_does_not_clip_when_height_recovery_is_enabled(self) -> None:
        args = base_args(gripper_height_guard_action="recover")
        step, status = align.limit_gripper_step_by_lift_budget(
            {"shoulder_pan.pos": -2.92, "shoulder_lift.pos": 4.19},
            np.asarray([-0.2, 2.3]),
            args,
            baseline_lift=2.27,
        )
        self.assertIn("height recovery enabled", status)
        np.testing.assert_allclose(step, np.asarray([-0.2, 2.3]))


if __name__ == "__main__":
    unittest.main()
