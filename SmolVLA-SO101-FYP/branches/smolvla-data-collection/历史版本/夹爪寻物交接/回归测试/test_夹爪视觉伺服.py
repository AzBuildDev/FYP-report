from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from 旧版夹爪视觉伺服 import (  # noqa: E402
    broyden_update,
    damped_least_squares_step,
    estimate_image_jacobian,
    jacobian_is_usable,
    robust_reference_cluster,
)


class RobustReferenceTests(unittest.TestCase):
    def test_rejects_far_reference_outliers(self) -> None:
        points = [
            (347.1, 283.2),
            (350.0, 280.2),
            (343.4, 285.0),
            (352.8, 282.5),
            (345.6, 285.0),
            (144.5, 305.8),
        ]
        result = robust_reference_cluster(points, hard_limit_px=30.0)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertNotIn(5, result.inlier_indices)
        self.assertLess(np.linalg.norm(np.asarray(result.center_uv) - np.asarray((347.1, 283.2))), 3.0)


class JacobianTests(unittest.TestCase):
    def test_recovers_measured_local_mapping(self) -> None:
        expected = np.array([[-8.0, -6.0], [-9.0, 2.0]])
        transitions = []
        for delta in (
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
            np.array([0.8, 0.1]),
        ):
            transitions.append((delta, expected @ delta))
        estimate = estimate_image_jacobian(transitions)
        self.assertTrue(jacobian_is_usable(estimate, max_condition=20.0, min_px_per_deg=1.0))
        assert estimate is not None
        np.testing.assert_allclose(estimate.matrix, expected, atol=0.01)

    def test_rejects_two_sample_zero_rmse_jacobian(self) -> None:
        expected = np.array([[-8.0, -6.0], [-9.0, 2.0]])
        transitions = [
            (np.array([1.0, 0.0]), expected @ np.array([1.0, 0.0])),
            (np.array([0.0, 1.0]), expected @ np.array([0.0, 1.0])),
        ]
        self.assertIsNone(estimate_image_jacobian(transitions))

    def test_requires_symmetric_axis_coverage(self) -> None:
        expected = np.array([[-8.0, -6.0], [-9.0, 2.0]])
        transitions = [
            (np.array([1.0, 0.0]), expected @ np.array([1.0, 0.0])),
            (np.array([0.9, 0.0]), expected @ np.array([0.9, 0.0])),
            (np.array([0.0, 1.0]), expected @ np.array([0.0, 1.0])),
            (np.array([0.0, -1.0]), expected @ np.array([0.0, -1.0])),
        ]
        self.assertIsNone(estimate_image_jacobian(transitions))

    def test_damped_step_reduces_error(self) -> None:
        jacobian = np.array([[-8.0, -6.0], [-9.0, 2.0]])
        error = np.array([-80.0, 20.0])
        step = damped_least_squares_step(
            jacobian,
            error,
            gain=0.7,
            damping=2.0,
            max_abs_step_deg=2.3,
        )
        new_error = error + jacobian @ step
        self.assertLess(np.linalg.norm(new_error), np.linalg.norm(error))
        self.assertLessEqual(float(np.max(np.abs(step))), 2.3 + 1e-9)

    def test_broyden_learns_changed_response(self) -> None:
        initial = np.array([[-8.0, -6.0], [-9.0, 2.0]])
        actual = np.array([[-5.0, -8.0], [-7.0, 4.0]])
        delta = np.array([1.0, 0.5])
        updated = broyden_update(initial, delta, actual @ delta, blend=1.0)
        self.assertLess(np.linalg.norm(actual @ delta - updated @ delta), 1e-8)


if __name__ == "__main__":
    unittest.main()
