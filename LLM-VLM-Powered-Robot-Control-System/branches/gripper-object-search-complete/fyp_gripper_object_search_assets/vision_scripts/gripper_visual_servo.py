"""Pure math helpers for the gripper-camera image-based visual servo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RobustReference:
    center_uv: tuple[float, float]
    inlier_indices: tuple[int, ...]
    radius_px: float
    threshold_px: float


@dataclass(frozen=True)
class JacobianEstimate:
    matrix: np.ndarray
    condition: float
    min_singular_value: float
    rmse_px: float
    sample_count: int


def robust_reference_cluster(
    points: Sequence[Sequence[float]],
    *,
    hard_limit_px: float = 30.0,
    mad_scale: float = 2.5,
    min_radius_px: float = 8.0,
) -> RobustReference | None:
    """Return a robust center and circular spread for recorded aligned pixels."""
    if not points:
        return None
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError("reference points must be a finite Nx2 array")

    initial_center = np.median(values, axis=0)
    distances = np.linalg.norm(values - initial_center, axis=1)
    if len(values) >= 5:
        median_distance = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median_distance)))
        adaptive_limit = median_distance + max(1.0, float(mad_scale)) * 1.4826 * max(mad, 0.5)
    else:
        adaptive_limit = float("inf")

    threshold = max(float(min_radius_px), adaptive_limit)
    if hard_limit_px > 0:
        threshold = min(threshold, float(hard_limit_px))
    inliers = np.flatnonzero(distances <= threshold)
    required = min(3, len(values))
    if len(inliers) < required:
        inliers = np.argsort(distances)[:required]

    inlier_values = values[inliers]
    center = np.median(inlier_values, axis=0)
    final_distances = np.linalg.norm(inlier_values - center, axis=1)
    radius = float(np.percentile(final_distances, 90)) if len(final_distances) > 1 else 0.0
    radius = max(float(min_radius_px), radius)
    return RobustReference(
        center_uv=(float(center[0]), float(center[1])),
        inlier_indices=tuple(int(index) for index in inliers),
        radius_px=radius,
        threshold_px=float(threshold),
    )


def estimate_image_jacobian(
    transitions: Iterable[tuple[Sequence[float], Sequence[float]]],
    *,
    ridge: float = 1e-4,
    min_joint_delta_deg: float = 0.12,
    min_samples: int = 4,
    require_symmetric_axes: bool = True,
) -> JacobianEstimate | None:
    """Fit dpixel = J * djoint from measured, settled transitions."""
    joint_rows: list[np.ndarray] = []
    pixel_rows: list[np.ndarray] = []
    for delta_joint, delta_pixel in transitions:
        dq = np.asarray(delta_joint, dtype=np.float64).reshape(-1)
        dp = np.asarray(delta_pixel, dtype=np.float64).reshape(-1)
        if dq.shape != (2,) or dp.shape != (2,) or not np.all(np.isfinite(dq)) or not np.all(np.isfinite(dp)):
            continue
        if float(np.linalg.norm(dq)) < float(min_joint_delta_deg):
            continue
        joint_rows.append(dq)
        pixel_rows.append(dp)
    required_samples = max(4, int(min_samples))
    if len(joint_rows) < required_samples:
        return None

    q = np.stack(joint_rows)
    p = np.stack(pixel_rows)
    if require_symmetric_axes:
        tol = max(1e-6, float(min_joint_delta_deg))
        has_pan_pos = bool(np.any((q[:, 0] >= tol) & (np.abs(q[:, 0]) >= np.abs(q[:, 1]))))
        has_pan_neg = bool(np.any((q[:, 0] <= -tol) & (np.abs(q[:, 0]) >= np.abs(q[:, 1]))))
        has_lift_pos = bool(np.any((q[:, 1] >= tol) & (np.abs(q[:, 1]) >= np.abs(q[:, 0]))))
        has_lift_neg = bool(np.any((q[:, 1] <= -tol) & (np.abs(q[:, 1]) >= np.abs(q[:, 0]))))
        if not (has_pan_pos and has_pan_neg and has_lift_pos and has_lift_neg):
            return None
    if np.linalg.matrix_rank(q, tol=max(1e-6, float(min_joint_delta_deg) * 0.1)) < 2:
        return None
    regularized = q.T @ q + max(0.0, float(ridge)) * np.eye(2)
    jacobian_t = np.linalg.solve(regularized, q.T @ p)
    jacobian = jacobian_t.T
    singular = np.linalg.svd(jacobian, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-9))
    residual = p - q @ jacobian.T
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    return JacobianEstimate(
        matrix=jacobian,
        condition=condition,
        min_singular_value=float(singular[-1]),
        rmse_px=rmse,
        sample_count=len(joint_rows),
    )


def jacobian_is_usable(
    estimate: JacobianEstimate | None,
    *,
    max_condition: float,
    min_px_per_deg: float,
) -> bool:
    if estimate is None:
        return False
    return (
        np.all(np.isfinite(estimate.matrix))
        and estimate.condition <= float(max_condition)
        and estimate.min_singular_value >= float(min_px_per_deg)
    )


def damped_least_squares_step(
    jacobian: Sequence[Sequence[float]],
    error_uv: Sequence[float],
    *,
    gain: float,
    damping: float,
    max_abs_step_deg: float,
) -> np.ndarray:
    """Compute a bounded joint correction that reduces the image error."""
    j = np.asarray(jacobian, dtype=np.float64).reshape(2, 2)
    error = np.asarray(error_uv, dtype=np.float64).reshape(2)
    damping_sq = max(0.0, float(damping)) ** 2
    step = -float(gain) * j.T @ np.linalg.solve(j @ j.T + damping_sq * np.eye(2), error)
    limit = max(0.0, float(max_abs_step_deg))
    max_component = float(np.max(np.abs(step)))
    if limit <= 0:
        return np.zeros(2, dtype=np.float64)
    if max_component > limit:
        step = step * (limit / max_component)
    return step


def broyden_update(
    jacobian: Sequence[Sequence[float]],
    delta_joint: Sequence[float],
    delta_pixel: Sequence[float],
    *,
    blend: float = 0.65,
) -> np.ndarray:
    """Update J with one measured secant while damping noisy camera motion."""
    j = np.asarray(jacobian, dtype=np.float64).reshape(2, 2)
    dq = np.asarray(delta_joint, dtype=np.float64).reshape(2)
    dp = np.asarray(delta_pixel, dtype=np.float64).reshape(2)
    denominator = float(dq @ dq)
    if denominator <= 1e-8:
        return j.copy()
    correction = np.outer(dp - j @ dq, dq) / denominator
    alpha = min(1.0, max(0.0, float(blend)))
    return j + alpha * correction


def ensure_minimum_effective_step(
    step: Sequence[float],
    *,
    minimum_abs_step_deg: float,
    maximum_abs_step_deg: float,
) -> np.ndarray:
    """Raise a non-zero command above servo deadband without changing direction."""
    value = np.asarray(step, dtype=np.float64).reshape(2)
    maximum = max(0.0, float(maximum_abs_step_deg))
    minimum = min(maximum, max(0.0, float(minimum_abs_step_deg)))
    largest = float(np.max(np.abs(value)))
    if largest <= 1e-9 or minimum <= 0 or largest >= minimum:
        return value
    return value * (minimum / largest)
