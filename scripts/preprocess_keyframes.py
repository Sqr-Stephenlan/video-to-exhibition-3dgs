"""Coverage-aware keyframe analysis and selection primitives.

The module deliberately contains no video I/O or file writes.  The preprocess
entrypoint owns both so the policy can be unit-tested with synthetic images.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import cv2
import numpy as np


KEYFRAME_DEFAULT_SETTINGS: dict[str, Any] = {
    "keyframe_policy": "legacy",
    "quality_analysis_long_edge": 512,
    "flow_analysis_long_edge": 320,
    "adaptive_blur_percentile": 15.0,
    "selection_min_gap_sec": 0.10,
    "selection_target_gap_sec": 0.20,
    "selection_max_gap_sec": 0.30,
    "duplicate_window_sec": 2.0,
    "min_tracked_points": 64,
    "min_motion_inlier_ratio": 0.65,
    "min_motion_grid_coverage": 0.375,
    "max_motion_residual_diag_ratio": 0.015,
    "max_median_displacement_diag_ratio": 0.25,
    "max_affine_rotation_deg": 12.0,
    "min_affine_scale": 0.80,
    "max_affine_scale": 1.25,
    "motion_model": "affine",
    "flow_forward_backward_max_error_px": 1.5,
    "fundamental_ransac_threshold_px": 1.5,
    "fundamental_ransac_confidence": 0.999,
    "bridge_min_blur_ratio": 1.0,
    "max_bridge_window_sec": 0.0,
    "max_bridge_fraction": 1.0,
}


@dataclass(frozen=True)
class KeyframePolicy:
    name: str
    blur_floor: float
    max_overexposed_ratio: float
    max_underexposed_ratio: float
    duplicate_hash_threshold: int
    quality_analysis_long_edge: int
    flow_analysis_long_edge: int
    adaptive_blur_percentile: float
    min_gap_sec: float
    target_gap_sec: float
    max_gap_sec: float
    duplicate_window_sec: float
    min_tracked_points: int
    min_motion_inlier_ratio: float
    min_motion_grid_coverage: float
    max_motion_residual_diag_ratio: float
    max_median_displacement_diag_ratio: float
    max_affine_rotation_deg: float
    min_affine_scale: float
    max_affine_scale: float
    motion_model: str = "affine"
    flow_forward_backward_max_error_px: float = 1.5
    fundamental_ransac_threshold_px: float = 1.5
    fundamental_ransac_confidence: float = 0.999
    bridge_min_blur_ratio: float = 1.0
    max_bridge_window_sec: float = 0.0
    max_bridge_fraction: float = 1.0

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "KeyframePolicy":
        return cls(
            name=str(settings["keyframe_policy"]),
            blur_floor=float(settings["blur_threshold"]),
            max_overexposed_ratio=float(settings["overexposed_ratio"]),
            max_underexposed_ratio=float(settings["underexposed_ratio"]),
            duplicate_hash_threshold=int(settings["duplicate_hash_threshold"]),
            quality_analysis_long_edge=int(settings["quality_analysis_long_edge"]),
            flow_analysis_long_edge=int(settings["flow_analysis_long_edge"]),
            adaptive_blur_percentile=float(settings["adaptive_blur_percentile"]),
            min_gap_sec=float(settings["selection_min_gap_sec"]),
            target_gap_sec=float(settings["selection_target_gap_sec"]),
            max_gap_sec=float(settings["selection_max_gap_sec"]),
            duplicate_window_sec=float(settings["duplicate_window_sec"]),
            min_tracked_points=int(settings["min_tracked_points"]),
            min_motion_inlier_ratio=float(settings["min_motion_inlier_ratio"]),
            min_motion_grid_coverage=float(settings["min_motion_grid_coverage"]),
            max_motion_residual_diag_ratio=float(settings["max_motion_residual_diag_ratio"]),
            max_median_displacement_diag_ratio=float(settings["max_median_displacement_diag_ratio"]),
            max_affine_rotation_deg=float(settings["max_affine_rotation_deg"]),
            min_affine_scale=float(settings["min_affine_scale"]),
            max_affine_scale=float(settings["max_affine_scale"]),
            motion_model=str(settings.get("motion_model", "affine")),
            flow_forward_backward_max_error_px=float(settings.get("flow_forward_backward_max_error_px", 1.5)),
            fundamental_ransac_threshold_px=float(settings.get("fundamental_ransac_threshold_px", 1.5)),
            fundamental_ransac_confidence=float(settings.get("fundamental_ransac_confidence", 0.999)),
            bridge_min_blur_ratio=float(settings.get("bridge_min_blur_ratio", 1.0)),
            max_bridge_window_sec=float(settings.get("max_bridge_window_sec", 0.0)),
            max_bridge_fraction=float(settings.get("max_bridge_fraction", 1.0)),
        )


@dataclass(frozen=True)
class ScannedFrame:
    id: str
    segment_id: str
    sample_index: int
    timestamp_sec: float
    frame_index: int
    blur_score: float
    calibrated_blur_score: float
    overexposed_ratio: float
    underexposed_ratio: float
    average_hash_bits: np.ndarray
    flow_gray: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class MotionMetrics:
    valid: bool
    tracked_count: int
    inlier_count: int
    inlier_ratio: float
    grid_coverage: float
    median_displacement_diag_ratio: float
    affine_rotation_deg: float | None
    affine_scale: float | None
    residual_rmse_diag_ratio: float | None
    reason: str | None = None
    model: str = "affine"
    forward_backward_rmse_px: float | None = None

    def to_manifest(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "tracked_count": self.tracked_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": round(self.inlier_ratio, 6),
            "grid_coverage": round(self.grid_coverage, 6),
            "median_displacement_diag_ratio": round(self.median_displacement_diag_ratio, 6),
            "affine_rotation_deg": None if self.affine_rotation_deg is None else round(self.affine_rotation_deg, 6),
            "affine_scale": None if self.affine_scale is None else round(self.affine_scale, 6),
            "residual_rmse_diag_ratio": (
                None if self.residual_rmse_diag_ratio is None else round(self.residual_rmse_diag_ratio, 6)
            ),
            "reason": self.reason,
            "model": self.model,
            "forward_backward_rmse_px": (
                None if self.forward_backward_rmse_px is None else round(self.forward_backward_rmse_px, 6)
            ),
        }


@dataclass(frozen=True)
class KeyframeDecision:
    selected: bool
    reject_reasons: tuple[str, ...]
    duplicate_score: int | None
    matched_frame_id: str | None
    reference_frame_id: str | None
    bridge: bool
    adaptive_blur_threshold: float
    quality_score: float
    motion: MotionMetrics | None
    component_id: int | None
    bridge_reason: str | None = None


def _invalid_motion(reason: str, tracked_count: int = 0) -> MotionMetrics:
    return MotionMetrics(False, tracked_count, 0, 0.0, 0.0, 0.0, None, None, None, reason)


def make_analysis_gray(frame: np.ndarray, long_edge: int) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be HxWx3 BGR")
    height, width = frame.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("frame dimensions must both be at least 2")
    if long_edge < 2:
        raise ValueError("long_edge must be at least 2")
    scale = float(long_edge) / float(max(width, height))
    target_width = max(2, int(round(width * scale)))
    target_height = max(2, int(round(height * scale)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (target_width, target_height) != (width, height):
        # AREA keeps the metric comparable when source videos arrive at very
        # different resolutions; the analysis image is always the same size.
        gray = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return gray


def calibrated_blur_score(frame: np.ndarray, long_edge: int) -> float:
    gray = make_analysis_gray(frame, long_edge)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _grid_coverage(points: np.ndarray, image_shape: tuple[int, int]) -> float:
    height, width = image_shape
    cells_x = np.clip((points[:, 0] * 4 / max(width, 1)).astype(int), 0, 3)
    cells_y = np.clip((points[:, 1] * 4 / max(height, 1)).astype(int), 0, 3)
    return float(len(set(zip(cells_x.tolist(), cells_y.tolist())))) / 16.0


def _coarse_affine_diagnostics(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
) -> tuple[float | None, float | None]:
    if len(reference_points) < 3:
        return None, None
    design = np.column_stack([reference_points, np.ones(len(reference_points), dtype=np.float32)])
    try:
        coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(design, candidate_points, rcond=None)
    except np.linalg.LinAlgError:
        return None, None
    a, b = float(coefficients[0, 0]), float(coefficients[0, 1])
    scale = float(np.hypot(a, b))
    rotation = float(np.degrees(np.arctan2(b, a)))
    if not math.isfinite(scale) or not math.isfinite(rotation):
        return None, None
    return rotation, scale


def _motion_metrics(
    *,
    model: str,
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    inliers: np.ndarray,
    residuals_px: np.ndarray,
    image_shape: tuple[int, int],
    rotation: float | None,
    scale: float | None,
) -> MotionMetrics:
    tracked_count = len(reference_points)
    inlier_count = int(np.count_nonzero(inliers))
    if not inlier_count:
        return _invalid_motion(f"no_{model}_inliers", tracked_count)
    height, width = image_shape
    diagonal = float(np.hypot(width, height))
    if not math.isfinite(diagonal) or diagonal <= 0:
        return _invalid_motion("invalid_image_shape", tracked_count)
    inlier_residuals = residuals_px[inliers]
    displacement = np.linalg.norm(candidate_points - reference_points, axis=1)
    values = [
        float(np.median(displacement)) / diagonal,
        _grid_coverage(reference_points[inliers], image_shape),
        float(np.sqrt(np.mean(np.square(inlier_residuals)))) / diagonal,
    ]
    if not all(math.isfinite(value) for value in values) or rotation is None or scale is None:
        return _invalid_motion(f"non_finite_{model}_metrics", tracked_count)
    return MotionMetrics(
        True,
        tracked_count,
        inlier_count,
        float(inlier_count) / float(tracked_count),
        values[1],
        values[0],
        rotation,
        scale,
        values[2],
        model=model,
    )


def _affine_motion_metrics(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    image_shape: tuple[int, int],
) -> MotionMetrics:
    cv2.setRNGSeed(0)
    try:
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            reference_points,
            candidate_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
    except cv2.error:
        return _invalid_motion("affine_estimation_failed", len(reference_points))
    if affine is None or inlier_mask is None or not np.all(np.isfinite(affine)):
        return _invalid_motion("affine_estimation_failed", len(reference_points))
    inliers = inlier_mask.reshape(-1).astype(bool)
    predicted = cv2.transform(reference_points.reshape(-1, 1, 2), affine).reshape(-1, 2)
    residuals = np.linalg.norm(predicted - candidate_points, axis=1)
    a, b = float(affine[0, 0]), float(affine[1, 0])
    return _motion_metrics(
        model="affine",
        reference_points=reference_points,
        candidate_points=candidate_points,
        inliers=inliers,
        residuals_px=residuals,
        image_shape=image_shape,
        rotation=float(np.degrees(np.arctan2(b, a))),
        scale=float(np.hypot(a, b)),
    )


def _fundamental_motion_metrics(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    image_shape: tuple[int, int],
    policy: KeyframePolicy,
) -> MotionMetrics:
    cv2.setRNGSeed(0)
    try:
        fundamental, inlier_mask = cv2.findFundamentalMat(
            reference_points,
            candidate_points,
            cv2.FM_RANSAC,
            policy.fundamental_ransac_threshold_px,
            policy.fundamental_ransac_confidence,
        )
    except cv2.error:
        return _invalid_motion("fundamental_estimation_failed", len(reference_points))
    if (
        fundamental is None
        or inlier_mask is None
        or fundamental.shape != (3, 3)
        or not np.all(np.isfinite(fundamental))
    ):
        return _invalid_motion("fundamental_estimation_failed", len(reference_points))
    inliers = inlier_mask.reshape(-1).astype(bool)
    if int(np.count_nonzero(inliers)) < max(8, policy.min_tracked_points):
        return _invalid_motion("insufficient_fundamental_inliers", len(reference_points))
    homogeneous_reference = np.column_stack([reference_points, np.ones(len(reference_points))])
    homogeneous_candidate = np.column_stack([candidate_points, np.ones(len(candidate_points))])
    fundamental = fundamental.astype(np.float64)
    forward = homogeneous_reference @ fundamental.T
    backward = homogeneous_candidate @ fundamental
    numerator = np.sum(homogeneous_candidate * forward, axis=1)
    denominator = (
        np.square(forward[:, 0])
        + np.square(forward[:, 1])
        + np.square(backward[:, 0])
        + np.square(backward[:, 1])
    )
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 1e-12):
        return _invalid_motion("invalid_fundamental_residual", len(reference_points))
    sampson_distances = np.square(numerator) / denominator
    if not np.all(np.isfinite(sampson_distances)) or np.any(sampson_distances < 0):
        return _invalid_motion("invalid_fundamental_residual", len(reference_points))
    rotation, scale = _coarse_affine_diagnostics(reference_points[inliers], candidate_points[inliers])
    return _motion_metrics(
        model="fundamental",
        reference_points=reference_points,
        candidate_points=candidate_points,
        inliers=inliers,
        residuals_px=np.sqrt(sampson_distances),
        image_shape=image_shape,
        rotation=rotation,
        scale=scale,
    )


def _motion_support(metrics: MotionMetrics) -> tuple[float, float, float]:
    residual = metrics.residual_rmse_diag_ratio
    return (
        metrics.inlier_ratio if metrics.valid else -1.0,
        metrics.grid_coverage if metrics.valid else -1.0,
        -(residual if residual is not None and math.isfinite(residual) else float("inf")),
    )


def estimate_motion_from_tracks(
    reference_points: np.ndarray,
    candidate_points: np.ndarray,
    image_shape: tuple[int, int],
    policy: KeyframePolicy,
) -> MotionMetrics:
    reference = np.asarray(reference_points, dtype=np.float32).reshape(-1, 2)
    candidate = np.asarray(candidate_points, dtype=np.float32).reshape(-1, 2)
    if len(reference) != len(candidate) or len(reference) < policy.min_tracked_points:
        return _invalid_motion("insufficient_tracked_points", min(len(reference), len(candidate)))
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        return _invalid_motion("non_finite_tracks", len(reference))
    affine = _affine_motion_metrics(reference, candidate, image_shape)
    if policy.motion_model == "affine":
        return affine
    fundamental = _fundamental_motion_metrics(reference, candidate, image_shape, policy)
    if motion_passes_hard_gate(affine, policy)[0]:
        return affine
    if motion_passes_hard_gate(fundamental, policy)[0]:
        return fundamental
    return max((affine, fundamental), key=_motion_support)


def estimate_pair_motion(
    reference_gray: np.ndarray,
    candidate_gray: np.ndarray,
    policy: KeyframePolicy,
) -> MotionMetrics:
    if reference_gray.shape != candidate_gray.shape:
        candidate_gray = cv2.resize(candidate_gray, (reference_gray.shape[1], reference_gray.shape[0]))
    points = cv2.goodFeaturesToTrack(reference_gray, maxCorners=800, qualityLevel=0.01, minDistance=5)
    if points is None or len(points) < policy.min_tracked_points:
        return _invalid_motion("insufficient_features", 0 if points is None else len(points))
    forward, forward_status, _forward_error = cv2.calcOpticalFlowPyrLK(reference_gray, candidate_gray, points, None)
    if forward is None or forward_status is None:
        return _invalid_motion("optical_flow_failed")
    reference = points.reshape(-1, 2)
    candidate = forward.reshape(-1, 2)
    forward_mask = forward_status.reshape(-1).astype(bool) & np.all(np.isfinite(candidate), axis=1)
    reference = reference[forward_mask]
    candidate = candidate[forward_mask]
    if len(reference) < policy.min_tracked_points:
        return _invalid_motion("insufficient_tracked_points", len(reference))
    backward, backward_status, _backward_error = cv2.calcOpticalFlowPyrLK(
        candidate_gray, reference_gray, candidate.reshape(-1, 1, 2), None
    )
    if backward is None or backward_status is None:
        return _invalid_motion("backward_optical_flow_failed", len(reference))
    returned = backward.reshape(-1, 2)
    backward_mask = backward_status.reshape(-1).astype(bool) & np.all(np.isfinite(returned), axis=1)
    errors = np.linalg.norm(returned - reference, axis=1)
    retained = backward_mask & np.isfinite(errors) & (errors <= policy.flow_forward_backward_max_error_px)
    reference = reference[retained]
    candidate = candidate[retained]
    errors = errors[retained]
    if len(reference) < policy.min_tracked_points:
        return _invalid_motion("insufficient_bidirectional_tracks", len(reference))
    metrics = estimate_motion_from_tracks(reference, candidate, reference_gray.shape[:2], policy)
    return replace(metrics, forward_backward_rmse_px=float(np.sqrt(np.mean(np.square(errors)))))


def motion_passes_hard_gate(metrics: MotionMetrics, policy: KeyframePolicy) -> tuple[bool, tuple[str, ...]]:
    if not metrics.valid:
        return False, (metrics.reason or "motion_invalid",)
    reasons: list[str] = []
    if metrics.tracked_count < policy.min_tracked_points:
        reasons.append("insufficient_tracked_points")
    if metrics.inlier_ratio < policy.min_motion_inlier_ratio:
        reasons.append("motion_inlier_ratio")
    if metrics.grid_coverage < policy.min_motion_grid_coverage:
        reasons.append("motion_grid_coverage")
    if (
        metrics.residual_rmse_diag_ratio is None
        or metrics.residual_rmse_diag_ratio > policy.max_motion_residual_diag_ratio
    ):
        reasons.append("motion_residual")
    if metrics.median_displacement_diag_ratio > policy.max_median_displacement_diag_ratio:
        reasons.append("motion_displacement")
    if metrics.affine_rotation_deg is None or abs(metrics.affine_rotation_deg) > policy.max_affine_rotation_deg:
        reasons.append("motion_rotation")
    if metrics.affine_scale is None or not policy.min_affine_scale <= metrics.affine_scale <= policy.max_affine_scale:
        reasons.append("motion_scale")
    return not reasons, tuple(reasons)


def motion_passes_bridge_gate(metrics: MotionMetrics, policy: KeyframePolicy) -> bool:
    """A bridge can relax quality, never the basic trackability safety floor."""
    return bool(
        metrics.valid
        and metrics.tracked_count >= policy.min_tracked_points
        and metrics.inlier_ratio >= policy.min_motion_inlier_ratio * 0.8
        and metrics.grid_coverage >= policy.min_motion_grid_coverage * 0.8
        and metrics.residual_rmse_diag_ratio is not None
        and metrics.residual_rmse_diag_ratio <= policy.max_motion_residual_diag_ratio * 1.5
        and metrics.median_displacement_diag_ratio <= policy.max_median_displacement_diag_ratio
        and metrics.affine_rotation_deg is not None
        and abs(metrics.affine_rotation_deg) <= policy.max_affine_rotation_deg
        and metrics.affine_scale is not None
        and policy.min_affine_scale <= metrics.affine_scale <= policy.max_affine_scale
    )


def hash_distance(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def _bridge_eligible(frame: ScannedFrame, policy: KeyframePolicy, adaptive_blur_threshold: float) -> bool:
    bridge_threshold = max(policy.blur_floor, adaptive_blur_threshold * policy.bridge_min_blur_ratio)
    return bool(
        frame.calibrated_blur_score >= bridge_threshold
        and frame.overexposed_ratio <= policy.max_overexposed_ratio
        and frame.underexposed_ratio <= policy.max_underexposed_ratio
    )


def _path_better(
    candidate: tuple[int, float, tuple[int, ...]],
    current: tuple[int, float, tuple[int, ...]],
) -> bool:
    if candidate[0] != current[0]:
        return candidate[0] < current[0]
    if not math.isclose(candidate[1], current[1]):
        return candidate[1] > current[1]
    return candidate[2] < current[2]


def _find_coverage_path(
    scanned: list[ScannedFrame],
    decisions: list[KeyframeDecision],
    left_index: int,
    right_index: int,
    policy: KeyframePolicy,
) -> tuple[list[int] | None, dict[tuple[int, int], MotionMetrics]]:
    adaptive = decisions[left_index].adaptive_blur_threshold
    eligible = {
        index
        for index in range(left_index, right_index + 1)
        if decisions[index].selected or _bridge_eligible(scanned[index], policy, adaptive)
    }
    if left_index not in eligible or right_index not in eligible:
        return None, {}
    motion_cache: dict[tuple[int, int], MotionMetrics] = {}

    def edge_motion(left: int, right: int) -> MotionMetrics:
        key = (left, right)
        if key not in motion_cache:
            motion_cache[key] = estimate_pair_motion(scanned[left].flow_gray, scanned[right].flow_gray, policy)
        return motion_cache[key]

    routes: dict[int, tuple[tuple[int, float, tuple[int, ...]], list[int]]] = {
        left_index: ((0, 1.0, (scanned[left_index].frame_index,)), [left_index])
    }
    for right in sorted(index for index in eligible if index > left_index):
        best: tuple[tuple[int, float, tuple[int, ...]], list[int]] | None = None
        for left in sorted(routes):
            if left >= right:
                continue
            delta = round(scanned[right].timestamp_sec - scanned[left].timestamp_sec, 3)
            if delta <= 0 or delta > round(policy.max_gap_sec, 3):
                continue
            motion = edge_motion(left, right)
            if not motion_passes_hard_gate(motion, policy)[0]:
                continue
            previous_score, previous_path = routes[left]
            normalized_blur = scanned[right].calibrated_blur_score / max(adaptive, 1e-6)
            score = (
                previous_score[0] + (0 if decisions[right].selected else 1),
                min(previous_score[1], normalized_blur),
                previous_score[2] + (scanned[right].frame_index,),
            )
            candidate = (score, previous_path + [right])
            if best is None or _path_better(score, best[0]):
                best = candidate
        if best is not None:
            routes[right] = best
    route = routes.get(right_index)
    return (None if route is None else route[1]), motion_cache


def _apply_coverage_path(
    scanned: list[ScannedFrame],
    decisions: list[KeyframeDecision],
    path: list[int],
    policy: KeyframePolicy,
    motion_cache: dict[tuple[int, int], MotionMetrics],
) -> list[KeyframeDecision]:
    updated = list(decisions)
    original_selected = {index for index, decision in enumerate(decisions) if decision.selected}
    adaptive = decisions[path[0]].adaptive_blur_threshold
    for position, index in enumerate(path[1:], start=1):
        previous = path[position - 1]
        motion = motion_cache.get((previous, index))
        if motion is None:
            motion = estimate_pair_motion(scanned[previous].flow_gray, scanned[index].flow_gray, policy)
        if not motion_passes_hard_gate(motion, policy)[0]:
            continue
        is_new = index not in original_selected
        is_endpoint = position == len(path) - 1
        is_bridge = is_new and (
            not is_endpoint or scanned[index].calibrated_blur_score < adaptive
        )
        updated[index] = replace(
            updated[index],
            selected=True,
            reject_reasons=(),
            duplicate_score=None,
            matched_frame_id=None,
            reference_frame_id=scanned[previous].id,
            bridge=is_bridge,
            bridge_reason="coverage_graph" if is_bridge else None,
            motion=motion,
            component_id=0,
        )
    return updated


def repair_coverage_gaps(
    scanned: list[ScannedFrame],
    decisions: list[KeyframeDecision],
    policy: KeyframePolicy,
    segment_start_sec: float,
    segment_end_sec: float,
) -> list[KeyframeDecision]:
    if not scanned or not decisions:
        return decisions
    updated = list(decisions)
    max_gap = round(policy.max_gap_sec, 3)
    adaptive = decisions[0].adaptive_blur_threshold

    def selected_indices() -> list[int]:
        return [index for index, decision in enumerate(updated) if decision.selected]

    def apply_path(left: int, right: int) -> bool:
        if (
            policy.max_bridge_window_sec > 0
            and scanned[right].timestamp_sec - scanned[left].timestamp_sec
            > policy.max_bridge_window_sec
        ):
            return False
        path, motion_cache = _find_coverage_path(scanned, updated, left, right, policy)
        if path is None:
            return False
        updated[:] = _apply_coverage_path(scanned, updated, path, policy, motion_cache)
        return any(updated[index].selected and not decisions[index].selected for index in path)

    # A virtual start boundary can select the earliest safe candidate without
    # inventing an incoming motion edge.
    selected = selected_indices()
    if not selected:
        candidates = [
            index for index, frame in enumerate(scanned)
            if frame.timestamp_sec - segment_start_sec <= max_gap
            and _bridge_eligible(frame, policy, adaptive)
        ]
        if candidates:
            index = min(candidates)
            updated[index] = replace(
                updated[index], selected=True, reject_reasons=(), duplicate_score=None,
                matched_frame_id=None, reference_frame_id=None,
                bridge=scanned[index].calibrated_blur_score < adaptive,
                bridge_reason="coverage_graph" if scanned[index].calibrated_blur_score < adaptive else None,
                motion=None, component_id=0,
            )
    else:
        first = selected[0]
        if scanned[first].timestamp_sec - segment_start_sec > max_gap:
            candidates = [
                index for index, frame in enumerate(scanned[:first])
                if frame.timestamp_sec - segment_start_sec <= max_gap
                and _bridge_eligible(frame, policy, adaptive)
            ]
            for index in sorted(candidates):
                original = updated[index]
                updated[index] = replace(
                    updated[index], selected=True, reject_reasons=(), duplicate_score=None,
                    matched_frame_id=None, reference_frame_id=None,
                    bridge=scanned[index].calibrated_blur_score < adaptive,
                    bridge_reason="coverage_graph" if scanned[index].calibrated_blur_score < adaptive else None,
                    motion=None, component_id=0,
                )
                if apply_path(index, first):
                    break
                updated[index] = original

    # Promote deterministic selected-to-selected paths until no repairable gap remains.
    for _ in range(len(scanned) + 1):
        changed = False
        selected = selected_indices()
        for left, right in zip(selected, selected[1:]):
            if round(scanned[right].timestamp_sec - scanned[left].timestamp_sec, 3) > max_gap:
                changed = apply_path(left, right) or changed
        if not changed:
            break

    selected = selected_indices()
    if selected and segment_end_sec - scanned[selected[-1]].timestamp_sec > max_gap:
        candidates = [
            index for index, frame in enumerate(scanned[selected[-1] + 1:], start=selected[-1] + 1)
            if segment_end_sec - frame.timestamp_sec <= max_gap
            and _bridge_eligible(frame, policy, adaptive)
        ]
        for endpoint in sorted(candidates, reverse=True):
            if apply_path(selected[-1], endpoint):
                break
    return updated


def select_coverage_frames(
    scanned: list[ScannedFrame],
    policy: KeyframePolicy,
    *,
    segment_start_sec: float | None = None,
    segment_end_sec: float | None = None,
) -> list[KeyframeDecision]:
    if not scanned:
        return []
    adaptive_blur_threshold = max(
        policy.blur_floor,
        float(np.percentile([frame.calibrated_blur_score for frame in scanned], policy.adaptive_blur_percentile)),
    )
    decisions: list[KeyframeDecision] = []
    selected_indices: list[int] = []
    for index, frame in enumerate(scanned):
        reasons: list[str] = []
        duplicate_score: int | None = None
        matched_frame_id: str | None = None
        reference_index = selected_indices[-1] if selected_indices else None
        reference_frame_id = scanned[reference_index].id if reference_index is not None else None
        if frame.calibrated_blur_score < adaptive_blur_threshold:
            reasons.append("blur")
        if frame.overexposed_ratio > policy.max_overexposed_ratio:
            reasons.append("overexposed")
        if frame.underexposed_ratio > policy.max_underexposed_ratio:
            reasons.append("underexposed")
        eligible_matches = [
            selected_index
            for selected_index in selected_indices
            if frame.timestamp_sec - scanned[selected_index].timestamp_sec <= policy.duplicate_window_sec
        ]
        if eligible_matches:
            duplicate_score, matched_index = min(
                (hash_distance(frame.average_hash_bits, scanned[selected_index].average_hash_bits), selected_index)
                for selected_index in eligible_matches
            )
            matched_frame_id = scanned[matched_index].id
        gap = 0.0 if reference_index is None else frame.timestamp_sec - scanned[reference_index].timestamp_sec
        motion = (
            None
            if reference_index is None
            else estimate_pair_motion(scanned[reference_index].flow_gray, frame.flow_gray, policy)
        )
        motion_ok, motion_reasons = (True, ()) if motion is None else motion_passes_hard_gate(motion, policy)
        bridge = False
        if (
            duplicate_score is not None
            and duplicate_score <= policy.duplicate_hash_threshold
            and gap < policy.target_gap_sec
        ):
            reasons.append("duplicate")
        if reference_index is not None and gap < policy.min_gap_sec:
            reasons.append("cadence")
        if reference_index is not None and gap >= policy.min_gap_sec and not motion_ok:
            reasons.extend(motion_reasons)
        selected = not reasons
        if selected:
            selected_indices.append(index)
        quality_score = min(1.0, frame.calibrated_blur_score / max(adaptive_blur_threshold, 1e-6))
        decisions.append(
            KeyframeDecision(
                selected=selected,
                reject_reasons=tuple(dict.fromkeys(reasons)),
                duplicate_score=duplicate_score,
                matched_frame_id=matched_frame_id if duplicate_score is not None else None,
                reference_frame_id=reference_frame_id,
                bridge=bridge,
                adaptive_blur_threshold=adaptive_blur_threshold,
                quality_score=quality_score,
                motion=motion,
                component_id=0 if selected else None,
            )
        )
    return repair_coverage_gaps(
        scanned,
        decisions,
        policy,
        scanned[0].timestamp_sec if segment_start_sec is None else segment_start_sec,
        scanned[-1].timestamp_sec if segment_end_sec is None else segment_end_sec,
    )


def validate_keyframe_settings(settings: dict[str, Any]) -> None:
    if settings["keyframe_policy"] not in {"legacy", "coverage_v1"}:
        raise ValueError("keyframe_policy must be one of: coverage_v1, legacy")
    numeric_names = (
        "adaptive_blur_percentile",
        "selection_min_gap_sec",
        "selection_target_gap_sec",
        "selection_max_gap_sec",
        "duplicate_window_sec",
        "min_motion_inlier_ratio",
        "min_motion_grid_coverage",
        "max_motion_residual_diag_ratio",
        "max_median_displacement_diag_ratio",
        "max_affine_rotation_deg",
        "min_affine_scale",
        "max_affine_scale",
        "flow_forward_backward_max_error_px",
        "fundamental_ransac_threshold_px",
        "fundamental_ransac_confidence",
        "bridge_min_blur_ratio",
        "max_bridge_window_sec",
        "max_bridge_fraction",
    )
    for name in numeric_names:
        if not math.isfinite(float(settings[name])):
            raise ValueError(f"{name} must be finite")
    for name in ("quality_analysis_long_edge", "flow_analysis_long_edge", "min_tracked_points"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{name} must be an integer of at least 2")
    if not 0.0 <= settings["adaptive_blur_percentile"] <= 100.0:
        raise ValueError("adaptive_blur_percentile must be between 0 and 100")
    for name in ("min_motion_inlier_ratio", "min_motion_grid_coverage"):
        if not 0.0 <= settings[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not (
        0
        < settings["selection_min_gap_sec"]
        <= settings["selection_target_gap_sec"]
        <= settings["selection_max_gap_sec"]
    ):
        raise ValueError("selection gap settings must satisfy 0 < min <= target <= max")
    if (
        settings["keyframe_policy"] == "coverage_v1"
        and settings["selection_max_gap_sec"] < 1.0 / settings["target_fps"]
    ):
        raise ValueError("selection_max_gap_sec must be at least one target-fps sampling interval")
    for name in (
        "duplicate_window_sec",
        "max_motion_residual_diag_ratio",
        "max_median_displacement_diag_ratio",
        "max_affine_rotation_deg",
    ):
        if settings[name] <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if not 0 < settings["min_affine_scale"] <= 1.0 <= settings["max_affine_scale"]:
        raise ValueError("affine scale settings must satisfy 0 < min <= 1 <= max")
    if settings.get("motion_model", "affine") not in {"affine", "auto"}:
        raise ValueError("motion_model must be one of: affine, auto")
    if settings["flow_forward_backward_max_error_px"] <= 0:
        raise ValueError("flow_forward_backward_max_error_px must be greater than zero")
    if settings["fundamental_ransac_threshold_px"] <= 0:
        raise ValueError("fundamental_ransac_threshold_px must be greater than zero")
    if not 0 < settings["fundamental_ransac_confidence"] <= 1:
        raise ValueError("fundamental_ransac_confidence must be between 0 and 1")
    if not 0 < settings["bridge_min_blur_ratio"] <= 1:
        raise ValueError("bridge_min_blur_ratio must be between 0 and 1")
    if settings["max_bridge_window_sec"] < 0:
        raise ValueError("max_bridge_window_sec must be at least zero")
    if not 0 <= settings["max_bridge_fraction"] <= 1:
        raise ValueError("max_bridge_fraction must be between 0 and 1")
