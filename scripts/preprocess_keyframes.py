"""Coverage-aware keyframe analysis and selection primitives.

The module deliberately contains no video I/O or file writes.  The preprocess
entrypoint owns both so the policy can be unit-tested with synthetic images.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def estimate_pair_motion(reference_gray: np.ndarray, candidate_gray: np.ndarray, policy: KeyframePolicy) -> MotionMetrics:
    if reference_gray.shape != candidate_gray.shape:
        candidate_gray = cv2.resize(candidate_gray, (reference_gray.shape[1], reference_gray.shape[0]))
    points = cv2.goodFeaturesToTrack(reference_gray, maxCorners=800, qualityLevel=0.01, minDistance=5)
    if points is None or len(points) < policy.min_tracked_points:
        return _invalid_motion("insufficient_features", 0 if points is None else len(points))
    candidate_points, status, _error = cv2.calcOpticalFlowPyrLK(reference_gray, candidate_gray, points, None)
    if candidate_points is None or status is None:
        return _invalid_motion("optical_flow_failed")
    tracked_mask = status.reshape(-1).astype(bool)
    reference_points = points.reshape(-1, 2)[tracked_mask]
    candidate_points = candidate_points.reshape(-1, 2)[tracked_mask]
    tracked_count = len(reference_points)
    if tracked_count < policy.min_tracked_points:
        return _invalid_motion("insufficient_tracked_points", tracked_count)
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        reference_points,
        candidate_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if affine is None or inlier_mask is None:
        return _invalid_motion("affine_estimation_failed", tracked_count)
    inliers = inlier_mask.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count) / float(tracked_count)
    if not inlier_count:
        return _invalid_motion("no_affine_inliers", tracked_count)
    height, width = reference_gray.shape[:2]
    diagonal = max(float(np.hypot(width, height)), 1.0)
    displacement = np.linalg.norm(candidate_points - reference_points, axis=1)
    median_displacement = float(np.median(displacement)) / diagonal
    inlier_reference = reference_points[inliers]
    cells_x = np.clip((inlier_reference[:, 0] * 4 / max(width, 1)).astype(int), 0, 3)
    cells_y = np.clip((inlier_reference[:, 1] * 4 / max(height, 1)).astype(int), 0, 3)
    grid_coverage = float(len(set(zip(cells_x.tolist(), cells_y.tolist())))) / 16.0
    predicted = cv2.transform(reference_points[inliers].reshape(-1, 1, 2), affine).reshape(-1, 2)
    residual = np.linalg.norm(predicted - candidate_points[inliers], axis=1)
    residual_rmse = float(np.sqrt(np.mean(np.square(residual)))) / diagonal
    a, b = float(affine[0, 0]), float(affine[1, 0])
    scale = float(np.hypot(a, b))
    rotation = float(np.degrees(np.arctan2(b, a)))
    return MotionMetrics(
        True,
        tracked_count,
        inlier_count,
        inlier_ratio,
        grid_coverage,
        median_displacement,
        rotation,
        scale,
        residual_rmse,
    )


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
    if metrics.residual_rmse_diag_ratio is None or metrics.residual_rmse_diag_ratio > policy.max_motion_residual_diag_ratio:
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


def select_coverage_frames(scanned: list[ScannedFrame], policy: KeyframePolicy) -> list[KeyframeDecision]:
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
        motion = None if reference_index is None else estimate_pair_motion(scanned[reference_index].flow_gray, frame.flow_gray, policy)
        motion_ok, motion_reasons = (True, ()) if motion is None else motion_passes_hard_gate(motion, policy)
        bridge = False
        if duplicate_score is not None and duplicate_score <= policy.duplicate_hash_threshold and gap < policy.target_gap_sec:
            reasons.append("duplicate")
        if reference_index is not None and gap < policy.min_gap_sec:
            reasons.append("cadence")
        if reference_index is not None and gap >= policy.min_gap_sec and not motion_ok:
            if gap >= policy.max_gap_sec and motion is not None and motion_passes_bridge_gate(motion, policy):
                bridge = True
            elif gap >= policy.max_gap_sec and index > 0:
                local_motion = estimate_pair_motion(scanned[index - 1].flow_gray, frame.flow_gray, policy)
                local_motion_ok, _local_motion_reasons = motion_passes_hard_gate(local_motion, policy)
                if local_motion_ok or motion_passes_bridge_gate(local_motion, policy):
                    motion = local_motion
                    # Reinitialization may reference a rejected scanned frame;
                    # the selected current frame becomes the next stable anchor.
                    reference_frame_id = scanned[index - 1].id
                    bridge = True
                else:
                    reasons.extend(motion_reasons)
            else:
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
    return decisions


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
    if not 0 < settings["selection_min_gap_sec"] <= settings["selection_target_gap_sec"] <= settings["selection_max_gap_sec"]:
        raise ValueError("selection gap settings must satisfy 0 < min <= target <= max")
    if settings["keyframe_policy"] == "coverage_v1" and settings["selection_max_gap_sec"] < 1.0 / settings["target_fps"]:
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
