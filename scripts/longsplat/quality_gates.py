"""Immutable quality gate configuration for LongSplat pipeline.

Defines :class:`PoseGateConfig`, :class:`VdaGateConfig`,
:class:`PlyGateConfig`, and the top-level :class:`QualityGateConfig` that
aggregates them.  Also provides :func:`audit_pose_quality` for post-training
pose verification before conversion.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

GateMode = Literal["off", "observe", "enforce"]
_VALID_MODES = frozenset({"off", "observe", "enforce"})


def _check_mode(value: str) -> str:
    if value not in _VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {value!r}")
    return value


def _check_float_nonnegative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")
    return value


def _check_ratio(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


@dataclass(frozen=True)
class PoseGateConfig:
    """Thresholds for online incremental PnP pose acceptance."""

    mode: Literal["off", "observe", "enforce"] = "off"
    min_match_count: int = 128
    min_inlier_count: int = 64
    min_inlier_ratio: float = 0.80
    max_reprojection_rmse_px: float = 2.0
    min_grid_coverage: float = 0.375
    min_positive_depth_ratio: float = 0.95
    max_rotation_step_deg: float = 25.0
    max_translation_step_ratio: float = 6.0
    reference_lookback: int = 3

    def __post_init__(self) -> None:
        _check_mode(self.mode)
        if self.min_match_count < 0:
            raise ValueError(f"min_match_count must be >= 0, got {self.min_match_count}")
        if self.min_inlier_count < 0:
            raise ValueError(f"min_inlier_count must be >= 0, got {self.min_inlier_count}")
        _check_ratio(self.min_inlier_ratio, "min_inlier_ratio")
        _check_float_nonnegative(self.max_reprojection_rmse_px, "max_reprojection_rmse_px")
        _check_ratio(self.min_grid_coverage, "min_grid_coverage")
        _check_ratio(self.min_positive_depth_ratio, "min_positive_depth_ratio")
        _check_float_nonnegative(self.max_rotation_step_deg, "max_rotation_step_deg")
        _check_float_nonnegative(self.max_translation_step_ratio, "max_translation_step_ratio")
        if self.reference_lookback < 0:
            raise ValueError(f"reference_lookback must be >= 0, got {self.reference_lookback}")


@dataclass(frozen=True)
class VdaGateConfig:
    """Thresholds for online VDA depth alignment acceptance."""

    mode: Literal["off", "observe", "enforce"] = "off"
    min_correlation: float = 0.90
    min_inlier_ratio: float = 0.95
    max_normalized_rmse: float = 0.10
    min_aligned_fraction: float = 0.98

    def __post_init__(self) -> None:
        _check_mode(self.mode)
        _check_ratio(self.min_correlation, "min_correlation")
        _check_ratio(self.min_inlier_ratio, "min_inlier_ratio")
        _check_float_nonnegative(self.max_normalized_rmse, "max_normalized_rmse")
        _check_ratio(self.min_aligned_fraction, "min_aligned_fraction")


@dataclass(frozen=True)
class PlyGateConfig:
    """Thresholds for post-conversion PLY publication acceptance."""

    mode: Literal["off", "observe", "enforce"] = "off"
    min_effective_fraction: float = 0.35
    max_anisotropy_q99: float = 50.0
    min_unit_quaternion_fraction: float = 0.9999

    def __post_init__(self) -> None:
        _check_mode(self.mode)
        _check_ratio(self.min_effective_fraction, "min_effective_fraction")
        _check_float_nonnegative(self.max_anisotropy_q99, "max_anisotropy_q99")
        _check_ratio(self.min_unit_quaternion_fraction, "min_unit_quaternion_fraction")


@dataclass(frozen=True)
class QualityGateConfig:
    """Aggregate quality gate configuration for a LongSplat run."""

    pose: PoseGateConfig = field(default_factory=PoseGateConfig)
    vda: VdaGateConfig = field(default_factory=VdaGateConfig)
    ply: PlyGateConfig = field(default_factory=PlyGateConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityGateConfig:
        """Parse from a JSON-safe dict, rejecting unknown keys."""
        known = {"pose", "vda", "ply"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown quality_gates keys: {sorted(unknown)}")
        pose_data = data.get("pose", {})
        vda_data = data.get("vda", {})
        ply_data = data.get("ply", {})
        return cls(
            pose=PoseGateConfig(**pose_data) if isinstance(pose_data, dict) else PoseGateConfig(),
            vda=VdaGateConfig(**vda_data) if isinstance(vda_data, dict) else VdaGateConfig(),
            ply=PlyGateConfig(**ply_data) if isinstance(ply_data, dict) else PlyGateConfig(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose": {
                "mode": self.pose.mode,
                "min_match_count": self.pose.min_match_count,
                "min_inlier_count": self.pose.min_inlier_count,
                "min_inlier_ratio": self.pose.min_inlier_ratio,
                "max_reprojection_rmse_px": self.pose.max_reprojection_rmse_px,
                "min_grid_coverage": self.pose.min_grid_coverage,
                "min_positive_depth_ratio": self.pose.min_positive_depth_ratio,
                "max_rotation_step_deg": self.pose.max_rotation_step_deg,
                "max_translation_step_ratio": self.pose.max_translation_step_ratio,
                "reference_lookback": self.pose.reference_lookback,
            },
            "vda": {
                "mode": self.vda.mode,
                "min_correlation": self.vda.min_correlation,
                "min_inlier_ratio": self.vda.min_inlier_ratio,
                "max_normalized_rmse": self.vda.max_normalized_rmse,
                "min_aligned_fraction": self.vda.min_aligned_fraction,
            },
            "ply": {
                "mode": self.ply.mode,
                "min_effective_fraction": self.ply.min_effective_fraction,
                "max_anisotropy_q99": self.ply.max_anisotropy_q99,
                "min_unit_quaternion_fraction": self.ply.min_unit_quaternion_fraction,
            },
        }


# ---------------------------------------------------------------------------
# Post-training pose audit
# ---------------------------------------------------------------------------

_CHKPNT_RE = re.compile(r"chkpnt(\d+)\.pth$")


def audit_pose_quality(
    cameras_path: Path,
    pose_telemetry: dict[str, Any],
    *,
    model_path: Path | None = None,
    expected_native_checkpoint_iteration: int | None = None,
    max_orthogonality_error: float = 1e-4,
    max_determinant_error: float = 1e-4,
    max_rotation_step_deg: float = 25.0,
    max_translation_step_ratio: float = 6.0,
) -> dict[str, Any]:
    """Audit camera poses post-training, before conversion.

    Hard gates verified:
    - All R/T finite
    - max orthogonality error ≤ threshold
    - max |det(R)-1| ≤ threshold
    - max rotation step ≤ threshold
    - max translation step / median step ≤ threshold
    - accepted terminal frame count equals train camera count
    - native checkpoint exists at expected iteration (when configured)

    Returns a dict with ``passed``, ``reasons`` (failure reasons),
    ``trajectory``, and ``telemetry``.
    """
    from .quality_metrics import analyze_camera_trajectory

    reasons: list[str] = []
    cameras_path = Path(cameras_path)

    # --- 1. Trajectory audit ---
    trajectory = analyze_camera_trajectory(cameras_path)

    if not trajectory.get("all_rotations_finite", False):
        reasons.append("non-finite rotation matrix detected")
    if not trajectory.get("all_positions_finite", False):
        reasons.append("non-finite position vector detected")

    max_ortho = trajectory.get("max_orthogonality_error")
    if max_ortho is not None and max_ortho > max_orthogonality_error:
        reasons.append(
            f"max orthogonality error {max_ortho:.2e} > {max_orthogonality_error:.1e}"
        )

    max_det = trajectory.get("max_determinant_error")
    if max_det is not None and max_det > max_determinant_error:
        reasons.append(
            f"max determinant error {max_det:.2e} > {max_determinant_error:.1e}"
        )

    max_rot = trajectory.get("max_rotation_step_deg", 0.0) or 0.0
    if max_rot > max_rotation_step_deg:
        reasons.append(
            f"max rotation step {max_rot:.2f}° > {max_rotation_step_deg}°"
        )

    # Translation step ratio: max consecutive step / median step
    max_trans = trajectory.get("max_translation_step", 0.0) or 0.0
    if max_trans > 0:
        cam_count = trajectory.get("camera_count", 0)
        if cam_count >= 2:
            # Compute per-pair steps for median reference
            with open(cameras_path, encoding="utf-8") as fh:
                cameras = json.load(fh)
            positions = []
            for cam in cameras:
                t = np.array(
                    cam.get("T", cam.get("position")), dtype=np.float64
                )
                if np.all(np.isfinite(t)):
                    positions.append(t)
            if len(positions) >= 2:
                steps = [
                    float(np.linalg.norm(positions[i] - positions[i - 1]))
                    for i in range(1, len(positions))
                ]
                median_step = float(np.median(steps)) if steps else 1.0
                if median_step > 1e-12:
                    ratio = max_trans / median_step
                    if ratio > max_translation_step_ratio:
                        reasons.append(
                            f"max translation step ratio {ratio:.2f} "
                            f"> {max_translation_step_ratio}"
                        )

    # --- 2. Camera acceptance check ---
    accepted_count = pose_telemetry.get("accepted_camera_count", 0)
    camera_count = trajectory.get("camera_count", 0)
    if accepted_count != camera_count:
        reasons.append(
            f"accepted cameras {accepted_count} != train cameras {camera_count}"
        )

    # --- 3. Native checkpoint check ---
    if expected_native_checkpoint_iteration is not None and model_path is not None:
        mp = Path(model_path)
        found_iter: int | None = None
        if mp.is_dir():
            for child in mp.iterdir():
                m = _CHKPNT_RE.match(child.name)
                if m:
                    found_iter = int(m.group(1))
                    break
        if found_iter is None:
            reasons.append(
                f"native checkpoint not found in {mp} "
                f"(expected iteration {expected_native_checkpoint_iteration})"
            )
        elif found_iter != expected_native_checkpoint_iteration:
            reasons.append(
                f"native checkpoint iteration {found_iter} "
                f"!= expected {expected_native_checkpoint_iteration}"
            )

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "trajectory": trajectory,
        "telemetry": {
            "accepted_camera_count": accepted_count,
            "camera_count": camera_count,
        },
    }
