"""Immutable quality gate configuration for LongSplat pipeline.

Defines :class:`PoseGateConfig`, :class:`VdaGateConfig`,
:class:`PlyGateConfig`, and the top-level :class:`QualityGateConfig` that
aggregates them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

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
