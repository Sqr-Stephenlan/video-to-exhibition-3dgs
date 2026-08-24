"""Deterministic, metric-aware frame selection bound to canonical pixels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .pipeline_contract import PipelineBlocked, stable_sha256


FRAME_SELECTION_SCHEMA = "frame-selection-v1"


def _finite(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class FrameSelectionConfig:
    """Selection limits; none of these are tied to a historical video."""

    target_fps: float = 2.0
    min_frames: int = 12
    max_frames: int = 240
    max_overexposed_ratio: float = 0.35
    max_underexposed_ratio: float = 0.35
    duplicate_hash_threshold: int = 3

    def __post_init__(self) -> None:
        if self.target_fps <= 0 or self.min_frames <= 0 or self.max_frames < self.min_frames:
            raise ValueError("frame budget limits must be positive and ordered")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_fps": self.target_fps,
            "min_frames": self.min_frames,
            "max_frames": self.max_frames,
            "max_overexposed_ratio": self.max_overexposed_ratio,
            "max_underexposed_ratio": self.max_underexposed_ratio,
            "duplicate_hash_threshold": self.duplicate_hash_threshold,
        }


@dataclass(frozen=True)
class FrameMetric:
    frame_id: str
    timestamp_sec: float
    frame_index: int
    path: str | None
    width: int
    height: int
    sha256: str | None = None
    sharpness: float | None = None
    overexposed_ratio: float | None = None
    underexposed_ratio: float | None = None
    motion_score: float | None = None
    coverage_score: float | None = None
    duplicate_score: int | None = None
    preprocess_selected: bool | None = None
    preprocess_reject_reasons: tuple[str, ...] = ()
    provenance: str = "unknown"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], index: int) -> "FrameMetric":
        keyframe = value.get("keyframe") if isinstance(value.get("keyframe"), Mapping) else {}
        motion_value = value.get("motion_score")
        if motion_value is None and isinstance(keyframe, Mapping):
            motion = keyframe.get("motion")
            if isinstance(motion, Mapping):
                motion_value = motion.get("score", motion.get("inlier_ratio"))
        reasons = value.get("reject_reasons", value.get("rejected_reasons", ()))
        if isinstance(reasons, str):
            reasons = (reasons,)
        if not isinstance(reasons, (list, tuple)):
            reasons = ()
        frame_id = value.get("frame_id", value.get("id", f"frame_{index:06d}"))
        if not isinstance(frame_id, str) or not frame_id:
            raise PipelineBlocked(f"frame {index} has no stable frame id")
        timestamp = _finite(value.get("timestamp_sec", value.get("timestamp")))
        if timestamp is None or timestamp < 0:
            raise PipelineBlocked(f"frame {frame_id} has invalid timestamp")
        frame_index = value.get("frame_index", value.get("producer_frame_index", index))
        if not isinstance(frame_index, int) or frame_index < 0:
            raise PipelineBlocked(f"frame {frame_id} has invalid frame index")
        width = value.get("width")
        height = value.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise PipelineBlocked(f"frame {frame_id} has invalid dimensions")
        duplicate = value.get("duplicate_score")
        if duplicate is not None:
            try:
                duplicate = int(duplicate)
            except (TypeError, ValueError) as exc:
                raise PipelineBlocked(f"frame {frame_id} has invalid duplicate score") from exc
        return cls(
            frame_id=frame_id,
            timestamp_sec=timestamp,
            frame_index=frame_index,
            path=value.get("path"),
            width=width,
            height=height,
            sha256=value.get("sha256"),
            sharpness=_finite(value.get("sharpness", value.get("blur_score"))),
            overexposed_ratio=_finite(value.get("overexposed_ratio")),
            underexposed_ratio=_finite(value.get("underexposed_ratio")),
            motion_score=_finite(motion_value),
            coverage_score=_finite(value.get("coverage_score", value.get("quality_score"))),
            duplicate_score=duplicate,
            preprocess_selected=(
                value.get("selected") if isinstance(value.get("selected"), bool) else None
            ),
            preprocess_reject_reasons=tuple(str(item) for item in reasons),
            provenance=str(value.get("provenance", "preprocess-manifest")),
        )


def metrics_from_preprocess_manifest(manifest: Mapping[str, Any]) -> list[FrameMetric]:
    """Reserved adapter entry point; weak cross-worktree manifests are blocked."""

    raise PipelineBlocked(
        "preprocess manifest adapter is disabled until source/canonical/pixel provenance is verified"
    )


def _quality_score(metric: FrameMetric) -> float:
    sharpness = 0.5 if metric.sharpness is None else 1.0 - math.exp(-max(metric.sharpness, 0.0) / 100.0)
    over = _clamp(metric.overexposed_ratio or 0.0)
    under = _clamp(metric.underexposed_ratio or 0.0)
    exposure = _clamp(1.0 - max(over, under))
    motion = 0.5 if metric.motion_score is None else _clamp(metric.motion_score)
    coverage = 0.5 if metric.coverage_score is None else _clamp(metric.coverage_score)
    return round(0.45 * sharpness + 0.25 * exposure + 0.15 * motion + 0.15 * coverage, 9)


def _hard_reasons(metric: FrameMetric, config: FrameSelectionConfig) -> list[str]:
    reasons = list(dict.fromkeys(metric.preprocess_reject_reasons))
    if metric.overexposed_ratio is not None and metric.overexposed_ratio > config.max_overexposed_ratio:
        reasons.append("overexposed")
    if metric.underexposed_ratio is not None and metric.underexposed_ratio > config.max_underexposed_ratio:
        reasons.append("underexposed")
    if metric.duplicate_score is not None and metric.duplicate_score <= config.duplicate_hash_threshold:
        reasons.append("duplicate")
    return list(dict.fromkeys(reasons))


def _tie_key(metric: FrameMetric, score: float) -> tuple[float, float, int, str]:
    return (-score, metric.timestamp_sec, metric.frame_index, metric.frame_id)


def select_frames(
    metrics: Sequence[FrameMetric | Mapping[str, Any]],
    *,
    duration_sec: float,
    source_video_sha256: str,
    canonical_media_sha256: str,
    canonical_width: int,
    canonical_height: int,
    config: FrameSelectionConfig | None = None,
) -> dict[str, Any]:
    """Select a duration-derived budget with deterministic coverage tie-breaks."""

    if duration_sec <= 0 or not source_video_sha256 or not canonical_media_sha256:
        raise PipelineBlocked("duration and source/canonical SHA bindings are required")
    if canonical_width <= 0 or canonical_height <= 0:
        raise PipelineBlocked("canonical dimensions must be positive")
    config = config or FrameSelectionConfig()
    normalized = [
        item if isinstance(item, FrameMetric) else FrameMetric.from_mapping(item, index)
        for index, item in enumerate(metrics)
    ]
    if not normalized:
        raise PipelineBlocked("cannot select frames from an empty metric set")
    ordered = sorted(normalized, key=lambda item: (item.timestamp_sec, item.frame_index, item.frame_id))
    budget = max(config.min_frames, min(config.max_frames, int(math.ceil(duration_sec * config.target_fps))))
    budget = min(budget, len(ordered))
    scored = {metric.frame_id: _quality_score(metric) for metric in ordered}
    reasons = {metric.frame_id: _hard_reasons(metric, config) for metric in ordered}
    eligible = [metric for metric in ordered if not reasons[metric.frame_id] and metric.path]
    if not eligible:
        raise PipelineBlocked("all frames were rejected by quality/coverage metrics")

    selected: set[str] = set()
    selection_reasons: dict[str, str] = {}
    # Uniform temporal anchors provide coverage before quality fills the budget.
    anchors = [
        duration_sec * index / max(1, budget - 1)
        for index in range(budget)
    ]
    for anchor in anchors:
        candidates = [metric for metric in eligible if metric.frame_id not in selected]
        if not candidates:
            break
        chosen = min(
            candidates,
            key=lambda metric: (
                abs(metric.timestamp_sec - anchor),
                _tie_key(metric, scored[metric.frame_id]),
            ),
        )
        selected.add(chosen.frame_id)
        selection_reasons[chosen.frame_id] = "temporal_coverage"
        if len(selected) >= budget:
            break
    for metric in sorted(eligible, key=lambda item: _tie_key(item, scored[item.frame_id])):
        if len(selected) >= budget:
            break
        if metric.frame_id not in selected:
            selected.add(metric.frame_id)
            selection_reasons[metric.frame_id] = "quality_fill"

    # If a configured minimum cannot be met, make the safe fallback explicit in
    # the record rather than silently changing the budget.
    if len(selected) < min(config.min_frames, len(ordered)):
        fallback = [metric for metric in ordered if metric.frame_id not in selected and metric.path]
        for metric in sorted(fallback, key=lambda item: _tie_key(item, scored[item.frame_id])):
            if len(selected) >= min(config.min_frames, len(ordered)):
                break
            selected.add(metric.frame_id)
            selection_reasons[metric.frame_id] = "budget_fallback"

    selected_metrics = [metric for metric in ordered if metric.frame_id in selected]
    times = [metric.timestamp_sec for metric in selected_metrics]
    internal_gaps = [right - left for left, right in zip(times, times[1:])]
    boundary_gaps = []
    if times:
        boundary_gaps = [times[0], max(0.0, duration_sec - times[-1])]
    records: list[dict[str, Any]] = []
    for metric in ordered:
        is_selected = metric.frame_id in selected
        record = {
            "frame_id": metric.frame_id,
            "frame_index": metric.frame_index,
            "timestamp_sec": metric.timestamp_sec,
            "path": metric.path,
            "width": metric.width,
            "height": metric.height,
            "sha256": metric.sha256,
            "metrics": {
                "sharpness": metric.sharpness,
                "overexposed_ratio": metric.overexposed_ratio,
                "underexposed_ratio": metric.underexposed_ratio,
                "motion_score": metric.motion_score,
                "coverage_score": metric.coverage_score,
                "duplicate_score": metric.duplicate_score,
            },
            "quality_score": scored[metric.frame_id],
            "selected": is_selected,
            "selection_reason": selection_reasons.get(metric.frame_id),
            "rejected_reasons": [] if is_selected else reasons[metric.frame_id] or ["budget_limit"],
            "provenance": metric.provenance,
        }
        records.append(record)
    selection = {
        "schema_version": FRAME_SELECTION_SCHEMA,
        "binding": {
            "source_video_sha256": source_video_sha256,
            "canonical_media_sha256": canonical_media_sha256,
            "canonical_width": canonical_width,
            "canonical_height": canonical_height,
        },
        "config": config.to_dict(),
        "duration_sec": duration_sec,
        "frame_budget": budget,
        "selected_count": len(selected_metrics),
        "selected_frame_ids": [metric.frame_id for metric in selected_metrics],
        "rejected_count": len(records) - len(selected_metrics),
        "max_internal_gap_sec": max(internal_gaps, default=0.0),
        "max_boundary_gap_sec": max(boundary_gaps, default=0.0),
        "max_time_gap_sec": max(internal_gaps + boundary_gaps, default=0.0),
        "frames": records,
    }
    selection["selection_sha256"] = stable_sha256(selection)
    return selection
