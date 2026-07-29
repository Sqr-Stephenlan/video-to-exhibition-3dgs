"""Adapt schema-2 preprocess manifests to the LongSplat consumer contract.

The full manifest schema is owned by ``feature/preprocess-video``.  This module
accepts its exact string schema version ``"2.0"`` and emits the smaller
LongSplat-internal integer schema version ``1``.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class AdapterError(Exception):
    """Raised when a producer manifest cannot be adapted."""


def adapt_manifest(
    producer_manifest: dict[str, Any],
    segment_id: str,
    repo_root: str | Path,
    *,
    quality_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one producer segment into a LongSplat consumer manifest."""
    _validate_producer_schema(producer_manifest)

    root = Path(repo_root).resolve()
    _find_segment(producer_manifest, segment_id)
    _validate_quality_handoff(producer_manifest, quality_report)

    producer_run_id = producer_manifest["run"]["id"]

    consumer_frames: list[dict[str, Any]] = []
    for frame in producer_manifest["frames"]:
        if not isinstance(frame, dict) or frame.get("segment_id") != segment_id:
            continue
        if frame.get("selected") is not True:
            continue

        producer_frame_id = _selected_frame_id(frame, segment_id)
        frame_path, absolute_path = _selected_frame_path(
            frame, producer_frame_id, segment_id, root
        )
        width = _positive_int(
            frame.get("width"), f"Selected frame {producer_frame_id!r} width"
        )
        height = _positive_int(
            frame.get("height"), f"Selected frame {producer_frame_id!r} height"
        )
        producer_frame_index = _nonnegative_int(
            frame.get("frame_index"),
            f"Selected frame {producer_frame_id!r} frame_index",
        )
        timestamp = frame.get("timestamp_sec")
        if (
            not isinstance(timestamp, (int, float))
            or isinstance(timestamp, bool)
            or not math.isfinite(float(timestamp))
        ):
            raise AdapterError(
                f"Selected frame {producer_frame_id!r} timestamp_sec must be "
                "a finite number"
            )

        sha256 = frame.get("sha256")
        if not isinstance(sha256, str) or _SHA256_RE.fullmatch(sha256) is None:
            raise AdapterError(
                f"Selected frame {producer_frame_id!r} sha256 must be a "
                "64-character hexadecimal string"
            )
        if _sha256_hex(absolute_path).lower() != sha256.lower():
            raise AdapterError(
                f"Selected frame {producer_frame_id!r} sha256 does not match file: "
                f"{frame_path}"
            )
        decoded_width, decoded_height = _decode_image_dimensions(
            absolute_path,
            producer_frame_id,
        )
        if (decoded_width, decoded_height) != (width, height):
            raise AdapterError(
                f"Selected frame {producer_frame_id!r} decoded dimensions "
                f"{decoded_width}x{decoded_height} do not match manifest "
                f"{width}x{height}"
            )

        consumer_frame: dict[str, Any] = {
            "frame_id": len(consumer_frames),
            "path": frame_path,
            "width": width,
            "height": height,
            "sha256": sha256,
            "timestamp": timestamp,
            "producer_frame_id": producer_frame_id,
            "producer_frame_index": producer_frame_index,
            "producer_run_id": producer_run_id,
        }
        consumer_frames.append(consumer_frame)

    if not consumer_frames:
        raise AdapterError(f"No selected frames found for segment {segment_id!r}")

    return {
        "schema_version": 1,
        "segment_id": segment_id,
        "base": str(root),
        "frames": consumer_frames,
    }


def _validate_producer_schema(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise AdapterError("Producer manifest must be a JSON object")

    schema_version = manifest.get("schema_version")
    if schema_version != "2.0":
        raise AdapterError(
            f"Unsupported producer schema_version: {schema_version!r} (expected '2.0')"
        )

    if not isinstance(manifest.get("segments"), list) or not manifest["segments"]:
        raise AdapterError("Producer manifest must contain a non-empty 'segments' list")
    if not isinstance(manifest.get("frames"), list):
        raise AdapterError("Producer manifest must contain a 'frames' list")

    segment_ids = _unique_segment_ids(
        manifest["segments"], "Producer manifest segments"
    )
    _segment_windows(manifest["segments"])
    seen_frame_ids: set[str] = set()
    previous_timestamp_by_segment: dict[str, float] = {}
    for index, frame in enumerate(manifest["frames"]):
        if not isinstance(frame, dict):
            raise AdapterError(f"Producer manifest frames[{index}] must be an object")
        frame_id = frame.get("id")
        if not isinstance(frame_id, str) or not frame_id:
            raise AdapterError(
                f"Producer manifest frames[{index}].id must be a non-empty string"
            )
        if frame_id in seen_frame_ids:
            raise AdapterError(
                f"Duplicate producer frame id {frame_id!r} at frames[{index}]"
            )
        seen_frame_ids.add(frame_id)

        frame_segment_id = frame.get("segment_id")
        if not isinstance(frame_segment_id, str) or frame_segment_id not in segment_ids:
            raise AdapterError(
                f"Producer manifest frames[{index}].segment_id must reference "
                "a declared segment"
            )
        if not isinstance(frame.get("selected"), bool):
            raise AdapterError(
                f"Producer manifest frames[{index}].selected must be boolean"
            )
        _nonnegative_int(
            frame.get("frame_index"),
            f"Producer manifest frames[{index}].frame_index",
        )
        timestamp = _finite_number(
            frame.get("timestamp_sec"),
            f"Producer manifest frames[{index}].timestamp_sec",
        )
        previous_timestamp = previous_timestamp_by_segment.get(frame_segment_id)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise AdapterError(
                f"Producer manifest timestamps for segment {frame_segment_id!r} "
                "must be strictly increasing in frames[] order"
            )
        previous_timestamp_by_segment[frame_segment_id] = timestamp

    video_id = manifest.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        raise AdapterError("Producer manifest video_id must be a non-empty string")

    settings = manifest.get("settings")
    if not isinstance(settings, dict):
        raise AdapterError("Producer manifest settings must be an object")
    if settings.get("keyframe_policy") not in {"legacy", "coverage_v1"}:
        raise AdapterError(
            "Producer manifest settings.keyframe_policy must be "
            "'legacy' or 'coverage_v1'"
        )

    if not isinstance(manifest.get("summary"), dict):
        raise AdapterError("Producer manifest summary must be an object")

    run = manifest.get("run")
    if not isinstance(run, dict):
        raise AdapterError("Producer manifest run must be an object")
    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id:
        raise AdapterError("Producer manifest run.id must be a non-empty string")


def _find_segment(manifest: dict[str, Any], segment_id: str) -> dict[str, Any]:
    if not isinstance(segment_id, str) or not segment_id:
        raise AdapterError("Requested segment_id must be a non-empty string")

    matches = [
        segment
        for segment in manifest["segments"]
        if isinstance(segment, dict) and segment.get("id") == segment_id
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise AdapterError(
            f"Segment {segment_id!r} must be unique in producer manifest"
        )

    available = [
        segment.get("id", "?")
        for segment in manifest["segments"]
        if isinstance(segment, dict)
    ]
    raise AdapterError(
        f"Segment {segment_id!r} not found in producer manifest. "
        f"Available segments: {available}"
    )


def _validate_quality_handoff(
    manifest: dict[str, Any],
    report: dict[str, Any] | None,
) -> None:
    policy = manifest["settings"]["keyframe_policy"]
    inline_quality = manifest["summary"].get("keyframe_quality")

    if policy == "coverage_v1":
        if not isinstance(inline_quality, dict):
            raise AdapterError(
                "coverage_v1 manifest summary.keyframe_quality must be an object"
            )
        _validate_inline_quality(policy, inline_quality)
        if report is None:
            raise AdapterError("coverage_v1 manifest requires a quality report")
        _validate_coverage_report(manifest, report)
        return

    if inline_quality is not None:
        if not isinstance(inline_quality, dict):
            raise AdapterError("Manifest summary.keyframe_quality must be an object")
        _validate_inline_quality(policy, inline_quality)


def _validate_inline_quality(
    policy: str,
    inline_quality: dict[str, Any],
) -> None:
    if inline_quality.get("policy") != policy:
        raise AdapterError(
            "Manifest settings.keyframe_policy must match "
            "summary.keyframe_quality.policy"
        )
    if inline_quality.get("status") != "passed":
        raise AdapterError("Manifest summary.keyframe_quality.status must be 'passed'")


def _validate_coverage_report(
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> None:
    if not isinstance(report, dict):
        raise AdapterError("Quality report must be a JSON object")
    if report.get("schema_version") != "2.0":
        raise AdapterError("Quality report schema_version must be '2.0'")
    if report.get("video_id") != manifest["video_id"]:
        raise AdapterError("Quality report video_id must match producer manifest")
    if report.get("policy") != "coverage_v1":
        raise AdapterError("Quality report policy must be 'coverage_v1'")
    if report.get("status") != "passed":
        raise AdapterError("Quality report status must be 'passed'")

    manifest_segment_order = [segment["id"] for segment in manifest["segments"]]
    manifest_segment_ids = set(manifest_segment_order)
    windows = _segment_windows(manifest["segments"])
    report_segments = report.get("segments")
    if not isinstance(report_segments, list):
        raise AdapterError("Quality report segments must be a list")
    report_segment_ids = _unique_segment_ids(
        report_segments, "Quality report segments", key="segment_id"
    )
    if report_segment_ids != manifest_segment_ids:
        raise AdapterError(
            "Quality report segment_id set must match producer manifest segments"
        )
    if [item["segment_id"] for item in report_segments] != manifest_segment_order:
        raise AdapterError(
            "Quality report segments must preserve producer manifest segment order"
        )

    report_by_segment = {item["segment_id"]: item for item in report_segments}
    expected_segment_summaries: list[dict[str, Any]] = []
    for current_segment_id in manifest_segment_order:
        summary = report_by_segment[current_segment_id]
        _validate_quality_segment_fields(current_segment_id, summary)

        segment_frames = [
            frame
            for frame in manifest["frames"]
            if frame["segment_id"] == current_segment_id
        ]
        bridge_frames = 0
        selected_timestamps: list[float] = []
        for frame in segment_frames:
            keyframe = frame.get("keyframe")
            if not isinstance(keyframe, dict):
                raise AdapterError(
                    f"coverage_v1 frame {frame['id']!r} keyframe must be an object"
                )
            if keyframe.get("policy") != "coverage_v1":
                raise AdapterError(
                    f"coverage_v1 frame {frame['id']!r} keyframe.policy "
                    "must be 'coverage_v1'"
                )
            if keyframe.get("selected_by_policy") is not frame["selected"]:
                raise AdapterError(
                    f"coverage_v1 frame {frame['id']!r} "
                    "keyframe.selected_by_policy must match selected"
                )
            bridge = keyframe.get("bridge")
            if not isinstance(bridge, bool):
                raise AdapterError(
                    f"coverage_v1 frame {frame['id']!r} keyframe.bridge must be boolean"
                )
            if bridge and not frame["selected"]:
                raise AdapterError(
                    f"coverage_v1 frame {frame['id']!r} cannot be a rejected bridge"
                )
            if frame["selected"]:
                selected_timestamps.append(float(frame["timestamp_sec"]))
                bridge_frames += int(bridge)

        segment_start, segment_end = windows[current_segment_id]
        gaps = [
            right - left
            for left, right in zip(
                selected_timestamps,
                selected_timestamps[1:],
            )
        ]
        if selected_timestamps:
            gaps.extend(
                [
                    selected_timestamps[0] - segment_start,
                    segment_end - selected_timestamps[-1],
                ]
            )
        if any(gap < 0 for gap in gaps):
            raise AdapterError(
                f"Producer frame timestamps for segment {current_segment_id!r} "
                "must lie within the segment time window"
            )
        max_gap = round(max(gaps, default=0.0), 3)
        max_allowed_gap = _finite_number(
            manifest["settings"].get("selection_max_gap_sec"),
            "Producer manifest settings.selection_max_gap_sec",
        )
        if max_allowed_gap <= 0:
            raise AdapterError(
                "Producer manifest settings.selection_max_gap_sec must be positive"
            )
        failed_reasons: list[str] = []
        if not selected_timestamps:
            failed_reasons.append("no_selected_frames")
        if max_gap > max_allowed_gap:
            failed_reasons.append("max_selected_gap_sec")

        expected_counts = {
            "selected_frames": sum(
                frame["selected"] is True for frame in segment_frames
            ),
            "rejected_frames": sum(
                frame["selected"] is False for frame in segment_frames
            ),
            "total_frames": len(segment_frames),
            "component_count": 1 if selected_timestamps else 0,
            "max_selected_gap_sec": max_gap,
            "bridge_frames": bridge_frames,
            "failed_reasons": failed_reasons,
        }
        for field, expected in expected_counts.items():
            actual = summary[field]
            if actual != expected:
                raise AdapterError(
                    f"Quality report segment {current_segment_id!r} {field} "
                    f"must match producer manifest frames ({expected}), "
                    f"got {actual!r}"
                )
        expected_segment_summaries.append(expected_counts)

    expected_failed_segments = [
        segment_id
        for segment_id, summary in zip(
            manifest_segment_order,
            expected_segment_summaries,
        )
        if summary["failed_reasons"]
    ]
    expected_status = "failed" if expected_failed_segments else "passed"
    if report["status"] != expected_status:
        raise AdapterError(
            "Quality report status must match recomputed segment failures"
        )

    inline_quality = manifest["summary"]["keyframe_quality"]
    expected_inline = {
        "selected_frames": sum(
            summary["selected_frames"] for summary in expected_segment_summaries
        ),
        "bridge_frames": sum(
            summary["bridge_frames"] for summary in expected_segment_summaries
        ),
        "component_count": max(
            (summary["component_count"] for summary in expected_segment_summaries),
            default=0,
        ),
        "max_selected_gap_sec": round(
            max(
                (
                    summary["max_selected_gap_sec"]
                    for summary in expected_segment_summaries
                ),
                default=0.0,
            ),
            3,
        ),
        "failed_segments": expected_failed_segments,
        "status": expected_status,
    }
    for field, expected in expected_inline.items():
        if inline_quality.get(field) != expected:
            raise AdapterError(
                f"Manifest summary.keyframe_quality.{field} must match "
                f"the coverage report ({expected!r})"
            )
    for field, setting_name in (
        ("min_motion_inlier_ratio", "min_motion_inlier_ratio"),
        ("min_motion_grid_coverage", "min_motion_grid_coverage"),
    ):
        expected = manifest["settings"].get(setting_name)
        if inline_quality.get(field) != expected:
            raise AdapterError(
                f"Manifest summary.keyframe_quality.{field} must match "
                f"settings.{setting_name}"
            )


def _validate_quality_segment_fields(
    segment_id: str,
    summary: dict[str, Any],
) -> None:
    for field in (
        "total_frames",
        "selected_frames",
        "rejected_frames",
        "component_count",
        "bridge_frames",
    ):
        value = summary.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise AdapterError(
                f"Quality report segment {segment_id!r} {field} must be "
                "a non-negative integer"
            )

    max_gap = summary.get("max_selected_gap_sec")
    if (
        not isinstance(max_gap, (int, float))
        or isinstance(max_gap, bool)
        or not math.isfinite(float(max_gap))
        or max_gap < 0
    ):
        raise AdapterError(
            f"Quality report segment {segment_id!r} "
            "max_selected_gap_sec must be a finite non-negative number"
        )

    failed_reasons = summary.get("failed_reasons")
    if not isinstance(failed_reasons, list) or any(
        not isinstance(reason, str) for reason in failed_reasons
    ):
        raise AdapterError(
            f"Quality report segment {segment_id!r} failed_reasons "
            "must be a list of strings"
        )
    if failed_reasons:
        raise AdapterError(
            f"Quality report status is 'passed' but segment {segment_id!r} "
            f"failed_reasons is not empty"
        )


def _unique_segment_ids(
    segments: list[Any],
    label: str,
    *,
    key: str = "id",
) -> set[str]:
    identifiers: list[str] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise AdapterError(f"{label}[{index}] must be an object")
        value = segment.get(key)
        if not isinstance(value, str) or not value:
            raise AdapterError(f"{label}[{index}].{key} must be a non-empty string")
        identifiers.append(value)
    if len(identifiers) != len(set(identifiers)):
        raise AdapterError(f"{label} must contain unique {key} values")
    return set(identifiers)


def _selected_frame_id(frame: dict[str, Any], segment_id: str) -> str:
    frame_id = frame.get("id")
    if not isinstance(frame_id, str) or not frame_id:
        raise AdapterError(
            f"Selected frame in segment {segment_id!r} has no producer frame id"
        )
    return frame_id


def _selected_frame_path(
    frame: dict[str, Any],
    frame_id: str,
    segment_id: str,
    root: Path,
) -> tuple[str, Path]:
    frame_path = frame.get("path")
    if not isinstance(frame_path, str) or not frame_path.strip():
        raise AdapterError(
            f"Selected frame {frame_id!r} in segment {segment_id!r} "
            "has no non-empty path"
        )

    relative_path = Path(frame_path)
    if relative_path.is_absolute():
        raise AdapterError(
            f"Selected frame {frame_id!r} path must be repository-relative"
        )
    try:
        absolute_path = (root / relative_path).resolve()
        absolute_path.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise AdapterError(
            f"Selected frame {frame_id!r} path escapes or is outside repo_root: "
            f"{frame_path!r}"
        )
    if not absolute_path.is_file():
        raise AdapterError(
            f"Frame file not found: {absolute_path} "
            f"(frame {frame_id!r} in segment {segment_id!r})"
        )
    return frame_path, absolute_path


def _segment_windows(
    segments: list[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    windows: dict[str, tuple[float, float]] = {}
    for index, segment in enumerate(segments):
        start = _finite_number(
            segment.get("start_sec"),
            f"Producer manifest segments[{index}].start_sec",
        )
        end = _finite_number(
            segment.get("end_sec"),
            f"Producer manifest segments[{index}].end_sec",
        )
        if end < start:
            raise AdapterError(
                f"Producer manifest segments[{index}] end_sec must be >= start_sec"
            )
        windows[segment["id"]] = (start, end)
    return windows


def _decode_image_dimensions(path: Path, frame_id: str) -> tuple[int, int]:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise AdapterError(
            f"Selected frame {frame_id!r} cannot be read for image decode: {path}"
        ) from exc
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise AdapterError(
            f"Selected frame {frame_id!r} cannot be decoded as an image: {path}"
        ) from exc
    if image is None or image.ndim < 2:
        raise AdapterError(
            f"Selected frame {frame_id!r} cannot be decoded as an image: {path}"
        )
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise AdapterError(f"Selected frame {frame_id!r} decoded to invalid dimensions")
    return int(width), int(height)


def _finite_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise AdapterError(f"{label} must be a finite number")
    return float(value)


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AdapterError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AdapterError(f"{label} must be a positive integer")
    return value


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
