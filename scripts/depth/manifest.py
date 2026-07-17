"""frames_manifest / depth_manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.depth.config import require_schema_version, validate_frame_id


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path.as_posix()}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest root must be an object: {path.as_posix()}")
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def selected_frames(frames_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return selected frames in VDA input order.

    Mapping to depth slices is strict-positional: depths[i] <-> selected[i].
    When every selected frame has timestamp_sec, sort ascending so temp-video
    order matches capture time even if the manifest list is unordered.
    """
    require_schema_version(frames_manifest, label="frames_manifest")
    frames = frames_manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames_manifest.frames must be a non-empty list")
    selected: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{index}] must be an object")
        if frame.get("selected", True) is False:
            continue
        for key in ("frame_id", "path"):
            if key not in frame or not frame[key]:
                raise ValueError(f"frames[{index}].{key} is required")
        validate_frame_id(str(frame["frame_id"]))
        selected.append(frame)
    if not selected:
        raise ValueError("No selected frames in frames_manifest")

    has_ts = ["timestamp_sec" in frame for frame in selected]
    if any(has_ts) and not all(has_ts):
        raise ValueError(
            "frames_manifest: either all selected frames must include timestamp_sec "
            "or none of them may (mixed timestamps are ambiguous for VDA ordering)"
        )
    if all(has_ts):
        try:
            selected = sorted(selected, key=lambda frame: float(frame["timestamp_sec"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "frames_manifest: timestamp_sec must be numeric on selected frames"
            ) from exc
    return selected


def build_depth_manifest(
    *,
    frames_manifest: dict[str, Any],
    frame_records: list[dict[str, Any]],
    backend: dict[str, Any],
    depth_type: str,
    frames_manifest_path: str | None = None,
) -> dict[str, Any]:
    """
    Build depth_manifest.json.

    frame_depth_mapping is always strict_positional: depths[i] matches the i-th
    selected frame after selected_frames() ordering (optional timestamp sort).
    """
    return {
        "schema_version": "1.0",
        "source_video_id": frames_manifest.get("video_id"),
        "source_frames_manifest": frames_manifest_path,
        "frame_depth_mapping": "strict_positional",
        "depth_scale": {
            "mode": depth_type,
            # relative: unitless soft prior; metric: meters when backend supports it
            "unit": None if depth_type == "relative" else "meters",
        },
        "backend": {
            "name": backend.get("name"),
            "commit": backend.get("commit"),
            "encoder": backend.get("encoder"),
            "depth_type": depth_type,
            "checkpoint": backend.get("checkpoint") or None,
        },
        "frames": frame_records,
    }
