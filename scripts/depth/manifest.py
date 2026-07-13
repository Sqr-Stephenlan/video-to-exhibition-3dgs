"""frames_manifest / depth_manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
        selected.append(frame)
    if not selected:
        raise ValueError("No selected frames in frames_manifest")
    return selected


def build_depth_manifest(
    *,
    frames_manifest: dict[str, Any],
    frame_records: list[dict[str, Any]],
    backend: dict[str, Any],
    depth_type: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "source_frames_manifest": frames_manifest.get("video_id"),
        "backend": {
            "name": backend.get("name"),
            "commit": backend.get("commit"),
            "encoder": backend.get("encoder"),
            "depth_type": depth_type,
            "checkpoint": backend.get("checkpoint") or None,
        },
        "frames": frame_records,
    }
