"""Contract helpers for LongSplat consumers of depth_manifest (no LongSplat import)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def prepared_depth_npy_name(prepared_image_name: str) -> str:
    """Depth file stem must match LongSplat prepared image stem."""
    stem = Path(prepared_image_name).stem
    if not stem:
        raise ValueError(f"Invalid prepared image name: {prepared_image_name!r}")
    return f"{stem}_depth.npy"


def bind_depth_manifest_to_frame_mapping(
    *,
    depth_manifest: dict[str, Any],
    frame_mapping: dict[str, Any],
    require_all: bool = True,
) -> list[dict[str, Any]]:
    """
    Bind depth_manifest frames to LongSplat prepared names via frame_mapping.

    frame_mapping entries must identify the depth frame by `frame_id` (preferred)
    or `rgb_path` matching depth_manifest.frames[].rgb_path.

    Raises ValueError when require_all and any prepared frame lacks depth.
    """
    depth_frames = depth_manifest.get("frames")
    if not isinstance(depth_frames, list) or not depth_frames:
        raise ValueError("depth_manifest.frames must be a non-empty list")

    by_id = {str(frame["frame_id"]): frame for frame in depth_frames if "frame_id" in frame}
    by_rgb = {
        str(frame["rgb_path"]).replace("\\", "/"): frame
        for frame in depth_frames
        if frame.get("rgb_path")
    }

    bindings: list[dict[str, Any]] = []
    missing: list[str] = []
    for prepared_name, meta in frame_mapping.items():
        if not isinstance(meta, dict):
            raise ValueError(f"frame_mapping[{prepared_name!r}] must be an object")
        frame = None
        frame_id = meta.get("frame_id")
        rgb_path = meta.get("rgb_path") or meta.get("path")
        if frame_id is not None:
            frame = by_id.get(str(frame_id))
        if frame is None and rgb_path:
            frame = by_rgb.get(str(rgb_path).replace("\\", "/"))
        if frame is None:
            missing.append(str(prepared_name))
            continue
        bindings.append(
            {
                "prepared_image": prepared_name,
                "prepared_depth_npy": prepared_depth_npy_name(str(prepared_name)),
                "frame_id": frame["frame_id"],
                "depth_path": frame.get("depth_path"),
                "rgb_path": frame.get("rgb_path"),
            }
        )

    if require_all and missing:
        raise ValueError(
            "LongSplat frame_mapping entries missing depth_manifest coverage "
            f"(fail closed, do not silently fall back to MASt3R): {missing}"
        )
    return bindings
