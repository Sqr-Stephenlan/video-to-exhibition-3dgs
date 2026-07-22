"""frames_manifest / depth_manifest helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.depth.config import require_schema_version, resolve_repo_path, validate_frame_id


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dedupe_identical_timestamps(
    frames: list[dict[str, Any]],
    *,
    root: Path,
) -> list[dict[str, Any]]:
    """
    Collapse frames that share the same timestamp_sec when image bytes match.

    Different content at the same timestamp is an error (ambiguous VDA ordering).
    """
    by_ts: dict[float, list[dict[str, Any]]] = {}
    for frame in frames:
        key = float(frame["timestamp_sec"])
        by_ts.setdefault(key, []).append(frame)

    kept: list[dict[str, Any]] = []
    for timestamp in sorted(by_ts):
        group = by_ts[timestamp]
        if len(group) == 1:
            kept.append(group[0])
            continue
        hashes = {
            _sha256_file(resolve_repo_path(root, frame["path"])): frame for frame in group
        }
        if len(hashes) != 1:
            ids = [frame["frame_id"] for frame in group]
            raise ValueError(
                "Duplicate timestamp_sec with differing image content; "
                f"timestamp={timestamp}, frame_ids={ids}. "
                "Split by segment or resolve the conflict before depth-prior."
            )
        # Identical bytes: keep the first occurrence (stable provenance).
        kept.append(group[0])
    return kept


def selected_frames(
    frames_manifest: dict[str, Any],
    *,
    root: Path | None = None,
    dedupe_timestamps: bool = True,
) -> list[dict[str, Any]]:
    """
    Return selected frames in VDA input order.

    Mapping to depth slices is strict-positional: depths[i] <-> selected[i].
    When every selected frame has timestamp_sec, sort ascending and optionally
    collapse identical-timestamp duplicates (same image bytes).
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

    ids = [str(frame["frame_id"]) for frame in selected]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate frame_id in selected frames: {ids}")

    def _has_timestamp(frame: dict[str, Any]) -> bool:
        return "timestamp_sec" in frame and frame["timestamp_sec"] is not None

    has_ts = [_has_timestamp(frame) for frame in selected]
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
        if dedupe_timestamps:
            if root is None:
                raise ValueError(
                    "dedupe_timestamps requires project root to hash frame files"
                )
            selected = dedupe_identical_timestamps(selected, root=root)
    return selected


def assert_outputs_not_conflicting(
    *,
    root: Path,
    depth_dir: Path,
    depth_manifest_path: Path,
    run_record_path: Path,
    frames: list[dict[str, Any]],
    overwrite: bool,
) -> None:
    """Refuse to clobber prior run artifacts unless overwrite is enabled."""
    if overwrite:
        return
    conflicts: list[str] = []
    for frame in frames:
        dest = depth_dir / f"{frame['frame_id']}.npz"
        if dest.is_file():
            conflicts.append(dest.as_posix())
    for path in (depth_manifest_path, run_record_path):
        if path.is_file():
            conflicts.append(path.as_posix())
    if conflicts:
        preview = "\n  ".join(conflicts[:8])
        more = "" if len(conflicts) <= 8 else f"\n  ... and {len(conflicts) - 8} more"
        raise FileExistsError(
            "Refusing to overwrite existing depth-prior outputs "
            "(set runtime.overwrite: true to allow):\n  "
            f"{preview}{more}"
        )


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
    selected frame after selected_frames() ordering (optional timestamp sort/dedupe).
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
