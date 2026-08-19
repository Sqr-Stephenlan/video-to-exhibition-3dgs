#!/usr/bin/env python3
"""Adapt preprocess frames_manifest.json into depth-prior frames_manifest.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.depth.config import (
    format_io_template,
    resolve_repo_path,
    to_repo_relative,
    validate_frame_id,
    validate_video_id,
)
from scripts.depth.manifest import save_json

# Upstream preprocess (PR #2) ships schema 2.0; 1.0 retained for older fixtures.
SUPPORTED_PREPROCESS_SCHEMAS = {"1.0", "2.0"}

# Provenance keys copied from preprocess when present.
_PROVENANCE_KEYS = (
    "segment_id",
    "frame_index",
    "blur_score",
    "overexposed_ratio",
    "underexposed_ratio",
    "duplicate_score",
    "reject_reasons",
    "calibrated_blur_score",
    "keyframe",
    "sha256",
)


def _coerce_timestamp(frame: dict[str, Any], *, index: int) -> float | None:
    if "timestamp_sec" not in frame or frame["timestamp_sec"] is None:
        return None
    try:
        return float(frame["timestamp_sec"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"frames[{index}].timestamp_sec must be numeric when present; "
            f"got {frame['timestamp_sec']!r}"
        ) from exc


def _read_image_size(root: Path, relative_path: str) -> tuple[int, int]:
    from PIL import Image

    path = resolve_repo_path(root, relative_path)
    with Image.open(path) as image:
        width, height = image.size
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image size for {relative_path}: {width}x{height}")
    return int(width), int(height)


def _selection_reason(frame: dict[str, Any]) -> str | None:
    if frame.get("reason") not in (None, ""):
        return str(frame["reason"])
    reject = frame.get("reject_reasons")
    if isinstance(reject, list) and reject:
        return ",".join(str(item) for item in reject)
    if frame.get("selected", True) is True:
        return "selected"
    return None


def _assert_coverage_quality_gate(preprocess_manifest: dict[str, Any]) -> None:
    """Fail closed when coverage keyframe quality is present but not passed."""
    summary = preprocess_manifest.get("summary")
    if not isinstance(summary, dict):
        return
    quality = summary.get("keyframe_quality")
    if not isinstance(quality, dict):
        return
    status = quality.get("status")
    if status != "passed":
        raise ValueError(
            "preprocess summary.keyframe_quality.status must be 'passed' before "
            f"depth-prior adaptation; got {status!r}"
        )


def adapt_preprocess_to_frames_manifest(
    preprocess_manifest: dict[str, Any],
    *,
    root: Path = ROOT,
    frames_manifest_path: str | None = None,
    fill_missing_size: bool = True,
) -> dict[str, Any]:
    """
    Map preprocess output to the depth-prior frames_manifest contract.

    Accepts upstream preprocess schema 2.0 (PR #2) and legacy 1.0 fixtures.
    Depth-prior output schema remains 1.0.

    - frames[].id -> frames[].frame_id
    - keep segment_id / quality provenance when present
    - width/height required (filled from image bytes when missing)
    - selected=true with missing path is an error
    - timestamp_sec must be all-present or all-absent after adaptation
    - if summary.keyframe_quality is present, status must be "passed"
    """
    schema = str(preprocess_manifest.get("schema_version", "1.0"))
    if schema not in SUPPORTED_PREPROCESS_SCHEMAS:
        raise ValueError(
            "Unsupported preprocess schema_version: "
            f"{schema!r}; supported={sorted(SUPPORTED_PREPROCESS_SCHEMAS)}"
        )
    _assert_coverage_quality_gate(preprocess_manifest)

    video_id = preprocess_manifest.get("video_id")
    if video_id not in (None, ""):
        video_id = validate_video_id(str(video_id))

    frames_in = preprocess_manifest.get("frames")
    if not isinstance(frames_in, list) or not frames_in:
        raise ValueError("preprocess frames_manifest.frames must be a non-empty list")

    frames_out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, frame in enumerate(frames_in):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{index}] must be an object")
        if frame.get("selected", True) is False:
            continue
        path = frame.get("path")
        if path is None or path == "":
            raise ValueError(
                f"frames[{index}] is selected but path is missing/null; "
                "refuse to silently drop selected frames"
            )
        resolve_repo_path(root, str(path))
        frame_id = frame.get("id") or frame.get("frame_id")
        if not frame_id:
            raise ValueError(f"frames[{index}] missing id/frame_id")
        frame_id = validate_frame_id(str(frame_id))
        if frame_id in seen_ids:
            raise ValueError(f"duplicate frame_id after adaptation: {frame_id!r}")
        seen_ids.add(frame_id)

        width = frame.get("width")
        height = frame.get("height")
        if width is None or height is None:
            if not fill_missing_size:
                raise ValueError(
                    f"frames[{index}] missing width/height and fill_missing_size is false"
                )
            width, height = _read_image_size(root, str(path))
        else:
            width = int(width)
            height = int(height)

        record: dict[str, Any] = {
            "frame_id": frame_id,
            "path": str(path).replace("\\", "/"),
            "selected": True,
            "width": width,
            "height": height,
        }
        timestamp = _coerce_timestamp(frame, index=index)
        if timestamp is not None:
            record["timestamp_sec"] = timestamp
        reason = _selection_reason(frame)
        if reason is not None:
            record["reason"] = reason
        for key in _PROVENANCE_KEYS:
            if key in frame and frame[key] is not None:
                record[key] = frame[key]
        frames_out.append(record)

    if not frames_out:
        raise ValueError("No selected frames with paths in preprocess frames_manifest")

    has_ts = ["timestamp_sec" in frame for frame in frames_out]
    if any(has_ts) and not all(has_ts):
        raise ValueError(
            "adapted frames_manifest would mix present/absent timestamp_sec; "
            "either all selected frames must include a numeric timestamp_sec or none"
        )

    source = preprocess_manifest.get("source") or {}
    source_path = source.get("path") if isinstance(source, dict) else None
    if source_path:
        resolve_repo_path(root, str(source_path))
        source_path = str(source_path).replace("\\", "/")
    return {
        "schema_version": "1.0",
        "video_id": video_id,
        "source_video": source_path,
        "source_preprocess_schema": schema,
        "source_preprocess_manifest": frames_manifest_path,
        "frames": frames_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt preprocess frames_manifest.json (schema 2.0 from PR #2, or "
            "legacy 1.0) to depth-prior frames_manifest.json"
        )
    )
    parser.add_argument(
        "preprocess_manifest",
        help=(
            "Repository-relative path to preprocess frames_manifest.json "
            "(typically data/manifests/<video_id>/frames_manifest.json)"
        ),
    )
    parser.add_argument(
        "--output",
        default="data/manifests/{video_id}/{run_id}/frames_manifest.json",
        help="Repository-relative depth-prior output path (supports {video_id} and {run_id})",
    )
    parser.add_argument(
        "--run-id",
        default="default",
        help="Namespace for {run_id} in --output (default: default)",
    )
    args = parser.parse_args(argv)

    try:
        src = resolve_repo_path(ROOT, args.preprocess_manifest)
    except ValueError as exc:
        print(f"path error: {exc}")
        return 2

    if not src.is_file():
        print(f"missing: {to_repo_relative(ROOT, src)}")
        return 2

    payload = json.loads(src.read_text(encoding="utf-8"))
    try:
        adapted = adapt_preprocess_to_frames_manifest(
            payload,
            root=ROOT,
            frames_manifest_path=to_repo_relative(ROOT, src),
        )
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"adapt error: {exc}")
        return 1

    try:
        out_rel = format_io_template(
            args.output,
            video_id=adapted.get("video_id"),
            run_id=args.run_id,
        )
        dest = resolve_repo_path(ROOT, out_rel)
    except ValueError as exc:
        print(f"path error: {exc}")
        return 2

    save_json(dest, adapted)
    print(f"wrote {to_repo_relative(ROOT, dest)} ({len(adapted['frames'])} selected frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
