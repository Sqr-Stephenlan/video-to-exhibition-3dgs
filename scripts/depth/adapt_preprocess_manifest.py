#!/usr/bin/env python3
"""Convert preprocess_manifest.json into depth-prior frames_manifest.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.depth.config import resolve_repo_path, to_repo_relative, validate_frame_id
from scripts.depth.manifest import save_json


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


def adapt_preprocess_to_frames_manifest(
    preprocess_manifest: dict[str, Any],
    *,
    frames_manifest_path: str | None = None,
) -> dict[str, Any]:
    """
    Map preprocess output to the depth-prior frames_manifest contract.

    - frames[].id -> frames[].frame_id
    - keep only selected frames that have a non-null path
    - selected=true with missing path is an error (do not silently drop)
    - timestamp_sec must be all-present or all-absent after adaptation
    """
    schema = preprocess_manifest.get("schema_version", "1.0")
    if schema != "1.0":
        raise ValueError(f"Unsupported preprocess schema_version: {schema!r}")

    frames_in = preprocess_manifest.get("frames")
    if not isinstance(frames_in, list) or not frames_in:
        raise ValueError("preprocess_manifest.frames must be a non-empty list")

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
        # Reject absolute / escape paths early (repo-relative contract).
        resolve_repo_path(ROOT, str(path))
        frame_id = frame.get("id") or frame.get("frame_id")
        if not frame_id:
            raise ValueError(f"frames[{index}] missing id/frame_id")
        frame_id = validate_frame_id(str(frame_id))
        if frame_id in seen_ids:
            raise ValueError(f"duplicate frame_id after adaptation: {frame_id!r}")
        seen_ids.add(frame_id)

        record: dict[str, Any] = {
            "frame_id": frame_id,
            "path": str(path).replace("\\", "/"),
            "selected": True,
        }
        timestamp = _coerce_timestamp(frame, index=index)
        if timestamp is not None:
            record["timestamp_sec"] = timestamp
        if "width" in frame and frame["width"] is not None:
            record["width"] = frame["width"]
        if "height" in frame and frame["height"] is not None:
            record["height"] = frame["height"]
        frames_out.append(record)

    if not frames_out:
        raise ValueError("No selected frames with paths in preprocess_manifest")

    has_ts = ["timestamp_sec" in frame for frame in frames_out]
    if any(has_ts) and not all(has_ts):
        raise ValueError(
            "adapted frames_manifest would mix present/absent timestamp_sec; "
            "either all selected frames must include a numeric timestamp_sec or none"
        )

    source = preprocess_manifest.get("source") or {}
    source_path = source.get("path") if isinstance(source, dict) else None
    if source_path:
        resolve_repo_path(ROOT, str(source_path))
        source_path = str(source_path).replace("\\", "/")
    return {
        "schema_version": "1.0",
        "video_id": preprocess_manifest.get("video_id"),
        "source_video": source_path,
        "source_preprocess_manifest": frames_manifest_path,
        "frames": frames_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adapt preprocess_manifest.json to depth-prior frames_manifest.json"
    )
    parser.add_argument(
        "preprocess_manifest",
        help="Repository-relative path to preprocess_manifest.json",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/frames_manifest.json",
        help="Repository-relative output frames_manifest path",
    )
    args = parser.parse_args(argv)

    try:
        src = resolve_repo_path(ROOT, args.preprocess_manifest)
        out_rel = args.output
        dest = resolve_repo_path(ROOT, out_rel)
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
            frames_manifest_path=to_repo_relative(ROOT, src),
        )
    except ValueError as exc:
        print(f"adapt error: {exc}")
        return 1

    save_json(dest, adapted)
    print(f"wrote {to_repo_relative(ROOT, dest)} ({len(adapted['frames'])} selected frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
