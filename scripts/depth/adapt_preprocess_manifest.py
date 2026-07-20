#!/usr/bin/env python3
"""Convert preprocess_manifest.json into depth-prior frames_manifest.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.depth.config import validate_frame_id
from scripts.depth.manifest import save_json


def adapt_preprocess_to_frames_manifest(
    preprocess_manifest: dict,
    *,
    frames_manifest_path: str | None = None,
) -> dict:
    """
    Map preprocess output to the depth-prior frames_manifest contract.

    - frames[].id -> frames[].frame_id
    - keep only selected frames with a non-null path
    - preserve timestamp_sec for ordering in depth-prior
    """
    schema = preprocess_manifest.get("schema_version", "1.0")
    if schema != "1.0":
        raise ValueError(f"Unsupported preprocess schema_version: {schema!r}")

    frames_in = preprocess_manifest.get("frames")
    if not isinstance(frames_in, list) or not frames_in:
        raise ValueError("preprocess_manifest.frames must be a non-empty list")

    frames_out: list[dict] = []
    for index, frame in enumerate(frames_in):
        if not isinstance(frame, dict):
            raise ValueError(f"frames[{index}] must be an object")
        if frame.get("selected", True) is False:
            continue
        path = frame.get("path")
        if not path:
            continue
        frame_id = frame.get("id") or frame.get("frame_id")
        if not frame_id:
            raise ValueError(f"frames[{index}] missing id/frame_id")
        validate_frame_id(str(frame_id))
        record = {
            "frame_id": str(frame_id),
            "path": str(path).replace("\\", "/"),
            "selected": True,
            "timestamp_sec": float(frame["timestamp_sec"])
            if "timestamp_sec" in frame
            else None,
        }
        if record["timestamp_sec"] is None:
            del record["timestamp_sec"]
        if "width" in frame:
            record["width"] = frame["width"]
        if "height" in frame:
            record["height"] = frame["height"]
        frames_out.append(record)

    if not frames_out:
        raise ValueError("No selected frames with paths in preprocess_manifest")

    # depth-prior sorts when all have timestamp_sec; drop mixed nulls already handled
    source = preprocess_manifest.get("source") or {}
    source_path = source.get("path") if isinstance(source, dict) else None
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

    src = ROOT / args.preprocess_manifest
    if not src.is_file():
        print(f"missing: {args.preprocess_manifest}")
        return 2
    payload = json.loads(src.read_text(encoding="utf-8"))
    adapted = adapt_preprocess_to_frames_manifest(
        payload,
        frames_manifest_path=args.preprocess_manifest.replace("\\", "/"),
    )
    dest = ROOT / args.output
    save_json(dest, adapted)
    print(f"wrote {args.output} ({len(adapted['frames'])} selected frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
