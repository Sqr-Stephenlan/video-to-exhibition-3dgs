#!/usr/bin/env python3
"""Split a frames_manifest into index-range chunks for bounded VDA runs.

Overlap is recorded as a **frame count** (overlap_prefix_count), never as the
source start index. Example: A=[0,49), B=[39,91) → B.overlap_prefix_count=10.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.depth.config import require_schema_version, to_repo_relative
from scripts.depth.manifest import load_json, save_json, selected_frames


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_range(text: str) -> tuple[int, int]:
    raw = text.strip()
    if "," in raw:
        left, right = raw.split(",", 1)
    elif ":" in raw:
        left, right = raw.split(":", 1)
    else:
        raise ValueError(
            f"range must look like start:end or start,end (end exclusive); got {text!r}"
        )
    start = int(left.strip())
    end = int(right.strip())
    if start < 0 or end <= start:
        raise ValueError(f"invalid range {text!r}: need 0 <= start < end")
    return start, end


def build_chunk_manifest(
    *,
    source_manifest: dict[str, Any],
    selected: list[dict[str, Any]],
    chunk_id: str,
    start: int,
    end: int,
    previous_end: int | None,
    previous_chunk_id: str | None,
    source_manifest_path: str | None,
) -> dict[str, Any]:
    if end > len(selected):
        raise ValueError(
            f"chunk {chunk_id}: end={end} exceeds selected frame count {len(selected)}"
        )
    frames = selected[start:end]
    if not frames:
        raise ValueError(f"chunk {chunk_id}: empty frame slice [{start},{end})")

    overlap_prefix_count = 0
    overlap_with = None
    if previous_end is not None and previous_chunk_id is not None:
        # Leading frames of this chunk that still fall inside the previous range.
        overlap_prefix_count = max(0, min(end, previous_end) - start)
        if overlap_prefix_count:
            overlap_with = previous_chunk_id

    out = {
        "schema_version": "1.0",
        "video_id": source_manifest.get("video_id"),
        "source_video": source_manifest.get("source_video"),
        "source_frames_manifest": source_manifest_path,
        "chunk": {
            "chunk_id": chunk_id,
            "source_index_start": start,
            "source_index_end": end,
            "frame_count": len(frames),
            "overlap_prefix_count": overlap_prefix_count,
            "overlap_with_chunk_id": overlap_with,
        },
        "frames": frames,
    }
    require_schema_version(out, label=f"chunk:{chunk_id}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Split selected frames into chunk manifests for bounded VDA inference. "
            "Records overlap_prefix_count as overlapping frame count, not start index."
        )
    )
    parser.add_argument("frames_manifest", help="Repository-relative or absolute path")
    parser.add_argument(
        "--chunk",
        action="append",
        required=True,
        metavar="ID=START:END",
        help="Chunk spec, e.g. a=0:49 (end exclusive). Order matters for overlap.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for chunk frames_manifest JSON files",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional assembly/split report JSON path",
    )
    args = parser.parse_args(argv)

    root = project_root()
    source_path = Path(args.frames_manifest)
    if not source_path.is_absolute():
        source_path = (root / source_path).resolve()
    source = load_json(source_path)
    selected = selected_frames(
        source, root=root, dedupe_timestamps=False, infer_per_segment=False
    )
    source_rel = to_repo_relative(root, source_path)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks_meta: list[dict[str, Any]] = []
    previous_end: int | None = None
    previous_id: str | None = None
    for spec in args.chunk:
        if "=" not in spec:
            raise SystemExit(f"--chunk must be ID=START:END, got {spec!r}")
        chunk_id, range_text = spec.split("=", 1)
        chunk_id = chunk_id.strip()
        start, end = _parse_range(range_text)
        manifest = build_chunk_manifest(
            source_manifest=source,
            selected=selected,
            chunk_id=chunk_id,
            start=start,
            end=end,
            previous_end=previous_end,
            previous_chunk_id=previous_id,
            source_manifest_path=source_rel,
        )
        out_path = out_dir / f"frames_manifest_chunk_{chunk_id}.json"
        save_json(out_path, manifest)
        chunk_info = dict(manifest["chunk"])
        chunk_info["path"] = to_repo_relative(root, out_path)
        chunks_meta.append(chunk_info)
        print(
            f"wrote {chunk_info['path']} "
            f"[{start}:{end}) count={chunk_info['frame_count']} "
            f"overlap_prefix_count={chunk_info['overlap_prefix_count']}"
        )
        previous_end = end
        previous_id = chunk_id

    report = {
        "source_frames_manifest": source_rel,
        "selected_frame_count": len(selected),
        "chunks": chunks_meta,
        "note": (
            "overlap_prefix_count is the number of leading frames in a chunk that "
            "also belong to the previous chunk's index range. It is not the start index."
        ),
    }
    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = (root / report_path).resolve()
        save_json(report_path, report)
        print(f"report: {to_repo_relative(root, report_path)}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
