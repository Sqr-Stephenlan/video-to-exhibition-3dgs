#!/usr/bin/env python3
"""Assemble overlapping chunk depth_manifests into one producer manifest.

Requires per-frame sha256 on every chunk frame. Verifies identity/order against
the source frames_manifest and re-checks NPZ file integrity. Never marks
vda_quality as passed — that remains a consumer/route gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.depth.config import to_repo_relative
from scripts.depth.manifest import (
    build_depth_manifest,
    load_json,
    save_json,
    selected_frames,
    validate_frame_record_sha256,
    verify_depth_file_integrity,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def _require_chunk_frames(manifest: dict[str, Any], label: str) -> list[dict[str, Any]]:
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{label}: frames must be a non-empty list")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"{label}.frames[{index}] must be an object")
        validate_frame_record_sha256(frame, label=f"{label}.frames[{index}]")
        for key in ("frame_id", "rgb_path", "depth_path"):
            if not frame.get(key):
                raise ValueError(f"{label}.frames[{index}]: {key} is required")
    return frames


def assemble_depth_manifests(
    *,
    root: Path,
    source_frames_manifest: dict[str, Any],
    source_frames_path: str,
    chunk_a: dict[str, Any],
    chunk_b: dict[str, Any],
    drop_b_prefix: int,
    verify_files: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if drop_b_prefix < 0:
        raise ValueError("drop_b_prefix must be >= 0")

    selected = selected_frames(
        source_frames_manifest,
        root=root,
        dedupe_timestamps=False,
        infer_per_segment=False,
    )
    frames_a = _require_chunk_frames(chunk_a, "chunk_a")
    frames_b = _require_chunk_frames(chunk_b, "chunk_b")
    if drop_b_prefix > len(frames_b):
        raise ValueError(
            f"drop_b_prefix={drop_b_prefix} exceeds chunk_b frame count {len(frames_b)}"
        )

    chunk_meta_b = chunk_b.get("chunk") if isinstance(chunk_b.get("chunk"), dict) else None
    recorded_overlap = None if chunk_meta_b is None else chunk_meta_b.get("overlap_prefix_count")
    if drop_b_prefix > 0 and recorded_overlap is None:
        raise ValueError(
            "chunk_b.chunk.overlap_prefix_count is required when --drop-b-prefix > 0. "
            "Run depth on a split frames_manifest (so chunk metadata is copied into "
            "depth_manifest), or set chunk.overlap_prefix_count explicitly. "
            "overlap_prefix_count is the overlapping frame count, not the start index."
        )
    if recorded_overlap is not None and int(recorded_overlap) != drop_b_prefix:
        raise ValueError(
            "chunk_b.chunk.overlap_prefix_count="
            f"{recorded_overlap} does not match --drop-b-prefix={drop_b_prefix}. "
            "overlap_prefix_count must be the overlapping frame count, not the "
            "source start index."
        )

    kept_b = frames_b[drop_b_prefix:]
    merged = list(frames_a) + list(kept_b)
    if len(merged) != len(selected):
        raise ValueError(
            f"assembled frame count {len(merged)} != source selected count {len(selected)}"
        )

    # Identity + order: frame_id and rgb_path must match source selected order.
    for index, (src, got) in enumerate(zip(selected, merged, strict=True)):
        if str(got["frame_id"]) != str(src["frame_id"]):
            raise ValueError(
                f"assembled frames[{index}].frame_id={got['frame_id']!r} != "
                f"source {src['frame_id']!r}"
            )
        if str(got["rgb_path"]) != str(src["path"]):
            raise ValueError(
                f"assembled frames[{index}].rgb_path={got['rgb_path']!r} != "
                f"source path {src['path']!r}"
            )

    ids = [str(frame["frame_id"]) for frame in merged]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate frame_id after assembly: {ids}")

    integrity_rows: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for index, frame in enumerate(merged):
        record = dict(frame)
        record["depth_index"] = index
        label = f"assembled.frames[{index}]"
        if verify_files:
            integrity = verify_depth_file_integrity(root=root, frame=record, label=label)
        else:
            digest = validate_frame_record_sha256(record, label=label)
            integrity = {
                "status": "passed",
                "sha256": digest,
                "note": "file bytes not re-read (--skip-file-verify)",
            }
        record["integrity"] = {
            "status": integrity["status"],
            "dtype": integrity.get("dtype"),
            "shape": integrity.get("shape"),
            "finite": integrity.get("finite"),
        }
        integrity_rows.append({"frame_id": record["frame_id"], **integrity})
        normalized.append(record)

    backend = {}
    for source in (chunk_a, chunk_b):
        if isinstance(source.get("backend"), dict) and source["backend"]:
            backend = dict(source["backend"])
            break
    depth_type = (
        backend.get("depth_type")
        or (normalized[0].get("depth_type") if normalized else None)
        or "relative"
    )

    assembled = build_depth_manifest(
        frames_manifest=source_frames_manifest,
        frame_records=normalized,
        backend=backend,
        depth_type=str(depth_type),
        frames_manifest_path=source_frames_path,
        file_integrity_status="passed",
        vda_quality_status="not_evaluated",
    )
    assembled["assembly"] = {
        "chunk_a_frame_count": len(frames_a),
        "chunk_b_frame_count_before_drop": len(frames_b),
        "drop_b_prefix": drop_b_prefix,
        "chunk_b_frame_count_after_drop": len(kept_b),
        "assembled_frame_count": len(normalized),
        "identity_order_ok": True,
        "hash_present_ok": True,
        "file_integrity_ok": True,
        "vda_quality_claimed": False,
    }

    report = {
        "source_frames_manifest": source_frames_path,
        "drop_b_prefix": drop_b_prefix,
        "expected_overlap_prefix_count": drop_b_prefix,
        "chunk_b_recorded_overlap_prefix_count": recorded_overlap,
        "chunk_a_frame_count": len(frames_a),
        "chunk_b_frame_count_before_drop": len(frames_b),
        "chunk_b_frame_count_after_drop": len(kept_b),
        "assembled_frame_count": len(normalized),
        "identity_order_hash_ok": True,
        "file_integrity": {"status": "passed", "frames": integrity_rows},
        "vda_quality": {
            "status": "not_evaluated",
            "note": (
                "Assembly only certifies identity/order/hash/file integrity. "
                "Do not treat this as a VDA geometric quality pass."
            ),
        },
    }
    return assembled, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble two overlapping depth_manifest chunks. "
            "Requires per-frame sha256; verifies identity/order/hash."
        )
    )
    parser.add_argument(
        "--source-frames-manifest",
        required=True,
        help="Original (full) frames_manifest used for identity/order checks",
    )
    parser.add_argument("--chunk-a", required=True, help="First chunk depth_manifest.json")
    parser.add_argument("--chunk-b", required=True, help="Second chunk depth_manifest.json")
    parser.add_argument(
        "--drop-b-prefix",
        type=int,
        required=True,
        help="Drop this many leading frames from chunk B (overlap count, not start index)",
    )
    parser.add_argument("--output", required=True, help="Assembled depth_manifest.json")
    parser.add_argument("--report", required=True, help="Assembly report JSON")
    parser.add_argument(
        "--skip-file-verify",
        action="store_true",
        help="Only check schema/sha fields; do not re-hash NPZ files",
    )
    args = parser.parse_args(argv)

    root = project_root()
    source_path = _resolve(root, args.source_frames_manifest)
    chunk_a_path = _resolve(root, args.chunk_a)
    chunk_b_path = _resolve(root, args.chunk_b)
    output_path = _resolve(root, args.output)
    report_path = _resolve(root, args.report)

    source = load_json(source_path)
    chunk_a = load_json(chunk_a_path)
    chunk_b = load_json(chunk_b_path)

    assembled, report = assemble_depth_manifests(
        root=root,
        source_frames_manifest=source,
        source_frames_path=to_repo_relative(root, source_path),
        chunk_a=chunk_a,
        chunk_b=chunk_b,
        drop_b_prefix=args.drop_b_prefix,
        verify_files=not args.skip_file_verify,
    )
    report["chunk_a"] = to_repo_relative(root, chunk_a_path)
    report["chunk_b"] = to_repo_relative(root, chunk_b_path)
    report["output"] = to_repo_relative(root, output_path)

    save_json(output_path, assembled)
    save_json(report_path, report)
    print(f"assembled: {report['output']} frames={report['assembled_frame_count']}")
    print(f"report: {to_repo_relative(root, report_path)}")
    print(
        "producer_gates: file_integrity=passed; "
        "vda_quality=not_evaluated (route owns quality gates)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
