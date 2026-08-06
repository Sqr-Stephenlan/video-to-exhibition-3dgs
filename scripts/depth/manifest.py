"""frames_manifest / depth_manifest helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

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

    Different content at the same timestamp is an error (ambiguous ordering
    within one VDA invocation). Call this per inference batch / segment.
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
                "Duplicate timestamp_sec with differing image content within one "
                f"VDA batch; timestamp={timestamp}, frame_ids={ids}. "
                "Ensure segment_id separates conflicting frames, or resolve upstream."
            )
        # Identical bytes: keep the first occurrence (stable provenance).
        kept.append(group[0])
    return kept


def _has_timestamp(frame: dict[str, Any]) -> bool:
    return "timestamp_sec" in frame and frame["timestamp_sec"] is not None


def _has_segment_id(frame: dict[str, Any]) -> bool:
    value = frame.get("segment_id")
    return value is not None and value != ""


def _sort_and_dedupe_batch(
    frames: list[dict[str, Any]],
    *,
    root: Path | None,
    dedupe_timestamps: bool,
) -> list[dict[str, Any]]:
    selected = list(frames)
    if selected and all(_has_timestamp(frame) for frame in selected):
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


def prepare_vda_batches(
    frames: list[dict[str, Any]],
    *,
    infer_per_segment: bool = True,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """
    Split selected frames into separate VDA invocations.

    Pinned VDA consumes a decoded frame array / index windows and does not use
    container PTS for temporal reset. Large gaps therefore do not reset the
    model; separate invocations (typically one per segment_id) do.
    """
    if not frames:
        raise ValueError("No frames to partition for VDA")
    if not infer_per_segment:
        return [("__all__", list(frames))]

    flags = [_has_segment_id(frame) for frame in frames]
    if any(flags) and not all(flags):
        raise ValueError(
            "frames_manifest: either all selected frames must include segment_id "
            "or none of them may when runtime.infer_per_segment is enabled"
        )
    if not all(flags):
        return [("__all__", list(frames))]

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for frame in frames:
        key = str(frame["segment_id"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(frame)

    def _segment_sort_key(segment_id: str) -> tuple[float, str]:
        batch = grouped[segment_id]
        if all(_has_timestamp(frame) for frame in batch):
            return (min(float(frame["timestamp_sec"]) for frame in batch), segment_id)
        return (float(order.index(segment_id)), segment_id)

    ordered_ids = sorted(order, key=_segment_sort_key)
    return [(segment_id, grouped[segment_id]) for segment_id in ordered_ids]


def selected_frames(
    frames_manifest: dict[str, Any],
    *,
    root: Path | None = None,
    dedupe_timestamps: bool = True,
    infer_per_segment: bool = True,
) -> list[dict[str, Any]]:
    """
    Return selected frames in VDA input order (flattened across batches).

    Mapping to depth slices is strict-positional: depths[i] <-> selected[i].
    When infer_per_segment is true and every frame has segment_id, ordering and
    timestamp dedupe are applied within each segment, then segments are ordered
    by earliest timestamp (or first-seen order).
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

    has_ts = [_has_timestamp(frame) for frame in selected]
    if any(has_ts) and not all(has_ts):
        raise ValueError(
            "frames_manifest: either all selected frames must include timestamp_sec "
            "or none of them may (mixed timestamps are ambiguous for VDA ordering)"
        )

    has_seg = [_has_segment_id(frame) for frame in selected]
    if any(has_seg) and not all(has_seg):
        raise ValueError(
            "frames_manifest: either all selected frames must include segment_id "
            "or none of them may (mixed segment_id is ambiguous for VDA batching)"
        )

    if infer_per_segment and all(has_seg):
        provisional = prepare_vda_batches(selected, infer_per_segment=True)
        processed: list[tuple[str, list[dict[str, Any]]]] = []
        for segment_id, batch in provisional:
            processed.append(
                (
                    segment_id,
                    _sort_and_dedupe_batch(
                        batch, root=root, dedupe_timestamps=dedupe_timestamps
                    ),
                )
            )

        def _batch_sort_key(
            item: tuple[str, list[dict[str, Any]]],
        ) -> tuple[float, str]:
            segment_id, batch = item
            if batch and all(_has_timestamp(frame) for frame in batch):
                return (
                    min(float(frame["timestamp_sec"]) for frame in batch),
                    segment_id,
                )
            return (float([key for key, _ in provisional].index(segment_id)), segment_id)

        processed.sort(key=_batch_sort_key)
        return [frame for _, batch in processed for frame in batch]

    return _sort_and_dedupe_batch(
        selected, root=root, dedupe_timestamps=dedupe_timestamps
    )


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


_SHA256_HEX = frozenset("0123456789abcdef")


def validate_frame_record_sha256(frame: dict[str, Any], *, label: str) -> str:
    """Require a 64-char lowercase hex sha256 on a depth frame record."""
    digest = frame.get("sha256")
    if not isinstance(digest, str) or not digest:
        raise ValueError(f"{label}: sha256 is required (64 lowercase hex chars)")
    if len(digest) != 64 or any(ch not in _SHA256_HEX for ch in digest):
        raise ValueError(
            f"{label}: sha256 must be 64 lowercase hex chars, got {digest!r}"
        )
    return digest


def verify_depth_file_integrity(
    *,
    root: Path,
    frame: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    """
    Verify NPZ file integrity for one producer frame record.

    This gate covers hash / dtype / finite / shape only. It does **not** evaluate
    VDA geometric quality (correlation, inlier ratio, nRMSE).
    """
    digest = validate_frame_record_sha256(frame, label=label)
    depth_rel = frame.get("depth_path")
    if not isinstance(depth_rel, str) or not depth_rel:
        raise ValueError(f"{label}: depth_path is required")
    path = resolve_repo_path(root, depth_rel)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: depth file missing: {depth_rel}")
    actual = _sha256_file(path)
    if actual != digest:
        raise ValueError(
            f"{label}: depth NPZ sha256 mismatch: expected {digest}, got {actual}"
        )

    with np.load(path, allow_pickle=False) as data:
        if "depth" not in data:
            raise ValueError(f"{label}: NPZ missing 'depth' key")
        depth = data["depth"]
    if depth.dtype != np.float32:
        raise ValueError(f"{label}: depth dtype must be float32, got {depth.dtype}")
    if depth.ndim != 2 or depth.shape[0] == 0 or depth.shape[1] == 0:
        raise ValueError(f"{label}: depth must be non-empty 2-D, got {depth.shape}")
    if not np.all(np.isfinite(depth)):
        raise ValueError(f"{label}: depth contains NaN or Inf")
    return {
        "status": "passed",
        "sha256": digest,
        "dtype": "float32",
        "shape": [int(depth.shape[0]), int(depth.shape[1])],
        "finite": True,
    }


def producer_gates_block(
    *,
    file_integrity_status: str = "passed",
    vda_quality_status: str = "not_evaluated",
) -> dict[str, Any]:
    """
    Explicit dual-gate block for depth_manifest.

    ``file_integrity`` may pass when NPZ files are readable and hashed.
    ``vda_quality`` stays ``not_evaluated`` on the producer unless a consumer
    quality audit is attached later; never treat integrity as quality pass.
    """
    if file_integrity_status not in {"passed", "failed", "not_evaluated"}:
        raise ValueError(f"invalid file_integrity status: {file_integrity_status}")
    if vda_quality_status not in {"passed", "failed", "not_evaluated"}:
        raise ValueError(f"invalid vda_quality status: {vda_quality_status}")
    return {
        "file_integrity": {
            "status": file_integrity_status,
            # Checks actually enforced when writing/verifying per-frame NPZ.
            # Identity/order vs a full source manifest is enforced by assemble, not here.
            "checks": ["sha256", "float32", "finite", "shape_2d"],
        },
        "vda_quality": {
            "status": vda_quality_status,
            "owner": "consumer/route",
            "checks": ["correlation", "inlier", "nRMSE", "aligned"],
            "note": (
                "Depth-prior producer does not claim geometric VDA quality. "
                "Route pose/VDA quality gates remain authoritative."
            ),
        },
    }


def _normalized_chunk_metadata(chunk: Any, *, label: str) -> dict[str, Any]:
    """Validate and copy split-chunk provenance onto depth_manifest."""
    if not isinstance(chunk, dict):
        raise ValueError(f"{label} must be an object when present")
    for key in (
        "chunk_id",
        "source_index_start",
        "source_index_end",
        "frame_count",
        "overlap_prefix_count",
    ):
        if key not in chunk:
            raise ValueError(f"{label}.{key} is required")
    chunk_id = chunk["chunk_id"]
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError(f"{label}.chunk_id must be a non-empty string")
    start = int(chunk["source_index_start"])
    end = int(chunk["source_index_end"])
    frame_count = int(chunk["frame_count"])
    overlap_prefix_count = int(chunk["overlap_prefix_count"])
    if start < 0 or end <= start:
        raise ValueError(f"{label}: need 0 <= source_index_start < source_index_end")
    if frame_count != end - start:
        raise ValueError(
            f"{label}: frame_count={frame_count} != "
            f"source_index_end-start ({end - start})"
        )
    if overlap_prefix_count < 0 or overlap_prefix_count > frame_count:
        raise ValueError(
            f"{label}.overlap_prefix_count must be in [0, frame_count], "
            f"got {overlap_prefix_count}"
        )
    overlap_with = chunk.get("overlap_with_chunk_id")
    if overlap_with is not None and (
        not isinstance(overlap_with, str) or not overlap_with
    ):
        raise ValueError(f"{label}.overlap_with_chunk_id must be a non-empty string or null")
    if overlap_prefix_count > 0 and not overlap_with:
        raise ValueError(
            f"{label}: overlap_with_chunk_id is required when overlap_prefix_count > 0"
        )
    return {
        "chunk_id": chunk_id,
        "source_index_start": start,
        "source_index_end": end,
        "frame_count": frame_count,
        "overlap_prefix_count": overlap_prefix_count,
        "overlap_with_chunk_id": overlap_with,
    }


def build_depth_manifest(
    *,
    frames_manifest: dict[str, Any],
    frame_records: list[dict[str, Any]],
    backend: dict[str, Any],
    depth_type: str,
    frames_manifest_path: str | None = None,
    file_integrity_status: str = "passed",
    vda_quality_status: str = "not_evaluated",
) -> dict[str, Any]:
    """
    Build depth_manifest.json.

    frame_depth_mapping is always strict_positional: depths[i] matches the i-th
    selected frame after selected_frames() ordering (optional timestamp sort/dedupe).

    Every frame record must include ``sha256`` (64 lowercase hex of the NPZ).
    When the input frames_manifest carries ``chunk`` provenance (from
    ``split_selected_manifest.py``), it is copied onto the depth_manifest so
    assemble can verify overlap_prefix_count.
    """
    if not frame_records:
        raise ValueError("frame_records must be non-empty")
    for index, frame in enumerate(frame_records):
        validate_frame_record_sha256(frame, label=f"frames[{index}]")
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "source_video_id": frames_manifest.get("video_id"),
        "source_frames_manifest": frames_manifest_path,
        "frame_depth_mapping": "strict_positional",
        "producer_gates": producer_gates_block(
            file_integrity_status=file_integrity_status,
            vda_quality_status=vda_quality_status,
        ),
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
    if "chunk" in frames_manifest and frames_manifest["chunk"] is not None:
        payload["chunk"] = _normalized_chunk_metadata(
            frames_manifest["chunk"], label="frames_manifest.chunk"
        )
        if payload["chunk"]["frame_count"] != len(frame_records):
            raise ValueError(
                "frames_manifest.chunk.frame_count="
                f"{payload['chunk']['frame_count']} != depth frame_records "
                f"{len(frame_records)}"
            )
    return payload
