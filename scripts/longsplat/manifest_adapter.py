"""
Adapt real ``feature/preprocess-video`` manifests to the LongSplat consumer contract.

The full manifest schema is owned by ``feature/preprocess-video``.  This module
reads manifests produced by that pipeline and translates them into the subset
required by the LongSplat backend (``validate_input.py``).  No schema is
invented here — every field maps to a producer-defined source.

Versioned handoff contract
--------------------------
* **Producer**: ``feature/preprocess-video``, schema_version ``"1.0"`` (string).
* **Consumer**: ``research/longsplat-route``, internal schema_version ``1`` (int).
* **Translation**: semi-automatic — callers pick a ``segment_id`` from the
  producer manifest and the adapter builds a single-segment consumer manifest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AdapterError(Exception):
    """Raised when a producer manifest cannot be adapted."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def adapt_manifest(
    producer_manifest: dict[str, Any],
    segment_id: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Convert a preprocess-video manifest to a LongSplat consumer manifest.

    Parameters
    ----------
    producer_manifest : dict
        Full manifest as produced by ``feature/preprocess-video``
        (schema_version ``"1.0"``).
    segment_id : str
        Which segment to extract from ``producer_manifest["segments"]``.
    repo_root : str or Path
        Repository root for resolving relative paths.

    Returns
    -------
    dict
        Consumer manifest satisfying the LongSplat ``validate_input`` contract.
    """
    _validate_producer_schema(producer_manifest)

    root = Path(repo_root).resolve()
    segment = _find_segment(producer_manifest, segment_id, root)

    # Frame dimensions come from the producer's normalized metadata
    normalized = producer_manifest["normalized"]
    frame_w = normalized["width"]
    frame_h = normalized["height"]

    consumer_frames: list[dict[str, Any]] = []
    for i, frame in enumerate(producer_manifest["frames"]):
        if frame.get("segment_id") != segment_id:
            continue
        if not frame.get("selected", False):
            continue

        frame_path = frame.get("path")
        if not frame_path or not isinstance(frame_path, str):
            raise AdapterError(
                f"Selected frame {frame.get('id', '?')} in segment {segment_id!r} "
                f"has no path — cannot prepare input."
            )

        abs_path = (root / frame_path).resolve()
        if not abs_path.is_file():
            raise AdapterError(
                f"Frame file not found: {abs_path} "
                f"(frame {frame.get('id', '?')} in segment {segment_id!r})"
            )

        sha = _sha256_hex(abs_path)

        consumer_frames.append(
            {
                "frame_id": i,
                "path": frame_path,
                "width": frame_w,
                "height": frame_h,
                "sha256": sha,
            }
        )

    if not consumer_frames:
        raise AdapterError(
            f"No selected frames found for segment {segment_id!r}"
        )

    return {
        "schema_version": 1,
        "segment_id": segment_id,
        "base": str(root),
        "frames": consumer_frames,
    }


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------


def _validate_producer_schema(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise AdapterError("Producer manifest must be a JSON object")

    sv = manifest.get("schema_version")
    if sv != "1.0":
        raise AdapterError(
            f"Unsupported producer schema_version: {sv!r} (expected '1.0')"
        )

    if not isinstance(manifest.get("segments"), list) or len(manifest["segments"]) == 0:
        raise AdapterError("Producer manifest must contain a non-empty 'segments' list")

    if not isinstance(manifest.get("frames"), list):
        raise AdapterError("Producer manifest must contain a 'frames' list")

    normalized = manifest.get("normalized", {})
    if not isinstance(normalized.get("width"), int) or normalized["width"] <= 0:
        raise AdapterError("Producer manifest normalized.width is missing or invalid")
    if not isinstance(normalized.get("height"), int) or normalized["height"] <= 0:
        raise AdapterError("Producer manifest normalized.height is missing or invalid")


def _find_segment(
    manifest: dict[str, Any],
    segment_id: str,
    repo_root: Path,
) -> dict[str, Any]:
    for seg in manifest["segments"]:
        if seg.get("id") == segment_id:
            return seg
    available = [s.get("id", "?") for s in manifest["segments"]]
    raise AdapterError(
        f"Segment {segment_id!r} not found in producer manifest. "
        f"Available segments: {available}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
