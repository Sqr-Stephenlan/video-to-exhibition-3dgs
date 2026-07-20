"""
Frame manifest validator for the LongSplat module.

Read-only consumer: validates a LongSplat-internal manifest against the
contract required by the LongSplat backend.

The full manifest schema is owned by ``feature/preprocess-video``.  Use
:mod:`manifest_adapter` to translate real preprocess manifests into the
format validated here.  This module only enforces the subset needed by
the LongSplat backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Frame record contract (consumption side — not a producer schema)
# ---------------------------------------------------------------------------

_REQUIRED_TOP_KEYS = frozenset(
    {
        "schema_version",
        "segment_id",
        "frames",
    }
)

_REQUIRED_FRAME_KEYS = frozenset(
    {
        "frame_id",
        "path",
        "width",
        "height",
        "sha256",
    }
)

_OPTIONAL_FRAME_KEYS = frozenset(
    {
        "timestamp",
        "pts",
        "time_base",
        "selection_status",
        "selection_reason",
        "producer_run_id",
    }
)

_ALL_KNOWN_FRAME_KEYS = _REQUIRED_FRAME_KEYS | _OPTIONAL_FRAME_KEYS


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class ManifestValidationError(Exception):
    """Raised when a manifest does not satisfy the consumption contract."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate a frame manifest against the LongSplat input contract.

    Parameters
    ----------
    manifest_path : str or Path
        Path to the JSON manifest file.

    Returns
    -------
    dict
        The validated manifest dict (read-only, caller must not mutate).

    Raises
    ------
    ManifestValidationError
        If any contract requirement is violated.
    FileNotFoundError
        If *manifest_path* does not exist or is not a file.
    """
    mp = Path(manifest_path)
    if not mp.is_file():
        raise FileNotFoundError(f"Manifest not found: {mp}")

    with open(mp, encoding="utf-8") as fh:
        manifest = json.load(fh)

    _validate_top_level(manifest, mp)
    _validate_frames(manifest, mp)
    return manifest


# ---------------------------------------------------------------------------
# Internal validators
# ---------------------------------------------------------------------------


def _validate_top_level(manifest: dict[str, Any], source: Path) -> None:
    if not isinstance(manifest, dict):
        raise ManifestValidationError(f"Manifest must be a JSON object: {source}")

    missing = _REQUIRED_TOP_KEYS - set(manifest.keys())
    if missing:
        raise ManifestValidationError(
            f"Manifest {source} missing top-level keys: {sorted(missing)}"
        )

    schema_version = manifest["schema_version"]
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ManifestValidationError(
            f"Manifest {source}: schema_version must be int >= 1, got {schema_version!r}"
        )

    segment_id = manifest["segment_id"]
    if not isinstance(segment_id, str) or not segment_id.strip():
        raise ManifestValidationError(
            f"Manifest {source}: segment_id must be a non-empty string"
        )

    frames = manifest["frames"]
    if not isinstance(frames, list) or len(frames) == 0:
        raise ManifestValidationError(
            f"Manifest {source}: frames must be a non-empty list"
        )


def _validate_frames(manifest: dict[str, Any], source: Path) -> None:
    frames: list[dict[str, Any]] = manifest["frames"]
    seen_ids: set[int] = set()
    base = Path(manifest.get("base", "."))

    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ManifestValidationError(
                f"Manifest {source}: frames[{i}] must be an object"
            )

        # --- required keys -------------------------------------------------
        missing = _REQUIRED_FRAME_KEYS - set(frame.keys())
        if missing:
            raise ManifestValidationError(
                f"Manifest {source}: frames[{i}] missing keys: {sorted(missing)}"
            )

        # --- frame_id ------------------------------------------------------
        frame_id = frame["frame_id"]
        if not isinstance(frame_id, int) or frame_id < 0:
            raise ManifestValidationError(
                f"Manifest {source}: frames[{i}].frame_id must be int >= 0, "
                f"got {frame_id!r}"
            )
        if frame_id in seen_ids:
            raise ManifestValidationError(
                f"Manifest {source}: duplicate frame_id {frame_id} at index {i}"
            )
        seen_ids.add(frame_id)

        # --- path (must not escape base) -----------------------------------
        rel = frame["path"]
        if not isinstance(rel, str) or not rel.strip():
            raise ManifestValidationError(
                f"Manifest {source}: frames[{i}].path must be a non-empty string"
            )
        resolved = (base / rel).resolve()
        try:
            resolved.relative_to(base.resolve())
        except ValueError:
            raise ManifestValidationError(
                f"Manifest {source}: frames[{i}].path {rel!r} escapes base directory"
            )

        # --- width / height ------------------------------------------------
        w, h = frame["width"], frame["height"]
        if not isinstance(w, int) or w <= 0:
            raise ManifestValidationError(
                f"Manifest {source}: frames[{i}].width must be positive int, got {w!r}"
            )
        if not isinstance(h, int) or h <= 0:
            raise ManifestValidationError(
                f"Manifest {source}: frames[{i}].height must be positive int, got {h!r}"
            )

        # --- sha256 --------------------------------------------------------
        sha = frame["sha256"]
        if not isinstance(sha, str) or len(sha) != 64:
            raise ManifestValidationError(
                f"Manifest {source}: frames[{i}].sha256 must be a 64-char hex string"
            )

        # --- optional fields -----------------------------------------------
        unknown = set(frame.keys()) - _ALL_KNOWN_FRAME_KEYS
        if unknown:
            raise ManifestValidationError(
                f"Manifest {source}: frames[{i}] has unknown keys: {sorted(unknown)}"
            )

        ts = frame.get("timestamp")
        if ts is not None and not isinstance(ts, (int, float)):
            raise ManifestValidationError(
                f"Manifest {source}: frames[{i}].timestamp must be numeric, got {ts!r}"
            )

    _validate_ordering(frames, source)


def _validate_ordering(frames: list[dict[str, Any]], source: Path) -> None:
    """Check that frames are ordered deterministically (by frame_id).

    The manifest contract requires frames in ascending frame_id order so
    that downstream consumers do not need to re-sort.
    """
    ids = [f["frame_id"] for f in frames]
    if ids != sorted(ids):
        raise ManifestValidationError(
            f"Manifest {source}: frames must be sorted by ascending frame_id"
        )
