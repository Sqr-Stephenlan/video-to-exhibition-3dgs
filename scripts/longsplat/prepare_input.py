"""
Prepare LongSplat input directory from a validated frame manifest.

Copies frames to a run-scoped input directory and saves a mapping from
source frame IDs to prepared filenames.  Only the copy strategy is
implemented in this version (no hardlink / symlink).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_input(
    manifest: dict[str, Any],
    run_dir: str | Path,
) -> Path:
    """Copy frames listed in *manifest* into ``<run_dir>/input/images/``.

    Parameters
    ----------
    manifest : dict
        A validated frame manifest (see :func:`validate_input.validate_manifest`).
    run_dir : str or Path
        Immutable run directory.  A subdirectory ``input/images/`` will be
        created inside it.

    Returns
    -------
    Path
        The prepared image directory (``<run_dir>/input/images``).

    Side effects
    ------------
    - Copies each frame into the prepared directory.
    - Writes ``input/frame_mapping.json`` mapping source ``frame_id`` →
      prepared filename and original path.
    - Verifies SHA-256 of each copied file against the manifest.
    """
    rd = Path(run_dir)
    img_dir = rd / "input" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    base = Path(manifest.get("base", "."))
    frames: list[dict[str, Any]] = manifest["frames"]
    mapping: list[dict[str, Any]] = []

    for frame in frames:
        src = (base / frame["path"]).resolve()
        expected_sha = frame["sha256"]
        frame_id: int = frame["frame_id"]

        # Use a stable prepared name: frame_<id:06d>.<ext>
        ext = src.suffix or ".jpg"
        dst_name = f"frame_{frame_id:06d}{ext}"
        dst = img_dir / dst_name

        shutil.copy2(src, dst)

        actual_sha = _sha256_hex(dst)
        if actual_sha != expected_sha:
            raise PreparedFileHashMismatch(
                f"Hash mismatch for frame {frame_id}: "
                f"expected {expected_sha}, got {actual_sha}"
            )

        mapping.append(
            {
                "frame_id": frame_id,
                "source_path": frame["path"],
                "prepared_name": dst_name,
                "sha256": actual_sha,
            }
        )

    mapping_path = rd / "input" / "frame_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)

    return img_dir


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PreparedFileHashMismatch(Exception):
    """Raised when a copied file's SHA-256 does not match the manifest."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
