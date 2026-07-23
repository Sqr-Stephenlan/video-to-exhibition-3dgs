"""
Strict depth bridge: validate and materialise VDA depth NPZ → npy.

Every NPZ is checked for project-root containment, SHA-256, 2-D shape,
float32 dtype, and finite values before being atomically written as an
npy file.  No depth reaches the backend without passing every check.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterializedDepth:
    """One successfully materialised depth frame."""

    rgb_path: str
    prepared_name: str
    source_path: str
    source_sha256: str
    output_path: str
    output_sha256: str
    shape: tuple[int, int]
    dtype: str


@dataclass(frozen=True)
class DepthMaterializationResult:
    """Aggregate result of materialising all depth frames."""

    expected_count: int
    materialized_count: int
    depth_manifest_sha256: str
    frames: tuple[MaterializedDepth, ...]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DepthContractError(Exception):
    """Raised when a depth input violates the consumption contract."""


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------


def resolve_relative_file(project_root: Path, value: str, label: str) -> Path:
    """Resolve *value* relative to *project_root* with strict containment.

    Rejects absolute paths, ``..`` traversal, symlink escape, and
    non-file targets.  Returns the resolved ``Path``.
    """
    root = project_root.resolve(strict=True)
    relative = Path(value)
    if relative.is_absolute():
        raise DepthContractError(f"{label} must be relative to project_root: {value}")
    if ".." in relative.parts:
        raise DepthContractError(f"{label} escapes project_root: {value}")
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DepthContractError(f"{label} escapes project_root: {value}") from exc
    if not resolved.is_file():
        raise DepthContractError(f"{label} is not a file: {value}")
    return resolved


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def validate_depth_manifest(manifest: dict, project_root: Path) -> list[dict]:
    """Validate depth manifest structure and return the normalised frame list.

    Raises ``DepthContractError`` on any violation:
    - frames is a non-empty list
    - every frame has non-empty string ``rgb_path``, ``depth_path``, ``sha256``
    - ``sha256`` is 64 lowercase hex characters
    - no duplicate normalised ``rgb_path`` entries
    """
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise DepthContractError("depth manifest has no non-empty 'frames' array")

    seen_rgb: set[str] = set()
    normalized: list[dict] = []

    for i, frame in enumerate(frames):
        label = f"depth_manifest.frames[{i}]"

        rgb_path = frame.get("rgb_path")
        if not rgb_path or not isinstance(rgb_path, str):
            raise DepthContractError(f"{label}: rgb_path must be a non-empty string")
        if rgb_path in seen_rgb:
            raise DepthContractError(f"{label}: duplicate rgb_path {rgb_path!r}")
        seen_rgb.add(rgb_path)

        depth_path = frame.get("depth_path")
        if not depth_path or not isinstance(depth_path, str):
            raise DepthContractError(f"{label}: depth_path must be a non-empty string")

        sha256 = frame.get("sha256")
        if not sha256 or not isinstance(sha256, str):
            raise DepthContractError(f"{label}: sha256 must be a non-empty string")
        if not _SHA256_RE.match(sha256):
            raise DepthContractError(
                f"{label}: sha256 must be 64 lowercase hex chars, got {sha256!r}"
            )

        normalized.append(
            {
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "sha256": sha256,
            }
        )

    return normalized


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_RE = _SHA256_HEX_RE


# ---------------------------------------------------------------------------
# NPZ loading and validation
# ---------------------------------------------------------------------------


def validate_and_load_npz(npz_path: Path, expected_sha256: str) -> np.ndarray:
    """Load and validate a single depth NPZ.

    1. File SHA-256 must match *expected_sha256*.
    2. Must contain key ``"depth"``.
    3. Array must be 2-D with positive dimensions.
    4. Dtype must be exactly ``float32``.
    5. Every value must be finite.

    Returns the validated ``float32`` 2-D array.
    """
    # SHA-256
    actual_sha = _sha256_hex(npz_path)
    if actual_sha != expected_sha256:
        raise DepthContractError(
            f"{npz_path}: SHA-256 mismatch — "
            f"expected {expected_sha256}, got {actual_sha}"
        )

    # Load
    data = np.load(npz_path, allow_pickle=False)
    if "depth" not in data:
        raise DepthContractError(
            f"{npz_path}: NPZ missing 'depth' key; keys found: {sorted(data.keys())}"
        )

    depth = data["depth"]

    # Shape
    if depth.ndim != 2:
        raise DepthContractError(
            f"{npz_path}: depth must be 2-D, got ndim={depth.ndim} shape={depth.shape}"
        )
    if depth.shape[0] == 0 or depth.shape[1] == 0:
        raise DepthContractError(
            f"{npz_path}: depth dimensions must be positive, got {depth.shape}"
        )

    # Dtype
    if depth.dtype != np.float32:
        raise DepthContractError(
            f"{npz_path}: depth dtype must be float32, got {depth.dtype}"
        )

    # Finite
    if not np.all(np.isfinite(depth)):
        raise DepthContractError(f"{npz_path}: depth contains NaN or Inf values")

    return depth


# ---------------------------------------------------------------------------
# Atomic npy write
# ---------------------------------------------------------------------------


def atomic_save_npy(destination: Path, depth: np.ndarray) -> None:
    """Atomically write *depth* as a ``.npy`` file at *destination*.

    Uses mkstemp + fsync + os.replace.  On failure the destination is
    left unchanged.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(suffix=".npy", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, depth, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
    finally:
        Path(temp_name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def materialize_all(
    depth_manifest_path: Path,
    frame_mapping: list[dict],
    project_root: Path,
    output_depth_dir: Path,
) -> DepthMaterializationResult:
    """Materialise every depth frame and return a structured result.

    Parameters
    ----------
    depth_manifest_path : Path
        Path to ``depth_manifest.json``.
    frame_mapping : list[dict]
        Prepared frame mapping; each entry has ``source_path`` and
        ``prepared_name``.
    project_root : Path
        Repository root for path resolution and containment.
    output_depth_dir : Path
        Directory where ``<stem>_depth.npy`` files are written.

    Returns
    -------
    DepthMaterializationResult
    """
    with depth_manifest_path.open("rb") as fh:
        raw = fh.read()
    manifest = json.loads(raw)
    manifest_sha = hashlib.sha256(raw).hexdigest()

    frames = validate_depth_manifest(manifest, project_root)

    # Build lookup: producer rgb_path → {depth_path, sha256}
    depth_by_rgb: dict[str, dict[str, str]] = {}
    for f in frames:
        depth_by_rgb[f["rgb_path"]] = {
            "depth_path": f["depth_path"],
            "sha256": f["sha256"],
        }

    # --- Exact coverage: set equality between manifest and mapping ---
    mapping_sources = [entry["source_path"] for entry in frame_mapping]
    destinations = [
        f"{Path(entry['prepared_name']).stem}_depth.npy" for entry in frame_mapping
    ]
    if len(set(mapping_sources)) != len(mapping_sources):
        raise DepthContractError("frame_mapping contains duplicate source_path")
    if len(set(destinations)) != len(destinations):
        raise DepthContractError("frame_mapping creates duplicate depth output")

    manifest_sources = set(depth_by_rgb)
    mapping_source_set = set(mapping_sources)
    missing = sorted(mapping_source_set - manifest_sources)
    unused = sorted(manifest_sources - mapping_source_set)
    if missing:
        raise DepthContractError(f"missing depth entries: {missing}")
    if unused:
        raise DepthContractError(f"unused depth entries: {unused}")

    output_depth_dir.mkdir(parents=True, exist_ok=True)

    materialized: list[MaterializedDepth] = []
    expected = len(frame_mapping)

    for entry in frame_mapping:
        source = entry.get("source_path", "")
        prepared_name = entry["prepared_name"]
        stem = Path(prepared_name).stem  # preserves "frame_001_t10.000"

        df = depth_by_rgb.get(source)
        if df is None:
            raise DepthContractError(
                f"No depth entry for source_path {source!r} "
                f"(prepared_name={prepared_name!r})"
            )

        npz_path = resolve_relative_file(
            project_root,
            df["depth_path"],
            f"depth for {prepared_name}",
        )

        depth_arr = validate_and_load_npz(npz_path, df["sha256"])

        npy_dest = output_depth_dir / f"{stem}_depth.npy"
        atomic_save_npy(npy_dest, depth_arr)

        output_sha = _sha256_hex(npy_dest)
        materialized.append(
            MaterializedDepth(
                rgb_path=source,
                prepared_name=prepared_name,
                source_path=str(npz_path),
                source_sha256=df["sha256"],
                output_path=str(npy_dest),
                output_sha256=output_sha,
                shape=(int(depth_arr.shape[0]), int(depth_arr.shape[1])),
                dtype=str(depth_arr.dtype),
            )
        )

    return DepthMaterializationResult(
        expected_count=expected,
        materialized_count=len(materialized),
        depth_manifest_sha256=manifest_sha,
        frames=tuple(materialized),
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
