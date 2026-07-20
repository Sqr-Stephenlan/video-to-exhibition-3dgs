"""
Standard 3DGS PLY converter for LongSplat outputs.

Invokes the locked ``convert_3dgs.py`` to produce a standard 3DGS PLY
from the native LongSplat anchor+MLP model.  Validates the output.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.longsplat.runner import (
    LongSplatConfig,
    _check_repo,
    build_convert_command,
)

# Standard 3DGS PLY attributes expected in the converted output.
_STANDARD_3DGS_ATTRS = frozenset(
    {
        "x", "y", "z",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }
)

class ConverterError(Exception):
    """Raised when conversion fails or produces invalid output."""


class ConvertedPLYValidationError(Exception):
    """Raised when the converted PLY fails schema or content validation."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_and_validate(
    repo_root: str | Path,
    config: LongSplatConfig,
    python_exe: str = "python",
) -> Path:
    """Run ``convert_3dgs.py`` and validate the resulting PLY.

    Parameters
    ----------
    repo_root : str or Path
        Root of the LongSplat repository at the locked commit.
    config : LongSplatConfig
        Must include ``model_path`` pointing to a trained LongSplat model.
    python_exe : str
        Python executable for the backend runtime.

    Returns
    -------
    Path
        Path to the validated converted PLY
        (``<model_path>/converted_3dgs/point_cloud.ply``).

    Raises
    ------
    BackendValidationError
        If the backend repo or python exe is invalid.
    ConverterError
        If the subprocess exits non-zero.
    ConvertedPLYValidationError
        If the output PLY is missing, empty, or has wrong schema.
    """
    _check_repo(repo_root)

    cmd = build_convert_command(repo_root, config, python_exe)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ConverterError(
            f"convert_3dgs.py exited with {result.returncode}:\n"
            f"{result.stderr[-2000:]}"
        )

    ply_path = (
        Path(config.model_path) / "converted_3dgs" / "point_cloud.ply"
    )
    _validate_converted_ply(ply_path)
    return ply_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_converted_ply(ply_path: str | Path) -> dict[str, Any]:
    """Validate a converted PLY without running the converter.

    Use this on an existing PLY to verify it satisfies the standard 3DGS
    contract before passing it to downstream consumers.

    Returns a metadata dict with vertex count, attributes, and SHA-256.
    """
    return _validate_converted_ply(Path(ply_path))


def clean_converted_ply(ply_path: str | Path) -> int:
    """Remove vertices with NaN/Inf in core 3DGS attributes.

    Returns the number of vertices removed.  The file is overwritten in-place
    only when invalid vertices are found.
    """
    import io
    import tempfile
    from plyfile import PlyData, PlyElement

    pp = Path(ply_path).resolve()
    if not pp.is_file():
        raise ConvertedPLYValidationError(f"PLY not found: {pp}")

    with open(pp, "rb") as fh:
        raw = fh.read()
    ply = PlyData.read(io.BytesIO(raw))
    vertex = ply["vertex"]
    if vertex.count == 0:
        return 0

    fields = [
        "x", "y", "z", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    available = [f for f in fields if f in vertex.data.dtype.names]
    mask = np.ones(vertex.count, dtype=bool)
    for f in available:
        mask &= np.isfinite(vertex.data[f])

    removed = vertex.count - int(mask.sum())
    if removed == 0:
        return 0

    filtered = vertex.data[mask]
    elements = np.empty(len(filtered), dtype=vertex.data.dtype)
    for name in vertex.data.dtype.names:
        elements[name] = filtered[name]

    el = PlyElement.describe(elements, "vertex")
    data = PlyData([el])
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ply", dir=str(pp.parent))
    os.close(tmp_fd)
    data.write(tmp_path)
    pp.unlink()
    Path(tmp_path).rename(pp)
    return removed


def _validate_converted_ply(ply_path: Path) -> dict[str, Any]:
    if not ply_path.is_file():
        raise ConvertedPLYValidationError(
            f"Converted PLY not found: {ply_path}"
        )

    file_size = ply_path.stat().st_size
    if file_size < 100:
        raise ConvertedPLYValidationError(
            f"Converted PLY is too small ({file_size} bytes): {ply_path}"
        )

    sha = _sha256_hex(ply_path)

    from plyfile import PlyData

    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ConvertedPLYValidationError(
            f"Converted PLY has no 'vertex' element: {ply_path}"
        )
    vertex = ply["vertex"]
    if vertex.count == 0:
        raise ConvertedPLYValidationError(
            f"Converted PLY has zero vertices: {ply_path}"
        )

    attr_names = set(vertex.data.dtype.names)
    missing_core = _STANDARD_3DGS_ATTRS - attr_names
    if missing_core:
        raise ConvertedPLYValidationError(
            f"Converted PLY missing core 3DGS attributes: {sorted(missing_core)}"
        )

    # --- dtype checks ---
    _check_dtype(vertex, "x", "f4")
    _check_dtype(vertex, "y", "f4")
    _check_dtype(vertex, "z", "f4")
    _check_dtype(vertex, "opacity", "f4")
    for attr in ("scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"):
        _check_dtype(vertex, attr, "f4")

    # --- NaN/Inf checks on position, opacity, scale, rotation ---
    _finite_fields = [
        "x", "y", "z", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    for field in _finite_fields:
        if field in attr_names:
            values = vertex.data[field]
            if not np.all(np.isfinite(values)):
                raise ConvertedPLYValidationError(
                    f"Converted PLY contains NaN/Inf in '{field}': {ply_path}"
                )

    # --- quaternion non-degeneracy check (sum of squares > 0) ---
    rot_fields = ["rot_0", "rot_1", "rot_2", "rot_3"]
    if all(r in attr_names for r in rot_fields):
        rot_data = np.stack(
            [vertex.data[r] for r in rot_fields], axis=-1
        )
        rot_norms = np.sum(rot_data ** 2, axis=-1)
        if np.any(rot_norms == 0):
            raise ConvertedPLYValidationError(
                f"Converted PLY contains degenerate quaternions (zero norm): {ply_path}"
            )

    # --- SH attribute count consistency (informational, not a hard failure) ---
    sh_attrs = [a for a in attr_names if a.startswith("f_rest_")]
    sh_order = len(sh_attrs)

    return {
        "path": str(ply_path),
        "vertex_count": int(vertex.count),
        "attributes": sorted(attr_names),
        "sha256": sha,
        "file_size": file_size,
        "sh_rest_count": sh_order,
    }


def _check_dtype(vertex, name: str, expected: str) -> None:
    """Check that a vertex attribute has the expected numpy dtype."""
    if name not in vertex.data.dtype.names:
        return
    actual_dtype = vertex.data.dtype[name]
    if np.dtype(actual_dtype) != np.dtype(expected):
        raise ConvertedPLYValidationError(
            f"PLY attribute '{name}' has dtype {actual_dtype}, "
            f"expected {expected}"
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
