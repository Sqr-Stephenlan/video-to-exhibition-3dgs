"""
Standard 3DGS PLY converter for LongSplat outputs.

Invokes the locked ``convert_3dgs.py`` to produce a standard 3DGS PLY
from the native LongSplat anchor+MLP model.  Validates the output.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

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

# Additional SH attributes that may be present (degrees 1-3).
_SH_ATTR_PATTERNS = ("f_rest_",)


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


def _validate_converted_ply(ply_path: Path) -> dict[str, Any]:
    if not ply_path.is_file():
        raise ConvertedPLYValidationError(
            f"Converted PLY not found: {ply_path}"
        )

    # Basic size check — empty PLY header is ~20 bytes
    file_size = ply_path.stat().st_size
    if file_size < 100:
        raise ConvertedPLYValidationError(
            f"Converted PLY is too small ({file_size} bytes): {ply_path}"
        )

    sha = _sha256_hex(ply_path)

    # Read PLY header to validate attributes
    from plyfile import PlyData

    ply = PlyData.read(ply_path)
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

    return {
        "path": str(ply_path),
        "vertex_count": int(vertex.count),
        "attributes": sorted(attr_names),
        "sha256": sha,
        "file_size": file_size,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
