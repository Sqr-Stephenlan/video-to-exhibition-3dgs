"""
Converter contract tests.

CPU-only — uses a small synthetic PLY to test validation logic.
No real GPU, LongSplat training, or checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.longsplat.convert import (  # noqa: E402
    ConverterError,
    ConvertedPLYValidationError,
    convert_and_validate,
    validate_converted_ply,
)
from scripts.longsplat.runner import LongSplatConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ply(path: Path, attributes: list[str], count: int = 10) -> None:
    """Write a minimal synthetic PLY for testing with valid defaults."""
    dtype = [(a, "f4") for a in attributes]
    verts = np.zeros(count, dtype=dtype)
    # Use identity quaternions (rot_0=1) to pass degeneracy checks.
    for attr in ("rot_0", "scale_0", "scale_1", "scale_2", "opacity"):
        if attr in attributes:
            verts[attr] = 1.0
    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=True).write(str(path))


# ---------------------------------------------------------------------------
# validate_converted_ply
# ---------------------------------------------------------------------------


def test_finite_conversion_telemetry_patch_contract():
    """The ordered research patch carries the root fix and all marker schemas."""
    patch = (
        _project_root
        / "docs"
        / "longsplat"
        / "patches"
        / "longsplat_conversion_telemetry.patch"
    )
    text = patch.read_text(encoding="utf-8")

    assert "torch.clamp(dist2_raw, min=1e-7" in text
    assert "if not torch.isfinite(scales).all()" in text
    assert "CONVERSION_TELEMETRY" in text
    assert "POSE_TELEMETRY" in text
    assert "VDA_TELEMETRY" in text


def test_valid_ply_passes(tmp_path):
    ply_path = tmp_path / "ok.ply"
    _make_ply(
        ply_path,
        [
            "x",
            "y",
            "z",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ],
    )
    meta = validate_converted_ply(str(ply_path))
    assert meta["vertex_count"] == 10
    assert len(meta["sha256"]) == 64
    assert meta["file_size"] > 0


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConvertedPLYValidationError, match="not found"):
        validate_converted_ply(str(tmp_path / "nope.ply"))


def test_empty_ply_raises(tmp_path):
    ply_path = tmp_path / "empty.ply"
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    verts = np.empty(0, dtype=dtype)
    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=True).write(str(ply_path))
    with pytest.raises(ConvertedPLYValidationError, match="zero vertices"):
        validate_converted_ply(str(ply_path))


def test_missing_core_attrs_raises(tmp_path):
    ply_path = tmp_path / "partial.ply"
    _make_ply(ply_path, ["x", "y", "z"])
    with pytest.raises(ConvertedPLYValidationError, match="missing core"):
        validate_converted_ply(str(ply_path))


def test_tiny_file_raises(tmp_path):
    ply_path = tmp_path / "tiny.ply"
    ply_path.write_text("not a ply")
    with pytest.raises(ConvertedPLYValidationError, match="too small"):
        validate_converted_ply(str(ply_path))


def test_extra_attrs_allowed(tmp_path):
    """Converted PLY may have additional SH attributes beyond the core set."""
    ply_path = tmp_path / "with_sh.ply"
    _make_ply(
        ply_path,
        [
            "x",
            "y",
            "z",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "f_rest_0",
            "f_rest_1",
        ],
    )
    meta = validate_converted_ply(str(ply_path))
    assert meta["vertex_count"] == 10


# ---------------------------------------------------------------------------
# convert_and_validate (fake subprocess — no real GPU / LongSplat)
# ---------------------------------------------------------------------------


def test_convert_and_validate_success(tmp_path):
    """FV-01: convert_and_validate delegates to run_conversion, validates PLY."""
    config = LongSplatConfig(
        source_path=str(tmp_path / "input"),
        model_path=str(tmp_path / "model"),
        iterations=100,
        seed=0,
    )

    fake_cp = mock.MagicMock(returncode=0, stderr="")
    with mock.patch(
        "scripts.longsplat.convert.run_conversion", return_value=fake_cp
    ) as mock_conv:
        with mock.patch(
            "scripts.longsplat.convert._validate_converted_ply"
        ) as mock_val:
            mock_val.return_value = {
                "path": str(tmp_path / "model" / "converted_3dgs" / "point_cloud.ply"),
                "vertex_count": 100,
                "attributes": ["x", "y", "z"],
                "sha256": "a" * 64,
                "file_size": 1024,
            }
            result = convert_and_validate("/fake/repo", config)

    assert "point_cloud.ply" in str(result)
    mock_conv.assert_called_once()
    mock_val.assert_called_once()


def test_convert_and_validate_nonzero_exit_raises(tmp_path):
    """FV-01: if run_conversion returns non-zero, raise ConverterError."""
    config = LongSplatConfig(
        source_path=str(tmp_path / "input"),
        model_path=str(tmp_path / "model"),
        iterations=100,
        seed=0,
    )

    fake_cp = mock.MagicMock(returncode=1, stderr="CUDA OOM")
    with mock.patch("scripts.longsplat.convert.run_conversion", return_value=fake_cp):
        with pytest.raises(ConverterError, match="exited with 1"):
            convert_and_validate("/fake/repo", config)


def test_convert_and_validate_path_construction(tmp_path):
    """FV-01: verify the PLY output path is constructed correctly."""
    model = tmp_path / "trained_model"
    config = LongSplatConfig(
        source_path=str(tmp_path / "input"),
        model_path=str(model),
        iterations=100,
        seed=0,
    )

    fake_cp = mock.MagicMock(returncode=0, stderr="")
    with mock.patch("scripts.longsplat.convert.run_conversion", return_value=fake_cp):
        with mock.patch(
            "scripts.longsplat.convert._validate_converted_ply"
        ) as mock_val:
            mock_val.return_value = {
                "path": "...",
                "vertex_count": 1,
                "attributes": [],
                "sha256": "a" * 64,
                "file_size": 200,
            }
            result = convert_and_validate("/fake/repo", config)

    expected = model / "converted_3dgs" / "point_cloud.ply"
    assert result == expected
