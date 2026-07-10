"""
Shared pytest fixtures for video-to-exhibition-3dgs tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make scripts importable from the tests directory
_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_scripts_dir))


# ---------------------------------------------------------------------------
# General fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root():
    """Path to the project root (parent of tests/)."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_workdir(tmp_path):
    """Temporary working directory with data/ subdirectory structure."""
    (tmp_path / "data").mkdir(exist_ok=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Synthetic 3D point cloud data
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_xyz():
    """A small point cloud (N, 3)."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(100, 3)).astype(np.float32)


@pytest.fixture
def synthetic_features():
    """Feature dict matching synthetic_xyz."""
    rng = np.random.default_rng(42)
    return {
        "opacity": rng.uniform(0.01, 0.99, size=100).astype(np.float32),
        "scale_0": rng.uniform(0.1, 1.0, size=100).astype(np.float32),
        "scale_1": rng.uniform(0.1, 1.0, size=100).astype(np.float32),
        "scale_2": rng.uniform(0.1, 1.0, size=100).astype(np.float32),
        "rot_0": np.full(100, 1.0, dtype=np.float32),
        "rot_1": np.full(100, 0.0, dtype=np.float32),
        "rot_2": np.full(100, 0.0, dtype=np.float32),
        "rot_3": np.full(100, 0.0, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Synthetic PLY data (for _ply_transform tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_ply_vertex_count():
    return 10


@pytest.fixture
def synthetic_ply(synthetic_ply_vertex_count):
    """Create a minimal 3DGS-style PlyData for transform testing."""
    from plyfile import PlyData, PlyElement

    n = synthetic_ply_vertex_count
    rng = np.random.default_rng(42)

    dtype_spec = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]

    verts = np.empty(n, dtype=dtype_spec)
    verts["x"] = rng.normal(0, 2, n).astype(np.float32)
    verts["y"] = rng.normal(0, 2, n).astype(np.float32)
    verts["z"] = rng.normal(0, 2, n).astype(np.float32)
    verts["nx"] = rng.normal(0, 1, n).astype(np.float32)
    verts["ny"] = rng.normal(0, 1, n).astype(np.float32)
    verts["nz"] = rng.normal(0, 1, n).astype(np.float32)
    verts["f_dc_0"] = rng.normal(0, 1, n).astype(np.float32)
    verts["f_dc_1"] = rng.normal(0, 1, n).astype(np.float32)
    verts["f_dc_2"] = rng.normal(0, 1, n).astype(np.float32)
    verts["opacity"] = rng.uniform(0.01, 0.99, n).astype(np.float32)
    verts["scale_0"] = rng.uniform(0.1, 1.0, n).astype(np.float32)
    verts["scale_1"] = rng.uniform(0.1, 1.0, n).astype(np.float32)
    verts["scale_2"] = rng.uniform(0.1, 1.0, n).astype(np.float32)
    verts["rot_0"] = np.ones(n, dtype=np.float32)
    verts["rot_1"] = np.zeros(n, dtype=np.float32)
    verts["rot_2"] = np.zeros(n, dtype=np.float32)
    verts["rot_3"] = np.zeros(n, dtype=np.float32)

    el = PlyElement.describe(verts, "vertex")
    return PlyData([el], text=True)
