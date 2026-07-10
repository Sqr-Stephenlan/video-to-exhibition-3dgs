"""Tests for _ply_transform.py: Sim(3) PLY attribute transformation."""

from __future__ import annotations

import numpy as np
import pytest
from _ply_transform import (
    SchemaError,
    _classify_attr,
    _matrix_to_quat,
    _quat_compose,
    transform_ply_attributes,
)

# ---------------------------------------------------------------------------
# _classify_attr
# ---------------------------------------------------------------------------


def test_classify_pos():
    assert _classify_attr("x") == "pos"
    assert _classify_attr("y") == "pos"
    assert _classify_attr("z") == "pos"


def test_classify_normal():
    assert _classify_attr("nx") == "normal"
    assert _classify_attr("ny") == "normal"
    assert _classify_attr("nz") == "normal"


def test_classify_rot():
    assert _classify_attr("rot_0") == "rot"
    assert _classify_attr("rot_3") == "rot"


def test_classify_scale():
    assert _classify_attr("scale_0") == "scale"
    assert _classify_attr("scale_2") == "scale"


def test_classify_passthrough():
    assert _classify_attr("opacity") == "passthrough"
    assert _classify_attr("f_dc_0") == "passthrough"


# ---------------------------------------------------------------------------
# _quat_compose
# ---------------------------------------------------------------------------


def test_quat_compose_identity():
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    result = _quat_compose(q, q)
    np.testing.assert_allclose(result, q, atol=1e-6)


def test_quat_compose_batch():
    n = 10
    q1 = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32)
    q2 = np.tile([0.0, 1.0, 0.0, 0.0], (n, 1)).astype(np.float32)
    result = _quat_compose(q1, q2)
    np.testing.assert_allclose(result, q2, atol=1e-6)


# ---------------------------------------------------------------------------
# _matrix_to_quat
# ---------------------------------------------------------------------------


def test_matrix_to_quat_identity():
    r_mat = np.eye(3, dtype=np.float32)
    q = _matrix_to_quat(r_mat)
    # quaternion (w, x, y, z) for identity = (1, 0, 0, 0)
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-5)


def test_matrix_to_quat_pi_around_z():
    theta = np.pi
    r_mat = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    q = _matrix_to_quat(r_mat)
    # quaternion for 180° around z = (0, 0, 0, 1)
    np.testing.assert_allclose(np.abs(q[0]), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.abs(q[3]), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# transform_ply_attributes — identity
# ---------------------------------------------------------------------------


def test_transform_identity(synthetic_ply):
    result = transform_ply_attributes(
        synthetic_ply,
        scale=1.0,
        R=np.eye(3, dtype=np.float32),
        t=np.zeros(3, dtype=np.float32),
    )
    vert_in = synthetic_ply["vertex"]
    vert_out = result["vertex"]
    for attr in ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"):
        np.testing.assert_allclose(
            np.asarray(vert_out[attr]),
            np.asarray(vert_in[attr]),
            atol=1e-5,
            err_msg=f"Mismatch in {attr}",
        )


# ---------------------------------------------------------------------------
# transform_ply_attributes — translation only
# ---------------------------------------------------------------------------


def test_transform_translation(synthetic_ply):
    t = np.array([10.0, -5.0, 3.0], dtype=np.float32)
    result = transform_ply_attributes(
        synthetic_ply,
        scale=1.0,
        R=np.eye(3, dtype=np.float32),
        t=t,
    )
    vert_out = result["vertex"]
    xyz = np.stack([vert_out["x"], vert_out["y"], vert_out["z"]], axis=1)
    xyz_orig = np.stack(
        [synthetic_ply["vertex"]["x"], synthetic_ply["vertex"]["y"], synthetic_ply["vertex"]["z"]],
        axis=1,
    )
    np.testing.assert_allclose(xyz, xyz_orig + t, atol=1e-4)


# ---------------------------------------------------------------------------
# transform_ply_attributes — scale only
# ---------------------------------------------------------------------------


def test_transform_scale(synthetic_ply):
    scale = 2.0
    result = transform_ply_attributes(
        synthetic_ply,
        scale=scale,
        R=np.eye(3, dtype=np.float32),
        t=np.zeros(3, dtype=np.float32),
    )
    vert_out = result["vertex"]
    xyz = np.stack([vert_out["x"], vert_out["y"], vert_out["z"]], axis=1)
    xyz_orig = np.stack(
        [synthetic_ply["vertex"]["x"], synthetic_ply["vertex"]["y"], synthetic_ply["vertex"]["z"]],
        axis=1,
    )
    np.testing.assert_allclose(xyz, xyz_orig * scale, atol=1e-4)
    np.testing.assert_allclose(vert_out["scale_0"], synthetic_ply["vertex"]["scale_0"] * scale)


# ---------------------------------------------------------------------------
# transform_ply_attributes — full Sim(3)
# ---------------------------------------------------------------------------


def test_transform_full_sim3(synthetic_ply):
    scale = 3.0
    theta = np.pi / 4
    r_mat = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    t = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = transform_ply_attributes(synthetic_ply, scale, r_mat, t)

    # Check positions
    vert_out = result["vertex"]
    xyz = np.stack([vert_out["x"], vert_out["y"], vert_out["z"]], axis=1)
    xyz_orig = np.stack(
        [synthetic_ply["vertex"]["x"], synthetic_ply["vertex"]["y"], synthetic_ply["vertex"]["z"]],
        axis=1,
    )
    expected_xyz = (scale * (r_mat @ xyz_orig.T)).T + t
    np.testing.assert_allclose(xyz, expected_xyz, atol=1e-4)

    # Passthrough attributes preserved
    np.testing.assert_allclose(vert_out["opacity"], synthetic_ply["vertex"]["opacity"])
    np.testing.assert_allclose(vert_out["f_dc_0"], synthetic_ply["vertex"]["f_dc_0"])


# ---------------------------------------------------------------------------
# SchemaError on malformed rot attributes
# ---------------------------------------------------------------------------


def test_transform_bad_rot_schema():
    """PLY with only 3 rot attributes (instead of 4) should raise SchemaError."""
    from plyfile import PlyData, PlyElement

    dtype_spec = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),  # only 3
    ]
    n = 5
    verts = np.zeros(n, dtype=dtype_spec)
    verts["rot_0"] = np.ones(n, dtype=np.float32)
    verts["opacity"] = np.full(n, 0.5, dtype=np.float32)
    verts["scale_0"] = verts["scale_1"] = verts["scale_2"] = np.full(n, 0.1, dtype=np.float32)
    el = PlyElement.describe(verts, "vertex")
    ply = PlyData([el], text=True)

    with pytest.raises(SchemaError, match="Expected 4 rotation"):
        transform_ply_attributes(
            ply, 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        )
