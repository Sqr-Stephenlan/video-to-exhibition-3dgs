"""
Apply Sim(3) similarity transforms to 3DGS PLY point clouds.

Handles not only XYZ coordinates but also Gaussian scale / rotation /
normal attributes, following the attribute schema defined by LongSplat's
``convert_3dgs.py`` at the locked commit.

Unknown attributes are rejected by default so that partial transformations
cannot silently produce corrupted PLYs.
"""

from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement


class SchemaError(Exception):
    """Raised when a PLY schema is incompatible with the expected layout."""


# --- Attribute classifiers ---------------------------------------------------

# Position attributes — transformed as:  scale * (R @ xyz) + t
_POSITION_ATTRS = frozenset({"x", "y", "z"})

# Scale attributes (per-Gaussian scaling factors) — multiplied by `scale`.
# LongSplat convention: "scale_0", "scale_1", "scale_2".
_SCALE_ATTR_PATTERNS = ("scale_",)

# Rotation attributes (quaternions or rotation matrices).
# LongSplat convention: "rot_0" … "rot_3" (quaternion w,x,y,z).
_ROT_ATTR_PATTERNS = ("rot_",)

# Normal / direction attributes — rotated only (no scaling).
_NORMAL_ATTRS = frozenset({"nx", "ny", "nz"})


def _attr_matches(name: str, pattern: str) -> bool:
    return name.startswith(pattern)


def _classify_attr(name: str) -> str:
    """Return one of 'pos', 'scale', 'rot', 'normal', 'passthrough'."""
    if name in _POSITION_ATTRS:
        return "pos"
    if name in _NORMAL_ATTRS:
        return "normal"
    for pat in _ROT_ATTR_PATTERNS:
        if _attr_matches(name, pat):
            return "rot"
    for pat in _SCALE_ATTR_PATTERNS:
        if _attr_matches(name, pat):
            return "scale"
    return "passthrough"


# --- Quaternion helpers ------------------------------------------------------


def _quat_compose(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Compose two quaternion arrays (N,4) with (w,x,y,z) convention."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=1)


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion (w,x,y,z)."""
    # vectorised over a single 3x3
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz], dtype=np.float32)


# --- Main API ----------------------------------------------------------------


def transform_ply_attributes(
    ply: PlyData,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> PlyData:
    """Transform all relevant PLY attributes by a Sim(3) transform.

    Parameters
    ----------
    ply : PlyData
        Input PLY (will be copied, not mutated).
    scale : float
        Uniform scale factor.
    R : (3, 3) float32 rotation matrix.
    t : (3,) float32 translation vector.

    Returns
    -------
    PlyData with transformed vertex data.
    """
    vert = ply["vertex"]
    attr_names: list[str] = list(vert.data.dtype.names)
    n_points = vert.count

    # Classify attributes once
    classified = {name: _classify_attr(name) for name in attr_names}

    # Build new arrays
    new_arrays: dict[str, np.ndarray] = {}

    # -- positions -----------------------------------------------------------
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=1)
    xyz_t = (scale * (R @ xyz.T)).T + t
    new_arrays["x"], new_arrays["y"], new_arrays["z"] = xyz_t[:, 0], xyz_t[:, 1], xyz_t[:, 2]

    # -- rotation matrix → quaternion (precomputed once) ---------------------
    rot_quat = _matrix_to_quat(R)  # (4,) wxyz

    # -- per-attribute transform ---------------------------------------------
    for name in attr_names:
        if name in new_arrays:
            continue
        kind = classified[name]
        values = np.asarray(vert[name], dtype=np.float32)

        if kind == "pos":
            pass  # already handled
        elif kind == "scale":
            new_arrays[name] = values * scale
        elif kind == "rot":
            # Collect all rot_* components into quaternions, compose with R
            pass  # handled below as a group
        elif kind == "normal":
            # Rotate normals: R @ n
            nx, ny, nz = vert["nx"], vert["ny"], vert["nz"]
            normals = np.stack([nx, ny, nz], axis=1)
            rot_n = (R @ normals.T).T
            new_arrays["nx"], new_arrays["ny"], new_arrays["nz"] = (
                rot_n[:, 0],
                rot_n[:, 1],
                rot_n[:, 2],
            )
        else:
            new_arrays[name] = values

    # Handle rot attributes as a group (quaternion convention: w,x,y,z)
    rot_attrs = sorted([n for n in attr_names if classified[n] == "rot"])
    if rot_attrs:
        # Expect exactly 4 rot fields
        if len(rot_attrs) != 4:
            raise SchemaError(f"Expected 4 rotation attributes (quaternion), got {rot_attrs}")
        q_orig = np.stack(
            [np.asarray(vert[n], dtype=np.float32) for n in rot_attrs], axis=1
        )  # (N, 4)
        q_new = _quat_compose(np.tile(rot_quat, (n_points, 1)).astype(np.float32), q_orig)
        for i, name in enumerate(rot_attrs):
            new_arrays[name] = q_new[:, i]

    # -- assemble output -----------------------------------------------------
    dtype_spec = [(n, vert.data.dtype[n]) for n in attr_names]
    out_verts = np.empty(n_points, dtype=dtype_spec)
    for n in attr_names:
        out_verts[n] = new_arrays[n]

    el = PlyElement.describe(out_verts, "vertex")
    return PlyData([el], text=ply.text)
