"""RED tests for LongSplat quality metrics — Task 1 Step 1.

All tests MUST fail before the implementation modules exist.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ply_text(path: Path, vertex_data: list[dict]) -> None:
    """Write a text-format PLY with the given vertex dicts.

    All dicts must share the same keys; dtype is f4 for everything.
    """
    fields = list(vertex_data[0].keys())
    dtype = [(f, "f4") for f in fields]
    verts = np.zeros(len(vertex_data), dtype=dtype)
    for i, row in enumerate(vertex_data):
        for f in fields:
            verts[i][f] = row[f]
    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=True).write(str(path))


def _make_ply_binary(path: Path, vertex_data: list[dict]) -> None:
    """Write a binary-format PLY."""
    fields = list(vertex_data[0].keys())
    dtype = [(f, "f4") for f in fields]
    verts = np.zeros(len(vertex_data), dtype=dtype)
    for i, row in enumerate(vertex_data):
        for f in fields:
            verts[i][f] = row[f]
    el = PlyElement.describe(verts, "vertex")
    PlyData([el]).write(str(path))


def _make_cameras_json(path: Path, cameras: list[dict]) -> None:
    """Write a minimal cameras JSON array."""
    path.write_text(json.dumps(cameras), encoding="utf-8")


def _rotation_matrix(axis_deg: float, axis: str = "y") -> list[list[float]]:
    """Simple axis-aligned rotation matrix."""
    r = math.radians(axis_deg)
    c, s = math.cos(r), math.sin(r)
    if axis == "y":
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    if axis == "z":
        return [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


# ---------------------------------------------------------------------------
# Test 1: sigmoid opacity and log-scale anisotropy computation
# ---------------------------------------------------------------------------


def test_sigmoid_opacity_on_small_ply(tmp_path):
    """Sigmoid opacity and log-scale anisotropy for a clean 3DGS PLY."""
    from scripts.longsplat.quality_metrics import analyze_ply_quality

    # opacity logit: sigmoid(0)=0.5, sigmoid(2)≈0.8808, sigmoid(-1)≈0.2689
    ply_path = tmp_path / "clean.ply"
    _make_ply_text(
        ply_path,
        [
            {
                "x": 0, "y": 0, "z": 0,
                "f_dc_0": 0.5, "f_dc_1": 0.3, "f_dc_2": 0.2,
                "opacity": 0.0,  # sigmoid(0) = 0.5
                "scale_0": 0.0, "scale_1": 0.0, "scale_2": 0.0,  # exp(0)=1, iso
                "rot_0": 1.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0,
            },
            {
                "x": 0, "y": 0, "z": 0,
                "f_dc_0": 0.5, "f_dc_1": 0.3, "f_dc_2": 0.2,
                "opacity": 2.0,  # sigmoid(2) ≈ 0.8808
                "scale_0": math.log(3), "scale_1": math.log(1), "scale_2": math.log(1),
                # anisotropy = exp(log(3)-log(1)) = 3
                "rot_0": 1.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0,
            },
            {
                "x": 0, "y": 0, "z": 0,
                "f_dc_0": 0.5, "f_dc_1": 0.3, "f_dc_2": 0.2,
                "opacity": -1.0,  # sigmoid(-1) ≈ 0.2689
                "scale_0": math.log(10), "scale_1": math.log(1), "scale_2": math.log(1),
                # anisotropy = 10
                "rot_0": 1.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0,
            },
        ],
    )

    result = analyze_ply_quality(ply_path)

    assert result["vertex_count"] == 3
    assert result["finite_core_fraction"] == 1.0
    # effective = all finite + activated opacity > 0.1
    # sigmoid(0)=0.5 > 0.1, sigmoid(2)≈0.8808 > 0.1, sigmoid(-1)≈0.2689 > 0.1
    # all three are effective
    assert result["effective_fraction"] == 1.0
    assert "opacity_q50" in result
    assert "anisotropy_q50" in result
    assert "anisotropy_q99" in result
    # q50 should be iso=1 (two of three values are 1.0)
    # median of [1.0, 3.0, 10.0]
    assert result["anisotropy_q50"] == pytest.approx(3.0, abs=0.1)
    assert result["anisotropy_q99"] >= 9.0
    assert result["quaternion_within_1pct_fraction"] == 1.0
    assert result["metric_schema"] == "longsplat-quality-v1"


# ---------------------------------------------------------------------------
# Test 2: fixture with -inf scale, NaN SH, zero quaternion
# ---------------------------------------------------------------------------


def test_degenerate_ply_reports_fractions_not_cleaning(tmp_path):
    """Ratios are reported; the function must not modify the file."""
    from scripts.longsplat.quality_metrics import analyze_ply_quality

    ply_path = tmp_path / "degenerate.ply"
    _make_ply_text(
        ply_path,
        [
            # Good vertex
            {
                "x": 0, "y": 0, "z": 0,
                "f_dc_0": 0.5, "f_dc_1": 0.3, "f_dc_2": 0.2,
                "opacity": 0.0,
                "scale_0": 0.0, "scale_1": 0.0, "scale_2": 0.0,
                "rot_0": 1.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0,
            },
            # -inf scale
            {
                "x": 0, "y": 0, "z": 0,
                "f_dc_0": 0.5, "f_dc_1": 0.3, "f_dc_2": 0.2,
                "opacity": 0.0,
                "scale_0": float("-inf"), "scale_1": 0.0, "scale_2": 0.0,
                "rot_0": 1.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0,
            },
            # NaN SH
            {
                "x": 0, "y": 0, "z": 0,
                "f_dc_0": float("nan"), "f_dc_1": 0.3, "f_dc_2": 0.2,
                "opacity": 0.0,
                "scale_0": 0.0, "scale_1": 0.0, "scale_2": 0.0,
                "rot_0": 1.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0,
            },
            # Zero quaternion
            {
                "x": 0, "y": 0, "z": 0,
                "f_dc_0": 0.5, "f_dc_1": 0.3, "f_dc_2": 0.2,
                "opacity": 0.0,
                "scale_0": 0.0, "scale_1": 0.0, "scale_2": 0.0,
                "rot_0": 0.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0,
            },
        ],
    )

    # Snapshot original content
    original = ply_path.read_bytes()

    result = analyze_ply_quality(ply_path)

    # File untouched
    assert ply_path.read_bytes() == original

    assert result["vertex_count"] == 4
    # Vertices 0 and 3 have all core fields finite;
    # vertex 1 has -inf scale, vertex 2 has NaN SH.
    assert result["finite_core_fraction"] == 0.5
    # Only vertices 0 and 3 are effective (sigmoid(0)=0.5 > 0.1)
    assert result["effective_fraction"] == 0.5
    # Vertices 0,1,2 have unit quaternion; vertex 3 has zero quaternion
    assert result["quaternion_within_1pct_fraction"] == 0.75
    # Non-finite rows excluded from anisotropy quantiles
    assert "anisotropy_q50" in result


# ---------------------------------------------------------------------------
# Test 3: known plane with outliers, repeatable occupancy and residual
# ---------------------------------------------------------------------------


def test_plane_metrics_repeatable(tmp_path):
    """PCA plane residual and occupancy on a known plane with outliers."""
    from scripts.longsplat.quality_metrics import analyze_ply_quality

    rng = np.random.RandomState(42)

    n_plane = 5000
    n_outlier = 50
    xs = rng.uniform(-1, 1, n_plane)
    ys = rng.uniform(-1, 1, n_plane)
    zs = np.zeros(n_plane) + rng.normal(0, 0.001, n_plane)  # near-flat

    ox = rng.uniform(-1, 1, n_outlier)
    oy = rng.uniform(-1, 1, n_outlier)
    oz = rng.normal(0, 0.5, n_outlier)  # far from plane

    all_x = np.concatenate([xs, ox])
    all_y = np.concatenate([ys, oy])
    all_z = np.concatenate([zs, oz])

    n = n_plane + n_outlier
    vertex_data = []
    for i in range(n):
        vertex_data.append(
            {
                "x": float(all_x[i]),
                "y": float(all_y[i]),
                "z": float(all_z[i]),
                "f_dc_0": 0.5, "f_dc_1": 0.3, "f_dc_2": 0.2,
                "opacity": 1.0,  # high enough to be effective
                "scale_0": 0.0, "scale_1": 0.0, "scale_2": 0.0,
                "rot_0": 1.0, "rot_1": 0.0, "rot_2": 0.0, "rot_3": 0.0,
            }
        )

    ply_path = tmp_path / "plane.ply"
    _make_ply_text(ply_path, vertex_data)

    r1 = analyze_ply_quality(ply_path, plane_grid_size=32)
    r2 = analyze_ply_quality(ply_path, plane_grid_size=32)

    assert r1["plane_median_residual_diag_fraction"] is not None
    assert r1["plane_grid_occupancy"] is not None
    assert r1["plane_largest_component_fraction"] is not None

    # Repeatable
    assert r1["plane_median_residual_diag_fraction"] == pytest.approx(
        r2["plane_median_residual_diag_fraction"]
    )
    assert r1["plane_grid_occupancy"] == pytest.approx(r2["plane_grid_occupancy"])

    # Plane with small z-variance → residual should be small relative to diagonal
    assert r1["plane_median_residual_diag_fraction"] < 0.01
    # Most cells occupied
    assert r1["plane_grid_occupancy"] > 0.5


# ---------------------------------------------------------------------------
# Test 4: camera trajectory — legal rotations, non-orthogonal, 90° jump
# ---------------------------------------------------------------------------


def test_camera_trajectory_legal_rotations(tmp_path):
    """Two legal SO(3) cameras produce clean metrics."""
    from scripts.longsplat.quality_metrics import analyze_camera_trajectory

    cam_path = tmp_path / "cameras.json"
    R0 = _rotation_matrix(0, "y")
    R1 = _rotation_matrix(15, "y")  # 15° step
    _make_cameras_json(
        cam_path,
        [
            {"id": 0, "position": [0, 0, 0], "rotation": R0},
            {"id": 1, "position": [0, 0, 0.5], "rotation": R1},
        ],
    )

    result = analyze_camera_trajectory(cam_path)

    assert result["camera_count"] == 2
    assert result["max_rotation_step_deg"] is not None
    assert result["max_rotation_step_deg"] == pytest.approx(15.0, abs=1.0)
    assert result["max_orthogonality_error"] < 1e-4
    assert result["max_determinant_error"] < 1e-4
    assert result["all_rotations_finite"] is True


def test_camera_trajectory_non_orthogonal(tmp_path):
    """A non-orthogonal rotation is detected."""
    from scripts.longsplat.quality_metrics import analyze_camera_trajectory

    cam_path = tmp_path / "cameras_nonortho.json"
    # Non-orthogonal: scale first row
    bad_R = [[2.0, 0, 0], [0, 1, 0], [0, 0, 1]]
    _make_cameras_json(
        cam_path,
        [
            {"id": 0, "position": [0, 0, 0], "rotation": bad_R},
        ],
    )

    result = analyze_camera_trajectory(cam_path)
    assert result["max_orthogonality_error"] > 0.1
    assert result["all_rotations_finite"] is True


def test_camera_trajectory_90deg_jump(tmp_path):
    """A 90° rotation step is captured."""
    from scripts.longsplat.quality_metrics import analyze_camera_trajectory

    cam_path = tmp_path / "cameras_jump.json"
    R0 = _rotation_matrix(0, "y")
    R1 = _rotation_matrix(90, "y")
    _make_cameras_json(
        cam_path,
        [
            {"id": 0, "position": [0, 0, 0], "rotation": R0},
            {"id": 1, "position": [0, 0, 1], "rotation": R1},
        ],
    )

    result = analyze_camera_trajectory(cam_path)
    assert result["max_rotation_step_deg"] == pytest.approx(90.0, abs=2.0)


def test_camera_trajectory_with_timestamps(tmp_path):
    """Camera trajectory with explicit timestamps."""
    from scripts.longsplat.quality_metrics import analyze_camera_trajectory

    cam_path = tmp_path / "cameras_ts.json"
    R0 = _rotation_matrix(0, "y")
    R1 = _rotation_matrix(10, "y")
    R2 = _rotation_matrix(30, "y")
    _make_cameras_json(
        cam_path,
        [
            {"id": 0, "position": [0, 0, 0], "rotation": R0},
            {"id": 1, "position": [0, 0, 1], "rotation": R1},
            {"id": 2, "position": [0, 0, 2], "rotation": R2},
        ],
    )

    result = analyze_camera_trajectory(cam_path, timestamps=[0.0, 1.0, 2.0])
    assert result["camera_count"] == 3
    assert result["max_rotation_step_deg"] == pytest.approx(20.0, abs=2.0)


# ---------------------------------------------------------------------------
# Test 5: legacy loss log q95/q99/gt_10_ratio
# ---------------------------------------------------------------------------


def test_loss_log_legacy_text(tmp_path):
    """Parse legacy 'Loss=' lines for q95, q99, and >10 ratio."""
    from scripts.longsplat.quality_metrics import analyze_loss_log

    log_path = tmp_path / "train.log"
    losses = [0.5, 0.8, 1.2, 3.0, 5.0, 10.0, 12.0, 15.0, 25.0, 68.923]
    lines = []
    for v in losses:
        lines.append(f"[22/07 16:02:03] Training progress: Loss={v:.4f}")
    log_path.write_text("\n".join(lines), encoding="utf-8")

    result = analyze_loss_log(log_path)

    assert result["sample_count"] == 10
    assert result["source"] == "legacy_text"
    # q95 interpolates between values at indices 8 (25.0) and 9 (68.923)
    assert result["loss_q95"] == pytest.approx(49.16, abs=0.5)
    # q99 interpolates to ~64.97 with 10 samples
    assert result["loss_q99"] >= 60.0
    # >10: values 12, 15, 25, 68.923 → 4/10 = 0.4
    assert result["loss_gt_10_ratio"] == pytest.approx(0.4, abs=0.01)
    assert result["loss_max"] == pytest.approx(68.923)


def test_loss_log_large_values(tmp_path):
    """Loss > 10 ratio captures tail correctly."""
    from scripts.longsplat.quality_metrics import analyze_loss_log

    log_path = tmp_path / "train2.log"
    losses = [0.1] * 90 + [15.0] * 5 + [30.0] * 5
    lines = [f"Loss={v:.4f}" for v in losses]
    log_path.write_text("\n".join(lines), encoding="utf-8")

    result = analyze_loss_log(log_path)
    assert result["sample_count"] == 100
    assert result["loss_gt_10_ratio"] == pytest.approx(0.10, abs=0.01)
