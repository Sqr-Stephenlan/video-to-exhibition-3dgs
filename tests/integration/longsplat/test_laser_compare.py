"""Tests for Sim(3) laser alignment and voxel distance evaluation (Task 9)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Umeyama Sim(3) tests
# ---------------------------------------------------------------------------


def test_estimate_sim3_umeyama_recovers_known_transform():
    """Umeyama recovers a known scale/rotation/translation exactly."""
    from scripts.longsplat.laser_compare import estimate_sim3_umeyama

    rng = np.random.RandomState(42)
    source = rng.randn(50, 3).astype(np.float64)

    # Known transform
    scale = 0.75
    angle = np.deg2rad(30.0)
    R = np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1],
    ], dtype=np.float64)
    t = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    target = scale * (source @ R.T) + t

    s, R_est, t_est = estimate_sim3_umeyama(source, target)

    assert abs(s - scale) < 1e-10
    assert np.allclose(R_est @ R_est.T, np.eye(3), atol=1e-10)
    assert abs(np.linalg.det(R_est) - 1.0) < 1e-10
    assert np.allclose(scale * (source @ R.T) + t, s * (source @ R_est.T) + t_est, atol=1e-10)
    assert np.allclose(t_est, t, atol=1e-10)


def test_estimate_sim3_umeyama_identity():
    """Umeyama returns identity for identical point clouds."""
    from scripts.longsplat.laser_compare import estimate_sim3_umeyama

    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    s, R, t = estimate_sim3_umeyama(source, source.copy())

    assert abs(s - 1.0) < 1e-10
    assert np.allclose(R, np.eye(3), atol=1e-10)
    assert np.allclose(t, np.zeros(3), atol=1e-10)


def test_estimate_sim3_umeyama_refuses_fewer_than_4_pairs():
    """Umeyama raises ValueError with fewer than 4 point pairs."""
    from scripts.longsplat.laser_compare import estimate_sim3_umeyama

    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    target = source.copy()
    with pytest.raises(ValueError, match="at least 4"):
        estimate_sim3_umeyama(source, target)


def test_estimate_sim3_umeyama_refuses_coplanar():
    """Umeyama refuses purely coplanar points (degenerate)."""
    from scripts.longsplat.laser_compare import estimate_sim3_umeyama

    # All points in z=0 plane
    source = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    ], dtype=np.float64)
    target = source.copy() + np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="coplanar"):
        estimate_sim3_umeyama(source, target)


# ---------------------------------------------------------------------------
# Voxel neighbor distance tests
# ---------------------------------------------------------------------------


def test_voxel_neighbor_distances_self_is_zero():
    """Distance from a point cloud to itself is near zero."""
    from scripts.longsplat.laser_compare import voxel_neighbor_distances

    rng = np.random.RandomState(42)
    pts = rng.randn(500, 3).astype(np.float32)
    dists = voxel_neighbor_distances(pts, pts, voxel_size=0.02, max_distance=1.0)
    assert len(dists) == len(pts)
    assert dists.max() < 0.05  # within voxel grid resolution


def test_voxel_neighbor_distances_separated_clouds():
    """Points far apart produce large distances."""
    from scripts.longsplat.laser_compare import voxel_neighbor_distances

    source = np.array([[0, 0, 0], [0.01, 0, 0]], dtype=np.float32)
    target = np.array([[5, 0, 0], [5.01, 0, 0]], dtype=np.float32)
    dists = voxel_neighbor_distances(source, target, voxel_size=0.02, max_distance=10.0)
    assert np.all(dists > 4.0)


def test_voxel_neighbor_distances_respects_max_distance():
    """Distances beyond max_distance are capped."""
    from scripts.longsplat.laser_compare import voxel_neighbor_distances

    source = np.array([[0, 0, 0]], dtype=np.float32)
    target = np.array([[10, 0, 0]], dtype=np.float32)
    dists = voxel_neighbor_distances(source, target, voxel_size=0.02, max_distance=1.0)
    assert dists[0] == 1.0


def test_voxel_neighbor_distances_empty_source():
    """Empty source returns empty array."""
    from scripts.longsplat.laser_compare import voxel_neighbor_distances

    source = np.empty((0, 3), dtype=np.float32)
    target = np.array([[0, 0, 0]], dtype=np.float32)
    dists = voxel_neighbor_distances(source, target, voxel_size=0.02, max_distance=1.0)
    assert len(dists) == 0


# ---------------------------------------------------------------------------
# compare_registered_wall integration tests
# ---------------------------------------------------------------------------


def test_compare_registered_wall_basic(tmp_path):
    """Full pipeline on synthetic data produces expected metric keys."""
    from scripts.longsplat.laser_compare import compare_registered_wall

    rng = np.random.RandomState(42)

    # Create synthetic laser PLY (a flat wall with some features)
    n_laser = 1000
    laser_xyz = np.zeros((n_laser, 3), dtype=np.float32)
    laser_xyz[:, 0] = rng.uniform(-2, 2, n_laser)
    laser_xyz[:, 1] = rng.uniform(-1, 1, n_laser)
    laser_xyz[:, 2] = 0.02 * rng.randn(n_laser).astype(np.float32)
    # Add a protruding box feature
    box_mask = (np.abs(laser_xyz[:, 0]) < 0.3) & (np.abs(laser_xyz[:, 1]) < 0.3)
    laser_xyz[box_mask, 2] += 0.1

    _write_minimal_ply(tmp_path / "laser.ply", laser_xyz)

    # Create synthetic model PLY (same wall, slightly noisy, shifted)
    angle = np.deg2rad(5.0)
    R = np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1],
    ], dtype=np.float64)
    t = np.array([0.05, -0.03, 0.01], dtype=np.float64)
    model_xyz = (laser_xyz @ R.T + t).astype(np.float32)
    model_xyz += 0.005 * rng.randn(*model_xyz.shape).astype(np.float32)
    _write_minimal_ply(tmp_path / "model.ply", model_xyz)

    # Correspondences (4 non-coplanar semantic points)
    # Pick corners of the box feature
    correspondences = [
        {"label": "box_corner_1", "laser_xyz": laser_xyz[100].tolist(),
         "model_xyz": model_xyz[100].tolist()},
        {"label": "box_corner_2", "laser_xyz": laser_xyz[200].tolist(),
         "model_xyz": model_xyz[200].tolist()},
        {"label": "box_corner_3", "laser_xyz": laser_xyz[300].tolist(),
         "model_xyz": model_xyz[300].tolist()},
        {"label": "box_corner_4", "laser_xyz": laser_xyz[400].tolist(),
         "model_xyz": model_xyz[400].tolist()},
    ]
    corr_path = tmp_path / "corr.json"
    corr_path.write_text(json.dumps(correspondences))

    # ROI
    roi = {"x_min": -2.0, "x_max": 2.0, "y_min": -1.0, "y_max": 1.0,
           "z_min": -0.5, "z_max": 0.5, "label": "test_wall"}
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps(roi))

    result = compare_registered_wall(
        tmp_path / "laser.ply",
        tmp_path / "model.ply",
        corr_path,
        roi_path,
    )

    assert "accuracy_median" in result
    assert "completeness_median" in result
    assert "f_score_2cm" in result
    assert "f_score_5cm" in result
    assert "sim3_scale" in result
    assert result["accuracy_median"] >= 0
    assert result["completeness_median"] >= 0
    assert 0 <= result["f_score_2cm"] <= 1
    assert result["sim3_scale"] > 0


def test_compare_registered_wall_30pct_missing_reduces_completeness(tmp_path):
    """Deleting 30% of model points reduces completeness but not accuracy much."""
    from scripts.longsplat.laser_compare import compare_registered_wall

    rng = np.random.RandomState(42)
    n = 500
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = rng.uniform(-1, 1, n)
    pts[:, 1] = rng.uniform(-1, 1, n)
    pts[:, 2] = 0.01 * rng.randn(n).astype(np.float32)

    _write_minimal_ply(tmp_path / "laser.ply", pts)

    # Full model
    full_model = pts + 0.002 * rng.randn(*pts.shape).astype(np.float32)
    _write_minimal_ply(tmp_path / "model_full.ply", full_model)

    # Partial model (70% of points)
    keep = rng.choice(n, int(n * 0.7), replace=False)
    partial_model = full_model[keep]
    _write_minimal_ply(tmp_path / "model_partial.ply", partial_model)

    corr = [
        {"label": "p1", "laser_xyz": pts[10].tolist(), "model_xyz": full_model[10].tolist()},
        {"label": "p2", "laser_xyz": pts[50].tolist(), "model_xyz": full_model[50].tolist()},
        {"label": "p3", "laser_xyz": pts[100].tolist(), "model_xyz": full_model[100].tolist()},
        {"label": "p4", "laser_xyz": pts[200].tolist(), "model_xyz": full_model[200].tolist()},
    ]
    corr_path = tmp_path / "corr.json"
    corr_path.write_text(json.dumps(corr))
    roi = {"x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -0.5, "z_max": 0.5}
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps(roi))

    result_full = compare_registered_wall(
        tmp_path / "laser.ply", tmp_path / "model_full.ply", corr_path, roi_path,
    )
    result_partial = compare_registered_wall(
        tmp_path / "laser.ply", tmp_path / "model_partial.ply", corr_path, roi_path,
    )

    # Completeness should drop more than accuracy
    assert result_partial["completeness_median"] >= result_full["completeness_median"]
    # F-score should be lower for partial
    assert result_partial["f_score_2cm"] <= result_full["f_score_2cm"]


def test_compare_registered_wall_missing_correspondence_file(tmp_path):
    """Raises FileNotFoundError for missing correspondence JSON."""
    from scripts.longsplat.laser_compare import compare_registered_wall

    laser = tmp_path / "laser.ply"
    model = tmp_path / "model.ply"
    _write_minimal_ply(laser, np.zeros((1, 3), dtype=np.float32))
    _write_minimal_ply(model, np.zeros((1, 3), dtype=np.float32))
    roi_path = tmp_path / "roi.json"
    roi_path.write_text(json.dumps({"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1}))

    with pytest.raises(FileNotFoundError):
        compare_registered_wall(laser, model, tmp_path / "nonexistent.json", roi_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_ply(path: Path, xyz: np.ndarray) -> None:
    """Write a minimal 3DGS PLY with SH and scale fields."""
    from plyfile import PlyData, PlyElement

    n = len(xyz)
    dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("f_dc_0", np.float32), ("f_dc_1", np.float32), ("f_dc_2", np.float32),
        ("opacity", np.float32),
        ("scale_0", np.float32), ("scale_1", np.float32), ("scale_2", np.float32),
        ("rot_0", np.float32), ("rot_1", np.float32), ("rot_2", np.float32), ("rot_3", np.float32),
    ])
    data = np.zeros(n, dtype=dtype)
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["f_dc_0"] = 0.5
    data["f_dc_1"] = 0.3
    data["f_dc_2"] = 0.1
    data["opacity"] = 0.9
    data["scale_0"] = 0.01
    data["scale_1"] = 0.01
    data["scale_2"] = 0.01
    data["rot_0"] = 1.0
    el = PlyElement.describe(data, "vertex")
    PlyData([el]).write(str(path))
