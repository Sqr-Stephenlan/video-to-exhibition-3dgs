"""Laser-reference wall evaluation for LongSplat 3DGS models.

Pure NumPy implementation — no Open3D, scikit-learn, or other new
dependencies.  Provides Sim(3) alignment via Umeyama's method,
voxel-hash neighbour search, and registered wall comparisons.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_sim3_umeyama(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate Sim(3) via Umeyama's method (scale + rotation + translation).

    Parameters
    ----------
    source_points : (N, 3) float64
    target_points : (N, 3) float64

    Returns
    -------
    (scale, rotation, translation)
        scale : float
        rotation : (3, 3) ndarray  (proper SO(3))
        translation : (3,) ndarray
    """
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(target_points, dtype=np.float64)

    if src.shape != dst.shape:
        raise ValueError(f"shape mismatch: {src.shape} vs {dst.shape}")
    n = src.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 point pairs, got {n}")

    # Check for coplanarity
    src_centered = src - src.mean(axis=0)
    _, s, _ = np.linalg.svd(src_centered)
    if s[-1] < 1e-10 * s[0]:
        raise ValueError("source points are coplanar or degenerate")

    mu_s = src.mean(axis=0)
    mu_t = dst.mean(axis=0)

    src_c = src - mu_s
    dst_c = dst - mu_t

    var_s = np.sum(src_c ** 2) / n

    cross_cov = (dst_c.T @ src_c) / n

    u, d, vh = np.linalg.svd(cross_cov)

    # Handle reflection case
    det_sign = np.sign(np.linalg.det(u @ vh))
    s_mat = np.eye(3)
    s_mat[-1, -1] = det_sign

    R = u @ s_mat @ vh
    scale = np.trace(np.diag(d) @ s_mat) / var_s

    t = mu_t - scale * (R @ mu_s)

    return float(scale), R, t


def voxel_neighbor_distances(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    voxel_size: float,
    max_distance: float,
) -> np.ndarray:
    """Per-point nearest-neighbour distances using voxel hashing.

    Parameters
    ----------
    source_points : (M, 3) float32/float64
    target_points : (N, 3) float32/float64
    voxel_size : float
        Edge length of each voxel cell (scene units).
    max_distance : float
        Distances beyond this are clamped.

    Returns
    -------
    distances : (M,) float64
    """
    src = np.asarray(source_points)
    tgt = np.asarray(target_points)

    if len(src) == 0:
        return np.array([], dtype=np.float64)

    if len(tgt) == 0:
        return np.full(len(src), max_distance, dtype=np.float64)

    # Build voxel hash map for target points
    inv_voxel = 1.0 / max(voxel_size, 1e-12)
    tgt_coords = (tgt * inv_voxel).astype(np.int64)

    voxel_map: dict[tuple[int, int, int], list[int]] = {}
    for i, coord in enumerate(tgt_coords):
        key = (int(coord[0]), int(coord[1]), int(coord[2]))
        voxel_map.setdefault(key, []).append(i)

    distances = np.full(len(src), max_distance, dtype=np.float64)
    max_dist_sq = max_distance * max_distance

    for i in range(len(src)):
        vc = (src[i] * inv_voxel).astype(np.int64)
        best_sq = max_dist_sq

        # Search 27-neighbourhood
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (int(vc[0] + dx), int(vc[1] + dy), int(vc[2] + dz))
                    indices = voxel_map.get(key)
                    if indices is None:
                        continue
                    for j in indices:
                        sq = np.sum((src[i] - tgt[j]) ** 2)
                        if sq < best_sq:
                            best_sq = sq

        distances[i] = np.sqrt(best_sq)

    return np.minimum(distances, max_distance)


def compare_registered_wall(
    laser_ply: Path | str,
    model_ply: Path | str,
    correspondences_json: Path | str,
    roi_json: Path | str,
    *,
    dedup_voxel_size: float = 0.01,
    search_voxel_size: float = 0.02,
    max_distance: float = 1.0,
    plane_grid_size: int = 128,
) -> dict[str, Any]:
    """Compare a LongSplat model against laser reference in a common ROI.

    Parameters
    ----------
    laser_ply : Path
        Laser reference 3DGS PLY file.
    model_ply : Path
        LongSplat converted 3DGS PLY file.
    correspondences_json : Path
        JSON array of ``{"label", "laser_xyz", "model_xyz"}`` objects
        (at least 4 non-coplanar pairs in laser coordinates).
    roi_json : Path
        JSON dict with ``x_min, x_max, y_min, y_max, z_min, z_max`` in
        laser coordinates.
    dedup_voxel_size : float
        Voxel size for point deduplication (metres).
    search_voxel_size : float
        Voxel size for neighbour search (metres).
    max_distance : float
        Maximum distance clamp (metres).
    plane_grid_size : int
        Grid resolution for plane occupancy computation.

    Returns
    -------
    dict
        Metrics keyed by name, plus input SHA-256 hashes.
    """
    from plyfile import PlyData

    laser_p = Path(laser_ply)
    model_p = Path(model_ply)
    corr_p = Path(correspondences_json)

    if not corr_p.is_file():
        raise FileNotFoundError(f"correspondence file not found: {corr_p}")

    # --- Load correspondences ---
    with open(corr_p, encoding="utf-8") as fh:
        corr_data = json.load(fh)
    if not isinstance(corr_data, list) or len(corr_data) < 4:
        raise ValueError(f"need at least 4 correspondences, got {len(corr_data) if isinstance(corr_data, list) else 'non-list'}")

    laser_corr = np.array([c["laser_xyz"] for c in corr_data], dtype=np.float64)
    model_corr = np.array([c["model_xyz"] for c in corr_data], dtype=np.float64)

    # --- Load ROI ---
    with open(roi_json, encoding="utf-8") as fh:
        roi = json.load(fh)
    roi_min = np.array([roi["x_min"], roi["y_min"], roi["z_min"]], dtype=np.float64)
    roi_max = np.array([roi["x_max"], roi["y_max"], roi["z_max"]], dtype=np.float64)

    # --- Load point clouds ---
    laser_ply_data = PlyData.read(str(laser_p))
    laser_xyz = np.stack([
        laser_ply_data["vertex"]["x"],
        laser_ply_data["vertex"]["y"],
        laser_ply_data["vertex"]["z"],
    ], axis=-1).astype(np.float64)

    model_ply_data = PlyData.read(str(model_p))
    model_xyz = np.stack([
        model_ply_data["vertex"]["x"],
        model_ply_data["vertex"]["y"],
        model_ply_data["vertex"]["z"],
    ], axis=-1).astype(np.float64)

    # --- Estimate Sim(3) from correspondences ---
    scale, R, t = estimate_sim3_umeyama(model_corr, laser_corr)

    # --- Transform model to laser coordinates ---
    model_aligned = scale * (model_xyz @ R.T) + t

    # --- Crop to ROI (in laser coordinates) ---
    laser_in_roi = (
        (laser_xyz[:, 0] >= roi_min[0]) & (laser_xyz[:, 0] <= roi_max[0])
        & (laser_xyz[:, 1] >= roi_min[1]) & (laser_xyz[:, 1] <= roi_max[1])
        & (laser_xyz[:, 2] >= roi_min[2]) & (laser_xyz[:, 2] <= roi_max[2])
    )
    model_in_roi = (
        (model_aligned[:, 0] >= roi_min[0]) & (model_aligned[:, 0] <= roi_max[0])
        & (model_aligned[:, 1] >= roi_min[1]) & (model_aligned[:, 1] <= roi_max[1])
        & (model_aligned[:, 2] >= roi_min[2]) & (model_aligned[:, 2] <= roi_max[2])
    )

    laser_crop = laser_xyz[laser_in_roi].astype(np.float32)
    model_crop = model_aligned[model_in_roi].astype(np.float32)

    # --- Deduplicate (1 cm voxel) ---
    laser_crop = _dedup_voxel(laser_crop, dedup_voxel_size)
    model_crop = _dedup_voxel(model_crop, dedup_voxel_size)

    # --- Bidirectional distances ---
    accuracy = voxel_neighbor_distances(
        model_crop, laser_crop,
        voxel_size=search_voxel_size, max_distance=max_distance,
    )
    completeness = voxel_neighbor_distances(
        laser_crop, model_crop,
        voxel_size=search_voxel_size, max_distance=max_distance,
    )

    # --- F-scores ---
    f_2cm = _f_score(accuracy, completeness, threshold=0.02)
    f_5cm = _f_score(accuracy, completeness, threshold=0.05)

    # --- Plane metrics on laser ROI points ---
    plane_median, occupancy, largest_comp = _plane_metrics(
        model_crop.astype(np.float64), grid_size=plane_grid_size,
    )

    # --- SHA-256 ---
    hashes = {
        "laser_sha256": _sha256_hex(laser_p),
        "model_sha256": _sha256_hex(model_p),
        "correspondences_sha256": _sha256_hex(corr_p),
        "roi_sha256": _sha256_hex(roi_json),
    }

    result: dict[str, Any] = {
        "laser_point_count": len(laser_crop),
        "model_point_count": len(model_crop),
        "accuracy_median": float(np.median(accuracy)) if len(accuracy) > 0 else None,
        "accuracy_p95": float(np.percentile(accuracy, 95)) if len(accuracy) > 0 else None,
        "completeness_median": float(np.median(completeness)) if len(completeness) > 0 else None,
        "completeness_p95": float(np.percentile(completeness, 95)) if len(completeness) > 0 else None,
        "f_score_2cm": f_2cm,
        "f_score_5cm": f_5cm,
        "plane_median_residual_diag_fraction": plane_median,
        "plane_grid_occupancy": occupancy,
        "plane_largest_component_fraction": largest_comp,
        "sim3_scale": scale,
        "sim3_rotation": R.tolist(),
        "sim3_translation": t.tolist(),
        **hashes,
        "parameters": {
            "dedup_voxel_size": dedup_voxel_size,
            "search_voxel_size": search_voxel_size,
            "max_distance": max_distance,
            "plane_grid_size": plane_grid_size,
        },
        "metric_schema": "longsplat-laser-wall-v1",
    }
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dedup_voxel(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Deduplicate points by keeping the first point in each voxel cell."""
    if len(xyz) == 0:
        return xyz
    inv = 1.0 / max(voxel_size, 1e-12)
    coords = (xyz * inv).astype(np.int64)
    seen: dict[tuple[int, int, int], int] = {}
    keep: list[int] = []
    for i in range(len(xyz)):
        key = (int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]))
        if key not in seen:
            seen[key] = i
            keep.append(i)
    return xyz[np.array(keep)]


def _f_score(
    accuracy: np.ndarray,
    completeness: np.ndarray,
    *,
    threshold: float,
) -> float:
    """Compute F-score at a given distance threshold."""
    prec = (accuracy <= threshold).mean()
    rec = (completeness <= threshold).mean()
    if prec + rec == 0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def _plane_metrics(
    xyz: np.ndarray,
    *,
    grid_size: int = 128,
) -> tuple[float | None, float | None, float | None]:
    """PCA-based plane metrics for effective wall points.

    Returns (median_residual_diag_fraction, grid_occupancy, largest_component_fraction).
    Each is None when fewer than 4 effective points are available.
    """
    n = xyz.shape[0]
    if n < 4:
        return None, None, None

    centroid = xyz.mean(axis=0)
    centered = xyz - centroid
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]

    residuals = np.abs(centered @ normal)
    median_residual = float(np.median(residuals))

    aabb_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    if aabb_diag < 1e-12:
        return 0.0, 0.0, 0.0

    diag_fraction = median_residual / aabb_diag

    # Project to first two principal axes
    proj_2d = centered @ vh[:2].T
    p_min = proj_2d.min(axis=0)
    p_max = proj_2d.max(axis=0)
    span = p_max - p_min
    if np.any(span < 1e-12):
        return diag_fraction, 0.0, 0.0

    grid_coords = ((proj_2d - p_min) / span * (grid_size - 1)).astype(int)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    grid[grid_coords[:, 0], grid_coords[:, 1]] = True
    occupancy = float(grid.mean())

    largest_comp = _largest_connected_component_fraction(grid)

    return diag_fraction, occupancy, largest_comp


def _largest_connected_component_fraction(grid: np.ndarray) -> float:
    """Fraction of occupied cells in largest 4-connected component."""
    occupied = grid.sum()
    if occupied == 0:
        return 0.0
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    max_size = 0
    for r in range(rows):
        for c in range(cols):
            if not grid[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                for nr, nc in [(cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)]:
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            max_size = max(max_size, size)
    return float(max_size) / float(occupied)


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
