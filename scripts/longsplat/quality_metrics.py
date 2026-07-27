"""LongSplat quality metrics for PLY, camera trajectory, and loss logs.

All computations are deterministic and dependency-free beyond numpy and
plyfile.  No file modification — analysis is strictly read-only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_CORE_FIELDS = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]

_SCALE_FIELDS = ["scale_0", "scale_1", "scale_2"]
_ROT_FIELDS = ["rot_0", "rot_1", "rot_2", "rot_3"]

_LEGACY_LOSS_RE = re.compile(r"Loss=([\d.]+(?:e[+-]?\d+)?)")


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlyQualityMetrics:
    vertex_count: int
    finite_core_fraction: float
    effective_fraction: float
    opacity_q50: float
    opacity_gt_01_fraction: float
    opacity_gt_05_fraction: float
    anisotropy_q50: float
    anisotropy_q99: float
    quaternion_within_1pct_fraction: float
    plane_median_residual_diag_fraction: float | None
    plane_grid_occupancy: float | None
    plane_largest_component_fraction: float | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_ply_quality(
    path: Path | str,
    *,
    opacity_threshold: float = 0.1,
    plane_grid_size: int = 128,
) -> dict[str, Any]:
    """Analyze a 3DGS PLY file and return quality metrics.

    Parameters
    ----------
    path : Path
        Path to the PLY file.
    opacity_threshold : float
        Minimum activated (sigmoid) opacity for a vertex to be "effective".
    plane_grid_size : int
        Grid resolution for plane occupancy computation.

    Returns
    -------
    dict
        Quality metrics dict keyed by metric name, plus ``metric_schema``,
        ``sha256``, and ``opacity_threshold``.
    """
    from plyfile import PlyData

    pp = Path(path).resolve()
    sha = _sha256_hex(pp)
    ply = PlyData.read(str(pp))
    vertex = ply["vertex"]
    n = vertex.count

    attr_names = set(vertex.data.dtype.names)
    core_present = [f for f in _CORE_FIELDS if f in attr_names]

    # --- core finite mask ---
    core_mask = np.ones(n, dtype=bool)
    for f in core_present:
        core_mask &= np.isfinite(vertex.data[f])

    finite_core_fraction = float(core_mask.sum()) / n if n > 0 else 0.0

    # --- activated opacity (sigmoid of logit) ---
    if "opacity" in attr_names:
        raw_opacity = vertex.data["opacity"]
        activated = _stable_sigmoid(raw_opacity)
    else:
        activated = np.zeros(n, dtype=np.float64)

    # --- effective mask: core finite + activated opacity > threshold ---
    effective_mask = core_mask & (activated > opacity_threshold)
    effective_fraction = float(effective_mask.sum()) / n if n > 0 else 0.0

    # --- opacity metrics on finite core ---
    finite_activated = activated[core_mask]
    if len(finite_activated) > 0:
        opacity_q50 = float(np.percentile(finite_activated, 50))
        opacity_gt_01_fraction = float((finite_activated > 0.1).mean())
        opacity_gt_05_fraction = float((finite_activated > 0.5).mean())
    else:
        opacity_q50 = 0.0
        opacity_gt_01_fraction = 0.0
        opacity_gt_05_fraction = 0.0

    # --- anisotropy from log-scale ---
    scales_present = [f for f in _SCALE_FIELDS if f in attr_names]
    if len(scales_present) == 3 and core_mask.sum() > 0:
        log_scales = np.stack(
            [vertex.data[f][core_mask] for f in _SCALE_FIELDS], axis=-1
        )
        max_log = log_scales.max(axis=-1)
        min_log = log_scales.min(axis=-1)
        log_ratio = np.clip(max_log - min_log, 0.0, 80.0)
        anisotropy = np.exp(log_ratio)
        finite_aniso = anisotropy[np.isfinite(anisotropy)]
        if len(finite_aniso) > 0:
            anisotropy_q50 = float(np.percentile(finite_aniso, 50))
            anisotropy_q99 = float(np.percentile(finite_aniso, 99))
        else:
            anisotropy_q50 = 0.0
            anisotropy_q99 = 0.0
    else:
        anisotropy_q50 = 0.0
        anisotropy_q99 = 0.0

    # --- quaternion quality ---
    rots_present = [f for f in _ROT_FIELDS if f in attr_names]
    if len(rots_present) == 4:
        q = np.stack([vertex.data[f] for f in _ROT_FIELDS], axis=-1)
        q_norms = np.linalg.norm(q, axis=-1)
        # within 1% of unit norm
        within_1pct = np.abs(q_norms - 1.0) <= 0.01
        quaternion_within_1pct_fraction = float(within_1pct.mean()) if n > 0 else 0.0
    else:
        quaternion_within_1pct_fraction = 0.0

    # --- plane metrics on effective points ---
    plane_median_residual = None
    plane_occupancy = None
    plane_largest_comp = None

    if effective_mask.sum() >= 4:
        xyz = np.stack(
            [vertex.data["x"], vertex.data["y"], vertex.data["z"]], axis=-1
        )
        effective_xyz = xyz[effective_mask]
        plane_median_residual, plane_occupancy, plane_largest_comp = (
            _plane_metrics(effective_xyz, grid_size=plane_grid_size)
        )

    result: dict[str, Any] = {
        "vertex_count": n,
        "finite_core_fraction": finite_core_fraction,
        "effective_fraction": effective_fraction,
        "opacity_q50": opacity_q50,
        "opacity_gt_01_fraction": opacity_gt_01_fraction,
        "opacity_gt_05_fraction": opacity_gt_05_fraction,
        "anisotropy_q50": anisotropy_q50,
        "anisotropy_q99": anisotropy_q99,
        "quaternion_within_1pct_fraction": quaternion_within_1pct_fraction,
        "plane_median_residual_diag_fraction": plane_median_residual,
        "plane_grid_occupancy": plane_occupancy,
        "plane_largest_component_fraction": plane_largest_comp,
        "sha256": sha,
        "opacity_threshold": opacity_threshold,
        "metric_schema": "longsplat-quality-v1",
    }
    return result


def analyze_camera_trajectory(
    cameras_path: Path | str,
    *,
    timestamps: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Analyze camera trajectory for SO(3) validity and inter-frame jumps.

    Parameters
    ----------
    cameras_path : Path
        Path to a JSON array of camera dicts with ``position`` (3,) and
        ``rotation`` (3×3 list-of-lists).
    timestamps : optional
        Per-camera timestamps; if absent, consecutive cameras are assumed
        to be temporally ordered.

    Returns
    -------
    dict
        Metrics including orthogonality error, determinant error, rotation
        and translation step sizes.
    """
    pp = Path(cameras_path)
    with open(pp, encoding="utf-8") as fh:
        cameras = json.load(fh)

    n = len(cameras)
    if n == 0:
        return {
            "camera_count": 0,
            "max_orthogonality_error": None,
            "max_determinant_error": None,
            "max_rotation_step_deg": None,
            "max_translation_step": None,
            "all_rotations_finite": True,
            "all_positions_finite": True,
        }

    rotations = []
    positions = []
    all_rot_finite = True
    all_pos_finite = True

    for cam in cameras:
        R = np.array(cam.get("R", cam.get("rotation")), dtype=np.float64)
        t = np.array(cam.get("T", cam.get("position")), dtype=np.float64)

        all_rot_finite = all_rot_finite and bool(np.all(np.isfinite(R)))
        all_pos_finite = all_pos_finite and bool(np.all(np.isfinite(t)))
        rotations.append(R)
        positions.append(t)

    # SO(3) errors
    max_ortho = 0.0
    max_det_err = 0.0
    for R in rotations:
        if not np.all(np.isfinite(R)):
            continue
        # R @ R^T should equal I
        ortho_err = np.max(np.abs(R @ R.T - np.eye(3)))
        max_ortho = max(max_ortho, ortho_err)
        det_err = abs(np.linalg.det(R) - 1.0)
        max_det_err = max(max_det_err, det_err)

    # Inter-frame rotation steps (in degrees)
    max_rot_step = 0.0
    max_trans_step = 0.0
    for i in range(1, n):
        if not np.all(np.isfinite(rotations[i])) or not np.all(
            np.isfinite(rotations[i - 1])
        ):
            continue
        R_rel = rotations[i] @ rotations[i - 1].T
        # angle = arccos((trace(R)-1)/2), clamped for numerical safety
        trace_val = np.trace(R_rel)
        cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))
        max_rot_step = max(max_rot_step, angle_deg)

        trans_step = float(np.linalg.norm(positions[i] - positions[i - 1]))
        max_trans_step = max(max_trans_step, trans_step)

    return {
        "camera_count": n,
        "max_orthogonality_error": float(max_ortho),
        "max_determinant_error": float(max_det_err),
        "max_rotation_step_deg": float(max_rot_step) if max_rot_step > 0 else 0.0,
        "max_translation_step": float(max_trans_step),
        "all_rotations_finite": all_rot_finite,
        "all_positions_finite": all_pos_finite,
    }


def analyze_loss_log(path: Path | str) -> dict[str, Any]:
    """Parse a LongSplat training log for loss statistics.

    Prefers ``LOSS_TELEMETRY`` JSON markers when present; falls back to
    legacy ``Loss=<float>`` text lines.

    Returns
    -------
    dict
        ``sample_count``, ``loss_q50``, ``loss_q95``, ``loss_q99``,
        ``loss_max``, ``loss_gt_10_ratio``, ``source``.
    """
    pp = Path(path)
    text = pp.read_text(encoding="utf-8")

    # Try new LOSS_TELEMETRY first
    losses: list[float] = []
    source = "legacy_text"

    for line in text.splitlines():
        if "LOSS_TELEMETRY" in line:
            # Parse JSON marker
            try:
                idx = line.index("LOSS_TELEMETRY") + len("LOSS_TELEMETRY")
                payload = line[idx:].strip()
                # Strip trailing timestamp suffix
                payload = _strip_timestamp_suffix(payload)
                record = json.loads(payload)
                if isinstance(record, dict) and "total" in record:
                    val = float(record["total"])
                    if math.isfinite(val):
                        losses.append(val)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                pass

    if losses:
        source = "loss_telemetry"
    else:
        # Legacy "Loss=" pattern
        for match in _LEGACY_LOSS_RE.finditer(text):
            try:
                val = float(match.group(1))
                if math.isfinite(val):
                    losses.append(val)
            except ValueError:
                pass

    if not losses:
        return {
            "sample_count": 0,
            "loss_q50": None,
            "loss_q95": None,
            "loss_q99": None,
            "loss_max": None,
            "loss_gt_10_ratio": None,
            "source": "legacy_text",
        }

    arr = np.array(losses, dtype=np.float64)
    return {
        "sample_count": len(arr),
        "loss_q50": float(np.percentile(arr, 50)),
        "loss_q95": float(np.percentile(arr, 95)),
        "loss_q99": float(np.percentile(arr, 99)),
        "loss_max": float(arr.max()),
        "loss_gt_10_ratio": float((arr > 10.0).mean()),
        "source": source,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: :math:`1 / (1 + exp(-clip(x)))`."""
    clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _plane_metrics(
    xyz: np.ndarray,
    *,
    grid_size: int = 128,
) -> tuple[float, float, float]:
    """PCA-based plane metrics.

    Returns
    -------
    (median_residual_diag_fraction, grid_occupancy, largest_component_fraction)
    """
    # Center points
    centroid = xyz.mean(axis=0)
    centered = xyz - centroid

    # PCA via SVD on centered points
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    # vh rows are principal directions; last row = min variance = plane normal
    normal = vh[-1]  # shape (3,)

    # Residual: distance from each point to the plane
    residuals = np.abs(centered @ normal)
    median_residual = float(np.median(residuals))

    # AABB diagonal of effective points
    aabb_min = xyz.min(axis=0)
    aabb_max = xyz.max(axis=0)
    diagonal = float(np.linalg.norm(aabb_max - aabb_min))
    if diagonal < 1e-12:
        return 0.0, 0.0, 0.0

    diag_fraction = median_residual / diagonal

    # Project points onto first two principal axes
    proj_2d = centered @ vh[:2].T  # shape (n, 2)

    # Map to grid coordinates
    p_min = proj_2d.min(axis=0)
    p_max = proj_2d.max(axis=0)
    span = p_max - p_min
    if np.any(span < 1e-12):
        return diag_fraction, 0.0, 0.0

    grid_coords = ((proj_2d - p_min) / span * (grid_size - 1)).astype(int)
    grid_coords = np.clip(grid_coords, 0, grid_size - 1)

    # Occupancy grid
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    grid[grid_coords[:, 0], grid_coords[:, 1]] = True
    occupancy = float(grid.mean())

    # Largest connected component in the grid
    largest_comp_fraction = _largest_connected_component_fraction(grid)

    return diag_fraction, occupancy, largest_comp_fraction


def _largest_connected_component_fraction(grid: np.ndarray) -> float:
    """Fraction of occupied cells belonging to the largest 4-connected component."""
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
            # BFS
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


def _strip_timestamp_suffix(payload: str) -> str:
    """Remove the LongSplat safe-state timestamp suffix like ' [22/07 16:02:03]'."""
    import re as _re

    return _re.sub(
        r"\s+\[(?:0[1-9]|[12][0-9]|3[01])/(?:0[1-9]|1[0-2])\s+"
        r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\]$",
        "",
        payload,
    )
