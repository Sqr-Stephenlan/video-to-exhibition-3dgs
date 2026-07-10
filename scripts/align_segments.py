"""
Cross-segment PLY alignment and fusion for multi-segment LongSplat outputs.

Workflow
--------
1. Load and validate per-segment PLY + camera metadata.
2. Match overlapping frames across adjacent segments via MASt3R.
3. Estimate Sim(3) (scale, rotation, translation) between segments using
   RANSAC on the multi-point 3D correspondences.
4. Build a pose graph relative to a reference segment and propagate
   cumulative transforms.
5. Transform PLY attributes (xyz, scale, rotation, normals) into the
   global frame.
6. Deduplicate overlapping Gaussians with a spatial index.
7. Write the merged PLY and a machine-readable report.

Usage
-----
::

    python scripts/align_segments.py \\
        --segments outputs/seg_01 outputs/seg_02 outputs/seg_03 \\
        --output outputs/merged/combined.ply \\
        --overlap_frames_json data/overlap_frames.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from _common import (
    now_utc_iso,
    write_json_atomic,
)
from _dedup import deduplicate_spatial
from _mast3r_adapter import (
    InsufficientMatchesError,
    Mast3rMatcher,
    estimate_sim3_ransac,
)
from _ply_transform import SchemaError, transform_ply_attributes
from plyfile import PlyData

# ---------------------------------------------------------------------------
# Validation (CR-07)
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Aggregated pre-flight validation failures."""


def _find_iter_dir(seg_path: Path) -> Path:
    """Return the highest-numbered ``point_cloud/iteration_*`` directory."""
    iter_dirs = sorted(
        seg_path.glob("point_cloud/iteration_*"),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not iter_dirs:
        raise ValidationError(f"No point_cloud/iteration_* found in {seg_path}")
    return iter_dirs[-1]


def _load_cameras(cam_path: Path) -> list[dict[str, Any]]:
    """Load camera list from a ``cameras_all_train.json`` file."""
    import json

    if not cam_path.exists():
        raise ValidationError(f"Camera file missing: {cam_path}")
    with open(cam_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or len(data) == 0:
        raise ValidationError(f"Camera file is empty or not a list: {cam_path}")
    return data


def _get_camera_center(cam: dict[str, Any]) -> np.ndarray:
    """Extract camera center from a camera dict.

    Supports both ``position`` (explicit) and ``T_world_cam`` (4x4 matrix)
    conventions commonly found in COLMAP / LongSplat outputs.
    """
    pos = cam.get("position")
    if pos is not None:
        return np.array(pos, dtype=np.float64)

    T = cam.get("T_world_cam") or cam.get("transform")
    if T is not None:
        T = np.array(T, dtype=np.float64)
        # camera center = -R.T @ t  (for world = R @ cam + t convention)
        R_inv = T[:3, :3].T
        return (-R_inv @ T[:3, 3]).astype(np.float64)

    raise ValidationError("Camera dict has neither 'position' nor 'T_world_cam'/'transform' keys")


def validate_segments(
    segments: list[str], ref_idx: int
) -> tuple[list[Path], list[Path], list[dict[str, Any]]]:
    """Pre-flight check of all segment directories.

    Returns ``(seg_dirs, ply_paths, schemas)``.
    """
    if len(segments) < 2:
        raise ValidationError("Need at least 2 segments for alignment.")

    if not (0 <= ref_idx < len(segments)):
        raise ValidationError(f"reference_segment {ref_idx} out of range (0–{len(segments) - 1})")

    seg_dirs: list[Path] = []
    ply_paths: list[Path] = []
    schemas: list[dict[str, Any]] = []

    for i, s in enumerate(segments):
        sd = Path(s).resolve()
        if not sd.is_dir():
            raise ValidationError(f"Segment {i} is not a directory: {sd}")

        iter_dir = _find_iter_dir(sd)
        ply_path = iter_dir / "point_cloud.ply"
        if not ply_path.exists():
            raise ValidationError(f"PLY missing: {ply_path}")

        cam_path = sd / "cameras_all_train.json"
        _load_cameras(cam_path)  # validates existence + format

        # Read schema for compatibility checks
        ply = PlyData.read(ply_path)
        schema = {
            "vertex_count": ply["vertex"].count,
            "attributes": list(ply["vertex"].data.dtype.names),
            "dtypes": {n: str(ply["vertex"].data.dtype[n]) for n in ply["vertex"].data.dtype.names},
        }
        schemas.append(schema)
        seg_dirs.append(sd)
        ply_paths.append(ply_path)

    # Cross-segment schema compatibility
    base = schemas[0]
    for i, s in enumerate(schemas[1:], 1):
        if s["attributes"] != base["attributes"]:
            raise ValidationError(
                f"Schema mismatch: segment 0 has {base['attributes']}, "
                f"segment {i} has {s['attributes']}"
            )

    return seg_dirs, ply_paths, schemas


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Align and merge multi-segment 3DGS PLYs")
    parser.add_argument(
        "--segments",
        nargs="+",
        required=True,
        help="Segment output dirs (each must have point_cloud/ and cameras_all_train.json)",
    )
    parser.add_argument("--output", required=True, help="Path for merged output PLY")
    parser.add_argument(
        "--overlap_frames_json",
        required=True,
        help="JSON mapping adjacent segment pairs to lists of overlapping frame paths",
    )
    parser.add_argument(
        "--reference_segment",
        type=int,
        default=0,
        help="Index of the reference segment (anchor) in --segments (default: 0)",
    )
    parser.add_argument(
        "--ransac_thresh",
        type=float,
        default=0.05,
        help="RANSAC inlier distance threshold",
    )
    parser.add_argument(
        "--ransac_max_iters",
        type=int,
        default=1000,
        help="Maximum RANSAC iterations",
    )
    parser.add_argument(
        "--min_inliers",
        type=int,
        default=10,
        help="Minimum required RANSAC inliers per segment pair",
    )
    parser.add_argument(
        "--dedup_radius",
        type=float,
        default=0.01,
        help="Spatial radius for Gaussian deduplication",
    )
    parser.add_argument(
        "--mast3r_checkpoint",
        default=None,
        help="Path to MASt3R checkpoint (default: use MASt3R default)",
    )
    parser.add_argument(
        "--mast3r_min_confidence",
        type=float,
        default=0.5,
        help="Minimum MASt3R match confidence",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Phase 0: Validate everything before any computation (CR-07)
    # ------------------------------------------------------------------
    print("=== Phase 0: Validation ===")
    try:
        seg_dirs, ply_paths, schemas = validate_segments(args.segments, args.reference_segment)
    except ValidationError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(seg_dirs)} segments, reference = {args.reference_segment}")

    # -- overlap frames -------------------------------------------------
    import json

    overlap_path = Path(args.overlap_frames_json)
    if not overlap_path.exists():
        print(f"FATAL: overlap_frames_json not found: {overlap_path}", file=sys.stderr)
        sys.exit(1)
    with open(overlap_path, encoding="utf-8") as fh:
        overlap_map = json.load(fh)

    n_segs = len(seg_dirs)
    for i in range(n_segs - 1):
        key = f"{i}_{i + 1}"
        if key not in overlap_map:
            print(f"FATAL: overlap_frames_json missing pair '{key}'", file=sys.stderr)
            sys.exit(1)
        frames = overlap_map[key]
        if not isinstance(frames, list) or len(frames) < 1:
            print(f"FATAL: pair '{key}' has insufficient overlap frames", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 1: MASt3R pairwise matching + RANSAC Sim(3) (CR-01)
    # ------------------------------------------------------------------
    print("=== Phase 1: MASt3R matching + Sim(3) estimation ===")

    matcher = Mast3rMatcher(
        checkpoint=args.mast3r_checkpoint,
        min_confidence=args.mast3r_min_confidence,
    )
    pair_results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    for i in range(n_segs - 1):
        j = i + 1
        key = f"{i}_{j}"
        print(f"  Pair {key} …")

        try:
            # Load overlapping frame paths
            entry = overlap_map[key]
            if isinstance(entry, dict):
                frames_i = entry.get("frames_i", entry.get("frames_src", []))
                frames_j = entry.get("frames_j", entry.get("frames_dst", []))
            else:
                # Simple list: split evenly (first half = seg i, second = seg j)
                half = len(entry) // 2
                frames_i = entry[:half]
                frames_j = entry[half:]

            src_pts, dst_pts, confidences = matcher.match_pair(frames_i, frames_j)
            scale, R, t, n_inliers = estimate_sim3_ransac(
                src_pts,
                dst_pts,
                thresh=args.ransac_thresh,
                max_iters=args.ransac_max_iters,
                min_inliers=args.min_inliers,
            )
            pair_results[key] = {
                "scale": float(scale),
                "R": R.tolist(),
                "t": t.tolist(),
                "num_inliers": n_inliers,
                "num_correspondences": len(src_pts),
            }
            print(f"    {n_inliers} inliers / {len(src_pts)} correspondences")

        except (InsufficientMatchesError, NotImplementedError) as exc:
            failures.append(f"Pair {key}: {exc}")

    if failures:
        print("\nFATAL: Alignment failures:", file=sys.stderr)
        for f_msg in failures:
            print(f"  - {f_msg}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2: Pose graph -> cumulative transforms (CR-01)
    # ------------------------------------------------------------------
    print("=== Phase 2: Pose graph propagation ===")

    # Build adjacency: from segment i to segment i+1
    # global_T[i] = transform from segment i → global (reference) frame
    global_transforms: list[tuple[float, np.ndarray, np.ndarray]] = [
        (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)) for _ in range(n_segs)
    ]

    # Forward chain from reference to the right
    for i in range(args.reference_segment, n_segs - 1):
        key = f"{i}_{i + 1}"
        if key in pair_results:
            pr = pair_results[key]
            s_local = np.float32(pr["scale"])
            R_local = np.array(pr["R"], dtype=np.float32)
            t_local = np.array(pr["t"], dtype=np.float32)
            # Compose: global_T[i+1] = local * global_T[i]
            prev_s, prev_R, prev_t = global_transforms[i]
            new_s = prev_s * s_local
            new_R = R_local @ prev_R
            new_t = s_local * (R_local @ prev_t) + t_local
            global_transforms[i + 1] = (new_s, new_R, new_t)

    # Backward chain from reference to the left
    for i in range(args.reference_segment, 0, -1):
        key = f"{i - 1}_{i}"
        if key in pair_results:
            pr = pair_results[key]
            s_local = np.float32(pr["scale"])
            R_local = np.array(pr["R"], dtype=np.float32)
            t_local = np.array(pr["t"], dtype=np.float32)
            # Invert local transform: seg i → seg i-1
            R_inv = R_local.T
            s_inv = 1.0 / max(s_local, 1e-12)
            t_inv = -s_inv * (R_inv @ t_local)
            # Compose with global_T[i]
            prev_s, prev_R, prev_t = global_transforms[i]
            new_s = prev_s * s_inv
            new_R = R_inv @ prev_R
            new_t = s_inv * (R_inv @ prev_t) + t_inv
            global_transforms[i - 1] = (new_s, new_R, new_t)

    # ------------------------------------------------------------------
    # Phase 3: Transform PLYs to global frame (CR-01)
    # ------------------------------------------------------------------
    print("=== Phase 3: PLY transformation ===")

    transformed_xyz: list[np.ndarray] = []
    transformed_features: list[dict[str, np.ndarray]] = []

    for i, ply_path in enumerate(ply_paths):
        ply = PlyData.read(ply_path)
        scale, R, t = global_transforms[i]

        if i == args.reference_segment and abs(scale - 1.0) < 1e-8:
            vert = ply["vertex"]
            xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=1).astype(np.float32)
            features: dict[str, np.ndarray] = {
                n: np.asarray(vert[n], dtype=np.float32)
                for n in vert.data.dtype.names
                if n not in ("x", "y", "z", "nx", "ny", "nz")
            }
        else:
            try:
                t_ply = transform_ply_attributes(ply, scale, R, t)
            except SchemaError as exc:
                print(f"FATAL: PLY transform error in segment {i}: {exc}", file=sys.stderr)
                sys.exit(1)
            vert = t_ply["vertex"]
            xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=1).astype(np.float32)
            features = {
                n: np.asarray(vert[n], dtype=np.float32)
                for n in vert.data.dtype.names
                if n not in ("x", "y", "z", "nx", "ny", "nz")
            }

        transformed_xyz.append(xyz)
        transformed_features.append(features)
        print(
            f"  Segment {i}: {len(xyz)} points "
            f"(scale={scale:.4f}, |R|=1, |t|={np.linalg.norm(t):.4f})"
        )

    # ------------------------------------------------------------------
    # Phase 4: Merge + deduplicate (CR-08)
    # ------------------------------------------------------------------
    print("=== Phase 4: Merge + deduplicate ===")

    merged_xyz, merged_features = deduplicate_spatial(
        transformed_xyz, transformed_features, radius=args.dedup_radius
    )
    print(f"  {len(merged_xyz)} points after dedup")

    # ------------------------------------------------------------------
    # Phase 5: Write output (atomic) (CR-07)
    # ------------------------------------------------------------------
    print("=== Phase 5: Write output ===")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use the first segment's PLY as the structure template
    template = PlyData.read(ply_paths[0])
    vert_tpl = template["vertex"].data
    dtype_names = list(vert_tpl.dtype.names)
    dtype_formats = {n: vert_tpl.dtype[n] for n in dtype_names}

    arrays: dict[str, np.ndarray] = {
        "x": merged_xyz[:, 0],
        "y": merged_xyz[:, 1],
        "z": merged_xyz[:, 2],
    }
    for n in dtype_names:
        if n in ("x", "y", "z"):
            continue
        if n in ("nx", "ny", "nz"):
            arrays[n] = np.zeros(len(merged_xyz), dtype=np.float32)
        elif n in merged_features:
            arrays[n] = merged_features[n]
        else:
            arrays[n] = np.zeros(len(merged_xyz), dtype=dtype_formats[n])

    verts = np.empty(len(merged_xyz), dtype=[(n, dtype_formats[n]) for n in dtype_names])
    for n in dtype_names:
        verts[n] = arrays[n]

    from plyfile import PlyElement

    PlyData([PlyElement.describe(verts, "vertex")], text=True).write(str(output_path))
    print(f"  Wrote {output_path}")

    # ------------------------------------------------------------------
    # Phase 6: Report
    # ------------------------------------------------------------------
    report = {
        "mast3r_version": matcher.version,
        "num_segments": n_segs,
        "reference_segment": args.reference_segment,
        "pair_results": pair_results,
        "global_transforms": {
            str(i): {
                "scale": float(global_transforms[i][0]),
                "R": global_transforms[i][1].tolist(),
                "t": global_transforms[i][2].tolist(),
            }
            for i in range(n_segs)
        },
        "dedup_radius": args.dedup_radius,
        "output_points": len(merged_xyz),
        "output_path": str(output_path.resolve()),
        "created": now_utc_iso(),
    }
    report_path = output_path.with_suffix(".report.json")
    write_json_atomic(report, report_path)
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
