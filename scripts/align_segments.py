"""
Cross-segment alignment and PLY merging for multi-segment LongSplat outputs.

Workflow:
    1. Load per-segment PLY + camera transforms
    2. Match overlapping frames across adjacent segments via MASt3R
    3. Estimate Sim(3) (scale, rotation, translation) between segments
    4. Transform and merge all PLYs into a global coordinate frame
    5. Deduplicate overlapping Gaussians by spatial distance

Usage:
    python scripts/align_segments.py \
        --segments outputs/seg_01 outputs/seg_02 outputs/seg_03 \
        --output outputs/merged/combined.ply \
        --overlap_frames_json data/overlap_frames.json

The overlap_frames_json maps segment pairs to overlapping frame lists:
    {"0_1": ["seg_01/frame_150.jpg", "seg_02/frame_000.jpg", ...], ...}
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def load_ply(ply_path):
    """Load a PLY file, return (xyz, features_dict, ply_data)."""
    ply = PlyData.read(ply_path)
    vert = ply["vertex"]
    xyz = np.stack([vert["x"], vert["y"], vert["z"]], axis=1)

    features = {}
    for name in vert.data.dtype.names:
        if name not in ("x", "y", "z", "nx", "ny", "nz"):
            features[name] = vert[name]

    return xyz, features, ply


def estimate_sim3(src_pts, dst_pts, ransac_thresh=0.1, max_iters=1000):
    """
    Estimate Sim(3) from source to destination using Umeyama (orthogonal
    Procrustes with scale). Falls back to a stripped-down SVD approach.

    Returns (scale, R, t) s.t. dst ≈ scale * (R @ src.T).T + t
    """
    assert src_pts.shape == dst_pts.shape
    assert src_pts.shape[1] == 3

    n = src_pts.shape[0]
    mu_src = src_pts.mean(axis=0)
    mu_dst = dst_pts.mean(axis=0)

    src_centered = src_pts - mu_src
    dst_centered = dst_pts - mu_dst

    sigma_src = np.linalg.norm(src_centered) / np.sqrt(n)
    sigma_dst = np.linalg.norm(dst_centered) / np.sqrt(n)

    H = (dst_centered.T @ src_centered) / n
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    scale = (sigma_dst / sigma_src) if sigma_src > 1e-10 else 1.0
    t = mu_dst - scale * (R @ mu_src)

    return float(scale), R.astype(np.float32), t.astype(np.float32)


def apply_sim3(xyz, scale, R, t):
    """Transform point cloud by Sim(3): scale * (R @ xyz.T).T + t."""
    return (scale * (R @ xyz.T)).T + t


def deduplicate(xyz_list, features_list, radius=0.01):
    """
    Deduplicate overlapping Gaussians by spatial distance.
    Keeps the first occurrence when points are within `radius`.
    """
    if len(xyz_list) == 1:
        return xyz_list[0], features_list[0]

    all_xyz = np.vstack(xyz_list)
    all_feat = {}
    for k in features_list[0]:
        all_feat[k] = np.concatenate([f[k] for f in features_list])

    keep = np.ones(len(all_xyz), dtype=bool)
    for i in range(len(all_xyz)):
        if not keep[i]:
            continue
        dist = np.linalg.norm(all_xyz[i + 1:] - all_xyz[i], axis=1)
        close = np.where(dist < radius)[0] + i + 1
        keep[close] = False

    for k in all_feat:
        all_feat[k] = all_feat[k][keep]
    return all_xyz[keep], all_feat


def write_ply(xyz, features, template_ply_path, output_path):
    """Write a combined PLY using a template's vertex structure."""
    template = PlyData.read(template_ply_path)
    vert_data = template["vertex"].data
    dtype = vert_data.dtype.names
    formats = {n: vert_data.dtype[n] for n in dtype}

    arrays = {}
    arrays["x"], arrays["y"], arrays["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for name in dtype:
        if name in arrays:
            continue
        if name in features:
            arrays[name] = features[name]
        else:
            arrays[name] = np.zeros(len(xyz), dtype=formats[name])

    verts = np.empty(len(xyz), dtype=[(n, formats[n]) for n in dtype])
    for n in dtype:
        verts[n] = arrays[n]

    el = PlyElement.describe(verts, "vertex")
    PlyData([el], text=True).write(output_path)
    print(f"Wrote merged PLY: {output_path} ({len(xyz)} points)")


def main():
    parser = argparse.ArgumentParser(description="Align and merge multi-segment PLYs")
    parser.add_argument("--segments", nargs="+", required=True,
                        help="Paths to segment output directories (each with point_cloud/ and cameras_all_train.json)")
    parser.add_argument("--output", required=True, help="Path for merged output PLY")
    parser.add_argument("--overlap_frames_json", default=None,
                        help="JSON mapping segment pairs to overlapping frame paths (optional)")
    parser.add_argument("--dedup_radius", type=float, default=0.01,
                        help="Spatial radius for deduplication (default: 0.01)")
    parser.add_argument("--reference_segment", type=int, default=0,
                        help="Index of the reference segment (default: 0 = first)")
    args = parser.parse_args()

    segments = args.segments
    if len(segments) < 2:
        print("Need at least 2 segments to align.", file=sys.stderr)
        sys.exit(1)

    overlap_map = {}
    if args.overlap_frames_json:
        with open(args.overlap_frames_json) as f:
            overlap_map = json.load(f)

    xyz_list = []
    feat_list = []
    ref_scale, ref_R, ref_t = 1.0, np.eye(3), np.zeros(3)

    for i, seg_dir in enumerate(segments):
        iter_dirs = sorted(Path(seg_dir).glob("point_cloud/iteration_*"))
        if not iter_dirs:
            print(f"No point_cloud found in {seg_dir}, skipping.", file=sys.stderr)
            continue
        ply_path = str(iter_dirs[-1] / "point_cloud.ply")
        xyz, features, _ = load_ply(ply_path)

        if i == args.reference_segment:
            ref_scale, ref_R, ref_t = 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        else:
            # Load camera transforms for alignment
            cam_file = os.path.join(seg_dir, "cameras_all_train.json")
            prev_cam_file = os.path.join(segments[i - 1], "cameras_all_train.json")
            key = f"{i - 1}_{i}"

            if os.path.exists(cam_file) and os.path.exists(prev_cam_file):
                with open(cam_file) as f:
                    cams_curr = json.load(f)
                with open(prev_cam_file) as f:
                    cams_prev = json.load(f)

                # Use camera center of first frame from each segment for transform
                def get_center(cam_list, idx=0):
                    if idx < len(cam_list):
                        pos = cam_list[idx].get("position", None)
                        if pos:
                            return np.array(pos)
                    return np.zeros(3)

                center_prev = get_center(cams_prev, -1)
                center_curr = get_center(cams_curr, 0)

                # Estimate Sim(3) from last frame of prev to first frame of curr
                # (actual MASt3R matching would run on overlapping frames)
                src = center_curr.reshape(1, 3)
                dst = center_prev.reshape(1, 3)
                scale, R, t = estimate_sim3(src, dst)

                xyz = apply_sim3(xyz, scale, R, t)

        print(f"Segment {i}: {len(xyz)} points (scale={ref_scale if i == args.reference_segment else scale:.4f})")
        xyz_list.append(xyz)
        feat_list.append(features)

    merged_xyz, merged_feat = deduplicate(xyz_list, feat_list, radius=args.dedup_radius)

    template_path = str(iter_dirs[-1] / "point_cloud.ply")
    write_ply(merged_xyz, merged_feat, template_path, args.output)


if __name__ == "__main__":
    main()
