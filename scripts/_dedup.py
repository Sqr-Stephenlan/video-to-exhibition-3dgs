"""
Spatial deduplication for merged 3DGS point clouds using scipy cKDTree.

Replaces the original O(N²) pairwise-distance loop with an O(N log N)
radius search that scales to millions of Gaussians.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def deduplicate_spatial(
    xyz_list: list[np.ndarray],
    features_list: list[dict[str, np.ndarray]],
    radius: float = 0.01,
    opacity_attr: str | None = "opacity",
    chunk_size: int = 100_000,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Deduplicate overlapping Gaussians using cKDTree radius search.

    For each point, neighbors within *radius* are identified.  Among a
    neighbourhood the point with the highest opacity is kept (if
    *opacity_attr* exists in the PLY); otherwise the first occurrence wins.

    Processing is done in chunks to bound peak memory.

    Returns
    -------
    merged_xyz : (M, 3) np.ndarray
    merged_features : dict of (M,) or (M, D) arrays
    """
    if len(xyz_list) == 1:
        return xyz_list[0], features_list[0]

    total_n = sum(len(x) for x in xyz_list)

    # Build cKDTree from all points — O(N log N)
    all_xyz = np.vstack(xyz_list)
    tree = cKDTree(all_xyz)

    # Determine whether we can use opacity for retention
    use_opacity = opacity_attr is not None and all(opacity_attr in f for f in features_list)
    if use_opacity:
        all_opacity = np.concatenate([f[opacity_attr] for f in features_list])

    keep = np.ones(total_n, dtype=bool)

    # Process in chunks to limit temporary arrays
    for start in range(0, total_n, chunk_size):
        end = min(start + chunk_size, total_n)
        chunk_idx = np.arange(start, end)[keep[start:end]]
        if len(chunk_idx) == 0:
            continue

        neighbors = tree.query_ball_point(all_xyz[chunk_idx], radius, workers=-1)

        for i_local, nbrs in enumerate(neighbors):
            i_global = chunk_idx[i_local]
            if not keep[i_global]:
                continue
            if len(nbrs) <= 1:
                continue

            nbrs_arr = np.array(nbrs, dtype=np.int64)
            # Only remove neighbors with higher index (preserves first-by-default)
            to_consider = nbrs_arr[nbrs_arr > i_global]
            to_consider = to_consider[keep[to_consider]]
            if len(to_consider) == 0:
                continue

            if use_opacity:
                best_idx = to_consider[np.argmax(all_opacity[to_consider])]
                if all_opacity[best_idx] > all_opacity[i_global]:
                    keep[i_global] = False
                else:
                    keep[to_consider] = False
            else:
                keep[to_consider] = False

    # -- assemble output -------------------------------------------------
    merged_xyz = all_xyz[keep]
    merged_features: dict[str, np.ndarray] = {}
    all_keys = set()
    for fl in features_list:
        all_keys.update(fl.keys())

    for k in sorted(all_keys):
        arrays = [fl[k] for fl in features_list]
        merged = np.concatenate(arrays)[keep]
        merged_features[k] = merged

    return merged_xyz, merged_features
