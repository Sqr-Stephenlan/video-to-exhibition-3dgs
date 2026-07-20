"""Materialize depth-prior NPZ outputs as npy files for LongSplat consumption.

Reads a depth_manifest.json produced by the depth-prior module, converts each
per-frame NPZ (key ``depth``) to a flat .npy at the path that LongSplat's
external depth hook expects.

Output path pattern:
    <source_path>/depths/<image_stem>_depth.npy

where ``image_stem`` is derived from ``rgb_path`` in the depth manifest
(e.g., ``data/frames/wall_test/selected/seg_0002/frame_000001_t20.000.jpg``
→ ``frame_000001_t20.000``).

VDA values are written **as-is** (disparity-like: higher = nearer). Direction
conversion and scene-scale alignment happen inside LongSplat's training loop
via ``align_vda_depth()`` where a reference z-depth is available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def materialize(depth_manifest_path: Path, source_path: Path) -> int:
    """Convert all frames in *depth_manifest_path* to .npy under *source_path*.

    Returns the number of frames written (including unchanged re-runs).
    """
    if not depth_manifest_path.is_file():
        print(f"ERROR: depth manifest not found: {depth_manifest_path}")
        return 1

    with depth_manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        print("ERROR: depth manifest has no frames array")
        return 1

    depth_dir = source_path / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    missing = 0

    for frame in frames:
        frame_id = frame.get("frame_id", "?")
        npz_path = frame.get("depth_path")
        rgb_path = frame.get("rgb_path")

        if not npz_path:
            print(f"WARNING: {frame_id}: missing depth_path, skipping")
            missing += 1
            continue
        if not rgb_path:
            print(f"WARNING: {frame_id}: missing rgb_path, skipping")
            missing += 1
            continue

        npz_full = Path(npz_path)
        if not npz_full.is_file():
            print(f"WARNING: {frame_id}: NPZ not found at {npz_path}, skipping")
            missing += 1
            continue

        image_stem = Path(rgb_path).name.split(".")[0]
        npy_dest = depth_dir / f"{image_stem}_depth.npy"

        data = np.load(npz_full)
        if "depth" not in data:
            print(f"WARNING: {frame_id}: NPZ missing 'depth' key, skipping")
            missing += 1
            continue

        depth = data["depth"]

        # Idempotent: skip if destination already holds identical data.
        if npy_dest.is_file():
            existing = np.load(npy_dest)
            if existing.shape == depth.shape and np.allclose(existing, depth):
                skipped += 1
                continue

        np.save(npy_dest, depth)
        written += 1

    print(
        f"Materialized {written} depth frames to {depth_dir.as_posix()}"
        + (f" ({skipped} unchanged, {missing} missing)" if skipped or missing else "")
    )
    if missing:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize depth-prior NPZ outputs for LongSplat."
    )
    parser.add_argument(
        "depth_manifest",
        type=Path,
        help="Path to depth_manifest.json",
    )
    parser.add_argument(
        "source_path",
        type=Path,
        help="LongSplat source_path (depth npy files written to <source_path>/depths/)",
    )
    args = parser.parse_args()
    return materialize(args.depth_manifest, args.source_path)


if __name__ == "__main__":
    sys.exit(main())
