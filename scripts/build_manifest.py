"""
Generate a structured frame-path manifest for downstream reconstruction tools.

Usage::

    python scripts/build_manifest.py --input data/frames_filtered/ --segment scene_01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from _common import (
    ensure_output_dir,
    get_project_root,
    now_utc_iso,
    write_json_atomic,
)


def build_manifest(
    frames_dir: str,
    segment_name: str = "scene_01",
    output_dir: Path | None = None,
) -> str:
    in_dir = Path(frames_dir).resolve()
    out_dir = output_dir or (get_project_root() / "data" / "manifests")
    ensure_output_dir(out_dir)

    frames = sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.png"))
    if not frames:
        frames = sorted(in_dir.glob("frame_*.*"))
    if not frames:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")

    data_root = get_project_root() / "data"

    entries: list[dict] = []
    for idx, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            raise FileNotFoundError(f"Cannot read image (corrupt or missing): {fp}")
        h, w = img.shape[:2]

        # Compute path relative to data_root (or absolute if outside)
        try:
            rel_path = str(fp.relative_to(data_root))
        except ValueError:
            rel_path = str(fp)

        entries.append(
            {
                "id": idx,
                "file": rel_path,
                "width": w,
                "height": h,
            }
        )

    # Derived paths
    paths = {
        "frames_dir": str(in_dir),
        "depth_dir": f"{segment_name}/depth",
        "segments_dir": f"{segment_name}/segments",
        "mask_dir": f"{segment_name}/masks",
    }

    manifest = {
        "segment": segment_name,
        "num_frames": len(entries),
        "paths": paths,
        "frames": entries,
        "created": now_utc_iso(),
    }

    out_path = out_dir / f"{segment_name}_manifest.json"
    write_json_atomic(manifest, out_path)
    print(f"Manifest written: {out_path}  ({len(entries)} frames)")
    return str(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frame manifest")
    parser.add_argument("--input", required=True, help="Input frames directory")
    parser.add_argument("--segment", default="scene_01", help="Segment name")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <project>/data/manifests)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else get_project_root() / "data" / "manifests"

    build_manifest(args.input, args.segment, output_dir)


if __name__ == "__main__":
    main()
