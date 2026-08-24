"""CLI wrapper around ``depth_bridge`` for standalone depth materialisation.

For programmatic use prefer :func:`depth_bridge.materialize_all`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .depth_bridge import DepthContractError, materialize_all


def materialize(depth_manifest_path: Path, source_path: Path) -> int:
    """Convert depth NPZ files to npy under *source_path*/depths/.

    Thin wrapper that builds a minimal frame_mapping from the depth
    manifest and delegates to :func:`depth_bridge.materialize_all`.
    """
    if not depth_manifest_path.is_file():
        print(f"ERROR: depth manifest not found: {depth_manifest_path}")
        return 1

    try:
        with depth_manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in depth manifest: {exc}")
        return 1

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        print("ERROR: depth manifest has no frames array")
        return 1

    # Build a frame_mapping from the depth manifest itself so the CLI
    # can work standalone without an orchestrator-provided mapping.
    frame_mapping: list[dict] = []
    for frame in frames:
        rgb_path = frame.get("rgb_path")
        if not rgb_path:
            continue
        stem = Path(rgb_path).stem
        frame_mapping.append(
            {
                "source_path": rgb_path,
                "prepared_name": stem,
            }
        )

    if not frame_mapping:
        print("ERROR: no frames with valid rgb_path in depth manifest")
        return 1

    output_depth_dir = source_path / "depths"
    project_root = Path.cwd()

    try:
        result = materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=frame_mapping,
            project_root=project_root,
            output_depth_dir=output_depth_dir,
        )
    except DepthContractError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(
        f"Materialized {result.materialized_count}/{result.expected_count} "
        f"depth frames to {output_depth_dir.as_posix()}"
    )
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
