"""CLI for batch quality evaluation of LongSplat outputs.

Usage::

    ./dev.sh python -m scripts.longsplat.evaluate_model \\
        --ply <path> --cameras <path> --loss-log <path> \\
        --output <json-path>

The output file must NOT already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from scripts.longsplat.quality_metrics import (
    analyze_camera_trajectory,
    analyze_loss_log,
    analyze_ply_quality,
)


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate LongSplat model quality")
    parser.add_argument("--ply", required=True, type=Path, help="Path to 3DGS PLY")
    parser.add_argument(
        "--cameras",
        required=True,
        type=Path,
        help="Path to cameras_all_train.json",
    )
    parser.add_argument(
        "--loss-log",
        required=True,
        type=Path,
        help="Path to train.log",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON path (must not exist)",
    )
    args = parser.parse_args(argv)

    output_path = args.output.resolve()
    if output_path.exists():
        print(f"ERROR: output file already exists: {output_path}", file=sys.stderr)
        sys.exit(1)

    ply_path = args.ply.resolve()
    cameras_path = args.cameras.resolve()
    loss_path = args.loss_log.resolve()

    ply_quality = analyze_ply_quality(ply_path)
    trajectory = analyze_camera_trajectory(cameras_path)
    loss_stats = analyze_loss_log(loss_path)

    result = {
        "metric_schema": ply_quality["metric_schema"],
        "inputs": {
            "ply": {
                "path": str(ply_path),
                "sha256": _sha256_hex(ply_path),
            },
            "cameras": {
                "path": str(cameras_path),
                "sha256": _sha256_hex(cameras_path),
            },
            "loss_log": {
                "path": str(loss_path),
                "sha256": _sha256_hex(loss_path),
            },
        },
        "ply_quality": ply_quality,
        "trajectory": trajectory,
        "loss_stats": loss_stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Evaluation written to {output_path}")


if __name__ == "__main__":
    main()
