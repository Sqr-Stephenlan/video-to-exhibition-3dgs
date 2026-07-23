"""
Thin CLI entry point for LongSplat reproduction experiments.

Loads a config JSON and delegates everything to
:func:`scripts.longsplat.orchestrator.run_pipeline`.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from .orchestrator import run_pipeline
from .runner import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LongSplat reproduction experiment",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=str,
        help="Path to producer manifest (schema 1.0)",
    )
    parser.add_argument(
        "--segment-id",
        required=True,
        type=str,
        help="Segment ID within the manifest",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to LongSplatConfig JSON",
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        type=str,
        help="Root of the LongSplat backend repository",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Parent directory for run outputs",
    )
    parser.add_argument(
        "--project-root",
        required=True,
        type=str,
        help="Repository root for resolving manifest paths",
    )
    parser.add_argument(
        "--depth-manifest",
        default=None,
        type=str,
        help="Optional depth manifest from depth-prior",
    )
    parser.add_argument(
        "--backend-python",
        default="python",
        type=str,
        help="Python interpreter for the LongSplat backend",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        type=str,
        help="Custom run identifier (auto-generated if omitted)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_pipeline(
        manifest_path=args.manifest,
        segment_id=args.segment_id,
        config=load_config(args.config),
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        project_root=args.project_root,
        depth_manifest_path=args.depth_manifest,
        python_exe=args.backend_python,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
