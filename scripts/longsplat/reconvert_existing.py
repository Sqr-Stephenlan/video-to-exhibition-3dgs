"""Isolated LongSplat checkpoint reconversion.

Creates a minimal snapshot from an existing trained model directory and
re-runs conversion against a clean destination.  The original model is
never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.longsplat.runner import (
    LongSplatConfig,
    build_convert_command,
)


# Files whitelisted for snapshot copy.
_SNAPSHOT_WHITELIST = [
    "cfg_args",
    "cameras_all_train.json",
    "cameras_all_test.json",
]


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_conversion_snapshot(
    source_model: Path,
    destination_model: Path,
    *,
    checkpoint_iteration: int,
) -> dict[str, Any]:
    """Copy only cfg_args, camera JSON files, and one checkpoint.

    *destination_model* must not exist.  Source files are hashed before and
    after the copy to prove they are unchanged.

    Returns a provenance dict with source and destination SHAs.
    """
    source = Path(source_model).resolve()
    destination = Path(destination_model).resolve()

    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    checkpoint_dir_rel = Path("point_cloud") / f"iteration_{checkpoint_iteration}"
    checkpoint_files = [
        "point_cloud.ply",
        "color_mlp.pt",
        "cov_mlp.pt",
        "opacity_mlp.pt",
    ]

    # --- Validate source ---
    for rel in _SNAPSHOT_WHITELIST:
        p = source / rel
        if not p.is_file():
            raise FileNotFoundError(f"missing source file: {p}")
    for fname in checkpoint_files:
        p = source / checkpoint_dir_rel / fname
        if not p.is_file():
            raise FileNotFoundError(f"missing checkpoint file: {p}")

    # --- Hash source before ---
    hashes_before: dict[str, str] = {}
    all_src_files: list[Path] = []
    for rel in _SNAPSHOT_WHITELIST:
        all_src_files.append(source / rel)
    for fname in checkpoint_files:
        all_src_files.append(source / checkpoint_dir_rel / fname)
    for p in all_src_files:
        hashes_before[str(p)] = _sha256_hex(p)

    # --- Copy ---
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for rel in _SNAPSHOT_WHITELIST:
            src = source / rel
            dst = destination / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # Checkpoint files
        dst_checkpoint = destination / checkpoint_dir_rel
        dst_checkpoint.mkdir(parents=True, exist_ok=True)
        for fname in checkpoint_files:
            src = source / checkpoint_dir_rel / fname
            dst = dst_checkpoint / fname
            shutil.copy2(src, dst)

        # Copy cameras_all_train.json as cameras_all.json for Eval loader
        cameras_train_dst = destination / "cameras_all_train.json"
        cameras_all_dst = destination / "cameras_all.json"
        shutil.copy2(cameras_train_dst, cameras_all_dst)
    except BaseException:
        # Clean up partial destination on failure
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise

    # --- Hash source after (prove no modification) ---
    hashes_after: dict[str, str] = {}
    for p in all_src_files:
        hashes_after[str(p)] = _sha256_hex(p)

    mismatches = [
        str(p) for p in all_src_files if hashes_before[str(p)] != hashes_after[str(p)]
    ]
    if mismatches:
        raise RuntimeError(f"Source files modified during snapshot: {mismatches}")

    return {
        "source_model": str(source),
        "destination_model": str(destination),
        "checkpoint_iteration": checkpoint_iteration,
        "source_sha256": {str(k): v for k, v in hashes_before.items()},
        "copied_files": [
            str(d.relative_to(destination))
            for d in sorted(destination.rglob("*"))
            if d.is_file()
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-convert an existing LongSplat checkpoint in isolation",
    )
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--destination-model", required=True, type=Path)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--backend-python", required=True, type=Path)
    parser.add_argument("--checkpoint-iteration", required=True, type=int)
    parser.add_argument("--conversion-iterations", required=True, type=int)
    parser.add_argument("--prune-ratio", required=True, type=float)
    parser.add_argument("--backend-mode", default="research_local", type=str)
    parser.add_argument("--output-record", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _argparser()
    args = parser.parse_args(argv)

    source_model = args.source_model.resolve()
    destination_model = args.destination_model.resolve()
    output_record = args.output_record.resolve()

    if output_record.exists():
        print(f"ERROR: output record already exists: {output_record}", file=sys.stderr)
        sys.exit(1)

    # Step 1: create snapshot
    snapshot = create_conversion_snapshot(
        source_model,
        destination_model,
        checkpoint_iteration=args.checkpoint_iteration,
    )

    # Step 2: build config
    config = LongSplatConfig(
        source_path=str(args.source_path),
        model_path=str(destination_model),
        iterations=args.conversion_iterations,  # placeholder — not used for conversion only
        seed=0,
        backend_mode=args.backend_mode,
        convert_iteration=args.conversion_iterations,
        convert_prune_ratio=args.prune_ratio,
    )

    convert_cmd = build_convert_command(
        args.repo_root,
        config,
        str(args.backend_python),
    )

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "snapshot": snapshot,
        "command": list(convert_cmd),
        "conversion_iterations": args.conversion_iterations,
        "prune_ratio": args.prune_ratio,
    }

    if args.dry_run:
        print(f"[DRY-RUN] Would execute: {' '.join(convert_cmd)}")
        output_record.parent.mkdir(parents=True, exist_ok=True)
        output_record.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"Record written to {output_record}")
        return

    # Step 3: run conversion
    repo_root = args.repo_root.resolve()
    env = subprocess.os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (
        subprocess.os.pathsep + env.get("PYTHONPATH", "")
        if env.get("PYTHONPATH")
        else ""
    )

    result = subprocess.run(
        convert_cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=env,
    )

    record["conversion"] = {
        "returncode": result.returncode,
        "stdout_last_2000": result.stdout[-2000:] if result.stdout else "",
        "stderr_last_2000": result.stderr[-2000:] if result.stderr else "",
    }

    # Step 4: validate output PLY
    if result.returncode == 0:
        converted_ply = destination_model / "converted_3dgs" / "point_cloud.ply"
        if converted_ply.is_file():
            from scripts.longsplat.convert import validate_converted_ply

            try:
                validation = validate_converted_ply(converted_ply)
                record["validation"] = validation
            except Exception as exc:
                record["validation_error"] = str(exc)

            from scripts.longsplat.telemetry import summarize_conversion_telemetry

            telemetry = summarize_conversion_telemetry(result.stdout)
            record["conversion_telemetry"] = telemetry
        else:
            record["conversion_error"] = "converted PLY not found"

    output_record.parent.mkdir(parents=True, exist_ok=True)
    output_record.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(f"Record written to {output_record}")

    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
