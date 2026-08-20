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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.longsplat.backend_identity import (
    BackendIdentityError,
    resolve_backend_identity,
)
from scripts.longsplat.authority_manifest import (
    AuthorityManifestError,
    load_authority_manifest,
)
from scripts.longsplat.runner import (
    LONGSPLAT_COMMIT,
    BackendValidationError,
    LongSplatConfig,
    build_convert_command,
    run_conversion,
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

    # --- Copy (remove a partial snapshot on any failure) ---
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
    parser.add_argument("--authority-manifest", type=Path)
    parser.add_argument("--source-model", type=Path)
    parser.add_argument("--destination-model", required=True, type=Path)
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--backend-python", required=True, type=Path)
    parser.add_argument("--checkpoint-iteration", type=int)
    parser.add_argument("--conversion-iterations", type=int)
    parser.add_argument("--prune-ratio", type=float)
    parser.add_argument("--anisotropy-reg-weight", default=0.01, type=float)
    parser.add_argument("--anisotropy-soft-limit", default=30.0, type=float)
    parser.add_argument("--backend-mode", default="research_local", type=str)
    parser.add_argument("--output-record", required=True, type=Path)
    parser.add_argument(
        "--precreated-snapshot",
        action="store_true",
        help="Use an already verified destination snapshot without copying it",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _argparser()
    args = parser.parse_args(argv)
    authority: dict[str, Any] | None = None
    if args.authority_manifest is not None:
        try:
            authority = load_authority_manifest(args.authority_manifest)
        except AuthorityManifestError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)
        profile = authority["profile"]
        manifest = authority["manifest"]
        source_model = Path(str(manifest["training_model"]["path"])).resolve()
        source_path = Path(str(manifest["training_input"]["path"])).resolve()
        supplied = {
            "checkpoint_iteration": args.checkpoint_iteration,
            "conversion_iterations": args.conversion_iterations,
            "prune_ratio": args.prune_ratio,
            "anisotropy_reg_weight": args.anisotropy_reg_weight,
            "anisotropy_soft_limit": args.anisotropy_soft_limit,
            "backend_mode": args.backend_mode,
        }
        for key, value in supplied.items():
            if value is not None and value != profile[key]:
                print(f"ERROR: authority profile rejects override {key}={value!r}", file=sys.stderr)
                sys.exit(2)
        if args.source_model is not None and args.source_model.resolve() != source_model:
            print("ERROR: --source-model differs from authority manifest", file=sys.stderr)
            sys.exit(2)
        if args.source_path is not None and args.source_path.resolve() != source_path:
            print("ERROR: --source-path differs from authority manifest", file=sys.stderr)
            sys.exit(2)
        checkpoint_iteration = int(profile["checkpoint_iteration"])
        conversion_iterations = int(profile["conversion_iterations"])
        prune_ratio = float(profile["prune_ratio"])
        anisotropy_reg_weight = float(profile["anisotropy_reg_weight"])
        anisotropy_soft_limit = float(profile["anisotropy_soft_limit"])
        backend_mode = str(profile["backend_mode"])
    else:
        required = {
            "source-model": args.source_model,
            "source-path": args.source_path,
            "checkpoint-iteration": args.checkpoint_iteration,
            "conversion-iterations": args.conversion_iterations,
            "prune-ratio": args.prune_ratio,
            "anisotropy-reg-weight": args.anisotropy_reg_weight,
            "anisotropy-soft-limit": args.anisotropy_soft_limit,
            "backend-mode": args.backend_mode,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("missing required arguments without --authority-manifest: " + ", ".join(missing))
        source_model = args.source_model.resolve()
        source_path = args.source_path.resolve()
        checkpoint_iteration = int(args.checkpoint_iteration)
        conversion_iterations = int(args.conversion_iterations)
        prune_ratio = float(args.prune_ratio)
        anisotropy_reg_weight = float(args.anisotropy_reg_weight)
        anisotropy_soft_limit = float(args.anisotropy_soft_limit)
        backend_mode = str(args.backend_mode)
    destination_model = args.destination_model.resolve()
    output_record = args.output_record.resolve()

    if output_record.exists():
        print(f"ERROR: output record already exists: {output_record}", file=sys.stderr)
        sys.exit(1)

    # Step 1: create or bind the snapshot.  The frozen conversion executor
    # creates and hashes its stricter allowlist before invoking this wrapper;
    # this flag prevents a second copy and keeps the executor's snapshot
    # manifest authoritative.
    if args.precreated_snapshot:
        if not destination_model.is_dir() or destination_model.is_symlink():
            print(
                f"ERROR: precreated snapshot is missing or symlinked: {destination_model}",
                file=sys.stderr,
            )
            sys.exit(1)
        snapshot = {
            "source_model": str(source_model),
            "destination_model": str(destination_model),
            "checkpoint_iteration": checkpoint_iteration,
            "precreated": True,
        }
    else:
        snapshot = create_conversion_snapshot(
            source_model,
            destination_model,
            checkpoint_iteration=checkpoint_iteration,
        )

    # Step 2: build config
    config = LongSplatConfig(
        source_path=str(source_path),
        model_path=str(destination_model),
        iterations=conversion_iterations,  # placeholder — not used for conversion only
        seed=0,
        backend_mode=backend_mode,
        convert_iteration=conversion_iterations,
        convert_prune_ratio=prune_ratio,
        convert_anisotropy_reg_weight=anisotropy_reg_weight,
        convert_anisotropy_soft_limit=anisotropy_soft_limit,
    )

    repo_root_resolved = args.repo_root.resolve()
    convert_cmd = build_convert_command(
        repo_root_resolved,
        config,
        str(args.backend_python),
    )

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "dry_run": args.dry_run,
        "snapshot": snapshot,
        "command": list(convert_cmd),
        "conversion_profile_id": None if authority is None else authority["profile_id"],
        "authority_manifest": None if authority is None else authority["manifest_path"],
        "conversion_iterations": conversion_iterations,
        "prune_ratio": prune_ratio,
        "anisotropy_regularization": {
            "weight": anisotropy_reg_weight,
            "soft_limit": anisotropy_soft_limit,
        },
        "backend": {
            "repo_root": str(repo_root_resolved),
            "mode": config.backend_mode,
            "commit_expected": LONGSPLAT_COMMIT,
        },
    }

    if args.dry_run:
        record["status"] = "dry_run"
        print(f"[DRY-RUN] Would execute: {' '.join(convert_cmd)}")
        output_record.parent.mkdir(parents=True, exist_ok=True)
        output_record.write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
        print(f"Record written to {output_record}")
        return

    # Step 3: run conversion through the validated runner so the locked commit,
    # recursive submodule SHAs, backend mode, and interpreter are all checked.
    try:
        backend_identity = resolve_backend_identity(
            repo_root_resolved,
            backend_mode=config.backend_mode,
        )
        record["backend"].update(
            {
                "commit_actual": backend_identity["commit"],
                "dirty": backend_identity["dirty"],
                "submodules": backend_identity["submodules"],
                "diff_sha256": backend_identity["diff_sha256"],
                "submodule_diffs": backend_identity["submodule_diffs"],
                "python_exe_requested": str(args.backend_python),
                "python_exe_resolved": shutil.which(str(args.backend_python)),
            }
        )
        result = run_conversion(
            repo_root_resolved,
            config,
            str(args.backend_python),
        )
    except (BackendIdentityError, BackendValidationError, OSError) as exc:
        record["status"] = "failed"
        record["conversion_error"] = f"{type(exc).__name__}: {exc}"
        output_record.parent.mkdir(parents=True, exist_ok=True)
        output_record.write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Record written to {output_record}")
        sys.exit(1)

    record["conversion"] = {
        "returncode": result.returncode,
        "stdout_last_2000": result.stdout[-2000:] if result.stdout else "",
        "stderr_last_2000": result.stderr[-2000:] if result.stderr else "",
    }

    # Step 4: validate output PLY
    failure_code = result.returncode
    if result.returncode == 0:
        converted_ply = destination_model / "converted_3dgs" / "point_cloud.ply"
        if converted_ply.is_file():
            from scripts.longsplat.convert import validate_converted_ply

            try:
                validation = validate_converted_ply(converted_ply)
                record["validation"] = validation
            except Exception as exc:
                record["validation_error"] = str(exc)
                failure_code = 1

            from scripts.longsplat.telemetry import summarize_conversion_telemetry

            telemetry = summarize_conversion_telemetry(result.stdout)
            record["conversion_telemetry"] = telemetry
        else:
            record["conversion_error"] = "converted PLY not found"
            failure_code = 1

    record["status"] = "complete" if failure_code == 0 else "failed"

    output_record.parent.mkdir(parents=True, exist_ok=True)
    output_record.write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    print(f"Record written to {output_record}")

    if failure_code != 0:
        sys.exit(failure_code)


if __name__ == "__main__":
    main()
