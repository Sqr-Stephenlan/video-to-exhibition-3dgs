"""
Single LongSplat orchestration entry point.

Executes the full pipeline in order:

  validate → prepare → create_record → training → conversion →
  PLY validation → artifact_recording → terminal_status

Any non-zero exit, exception, or missing product creates a failed record
and returns a non-zero exit code.  This is the single public entry point
for running LongSplat from this repository.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .convert import clean_converted_ply, validate_converted_ply
from .manifest_adapter import adapt_manifest
from .prepare_input import prepare_input
from .run_record import (
    RunStatus,
    add_artifact,
    create_run_record,
    transition_status,
)
from .runner import (
    BackendValidationError,
    LongSplatConfig,
    _check_repo,
    build_convert_command,
    build_train_command,
)
from .validate_input import ManifestValidationError, validate_manifest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    manifest_path: str | Path,
    segment_id: str,
    config: LongSplatConfig,
    repo_root: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    python_exe: str = "python",
) -> int:
    """Run the full LongSplat pipeline for one segment.

    Parameters
    ----------
    manifest_path : str or Path
        Path to a preprocess manifest (schema_version ``"1.0"``).
    segment_id : str
        Which segment from the manifest to process.
    config : LongSplatConfig
        Training and conversion configuration.
    repo_root : str or Path
        Root of the locked LongSplat repository.
    output_dir : str or Path
        Parent directory for run-scoped output.  A timestamped run directory
        is created inside it.
    run_id : str or None
        Custom run identifier.  Auto-generated when omitted.
    python_exe : str
        Path to the Python interpreter for the LongSplat backend.

    Returns
    -------
    int
        0 on success, non-zero on failure.
    """
    failed = False
    run_dir: Path | None = None
    record: dict[str, Any] | None = None

    try:
        # 1. Validate the producer manifest and adapt to consumer format
        _echo("Validating producer manifest ...")
        with open(manifest_path, encoding="utf-8") as fh:
            producer = json.load(fh)
        consumer = adapt_manifest(producer, segment_id, Path.cwd())

        # 2. Create run directory
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        r_id = run_id or f"longsplat_{segment_id}_{ts}"
        run_dir = out / r_id
        run_dir.mkdir(parents=True)
        _echo(f"Run directory: {run_dir}")

        # Write adapted manifest for provenance (needed by create_run_record)
        consumer_manifest_path = run_dir / "consumer_manifest.json"
        consumer_manifest_path.write_text(
            json.dumps(consumer, indent=2) + "\n", encoding="utf-8"
        )

        # 3. Create run record
        _echo("Creating run record ...")
        repo_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        from .runner import LONGSPLAT_REPO_URL, LONGSPLAT_COMMIT

        record = create_run_record(
            run_dir=run_dir,
            manifest_path=consumer_manifest_path,
            config=config.__dict__,
            repo_sha=repo_sha,
            backend_repo_url=LONGSPLAT_REPO_URL,
            backend_commit=LONGSPLAT_COMMIT,
            python_exe=python_exe,
            seed=config.seed,
        )

        # 4. Prepare input
        _echo("Preparing input ...")
        prepare_input(consumer, run_dir)

        # Update config paths for training
        config.source_path = str(run_dir / "input")
        config.model_path = str(run_dir / "longsplat_model")

        # 5. Run training
        _echo("Running training ...")
        transition_status(
            record, run_dir, RunStatus.RUNNING,
            stage="training", details={"status": "started"},
        )
        train_start = time.monotonic()
        train_result = subprocess.run(
            build_train_command(repo_root, config, python_exe),
            capture_output=True, text=True,
            env=_subprocess_env(python_exe, Path(repo_root)),
        )
        train_duration = time.monotonic() - train_start

        if train_result.returncode != 0:
            transition_status(
                record, run_dir, RunStatus.FAILED,
                stage="training",
                details={
                    "status": "failed",
                    "exit_code": train_result.returncode,
                    "duration_s": round(train_duration, 1),
                    "stderr": train_result.stderr[-2000:],
                },
            )
            raise TrainingFailed(
                f"Training exited with code {train_result.returncode}",
                returncode=train_result.returncode,
                stdout=train_result.stdout,
                stderr=train_result.stderr,
            )

        _echo(f"Training completed in {train_duration:.1f}s")
        transition_status(
            record, run_dir, RunStatus.RUNNING,
            stage="training",
            details={
                "status": "completed",
                "exit_code": 0,
                "duration_s": round(train_duration, 1),
            },
        )
        add_artifact(
            record, run_dir,
            artifact_path=str(run_dir / "longsplat_model"),
            artifact_type="longsplat_model",
            stage="training",
            metadata={"iteration": config.iterations},
        )

        # 6. Run conversion
        _echo("Running conversion ...")
        # LongSplat's convert_3dgs expects cameras_all.json (without suffix)
        # when eval=False, but training outputs cameras_all_train.json.
        # Copy it so the converter can find it.
        cameras_src = Path(config.model_path) / "cameras_all_train.json"
        cameras_dst = Path(config.model_path) / "cameras_all.json"
        if cameras_src.exists() and not cameras_dst.exists():
            shutil.copyfile(cameras_src, cameras_dst)
        transition_status(
            record, run_dir, RunStatus.RUNNING,
            stage="conversion", details={"status": "started"},
        )
        conv_start = time.monotonic()
        conv_result = subprocess.run(
            build_convert_command(repo_root, config, python_exe),
            capture_output=True, text=True,
            env=_subprocess_env(python_exe, Path(repo_root)),
        )
        conv_duration = time.monotonic() - conv_start

        if conv_result.returncode != 0:
            transition_status(
                record, run_dir, RunStatus.FAILED,
                stage="conversion",
                details={
                    "status": "failed",
                    "exit_code": conv_result.returncode,
                    "duration_s": round(conv_duration, 1),
                    "stderr": conv_result.stderr[-2000:],
                },
            )
            raise ConversionFailed(
                f"Conversion exited with code {conv_result.returncode}",
                returncode=conv_result.returncode,
                stdout=conv_result.stdout,
                stderr=conv_result.stderr,
            )

        _echo(f"Conversion completed in {conv_duration:.1f}s")
        transition_status(
            record, run_dir, RunStatus.RUNNING,
            stage="conversion",
            details={
                "status": "completed",
                "exit_code": 0,
                "duration_s": round(conv_duration, 1),
            },
        )

        # 7. Clean and validate converted PLY
        converted_ply = (
            Path(config.model_path) / "converted_3dgs" / "point_cloud.ply"
        )
        removed = clean_converted_ply(converted_ply)
        if removed:
            _echo(f"Cleaned {removed} NaN/Inf vertices from PLY")
        _echo("Validating PLY ...")
        try:
            meta = validate_converted_ply(converted_ply)
            _echo(f"PLY valid: {meta['vertex_count']} vertices, sha256={meta['sha256'][:12]}...")
        except Exception as exc:
            transition_status(
                record, run_dir, RunStatus.FAILED,
                stage="ply_validation",
                details={"status": "failed", "error": str(exc)},
            )
            raise PLYValidationFailed(str(exc)) from exc

        add_artifact(
            record, run_dir,
            artifact_path=str(converted_ply),
            artifact_type="ply",
            stage="ply_validation",
            metadata={
                "iteration": config.convert_iteration,
                "prune_ratio": config.convert_prune_ratio,
                "vertex_count": meta["vertex_count"],
                "sha256": meta["sha256"],
            },
        )

        # 8. Final status
        transition_status(record, run_dir, RunStatus.COMPLETE)
        _echo("Pipeline complete.")
        return 0

    except Exception as exc:
        failed = True
        _echo(f"FAILED: {exc}", file=sys.stderr)
        return 1

    finally:
        if failed and run_dir is not None and run_dir.exists():
            _echo(f"Run directory preserved for debugging: {run_dir}")


# ---------------------------------------------------------------------------
# Pipeline-specific errors
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Base for pipeline-stage errors."""


class TrainingFailed(PipelineError):
    def __init__(self, msg: str, *, returncode: int, stdout: str, stderr: str):
        super().__init__(msg)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ConversionFailed(PipelineError):
    def __init__(self, msg: str, *, returncode: int, stdout: str, stderr: str):
        super().__init__(msg)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class PLYValidationFailed(PipelineError):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subprocess_env(python_exe: str, repo_root: Path) -> dict[str, str]:
    """Build environment for LongSplat subprocess calls."""
    import os as _os
    env = _os.environ.copy()
    mast3r_path = str((repo_root / "submodules" / "mast3r").resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = mast3r_path + (_os.pathsep + existing if existing else "")
    return env


def _echo(msg: str, file=sys.stdout) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=file, flush=True)
