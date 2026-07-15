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

from .convert import convert_and_validate
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

    try:
        # 1. Validate the producer manifest and adapt to consumer format
        _echo("Validating producer manifest ...")
        with open(manifest_path, encoding="utf-8") as fh:
            producer = json.load(fh)
        consumer = adapt_manifest(producer, segment_id, Path.cwd())

        # 2. Create run directory (fail if exists)
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        r_id = run_id or f"longsplat_{segment_id}_{ts}"
        run_dir = out / r_id
        run_dir.mkdir(parents=True)
        _echo(f"Run directory: {run_dir}")

        # Write adapted manifest for provenance
        consumer_manifest_path = run_dir / "consumer_manifest.json"
        consumer_manifest_path.write_text(
            json.dumps(consumer, indent=2) + "\n", encoding="utf-8"
        )

        # 3. Create run record
        _echo("Creating run record ...")
        recon_record = create_run_record(
            run_dir=run_dir,
            run_id=r_id,
            config=config,
        )
        record_path = recon_record["record_path"]

        # 4. Prepare input
        _echo("Preparing input ...")
        input_dir = Path(prepare_input(consumer, run_dir))

        # Update config source_path to point to the prepared input's parent
        config.source_path = str(run_dir / "input")
        config.model_path = str(run_dir / "longsplat_model")

        # 5. Run training
        _echo("Running training ...")
        transition_status(record_path, RunStatus.RUNNING)
        train_start = time.monotonic()
        train_result = subprocess.run(
            build_train_command(repo_root, config, python_exe),
            capture_output=True, text=True,
        )
        train_duration = time.monotonic() - train_start

        if train_result.returncode != 0:
            raise TrainingFailed(
                f"Training exited with code {train_result.returncode}",
                returncode=train_result.returncode,
                stdout=train_result.stdout,
                stderr=train_result.stderr,
            )

        _echo(f"Training completed in {train_duration:.1f}s")
        add_artifact(
            record_path,
            str(run_dir / "longsplat_model"),
            {"type": "longsplat_model", "iteration": config.iterations},
        )

        # 6. Run conversion
        _echo("Running conversion ...")
        conv_start = time.monotonic()
        conv_result = subprocess.run(
            build_convert_command(repo_root, config, python_exe),
            capture_output=True, text=True,
        )
        conv_duration = time.monotonic() - conv_start

        if conv_result.returncode != 0:
            raise ConversionFailed(
                f"Conversion exited with code {conv_result.returncode}",
                returncode=conv_result.returncode,
                stdout=conv_result.stdout,
                stderr=conv_result.stderr,
            )

        _echo(f"Conversion completed in {conv_duration:.1f}s")

        # 7. Validate PLY
        ply_path = run_dir / "longsplat_model" / f"point_cloud/iteration_{config.convert_iteration}/point_cloud.ply"
        _echo("Validating PLY ...")
        try:
            convert_and_validate(
                str(repo_root),
                config,
                python_exe=python_exe,
            )
        except Exception as exc:
            raise PLYValidationFailed(str(exc)) from exc

        add_artifact(
            record_path,
            str(ply_path),
            {
                "type": "ply",
                "iteration": config.convert_iteration,
                "prune_ratio": config.convert_prune_ratio,
            },
        )

        # 8. Final status
        transition_status(record_path, RunStatus.COMPLETE)
        _echo("Pipeline complete.")
        return 0

    except Exception as exc:
        failed = True
        _echo(f"FAILED: {exc}", file=sys.stderr)

        if run_dir is not None and run_dir.exists():
            record_files = list(run_dir.glob("reconstruction_run_*.json"))
            if record_files:
                try:
                    transition_status(record_files[0], RunStatus.FAILED)
                except Exception:
                    pass

        return 1

    finally:
        if failed and run_dir is not None and run_dir.exists():
            # Leave the run directory for forensic analysis
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


def _echo(msg: str, file=sys.stdout) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=file, flush=True)
