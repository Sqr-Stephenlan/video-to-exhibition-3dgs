"""
Immutable run record for LongSplat reconstruction runs.

Every run gets a unique run ID and an isolated directory.  The run record
tracks all inputs, configuration, commands, backend versions, and output
artifacts.  It is updated at each lifecycle transition.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Lifecycle states
# ---------------------------------------------------------------------------


class RunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETE = "complete"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

RUN_RECORD_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_run_record(
    run_dir: str | Path,
    manifest_path: str | Path,
    config: dict[str, Any],
    repo_sha: str,
    backend_repo_url: str,
    backend_commit: str,
    python_exe: str,
    seed: int,
    *,
    runtime_profile: str = "unvalidated",
) -> dict[str, Any]:
    """Create a new run record in ``planned`` state.

    Parameters
    ----------
    run_dir : str or Path
        Immutable run directory (created if it does not exist).
    manifest_path : str or Path
        Path to the input frame manifest consumed by this run.
    config : dict
        Effective configuration (LongSplatConfig serialised to dict).
    repo_sha : str
        Full 40-char SHA of the main repository at run time.
    backend_repo_url : str
        URL of the backend repository (e.g. LongSplat).
    backend_commit : str
        Full 40-char SHA of the backend repository.
    python_exe : str
        Python executable used for the backend command.
    seed : int
        Random seed.
    runtime_profile : str
        Label for the runtime environment (default: "unvalidated").

    Returns
    -------
    dict
        The run record dict (also written to ``<run_dir>/reconstruction_run.json``).
    """
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    manifest_sha = _sha256_hex(Path(manifest_path))

    record: dict[str, Any] = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "status": RunStatus.PLANNED.value,
        "created_at": now,
        "updated_at": now,
        "repo_sha": repo_sha,
        "backend": {
            "repo_url": backend_repo_url,
            "commit": backend_commit,
        },
        "runtime": {
            "python_exe": python_exe,
            "profile": runtime_profile,
        },
        "input": {
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": manifest_sha,
        },
        "config": config,
        "seed": seed,
        "stages": {},
    }

    _write_record(record, rd)
    return record


def transition_status(
    record: dict[str, Any],
    run_dir: str | Path,
    new_status: RunStatus,
    *,
    stage: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Transition the run record to a new status.

    Valid transitions:
    - planned → running
    - running → failed | complete

    Parameters
    ----------
    record : dict
        Existing run record.
    run_dir : str or Path
        Run directory.
    new_status : RunStatus
        Target status.
    stage : str or None
        Current stage name (e.g. "training", "conversion").
    details : dict or None
        Additional per-stage data (exit code, errors, artifacts).
    """
    current = RunStatus(record["status"])

    _valid = {
        RunStatus.PLANNED: {RunStatus.RUNNING},
        RunStatus.RUNNING: {RunStatus.FAILED, RunStatus.COMPLETE},
    }
    allowed = _valid.get(current, set())
    if new_status not in allowed:
        raise RunStatusTransitionError(
            f"Cannot transition from {current.value} to {new_status.value}"
        )

    record["status"] = new_status.value
    record["updated_at"] = datetime.now(UTC).isoformat()

    if stage and details:
        record["stages"][stage] = details

    _write_record(record, Path(run_dir))
    return record


def add_artifact(
    record: dict[str, Any],
    run_dir: str | Path,
    *,
    artifact_path: str | Path,
    artifact_type: str,
    stage: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an output artifact with its SHA-256."""
    ap = Path(artifact_path)
    sha = _sha256_hex(ap) if ap.is_file() else None

    entry: dict[str, Any] = {
        "path": str(ap.resolve()),
        "type": artifact_type,
        "sha256": sha,
        "file_size": ap.stat().st_size if ap.is_file() else None,
    }
    if metadata:
        entry["metadata"] = metadata

    record.setdefault("artifacts", []).append(entry)
    record["updated_at"] = datetime.now(UTC).isoformat()

    _write_record(record, Path(run_dir))
    return record


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RunStatusTransitionError(Exception):
    """Raised when an invalid status transition is attempted."""


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _write_record(record: dict[str, Any], run_dir: Path) -> None:
    path = run_dir / "reconstruction_run.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
