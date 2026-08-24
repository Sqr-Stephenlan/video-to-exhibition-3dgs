"""
Immutable run record for LongSplat reconstruction runs.

Every run gets a unique run ID and an isolated directory.  The run record
tracks all inputs, configuration, commands, backend versions, and output
artifacts.  It is updated at each lifecycle transition.

Schema v2 adds the ``requested`` section (recorded before preflight),
a richer ``backend`` block, and the ``commands`` section.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .runner import LONGSPLAT_COMMIT, LONGSPLAT_REPO_URL


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

RUN_RECORD_SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_run_record(
    *,
    run_dir: str | Path,
    run_id: str,
    manifest_path: str | Path,
    project_root: str | Path,
    repo_root: str | Path,
    depth_manifest_path: str | Path | None = None,
    backend_url: str = LONGSPLAT_REPO_URL,
    backend_expected_commit: str = LONGSPLAT_COMMIT,
) -> dict[str, Any]:
    """Create a new v2 run record in ``planned`` state.

    Call this **before** any fallible preflight check so every failure
    leaves a terminal record.  All fields that are not yet known are
    set to JSON ``null`` — never invented values.
    """
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    record: dict[str, Any] = {
        "schema_version": RUN_RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "status": RunStatus.PLANNED.value,
        "created_at": now,
        "updated_at": now,
        "requested": {
            "manifest_path": str(Path(manifest_path)),
            "depth_manifest_path": (
                None if depth_manifest_path is None else str(Path(depth_manifest_path))
            ),
            "project_root": str(Path(project_root)),
            "repo_root": str(Path(repo_root)),
        },
        "repo_sha": None,
        "backend": {
            "repo_url_expected": backend_url,
            "commit_expected": backend_expected_commit,
            "commit_actual": None,
            "dirty": None,
            "submodules": {},
            "mode": None,
            "diff_sha256": None,
            "submodule_diffs": {},
        },
        "commands": {},
        "artifacts": [],
        "stages": {},
    }

    write_run_record(record, rd)
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
    - planned → running | failed
    - running → running | failed | complete
    """
    current = RunStatus(record["status"])

    _valid = {
        RunStatus.PLANNED: {RunStatus.RUNNING, RunStatus.FAILED},
        RunStatus.RUNNING: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.COMPLETE},
    }
    allowed = _valid.get(current, set())
    if new_status not in allowed:
        raise RunStatusTransitionError(
            f"Cannot transition from {current.value} to {new_status.value}"
        )

    record["status"] = new_status.value
    record["updated_at"] = datetime.now(timezone.utc).isoformat()

    if stage and details:
        record["stages"][stage] = details

    write_run_record(record, Path(run_dir))
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
        "stage": stage,
        "sha256": sha,
        "file_size": ap.stat().st_size if ap.is_file() else None,
    }
    if metadata:
        entry["metadata"] = metadata

    record.setdefault("artifacts", []).append(entry)
    record["updated_at"] = datetime.now(timezone.utc).isoformat()

    write_run_record(record, Path(run_dir))
    return record


def write_run_record(record: dict[str, Any], run_dir: Path) -> None:
    """Atomically write the run record to *run_dir*."""
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


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RunStatusTransitionError(Exception):
    """Raised when an invalid status transition is attempted."""


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
