"""
Run record lifecycle tests.

CPU-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.longsplat.run_record import (  # noqa: E402
    RunStatus,
    RunStatusTransitionError,
    add_artifact,
    create_run_record,
    transition_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest_file(tmp_path):
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps({"schema_version": 1, "segment_id": "test", "frames": []}))
    return mf


@pytest.fixture
def base_config():
    return {
        "source_path": "/tmp/images",
        "model_path": "/tmp/model",
        "iterations": 100,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# create_run_record
# ---------------------------------------------------------------------------


def test_create_run_record_writes_file(manifest_file, base_config, tmp_path):
    run_dir = tmp_path / "run_01"
    record = create_run_record(
        run_dir, manifest_file, base_config,
        repo_sha="a" * 40,
        backend_repo_url="https://github.com/NVlabs/LongSplat",
        backend_commit="b" * 40,
        python_exe="python",
        seed=42,
    )
    assert record["status"] == "planned"
    assert len(record["run_id"]) == 36  # UUID
    assert record["schema_version"] == 1
    assert (run_dir / "reconstruction_run.json").is_file()


def test_create_run_record_unique_ids(manifest_file, base_config, tmp_path):
    r1 = create_run_record(
        tmp_path / "a", manifest_file, base_config,
        repo_sha="a" * 40, backend_repo_url="x", backend_commit="b" * 40,
        python_exe="python", seed=0,
    )
    r2 = create_run_record(
        tmp_path / "b", manifest_file, base_config,
        repo_sha="a" * 40, backend_repo_url="x", backend_commit="b" * 40,
        python_exe="python", seed=0,
    )
    assert r1["run_id"] != r2["run_id"]


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


def test_valid_transition_planned_to_running(manifest_file, base_config, tmp_path):
    run_dir = tmp_path / "run"
    record = create_run_record(
        run_dir, manifest_file, base_config,
        repo_sha="a" * 40, backend_repo_url="x", backend_commit="b" * 40,
        python_exe="python", seed=0,
    )
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    assert record["status"] == "running"


def test_valid_transition_running_to_complete(manifest_file, base_config, tmp_path):
    run_dir = tmp_path / "run"
    record = create_run_record(
        run_dir, manifest_file, base_config,
        repo_sha="a" * 40, backend_repo_url="x", backend_commit="b" * 40,
        python_exe="python", seed=0,
    )
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    record = transition_status(
        record, run_dir, RunStatus.COMPLETE,
        stage="training",
        details={"exit_code": 0, "duration_s": 123.4},
    )
    assert record["status"] == "complete"
    assert record["stages"]["training"]["exit_code"] == 0


def test_valid_transition_running_to_failed(manifest_file, base_config, tmp_path):
    run_dir = tmp_path / "run"
    record = create_run_record(
        run_dir, manifest_file, base_config,
        repo_sha="a" * 40, backend_repo_url="x", backend_commit="b" * 40,
        python_exe="python", seed=0,
    )
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    record = transition_status(
        record, run_dir, RunStatus.FAILED,
        stage="training",
        details={"exit_code": 1, "error": "CUDA OOM"},
    )
    assert record["status"] == "failed"
    assert "CUDA OOM" in record["stages"]["training"]["error"]


def test_invalid_transition_raises(manifest_file, base_config, tmp_path):
    run_dir = tmp_path / "run"
    record = create_run_record(
        run_dir, manifest_file, base_config,
        repo_sha="a" * 40, backend_repo_url="x", backend_commit="b" * 40,
        python_exe="python", seed=0,
    )
    # planned → complete is not allowed
    with pytest.raises(RunStatusTransitionError):
        transition_status(record, run_dir, RunStatus.COMPLETE)

    # planned → failed is not allowed
    with pytest.raises(RunStatusTransitionError):
        transition_status(record, run_dir, RunStatus.FAILED)


def test_complete_record_not_mutable_to_running(manifest_file, base_config, tmp_path):
    run_dir = tmp_path / "run"
    record = create_run_record(
        run_dir, manifest_file, base_config,
        repo_sha="a" * 40, backend_repo_url="x", backend_commit="b" * 40,
        python_exe="python", seed=0,
    )
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    record = transition_status(record, run_dir, RunStatus.COMPLETE)
    with pytest.raises(RunStatusTransitionError):
        transition_status(record, run_dir, RunStatus.RUNNING)


# ---------------------------------------------------------------------------
# Artifact tracking
# ---------------------------------------------------------------------------


def test_add_artifact(manifest_file, base_config, tmp_path):
    run_dir = tmp_path / "run"
    record = create_run_record(
        run_dir, manifest_file, base_config,
        repo_sha="a" * 40, backend_repo_url="x", backend_commit="b" * 40,
        python_exe="python", seed=0,
    )
    # Create a file as artifact
    art = run_dir / "output.ply"
    art.write_text("fake ply content")
    record = add_artifact(
        record, run_dir,
        artifact_path=art,
        artifact_type="converted_ply",
        stage="conversion",
    )
    assert len(record["artifacts"]) == 1
    assert record["artifacts"][0]["type"] == "converted_ply"
    assert len(record["artifacts"][0]["sha256"]) == 64
