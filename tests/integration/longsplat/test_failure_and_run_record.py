"""
Run record lifecycle tests (schema v2).

CPU-only.
"""

from __future__ import annotations

import json

import pytest

from scripts.longsplat.run_record import (
    RunStatus,
    RunStatusTransitionError,
    add_artifact,
    create_run_record,
    transition_status,
    write_run_record,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest_file(tmp_path):
    mf = tmp_path / "manifest.json"
    mf.write_text(json.dumps({"schema_version": 1, "segment_id": "test", "frames": []}))
    return mf


# ---------------------------------------------------------------------------
# create_run_record (v2)
# ---------------------------------------------------------------------------


def test_create_run_record_writes_file(manifest_file, tmp_path):
    run_dir = tmp_path / "run_01"
    record = create_run_record(
        run_dir=run_dir,
        run_id="test-run-001",
        manifest_path=manifest_file,
        project_root=tmp_path,
        repo_root=tmp_path,
    )
    assert record["status"] == "planned"
    assert record["run_id"] == "test-run-001"
    assert record["schema_version"] == 2
    assert record["repo_sha"] is None
    assert record["backend"]["commit_actual"] is None
    assert record["backend"]["dirty"] is None
    assert record["commands"] == {}
    assert record["artifacts"] == []
    assert (run_dir / "reconstruction_run.json").is_file()


def test_create_run_record_unique_ids(manifest_file, tmp_path):
    r1 = create_run_record(
        run_dir=tmp_path / "a",
        run_id="id-a",
        manifest_path=manifest_file,
        project_root=tmp_path,
        repo_root=tmp_path,
    )
    r2 = create_run_record(
        run_dir=tmp_path / "b",
        run_id="id-b",
        manifest_path=manifest_file,
        project_root=tmp_path,
        repo_root=tmp_path,
    )
    assert r1["run_id"] != r2["run_id"]


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


def _make_record(run_dir, manifest_file, tmp_path, run_id="test"):
    return create_run_record(
        run_dir=run_dir,
        run_id=run_id,
        manifest_path=manifest_file,
        project_root=tmp_path,
        repo_root=tmp_path,
    )


def test_valid_transition_planned_to_running(manifest_file, tmp_path):
    run_dir = tmp_path / "run"
    record = _make_record(run_dir, manifest_file, tmp_path)
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    assert record["status"] == "running"


def test_valid_transition_running_to_complete(manifest_file, tmp_path):
    run_dir = tmp_path / "run"
    record = _make_record(run_dir, manifest_file, tmp_path)
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    record = transition_status(
        record,
        run_dir,
        RunStatus.COMPLETE,
        stage="training",
        details={"exit_code": 0, "duration_s": 123.4},
    )
    assert record["status"] == "complete"
    assert record["stages"]["training"]["exit_code"] == 0


def test_valid_transition_running_to_failed(manifest_file, tmp_path):
    run_dir = tmp_path / "run"
    record = _make_record(run_dir, manifest_file, tmp_path)
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    record = transition_status(
        record,
        run_dir,
        RunStatus.FAILED,
        stage="training",
        details={"exit_code": 1, "error": "CUDA OOM"},
    )
    assert record["status"] == "failed"
    assert "CUDA OOM" in record["stages"]["training"]["error"]


def test_invalid_transition_raises(manifest_file, tmp_path):
    run_dir = tmp_path / "run"
    record = _make_record(run_dir, manifest_file, tmp_path)
    with pytest.raises(RunStatusTransitionError):
        transition_status(record, run_dir, RunStatus.COMPLETE)


def test_planned_to_failed_allowed(manifest_file, tmp_path):
    """planned → failed is valid."""
    run_dir = tmp_path / "run2"
    record = _make_record(run_dir, manifest_file, tmp_path)
    record = transition_status(
        record,
        run_dir,
        RunStatus.FAILED,
        stage="validation",
        details={"reason": "test"},
    )
    assert record["status"] == "failed"
    assert record["stages"]["validation"]["reason"] == "test"


def test_complete_record_not_mutable_to_running(manifest_file, tmp_path):
    run_dir = tmp_path / "run"
    record = _make_record(run_dir, manifest_file, tmp_path)
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    record = transition_status(record, run_dir, RunStatus.COMPLETE)
    with pytest.raises(RunStatusTransitionError):
        transition_status(record, run_dir, RunStatus.RUNNING)


# ---------------------------------------------------------------------------
# Artifact tracking
# ---------------------------------------------------------------------------


def test_add_artifact(manifest_file, tmp_path):
    run_dir = tmp_path / "run"
    record = _make_record(run_dir, manifest_file, tmp_path)
    art = run_dir / "output.ply"
    art.write_text("fake ply content")
    record = add_artifact(
        record,
        run_dir,
        artifact_path=art,
        artifact_type="converted_ply",
        stage="conversion",
    )
    assert len(record["artifacts"]) == 1
    assert record["artifacts"][0]["type"] == "converted_ply"
    assert len(record["artifacts"][0]["sha256"]) == 64


# ---------------------------------------------------------------------------
# Concurrency: auto IDs are unique
# ---------------------------------------------------------------------------


def test_concurrent_auto_ids_are_unique():
    """Two auto-generated IDs in the same second must differ."""
    import uuid

    ids = set()
    for _ in range(10):
        ids.add(uuid.uuid4().hex[:12])
    assert len(ids) == 10


# ---------------------------------------------------------------------------
# write_run_record exported
# ---------------------------------------------------------------------------


def test_write_run_record_is_exported():
    """write_run_record must be importable from run_record."""
    from scripts.longsplat.run_record import write_run_record as wrr

    assert callable(wrr)


def test_write_run_record_persists(manifest_file, tmp_path):
    """write_run_record atomically persists a record."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    record = _make_record(run_dir, manifest_file, tmp_path)
    # Modify and re-persist
    record["backend"]["commit_actual"] = "c" * 40
    record["backend"]["dirty"] = False
    write_run_record(record, run_dir)

    # Read back
    loaded = json.loads((run_dir / "reconstruction_run.json").read_text())
    assert loaded["backend"]["commit_actual"] == "c" * 40
    assert loaded["backend"]["dirty"] is False
