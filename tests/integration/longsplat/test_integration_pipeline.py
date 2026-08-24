"""
Cross-module integration tests (schema v2).

Chains validate_input → prepare_input → run_record across the full
reconstruction lifecycle.  CPU-only — no real GPU, backend, or checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.longsplat.validate_input import validate_manifest  # noqa: E402
from scripts.longsplat.prepare_input import prepare_input  # noqa: E402
from scripts.longsplat.run_record import (  # noqa: E402
    RunStatus,
    add_artifact,
    create_run_record,
    transition_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_data():
    """Create a temporary directory with a manifest and frame images."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        img_dir = base / "frames"
        img_dir.mkdir()
        fp = img_dir / "frame_001.jpg"
        fp.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 128)
        sha = _sha256_hex(fp)

        manifest = {
            "schema_version": 1,
            "segment_id": "integration_test_segment",
            "base": str(base),
            "frames": [
                {
                    "frame_id": 0,
                    "path": "frames/frame_001.jpg",
                    "width": 1920,
                    "height": 1080,
                    "sha256": sha,
                    "timestamp": 0.0,
                },
                {
                    "frame_id": 1,
                    "path": "frames/frame_001.jpg",
                    "width": 1920,
                    "height": 1080,
                    "sha256": sha,
                    "timestamp": 0.1,
                },
                {
                    "frame_id": 2,
                    "path": "frames/frame_001.jpg",
                    "width": 1920,
                    "height": 1080,
                    "sha256": sha,
                    "timestamp": 0.2,
                },
            ],
        }
        mf_path = base / "manifest.json"
        mf_path.write_text(json.dumps(manifest))
        yield base


# ---------------------------------------------------------------------------
# Full happy-path pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_happy_path(source_data):
    """validate → prepare → run_record lifecycle complete."""
    manifest_path = source_data / "manifest.json"

    manifest = validate_manifest(manifest_path)
    assert manifest["segment_id"] == "integration_test_segment"
    assert len(manifest["frames"]) == 3

    run_dir = source_data / "run_001"
    img_dir = prepare_input(manifest, run_dir)
    assert img_dir.is_dir()
    assert (img_dir / "frame_000000.jpg").is_file()
    mapping_path = run_dir / "input" / "frame_mapping.json"
    assert mapping_path.is_file()

    # V2: create record with preflight-known fields
    record = create_run_record(
        run_dir=run_dir,
        run_id="integration-test-001",
        manifest_path=manifest_path,
        project_root=source_data,
        repo_root=source_data,
    )
    assert record["status"] == "planned"
    assert record["schema_version"] == 2
    assert record["run_id"] == "integration-test-001"
    run_record_path = run_dir / "reconstruction_run.json"
    assert run_record_path.is_file()

    # Simulate orchestrator filling in post-preflight fields
    record["repo_sha"] = "a" * 40
    record["backend"]["commit_actual"] = "b" * 40
    record["backend"]["dirty"] = False
    record["backend"]["submodules"] = {}

    record = transition_status(record, run_dir, RunStatus.RUNNING)
    assert record["status"] == "running"

    record = transition_status(
        record,
        run_dir,
        RunStatus.COMPLETE,
        stage="training",
        details={"exit_code": 0, "duration_s": 300.0, "iterations": 100},
    )
    assert record["stages"]["training"]["exit_code"] == 0

    art = run_dir / "output.ply"
    art.write_text("synthetic ply content")
    record = add_artifact(
        record,
        run_dir,
        artifact_path=art,
        artifact_type="converted_ply",
        stage="conversion",
    )
    assert len(record["artifacts"]) == 1
    assert record["artifacts"][0]["sha256"] == _sha256_hex(art)

    with open(run_record_path) as fh:
        disk_record = json.load(fh)
    assert disk_record["status"] == "complete"
    assert len(disk_record["artifacts"]) == 1


# ---------------------------------------------------------------------------
# Error-path: training failure
# ---------------------------------------------------------------------------


def test_pipeline_training_failure(source_data):
    """validate → prepare → run_record → running → failed."""
    manifest_path = source_data / "manifest.json"
    manifest = validate_manifest(manifest_path)
    run_dir = source_data / "run_fail"
    prepare_input(manifest, run_dir)

    record = create_run_record(
        run_dir=run_dir,
        run_id="fail-test",
        manifest_path=manifest_path,
        project_root=source_data,
        repo_root=source_data,
    )
    record["repo_sha"] = "a" * 40
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    record = transition_status(
        record,
        run_dir,
        RunStatus.FAILED,
        stage="training",
        details={"exit_code": 1, "error": "CUDA OOM at iteration 500"},
    )

    assert record["status"] == "failed"
    assert "CUDA OOM" in record["stages"]["training"]["error"]

    with open(run_dir / "reconstruction_run.json") as fh:
        disk = json.load(fh)
    assert disk["status"] == "failed"
    assert disk["stages"]["training"]["exit_code"] == 1


# ---------------------------------------------------------------------------
# Multi-artifact tracking
# ---------------------------------------------------------------------------


def test_pipeline_multi_artifact(source_data):
    """Track multiple output artifacts."""
    manifest_path = source_data / "manifest.json"
    manifest = validate_manifest(manifest_path)
    run_dir = source_data / "run_multi"
    prepare_input(manifest, run_dir)

    record = create_run_record(
        run_dir=run_dir,
        run_id="multi-art",
        manifest_path=manifest_path,
        project_root=source_data,
        repo_root=source_data,
    )
    record["repo_sha"] = "a" * 40
    record = transition_status(record, run_dir, RunStatus.RUNNING)

    ckpt = run_dir / "checkpoint.pth"
    ckpt.write_text("model weights placeholder")
    record = add_artifact(
        record,
        run_dir,
        artifact_path=ckpt,
        artifact_type="training_checkpoint",
        stage="training",
    )

    ply = run_dir / "converted.ply"
    ply.write_text("ply placeholder")
    record = add_artifact(
        record,
        run_dir,
        artifact_path=ply,
        artifact_type="converted_ply",
        stage="conversion",
    )

    report = run_dir / "report.json"
    report.write_text(json.dumps({"summary": "ok"}))
    record = add_artifact(
        record,
        run_dir,
        artifact_path=report,
        artifact_type="summary_report",
        stage="reporting",
    )

    record = transition_status(record, run_dir, RunStatus.COMPLETE)
    assert len(record["artifacts"]) == 3
    types = [a["type"] for a in record["artifacts"]]
    assert types == ["training_checkpoint", "converted_ply", "summary_report"]


# ---------------------------------------------------------------------------
# Run record schema v2 completeness
# ---------------------------------------------------------------------------


def test_run_record_schema_completeness(source_data):
    """Every required field in the v2 run record must be populated."""
    manifest_path = source_data / "manifest.json"
    validate_manifest(manifest_path)
    run_dir = source_data / "run_schema"

    record = create_run_record(
        run_dir=run_dir,
        run_id="schema-test",
        manifest_path=manifest_path,
        project_root=source_data,
        repo_root=source_data,
    )

    # Top-level keys (v2)
    assert record["schema_version"] == 2
    assert record["run_id"] == "schema-test"
    assert record["status"] == "planned"
    assert record["repo_sha"] is None  # not yet known

    # Requested section
    assert "requested" in record
    assert record["requested"]["manifest_path"] == str(manifest_path)
    assert record["requested"]["project_root"] == str(source_data)

    # Backend identity (expectations filled, actuals null)
    assert record["backend"]["repo_url_expected"] is not None
    assert record["backend"]["commit_expected"] is not None
    assert record["backend"]["commit_actual"] is None
    assert record["backend"]["dirty"] is None
    assert isinstance(record["backend"]["submodules"], dict)

    # Commands (empty until built)
    assert record["commands"] == {}

    # Artifacts (empty list)
    assert record["artifacts"] == []

    # Stages (empty dict)
    assert record["stages"] == {}

    # Timestamps
    assert "created_at" in record
    assert "updated_at" in record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
