"""
Cross-module integration tests.

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


@pytest.fixture
def base_config():
    return {
        "source_path": "/tmp/images",
        "model_path": "/tmp/model",
        "iterations": 100,
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Full happy-path pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_happy_path(source_data, base_config):
    """validate → prepare → run_record lifecycle complete."""
    manifest_path = str(source_data / "manifest.json")

    # Phase 1: validate
    manifest = validate_manifest(manifest_path)
    assert manifest["segment_id"] == "integration_test_segment"
    assert len(manifest["frames"]) == 3

    # Phase 2: prepare input
    run_dir = source_data / "run_001"
    img_dir = prepare_input(manifest, run_dir)
    assert img_dir.is_dir()
    assert (img_dir / "frame_000000.jpg").is_file()
    mapping_path = run_dir / "input" / "frame_mapping.json"
    assert mapping_path.is_file()

    # Phase 3: create run record
    record = create_run_record(
        run_dir, manifest_path, base_config,
        repo_sha="a" * 40,
        backend_repo_url="https://github.com/NVlabs/LongSplat",
        backend_commit="b" * 40,
        python_exe="python",
        seed=42,
    )
    assert record["status"] == "planned"
    run_record_path = run_dir / "reconstruction_run.json"
    assert run_record_path.is_file()

    # Phase 4: transition to running
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    assert record["status"] == "running"

    # Phase 5: training succeeds
    record = transition_status(
        record, run_dir, RunStatus.COMPLETE,
        stage="training",
        details={"exit_code": 0, "duration_s": 300.0, "iterations": 100},
    )
    assert record["stages"]["training"]["exit_code"] == 0
    assert record["stages"]["training"]["iterations"] == 100

    # Phase 6: record artifacts
    art = run_dir / "output.ply"
    art.write_text("synthetic ply content")
    record = add_artifact(
        record, run_dir,
        artifact_path=art,
        artifact_type="converted_ply",
        stage="conversion",
    )
    assert len(record["artifacts"]) == 1
    assert record["artifacts"][0]["sha256"] == _sha256_hex(art)

    # Verify on-disk record is up-to-date
    with open(run_record_path) as fh:
        disk_record = json.load(fh)
    assert disk_record["status"] == "complete"
    assert len(disk_record["artifacts"]) == 1


# ---------------------------------------------------------------------------
# Error-path pipeline: training failure
# ---------------------------------------------------------------------------


def test_pipeline_training_failure(source_data, base_config):
    """validate → prepare → run_record → running → failed."""
    manifest_path = str(source_data / "manifest.json")
    manifest = validate_manifest(manifest_path)
    run_dir = source_data / "run_fail"
    prepare_input(manifest, run_dir)

    record = create_run_record(
        run_dir, manifest_path, base_config,
        repo_sha="a" * 40,
        backend_repo_url="x",
        backend_commit="b" * 40,
        python_exe="python", seed=42,
    )
    record = transition_status(record, run_dir, RunStatus.RUNNING)
    record = transition_status(
        record, run_dir, RunStatus.FAILED,
        stage="training",
        details={"exit_code": 1, "error": "CUDA OOM at iteration 500"},
    )

    assert record["status"] == "failed"
    assert "CUDA OOM" in record["stages"]["training"]["error"]

    # Verify on-disk
    with open(run_dir / "reconstruction_run.json") as fh:
        disk = json.load(fh)
    assert disk["status"] == "failed"
    assert disk["stages"]["training"]["exit_code"] == 1


# ---------------------------------------------------------------------------
# Multi-artifact tracking
# ---------------------------------------------------------------------------


def test_pipeline_multi_artifact(source_data, base_config):
    """Track multiple output artifacts (checkpoint + PLY + report)."""
    manifest_path = str(source_data / "manifest.json")
    manifest = validate_manifest(manifest_path)
    run_dir = source_data / "run_multi"
    prepare_input(manifest, run_dir)

    record = create_run_record(
        run_dir, manifest_path, base_config,
        repo_sha="a" * 40,
        backend_repo_url="x",
        backend_commit="b" * 40,
        python_exe="python", seed=42,
    )
    record = transition_status(record, run_dir, RunStatus.RUNNING)

    # Training checkpoint
    ckpt = run_dir / "checkpoint.pth"
    ckpt.write_text("model weights placeholder")
    record = add_artifact(record, run_dir, artifact_path=ckpt,
                          artifact_type="training_checkpoint", stage="training")

    # Converted PLY
    ply = run_dir / "converted.ply"
    ply.write_text("ply placeholder")
    record = add_artifact(record, run_dir, artifact_path=ply,
                          artifact_type="converted_ply", stage="conversion")

    # Summary report
    report = run_dir / "report.json"
    report.write_text(json.dumps({"summary": "ok"}))
    record = add_artifact(record, run_dir, artifact_path=report,
                          artifact_type="summary_report", stage="reporting")

    record = transition_status(record, run_dir, RunStatus.COMPLETE)

    assert len(record["artifacts"]) == 3
    types = [a["type"] for a in record["artifacts"]]
    assert types == ["training_checkpoint", "converted_ply", "summary_report"]


# ---------------------------------------------------------------------------
# Run record completeness: verify all required fields present
# ---------------------------------------------------------------------------


def test_run_record_schema_completeness(source_data, base_config):
    """Every required field in the run record must be populated."""
    manifest_path = str(source_data / "manifest.json")
    validate_manifest(manifest_path)
    run_dir = source_data / "run_schema"

    record = create_run_record(
        run_dir, manifest_path, base_config,
        repo_sha="c" * 40,
        backend_repo_url="https://github.com/NVlabs/LongSplat",
        backend_commit="d" * 40,
        python_exe="/usr/bin/python3",
        seed=123,
        runtime_profile="validated",
    )

    # Top-level keys
    assert record["schema_version"] == 1
    assert len(record["run_id"]) == 36
    assert record["status"] == "planned"
    assert record["repo_sha"] == "c" * 40
    assert record["seed"] == 123

    # Backend identity
    assert record["backend"]["repo_url"] == "https://github.com/NVlabs/LongSplat"
    assert record["backend"]["commit"] == "d" * 40

    # Runtime
    assert record["runtime"]["python_exe"] == "/usr/bin/python3"
    assert record["runtime"]["profile"] == "validated"

    # Input provenance
    assert "manifest_path" in record["input"]
    assert len(record["input"]["manifest_sha256"]) == 64

    # Config snapshot
    assert record["config"] == base_config

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
