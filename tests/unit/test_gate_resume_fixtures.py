from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.longsplat.gate_schema import GateSchemaBlocked, normalize_gate_evidence
from scripts.longsplat.reconstruct_pipeline import (
    STAGES,
    _technical_consumer_resume_allowed,
)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "longsplat" / "resume"
PRODUCER_STAGES = (
    "preflight",
    "probe",
    "frames",
    "colmap",
    "camera-staging",
    "longsplat-input",
    "convergence-smoke-plan",
    "convergence-smoke-training",
    "convergence-smoke-render",
    "convergence-smoke-render-postcheck",
)


# ---------------------------------------------------------------------------
# Fixture shape assertions (static external-root synthetic fixtures).
# ---------------------------------------------------------------------------

def test_gate_v2_flat_fixture_normalizes_directly():
    gate = json.loads((FIXTURE_DIR / "gate_v2_flat.json").read_text(encoding="utf-8"))
    normalized = normalize_gate_evidence(gate)
    assert normalized["schema_version"] == "automated-early-gate-v2"
    assert normalized.get("normalized_from_legacy_wrapper") is not True
    assert normalized["policy"]["policy_id"] == "automated-technical-v1"


def test_gate_v1_wrapper_fixture_is_promoted():
    gate = json.loads((FIXTURE_DIR / "gate_v1_wrapper.json").read_text(encoding="utf-8"))
    normalized = normalize_gate_evidence(gate)
    assert normalized["schema_version"] == "automated-early-gate-v2"
    assert normalized["normalized_from_legacy_wrapper"] is True
    assert normalized["computed_pass"] is True
    assert normalized["policy"]["policy_id"] == "automated-technical-v1"


def test_gpu_child_fixture_has_full_executor_shape():
    child = FIXTURE_DIR / "formal_gpu_child"
    for name in ("result.json", "request.json", "argv.json"):
        assert (child / name).is_file(), name
        assert not (child / name).is_symlink(), name
    result = json.loads((child / "result.json").read_text(encoding="utf-8"))
    assert result["exit_code"] == 0
    assert result["gpu_invoked"] is True
    assert result["stage"] == "training"
    assert isinstance(result["model_path"], str)
    assert (child / "model").is_dir() and not (child / "model").is_symlink()
    assert (child / "model" / "placeholder.txt").is_file()


def test_cpu_orphan_fixture_has_request_only():
    orphan = FIXTURE_DIR / "cpu_orphan"
    assert (orphan / "request.json").is_file()
    assert not (orphan / "result.json").exists()
    assert not (orphan / "argv.json").exists()
    assert not (orphan / "model").exists()


# ---------------------------------------------------------------------------
# State-machine scenarios for preserved_candidates==1 semantics (L3960-4055).
# ---------------------------------------------------------------------------

def _identity():
    return {
        "source_video_sha256": "source-sha",
        "source_video_path": "/immutable/video.mp4",
        "source_video_size_bytes": 12345,
        "tool_identity_sha256": "tool-sha",
        "canonical_config_sha256": "cfg-sha",
        "output_root_resolved": "/outputs",
        "run_root_resolved": "/outputs/run",
        # identical producer-scoped record in both identities -> equal producer
        # scoped digest, so the consumer-only continuation guard passes.
        "code_identity": {
            "files": [
                {"path": "scripts/longsplat/raw_pipeline.py", "sha256": "producer-code-sha", "size_bytes": 123},
            ],
        },
    }


def _config():
    return {
        "schema_version": "reconstruct-pipeline-config-v2",
        "depth_source": "disabled",
        "stage_order": list(STAGES),
        "smoke_profile": "smoke100-v1",
        "formal_profile": "formal30000-v1",
        "conversion_profile": "standard30000-v1",
    }


def _build_passed_producer(run_dir):
    stages: dict[str, list[dict[str, str]]] = {}
    for index, stage in enumerate(PRODUCER_STAGES, start=1):
        attempt = run_dir / "stages" / stage / "attempt-0001"
        attempt.mkdir(parents=True)
        result_path = attempt / "result.json"
        result_path.write_text(json.dumps({"status": "passed", "result": {"artifacts": []}}), encoding="utf-8")
        stages[stage] = [
            {"attempt": "attempt-0001", "status": "passed", "result_path": str(result_path.relative_to(run_dir))},
        ]
    return stages


def _install_executor(run_dir, *, kind: str, attempt: str = "attempt-0002") -> Path:
    """Copy the static executor template into a formal-training attempt.

    ``kind`` is "gpu" (full child: result.json + request + argv + model) or
    "orphan" (request.json only, no result.json).  For the gpu child the
    result.json ``model_path`` is rewritten to an absolute path under the run dir.
    """
    attempt_dir = run_dir / "stages" / "formal-training" / attempt
    executor_root = attempt_dir / "executor"
    executor_root.mkdir(parents=True)
    source = FIXTURE_DIR / ("formal_gpu_child" if kind == "gpu" else "cpu_orphan")
    for item in source.iterdir():
        dst = executor_root / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    if kind == "gpu":
        model_dir = executor_root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        result = json.loads((executor_root / "result.json").read_text(encoding="utf-8"))
        result["model_path"] = str(model_dir.resolve())
        (executor_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return executor_root


def _formal_training_entry(run_dir, *, kind: str, status: str, attempt: str = "attempt-0002") -> dict[str, str]:
    executor_root = _install_executor(run_dir, kind=kind, attempt=attempt)
    result_path = f"stages/formal-training/{attempt}/result.json"
    envelope = {"status": status, "result": {"executor_root": str(executor_root.resolve())}}
    (run_dir / "stages" / "formal-training" / attempt).mkdir(parents=True, exist_ok=True)
    (run_dir / result_path).write_text(json.dumps(envelope), encoding="utf-8")
    return {"attempt": attempt, "status": status, "result_path": result_path}


def _resume_run(run_dir, *, formal_entries):
    stages = _build_passed_producer(run_dir)
    stages["formal-training"] = formal_entries
    summary = {
        "status": "blocked",
        "last_stage": "formal-training",
        "blocked": {"stage": "formal-training"},
        "stages": stages,
    }
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    identity = _identity()
    config = _config()
    return dict(
        run_dir=run_dir,
        existing_identity=identity,
        current_identity=dict(identity),
        existing_config=config,
        current_config=config,
        requested_stop_after="formal-training",
        execute_gpu=True,
    )


def test_scenario_a_orphan_does_not_self_promote(tmp_path: Path):
    # A lone executor directory with only request.json (no result.json) must
    # NOT count as a preserved candidate: preserved_candidates==0 != 1, so the
    # consumer-only formal-training recovery is refused.
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    kwargs = _resume_run(run_dir, formal_entries=[_formal_training_entry(run_dir, kind="orphan", status="blocked")])
    assert _technical_consumer_resume_allowed(**kwargs) is False


def test_scenario_b_preserved_child_selected_exactly_once(tmp_path: Path):
    # Exactly one valid completed GPU child -> preserved_candidates==1 -> the
    # consumer-only formal-training recovery is allowed and returns True.
    run_dir = tmp_path / "run-b"
    run_dir.mkdir()
    kwargs = _resume_run(run_dir, formal_entries=[_formal_training_entry(run_dir, kind="gpu", status="blocked")])
    assert _technical_consumer_resume_allowed(**kwargs) is True


def test_scenario_c_executor_pass_but_ledger_unfinished(tmp_path: Path):
    # The executor child is a full pass (exit 0, gpu_invoked) yet the ledger
    # does not record formal-training as a blocked attempt (status is "passed").
    # preserved_candidates==1 is computed from blocked ledger entries only, so
    # the recovery must be refused.
    run_dir = tmp_path / "run-c"
    run_dir.mkdir()
    kwargs = _resume_run(run_dir, formal_entries=[_formal_training_entry(run_dir, kind="gpu", status="passed")])
    assert _technical_consumer_resume_allowed(**kwargs) is False


def test_scenario_d_two_preserved_children_are_rejected(tmp_path: Path):
    # Two completed GPU children (two blocked attempts, each a full executor)
    # -> preserved_candidates==2 != 1 -> recovery must be refused.
    run_dir = tmp_path / "run-d"
    run_dir.mkdir()
    first = _formal_training_entry(run_dir, kind="gpu", status="blocked")
    second = _formal_training_entry(run_dir, kind="gpu", status="blocked", attempt="attempt-0003")
    kwargs = _resume_run(run_dir, formal_entries=[first, second])
    assert _technical_consumer_resume_allowed(**kwargs) is False

