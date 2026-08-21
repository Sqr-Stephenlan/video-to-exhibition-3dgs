from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.longsplat.reconstruct_pipeline as reconstruct_pipeline
from scripts.longsplat.pipeline_contract import RunLedger
from scripts.longsplat.reconstruct_pipeline import (
    _converted_eval_postprocess_stage,
    _recover_failed_converted_evaluation,
    _mid_consumer_resume_allowed,
    _resolve_converted_evaluation,
    _stage_reusable,
)


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
DOWNSTREAM_STAGES = (
    "native-render",
    "formal-native-render-postcheck",
    "automated-formal-gate",
    "authority-manifest",
    "conversion",
)
ABSENT_STAGES = (
    "converted-eval-postprocess",
    "candidate-delivery",
    "automated-technical-delivery",
    "accepted-delivery",
)


def _identity() -> dict[str, object]:
    return {
        "source_video_sha256": "source-sha",
        "source_video_path": "/immutable/video.mp4",
        "source_video_size_bytes": 12345,
        "tool_identity_sha256": "tool-sha",
        "canonical_config_sha256": "cfg-sha",
        "output_root_resolved": "/outputs",
        "run_root_resolved": "/outputs/run",
        # identical producer-scoped record in both identities -> equal digest.
        "code_identity": {
            "files": [
                {"path": "scripts/longsplat/raw_pipeline.py", "sha256": "producer-code-sha", "size_bytes": 123},
            ],
        },
    }


def _config() -> dict[str, object]:
    return {"depth_source": "disabled", "formal_profile": "formal30000-v1"}


def _write_passed_stage(run_dir: Path, stage: str, *, attempt: str = "attempt-0001", **result_extra) -> dict[str, str]:
    attempt_dir = run_dir / "stages" / stage / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"artifacts": []}
    payload.update(result_extra)
    result_path = attempt_dir / "result.json"
    result_path.write_text(json.dumps({"status": "passed", "result": payload}), encoding="utf-8")
    return {"attempt": attempt, "status": "passed", "result_path": str(result_path.relative_to(run_dir))}


def _write_blocked_stage(run_dir: Path, stage: str, *, attempt: str = "attempt-0002") -> dict[str, str]:
    attempt_dir = run_dir / "stages" / stage / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    result_path = attempt_dir / "result.json"
    result_path.write_text(json.dumps({"status": "blocked", "result": {"artifacts": []}, "reason": "blocked"}), encoding="utf-8")
    return {"attempt": attempt, "status": "blocked", "result_path": str(result_path.relative_to(run_dir))}




def _write_failed_stage(run_dir: Path, stage: str, *, attempt: str = "attempt-0002", reason: str = "child failed") -> dict[str, str]:
    attempt_dir = run_dir / "stages" / stage / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    result_path = attempt_dir / "result.json"
    result_path.write_text(json.dumps({"status": "failed", "result": {"artifacts": []}, "reason": reason}), encoding="utf-8")
    return {"attempt": attempt, "status": "failed", "result_path": str(result_path.relative_to(run_dir))}

def _build_summary(run_dir: Path, **overrides) -> dict[str, object]:
    stages: dict[str, list[dict[str, str]]] = {}
    for stage in PRODUCER_STAGES:
        extra = {}
        if stage == "convergence-smoke-render-postcheck":
            extra = {"contact_sheet_layout": {"policy": "aspect-preserving-letterbox-v1"}}
        stages[stage] = [_write_passed_stage(run_dir, stage, **extra)]
    # v3 shape: formal-training earlier attempts blocked, latest passed.
    stages["formal-training"] = [
        _write_blocked_stage(run_dir, "formal-training", attempt="attempt-0003"),
        _write_blocked_stage(run_dir, "formal-training", attempt="attempt-0005"),
        _write_passed_stage(run_dir, "formal-training", attempt="attempt-0006"),
    ]
    for stage in DOWNSTREAM_STAGES:
        extra = {}
        if stage == "formal-native-render-postcheck":
            extra = {"contact_sheet_layout": {"policy": "aspect-preserving-letterbox-v1"}}
        stages[stage] = [_write_passed_stage(run_dir, stage, **extra)]
    stages["converted-eval"] = [_write_blocked_stage(run_dir, "converted-eval")]
    summary: dict[str, object] = {
        "status": "blocked",
        "blocked": {"stage": "converted-eval", "error": "mid consumer resume", "exit_code": 2},
        "stages": stages,
    }
    summary.update(overrides)
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
    run_dir = tmp_path / "run-v3"
    run_dir.mkdir()
    identity = _identity()
    config = _config()
    summary = _build_summary(run_dir)
    return run_dir, summary, identity, config


def _gate_kwargs(run_dir: Path, identity: dict, config: dict, **over) -> dict:
    kwargs = dict(
        run_dir=run_dir,
        existing_identity=identity,
        current_identity=dict(identity),
        existing_config=config,
        current_config=dict(config),
        requested_stop_after="automated-technical-delivery",
        execute_gpu=True,
    )
    kwargs.update(over)
    return kwargs


def _recovery_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object], Path]:
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["identity"] = identity
    camera_order = ["view-a", "view-b"]
    dimensions = {"width": 5, "height": 4}
    authority_manifest_path = run_dir / "authority-manifest.json"
    authority_manifest_path.write_text("{}", encoding="utf-8")
    authority_result = {
        "schema_version": "longsplat-reconstruct-stage-v2",
        "stage": "authority-manifest",
        "status": "passed",
        "computed_pass": True,
        "artifacts": [],
        "camera_count": len(camera_order),
        "camera_order": camera_order,
        "camera_dimensions": dimensions,
        "authority_manifest_path": str(authority_manifest_path.resolve()),
        "authority": {"manifest": {}},
    }
    authority_path = run_dir / "stages" / "authority-manifest" / "attempt-0001" / "result.json"
    authority_path.write_text(
        json.dumps(
            {
                "schema_version": "stage-attempt-result-v1",
                "stage": "authority-manifest",
                "status": "passed",
                "identity": identity,
                "result": authority_result,
            }
        ),
        encoding="utf-8",
    )

    eval_attempt = run_dir / "stages" / "converted-eval" / "attempt-0001"
    executor_root = eval_attempt / "executor"
    gt_root = executor_root / "same_camera_eval" / "evaluator_result_gt"
    converted_root = executor_root / "same_camera_eval" / "evaluator_result_renders"
    gt_root.mkdir(parents=True)
    converted_root.mkdir(parents=True)
    names = [f"{index:04d}_{name}.png" for index, name in enumerate(camera_order)]
    for index, name in enumerate(names):
        image = np.full((dimensions["height"], dimensions["width"], 3), index + 1, dtype=np.uint8)
        assert cv2.imwrite(str(gt_root / name), image)
        assert cv2.imwrite(str(converted_root / name), image)
    evaluator_result = {
        "schema_version": "longsplat-generic-converted-evaluation-v1",
        "stage": "converted-eval",
        "exit_code": 7,
        "SAME_CAMERA_VISUAL_PASS": "fail",
        "reason": "fixture evaluator failure",
    }
    evaluator_result_path = executor_root / "evaluation_result.json"
    evaluator_result_path.write_text(json.dumps(evaluator_result), encoding="utf-8")
    evaluator_stage = {
        "schema_version": "longsplat-reconstruct-stage-v2",
        "stage": "converted-eval",
        "status": "failed",
        "computed_pass": False,
        "artifacts": [
            {
                "path": str(evaluator_result_path.resolve()),
                "sha256": hashlib.sha256(evaluator_result_path.read_bytes()).hexdigest(),
            }
        ],
        "executor_root": str(executor_root.resolve()),
        "executor_result": evaluator_result,
        "reason": "fixture evaluator failure",
    }
    (eval_attempt / "result.json").write_text(
        json.dumps(
            {
                "schema_version": "stage-attempt-result-v1",
                "stage": "converted-eval",
                "status": "failed",
                "identity": identity,
                "result": evaluator_stage,
            }
        ),
        encoding="utf-8",
    )
    summary["stages"]["converted-eval"] = [
        {"attempt": "attempt-0001", "status": "failed", "result_path": "stages/converted-eval/attempt-0001/result.json"}
    ]
    blocked_attempt = run_dir / "stages" / "converted-eval-postprocess" / "attempt-0001"
    blocked_attempt.mkdir(parents=True)
    (blocked_attempt / "result.json").write_text(
        json.dumps(
            {
                "schema_version": "stage-attempt-result-v1",
                "stage": "converted-eval-postprocess",
                "status": "blocked",
                "identity": identity,
                "result": {
                    "schema_version": "longsplat-reconstruct-stage-v2",
                    "stage": "converted-eval-postprocess",
                    "status": "blocked",
                    "computed_pass": False,
                    "artifacts": [],
                },
            }
        ),
        encoding="utf-8",
    )
    summary["stages"]["converted-eval-postprocess"] = [
        {"attempt": "attempt-0001", "status": "blocked", "result_path": "stages/converted-eval-postprocess/attempt-0001/result.json"}
    ]
    summary["blocked"] = {"stage": "converted-eval-postprocess", "error": "postprocess failed", "exit_code": 2}
    summary.pop("active_stage", None)
    summary.pop("active_attempt", None)
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir, summary, identity, config, evaluator_result_path


# ---------------------------------------------------------------------------
# Positive: v3 blocked at converted-eval with passed downstream is resumable.
# ---------------------------------------------------------------------------

def test_mid_consumer_resume_allowed_for_v3_blocked_at_converted_eval(tmp_path: Path) -> None:
    run_dir, _summary, identity, config = _fixture(tmp_path)
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is True


def test_passed_downstream_stages_are_reusable_and_converted_eval_is_not(tmp_path: Path) -> None:
    """Code-level reuse evidence: _stage_reusable returns the passed stage
    results (so they are never rerun) and None for the blocked converted-eval
    (so it is the rerun target)."""
    run_dir, summary, identity, _cfg = _fixture(tmp_path)
    ledger = RunLedger(run_dir, identity, resumed=True)
    ledger.summary = dict(summary)
    for stage in PRODUCER_STAGES + DOWNSTREAM_STAGES + ("formal-training",):
        assert _stage_reusable(ledger, stage) is not None, stage
    assert _stage_reusable(ledger, "converted-eval") is None
    for stage in ABSENT_STAGES:
        assert _stage_reusable(ledger, stage) is None, stage


# ---------------------------------------------------------------------------
# Negatives.
# ---------------------------------------------------------------------------

def test_negative_converted_eval_not_blocked(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["status"] = "blocked"
    summary["blocked"] = {"stage": "converted-eval"}
    summary["stages"]["converted-eval"] = [_write_passed_stage(run_dir, "converted-eval", attempt="attempt-0003")]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_status_not_blocked(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["status"] = "stopped"
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_blocked_stage_not_converted_eval(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["blocked"] = {"stage": "conversion"}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_downstream_conversion_missing(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    del summary["stages"]["conversion"]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_downstream_not_passed(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["stages"]["native-render"] = [_write_blocked_stage(run_dir, "native-render")]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_producer_missing(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    del summary["stages"]["frames"]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_formal_training_not_passed(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["stages"]["formal-training"] = [
        _write_blocked_stage(run_dir, "formal-training", attempt="attempt-0003"),
        _write_blocked_stage(run_dir, "formal-training", attempt="attempt-0005"),
        _write_blocked_stage(run_dir, "formal-training", attempt="attempt-0007"),
    ]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_candidate_delivery_exists(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["stages"]["candidate-delivery"] = [_write_passed_stage(run_dir, "candidate-delivery")]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_active_stage_present(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["active_stage"] = "converted-eval"
    summary["active_attempt"] = "attempt-0002"
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_identity_mismatch(tmp_path: Path) -> None:
    run_dir, _summary, identity, config = _fixture(tmp_path)
    drifted = dict(identity)
    drifted["source_video_sha256"] = "source-sha-OTHER"
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, current_identity=drifted)) is False


def test_negative_config_mismatch(tmp_path: Path) -> None:
    run_dir, _summary, identity, config = _fixture(tmp_path)
    drifted = dict(config)
    drifted["depth_source"] = "enabled"
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, current_config=drifted)) is False


def test_negative_cross_run_summary_missing(tmp_path: Path) -> None:
    run_dir, _summary, identity, config = _fixture(tmp_path)
    other = tmp_path / "other-run"
    other.mkdir()
    (other / "run.json").write_text(json.dumps({"status": "blocked", "blocked": {"stage": "converted-eval"}, "stages": {}}), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(other, identity, config)) is False


def test_negative_run_json_symlinked(tmp_path: Path) -> None:
    run_dir, _summary, identity, config = _fixture(tmp_path)
    real = tmp_path / "real-summary.json"
    real.write_text((run_dir / "run.json").read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "run.json").unlink()
    (run_dir / "run.json").symlink_to(real)
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_requested_stop_after_not_default(tmp_path: Path) -> None:
    run_dir, _summary, identity, config = _fixture(tmp_path)
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, requested_stop_after="convergence-smoke-plan")) is False


def test_negative_execute_gpu_not_bool(tmp_path: Path) -> None:
    run_dir, _summary, identity, config = _fixture(tmp_path)
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, execute_gpu="yes")) is False


# ---------------------------------------------------------------------------
# Invariant: an orphan never self-promotes to a reused/rerun stage.
# ---------------------------------------------------------------------------

def test_orphan_for_reused_downstream_stage_blocks_resume(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["orphan_evidence"] = {"stage": "native-render", "attempt": "attempt-0002", "status": "not_committed_to_ledger"}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_orphan_for_absent_delivery_stage_blocks_resume(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["orphan_evidence"] = {"stage": "converted-eval-postprocess", "attempt": "attempt-0001", "status": "not_committed_to_ledger"}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_empty_formal_orphan_is_allowed(tmp_path: Path) -> None:
    run_dir, summary, identity, config = _fixture(tmp_path)
    formal_orphan = run_dir / "stages" / "formal-training" / "attempt-9999" / "executor"
    formal_orphan.mkdir(parents=True)
    summary["orphan_evidence"] = {"stage": "formal-training", "attempt": "attempt-9999", "status": "not_committed_to_ledger"}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is True


def test_fake_executor_records_only_converted_eval_as_rerun(tmp_path: Path) -> None:
    """Drive the conductor reuse decision through _stage_reusable with a
    recording fake executor: every passed producer/consumer stage is reused,
    and only the blocked converted-eval is dispatched to the executor (the
    rerun set)."""
    run_dir, summary, identity, _cfg = _fixture(tmp_path)
    ledger = RunLedger(run_dir, identity, resumed=True)
    ledger.summary = dict(summary)

    rerun: list[str] = []
    upstream = list(PRODUCER_STAGES) + ["formal-training"] + list(DOWNSTREAM_STAGES)
    for stage in upstream:
        reused = _stage_reusable(ledger, stage)
        assert reused is not None, stage
    # converted-eval is blocked (not reusable) -> the executor would run it.
    assert _stage_reusable(ledger, "converted-eval") is None
    fake_executor(stage="converted-eval", rerun=rerun)

    assert rerun == ["converted-eval"]
    for stage in DOWNSTREAM_STAGES:
        assert stage not in rerun
    for stage in PRODUCER_STAGES:
        assert stage not in rerun


def fake_executor(*, stage: str, rerun: list[str]) -> None:
    """Stand-in for the GPU/authority executor entrypoint that records runs."""
    rerun.append(stage)


# ---------------------------------------------------------------------------
# converted-eval rerun target accepts both blocked and failed (non-passed).
# ---------------------------------------------------------------------------

def test_mid_consumer_resume_allowed_for_converted_eval_failed(tmp_path: Path) -> None:
    """v3-after-#C shape: converted-eval latest = failed (nonzero child exit);
    a failed retryable attempt is a valid rerun target just like blocked."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["stages"]["converted-eval"] = [_write_failed_stage(run_dir, "converted-eval", attempt="attempt-0002")]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is True


def test_mid_consumer_resume_allowed_blocked_converted_eval_absent(tmp_path: Path) -> None:
    """Positive: an absent (never recorded / None) retryable blocked stage is
    still a valid rerun target (matches the real v3 delivery-absent shape)."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    del summary["stages"]["converted-eval"]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is True



def test_mid_consumer_resume_allowed_blocked_at_automated_technical_delivery(tmp_path: Path) -> None:
    """Positive (generalized): blocked at the final automated-technical-delivery
    with all prior consumers (incl. converted-eval and postprocess) passed."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["stages"]["converted-eval"] = [_write_passed_stage(run_dir, "converted-eval", attempt="attempt-0003")]
    summary["stages"]["converted-eval-postprocess"] = [_write_passed_stage(run_dir, "converted-eval-postprocess", attempt="attempt-0001")]
    summary["stages"]["automated-technical-delivery"] = [_write_failed_stage(run_dir, "automated-technical-delivery", attempt="attempt-0001", reason="delivery bug")]
    summary["blocked"] = {"stage": "automated-technical-delivery", "error": "expected str, bytes or os.PathLike object, not dict", "exit_code": 2}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is True


def test_mid_consumer_resume_allowed_blocked_at_converted_eval_postprocess(tmp_path: Path) -> None:
    """Positive (generalized): blocked at converted-eval-postprocess with
    converted-eval already passed."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["stages"]["converted-eval"] = [_write_passed_stage(run_dir, "converted-eval", attempt="attempt-0003")]
    summary["stages"]["converted-eval-postprocess"] = [_write_blocked_stage(run_dir, "converted-eval-postprocess", attempt="attempt-0001")]
    summary["blocked"] = {"stage": "converted-eval-postprocess", "error": "postprocess failed", "exit_code": 2}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is True


def test_failed_evaluator_artifacts_allow_cpu_postprocess_resume_without_gpu_retry(tmp_path: Path) -> None:
    run_dir, summary, identity, config, _evaluator_result_path = _recovery_fixture(tmp_path)
    authority_result = json.loads(
        (run_dir / "stages" / "authority-manifest" / "attempt-0001" / "result.json").read_text(encoding="utf-8")
    )["result"]
    recovered = _recover_failed_converted_evaluation(
        summary=summary,
        run_dir=run_dir,
        authority_result=authority_result,
    )
    assert recovered is not None
    assert recovered["executor_result"]["exit_code"] == 7
    assert recovered["render_artifacts_available"] is True
    assert recovered["render_reused"] is True
    assert recovered["cuda_rerun"] is False
    assert recovered["evaluation_gate"] == "deferred_to_cpu_streaming_postprocess"
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, execute_gpu=False)) is True


def test_failed_evaluator_recovery_rejects_incomplete_png_inventory(tmp_path: Path) -> None:
    run_dir, summary, identity, config, _ = _recovery_fixture(tmp_path)
    (run_dir / "stages" / "converted-eval" / "attempt-0001" / "executor" / "same_camera_eval" / "evaluator_result_renders" / "0001_view-b.png").unlink()
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, execute_gpu=False)) is False


def test_failed_evaluator_recovery_rejects_camera_identity_drift(tmp_path: Path) -> None:
    run_dir, summary, identity, config, _ = _recovery_fixture(tmp_path)
    authority_path = run_dir / "stages" / "authority-manifest" / "attempt-0001" / "result.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["result"]["camera_order"] = ["view-a", "different-view"]
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, execute_gpu=False)) is False


def test_failed_evaluator_recovery_rejects_camera_dimension_drift(tmp_path: Path) -> None:
    run_dir, summary, identity, config, _ = _recovery_fixture(tmp_path)
    authority_path = run_dir / "stages" / "authority-manifest" / "attempt-0001" / "result.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["result"]["camera_dimensions"] = {"width": 6, "height": 4}
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, execute_gpu=False)) is False


def test_failed_evaluator_recovery_rejects_symlinked_executor_root(tmp_path: Path) -> None:
    run_dir, summary, identity, config, _ = _recovery_fixture(tmp_path)
    actual_root = run_dir / "stages" / "converted-eval" / "attempt-0001" / "executor"
    alias_root = actual_root.parent / "executor-alias"
    alias_root.symlink_to(actual_root, target_is_directory=True)
    result_path = run_dir / "stages" / "converted-eval" / "attempt-0001" / "result.json"
    envelope = json.loads(result_path.read_text(encoding="utf-8"))
    envelope["result"]["executor_root"] = str(alias_root)
    result_path.write_text(json.dumps(envelope), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, execute_gpu=False)) is False


def test_failed_evaluator_recovery_rejects_failed_result_drift(tmp_path: Path) -> None:
    run_dir, summary, identity, config, evaluator_result_path = _recovery_fixture(tmp_path)
    evaluator_result = json.loads(evaluator_result_path.read_text(encoding="utf-8"))
    evaluator_result["exit_code"] = 0
    evaluator_result_path.write_text(json.dumps(evaluator_result), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, execute_gpu=False)) is False


def test_failed_evaluator_recovery_rejects_existing_subsequent_stage(tmp_path: Path) -> None:
    run_dir, summary, identity, config, _ = _recovery_fixture(tmp_path)
    summary["stages"]["automated-technical-delivery"] = [_write_passed_stage(run_dir, "automated-technical-delivery")]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config, execute_gpu=False)) is False


def test_default_formal_delivery_recovers_failed_eval_without_executor_and_appends_postprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, summary, identity, _config_value, evaluator_result_path = _recovery_fixture(tmp_path)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture video")

    def overwrite_stage(stage: str, result: dict[str, object]) -> None:
        entry = summary["stages"][stage][-1]
        path = run_dir / entry["result_path"]
        path.write_text(
            json.dumps(
                {
                    "schema_version": "stage-attempt-result-v1",
                    "stage": stage,
                    "status": "passed",
                    "identity": identity,
                    "result": result,
                }
            ),
            encoding="utf-8",
        )

    native_root = run_dir / "stages" / "native-render" / "attempt-0001" / "executor"
    native_root.mkdir(parents=True)
    overwrite_stage(
        "native-render",
        {
            "schema_version": "longsplat-reconstruct-stage-v2",
            "stage": "native-render",
            "status": "passed",
            "computed_pass": True,
            "artifacts": [],
            "executor_root": str(native_root.resolve()),
            "quality": {"quality_status": "pass"},
        },
    )
    formal_gate_result = {
        "schema_version": "longsplat-reconstruct-stage-v2",
        "stage": "automated-formal-gate",
        "status": "passed",
        "computed_pass": True,
        "artifacts": [],
    }
    overwrite_stage("automated-formal-gate", formal_gate_result)

    conversion_root = run_dir / "stages" / "conversion" / "attempt-0001" / "executor"
    conversion_root.mkdir(parents=True)
    converted_ply = conversion_root / "point_cloud.ply"
    converted_ply.write_bytes(b"fixture ply")
    conversion_result_path = conversion_root / "conversion_result.json"
    conversion_result_path.write_text(
        json.dumps(
            {
                "STRUCTURAL_CONVERSION_PASS": True,
                "structural_pass": True,
                "structural": {"path": str(converted_ply.resolve()), "vertex_count": 1},
            }
        ),
        encoding="utf-8",
    )
    overwrite_stage(
        "conversion",
        {
            "schema_version": "longsplat-reconstruct-stage-v2",
            "stage": "conversion",
            "status": "passed",
            "computed_pass": True,
            "artifacts": [],
            "executor_root": str(conversion_root.resolve()),
            "executor_result": {"structural": {"path": str(converted_ply.resolve())}},
        },
    )
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    old_failed_result = (run_dir / "stages" / "converted-eval" / "attempt-0001" / "result.json").read_bytes()
    old_blocked_result = (run_dir / "stages" / "converted-eval-postprocess" / "attempt-0001" / "result.json").read_bytes()
    evaluator_attempt_dirs = sorted(
        path.name for path in (run_dir / "stages" / "converted-eval").iterdir() if path.is_dir()
    )
    summary.update(
        {
            "run_id": run_dir.name,
            "computed_pass": False,
            "accepted": False,
            "delivery_reachable": False,
            "gpu_invoked": False,
        }
    )

    ledger = RunLedger(run_dir, identity, resumed=True)
    ledger.summary = summary
    executor_calls: list[str] = []

    def fail_if_executor_called(**kwargs: object) -> tuple[dict[str, object], str]:
        executor_calls.append(str(kwargs.get("stage")))
        raise AssertionError("failed evaluator recovery must not call the GPU authority executor")

    monkeypatch.setattr(reconstruct_pipeline, "_execute_authority_stage", fail_if_executor_called)
    monkeypatch.setattr(reconstruct_pipeline, "_profile_input", lambda _payload, _profile: {})

    def fake_postprocess(**kwargs: object) -> dict[str, object]:
        output = Path(str(kwargs["output_root"]))
        output.mkdir(parents=True)
        for name in ("postprocess_result.json", "metrics.json", "png_hashes.json", "fixed_gt_native_converted_contact_sheet.png"):
            (output / name).write_bytes(name.encode("ascii"))
        return {"STRUCTURAL_EVALUATION_PASS": True, "FULL_STREAM_VALIDATION_PASS": True, "visual_quality_pass": True}

    monkeypatch.setattr("scripts.longsplat.converted_eval_postprocess.run_postprocess", fake_postprocess)
    monkeypatch.setattr(
        "scripts.longsplat.conversion_evidence_schema.normalize_converted_evaluation",
        lambda *_args, **_kwargs: {
            "STRUCTURAL_EVALUATION_PASS": True,
            "visual_quality_pass": True,
            "legacy_compatibility_warnings": [],
        },
    )
    delivery_seen: dict[str, object] = {}

    def fake_delivery(**kwargs: object) -> tuple[dict[str, object], str]:
        delivery_seen.update(kwargs)
        return {
            "stage": "automated-technical-delivery",
            "status": "passed",
            "computed_pass": True,
            "accepted_by_automated_policy": True,
            "artifacts": [],
        }, "passed"

    monkeypatch.setattr(reconstruct_pipeline, "_automated_technical_delivery", fake_delivery)
    result = reconstruct_pipeline._run_default_formal_delivery(
        ledger=ledger,
        route=tmp_path,
        source=source,
        camera_run_dir=None,
        results={
            "automated-early-gate": {"stage": "automated-early-gate", "status": "passed", "computed_pass": True},
            "frames": {},
            "colmap": {},
        },
        stop_after="automated-technical-delivery",
        execute_gpu=False,
        authority_manifest=None,
        camera_model="SIMPLE_RADIAL",
        matching="sequential",
        tool_paths=None,
        tool_provider={},
    )

    assert executor_calls == []
    assert result["status"] == "stopped"
    assert result["delivery_reachable"] is True
    assert delivery_seen["evaluation"]["failed_evaluation_recovery"]["read_only"] is True
    assert delivery_seen["postprocess"]["computed_pass"] is True
    assert sorted(
        path.name for path in (run_dir / "stages" / "converted-eval").iterdir() if path.is_dir()
    ) == evaluator_attempt_dirs
    assert (run_dir / "stages" / "converted-eval-postprocess" / "attempt-0002" / "result.json").is_file()
    assert (run_dir / "stages" / "converted-eval" / "attempt-0001" / "result.json").read_bytes() == old_failed_result
    assert (run_dir / "stages" / "converted-eval-postprocess" / "attempt-0001" / "result.json").read_bytes() == old_blocked_result
    assert evaluator_result_path.is_file()


def test_negative_blocked_gpu_stage_native_render(tmp_path: Path) -> None:
    """Negative: GPU consumer native-render is not retryable."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["blocked"] = {"stage": "native-render", "error": "gpu", "exit_code": 2}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_blocked_gpu_stage_conversion(tmp_path: Path) -> None:
    """Negative: GPU consumer conversion is not retryable."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["blocked"] = {"stage": "conversion", "error": "gpu", "exit_code": 2}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False


def test_negative_stage_after_blocked_target_exists(tmp_path: Path) -> None:
    """Negative: a consumer stage after the blocked retryable target exists."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    # blocked at converted-eval (idx 5); a later consumer already exists.
    summary["stages"]["converted-eval"] = [_write_blocked_stage(run_dir, "converted-eval")]
    summary["stages"]["converted-eval-postprocess"] = [_write_passed_stage(run_dir, "converted-eval-postprocess", attempt="attempt-0001")]
    summary["blocked"] = {"stage": "converted-eval", "error": "eval", "exit_code": 2}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False



def test_mid_consumer_resume_allowed_blocked_delivery_absent(tmp_path: Path) -> None:
    """Positive (real-v3 shape): blocked at automated-technical-delivery with the
    delivery stage still absent from the ledger (no entry -> None), all prior
    consumers passed."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["stages"]["converted-eval"] = [_write_passed_stage(run_dir, "converted-eval", attempt="attempt-0003")]
    summary["stages"]["converted-eval-postprocess"] = [_write_passed_stage(run_dir, "converted-eval-postprocess", attempt="attempt-0001")]
    # automated-technical-delivery is deliberately absent (no ledger entry); the
    # fixture never creates it, so None is the retryable blocked-stage state.
    summary["blocked"] = {"stage": "automated-technical-delivery", "error": "delivery", "exit_code": 2}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is True


def test_negative_blocked_delivery_latest_passed(tmp_path: Path) -> None:
    """Negative: blocked at automated-technical-delivery but its latest ledger
    entry is passed - a passed stage is not retryable."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["stages"]["converted-eval"] = [_write_passed_stage(run_dir, "converted-eval", attempt="attempt-0003")]
    summary["stages"]["converted-eval-postprocess"] = [_write_passed_stage(run_dir, "converted-eval-postprocess", attempt="attempt-0001")]
    summary["stages"]["automated-technical-delivery"] = [_write_passed_stage(run_dir, "automated-technical-delivery", attempt="attempt-0001")]
    summary["blocked"] = {"stage": "automated-technical-delivery", "error": "delivery", "exit_code": 2}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False



def test_orphan_empty_retry_target_blocked_stage_allows(tmp_path: Path) -> None:
    """Positive (#L): orphan.stage == blocked_stage (automated-technical-delivery)
    with an empty attempt dir (no result.json) is allowed; the delivery is
    appended fresh after it, so no stage self-promotes."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["stages"]["converted-eval"] = [_write_passed_stage(run_dir, "converted-eval", attempt="attempt-0003")]
    summary["stages"]["converted-eval-postprocess"] = [_write_passed_stage(run_dir, "converted-eval-postprocess", attempt="attempt-0001")]
    summary["blocked"] = {"stage": "automated-technical-delivery", "error": "delivery", "exit_code": 2}
    orphan_executor = run_dir / "stages" / "automated-technical-delivery" / "attempt-0001" / "executor"
    orphan_executor.mkdir(parents=True)  # exists but has NO result.json -> empty orphan
    summary["orphan_evidence"] = {"stage": "automated-technical-delivery", "attempt": "attempt-0001", "status": "not_committed_to_ledger"}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is True


def test_orphan_retry_target_with_result_json_rejects(tmp_path: Path) -> None:
    """Negative (#L): orphan.stage == blocked_stage but its attempt has a real
    result.json is genuine evidence -> rejected, not self-promoted."""
    run_dir, summary, identity, config = _fixture(tmp_path)
    summary["stages"]["converted-eval"] = [_write_passed_stage(run_dir, "converted-eval", attempt="attempt-0003")]
    summary["stages"]["converted-eval-postprocess"] = [_write_passed_stage(run_dir, "converted-eval-postprocess", attempt="attempt-0001")]
    summary["blocked"] = {"stage": "automated-technical-delivery", "error": "delivery", "exit_code": 2}
    orphan_executor = run_dir / "stages" / "automated-technical-delivery" / "attempt-0001" / "executor"
    orphan_executor.mkdir(parents=True)
    (orphan_executor / "result.json").write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
    summary["orphan_evidence"] = {"stage": "automated-technical-delivery", "attempt": "attempt-0001", "status": "not_committed_to_ledger"}
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _mid_consumer_resume_allowed(**_gate_kwargs(run_dir, identity, config)) is False
