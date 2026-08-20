from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import pytest

import scripts.longsplat.reconstruct_pipeline as reconstruct_pipeline
from scripts.longsplat.pipeline_contract import PipelineBlocked, ResumeMismatchError, RunLedger
from scripts.longsplat.reconstruct_pipeline import (
    STAGES,
    _coverage_visual_gate,
    _coverage_plan_stage,
    _convergence_visual_gate,
    _convergence_consumer_binding,
    _compare_convergence_binding,
    _convergence_plan_matches_consumer,
    _diagnostic_consumer_resume_allowed,
    _input_video_identity,
    _formal_input,
    _route_output_root,
    _scoped_code_identity,
    _stage_result,
    _training_source_identity,
    _mark_existing_run_blocked,
    _manual_acceptance_valid,
    _visual_decision_valid,
    run_reconstruction,
)
from scripts.longsplat.reconstruct_validation import validate_existing_run


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "input video.mp4"
    source.write_bytes(b"synthetic video identity")
    return source


def _identity() -> dict[str, object]:
    return {"code_identity_sha256": "fixture-code"}


def _training_source_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object], dict[str, object]]:
    route = tmp_path / "route"
    run_dir = route / "outputs" / "run"
    training = run_dir / "raw" / "longsplat-input"
    training.mkdir(parents=True)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video identity")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    (training / "_model_bin").mkdir()
    (training / "_model_txt").mkdir()
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (training / "_model_bin" / name).write_bytes(name.encode("ascii"))
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (training / "_model_txt" / name).write_text(name, encoding="ascii")
    camera = {
        "schema_version": "camera-contract-v1",
        "contract_sha256": "camera-contract-sha",
        "source_video_sha256": source_sha,
        "frame_names": ["view-a.png", "view-b.png"],
        "camera": {"width": 32, "height": 24, "model": "PINHOLE"},
    }
    camera_path = training / "camera_contract-v1.json"
    camera_path.write_text(json.dumps(camera), encoding="utf-8")
    static_path = training / "static_contract.json"
    static_path.write_text(json.dumps({"source_video_sha256": source_sha}), encoding="utf-8")
    staging = {
        "source_path": str(training),
        "source_video_sha256": source_sha,
        "sparse_binary_evidence_path": str(training / "_model_bin"),
        "sparse_txt_evidence_path": str(training / "_model_txt"),
        "canonical_media_binding_sha256": "media-binding",
        "canonical_media_pixels_sha256": "media-pixels",
        "undistorted_input_pixel_aggregate_sha256": "undistorted-pixels",
        "final_staged_pixel_aggregate_sha256": "staged-pixels",
    }
    staging_path = training / "staging_manifest.json"
    staging_path.write_text(json.dumps(staging), encoding="utf-8")
    payload = {
        "source_path": str(training),
        "staging_manifest_path": str(staging_path),
        "static_contract_path": str(static_path),
        "static_contract_sha256": hashlib.sha256(static_path.read_bytes()).hexdigest(),
        "camera_contract_path": str(camera_path),
        **{key: value for key, value in staging.items() if key.endswith("sha256")},
    }
    input_result = {"source_path": str(training), "raw_result": payload}
    identity = {
        "source_video_sha256": source_sha,
        "run_identity_sha256": "run-identity",
        "code_identity": {"code_identity_sha256": "consumer-code"},
    }
    return route, run_dir, source, input_result, identity


def test_source_identity_separates_video_file_and_training_directory(tmp_path: Path) -> None:
    route, run_dir, source, input_result, identity = _training_source_fixture(tmp_path)
    video_identity = _input_video_identity(source, identity)
    training_identity = _training_source_identity(
        input_result=input_result,
        input_video_identity=video_identity,
        run_dir=run_dir,
        route=route,
    )
    assert video_identity["video_file"]["path"] == str(source.resolve())
    assert training_identity["training_source_directory"] == str((run_dir / "raw" / "longsplat-input").resolve())
    assert training_identity["immutable_contracts"]["staging_manifest"]["sha256"]
    assert training_identity["sparse_model"]["binary"]["files_sha256"]
    with pytest.raises(PipelineBlocked, match="regular file"):
        _input_video_identity(Path(str(input_result["source_path"])), identity)
    with pytest.raises(PipelineBlocked, match="regular directory|containment"):
        _training_source_identity(
            input_result={**input_result, "source_path": str(source)},
            input_video_identity=video_identity,
            run_dir=run_dir,
            route=route,
        )


def test_source_and_training_identity_drift_or_escape_hard_stops(tmp_path: Path) -> None:
    route, run_dir, source, input_result, identity = _training_source_fixture(tmp_path)
    video_identity = _input_video_identity(source, identity)
    source.write_bytes(b"mutated video")
    with pytest.raises(PipelineBlocked, match="SHA"):
        _input_video_identity(source, identity)
    source.write_bytes(b"video identity")
    outside = tmp_path / "outside-training"
    outside.mkdir()
    link = run_dir / "training-link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(PipelineBlocked, match="symlink|containment"):
        _training_source_identity(
            input_result={**input_result, "source_path": str(link)},
            input_video_identity=video_identity,
            run_dir=run_dir,
            route=route,
        )


def test_dynamic_output_root_is_not_route_frozen_and_rejects_aliases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    route = tmp_path / "route"
    route.mkdir()
    arbitrary = tmp_path / "external" / "outputs"
    resolved = _route_output_root(arbitrary, route)
    assert resolved == arbitrary.resolve()

    monkeypatch.chdir(tmp_path)
    assert _route_output_root(Path("external/../external/outputs"), route) == resolved

    link = tmp_path / "linked-output"
    link.symlink_to(arbitrary, target_is_directory=True)
    with pytest.raises(PipelineBlocked, match="symlink"):
        _route_output_root(link, route)
    with pytest.raises(PipelineBlocked, match="filesystem root"):
        _route_output_root(Path("/"), route)
    with pytest.raises(ResumeMismatchError, match="safe basename"):
        RunLedger.create_or_resume(output_root=arbitrary, run_id="../escape", identity={})


def test_input_video_resume_identity_freezes_path_size_and_sha(tmp_path: Path) -> None:
    source = _source(tmp_path)
    identity = {
        "source_video_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_video_path": str(source.resolve()),
        "source_video_size_bytes": source.stat().st_size,
        "run_identity_sha256": "run-identity",
    }
    record = _input_video_identity(source, identity)
    assert record["source_video_path"] == str(source.resolve())
    moved = tmp_path / "moved.mp4"
    moved.write_bytes(source.read_bytes())
    with pytest.raises(PipelineBlocked, match="path"):
        _input_video_identity(moved, identity)
    source.write_bytes(source.read_bytes() + b"drift")
    with pytest.raises(PipelineBlocked, match="SHA|size"):
        _input_video_identity(source, identity)


def test_scoped_consumer_identity_excludes_docs_and_keeps_producer_separate() -> None:
    code = {
        "files": [
            {"path": "scripts/longsplat/reconstruct_pipeline.py", "sha256": "consumer-a", "size_bytes": 1},
            {"path": "scripts/longsplat/pipeline_contract.py", "sha256": "shared-a", "size_bytes": 1},
            {"path": "scripts/longsplat/raw_pipeline.py", "sha256": "producer-a", "size_bytes": 1},
            {"path": "nested/scene/gaussian_model.py", "sha256": "nested-a", "size_bytes": 1},
            {"path": "docs/longsplat/README.md", "sha256": "docs-a", "size_bytes": 1},
            {"path": "tests/unit/test_reconstruct_pipeline.py", "sha256": "tests-a", "size_bytes": 1},
        ]
    }
    consumer = _scoped_code_identity(code, "convergence-consumer")
    producer = _scoped_code_identity(code, "producer")
    consumer_paths = {item["path"] for item in consumer["files"]}
    producer_paths = {item["path"] for item in producer["files"]}
    assert "docs/longsplat/README.md" not in consumer_paths | producer_paths
    assert "tests/unit/test_reconstruct_pipeline.py" not in consumer_paths | producer_paths
    assert "scripts/longsplat/reconstruct_pipeline.py" in consumer_paths
    assert "scripts/longsplat/raw_pipeline.py" in producer_paths
    assert "nested/scene/gaussian_model.py" in producer_paths


def test_stage_result_builder_normalizes_equal_and_blocks_conflicting_reserved_fields() -> None:
    equal = _stage_result(
        stage="synthetic",
        status="passed",
        computed_pass=True,
        reason="ok",
        plan=False,
        formal_auto_release=False,
        payload={"formal_auto_release": False, "adapter": "fixture"},
    )
    assert equal["status"] == "passed"
    assert equal["formal_auto_release"] is False
    assert equal["adapter"] == "fixture"

    conflicting = _stage_result(
        stage="synthetic",
        status="passed",
        computed_pass=True,
        reason="ok",
        plan=False,
        formal_auto_release=False,
        payload={"formal_auto_release": True},
    )
    assert conflicting["status"] == "blocked"
    assert conflicting["computed_pass"] is False
    assert "formal_auto_release" in conflicting["stage_result_conflict"]


def test_ledger_does_not_reuse_passed_orphan_artifact_outside_run(tmp_path: Path) -> None:
    ledger = RunLedger.create_or_resume(output_root=tmp_path / "outputs", run_id="orphan", identity={})
    attempt = ledger.begin_attempt("synthetic", {"stage": "synthetic"})
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    ledger.finish_attempt(
        stage="synthetic",
        attempt=attempt,
        status="passed",
        result={"computed_pass": True, "artifacts": [{"path": str(outside), "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}]},
    )
    assert ledger.latest_attempt("synthetic")["status"] == "passed"
    assert ledger.latest_attempt("synthetic")["reusable"] is False


def test_outer_exception_records_precise_stage_and_orphan_attempt(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    run_id = "outer-error"
    ledger = RunLedger.create_or_resume(output_root=output_root, run_id=run_id, identity={})
    attempt = ledger.begin_attempt("convergence-smoke-training", {"stage": "training"})
    _mark_existing_run_blocked(
        output_root=output_root,
        route_root=tmp_path / "route",
        run_id=run_id,
        error="synthetic outer exception",
        exit_code=2,
    )
    summary = json.loads(ledger.summary_path.read_text(encoding="utf-8"))
    assert summary["blocked"]["stage"] == "convergence-smoke-training"
    assert summary["blocked"]["error"] == "synthetic outer exception"
    assert summary["orphan_evidence"]["attempt"] == attempt.name


def test_new_invocation_moves_stale_active_stage_to_orphan_inventory(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    identity = {"source_video_sha256": "fixture"}
    first = RunLedger.create_or_resume(output_root=output_root, run_id="stale-active", identity=identity)
    attempt = first.begin_attempt("smoke100-training", {"stage": "training"})

    resumed = RunLedger.create_or_resume(output_root=output_root, run_id="stale-active", identity=identity)
    summary = json.loads(resumed.summary_path.read_text(encoding="utf-8"))
    assert "active_stage" not in summary
    assert "active_attempt" not in summary
    assert "current_intended_stage" not in summary
    assert summary["orphan_inventory"][-1]["stage"] == "smoke100-training"
    assert summary["orphan_inventory"][-1]["attempt"] == attempt.name


def test_outer_exception_prefers_current_invocation_intent_over_stale_active_stage(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    run_id = "precise-intent"
    ledger = RunLedger.create_or_resume(output_root=output_root, run_id=run_id, identity={})
    ledger.summary["active_stage"] = "smoke100-training"
    ledger.summary["active_attempt"] = "attempt-0009"
    ledger.summary["current_intended_stage"] = "convergence-smoke-plan"
    ledger._write_summary()
    _mark_existing_run_blocked(
        output_root=output_root,
        route_root=tmp_path,
        run_id=run_id,
        error="plan comparison failed",
    )
    summary = json.loads(ledger.summary_path.read_text(encoding="utf-8"))
    assert summary["blocked"]["stage"] == "convergence-smoke-plan"
    assert summary["blocked"]["error"] == "plan comparison failed"


def _coverage_fixture_files(tmp_path: Path, *, invalid_selection: bool = False) -> tuple[Path, Path]:
    names = ["view_17.jpg", "camera-A.png"]
    contract = tmp_path / "camera_contract-v1.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": "camera-contract-v1",
                "frame_names": names,
                "frame_count": len(names),
                "source_video_sha256": "fixture-source",
                "camera": {"width": 32, "height": 24, "model": "PINHOLE"},
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "frame_selection-v1.json"
    selection.write_text(
        "{}"
        if invalid_selection
        else json.dumps(
            {
                "schema_version": "frame-selection-v1",
                "config": {"max_frames": 240},
                "duration_sec": 1.0,
                "binding": {"source_video_sha256": "fixture-source"},
                "frames": [{"frame_id": name, "selected": True} for name in names],
            }
        ),
        encoding="utf-8",
    )
    return contract, selection


def test_coverage_helper_uses_named_artifact_and_ledger_is_canonical_writer(tmp_path: Path) -> None:
    contract, selection = _coverage_fixture_files(tmp_path)
    ledger = RunLedger.create_or_resume(
        output_root=tmp_path / "outputs",
        run_id="single-writer",
        identity=_identity(),
    )
    result, status = _coverage_plan_stage(
        ledger=ledger,
        input_result={"raw_result": {"camera_contract_path": str(contract)}},
        frames_result={"raw_result": {"artifacts": [{"path": str(selection)}]}},
        route=tmp_path,
        plan=False,
    )
    attempt = tmp_path / "outputs" / "single-writer" / "stages" / "coverage-smoke-plan" / "attempt-0001"
    canonical = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
    assert status == "passed"
    assert result["computed_pass"] is True
    assert canonical["schema_version"] == "stage-attempt-result-v1"
    assert (attempt / "coverage-smoke-result-v1.json").is_file()
    assert any(item["path"].endswith("coverage-smoke-result-v1.json") for item in canonical["result"]["artifacts"])
    assert not (attempt / "result.json").read_text(encoding="utf-8").startswith('{\n  "schema_version": "coverage-smoke-v1"')


def test_run_ledger_canonical_result_remains_exclusive(tmp_path: Path) -> None:
    ledger = RunLedger.create_or_resume(
        output_root=tmp_path / "outputs",
        run_id="exclusive-result",
        identity=_identity(),
    )
    attempt = ledger.begin_attempt("coverage-smoke-plan", {"fixture": True})
    ledger.finish_attempt(stage="coverage-smoke-plan", attempt=attempt, status="passed", result={"artifacts": []})
    original = (attempt / "result.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        ledger.finish_attempt(stage="coverage-smoke-plan", attempt=attempt, status="passed", result={"changed": True})
    assert (attempt / "result.json").read_text(encoding="utf-8") == original


def test_coverage_helper_exception_marks_authoritative_run_blocked(tmp_path: Path) -> None:
    contract, selection = _coverage_fixture_files(tmp_path, invalid_selection=True)
    ledger = RunLedger.create_or_resume(
        output_root=tmp_path / "outputs",
        run_id="blocked-coverage",
        identity=_identity(),
    )
    result, status = _coverage_plan_stage(
        ledger=ledger,
        input_result={"raw_result": {"camera_contract_path": str(contract)}},
        frames_result={"raw_result": {"artifacts": [{"path": str(selection)}]}},
        route=tmp_path,
        plan=False,
    )
    summary = json.loads((tmp_path / "outputs" / "blocked-coverage" / "run.json").read_text(encoding="utf-8"))
    assert status == "blocked"
    assert result["status"] == "blocked"
    assert summary["status"] == "blocked"
    assert summary["blocked"]["stage"] == "coverage-smoke-plan"
    assert summary["blocked"]["exit_code"] == 2
    assert summary["blocked"]["error"]


def test_outer_exception_status_closes_existing_run_without_overwriting_attempts(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "outer-blocked"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "status": "planned",
                "computed_pass": False,
                "accepted": False,
                "delivery_reachable": False,
                "stages": {"coverage-smoke-plan": [{"attempt": "attempt-0001"}]},
                "active_stage": "coverage-smoke-plan",
            }
        ),
        encoding="utf-8",
    )
    attempt_marker = run_dir / "stages" / "coverage-smoke-plan" / "attempt-0001" / "request.json"
    attempt_marker.parent.mkdir(parents=True)
    attempt_marker.write_text("preserve", encoding="utf-8")
    _mark_existing_run_blocked(
        output_root=tmp_path / "outputs",
        route_root=tmp_path,
        run_id="outer-blocked",
        error="synthetic outer exception",
    )
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["blocked"] == {"stage": "coverage-smoke-plan", "error": "synthetic outer exception", "exit_code": 2}
    assert attempt_marker.read_text(encoding="utf-8") == "preserve"


def test_passed_coverage_plan_enters_coverage_training_without_formal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import scripts.longsplat.reconstruct_pipeline as pipeline

    route = tmp_path / "route"
    (route / "outputs").mkdir(parents=True)
    (route / "dev.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    source = _source(tmp_path)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    plan_path = tmp_path / "future_smoke_plan.json"
    static_path = tmp_path / "static_contract.json"
    plan_path.write_text("{}", encoding="utf-8")
    static_path.write_text("{}", encoding="utf-8")
    payload = {
        "source_path": str(source),
        "future_smoke_plan_path": str(plan_path),
        "static_contract_path": str(static_path),
    }

    def fake_raw_video_pipeline(**_: object) -> dict[str, str]:
        return {"run_dir": str(raw_dir)}

    def fake_adapt(*, stage: str, raw_run_dir: Path, plan: bool) -> dict[str, object]:
        return {
            "stage": stage,
            "status": "passed",
            "computed_pass": True,
            "reason": "synthetic adapter fixture",
            "raw_run_dir": str(raw_run_dir),
            "raw_result": dict(payload),
            "source_path": str(source),
            "plan_path": str(plan_path),
            "static_contract_path": str(static_path),
        }

    calls: list[str] = []

    def fake_execute(**kwargs: object) -> tuple[dict[str, object], str]:
        stage = str(kwargs["stage"])
        calls.append(stage)
        result: dict[str, object] = {
            "stage": stage,
            "status": "passed",
            "computed_pass": True,
            "executor_root": str(tmp_path / stage),
        }
        if stage == "smoke100-render":
            result["quality"] = {"quality_status": "needs_review"}
        return result, "passed"

    def fake_coverage_plan(**_: object) -> tuple[dict[str, object], str]:
        return {
            "stage": "coverage-smoke-plan",
            "status": "passed",
            "computed_pass": True,
            "reason": "synthetic dynamic plan",
            "coverage_plan_path": str(plan_path),
        }, "passed"

    monkeypatch.setattr("scripts.longsplat.raw_pipeline.run_raw_video_pipeline", fake_raw_video_pipeline)
    monkeypatch.setattr(pipeline, "_adapt_raw_stage", fake_adapt)
    monkeypatch.setattr(pipeline, "_execute_training_or_render", fake_execute)
    monkeypatch.setattr(pipeline, "_coverage_plan_stage", fake_coverage_plan)
    result = run_reconstruction(
        input_video=source,
        output_root=route / "outputs",
        run_id="coverage-flow",
        stop_after="coverage-smoke-training",
        route_root=route,
        code_identity_override=_identity(),
        execute_gpu=True,
    )
    assert result["status"] == "stopped"
    assert calls == ["smoke100-training", "smoke100-render", "coverage-smoke-training"]
    assert "formal-training" not in calls


def test_plan_stop_after_records_order_without_gpu(tmp_path: Path) -> None:
    result = run_reconstruction(
        input_video=_source(tmp_path),
        output_root=tmp_path / "outputs",
        run_id="plan-run",
        stop_after="longsplat-input",
        plan=True,
        route_root=tmp_path,
        code_identity_override=_identity(),
    )
    assert result["status"] == "planned"
    assert result["gpu_invoked"] is False
    assert list(result["stage_results"]) == list(STAGES[:6])
    assert result["stage_results"]["longsplat-input"]["computed_pass"] is False


def test_plan_exposes_coverage_and_diagnostic_stages_without_auto_convergence(tmp_path: Path) -> None:
    result = run_reconstruction(
        input_video=_source(tmp_path),
        output_root=tmp_path / "outputs",
        run_id="full-plan",
        stop_after="accepted-delivery",
        plan=True,
        route_root=tmp_path,
        code_identity_override=_identity(),
    )
    assert result["status"] == "planned"
    stages = list(result["stage_results"])
    assert stages.index("coverage-smoke-plan") < stages.index("coverage-smoke-visual-gate")
    assert stages.index("coverage-smoke-visual-gate") < stages.index("formal-training")
    assert stages.index("coverage-smoke-visual-gate") < stages.index("convergence-smoke-plan")
    assert stages.index("convergence-smoke-visual-gate") < stages.index("formal-training")
    assert stages.index("converted-eval") < stages.index("converted-eval-postprocess")


def test_first_gpu_boundary_is_structured_and_does_not_execute(tmp_path: Path) -> None:
    result = run_reconstruction(
        input_video=_source(tmp_path),
        output_root=tmp_path / "outputs",
        run_id="gpu-boundary",
        stop_after="conversion",
        plan=False,
        route_root=tmp_path,
        code_identity_override=_identity(),
    )
    assert result["status"] == "blocked"
    assert result["gpu_invoked"] is False
    convergence = result["stage_results"]["convergence-smoke-training"]
    assert convergence["status"] == "requires_escalated_gpu_execution"
    assert convergence["reason"] == "requires_escalated_gpu_execution"


def test_source_mutation_cannot_resume_same_run(tmp_path: Path) -> None:
    source = _source(tmp_path)
    kwargs = dict(
        input_video=source,
        output_root=tmp_path / "outputs",
        run_id="mutation-run",
        stop_after="preflight",
        plan=True,
        route_root=tmp_path,
        code_identity_override=_identity(),
    )
    run_reconstruction(**kwargs)
    source.write_bytes(b"mutated source")
    with pytest.raises(ResumeMismatchError, match="source_video_sha256"):
        run_reconstruction(**kwargs)


def test_acceptance_token_requires_explicit_three_view_flags(tmp_path: Path) -> None:
    token = tmp_path / "acceptance.json"
    token.write_text(json.dumps({"accepted": True, "supersplat": True}), encoding="utf-8")
    with pytest.raises(Exception, match="three-view"):
        _manual_acceptance_valid(token)
    token.write_text(
        json.dumps({"accepted": True, "supersplat": True, "three_view_manual_acceptance": True}),
        encoding="utf-8",
    )
    record = _manual_acceptance_valid(token)
    assert record["record"]["accepted"] is True


def test_user_asserted_acceptance_keeps_screenshot_archive_state_separate(tmp_path: Path) -> None:
    token = tmp_path / "accepted.json"
    token.write_text(
        json.dumps(
            {
                "status": "ACCEPTED_BY_USER_PENDING_SCREENSHOT_ARCHIVE",
                "accepted": True,
                "supersplat": True,
                "three_view_manual_acceptance": True,
                "user_asserted_manual_acceptance": True,
                "screenshot_embeds_ply_sha": False,
                "screenshot_file_evidence": "missing",
                "screenshot_evidence_complete": False,
            }
        ),
        encoding="utf-8",
    )
    record = _manual_acceptance_valid(token)
    assert record["screenshot_file_evidence"] == "missing"
    assert record["screenshot_evidence_complete"] is False


def test_visual_gate_requires_explicit_human_token(tmp_path: Path) -> None:
    token = tmp_path / "visual.json"
    token.write_text(json.dumps({"decision": "pass"}), encoding="utf-8")
    with pytest.raises(PipelineBlocked, match="human visual review"):
        _visual_decision_valid(token, label="coverage token")
    token.write_text(
        json.dumps({"decision": "pass", "explicit_visual_review": True}),
        encoding="utf-8",
    )
    assert _visual_decision_valid(token, label="coverage token")["decision"] == "pass"


@pytest.mark.parametrize(
    "decision,diagnostic_profile,expected_status,expected_formal,expected_diagnostic",
    [
        ("pass", "convergence1000-v1", "passed", True, False),
        ("needs_review", None, "blocked", False, False),
        ("fail", "convergence1000-v1", "blocked", False, True),
    ],
)
def test_coverage_gate_keeps_visual_and_diagnostic_eligibility_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
    diagnostic_profile: str | None,
    expected_status: str,
    expected_formal: bool,
    expected_diagnostic: bool,
) -> None:
    expected_evidence = {
        "postcheck": {"path": str(tmp_path / "postcheck.json"), "sha256": "postcheck"},
        "quality_evidence": {"path": str(tmp_path / "quality.json"), "sha256": "quality"},
        "fixed_vs_gt": {"path": str(tmp_path / "fixed.png"), "sha256": "fixed"},
        "on_path": {"path": str(tmp_path / "on-path.png"), "sha256": "on-path"},
        "metrics": {"path": str(tmp_path / "metrics.json"), "sha256": "metrics"},
        "png_hashes": {"path": str(tmp_path / "png-hashes.json"), "sha256": "png-hashes"},
    }
    monkeypatch.setattr(reconstruct_pipeline, "_visual_evidence_binding", lambda *_args, **_kwargs: expected_evidence)
    token = tmp_path / "coverage-token.json"
    token.write_text(
        json.dumps({"decision": decision, "explicit_visual_review": True, "evidence": expected_evidence}),
        encoding="utf-8",
    )
    ledger = RunLedger.create_or_resume(
        output_root=tmp_path / "outputs",
        run_id=f"coverage-{decision}-{diagnostic_profile or 'none'}",
        identity=_identity(),
    )
    result, status = _coverage_visual_gate(
        ledger=ledger,
        smoke={"computed_pass": True},
        coverage_plan={"computed_pass": True},
        coverage_render={"computed_pass": True},
        coverage_postcheck={
            "computed_pass": True,
            "postcheck_result_path": str(tmp_path / "postcheck.json"),
            "quality_evidence_path": str(tmp_path / "quality.json"),
            "postcheck_result": {},
            "quality_evidence": {},
        },
        decision_token=token,
        diagnostic_profile=diagnostic_profile,
    )
    assert status == expected_status
    assert result["computed_pass"] is expected_formal
    assert result["formal_release_eligible"] is expected_formal
    assert result["diagnostic_release_eligible"] is expected_diagnostic
    assert result["formal_release_eligible"] is not result["diagnostic_release_eligible"] or expected_formal is False


def test_coverage_gate_rejects_arbitrary_diagnostic_profile(tmp_path: Path) -> None:
    ledger = RunLedger.create_or_resume(
        output_root=tmp_path / "outputs",
        run_id="coverage-invalid-profile",
        identity=_identity(),
    )
    with pytest.raises(PipelineBlocked, match="only the versioned convergence1000-v1"):
        _coverage_visual_gate(
            ledger=ledger,
            smoke={"computed_pass": True},
            coverage_plan={"computed_pass": True},
            coverage_render={"computed_pass": True},
            coverage_postcheck={},
            decision_token=None,
            diagnostic_profile="arbitrary-profile",
        )


def test_convergence_visual_gate_without_token_is_diagnostic_blocked(tmp_path: Path) -> None:
    ledger = RunLedger.create_or_resume(
        output_root=tmp_path / "outputs",
        run_id="convergence-gate",
        identity=_identity(),
    )
    result, status = _convergence_visual_gate(
        ledger=ledger,
        convergence_postcheck={
            "computed_pass": True,
            "postcheck_result_path": str(tmp_path / "postcheck.json"),
            "quality_evidence_path": str(tmp_path / "quality.json"),
            "postcheck_result": {},
            "quality_evidence": {},
        },
        decision_token=None,
    )
    assert status == "blocked"
    assert result["computed_pass"] is False
    assert result["diagnostic_only"] is True
    assert result["formal_release_eligible"] is False
    summary = json.loads(ledger.summary_path.read_text(encoding="utf-8"))
    assert summary["blocked"]["stage"] == "convergence-smoke-visual-gate"


def _diagnostic_resume_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object], Path]:
    run_dir = tmp_path / "diagnostic-resume"
    run_dir.mkdir()
    token = run_dir / "coverage-token.json"
    token.write_text(json.dumps({"decision": "needs_review", "explicit_visual_review": True}), encoding="utf-8")
    config: dict[str, object] = {
        "schema_version": "reconstruct-pipeline-config-v2",
        "depth_source": "disabled",
        "stage_order": list(STAGES),
        "smoke_profile": "smoke100-v1",
        "formal_profile": "formal30000-v1",
        "conversion_profile": "standard30000-v1",
    }
    existing_identity: dict[str, object] = {
        "source_video_sha256": "source-sha",
        "tool_identity_sha256": "tool-sha",
        "code_identity": {"code_identity_sha256": "producer-code"},
    }
    current_identity: dict[str, object] = {
        "source_video_sha256": "source-sha",
        "tool_identity_sha256": "tool-sha",
        "code_identity": {"code_identity_sha256": "consumer-code"},
    }
    stages: dict[str, list[dict[str, str]]] = {}
    for index, stage in enumerate(
        (
            "preflight",
            "probe",
            "frames",
            "colmap",
            "camera-staging",
            "longsplat-input",
            "smoke100-training",
            "smoke100-render",
            "coverage-smoke-plan",
            "coverage-smoke-training",
            "coverage-smoke-render",
            "coverage-smoke-render-postcheck",
        ),
        start=1,
    ):
        attempt = run_dir / "stages" / stage / "attempt-0001"
        attempt.mkdir(parents=True)
        result: dict[str, object] = {"artifacts": []}
        if stage == "coverage-smoke-plan":
            result.update(
                {
                    "profile_id": "convergence1000-v1",
                    "convergence_plan": {
                        "requested_iterations": 1000,
                        "active_camera_count": 2,
                        "active_camera_order": ["left.png", "right.png"],
                    },
                }
            )
        result_path = attempt / "result.json"
        result_path.write_text(json.dumps({"status": "passed", "result": result}), encoding="utf-8")
        stages[stage] = [{"attempt": "attempt-0001", "status": "passed", "result_path": str(result_path.relative_to(run_dir))}]
    summary = {
        "status": "stopped",
        "last_stage": "convergence-smoke-plan",
        "stages": stages,
    }
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return run_dir, existing_identity, current_identity, config, token


def test_diagnostic_resume_allows_cpu_plan_but_rejects_gpu_stage_or_config_drift(tmp_path: Path) -> None:
    run_dir, existing, current, config, token = _diagnostic_resume_fixture(tmp_path)
    kwargs = {
        "run_dir": run_dir,
        "existing_identity": existing,
        "current_identity": current,
        "current_config": config,
        "coverage_visual_token": token,
        "requested_stop_after": "convergence-smoke-visual-gate",
        "execute_gpu": True,
    }
    assert _diagnostic_consumer_resume_allowed(**kwargs) is True

    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    training_attempt = run_dir / "stages" / "convergence-smoke-training" / "attempt-0001"
    training_attempt.mkdir(parents=True)
    training_result = training_attempt / "result.json"
    training_result.write_text(json.dumps({"status": "blocked", "result": {"artifacts": []}}), encoding="utf-8")
    summary["stages"]["convergence-smoke-training"] = [{
        "attempt": "attempt-0001",
        "status": "blocked",
        "result_path": str(training_result.relative_to(run_dir)),
    }]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _diagnostic_consumer_resume_allowed(**kwargs) is False

    summary["stages"].pop("convergence-smoke-training")
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    drifted = dict(config)
    drifted["depth_source"] = "enabled"
    assert _diagnostic_consumer_resume_allowed(**{**kwargs, "current_config": drifted}) is False


def test_diagnostic_resume_allows_fresh_cpu_plan_after_plan_only_block(tmp_path: Path) -> None:
    run_dir, existing, current, config, token = _diagnostic_resume_fixture(tmp_path)
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    convergence_attempt = run_dir / "stages" / "convergence-smoke-plan" / "attempt-0001"
    convergence_attempt.mkdir(parents=True)
    convergence_result = convergence_attempt / "result.json"
    convergence_result.write_text(
        json.dumps(
            {
                "status": "passed",
                "result": {
                    "profile_id": "convergence1000-v1",
                    "convergence_plan": {
                        "requested_iterations": 1000,
                        "active_camera_count": 2,
                        "active_camera_order": ["left.png", "right.png"],
                        "identity_binding": {"camera_order_sha256": "fixture-order"},
                    },
                    "artifacts": [],
                },
            }
        ),
        encoding="utf-8",
    )
    summary["stages"]["convergence-smoke-plan"] = [{
        "attempt": "attempt-0001",
        "status": "passed",
        "result_path": str(convergence_result.relative_to(run_dir)),
    }]
    summary["status"] = "blocked"
    summary["blocked"] = {"stage": "convergence-smoke-plan", "error": "legacy source adapter", "exit_code": 2}
    summary["stages"]["convergence-smoke-plan"][-1]["status"] = "blocked"
    plan_result = run_dir / summary["stages"]["convergence-smoke-plan"][-1]["result_path"]
    plan_result.write_text(
        json.dumps(
            {
                "status": "blocked",
                "result": {
                    "profile_id": "convergence1000-v1",
                    "convergence_plan": {
                        "requested_iterations": 1000,
                        "active_camera_count": 2,
                        "active_camera_order": ["left.png", "right.png"],
                        "identity_binding": {"camera_order_sha256": "fixture-order"},
                    },
                    "artifacts": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")
    assert _diagnostic_consumer_resume_allowed(
        run_dir=run_dir,
        existing_identity=existing,
        current_identity=current,
        current_config=config,
        coverage_visual_token=token,
        requested_stop_after="convergence-smoke-plan",
        execute_gpu=False,
    ) is True


def test_legacy_convergence_plan_is_fresh_append_only_consumer_attempt(tmp_path: Path) -> None:
    comparison = _convergence_plan_matches_consumer(
        plan_result={"profile_id": "convergence1000-v1", "convergence_plan": {"requested_iterations": 1000}},
        input_result={},
        input_video_identity={},
        consumer_identity={},
        run_dir=tmp_path,
        route=tmp_path,
        coverage_visual_token=None,
        coverage_decision=None,
    )
    assert comparison["classification"] == "stale_rebuildable"
    assert comparison["stale_rebuildable"] is True


def test_convergence_binding_classifies_consumer_migration_and_immutable_drift() -> None:
    expected = {
        "schema_version": "convergence-consumer-binding-v3",
        "consumer_run_identity_sha256": "new-run",
        "consumer_code_identity_sha256": "new-consumer",
        "producer_code_identity_sha256": "producer",
        "profile_id": "convergence1000-v1",
        "active_camera_count": 2,
        "active_camera_order": ["left.png", "right.png"],
        "requested_iterations": 1000,
    }
    migrated = dict(expected)
    migrated["schema_version"] = "convergence-consumer-binding-v2"
    migrated["consumer_run_identity_sha256"] = "old-run"
    migrated["consumer_code_identity_sha256"] = "old-consumer"
    comparison = _compare_convergence_binding(migrated, expected)
    assert comparison["classification"] == "stale_rebuildable"

    drifted = dict(migrated)
    drifted["active_camera_count"] = 3
    comparison = _compare_convergence_binding(drifted, expected)
    assert comparison["classification"] == "unsafe_drift"
    assert "active_camera_count" in comparison["diff"]["unsafe_fields"]


def test_validate_only_reuses_dynamic_authority_candidate_and_cpu_recovery(tmp_path: Path) -> None:
    from tests.unit.test_authority_manifest import _fixture

    _manifest, authority_path, route = _fixture(
        tmp_path,
        "validate",
        camera_names=["left-view", "right-view"],
        width=19,
        height=13,
    )
    run_root = route / "outputs" / "validate"
    authority_stage = run_root / "stages" / "authority-manifest" / "attempt-0001"
    authority_stage.mkdir(parents=True)
    copied_authority = authority_stage / "authority_manifest-v1.json"
    shutil.copyfile(authority_path, copied_authority)
    conversion_stage = run_root / "stages" / "conversion" / "attempt-0001" / "executor"
    conversion_stage.mkdir(parents=True)
    ply = conversion_stage / "point_cloud.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n",
        encoding="ascii",
    )
    digest = hashlib.sha256(ply.read_bytes()).hexdigest()
    conversion = {
        "STRUCTURAL_CONVERSION_PASS": True,
        "structural_pass": True,
        "structural": {"path": str(ply), "sha256": digest, "file_size": ply.stat().st_size, "vertex_count": 1},
    }
    (conversion_stage / "conversion_result.json").write_text(json.dumps(conversion), encoding="utf-8")
    post_stage = run_root / "stages" / "converted-eval-postprocess" / "attempt-0001"
    post_stage.mkdir(parents=True)
    (post_stage / "postprocess_result.json").write_text(
        json.dumps(
            {
                "FULL_STREAM_VALIDATION_PASS": True,
                "SAME_CAMERA_VISUAL_PASS": "pass",
                "gpu_invoked": False,
                "render_reused": True,
                "cuda_rerun": False,
                "full_stream_validation": {"resident_full_resolution_frame_max": 3},
            }
        ),
        encoding="utf-8",
    )
    candidate = run_root / "stages" / "candidate-delivery" / "attempt-0001" / "candidate_delivery"
    candidate.mkdir(parents=True)
    shutil.copyfile(ply, candidate / "point_cloud.ply")
    (candidate / "candidate_manifest.json").write_text(json.dumps({"accepted": False, "supersplat": False}), encoding="utf-8")
    accepted = run_root / "accepted_delivery_20260814_v1"
    accepted.mkdir()
    shutil.copyfile(ply, accepted / "point_cloud.ply")
    accepted_record = {
        "status": "ACCEPTED_BY_USER_PENDING_SCREENSHOT_ARCHIVE",
        "accepted": True,
        "supersplat": True,
        "three_view_manual_acceptance": True,
        "user_asserted_manual_acceptance": True,
        "screenshot_file_evidence": "missing",
        "screenshot_evidence_complete": False,
        "screenshot_records": [],
        "contact_sheet": None,
        "point_cloud": {"sha256": digest, "size_bytes": ply.stat().st_size},
    }
    (accepted / "SUPERSPLAT_ACCEPTANCE.json").write_text(json.dumps(accepted_record), encoding="utf-8")
    result = validate_existing_run(
        route_root=route,
        run_root=run_root,
        input_video=Path(_manifest["source_video"]["path"]),
        authority_manifest=copied_authority,
    )
    assert result["validate_only"] is True
    assert result["gpu_invoked"] is False
    assert result["computed_pass"] is True
    assert result["stage_results"]["accepted-delivery"]["screenshot_file_evidence"] == "missing"
    assert result["recovery"]["cuda_rerun"] is False


def test_formal_input_reuses_passed_nested_producer_without_raw_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    route = tmp_path / "route"
    run_dir = route / "external-output" / "run"
    camera_run = run_dir / "raw-camera" / "camera"
    formal_run = run_dir / "raw-formal-input" / "formal-input"
    camera_stage = camera_run / "stages" / "camera-staging" / "attempt-0001"
    formal_stage = formal_run / "stages" / "longsplat-input" / "attempt-0001" / "longsplat_input"
    camera_stage.mkdir(parents=True)
    formal_stage.mkdir(parents=True)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"formal resume source")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()

    camera_contract = camera_stage / "camera_contract-v1.json"
    camera_contract.write_text(json.dumps({"schema_version": "camera-contract-v1", "count": 2}), encoding="utf-8")
    camera_contract_sha = hashlib.sha256(camera_contract.read_bytes()).hexdigest()
    camera_result = camera_stage / "result.json"
    camera_payload = {
        "contract_path": str(camera_contract),
        "contract": {"contract_sha256": camera_contract_sha},
        "artifacts": [{"path": str(camera_contract), "sha256": camera_contract_sha}],
    }
    camera_result.write_text(json.dumps({"status": "passed", "result": camera_payload}), encoding="utf-8")
    camera_identity = {
        "source_video_sha256": source_sha,
        "run_identity_sha256": "camera-run-identity",
        "code_identity": {"code_identity_sha256": "producer-code"},
    }
    (camera_run / "identity.json").parent.mkdir(parents=True, exist_ok=True)
    (camera_run / "identity.json").write_text(json.dumps(camera_identity), encoding="utf-8")
    (camera_run / "run.json").write_text(
        json.dumps({"stages": {"camera-staging": [{"status": "passed", "result_path": "stages/camera-staging/attempt-0001/result.json"}]}}),
        encoding="utf-8",
    )

    static_path = formal_stage / "static_contract.json"
    plan_path = formal_stage / "future_smoke_plan-v2.json"
    staging_path = formal_stage / "staging_manifest.json"
    formal_camera_path = formal_stage / "camera_contract-v1.json"
    for path, value in (
        (static_path, {"schema_version": "static-contract-v1"}),
        (plan_path, {"profile": "formal30000-v1", "iterations": 30000}),
        (staging_path, {"schema_version": "staging-manifest-v1"}),
        (formal_camera_path, {"schema_version": "camera-contract-v1", "count": 2}),
    ):
        path.write_text(json.dumps(value), encoding="utf-8")
    parent_binding = {
        "parent_run_root": str(camera_run),
        "parent_run_identity_sha256": camera_identity["run_identity_sha256"],
        "parent_code_identity_sha256": camera_identity["code_identity"]["code_identity_sha256"],
        "source_video_sha256": source_sha,
        "camera_contract_path": str(camera_contract),
        "camera_contract_file_sha256": camera_contract_sha,
        "camera_staging_result_path": str(camera_result),
        "camera_staging_result_file_sha256": hashlib.sha256(camera_result.read_bytes()).hexdigest(),
    }
    artifact_paths = [static_path, plan_path, staging_path, formal_camera_path]
    payload = {
        "status": "computed_pass",
        "source_path": str(formal_stage),
        "static_contract_path": str(static_path),
        "static_contract_sha256": hashlib.sha256(static_path.read_bytes()).hexdigest(),
        "future_smoke_plan_path": str(plan_path),
        "future_smoke_profile": "formal30000-v1",
        "future_smoke_iterations": 30000,
        "staging_manifest_path": str(staging_path),
        "camera_contract_path": str(formal_camera_path),
        "camera_contract_sha256": hashlib.sha256(formal_camera_path.read_bytes()).hexdigest(),
        "parent_binding": parent_binding,
        "artifacts": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in artifact_paths],
    }
    formal_result = formal_stage.parent / "result.json"
    formal_result.write_text(json.dumps({"status": "passed", "result": payload}), encoding="utf-8")
    (formal_run / "run.json").write_text(
        json.dumps({"stages": {"longsplat-input": [{"status": "passed", "result_path": "stages/longsplat-input/attempt-0001/result.json"}]}}),
        encoding="utf-8",
    )
    (formal_run / "identity.json").write_text(
        json.dumps({"source_video_sha256": source_sha, "run_root_resolved": str(formal_run)}),
        encoding="utf-8",
    )

    def fail_if_raw_pipeline_called(**_: object) -> None:
        raise AssertionError("passed formal input must not invoke raw pipeline on consumer resume")

    monkeypatch.setattr("scripts.longsplat.raw_pipeline.run_raw_video_pipeline", fail_if_raw_pipeline_called)
    ledger = RunLedger(run_dir, {"schema_version": "test", "code_identity": {}}, resumed=True)
    result = _formal_input(source=source, route=route, ledger=ledger, camera_run_dir=camera_run)
    assert result["reused"] is True
    assert result["profile"] == "formal30000-v1"
    assert Path(result["raw_run_dir"]) == formal_run.resolve()


def test_output_root_accepts_route_external_absolute_directory(tmp_path: Path) -> None:
    output_root = tmp_path / "outside-run"
    result = run_reconstruction(
        input_video=_source(tmp_path),
        output_root=output_root,
        run_id="external-run",
        stop_after="longsplat-input",
        plan=True,
        route_root=tmp_path,
        code_identity_override=_identity(),
    )
    assert result["status"] == "planned"
    assert Path(result["run_dir"]).parent == output_root.resolve()
