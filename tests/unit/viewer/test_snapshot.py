from __future__ import annotations

from types import SimpleNamespace

from scripts.viewer.api.snapshot import build_job_snapshot, normalize_run_status


def test_normalize_run_status_keeps_terminal_acceptance_separate():
    assert normalize_run_status("planned") == "queued"
    assert normalize_run_status("running") == "running"
    assert normalize_run_status("complete") == "complete"
    assert normalize_run_status("blocked") == "blocked"
    assert normalize_run_status("failed") == "failed"


def test_build_snapshot_maps_ledger_stage_and_observed_conversion():
    job = {
        "schema_version": "web-job-v1",
        "job_id": "job-1",
        "run_id": "job-1",
        "status": "running",
        "created_at": "2026-08-24T00:00:00Z",
        "input": {
            "name": "gallery.mp4",
            "size_bytes": 123,
            "sha256": "b" * 64,
        },
    }
    config = {
        "stage_order": ["probe", "conversion", "automated-technical-delivery"],
    }
    run = {
        "status": "running",
        "updated_at": "2026-08-24T00:01:00Z",
        "active_stage": "conversion",
        "active_attempt": "attempt-0001",
        "stages": {
            "probe": [
                {
                    "attempt": "attempt-0001",
                    "status": "passed",
                    "result_path": "stages/probe/attempt-0001/result.json",
                }
            ]
        },
        "acceptance": {"accepted": False, "reason": "technical delivery pending"},
    }
    progress = SimpleNamespace(
        stage="conversion",
        stage_position=2,
        stage_attempt="attempt-0001",
        stage_status="running",
        stage_started=1724457630.0,
        conversion=SimpleNamespace(
            iteration=12,
            total=30,
            marker_timestamp=1724457660.0,
        ),
        training=SimpleNamespace(iteration=None, total=None),
    )

    snapshot = build_job_snapshot(
        job_record=job,
        config_record=config,
        run_record=run,
        progress=progress,
        artifacts=[],
    )

    assert snapshot.status == "running"
    assert snapshot.current_stage is not None
    assert snapshot.current_stage.id == "conversion"
    assert snapshot.current_stage.index == 2
    assert snapshot.stages[0].status == "passed"
    assert snapshot.stages[1].status == "running"
    assert snapshot.observed_progress is not None
    assert snapshot.observed_progress.iteration == 12
    assert snapshot.observed_progress.total == 30
    assert snapshot.observed_progress.stage == "conversion"
    assert snapshot.observed_progress.attempt == "attempt-0001"
    assert snapshot.quality.accepted_by_automated_policy is False
    assert snapshot.quality.supersplat_runtime_verified is False
    assert not hasattr(snapshot, "percent")


def test_build_snapshot_promotes_progress_when_run_ledger_is_still_planned():
    snapshot = build_job_snapshot(
        job_record={
            "job_id": "job-1",
            "run_id": "job-1",
            "status": "running",
            "created_at": "2026-08-24T00:00:00Z",
        },
        config_record={"stage_order": ["formal-training"]},
        run_record={
            "status": "planned",
            "updated_at": "2026-08-24T00:01:00Z",
            "stages": {},
        },
        progress=SimpleNamespace(
            stage="formal-training",
            stage_active=True,
            stage_attempt="attempt-0001",
            stage_status="running",
            stage_started=1724457630.0,
            training=SimpleNamespace(iteration=250, total=30000, marker_timestamp=1724457660.0),
            conversion=SimpleNamespace(iteration=None, total=None),
        ),
    )

    assert snapshot.status == "running"
    assert snapshot.current_stage is not None
    assert snapshot.current_stage.id == "formal-training"
    assert snapshot.observed_progress is not None
    assert snapshot.observed_progress.iteration == 250


def test_build_snapshot_keeps_historical_training_progress_origin_explicit():
    snapshot = build_job_snapshot(
        job_record={
            "job_id": "job-1",
            "run_id": "job-1",
            "status": "blocked",
            "created_at": "2026-08-24T00:00:00Z",
        },
        config_record={"stage_order": ["formal-training", "conversion"]},
        run_record={
            "status": "blocked",
            "active_stage": "conversion",
            "active_attempt": "attempt-0001",
            "last_stage": "conversion",
            "blocked": {"stage": "conversion", "error": "converter unavailable"},
            "stages": {
                "formal-training": [{"attempt": "attempt-0001", "status": "passed"}],
                "conversion": [{"attempt": "attempt-0001", "status": "blocked"}],
            },
        },
        progress=SimpleNamespace(
            stage="conversion",
            stage_active=False,
            stage_attempt="attempt-0001",
            stage_status="blocked",
            stage_started=1724457630.0,
            training=SimpleNamespace(iteration=250, total=30000, marker_timestamp=1724457660.0),
            conversion=SimpleNamespace(iteration=None, total=None),
        ),
    )

    assert snapshot.observed_progress is not None
    assert snapshot.observed_progress.stage == "formal-training"
    assert snapshot.observed_progress.attempt == "attempt-0001"


def test_build_snapshot_does_not_promote_historical_progress_to_running():
    snapshot = build_job_snapshot(
        job_record={
            "job_id": "job-1",
            "run_id": "job-1",
            "status": "running",
            "created_at": "2026-08-24T00:00:00Z",
        },
        config_record={"stage_order": ["formal-training", "conversion"]},
        run_record={
            "status": "planned",
            "updated_at": "2026-08-24T00:01:00Z",
            "stages": {},
        },
        progress=SimpleNamespace(
            stage="formal-training",
            stage_active=False,
            stage_attempt="attempt-0001",
            stage_status="passed",
            stage_started=1724457630.0,
            training=SimpleNamespace(iteration=250, total=30000, marker_timestamp=1724457660.0),
            conversion=SimpleNamespace(iteration=None, total=None),
        ),
    )

    assert snapshot.status == "queued"
    assert snapshot.current_stage is None


def test_build_snapshot_keeps_conversion_progress_origin_after_stage_switch():
    snapshot = build_job_snapshot(
        job_record={
            "job_id": "job-1",
            "run_id": "job-1",
            "status": "running",
            "created_at": "2026-08-24T00:00:00Z",
        },
        config_record={"stage_order": ["conversion", "native-render"]},
        run_record={
            "status": "running",
            "active_stage": "native-render",
            "active_attempt": "attempt-0001",
            "stages": {
                "conversion": [{"attempt": "attempt-0001", "status": "passed"}],
                "native-render": [{"attempt": "attempt-0001", "status": "running"}],
            },
        },
        progress=SimpleNamespace(
            stage="native-render",
            stage_active=True,
            stage_attempt="attempt-0001",
            stage_status="running",
            stage_started=1724457690.0,
            conversion=SimpleNamespace(iteration=12, total=30, marker_timestamp=1724457660.0),
            training=SimpleNamespace(iteration=None, total=None),
        ),
    )

    assert snapshot.observed_progress is not None
    assert snapshot.observed_progress.stage == "conversion"
    assert snapshot.observed_progress.attempt == "attempt-0001"


def test_build_snapshot_keeps_terminal_blocked_stage_visible():
    snapshot = build_job_snapshot(
        job_record={
            "job_id": "job-1",
            "run_id": "job-1",
            "status": "blocked",
            "created_at": "2026-08-24T00:00:00Z",
        },
        config_record={"stage_order": ["formal-training", "conversion"]},
        run_record={
            "status": "blocked",
            "updated_at": "2026-08-24T00:01:00Z",
            "last_stage": "conversion",
            "blocked": {"stage": "conversion", "error": "conversion failed", "exit_code": 2},
            "stages": {
                "formal-training": [{"attempt": "attempt-0001", "status": "passed"}],
                "conversion": [{"attempt": "attempt-0001", "status": "blocked"}],
            },
        },
    )

    assert snapshot.status == "blocked"
    assert snapshot.current_stage is not None
    assert snapshot.current_stage.id == "conversion"
    assert snapshot.current_stage.status == "blocked"


def test_complete_snapshot_fails_closed_without_published_delivery_evidence():
    snapshot = build_job_snapshot(
        job_record={
            "job_id": "job-1",
            "run_id": "job-1",
            "status": "running",
            "created_at": "2026-08-24T00:00:00Z",
        },
        config_record={"stage_order": []},
        run_record={
            "status": "complete",
            "accepted_by_automated_policy": True,
            "supersplat_format_compatible": True,
        },
    )

    assert snapshot.status == "failed"
    assert snapshot.error is not None
    assert snapshot.error.code == "technical_delivery_incomplete"
