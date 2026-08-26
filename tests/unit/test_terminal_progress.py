from __future__ import annotations

import io
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts.longsplat.terminal_progress import (
    ProgressSnapshot,
    TerminalProgress,
    read_progress_snapshot,
    render_progress_line,
)


def _stamp(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def _stage_order() -> list[str]:
    return [
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
        "automated-early-gate",
        "formal-training",
        "native-render",
        "formal-native-render-postcheck",
        "automated-formal-gate",
        "authority-manifest",
        "conversion",
        "converted-eval",
        "converted-eval-postprocess",
        "automated-technical-delivery",
    ]


def _make_run(
    tmp_path: Path,
    *,
    stage: str = "conversion",
    status: str = "running",
    attempt_status: str = "running",
) -> Path:
    run_dir = tmp_path / "run"
    attempt = run_dir / "stages" / stage / "attempt-0001"
    (attempt / "executor").mkdir(parents=True)
    stage_order = _stage_order()
    (run_dir / "config.json").write_text(
        json.dumps({"stage_order": stage_order}),
        encoding="utf-8",
    )
    (attempt / "request.json").write_text(
        json.dumps({"started_at": _stamp(100)}),
        encoding="utf-8",
    )
    (attempt / "result.json").write_text(
        json.dumps(
            {
                "result": {
                    "executor_root": str(attempt / "executor"),
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "created_at": _stamp(0),
                "updated_at": _stamp(100),
                "status": status,
                "active_stage": stage,
                "active_attempt": "attempt-0001",
                "stages": {
                    stage: [
                        {
                            "attempt": "attempt-0001",
                            "status": attempt_status,
                            "result_path": f"stages/{stage}/attempt-0001/result.json",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _write_status_run(
    run_dir: Path,
    *,
    status: str,
    stage_statuses: dict[str, str],
    active_stage: str | None = None,
    last_stage: str | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    stages: dict[str, list[dict[str, str]]] = {}
    for stage, attempt_status in stage_statuses.items():
        attempt = run_dir / "stages" / stage / "attempt-0001"
        (attempt / "executor").mkdir(parents=True, exist_ok=True)
        (attempt / "request.json").write_text(
            json.dumps({"started_at": _stamp(100)}),
            encoding="utf-8",
        )
        (attempt / "result.json").write_text(
            json.dumps(
                {
                    "stage": stage,
                    "status": attempt_status,
                    "finished_at": _stamp(200),
                }
            ),
            encoding="utf-8",
        )
        stages[stage] = [
            {
                "attempt": "attempt-0001",
                "result_path": f"stages/{stage}/attempt-0001/result.json",
                "status": attempt_status,
            }
        ]
    (run_dir / "config.json").write_text(
        json.dumps({"stage_order": _stage_order()}),
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "created_at": _stamp(0),
        "updated_at": _stamp(200),
        "status": status,
        "stages": stages,
    }
    if active_stage is not None:
        summary["active_stage"] = active_stage
        summary["active_attempt"] = "attempt-0001"
    if last_stage is not None:
        summary["last_stage"] = last_stage
    if status in {"blocked", "failed"}:
        summary[status] = {"stage": last_stage}
    (run_dir / "run.json").write_text(
        json.dumps(summary, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir


def _activate_stage(run_dir: Path, stage: str, *, attempt: str = "attempt-0001", started: int = 101) -> None:
    attempt_dir = run_dir / "stages" / stage / attempt
    (attempt_dir / "executor").mkdir(parents=True, exist_ok=True)
    (attempt_dir / "request.json").write_text(
        json.dumps({"started_at": _stamp(started)}),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "created_at": _stamp(0),
                "updated_at": _stamp(started),
                "status": "running",
                "active_stage": stage,
                "active_attempt": attempt,
                "stages": {
                    stage: [
                        {"attempt": attempt, "status": "running"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def _make_raw_camera_run(tmp_path: Path, *, raw_stage: str = "colmap") -> Path:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "raw-camera" / "camera"
    attempt = raw_dir / "stages" / raw_stage / "attempt-0002"
    (attempt / "executor").mkdir(parents=True)
    (attempt / "request.json").write_text(
        json.dumps({"started_at": _stamp(100)}),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps({"stage_order": _stage_order()}),
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "created_at": _stamp(0),
                "updated_at": _stamp(100),
                "status": "running",
                "stages": {},
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "run.json").write_text(
        json.dumps(
            {
                "created_at": _stamp(0),
                "updated_at": _stamp(100),
                "status": "running",
                "active_stage": raw_stage,
                "active_attempt": "attempt-0002",
                "stages": {},
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _write_sampling_events(model: Path, iterations: int) -> None:
    model.mkdir(parents=True, exist_ok=True)
    model.joinpath("camera_sampling_telemetry-v1.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "camera-sampling-telemetry-v1",
                    "iteration": iteration,
                }
            )
            + "\n"
            for iteration in range(1, iterations + 1)
        ),
        encoding="utf-8",
    )


def _make_training_run(tmp_path: Path, *, stage: str = "convergence-smoke-training") -> Path:
    run_dir = _make_run(tmp_path, stage=stage)
    attempt = run_dir / "stages" / stage / "attempt-0001"
    model = attempt / "model"
    (attempt / "executor" / "request.json").write_text(
        json.dumps(
            {
                "stage": "training",
                "iterations": 1000,
                "model_path": str(model),
            }
        ),
        encoding="utf-8",
    )
    _write_sampling_events(model, 3)
    return run_dir


def test_conversion_marker_render_and_stale_warning(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    sidecar = run_dir / "stages" / "conversion" / "attempt-0001" / "executor" / "conversion-progress-v1.jsonl"
    sidecar.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": "conversion-progress-v1",
                        "phase": "child_started",
                        "total": 30000,
                        "timestamp_utc": _stamp(100),
                        "elapsed_seconds": 0,
                    }
                ),
                json.dumps(
                    {
                        "schema_version": "conversion-progress-v1",
                        "phase": "iteration",
                        "iteration": 14000,
                        "total": 30000,
                        "timestamp_utc": _stamp(100),
                        "elapsed_seconds": 3600,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = read_progress_snapshot(run_dir, now=1100)
    line = render_progress_line(snapshot, now=1100)

    assert snapshot is not None
    assert snapshot.conversion.iteration == 14000
    assert "conversion 14000/30000" in line
    assert line.startswith("[")
    assert "WARNING: no observed iteration heartbeat" in line
    assert "阶段 00:16:40" in line
    assert "总计 00:18:20" in line


def test_missing_marker_is_unobserved_and_does_not_fake_percent_or_eta(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    snapshot = read_progress_snapshot(run_dir, now=200)
    line = render_progress_line(snapshot, now=200)

    assert snapshot is not None
    assert snapshot.conversion.iteration is None
    assert "conversion (unobserved)" in line
    assert "0/30000" not in line
    assert "ETA" not in line


def test_partial_sidecar_line_is_ignored_and_previous_marker_survives(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    sidecar = run_dir / "stages" / "conversion" / "attempt-0001" / "executor" / "conversion-progress-v1.jsonl"
    valid = {
        "schema_version": "conversion-progress-v1",
        "phase": "iteration",
        "iteration": 1000,
        "total": 30000,
        "timestamp_utc": _stamp(100),
        "elapsed_seconds": 1,
    }
    sidecar.write_text(json.dumps(valid) + "\n{\"schema_version\":\"conversion-progress-v1\"", encoding="utf-8")

    snapshot = read_progress_snapshot(run_dir, now=110)

    assert snapshot is not None
    assert snapshot.conversion.iteration == 1000


def test_active_conversion_attempt_never_falls_back_to_previous_sidecar(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    previous = run_dir / "stages" / "conversion" / "attempt-0000" / "executor"
    previous.mkdir(parents=True)
    previous_sidecar = previous / "conversion-progress-v1.jsonl"
    previous_sidecar.write_text(
        json.dumps(
            {
                "schema_version": "conversion-progress-v1",
                "phase": "iteration",
                "iteration": 14000,
                "total": 30000,
                "timestamp_utc": _stamp(100),
                "elapsed_seconds": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot = read_progress_snapshot(run_dir, now=110)

    assert snapshot is not None
    assert snapshot.conversion.iteration is None


def test_blocked_and_failed_statuses_are_displayed_without_changing_stage_progress(tmp_path: Path) -> None:
    for status in ("blocked", "failed"):
        run_dir = _make_run(tmp_path / status, status=status)
        snapshot = read_progress_snapshot(run_dir, now=110)
        line = render_progress_line(snapshot, now=110)

        assert snapshot is not None
        assert f"| {status}" in line
        assert not line.startswith("[")


def test_latest_summary_stage_uses_canonical_order_not_mapping_order(tmp_path: Path) -> None:
    run_dir = _write_status_run(
        tmp_path / "canonical",
        status="passed",
        stage_statuses={"probe": "passed", "frames": "passed", "colmap": "passed"},
        last_stage="colmap",
    )

    snapshot = read_progress_snapshot(run_dir, now=200)
    line = render_progress_line(snapshot, now=200)

    assert snapshot is not None
    assert snapshot.stage == "colmap"
    assert snapshot.stage_status == "passed"
    assert "colmap | 完成" in line
    assert "probe | 完成" not in line


def test_blocked_or_failed_attempt_never_renders_as_completion(tmp_path: Path) -> None:
    for status in ("blocked", "failed"):
        run_dir = _write_status_run(
            tmp_path / status,
            status=status,
            stage_statuses={"probe": "passed", "colmap": status},
            last_stage="colmap",
        )
        snapshot = read_progress_snapshot(run_dir, now=200)
        line = render_progress_line(snapshot, now=200)

        assert snapshot is not None
        assert snapshot.stage == "colmap"
        assert snapshot.stage_status == status
        assert f"colmap | {status}" in line
        assert "colmap | 完成" not in line
        assert "probe | 完成" not in line


def test_missing_terminal_attempt_status_is_pending_not_completion(tmp_path: Path) -> None:
    run_dir = _write_status_run(
        tmp_path / "pending",
        status="stopped",
        stage_statuses={"colmap": "running"},
        last_stage="colmap",
    )

    snapshot = read_progress_snapshot(run_dir, now=200)
    line = render_progress_line(snapshot, now=200)

    assert snapshot is not None
    assert snapshot.stage == "colmap"
    assert "colmap | pending" in line
    assert "colmap | 完成" not in line


def test_pending_transition_is_not_frozen_and_later_flushes_canonical_completion(
    tmp_path: Path,
) -> None:
    run_dir = _write_status_run(
        tmp_path / "pending-transition",
        status="running",
        stage_statuses={"frames": "running", "colmap": "running"},
        active_stage="colmap",
        last_stage="colmap",
    )
    stream = io.StringIO()
    reporter = TerminalProgress("plain", stream=stream, env={"TERM": "x"})
    reporter.bind_run(run_dir)
    reporter.poll_once(force=True)
    assert stream.getvalue() == ""

    _write_status_run(
        run_dir,
        status="running",
        stage_statuses={"frames": "passed", "colmap": "running"},
        active_stage="colmap",
        last_stage="colmap",
    )
    reporter.poll_once()
    _write_status_run(
        run_dir,
        status="blocked",
        stage_statuses={"frames": "passed", "colmap": "blocked"},
        last_stage="colmap",
    )
    reporter.poll_once()

    lines = stream.getvalue().splitlines()
    assert lines[0].startswith("frames | 完成")
    assert lines[1].startswith("colmap |")
    assert "pending" not in lines[0]
    assert "colmap | blocked" in lines[2]
    assert "colmap | 完成" not in lines[2]
    reporter.close()


def test_active_retry_does_not_inherit_old_failed_attempt(tmp_path: Path) -> None:
    run_dir = _write_status_run(
        tmp_path / "retry",
        status="running",
        stage_statuses={"colmap": "failed"},
        active_stage="colmap",
        last_stage="colmap",
    )
    attempt = run_dir / "stages" / "colmap" / "attempt-0002"
    (attempt / "executor").mkdir(parents=True)
    (attempt / "request.json").write_text(
        json.dumps({"started_at": _stamp(300)}),
        encoding="utf-8",
    )
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary["active_attempt"] = "attempt-0002"
    (run_dir / "run.json").write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    snapshot = read_progress_snapshot(run_dir, now=400)
    line = render_progress_line(snapshot, now=400)

    assert snapshot is not None
    assert snapshot.stage == "colmap"
    assert snapshot.stage_attempt == "attempt-0002"
    assert snapshot.stage_status is None
    assert "failed" not in line
    assert "blocked" not in line
    assert "完成" not in line


def test_current_intended_retry_is_pending_until_its_own_attempt_is_recorded(
    tmp_path: Path,
) -> None:
    run_dir = _write_status_run(
        tmp_path / "current-intended-retry",
        status="blocked",
        stage_statuses={"colmap": "failed"},
        last_stage="colmap",
    )
    attempt = run_dir / "stages" / "colmap" / "attempt-0002"
    (attempt / "executor").mkdir(parents=True)
    (attempt / "request.json").write_text(
        json.dumps({"started_at": _stamp(300)}),
        encoding="utf-8",
    )
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary["current_intended_stage"] = "colmap"
    summary["active_attempt"] = "attempt-0002"
    (run_dir / "run.json").write_text(
        json.dumps(summary, sort_keys=True),
        encoding="utf-8",
    )

    snapshot = read_progress_snapshot(run_dir, now=400)
    line = render_progress_line(snapshot, now=400)

    assert snapshot is not None
    assert snapshot.stage == "colmap"
    assert snapshot.stage_active is True
    assert snapshot.stage_attempt == "attempt-0002"
    assert snapshot.stage_status is None
    assert "colmap | pending" in line
    assert "failed" not in line
    assert "完成" not in line

    stream = io.StringIO()
    reporter = TerminalProgress("plain", stream=stream, env={"TERM": "x"})
    reporter.bind_run(run_dir)
    reporter.poll_once(force=True)
    reporter.close()
    assert stream.getvalue() == ""


def test_close_reads_last_authoritative_snapshot_without_duplicate_terminal_line(
    tmp_path: Path,
) -> None:
    run_dir = _write_status_run(
        tmp_path / "close",
        status="blocked",
        stage_statuses={"colmap": "blocked"},
        last_stage="colmap",
    )
    stream = io.StringIO()
    reporter = TerminalProgress("plain", stream=stream, env={"TERM": "x"})
    reporter.bind_run(run_dir)
    reporter.close()

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("colmap | blocked")


def test_tty_pending_is_refresh_only_and_passed_replaces_it(tmp_path: Path) -> None:
    run_dir = _write_status_run(
        tmp_path / "tty-pending",
        status="running",
        stage_statuses={"frames": "running"},
        last_stage="frames",
    )

    class TTY(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = TTY()
    reporter = TerminalProgress("auto", stream=stream, env={"TERM": "x"})
    reporter.bind_run(run_dir)
    reporter.poll_once(force=True)
    assert "frames | pending" in stream.getvalue()
    assert "\n" not in stream.getvalue()

    _write_status_run(
        run_dir,
        status="passed",
        stage_statuses={"frames": "passed"},
        last_stage="frames",
    )
    reporter.poll_once()
    output = stream.getvalue()
    assert output.endswith("\n")
    assert output.count("\n") == 1
    assert "frames | 完成" in output
    reporter.close()


def test_raw_child_blocked_stage_overrides_stale_root_probe(tmp_path: Path) -> None:
    run_dir = _write_status_run(
        tmp_path / "child-blocked",
        status="running",
        stage_statuses={"probe": "passed"},
        last_stage="probe",
    )
    _write_status_run(
        run_dir / "raw-camera" / "camera",
        status="blocked",
        stage_statuses={"colmap": "blocked"},
        last_stage="colmap",
    )

    snapshot = read_progress_snapshot(run_dir, now=200)
    line = render_progress_line(snapshot, now=200)

    assert snapshot is not None
    assert snapshot.stage == "colmap"
    assert snapshot.stage_status == "blocked"
    assert "colmap | blocked" in line
    assert "probe | 完成" not in line


def test_raw_smoke_child_failure_keeps_root_history_in_canonical_order(tmp_path: Path) -> None:
    run_dir = _write_status_run(
        tmp_path / "raw-smoke-child",
        status="running",
        stage_statuses={"probe": "passed"},
        last_stage="probe",
    )
    _write_status_run(
        run_dir / "raw-smoke-input" / "smoke-input",
        status="blocked",
        stage_statuses={"colmap": "blocked"},
        last_stage="colmap",
    )
    stream = io.StringIO()
    reporter = TerminalProgress("plain", stream=stream, env={"TERM": "x"})
    reporter.bind_run(run_dir)
    reporter.poll_once(force=True)

    lines = stream.getvalue().splitlines()
    assert lines[0].startswith("probe | 完成")
    assert lines[1].startswith("colmap | blocked")
    assert all("probe | 完成" not in line for line in lines[1:])
    reporter.close()


def test_raw_child_completion_then_root_close_does_not_duplicate_or_reorder_lines(
    tmp_path: Path,
) -> None:
    run_dir = _write_status_run(
        tmp_path / "child-race",
        status="running",
        stage_statuses={"probe": "passed"},
        last_stage="probe",
    )
    raw_dir = run_dir / "raw-camera" / "camera"
    _write_status_run(
        raw_dir,
        status="running",
        stage_statuses={"colmap": "running"},
        active_stage="colmap",
        last_stage="colmap",
    )

    stream = io.StringIO()
    reporter = TerminalProgress("plain", stream=stream, env={"TERM": "x"}, now_fn=lambda: 200.0)
    reporter.bind_run(run_dir)
    reporter.poll_once(force=True)

    _write_status_run(
        raw_dir,
        status="passed",
        stage_statuses={"colmap": "passed"},
        last_stage="colmap",
    )
    reporter.poll_once()

    _write_status_run(
        run_dir,
        status="passed",
        stage_statuses={"probe": "passed", "colmap": "passed"},
        last_stage="colmap",
    )
    reporter.poll_once()

    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    assert "probe | 完成" in lines[0]
    assert lines[1].startswith("colmap |")
    assert "完成" not in lines[1]
    assert "colmap | 完成" in lines[2]
    assert sum("完成" in line for line in lines) == 2
    assert all("probe | 完成" not in line for line in lines[1:])
    reporter.close()


def test_raw_camera_reports_actual_colmap_and_current_attempt_timer(tmp_path: Path) -> None:
    run_dir = _make_raw_camera_run(tmp_path, raw_stage="colmap")

    snapshot = read_progress_snapshot(run_dir, now=200)
    line = render_progress_line(snapshot, now=200)

    assert snapshot is not None
    assert snapshot.stage == "colmap"
    assert snapshot.stage_active is True
    assert snapshot.stage_started == 100
    assert "colmap |" in line
    assert "camera staging" not in line
    assert "4/20" not in line
    assert not line.startswith("[")
    assert "阶段 00:01:40" in line


def test_current_stage_refreshes_in_place_on_tty(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, stage="colmap")

    class TTY(io.StringIO):
        def isatty(self):
            return True

    clock = [110.0]
    stream = TTY()
    reporter = TerminalProgress("auto", stream=stream, env={"TERM": "x"}, now_fn=lambda: clock[0])
    reporter.bind_run(run_dir)
    reporter.poll_once(force=True)
    clock[0] = 111.0
    reporter.poll_once()

    assert "\n" not in stream.getvalue()
    assert stream.getvalue().count("\r") == 2
    reporter.close()


def test_stage_switch_freezes_completion_and_starts_new_tty_line(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, stage="colmap", attempt_status="passed")

    class TTY(io.StringIO):
        def isatty(self):
            return True

    stream = TTY()
    reporter = TerminalProgress("auto", stream=stream, env={"TERM": "x"})
    reporter.bind_run(run_dir)
    reporter.poll_once(force=True)
    _activate_stage(run_dir, "formal-training")
    reporter.poll_once()

    output = stream.getvalue()
    assert "完成" in output
    assert "\nformal training" in output
    reporter.close()


def test_plain_stage_switch_is_low_noise_but_keeps_completion_line(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, stage="colmap", attempt_status="passed")
    clock = [100.0]
    stream = io.StringIO()
    reporter = TerminalProgress("plain", stream=stream, env={"TERM": "x"}, now_fn=lambda: clock[0])
    reporter.bind_run(run_dir)
    reporter.poll_once(force=True)
    first = stream.getvalue()
    reporter.poll_once()
    assert stream.getvalue() == first

    _activate_stage(run_dir, "formal-training")
    clock[0] = 101.0
    reporter.poll_once()
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    assert "colmap | 完成" in lines[0]
    assert "formal training" in lines[1]

    clock[0] = 162.0
    reporter.poll_once()
    assert len(stream.getvalue().splitlines()) == 3


def test_training_sampling_telemetry_is_real_iteration_not_quality_progress(tmp_path: Path) -> None:
    run_dir = _make_training_run(tmp_path)

    snapshot = read_progress_snapshot(run_dir, now=110)
    line = render_progress_line(snapshot, now=110)

    assert snapshot is not None
    assert snapshot.training.iteration == 3
    assert snapshot.training.total == 1000
    assert "camera sampling 3/1000" in line
    assert "quality" not in line
    assert line.startswith("[")


def test_training_progress_accepts_model_directory_elsewhere_within_run_root(tmp_path: Path) -> None:
    run_dir = _make_training_run(tmp_path)
    attempt = run_dir / "stages" / "convergence-smoke-training" / "attempt-0001"
    model = run_dir / "raw-formal-input" / "formal-input" / "model"
    _write_sampling_events(model, 3)
    (attempt / "executor" / "request.json").write_text(
        json.dumps(
            {
                "stage": "training",
                "iterations": 1000,
                "model_path": str(model),
            }
        ),
        encoding="utf-8",
    )

    snapshot = read_progress_snapshot(run_dir, now=110)

    assert snapshot is not None
    assert snapshot.training.iteration == 3
    assert snapshot.training.total == 1000


def test_completed_training_progress_remains_visible_after_later_stage_blocks(tmp_path: Path) -> None:
    run_dir = _make_training_run(tmp_path)
    summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary["status"] = "blocked"
    summary["active_stage"] = "conversion"
    summary["active_attempt"] = "attempt-0001"
    summary["last_stage"] = "conversion"
    summary["blocked"] = {"stage": "conversion", "error": "converter unavailable"}
    summary["stages"]["convergence-smoke-training"][0]["status"] = "passed"
    conversion_attempt = run_dir / "stages" / "conversion" / "attempt-0001"
    (conversion_attempt / "executor").mkdir(parents=True)
    (conversion_attempt / "request.json").write_text("{}", encoding="utf-8")
    (conversion_attempt / "result.json").write_text("{}", encoding="utf-8")
    summary["stages"]["conversion"] = [
        {
            "attempt": "attempt-0001",
            "status": "blocked",
            "result_path": "stages/conversion/attempt-0001/result.json",
        }
    ]
    (run_dir / "run.json").write_text(json.dumps(summary), encoding="utf-8")

    snapshot = read_progress_snapshot(run_dir, now=110)

    assert snapshot is not None
    assert snapshot.stage == "conversion"
    assert snapshot.training.iteration == 3
    assert snapshot.training.total == 1000


def test_training_telemetry_fails_closed_for_malformed_symlink_outside_and_old_attempt(
    tmp_path: Path,
) -> None:
    malformed = _make_training_run(tmp_path / "malformed")
    malformed_events = malformed / "stages" / "convergence-smoke-training" / "attempt-0001" / "model" / "camera_sampling_telemetry-v1.jsonl"
    malformed_events.write_text("not-json\n", encoding="utf-8")
    assert read_progress_snapshot(malformed).training.iteration is None

    symlinked = _make_training_run(tmp_path / "symlinked")
    symlinked_model = symlinked / "stages" / "convergence-smoke-training" / "attempt-0001" / "model"
    symlinked_events = symlinked_model / "camera_sampling_telemetry-v1.jsonl"
    outside_events = tmp_path / "symlinked-events.jsonl"
    outside_events.write_text("{}\n", encoding="utf-8")
    symlinked_events.unlink()
    try:
        symlinked_events.symlink_to(outside_events)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows symlink privilege is unavailable")
        raise
    assert read_progress_snapshot(symlinked).training.iteration is None

    outside = _make_training_run(tmp_path / "outside")
    outside_attempt = outside / "stages" / "convergence-smoke-training" / "attempt-0001"
    outside_model = tmp_path / "outside-model"
    _write_sampling_events(outside_model, 1)
    (outside_attempt / "executor" / "request.json").write_text(
        json.dumps({"stage": "training", "iterations": 1000, "model_path": str(outside_model)}),
        encoding="utf-8",
    )
    assert read_progress_snapshot(outside).training.iteration is None

    old_attempt = _make_training_run(tmp_path / "old-attempt")
    current_attempt = old_attempt / "stages" / "convergence-smoke-training" / "attempt-0002"
    current_model = current_attempt / "model"
    (current_attempt / "executor").mkdir(parents=True)
    (current_attempt / "executor" / "request.json").write_text(
        json.dumps({"stage": "training", "iterations": 1000, "model_path": str(current_model)}),
        encoding="utf-8",
    )
    (old_attempt / "run.json").write_text(
        json.dumps(
            {
                "created_at": _stamp(0),
                "updated_at": _stamp(110),
                "status": "running",
                "active_stage": "convergence-smoke-training",
                "active_attempt": "attempt-0002",
                "stages": {
                    "convergence-smoke-training": [
                        {"attempt": "attempt-0001", "status": "blocked"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    assert read_progress_snapshot(old_attempt).training.iteration is None


def test_atomic_json_transient_read_error_keeps_last_good_snapshot(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    stream = io.StringIO()
    reporter = TerminalProgress("plain", stream=stream, env={"TERM": "x"})
    reporter.bind_run(run_dir)
    reporter.poll_once(force=True)
    good = stream.getvalue()
    (run_dir / "run.json").write_text("{", encoding="utf-8")
    reporter.poll_once()
    assert stream.getvalue() == good


def test_tty_plain_off_no_color_and_stage_change_heartbeat_behavior(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, stage="conversion")
    class TTY(io.StringIO):
        def isatty(self):
            return True

    tty_stream = TTY()
    tty = TerminalProgress("auto", stream=tty_stream, env={"TERM": "x"})
    tty.bind_run(run_dir)
    tty.poll_once(force=True)
    tty.close()
    assert "\r" in tty_stream.getvalue()
    assert "\x1b[" not in tty_stream.getvalue()

    no_color_stream = TTY()
    no_color = TerminalProgress("auto", stream=no_color_stream, env={"TERM": "x", "NO_COLOR": "1"})
    no_color.bind_run(run_dir)
    no_color.poll_once(force=True)
    assert "\x1b[" not in no_color_stream.getvalue()
    assert "\n" in no_color_stream.getvalue()

    off_stream = io.StringIO()
    off = TerminalProgress("off", stream=off_stream)
    off.bind_run(run_dir)
    off.start()
    off.close()
    assert off_stream.getvalue() == ""
    assert off._thread is None

    heartbeat_stream = io.StringIO()
    clock = [100.0]
    plain = TerminalProgress("plain", stream=heartbeat_stream, env={"TERM": "x"}, now_fn=lambda: clock[0])
    plain.bind_run(run_dir)
    plain.poll_once(force=True)
    first = heartbeat_stream.getvalue()
    plain.poll_once()
    assert heartbeat_stream.getvalue() == first
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "created_at": _stamp(0),
                "updated_at": _stamp(101),
                "status": "running",
                "active_stage": "formal-training",
                "active_attempt": "attempt-0001",
                "stages": {
                    "conversion": [
                        {"attempt": "attempt-0001", "status": "passed"},
                    ],
                    "formal-training": [
                        {"attempt": "attempt-0001", "status": "running"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    clock[0] = 101.0
    plain.poll_once()
    assert "formal training" in heartbeat_stream.getvalue()
    clock[0] = 162.0
    plain.poll_once()
    assert heartbeat_stream.getvalue().count("\n") == 3


def test_download_callback_reports_only_sizes_without_url(tmp_path: Path) -> None:
    stream = io.StringIO()
    reporter = TerminalProgress("plain", stream=stream, env={"TERM": "x"})
    reporter.download_callback(1024, 4096)

    output = stream.getvalue()
    assert "download 1.0 KiB / 4.0 KiB" in output
    assert "token" not in output
    assert "http" not in output


def test_download_callback_without_content_length_uses_mib_only() -> None:
    stream = io.StringIO()
    reporter = TerminalProgress("plain", stream=stream, env={"TERM": "x"})
    reporter.download_callback(3 * 1024 * 1024, None)

    assert "download 3.0 MiB" in stream.getvalue()
    assert "/" not in stream.getvalue()


def test_background_thread_stops_and_joins(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    stream = io.StringIO()
    reporter = TerminalProgress("plain", stream=stream, env={"TERM": "x"}, interval=0.05)
    reporter.bind_run(run_dir)
    reporter.start()
    thread = reporter._thread
    reporter.close()

    assert thread is not None
    assert not thread.is_alive()
    assert not [item for item in threading.enumerate() if item.name == "video-to-3dgs-progress"]


def test_reporter_output_failure_does_not_escape_download_callback() -> None:
    class BrokenStream(io.StringIO):
        def write(self, _value: str) -> int:
            raise OSError("reporter output failed")

    reporter = TerminalProgress("plain", stream=BrokenStream(), env={"TERM": "x"})

    reporter.download_callback(1024, None)
    reporter.close()
