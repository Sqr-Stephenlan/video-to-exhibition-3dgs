from __future__ import annotations

import io
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from scripts.longsplat.terminal_progress import (
    ProgressSnapshot,
    TerminalProgress,
    read_progress_snapshot,
    render_progress_line,
)


def _stamp(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def _make_run(tmp_path: Path, *, stage: str = "conversion", status: str = "running") -> Path:
    run_dir = tmp_path / "run"
    attempt = run_dir / "stages" / stage / "attempt-0001"
    (attempt / "executor").mkdir(parents=True)
    stage_order = [
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
                            "status": "running",
                            "result_path": f"stages/{stage}/attempt-0001/result.json",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
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
    assert "17/20" in line
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
        assert "17/20" in line


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
                "stages": {},
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
