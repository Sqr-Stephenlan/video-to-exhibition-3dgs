from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from scripts.longsplat.conversion_observability import (
    AppendOnlyProgress,
    ConversionStreamCapture,
)
from scripts.longsplat.runner import EffectiveCommands, LongSplatConfig, run_conversion


def _config(tmp_path: Path, *, total: int = 30_000) -> LongSplatConfig:
    return LongSplatConfig(
        source_path=str(tmp_path / "input"),
        model_path=str(tmp_path / "model"),
        iterations=total,
        convert_iteration=total,
        seed=0,
        backend_mode="research_local",
    )


def _run_fake(
    tmp_path: Path,
    script: str,
    *,
    total: int = 30_000,
    with_observation_files: bool = True,
):
    evidence = tmp_path / "conversion-attempt"
    evidence.mkdir(parents=True)
    kwargs = {}
    if with_observation_files:
        kwargs = {
            "observability_root": evidence,
            "live_stdout_path": evidence / "conversion-stdout-live.log",
            "live_stderr_path": evidence / "conversion-stderr-live.log",
            "progress_path": evidence / "conversion-progress-v1.jsonl",
        }
    command = (sys.executable, "-u", "-c", script)
    with mock.patch("scripts.longsplat.runner._check_repo", return_value=tmp_path), mock.patch(
        "scripts.longsplat.runner._check_python"
    ):
        result = run_conversion(
            tmp_path,
            _config(tmp_path, total=total),
            commands=EffectiveCommands(train=(), convert=command),
            **kwargs,
        )
    return result, evidence


def _read_progress(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_interleaved_streams_large_tqdm_and_more_than_thirty_markers(tmp_path: Path) -> None:
    script = r'''
import sys
for index in range(1, 36):
    sys.stderr.write("\r" + ("tqdm-progress " * 20) + str(index))
    sys.stderr.flush()
    marker = (
        "CONVERSION_SHAPE_TELEMETRY "
        f"iteration={index * 500} anisotropy_loss=1.25e-03 "
        "weight=0.01 soft_limit=30.0\n"
    )
    split = 11
    sys.stdout.write(marker[:split])
    sys.stdout.flush()
    sys.stdout.write(marker[split:])
    sys.stdout.flush()
for _ in range(12000):
    sys.stderr.write("\r" + ("x" * 160))
sys.stderr.flush()
'''
    result, evidence = _run_fake(tmp_path, script)

    assert result.returncode == 0
    assert len(result.stdout) <= 2_000
    assert len(result.stderr) <= 2_000
    live_stdout = evidence / "conversion-stdout-live.log"
    live_stderr = evidence / "conversion-stderr-live.log"
    assert live_stdout.is_file()
    assert live_stderr.is_file()
    stdout_text = live_stdout.read_text(encoding="utf-8")
    stderr_text = live_stderr.read_text(encoding="utf-8")
    assert stdout_text.count("CONVERSION_SHAPE_TELEMETRY") == 35
    assert "tqdm-progress" in stderr_text
    assert len(stderr_text) > 1_000_000

    records = _read_progress(evidence / "conversion-progress-v1.jsonl")
    iterations = [record["iteration"] for record in records if record["phase"] == "iteration"]
    assert iterations == [index * 500 for index in range(1, 36)]
    assert all(record["total"] == 30_000 for record in records)
    assert records[0]["phase"] == "child_started"
    assert records[-1]["phase"] == "child_exited"


def test_large_tqdm_stream_does_not_fsync_per_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = r'''
import sys
for _ in range(12000):
    sys.stderr.write("\r" + ("x" * 160))
sys.stderr.flush()
'''
    fsync_calls: list[int] = []
    monkeypatch.setattr(
        "scripts.longsplat.runner._os.fsync",
        lambda file_descriptor: fsync_calls.append(file_descriptor),
    )

    result, _evidence = _run_fake(tmp_path, script)

    assert result.returncode == 0
    assert len(fsync_calls) <= 6


def test_duplicate_regression_out_of_range_and_malformed_markers_are_not_progress() -> None:
    captured: list[tuple[str, str, dict]] = []
    capture = ConversionStreamCapture(
        stream="stdout",
        on_marker=lambda stream, raw, fields: captured.append((stream, raw, fields)),
    )
    capture.feed(
        b"CONVERSION_SHAPE_TELEMETRY iteration=1000 value=1\n"
        b"CONVERSION_SHAPE_TELEMETRY iteration="
    )
    capture.feed(
        b"1000 value=duplicate\n"
        b"CONVERSION_SHAPE_TELEMETRY iteration=500 value=regression\n"
        b"CONVERSION_SHAPE_TELEMETRY iteration=oops value=bad\n"
        b"CONVERSION_SHAPE_TELEMETRY iteration=31000 value=out\n"
    )
    capture.finish()
    assert [item[2]["iteration"] for item in captured] == [1000, 1000, 500, 31000]


def test_progress_writer_enforces_integer_bounds_and_strict_monotonicity(tmp_path: Path) -> None:
    path = tmp_path / "conversion-progress-v1.jsonl"
    writer = AppendOnlyProgress(path, total=30_000, started_monotonic=time.monotonic())
    assert writer.append_marker(
        stream="stdout",
        raw_marker="CONVERSION_SHAPE_TELEMETRY iteration=1000",
        fields={"iteration": 1000},
        elapsed=1.0,
    ) is True
    for iteration in (1000, 999, 30_001, 0, True):
        assert writer.append_marker(
            stream="stdout",
            raw_marker=f"iteration={iteration}",
            fields={"iteration": iteration},
            elapsed=2.0,
        ) is False
    assert writer.append_marker(
        stream="stdout",
        raw_marker="CONVERSION_SHAPE_TELEMETRY iteration=2000",
        fields={"iteration": 2000},
        elapsed=3.0,
    ) is True
    writer.close()
    records = _read_progress(path)
    assert [record["iteration"] for record in records] == [1000, 2000]
    assert all(record["schema_version"] == "conversion-progress-v1" for record in records)


def test_nonzero_and_sigterm_child_are_recorded_without_validator_semantics(tmp_path: Path) -> None:
    nonzero, evidence = _run_fake(
        tmp_path / "nonzero",
        "import sys; print('failed-child'); sys.exit(7)",
    )
    assert nonzero.returncode == 7
    assert _read_progress(evidence / "conversion-progress-v1.jsonl")[-1]["returncode"] == 7

    terminated, term_evidence = _run_fake(
        tmp_path / "sigterm",
        "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
    )
    assert terminated.returncode == -signal.SIGTERM
    assert _read_progress(term_evidence / "conversion-progress-v1.jsonl")[-1]["phase"] == "child_exited"


def test_optional_observation_paths_leave_direct_caller_usable(tmp_path: Path) -> None:
    result, evidence = _run_fake(
        tmp_path,
        "import sys; sys.stdout.write('ok\\n'); sys.stdout.flush()",
        with_observation_files=False,
    )
    assert result.returncode == 0
    assert result.stdout.endswith("ok\n")
    assert list(evidence.iterdir()) == []


def test_keyboard_interrupt_reaps_only_current_child_and_closes_files(tmp_path: Path) -> None:
    evidence = tmp_path / "interrupt-attempt"
    evidence.mkdir()
    command = (
        sys.executable,
        "-u",
        "-c",
        "import time; time.sleep(30)",
    )
    original_popen = subprocess.Popen
    child: dict[str, subprocess.Popen] = {}

    def capture_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        child["process"] = process
        return process

    class InterruptingSelector:
        def __init__(self):
            self._keys = {}

        def register(self, fileobj, _events, data):
            self._keys[fileobj] = SimpleNamespace(fileobj=fileobj, data=data)

        def get_map(self):
            return self._keys

        def select(self, timeout=None):
            del timeout
            raise KeyboardInterrupt

        def unregister(self, fileobj):
            self._keys.pop(fileobj, None)

        def close(self):
            self._keys.clear()

    with mock.patch("scripts.longsplat.runner._check_repo", return_value=tmp_path), mock.patch(
        "scripts.longsplat.runner._check_python"
    ), mock.patch("scripts.longsplat.runner.subprocess.Popen", side_effect=capture_popen), mock.patch(
        "scripts.longsplat.runner.selectors.DefaultSelector", InterruptingSelector
    ):
        with pytest.raises(KeyboardInterrupt):
            run_conversion(
                tmp_path,
                _config(tmp_path),
                commands=EffectiveCommands(train=(), convert=command),
                observability_root=evidence,
                live_stdout_path=evidence / "conversion-stdout-live.log",
                live_stderr_path=evidence / "conversion-stderr-live.log",
                progress_path=evidence / "conversion-progress-v1.jsonl",
            )
    assert child["process"].poll() is not None
    assert child["process"].stdout is not None
    assert child["process"].stderr is not None
    with pytest.raises(ValueError):
        child["process"].stdout.fileno()
    with pytest.raises(ValueError):
        child["process"].stderr.fileno()


def test_reader_exception_reaps_child_and_records_parent_error(tmp_path: Path) -> None:
    evidence = tmp_path / "reader-error-attempt"
    evidence.mkdir()
    command = (sys.executable, "-u", "-c", "import time; print('x'); time.sleep(30)")
    original_read = os.read
    child_pipe_fds: set[int] = set()
    original_popen = subprocess.Popen

    def capture_popen(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        child_pipe_fds.update({process.stdout.fileno(), process.stderr.fileno()})
        return process

    def broken_read(fd, size):
        if fd in child_pipe_fds:
            raise RuntimeError("fake reader failure")
        return original_read(fd, size)

    with mock.patch("scripts.longsplat.runner._check_repo", return_value=tmp_path), mock.patch(
        "scripts.longsplat.runner._check_python"
    ), mock.patch("scripts.longsplat.runner.subprocess.Popen", side_effect=capture_popen), mock.patch(
        "scripts.longsplat.runner._os.read", side_effect=broken_read
    ):
        with pytest.raises(RuntimeError, match="fake reader failure"):
            run_conversion(
                tmp_path,
                _config(tmp_path),
                commands=EffectiveCommands(train=(), convert=command),
                observability_root=evidence,
                live_stdout_path=evidence / "conversion-stdout-live.log",
                live_stderr_path=evidence / "conversion-stderr-live.log",
                progress_path=evidence / "conversion-progress-v1.jsonl",
            )
    progress = _read_progress(evidence / "conversion-progress-v1.jsonl")
    assert progress[-1]["phase"] == "parent_error"
    assert progress[-1]["error_type"] == "RuntimeError"
