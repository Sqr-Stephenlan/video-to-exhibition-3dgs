"""Read-only terminal progress reporting for the canonical one-click chain."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO


_PROGRESS_MODES = {"auto", "plain", "off"}
_PROGRESS_WIDTH = 20
_HEARTBEAT_SECONDS = 60.0
_CONVERSION_WARNING_SECONDS = 15 * 60.0
_SIDECAR_TAIL_BYTES = 1024 * 1024
_SCHEMA = "conversion-progress-v1"
_SAMPLING_SCHEMA = "camera-sampling-telemetry-v1"
_SAMPLING_EVENTS_NAME = "camera_sampling_telemetry-v1.jsonl"
_TRAINING_STAGES = {"convergence-smoke-training", "formal-training"}
_FAILURE_STATUSES = {"blocked", "failed"}
_TERMINAL_STATUSES = {"passed", "blocked", "failed"}


@dataclass(frozen=True)
class ConversionProgress:
    iteration: int | None
    total: int | None
    marker_timestamp: float | None


@dataclass(frozen=True)
class TrainingProgress:
    iteration: int | None
    total: int | None
    marker_timestamp: float | None = None


@dataclass(frozen=True)
class ProgressSnapshot:
    stage_order: tuple[str, ...]
    stage: str | None
    stage_position: int | None
    run_started: float | None
    stage_started: float | None
    status: str | None
    conversion: ConversionProgress
    downloaded_bytes: int | None
    download_total: int | None
    stage_active: bool = True
    stage_attempt: str | None = None
    stage_status: str | None = None
    stage_finished: float | None = None
    training: TrainingProgress = TrainingProgress(None, None)
    stage_observations: tuple[_StageObservation, ...] = ()


@dataclass(frozen=True)
class _StageSelection:
    stage: str | None
    root: Path | None
    summary: Mapping[str, Any]
    active: bool
    status: str | None


@dataclass(frozen=True)
class _StageObservation:
    stage: str
    attempt: str | None
    status: str | None
    active: bool


@dataclass(frozen=True)
class _StageTiming:
    attempt: str | None
    started: float | None
    finished: float | None


def _read_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.timestamp()


def _contained_file(path: Path, root: Path) -> Path | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    current = resolved_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return None
    return resolved


def _contained_directory(path: Path, root: Path) -> Path | None:
    if path.is_symlink() or not path.is_dir():
        return None
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(resolved_root)
    except (OSError, ValueError):
        return None
    current = resolved_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return None
    return resolved


def _latest_attempt_name(
    summary: Mapping[str, Any],
    *,
    stage: str,
) -> str | None:
    stages = summary.get("stages")
    entries = stages.get(stage) if isinstance(stages, Mapping) else None
    if not isinstance(entries, list) or not entries:
        return None
    entry = entries[-1]
    attempt = entry.get("attempt") if isinstance(entry, Mapping) else None
    return attempt if isinstance(attempt, str) else None


def _active_attempt(summary: Mapping[str, Any], *, stage: str) -> str | None:
    active_stage = summary.get("active_stage")
    if not isinstance(active_stage, str):
        active_stage = summary.get("current_intended_stage")
    if active_stage != stage:
        return None
    attempt = summary.get("active_attempt")
    return attempt if isinstance(attempt, str) else None


def _stage_entry_for_attempt(
    entries: list[Any],
    *,
    attempt: str,
) -> Mapping[str, Any] | None:
    for entry in reversed(entries):
        if isinstance(entry, Mapping) and entry.get("attempt") == attempt:
            return entry
    return None


def _stage_start(
    summary: Mapping[str, Any],
    *,
    stage: str | None,
    raw_dir: Path | None = None,
) -> float | None:
    if not isinstance(stage, str):
        return None
    source_root = raw_dir
    source_summary = summary
    if raw_dir is not None:
        raw_path = _contained_file(raw_dir / "run.json", raw_dir)
        raw_summary = _read_json(raw_path) if raw_path is not None else None
        if raw_summary is None:
            return None
        source_summary = raw_summary
    timing = _stage_timing(source_root, source_summary, stage)
    return timing.started


def _read_sidecar(path: Path) -> ConversionProgress:
    if path.is_symlink() or not path.is_file():
        return ConversionProgress(None, None, None)
    last_iteration: int | None = None
    total: int | None = None
    marker_timestamp: float | None = None
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - _SIDECAR_TAIL_BYTES)
            handle.seek(start)
            payload = handle.read(size - start).decode("utf-8", errors="replace")
        if start:
            separator = payload.find("\n")
            if separator < 0:
                return ConversionProgress(None, None, None)
            payload = payload[separator + 1 :]
        for line in payload.splitlines():
            try:
                record = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(record, Mapping) or record.get("schema_version") != _SCHEMA:
                continue
            record_total = record.get("total")
            if (
                isinstance(record_total, bool)
                or not isinstance(record_total, int)
                or record_total <= 0
            ):
                continue
            if record.get("phase") != "iteration":
                continue
            iteration = record.get("iteration")
            if (
                isinstance(iteration, bool)
                or not isinstance(iteration, int)
                or iteration < 1
                or iteration > record_total
                or (last_iteration is not None and iteration <= last_iteration)
            ):
                continue
            last_iteration = iteration
            total = record_total
            marker_timestamp = _timestamp(record.get("timestamp_utc"))
    except OSError:
        return ConversionProgress(None, None, None)
    return ConversionProgress(last_iteration, total, marker_timestamp)


def _conversion_sidecar(summary: Mapping[str, Any], *, run_dir: Path) -> Path | None:
    active_attempt = summary.get("active_attempt")
    active_stage = summary.get("active_stage")
    if not isinstance(active_stage, str):
        active_stage = summary.get("current_intended_stage")
    if isinstance(active_attempt, str) and active_stage == "conversion":
        candidate = run_dir / "stages" / "conversion" / active_attempt / "executor" / "conversion-progress-v1.jsonl"
        return _contained_file(candidate, run_dir)
    latest_attempt = _latest_attempt_name(summary, stage="conversion")
    if latest_attempt is not None:
        candidate = run_dir / "stages" / "conversion" / latest_attempt / "executor" / "conversion-progress-v1.jsonl"
        checked = _contained_file(candidate, run_dir)
        return checked
    return None


def _active_summary_stage(summary: Mapping[str, Any], stage_order: tuple[str, ...]) -> str | None:
    for value in (summary.get("active_stage"), summary.get("current_intended_stage")):
        if isinstance(value, str) and value in stage_order:
            return value
    return None


def _latest_summary_stage(
    summary: Mapping[str, Any],
    stage_order: tuple[str, ...],
) -> tuple[str | None, str | None]:
    stages = summary.get("stages")
    if not isinstance(stages, Mapping):
        return None, None
    for value in reversed(stage_order):
        entries = stages.get(value)
        if not isinstance(entries, list) or not entries:
            continue
        entry = entries[-1]
        status = entry.get("status") if isinstance(entry, Mapping) else None
        return value, status if isinstance(status, str) else None
    return None, None


def _latest_stage_status(summary: Mapping[str, Any], stage: str) -> str | None:
    stages = summary.get("stages")
    entries = stages.get(stage) if isinstance(stages, Mapping) else None
    if not isinstance(entries, list) or not entries:
        return None
    active_attempt = _active_attempt(summary, stage=stage)
    if active_attempt is not None:
        # A retry can be active before its new ledger entry is visible.  Never
        # expose the previous attempt's terminal status as the new attempt's
        # status during that race.
        entry = _stage_entry_for_attempt(entries, attempt=active_attempt)
        if entry is None:
            return None
    else:
        entry = entries[-1]
    status = entry.get("status") if isinstance(entry, Mapping) else None
    return status if isinstance(status, str) else None


def _summary_active_stage(
    summary: Mapping[str, Any],
    stage_order: tuple[str, ...],
) -> str | None:
    active = summary.get("active_stage")
    if isinstance(active, str) and active in stage_order:
        return active
    intended = summary.get("current_intended_stage")
    if isinstance(intended, str) and intended in stage_order:
        return intended
    return None


def _summary_stage_observations(
    summary: Mapping[str, Any],
    *,
    stage_order: tuple[str, ...],
) -> tuple[_StageObservation, ...]:
    stages = summary.get("stages")
    if not isinstance(stages, Mapping):
        stages = {}
    active_stage = _summary_active_stage(summary, stage_order)
    observations: list[_StageObservation] = []
    for stage in stage_order:
        entries = stages.get(stage)
        if not isinstance(entries, list) or not entries:
            entries = []
        active = active_stage == stage
        entry: Mapping[str, Any] | None
        if active:
            active_attempt = _active_attempt(summary, stage=stage)
            if active_attempt is not None:
                entry = _stage_entry_for_attempt(entries, attempt=active_attempt)
                attempt = active_attempt
            else:
                entry = entries[-1] if entries else None
                attempt = (
                    entry.get("attempt")
                    if isinstance(entry, Mapping) and isinstance(entry.get("attempt"), str)
                    else None
                )
        elif entries:
            entry = entries[-1]
            attempt = (
                entry.get("attempt")
                if isinstance(entry, Mapping) and isinstance(entry.get("attempt"), str)
                else None
            )
        else:
            entry = None
            attempt = None
        if entry is None and not active:
            continue
        status = entry.get("status") if isinstance(entry, Mapping) else None
        observations.append(
            _StageObservation(
                stage=stage,
                attempt=attempt,
                status=status if isinstance(status, str) else None,
                active=active,
            )
        )
    return tuple(observations)


def _attempt_sort_key(attempt: str | None) -> tuple[int, str]:
    if not isinstance(attempt, str):
        return (-1, "")
    prefix, separator, suffix = attempt.rpartition("-")
    if separator and suffix.isdigit():
        return (int(suffix), prefix)
    return (-1, attempt)


def _preferred_status(observations: list[_StageObservation]) -> str | None:
    for status in ("failed", "blocked", "passed"):
        if any(item.status == status for item in observations):
            return status
    for item in observations:
        if item.status is not None:
            return item.status
    return None


def _merge_stage_observations(
    summaries: tuple[Mapping[str, Any], ...],
    *,
    stage_order: tuple[str, ...],
) -> tuple[_StageObservation, ...]:
    grouped: dict[str, list[_StageObservation]] = {stage: [] for stage in stage_order}
    for summary in summaries:
        for observation in _summary_stage_observations(summary, stage_order=stage_order):
            grouped[observation.stage].append(observation)

    merged: list[_StageObservation] = []
    for stage in stage_order:
        candidates = grouped[stage]
        if not candidates:
            continue
        active = [item for item in candidates if item.active]
        if active:
            attempts = [item.attempt for item in active if item.attempt is not None]
            selected_attempt = max(attempts, key=_attempt_sort_key) if attempts else None
            if selected_attempt is not None:
                candidates = [item for item in active if item.attempt == selected_attempt]
            else:
                candidates = active
            merged_status = _preferred_status(candidates)
            merged.append(
                _StageObservation(
                    stage=stage,
                    attempt=selected_attempt or candidates[0].attempt,
                    status=merged_status,
                    active=merged_status not in _TERMINAL_STATUSES,
                )
            )
            continue
        merged.append(
            _StageObservation(
                stage=stage,
                attempt=candidates[-1].attempt,
                status=_preferred_status(candidates),
                active=False,
            )
        )
    return tuple(merged)


def _terminal_summary_stage(
    summary: Mapping[str, Any],
    stage_order: tuple[str, ...],
) -> tuple[str | None, str | None]:
    overall_status = summary.get("status")
    if overall_status not in _FAILURE_STATUSES:
        return None, None
    marker = summary.get(overall_status)
    marker_stage = marker.get("stage") if isinstance(marker, Mapping) else None
    stage = marker_stage if isinstance(marker_stage, str) else summary.get("last_stage")
    if not isinstance(stage, str) or stage not in stage_order:
        return None, None
    attempt_status = _latest_stage_status(summary, stage)
    if attempt_status in _FAILURE_STATUSES:
        return stage, attempt_status
    if _summary_active_stage(summary, stage_order) == stage and isinstance(
        summary.get("active_attempt"), str
    ):
        # The overall summary can still carry the previous attempt's terminal
        # status while a retry is being opened.  The active attempt remains
        # unresolved until its own ledger entry appears.
        return None, None
    return stage, overall_status


def _summary_selection(
    summary: Mapping[str, Any],
    *,
    stage_order: tuple[str, ...],
    root: Path | None,
) -> _StageSelection:
    terminal_stage, terminal_status = _terminal_summary_stage(summary, stage_order)
    if terminal_stage is not None:
        return _StageSelection(
            terminal_stage,
            root,
            summary,
            False,
            terminal_status,
        )

    direct_stage = _active_summary_stage(summary, stage_order)
    if direct_stage is not None:
        status = _latest_stage_status(summary, direct_stage)
        overall_status = summary.get("status")
        if status is None and _active_attempt(summary, stage=direct_stage) is None:
            status = overall_status
        elif (
            status not in _TERMINAL_STATUSES
            and overall_status in _FAILURE_STATUSES
            and _stage_entry_for_attempt(
                (
                    summary.get("stages", {}).get(direct_stage, [])
                    if isinstance(summary.get("stages"), Mapping)
                    else []
                ),
                attempt=str(summary.get("active_attempt")),
            )
            is not None
        ):
            # Legacy summaries may leave the active attempt marked running
            # while the overall run already records the authoritative stop.
            # A real retry with no new ledger entry is handled by the branch
            # above and remains unresolved instead of inheriting old failure.
            status = overall_status
        return _StageSelection(
            direct_stage,
            root,
            summary,
            True,
            status if isinstance(status, str) else None,
        )

    completed_stage, completed_status = _latest_summary_stage(summary, stage_order)
    if completed_stage is not None:
        return _StageSelection(
            completed_stage,
            root,
            summary,
            False,
            completed_status,
        )
    return _StageSelection(None, root, summary, False, None)


def _stage_selection(
    summary: Mapping[str, Any],
    *,
    stage_order: tuple[str, ...],
    root: Path | None,
    raw_camera: Mapping[str, Any] | None,
    raw_camera_dir: Path | None,
    raw_smoke: Mapping[str, Any] | None,
    raw_smoke_dir: Path | None,
) -> _StageSelection:
    selections = [
        _summary_selection(summary, stage_order=stage_order, root=root),
    ]
    for child, child_dir in (
        (raw_camera, raw_camera_dir),
        (raw_smoke, raw_smoke_dir),
    ):
        if isinstance(child, Mapping):
            selections.append(_summary_selection(child, stage_order=stage_order, root=child_dir))

    available = [selection for selection in selections if selection.stage is not None]
    if not available:
        return _StageSelection(None, root, summary, False, None)

    def stage_rank(selection: _StageSelection) -> int:
        return stage_order.index(selection.stage) if selection.stage is not None else -1

    failures = [selection for selection in available if selection.status in _FAILURE_STATUSES]
    if failures:
        return max(failures, key=stage_rank)

    active = [selection for selection in available if selection.active]
    if active:
        return max(active, key=stage_rank)
    return max(available, key=stage_rank)


def _active_stage(
    summary: Mapping[str, Any],
    *,
    stage_order: tuple[str, ...],
    raw_camera: Mapping[str, Any] | None,
    raw_smoke: Mapping[str, Any] | None,
) -> tuple[str | None, Path | None]:
    selected = _stage_selection(
        summary,
        stage_order=stage_order,
        root=None,
        raw_camera=raw_camera,
        raw_camera_dir=None,
        raw_smoke=raw_smoke,
        raw_smoke_dir=None,
    )
    return selected.stage, None


def _stage_timing(
    root: Path | None,
    summary: Mapping[str, Any],
    stage: str | None,
) -> _StageTiming:
    if root is None or not isinstance(stage, str):
        return _StageTiming(None, None, None)
    active_stage = summary.get("active_stage")
    if not isinstance(active_stage, str):
        active_stage = summary.get("current_intended_stage")
    active = active_stage == stage
    attempt = summary.get("active_attempt") if active else _latest_attempt_name(summary, stage=stage)
    if not isinstance(attempt, str):
        return _StageTiming(None, None, None)
    attempt_dir = _contained_directory(root / "stages" / stage / attempt, root)
    if attempt_dir is None:
        return _StageTiming(attempt, None, None)
    request_path = _contained_file(attempt_dir / "request.json", root)
    request = _read_json(request_path) if request_path is not None else None
    started = _timestamp(request.get("started_at")) if request is not None else None
    result_path = _contained_file(attempt_dir / "result.json", root)
    result = _read_json(result_path) if result_path is not None else None
    finished = _timestamp(result.get("finished_at")) if result is not None else None
    return _StageTiming(attempt, started, finished)


def _read_training_events(path: Path, total: int) -> TrainingProgress:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            start = max(0, size - _SIDECAR_TAIL_BYTES)
            handle.seek(start)
            raw = handle.read(size - start)
        if not raw or not raw.endswith(b"\n"):
            return TrainingProgress(None, None)
        payload = raw
        if start:
            separator = payload.find(b"\n")
            if separator < 0:
                return TrainingProgress(None, None)
            payload = payload[separator + 1 :]
        text = payload.decode("utf-8")
    except (OSError, UnicodeError):
        return TrainingProgress(None, None)

    previous: int | None = None
    latest: int | None = None
    for line in text.splitlines():
        if not line:
            return TrainingProgress(None, None)
        try:
            record = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError):
            return TrainingProgress(None, None)
        if not isinstance(record, Mapping) or record.get("schema_version") != _SAMPLING_SCHEMA:
            return TrainingProgress(None, None)
        iteration = record.get("iteration")
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < 1
            or iteration > total
            or (previous is not None and iteration != previous + 1)
        ):
            return TrainingProgress(None, None)
        previous = iteration
        latest = iteration
    if latest is None:
        return TrainingProgress(None, None)
    try:
        marker_timestamp = path.stat().st_mtime
    except OSError:
        marker_timestamp = None
    return TrainingProgress(latest, total, marker_timestamp)


def _latest_training_stage(
    summary: Mapping[str, Any],
    *,
    stage_order: tuple[str, ...],
) -> str | None:
    stages = summary.get("stages")
    if not isinstance(stages, Mapping):
        return None
    for candidate in reversed(stage_order):
        if candidate not in _TRAINING_STAGES:
            continue
        entries = stages.get(candidate)
        if not isinstance(entries, list) or not entries:
            continue
        latest = entries[-1]
        status = latest.get("status") if isinstance(latest, Mapping) else None
        if str(status or "").strip().lower() in {
            "running",
            "passed",
            "complete",
            "completed",
            "accepted",
        }:
            return candidate
    return None


def _training_progress(
    root: Path | None,
    summary: Mapping[str, Any],
    stage: str | None,
) -> TrainingProgress:
    active_stage = summary.get("active_stage")
    if not isinstance(active_stage, str):
        active_stage = summary.get("current_intended_stage")
    if root is None or stage not in _TRAINING_STAGES:
        return TrainingProgress(None, None)
    attempt = summary.get("active_attempt") if active_stage == stage else _latest_attempt_name(summary, stage=stage)
    if not isinstance(attempt, str):
        return TrainingProgress(None, None)
    attempt_dir = _contained_directory(root / "stages" / stage / attempt, root)
    if attempt_dir is None:
        return TrainingProgress(None, None)
    request_path = _contained_file(attempt_dir / "executor" / "request.json", root)
    request = _read_json(request_path) if request_path is not None else None
    if request is None or request.get("stage") != "training":
        return TrainingProgress(None, None)
    total = request.get("iterations")
    model_value = request.get("model_path")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or not isinstance(model_value, str)
    ):
        return TrainingProgress(None, None)
    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = root / model_path
    model = _contained_directory(model_path, root)
    if model is None:
        return TrainingProgress(None, None)
    events = _contained_file(model / _SAMPLING_EVENTS_NAME, root)
    if events is None:
        return TrainingProgress(None, None)
    return _read_training_events(events, total)


def read_progress_snapshot(run_dir: str | Path, *, now: float | None = None) -> ProgressSnapshot | None:
    """Read only current-run JSON/sidecar state; malformed transient files are ignored."""

    root = Path(run_dir)
    if root.is_symlink() or not root.is_dir():
        return None
    summary_path = _contained_file(root / "run.json", root)
    config_path = _contained_file(root / "config.json", root)
    summary = _read_json(summary_path) if summary_path is not None else None
    config = _read_json(config_path) if config_path is not None else None
    if summary is None or config is None:
        return None
    raw_camera_path = root / "raw-camera" / "camera" / "run.json"
    raw_smoke_path = root / "raw-smoke-input" / "smoke-input" / "run.json"
    raw_camera_file = _contained_file(raw_camera_path, root)
    raw_smoke_file = _contained_file(raw_smoke_path, root)
    raw_camera = _read_json(raw_camera_file) if raw_camera_file is not None else None
    raw_smoke = _read_json(raw_smoke_file) if raw_smoke_file is not None else None
    raw_camera_dir = raw_camera_file.parent if raw_camera is not None and raw_camera_file is not None else None
    raw_smoke_dir = raw_smoke_file.parent if raw_smoke is not None and raw_smoke_file is not None else None
    stage_order_value = config.get("stage_order")
    stage_order = (
        tuple(value for value in stage_order_value if isinstance(value, str))
        if isinstance(stage_order_value, list)
        else ()
    )
    summaries = tuple(
        value
        for value in (summary, raw_camera, raw_smoke)
        if isinstance(value, Mapping)
    )
    stage_observations = _merge_stage_observations(
        summaries,
        stage_order=stage_order,
    )
    selected = _stage_selection(
        summary,
        stage_order=stage_order,
        root=root,
        raw_camera=raw_camera,
        raw_camera_dir=raw_camera_dir,
        raw_smoke=raw_smoke,
        raw_smoke_dir=raw_smoke_dir,
    )
    if selected.stage is None:
        return None if not stage_order else ProgressSnapshot(
            stage_order=stage_order,
            stage=None,
            stage_position=None,
            run_started=_timestamp(summary.get("created_at")),
            stage_started=None,
            status=summary.get("status") if isinstance(summary.get("status"), str) else None,
            conversion=ConversionProgress(None, None, None),
            downloaded_bytes=None,
            download_total=None,
            stage_active=False,
            stage_observations=stage_observations,
        )
    sidecar = _conversion_sidecar(summary, run_dir=root)
    conversion = _read_sidecar(sidecar) if sidecar is not None else ConversionProgress(None, None, None)
    position = stage_order.index(selected.stage) + 1
    timing = _stage_timing(selected.root, selected.summary, selected.stage)
    training_stage = selected.stage if selected.stage in _TRAINING_STAGES else _latest_training_stage(
        selected.summary,
        stage_order=stage_order,
    )
    training = _training_progress(selected.root, selected.summary, training_stage)
    status = selected.summary.get("status")
    if selected.active and selected.stage is not None and _active_attempt(
        selected.summary,
        stage=selected.stage,
    ) is not None:
        # Do not let a stale overall failure leak through while the selected
        # attempt is still being written.  Its own attempt status is the only
        # authoritative status for an active retry.
        status = selected.status
        if status not in _TERMINAL_STATUSES and not isinstance(status, str):
            status = None
    return ProgressSnapshot(
        stage_order=stage_order,
        stage=selected.stage,
        stage_position=position,
        run_started=_timestamp(summary.get("created_at")),
        stage_started=timing.started,
        status=status if isinstance(status, str) else None,
        conversion=conversion,
        downloaded_bytes=None,
        download_total=None,
        stage_active=selected.active,
        stage_attempt=timing.attempt,
        stage_status=selected.status,
        stage_finished=timing.finished,
        training=training,
        stage_observations=stage_observations,
    )


def _format_duration(seconds: float | None) -> str:
    value = max(0, int(seconds or 0))
    hours, remainder = divmod(value, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d}"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "0 B"
    amount = float(max(0, value))
    units = ("B", "KiB", "MiB", "GiB")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if amount < 1024 or candidate == units[-1]:
            break
        amount /= 1024
    return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"


def _format_download(downloaded: int, content_length: int | None) -> str:
    if content_length is None:
        return f"download {max(0, downloaded) / (1024 * 1024):.1f} MiB"
    return f"download {_format_bytes(downloaded)} / {_format_bytes(content_length)}"


def _display_stage(stage: str) -> str:
    return stage.replace("-smoke-", " ").replace("-", " ")


def render_progress_line(
    snapshot: ProgressSnapshot | None,
    *,
    now: float | None = None,
    width: int = _PROGRESS_WIDTH,
    download: tuple[int, int | None] | None = None,
    completed: bool | None = None,
) -> str:
    """Render one line without URLs, query strings, fragments, or tokens."""

    current = time.time() if now is None else now
    downloaded = snapshot.downloaded_bytes if snapshot is not None else None
    download_total = snapshot.download_total if snapshot is not None else None
    if download is not None:
        downloaded, download_total = download
    width = max(0, int(width))
    has_progress = False
    progress_value = 0
    progress_total = 0
    stage_started: float | None = None
    stage_finished: float | None = None
    run_started: float | None = None
    is_completed = completed is True
    if snapshot is None or not snapshot.stage_order:
        label = "preparing"
    else:
        stage = snapshot.stage or "preparing"
        label = _display_stage(stage)
        if stage == "conversion" and (
            snapshot.conversion.iteration is not None
            and snapshot.conversion.total is not None
        ):
            label = f"conversion {snapshot.conversion.iteration}/{snapshot.conversion.total}"
            progress_value = snapshot.conversion.iteration
            progress_total = snapshot.conversion.total
            has_progress = True
        elif stage == "conversion":
            label = "conversion (unobserved)"
        elif snapshot.training.iteration is not None and snapshot.training.total is not None:
            label = f"{label} | camera sampling {snapshot.training.iteration}/{snapshot.training.total}"
            progress_value = snapshot.training.iteration
            progress_total = snapshot.training.total
            has_progress = True
        if downloaded is not None:
            label = _format_download(downloaded, download_total)
        stage_started = snapshot.stage_started
        stage_finished = snapshot.stage_finished
        run_started = snapshot.run_started
        if completed is None:
            is_completed = not snapshot.stage_active
        if (
            stage == "conversion"
            and snapshot.conversion.marker_timestamp is not None
            and current - snapshot.conversion.marker_timestamp > _CONVERSION_WARNING_SECONDS
        ):
            label += " | WARNING: no observed iteration heartbeat"
        failure_status = next(
            (
                status
                for status in (snapshot.stage_status, snapshot.status)
                if status in _FAILURE_STATUSES
            ),
            None,
        )
        if is_completed:
            if failure_status is not None:
                label += f" | {failure_status}"
            elif snapshot.stage_status == "passed":
                label += " | 完成"
            else:
                label += " | pending"
        elif failure_status is not None:
            label += f" | {failure_status}"
        elif snapshot.stage_status is None:
            # A transition/retry can expose the active marker before its own
            # attempt ledger entry.  Keep this as a refresh-only pending state;
            # TerminalProgress never promotes it to completed history.
            label += " | pending"
    if downloaded is not None:
        label = _format_download(downloaded, download_total)
        if download_total is not None and download_total > 0:
            progress_value = min(max(0, downloaded), download_total)
            progress_total = download_total
            has_progress = True
    if is_completed and stage_finished is not None:
        stage_end = stage_finished
    elif is_completed:
        stage_end = current
    else:
        stage_end = current
    stage_elapsed = None if stage_started is None else stage_end - stage_started
    prefix = ""
    if has_progress and progress_total > 0:
        filled = min(width, max(0, round(width * progress_value / progress_total)))
        bar = "#" * filled + "-" * (width - filled)
        prefix = f"[{bar}] | "
    return (
        f"{prefix}{label} | "
        f"阶段 {_format_duration(stage_elapsed)} | "
        f"总计 {_format_duration(None if run_started is None else current - run_started)}"
    )


class TerminalProgress:
    """Background read-only observer with TTY and line-oriented modes."""

    def __init__(
        self,
        mode: str = "auto",
        *,
        stream: TextIO | None = None,
        env: Mapping[str, str] | None = None,
        now_fn: Callable[[], float] = time.time,
        interval: float = 1.0,
    ) -> None:
        if mode not in _PROGRESS_MODES:
            raise ValueError(f"progress mode must be one of {sorted(_PROGRESS_MODES)}")
        self.mode = mode
        self.stream = stream or sys.stderr
        self.env = dict(env or os.environ)
        self.now_fn = now_fn
        self.interval = max(0.05, interval)
        self._run_dir: Path | None = None
        self._download: tuple[int, int | None] | None = None
        self._last_snapshot: ProgressSnapshot | None = None
        self._display_snapshot: ProgressSnapshot | None = None
        self._display_started = False
        self._last_line = ""
        self._last_event: tuple[Any, ...] | None = None
        self._plain_current_key: tuple[Any, ...] | None = None
        self._stage_states: dict[str, _StageObservation] = {}
        self._stage_snapshots: dict[str, ProgressSnapshot] = {}
        self._emitted_terminal: dict[str, tuple[str | None, str]] = {}
        self._last_emit = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._tty = self._is_tty()

    def _is_tty(self) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "plain":
            return False
        if self.env.get("CI"):
            return False
        if self.env.get("TERM", "").lower() == "dumb" or "NO_COLOR" in self.env:
            return False
        return bool(getattr(self.stream, "isatty", lambda: False)())

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def bind_run(self, run_dir: str | Path) -> None:
        with self._lock:
            self._run_dir = Path(run_dir)
            self._download = None

    def download_callback(self, downloaded_bytes: int, content_length: int | None) -> None:
        with self._lock:
            self._download = (max(0, int(downloaded_bytes)), content_length)
        self.poll_once()

    def _read(self) -> ProgressSnapshot | None:
        with self._lock:
            run_dir = self._run_dir
        if run_dir is None:
            return self._last_snapshot
        snapshot = read_progress_snapshot(run_dir, now=self.now_fn())
        if snapshot is not None:
            self._last_snapshot = snapshot
        return self._last_snapshot

    def _line(self) -> str:
        snapshot = self._read()
        with self._lock:
            download = self._download
        return render_progress_line(snapshot, now=self.now_fn(), download=download)

    @staticmethod
    def _stage_key(snapshot: ProgressSnapshot | None) -> tuple[Any, ...]:
        if snapshot is None:
            return (None, None, False)
        return (snapshot.stage, snapshot.stage_attempt, snapshot.stage_active)

    @staticmethod
    def _event_key(
        snapshot: ProgressSnapshot | None,
        download: tuple[int, int | None] | None,
    ) -> tuple[Any, ...]:
        if snapshot is None:
            return (None, None, False, None, download is not None)
        return (
            snapshot.stage,
            snapshot.stage_attempt,
            snapshot.stage_active,
            snapshot.status,
            snapshot.stage_status,
            download is not None,
        )

    @staticmethod
    def _has_stage(snapshot: ProgressSnapshot | None) -> bool:
        return snapshot is not None and snapshot.stage is not None

    @staticmethod
    def _terminal_token(
        observation: _StageObservation | None,
    ) -> tuple[str | None, str] | None:
        if observation is None or observation.status not in _TERMINAL_STATUSES:
            return None
        return observation.attempt, observation.status

    def _record_snapshot(self, snapshot: ProgressSnapshot | None) -> None:
        if snapshot is None:
            return
        observations = snapshot.stage_observations
        if not observations and snapshot.stage is not None:
            observations = (
                _StageObservation(
                    stage=snapshot.stage,
                    attempt=snapshot.stage_attempt,
                    status=snapshot.stage_status,
                    active=snapshot.stage_active,
                ),
            )
        for observation in observations:
            previous = self._stage_states.get(observation.stage)
            if previous is not None and previous.attempt != observation.attempt:
                self._emitted_terminal.pop(observation.stage, None)
            if (
                previous is not None
                and previous.attempt == observation.attempt
                and previous.status in _TERMINAL_STATUSES
                and observation.status not in _TERMINAL_STATUSES
            ):
                # Atomic writer races may expose an older/lighter view after a
                # terminal record.  Terminal evidence is monotone for one
                # attempt; never turn it back into pending.
                observation = previous
            self._stage_states[observation.stage] = observation
        if snapshot.stage is not None:
            self._stage_snapshots[snapshot.stage] = snapshot

    def _display_ready(self, snapshot: ProgressSnapshot | None) -> bool:
        if snapshot is None or snapshot.stage is None:
            return True
        try:
            position = snapshot.stage_order.index(snapshot.stage)
        except ValueError:
            return True
        for stage in snapshot.stage_order[:position]:
            observation = self._stage_states.get(stage)
            if observation is not None and observation.status not in _TERMINAL_STATUSES:
                return False
        return True

    def _stage_line(
        self,
        stage: str,
        observation: _StageObservation,
        *,
        current: ProgressSnapshot | None,
        now: float,
    ) -> str:
        base = self._stage_snapshots.get(stage)
        if base is None or base.stage != stage:
            order = current.stage_order if current is not None else tuple(self._stage_states)
            position = order.index(stage) + 1 if stage in order else None
            base = ProgressSnapshot(
                stage_order=order,
                stage=stage,
                stage_position=position,
                run_started=current.run_started if current is not None else None,
                stage_started=None,
                status=None,
                conversion=ConversionProgress(None, None, None),
                downloaded_bytes=None,
                download_total=None,
                stage_active=False,
                stage_attempt=observation.attempt,
                stage_status=observation.status,
                stage_finished=None,
            )
        else:
            base = replace(
                base,
                stage_active=False,
                stage_attempt=observation.attempt,
                stage_status=observation.status,
                status=(
                    observation.status
                    if observation.status in _FAILURE_STATUSES
                    else base.status
                ),
            )
        return render_progress_line(base, now=now, completed=True)

    def _pending_terminal_lines(
        self,
        snapshot: ProgressSnapshot | None,
        *,
        now: float,
        force: bool = False,
    ) -> list[tuple[str, tuple[str | None, str], str]]:
        if snapshot is not None and snapshot.stage is not None:
            order = snapshot.stage_order
            max_position = order.index(snapshot.stage) if snapshot.stage in order else -1
        else:
            order = self._last_snapshot.stage_order if self._last_snapshot is not None else tuple(self._stage_states)
            positions = [order.index(stage) for stage in self._stage_states if stage in order]
            max_position = max(positions, default=-1)
        blocked_by_pending = False
        pending: list[tuple[str, tuple[str | None, str], str]] = []
        for stage in order[: max_position + 1]:
            observation = self._stage_states.get(stage)
            token = self._terminal_token(observation)
            if token is None:
                if observation is not None:
                    blocked_by_pending = True
                continue
            if self._emitted_terminal.get(stage) == token:
                continue
            if blocked_by_pending and not force:
                continue
            pending.append(
                (
                    stage,
                    token,
                    self._stage_line(stage, observation, current=snapshot, now=now),
                )
            )
        return pending

    def _mark_terminal_lines(
        self,
        lines: list[tuple[str, tuple[str | None, str], str]],
    ) -> None:
        for stage, token, _line in lines:
            self._emitted_terminal[stage] = token

    def _write_tty_output(
        self,
        terminal_lines: list[tuple[str, tuple[str | None, str], str]],
        current: ProgressSnapshot | None,
        *,
        current_line: str,
        current_ready: bool,
    ) -> None:
        had_current_line = bool(self._last_line)
        if terminal_lines:
            for index, (_stage, _token, completed_line) in enumerate(terminal_lines):
                if index == 0 and self._last_line:
                    padding = " " * max(0, len(self._last_line) - len(completed_line))
                    self.stream.write("\r" + completed_line + padding + "\n")
                else:
                    self.stream.write(completed_line + "\n")
            self._last_line = ""

        if not self._has_stage(current):
            if not terminal_lines and (not self._display_started or self._last_line):
                self.stream.write("\r" + current_line)
                self._last_line = current_line
            return

        observation = self._stage_states.get(current.stage)
        token = self._terminal_token(observation)
        terminal_already_written = current.stage in {item[0] for item in terminal_lines} or (
            token is not None and self._emitted_terminal.get(current.stage) == token
        )
        if terminal_already_written:
            return

        # A pending stage is allowed to remain a current TTY refresh, but it
        # is deliberately never newline-frozen until its attempt settles.
        if current_ready or not terminal_lines:
            padding = " " * max(0, len(self._last_line) - len(current_line))
            prefix = "" if terminal_lines and not had_current_line else (
                "\r" if self._last_line or not self._display_started else ""
            )
            self.stream.write(prefix + current_line + padding)
            self._last_line = current_line

    def _poll_once(self, *, force: bool = False) -> str | None:
        if not self.enabled:
            return None
        line = self._line()
        current = self._last_snapshot
        self._record_snapshot(current)
        now = self.now_fn()
        with self._lock:
            download = self._download
        current_ready = self._display_ready(current)
        terminal_lines = self._pending_terminal_lines(current, now=now, force=False)
        if self._tty:
            self._write_tty_output(
                terminal_lines,
                current,
                current_line=line,
                current_ready=current_ready,
            )
            if terminal_lines:
                self._mark_terminal_lines(terminal_lines)
            self.stream.flush()
            self._display_snapshot = current
            self._display_started = True
            return line
        event = self._event_key(current, download)
        lines: list[str] = []
        lines.extend(item[2] for item in terminal_lines)
        pending_stage_names = {item[0] for item in terminal_lines}
        current_observation = current.stage if current is not None else None
        current_token = self._terminal_token(
            self._stage_states.get(current_observation) if current_observation is not None else None
        )
        current_terminal_written = (
            current_observation is not None
            and current_token is not None
            and (
                current_observation in pending_stage_names
                or self._emitted_terminal.get(current_observation) == current_token
            )
        )
        should_refresh = force or event != self._last_event or now - self._last_emit >= _HEARTBEAT_SECONDS
        if self._has_stage(current):
            if current_ready and not current_terminal_written and current.stage_active:
                # Plain/CI output has no in-place refresh cursor.  An active
                # attempt with no authoritative ledger status is therefore
                # kept silent until it settles; otherwise ``pending`` would
                # become permanent history during the write race.
                transient = current.stage_status in {None, "pending", "unknown"}
                if not transient and (self._plain_current_key != event or should_refresh):
                    lines.append(line)
        elif not self._display_started or should_refresh:
            lines.append(line)
        if lines:
            self.stream.write("\n".join(lines) + "\n")
            self.stream.flush()
            if terminal_lines:
                self._mark_terminal_lines(terminal_lines)
            self._last_event = event
            self._last_emit = now
            if (
                self._has_stage(current)
                and not current_terminal_written
                and current_ready
                and current.stage_active
                and current.stage_status not in {None, "pending", "unknown"}
            ):
                self._plain_current_key = event
        self._display_snapshot = current
        self._display_started = True
        return line

    def poll_once(self, *, force: bool = False) -> str | None:
        """Poll best-effort; reporter failures never affect the canonical route."""
        try:
            return self._poll_once(force=force)
        except Exception:
            return None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self.poll_once(force=True)
        self._thread = threading.Thread(target=self._loop, name="video-to-3dgs-progress", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.poll_once()

    def _flush_terminal_on_close(self) -> None:
        snapshot = self._last_snapshot
        if snapshot is None:
            return
        now = self.now_fn()
        terminal_lines = self._pending_terminal_lines(snapshot, now=now, force=True)
        if not terminal_lines:
            return
        line = render_progress_line(snapshot, now=now)
        if self._tty:
            self._write_tty_output(
                terminal_lines,
                snapshot,
                current_line=line,
                current_ready=True,
            )
            self.stream.flush()
        else:
            self.stream.write("\n".join(item[2] for item in terminal_lines) + "\n")
            self.stream.flush()
        self._mark_terminal_lines(terminal_lines)

    def close(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        try:
            # Capture a final authoritative snapshot if the caller closes the
            # observer immediately after an executor writes its result.
            self._poll_once()
            self._flush_terminal_on_close()
        except Exception:
            pass
        if self._tty and self._last_line:
            try:
                self.stream.write("\r" + (" " * len(self._last_line)) + "\r")
                self.stream.flush()
            except Exception:
                pass
            self._last_line = ""
