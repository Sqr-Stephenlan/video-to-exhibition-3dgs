"""Read-only terminal progress reporting for the canonical one-click chain."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, TextIO


_PROGRESS_MODES = {"auto", "plain", "off"}
_PROGRESS_WIDTH = 20
_HEARTBEAT_SECONDS = 60.0
_CONVERSION_WARNING_SECONDS = 15 * 60.0
_SIDECAR_TAIL_BYTES = 1024 * 1024
_SCHEMA = "conversion-progress-v1"


@dataclass(frozen=True)
class ConversionProgress:
    iteration: int | None
    total: int | None
    marker_timestamp: float | None


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
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
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


def _stage_start(
    summary: Mapping[str, Any],
    *,
    stage: str | None,
    raw_dir: Path | None = None,
) -> float | None:
    if not isinstance(stage, str):
        return None
    if raw_dir is not None:
        raw_summary = _read_json(raw_dir / "run.json")
        if raw_summary:
            raw_started = _timestamp(raw_summary.get("created_at"))
            if raw_started is not None:
                return raw_started
    return _timestamp(summary.get("updated_at"))


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
    if isinstance(active_attempt, str) and summary.get("active_stage") == "conversion":
        candidate = run_dir / "stages" / "conversion" / active_attempt / "executor" / "conversion-progress-v1.jsonl"
        return _contained_file(candidate, run_dir)
    latest_attempt = _latest_attempt_name(summary, stage="conversion")
    if latest_attempt is not None:
        candidate = run_dir / "stages" / "conversion" / latest_attempt / "executor" / "conversion-progress-v1.jsonl"
        checked = _contained_file(candidate, run_dir)
        return checked
    return None


def _active_stage(
    summary: Mapping[str, Any],
    *,
    stage_order: tuple[str, ...],
    raw_camera: Mapping[str, Any] | None,
    raw_smoke: Mapping[str, Any] | None,
) -> tuple[str | None, Path | None]:
    for value in (summary.get("active_stage"), summary.get("current_intended_stage")):
        if isinstance(value, str) and value in stage_order:
            return value, None
    if isinstance(raw_camera, Mapping):
        raw_stage = raw_camera.get("active_stage") or raw_camera.get("current_intended_stage")
        if raw_stage:
            return ("camera-staging" if "camera-staging" in stage_order else None), None
    if isinstance(raw_smoke, Mapping):
        raw_stage = raw_smoke.get("active_stage") or raw_smoke.get("current_intended_stage")
        if raw_stage:
            return ("longsplat-input" if "longsplat-input" in stage_order else None), None
    stages = summary.get("stages")
    if isinstance(stages, Mapping):
        for value in reversed(list(stages)):
            if value in stage_order:
                return value, None
    return None, None


def read_progress_snapshot(run_dir: str | Path, *, now: float | None = None) -> ProgressSnapshot | None:
    """Read only current-run JSON/sidecar state; malformed transient files are ignored."""

    root = Path(run_dir)
    if root.is_symlink() or not root.is_dir():
        return None
    summary = _read_json(root / "run.json")
    config = _read_json(root / "config.json")
    if summary is None or config is None:
        return None
    raw_camera_path = root / "raw-camera" / "camera" / "run.json"
    raw_smoke_path = root / "raw-smoke-input" / "smoke-input" / "run.json"
    raw_camera = _read_json(raw_camera_path)
    raw_smoke = _read_json(raw_smoke_path)
    raw_camera_dir = raw_camera_path.parent if raw_camera is not None else None
    raw_smoke_dir = raw_smoke_path.parent if raw_smoke is not None else None
    stage_order_value = config.get("stage_order")
    stage_order = (
        tuple(value for value in stage_order_value if isinstance(value, str))
        if isinstance(stage_order_value, list)
        else ()
    )
    stage, _ = _active_stage(
        summary,
        stage_order=stage_order,
        raw_camera=raw_camera,
        raw_smoke=raw_smoke,
    )
    if stage is None:
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
        )
    sidecar = _conversion_sidecar(summary, run_dir=root)
    conversion = _read_sidecar(sidecar) if sidecar is not None else ConversionProgress(None, None, None)
    position = stage_order.index(stage) + 1
    started = _stage_start(
        summary,
        stage=stage,
        raw_dir=raw_camera_dir if stage == "camera-staging" else raw_smoke_dir if stage == "longsplat-input" else None,
    )
    return ProgressSnapshot(
        stage_order=stage_order,
        stage=stage,
        stage_position=position,
        run_started=_timestamp(summary.get("created_at")),
        stage_started=started,
        status=summary.get("status") if isinstance(summary.get("status"), str) else None,
        conversion=conversion,
        downloaded_bytes=None,
        download_total=None,
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
) -> str:
    """Render one line without URLs, query strings, fragments, or tokens."""

    current = time.time() if now is None else now
    downloaded = snapshot.downloaded_bytes if snapshot is not None else None
    download_total = snapshot.download_total if snapshot is not None else None
    if download is not None:
        downloaded, download_total = download
    if snapshot is None or not snapshot.stage_order:
        label = "preparing"
        position_text = ""
        filled = 0
        stage_started = None
        run_started = None
    else:
        stage = snapshot.stage or "preparing"
        label = _display_stage(stage)
        if stage == "conversion" and snapshot.conversion.iteration is not None:
            label = f"conversion {snapshot.conversion.iteration}/{snapshot.conversion.total}"
        elif stage == "conversion":
            label = "conversion (unobserved)"
        if downloaded is not None:
            label = _format_download(downloaded, download_total)
        position = snapshot.stage_position or 0
        total = len(snapshot.stage_order)
        position_text = f" {position}/{total}"
        filled = min(width, max(0, round(width * position / total)))
        stage_started = snapshot.stage_started
        run_started = snapshot.run_started
        if (
            stage == "conversion"
            and snapshot.conversion.marker_timestamp is not None
            and current - snapshot.conversion.marker_timestamp > _CONVERSION_WARNING_SECONDS
        ):
            label += " | WARNING: no observed iteration heartbeat"
        if snapshot.status in {"blocked", "failed"}:
            label += f" | {snapshot.status}"
    if downloaded is not None:
        label = _format_download(downloaded, download_total)
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{bar}]{position_text} | {label} | "
        f"阶段 {_format_duration(None if stage_started is None else current - stage_started)} | "
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
        self._last_line = ""
        self._last_event: tuple[Any, ...] | None = None
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

    def _poll_once(self, *, force: bool = False) -> str | None:
        if not self.enabled:
            return None
        line = self._line()
        now = self.now_fn()
        if self._tty:
            padding = " " * max(0, len(self._last_line) - len(line))
            self.stream.write("\r" + line + padding)
            self.stream.flush()
            self._last_line = line
            return line
        snapshot = self._last_snapshot
        event = (
            snapshot.stage if snapshot is not None else None,
            snapshot.status if snapshot is not None else None,
            self._download is not None,
        )
        if force or event != self._last_event or now - self._last_emit >= _HEARTBEAT_SECONDS:
            self.stream.write(line + "\n")
            self.stream.flush()
            self._last_event = event
            self._last_emit = now
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

    def close(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        if self._tty and self._last_line:
            try:
                self.stream.write("\r" + (" " * len(self._last_line)) + "\r")
                self.stream.flush()
            except Exception:
                pass
            self._last_line = ""
