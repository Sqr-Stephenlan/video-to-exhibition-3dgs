"""Small, standard-library-only helpers for live conversion observation.

The conversion child is the source of truth for iteration progress.  This
module deliberately understands one structured marker only; it does not try
to infer progress from tqdm output or process liveness.
"""

from __future__ import annotations

import codecs
import json
import math
import os
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


CONVERSION_SHAPE_MARKER = "CONVERSION_SHAPE_TELEMETRY"
CONVERSION_PROGRESS_SCHEMA = "conversion-progress-v1"
DEFAULT_TAIL_CHARS = 2_000
_MAX_PENDING_CHARS = 64 * 1024
_MAX_MARKER_CHARS = 4 * 1024
_FIELD_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class ObservabilityPathError(ValueError):
    """Raised when an observation artifact is outside its attempt root."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_symlink_components(path: Path, label: str) -> None:
    probe = Path(path.anchor)
    for component in path.parts[1:]:
        probe /= component
        if probe.is_symlink():
            raise ObservabilityPathError(f"{label} traverses a symlink: {probe}")


def validate_observability_file(path: str | Path, *, root: str | Path, label: str) -> Path:
    """Validate a new regular file path below an existing attempt root."""

    raw = Path(path)
    if not raw.is_absolute():
        raise ObservabilityPathError(f"{label} must be absolute: {raw}")
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ObservabilityPathError(f"observability root is missing or symlinked: {root_path}")
    _reject_symlink_components(raw, label)
    if raw.exists() or raw.is_symlink():
        raise ObservabilityPathError(f"{label} must be a fresh file: {raw}")
    parent = raw.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ObservabilityPathError(f"{label} parent is missing or symlinked: {parent}")
    root_resolved = root_path.resolve(strict=True)
    try:
        parent.resolve(strict=True).relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise ObservabilityPathError(
            f"{label} must remain below observability root {root_resolved}: {raw}"
        ) from exc
    return raw


def open_exclusive_binary(path: Path, *, label: str, append: bool = False):
    """Open a fresh ordinary file without following the final symlink."""

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if append:
        flags |= os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(path), flags, 0o644)
    except OSError as exc:
        raise ObservabilityPathError(f"cannot create {label}: {path}: {exc}") from exc
    return os.fdopen(fd, "wb")


def parse_conversion_shape_marker(line: str) -> dict[str, Any] | None:
    """Parse one exact ``CONVERSION_SHAPE_TELEMETRY`` marker.

    The iteration field is mandatory and strictly integer-valued.  Other
    key/value fields are retained when they are finite numbers or plain
    strings.  No generic progress-bar text is accepted.
    """

    prefix = f"{CONVERSION_SHAPE_MARKER} "
    candidate = line.strip()
    if not candidate.startswith(prefix):
        return None
    fields: dict[str, Any] = {}
    for token in candidate[len(prefix) :].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if _FIELD_NAME.fullmatch(key) is None:
            continue
        if key == "iteration":
            if not re.fullmatch(r"[0-9]+", value):
                return None
            iteration = int(value)
            if iteration <= 0:
                return None
            fields[key] = iteration
            continue
        try:
            number = float(value)
        except ValueError:
            fields[key] = value
        else:
            fields[key] = number if math.isfinite(number) else value
    if not isinstance(fields.get("iteration"), int):
        return None
    return fields


class BoundedTextTail:
    """Keep only the last ``max_chars`` decoded characters."""

    def __init__(self, max_chars: int = DEFAULT_TAIL_CHARS) -> None:
        self._max_chars = max(1, int(max_chars))
        self._parts: deque[str] = deque()
        self._length = 0

    def append(self, value: str) -> None:
        if not value:
            return
        if len(value) > self._max_chars:
            value = value[-self._max_chars :]
        self._parts.append(value)
        self._length += len(value)
        while self._length > self._max_chars:
            overflow = self._length - self._max_chars
            head = self._parts[0]
            if len(head) <= overflow:
                self._parts.popleft()
                self._length -= len(head)
            else:
                self._parts[0] = head[overflow:]
                self._length -= overflow

    def value(self) -> str:
        return "".join(self._parts)


class ConversionStreamCapture:
    """Decode a child stream, retain a bounded tail, and inspect exact markers."""

    def __init__(
        self,
        *,
        stream: str,
        on_marker: Callable[[str, str, dict[str, Any]], None] | None = None,
        tail_chars: int = DEFAULT_TAIL_CHARS,
    ) -> None:
        self.stream = stream
        self._on_marker = on_marker
        self._tail = BoundedTextTail(tail_chars)
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._pending = ""
        self._finished = False

    def _inspect(self, line: str) -> None:
        # tqdm uses carriage returns.  Splitting only on carriage returns here
        # lets an exact structured marker survive a refresh without treating
        # arbitrary tqdm text as progress.
        for candidate in line.split("\r"):
            marker = candidate.strip()
            fields = parse_conversion_shape_marker(marker)
            if fields is not None and self._on_marker is not None:
                self._on_marker(self.stream, marker[:_MAX_MARKER_CHARS], fields)

    def _consume_text(self, text: str) -> None:
        if not text:
            return
        self._tail.append(text)
        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._inspect(line)
        if len(self._pending) > _MAX_PENDING_CHARS:
            self._pending = self._pending[-_MAX_PENDING_CHARS:]

    def feed(self, data: bytes) -> None:
        if self._finished:
            return
        self._consume_text(self._decoder.decode(data, final=False))

    def finish(self) -> None:
        if self._finished:
            return
        self._consume_text(self._decoder.decode(b"", final=True))
        if self._pending:
            self._inspect(self._pending)
            self._pending = ""
        self._finished = True

    @property
    def tail(self) -> str:
        return self._tail.value()


class AppendOnlyProgress:
    """Write complete JSONL records and fsync each observation."""

    def __init__(self, path: Path, *, total: int, started_monotonic: float) -> None:
        self.path = path
        self.total = int(total)
        self.started_monotonic = started_monotonic
        self._handle = open_exclusive_binary(
            path,
            label="conversion progress sidecar",
            append=True,
        )
        self._closed = False
        self._last_iteration: int | None = None

    def append(self, *, phase: str, elapsed: float | None = None, **fields: Any) -> None:
        if self._closed:
            raise RuntimeError("conversion progress sidecar is closed")
        record: dict[str, Any] = {
            "schema_version": CONVERSION_PROGRESS_SCHEMA,
            "phase": phase,
            "total": self.total,
            "timestamp_utc": utc_now(),
            "elapsed_seconds": round(
                max(0.0, elapsed if elapsed is not None else 0.0),
                6,
            ),
        }
        record.update(fields)
        payload = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        self._handle.write(payload)
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def append_started(self, *, pid: int, elapsed: float) -> None:
        self.append(phase="child_started", elapsed=elapsed, pid=int(pid))

    def append_marker(
        self,
        *,
        stream: str,
        raw_marker: str,
        fields: dict[str, Any],
        elapsed: float,
    ) -> bool:
        iteration = fields.get("iteration")
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < 1
            or iteration > self.total
            or (
                self._last_iteration is not None
                and iteration <= self._last_iteration
            )
        ):
            return False
        self._last_iteration = iteration
        self.append(
            phase="iteration",
            elapsed=elapsed,
            iteration=iteration,
            marker_summary=fields,
            raw_marker=raw_marker,
            stream=stream,
        )
        return True

    def append_exited(self, *, returncode: int, elapsed: float) -> None:
        self.append(phase="child_exited", elapsed=elapsed, returncode=int(returncode))

    def append_parent_error(self, *, error_type: str, elapsed: float) -> None:
        self.append(phase="parent_error", elapsed=elapsed, error_type=error_type)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        finally:
            self._handle.close()
            self._closed = True
