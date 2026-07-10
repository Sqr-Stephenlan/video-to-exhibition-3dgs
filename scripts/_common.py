"""
Shared utilities for the video-to-exhibition-3dgs preprocessing pipeline.

Provides project-root resolution, safe media-metadata parsing, external-tool
error wrapping, output-directory lifecycle management, and atomic JSON I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path | None = None


def get_project_root() -> Path:
    """Return the absolute path of the repository root.

    Walks upward from this file until a ``.git`` directory or ``dev.sh`` is
    found.  The result is cached so repeated calls are cheap.
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    candidate = Path(__file__).resolve().parent
    for _ in range(10):
        if (candidate / ".git").exists() or (candidate / "dev.sh").exists():
            _PROJECT_ROOT = candidate
            return candidate
        candidate = candidate.parent

    # Fallback: assume the canonical layout <root>/scripts/_common.py
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    return _PROJECT_ROOT


# ---------------------------------------------------------------------------
# Safe media-metadata helpers
# ---------------------------------------------------------------------------


def safe_parse_fps(r_frame_rate: str) -> float:
    """Parse an ffprobe ``r_frame_rate`` string (e.g. ``"30000/1001"``).

    Uses :class:`fractions.Fraction` so that zero denominators are caught
    cleanly instead of raising :exc:`ZeroDivisionError` inside ``eval()``.
    """
    if not r_frame_rate or r_frame_rate.strip() == "":
        raise ValueError("r_frame_rate is empty")
    try:
        frac = Fraction(r_frame_rate.strip())
    except ZeroDivisionError:
        raise ValueError(f"r_frame_rate has zero denominator: {r_frame_rate!r}") from None
    return float(frac)


# ---------------------------------------------------------------------------
# External-tool execution
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Raised when an external tool exits with a non-zero return code."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        summary = stderr.strip().splitlines()
        tail = summary[-8:] if len(summary) > 8 else summary
        detail = "\n".join(tail) if tail else "(no stderr output)"
        super().__init__(f"Command failed (rc={returncode}): {' '.join(cmd)}\n{detail}")
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr


def run_tool(cmd: list[str], description: str = "") -> subprocess.CompletedProcess[str]:
    """Run an external tool, capturing and decoding stdout/stderr as text.

    On failure a :exc:`ToolError` is raised that includes a human-readable
    summary of the command and its stderr tail.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ToolError(cmd, exc.returncode, exc.stderr or "") from exc
    except FileNotFoundError:
        raise ToolError(cmd, -1, f"executable not found: {cmd[0]}") from None
    return result


# ---------------------------------------------------------------------------
# Output-directory lifecycle (CR-09 provenance foundation)
# ---------------------------------------------------------------------------


class OutputDirectoryError(Exception):
    """Raised when an output directory cannot be used as requested."""


def ensure_output_dir(
    dir_path: Path,
    *,
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    """Create or validate an output directory according to a lifecycle policy.

    * ``dir_path`` does not exist → create it (mkdir parents).
    * ``dir_path`` exists and is **empty** → proceed.
    * ``dir_path`` exists and is **non-empty**:
        * default (no flags) → :exc:`OutputDirectoryError`
        * ``overwrite=True`` → remove contents, then recreate
        * ``resume=True``  → proceed (caller is responsible for fingerprint
          checks)
    """
    resolved = dir_path.resolve()
    if not resolved.exists():
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    contents = list(resolved.iterdir())
    if not contents:
        return resolved

    if overwrite:
        shutil.rmtree(resolved)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    if resume:
        return resolved

    raise OutputDirectoryError(
        f"Output directory {resolved} is not empty ({len(contents)} items). "
        "Use --overwrite to clear it or --resume to reuse (with provenance check)."
    )


# ---------------------------------------------------------------------------
# Atomic JSON I/O
# ---------------------------------------------------------------------------


def read_json(path: Path) -> Any:
    """Read a JSON file with explicit UTF-8 encoding."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json_atomic(data: Any, path: Path) -> None:
    """Write JSON to a temp file then atomically rename to *path*."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Image I/O with validation
# ---------------------------------------------------------------------------


def check_imread(path: Path, flags: int = 1) -> np.ndarray:
    """``cv2.imread`` wrapper that raises if the image cannot be read."""
    import cv2

    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Cannot read image (corrupt or missing): {path}")
    return img


def check_imwrite(path: Path, img: np.ndarray) -> None:
    """``cv2.imwrite`` wrapper that raises on write failure."""
    import cv2

    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise OSError(f"Failed to write image: {path}")


# ---------------------------------------------------------------------------
# Hashing helpers (for provenance)
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dict_fingerprint(d: dict[str, Any]) -> str:
    """Stable SHA-256 fingerprint of a (JSON-serializable) dictionary."""
    raw = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def now_utc_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()
