"""Load and validate depth-prior YAML configs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

ALLOWED_ENCODERS = {"vits", "vitb", "vitl"}
ALLOWED_DEPTH_TYPES = {"relative", "metric"}
SUPPORTED_SCHEMA_VERSIONS = {"1.0"}
FRAME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def require_schema_version(data: dict[str, Any], *, label: str) -> str:
    version = data.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{label}.schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}; "
            f"got {version!r}"
        )
    return str(version)


def validate_video_id(video_id: str) -> str:
    if not isinstance(video_id, str) or not VIDEO_ID_RE.fullmatch(video_id):
        raise ValueError(
            "video_id must be 1-128 chars of [A-Za-z0-9._-] and start with alphanumeric; "
            f"got {video_id!r}"
        )
    return video_id


def format_io_template(
    template: str,
    *,
    video_id: str | None = None,
    run_id: str | None = None,
) -> str:
    """Expand optional {video_id} / {run_id} placeholders in repository-relative IO paths."""
    text = str(template)
    if "{video_id}" in text:
        if not video_id:
            raise ValueError(
                f"IO path template requires video_id but none was provided: {template}"
            )
        text = text.replace("{video_id}", validate_video_id(video_id))
    if "{run_id}" in text:
        if not run_id:
            raise ValueError(
                f"IO path template requires run_id but none was provided: {template}"
            )
        text = text.replace("{run_id}", validate_video_id(run_id))
    return text


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path.as_posix()}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path.as_posix()}")
    validate_config(data)
    return data


def validate_config(data: dict[str, Any]) -> None:
    require_schema_version(data, label="config")
    backend = data.get("backend")
    if not isinstance(backend, dict):
        raise ValueError("config.backend must be a mapping")
    encoder = backend.get("encoder")
    if encoder not in ALLOWED_ENCODERS:
        raise ValueError(f"backend.encoder must be one of {sorted(ALLOWED_ENCODERS)}")
    depth_type = backend.get("depth_type", "relative")
    if depth_type not in ALLOWED_DEPTH_TYPES:
        raise ValueError(f"backend.depth_type must be one of {sorted(ALLOWED_DEPTH_TYPES)}")
    io = data.get("io")
    if not isinstance(io, dict):
        raise ValueError("config.io must be a mapping")
    for key in ("frames_manifest", "depth_dir", "depth_manifest", "run_record"):
        if key not in io or not io[key]:
            raise ValueError(f"config.io.{key} is required")


def validate_frame_id(frame_id: str) -> str:
    if not isinstance(frame_id, str) or not FRAME_ID_RE.fullmatch(frame_id):
        raise ValueError(
            "frame_id must be 1-128 chars of [A-Za-z0-9._-] and start with alphanumeric; "
            f"got {frame_id!r}"
        )
    return frame_id


def resolve_repo_path(root: Path, relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed in configs: {path}")
    root_resolved = root.resolve()
    candidate = (root_resolved / path).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {relative}") from exc
    return candidate


def to_repo_relative(root: Path, path: Path) -> str:
    root_resolved = root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root_resolved).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path is outside repository root: {path}") from exc
