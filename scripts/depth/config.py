"""Load and validate depth-prior YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ALLOWED_ENCODERS = {"vits", "vitb", "vitl"}
ALLOWED_DEPTH_TYPES = {"relative", "metric"}


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


def resolve_repo_path(root: Path, relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"Absolute paths are not allowed in configs: {path}")
    return (root / path).resolve()
