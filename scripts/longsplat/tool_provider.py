"""Versioned, CPU-only provider layout for the raw-video chain."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from .pipeline_contract import PipelineBlocked, sha256_file


TOOL_PROVIDER_SCHEMA = "tool-provider-config-v1"
TOOL_NAMES = ("ffmpeg", "ffprobe", "colmap", "backend_python", "route_python")


def default_tool_paths(route_root: str | Path) -> dict[str, str]:
    """Return the historical known layout without making it mandatory."""

    route = Path(route_root).resolve()
    workspace_root = route.parent.parent
    return {
        "ffmpeg": str(workspace_root / "backend-envs/media-tools/bin/ffmpeg"),
        "ffprobe": str(workspace_root / "backend-envs/media-tools/bin/ffprobe"),
        "colmap": "colmap",
        "backend_python": str(workspace_root / "backend-envs/longsplat-cu128/bin/python"),
        "route_python": sys.executable,
    }


def _is_path_value(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or any(separator in value for separator in (os.sep, "\\"))


def _provider_file_identity(value: str) -> dict[str, Any]:
    requested = str(value)
    candidate: Path | None
    if _is_path_value(requested):
        candidate = Path(requested)
    else:
        resolved = shutil.which(requested)
        candidate = None if resolved is None else Path(resolved)
    if candidate is None:
        return {
            "requested_path": requested,
            "resolved_path": None,
            "resolved_target": None,
            "symlink": False,
            "regular_file": False,
            "executable": False,
            "sha256": None,
            "size_bytes": None,
        }
    symlink = candidate.is_symlink()
    resolved = candidate.resolve(strict=False)
    regular = resolved.is_file()
    executable = regular and os.access(resolved, os.X_OK)
    return {
        "requested_path": requested,
        "resolved_path": str(candidate.absolute()) if candidate.exists() else None,
        "resolved_target": str(resolved) if regular else None,
        "symlink": symlink,
        "regular_file": regular,
        "executable": executable,
        "sha256": sha256_file(resolved) if regular else None,
        "size_bytes": resolved.stat().st_size if regular else None,
    }


def resolve_tool_provider(
    route_root: str | Path,
    overrides: Mapping[str, str | Path] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Merge explicit providers with the known layout and bind their identity.

    Explicit values may point outside the route or output root.  The provider
    contract does not dereference them into output data; it records the
    requested path, resolved target, executable bit, and file digest.  A
    symlink is allowed when its resolved target is a regular executable, and
    both the link and target are recorded for resume drift checks.
    """

    effective = default_tool_paths(route_root)
    requested = dict(overrides or {})
    unknown = set(requested) - set(TOOL_NAMES) - {"backend_env"}
    if unknown:
        raise PipelineBlocked(f"unsupported tool provider keys: {sorted(unknown)}")
    backend_env = requested.pop("backend_env", None)
    if backend_env is not None:
        if "backend_python" in requested:
            raise PipelineBlocked("backend_env and backend_python cannot both be specified")
        effective["backend_python"] = str(Path(backend_env) / "bin" / "python")
    for name, value in requested.items():
        if value is None:
            continue
        effective[name] = str(value)
    providers = {
        name: _provider_file_identity(effective[name]) for name in TOOL_NAMES
    }
    config: dict[str, Any] = {
        "schema_version": TOOL_PROVIDER_SCHEMA,
        "source": "explicit" if overrides else "known_layout",
        "backend_env": None if backend_env is None else str(Path(backend_env).absolute()),
        "requested": {
            key: str(value) for key, value in sorted((overrides or {}).items()) if value is not None
        },
        "effective": {key: effective[key] for key in TOOL_NAMES},
        "providers": providers,
    }
    config["provider_identity_sha256"] = _stable_hash(config)
    return effective, config


def verify_tool_provider(config: Mapping[str, Any]) -> dict[str, Any]:
    """Re-resolve a persisted provider record and reject identity drift."""

    if config.get("schema_version") != TOOL_PROVIDER_SCHEMA:
        raise PipelineBlocked(f"tool provider schema must be {TOOL_PROVIDER_SCHEMA}")
    effective = config.get("effective")
    declared = config.get("providers")
    if not isinstance(effective, Mapping) or not isinstance(declared, Mapping):
        raise PipelineBlocked("tool provider effective/providers records are required")
    current = {name: _provider_file_identity(str(effective.get(name, ""))) for name in TOOL_NAMES}
    for name in TOOL_NAMES:
        expected = declared.get(name)
        if not isinstance(expected, Mapping):
            raise PipelineBlocked(f"tool provider record is missing: {name}")
        for key in ("requested_path", "resolved_target", "symlink", "regular_file", "executable", "sha256", "size_bytes"):
            if expected.get(key) != current[name].get(key):
                raise PipelineBlocked(f"tool provider identity drifted for {name}: {key}")
    unsigned = dict(config)
    unsigned.pop("provider_identity_sha256", None)
    if config.get("provider_identity_sha256") != _stable_hash(unsigned):
        raise PipelineBlocked("tool provider identity digest is invalid")
    return current


def _stable_hash(value: Any) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
