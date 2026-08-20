"""Versioned, CPU-only provider layout for the raw-video chain."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

from .pipeline_contract import PipelineBlocked, sha256_file


TOOL_PROVIDER_SCHEMA = "tool-provider-config-v1"
TOOL_NAMES = ("ffmpeg", "ffprobe", "colmap", "backend_python", "route_python")
PROVIDER_CONFIG_KEYS = frozenset(TOOL_NAMES) | {"backend_env"}


def discover_workspace_root(route_root: str | Path) -> Path | None:
    """Find a workspace ancestor that carries the optional backend envs."""

    route = Path(route_root).resolve()
    for candidate in (route, *route.parents):
        if (candidate / "backend-envs").is_dir():
            return candidate
    return None


_discover_workspace_root = discover_workspace_root


def _prefer_layout_or_path(layout_path: Path, command: str) -> str:
    if layout_path.is_file():
        return str(layout_path)
    return shutil.which(command) or command


def default_tool_paths(route_root: str | Path) -> dict[str, str]:
    """Discover portable workspace providers, then fall back to ``PATH``.

    The production route normally lives below the workspace containing
    ``backend-envs/media-tools`` and ``backend-envs/longsplat-cu128``.  A
    separate clone can instead provide explicit values through the local
    provider config or the existing CLI overrides; no user-specific absolute
    path is part of the tracked default.
    """

    workspace_root = discover_workspace_root(route_root)
    media_root = None if workspace_root is None else workspace_root / "backend-envs/media-tools/bin"
    longsplat_root = None if workspace_root is None else workspace_root / "backend-envs/longsplat-cu128/bin"
    backend_override = os.environ.get("LONGSPLAT_BACKEND_PYTHON")
    return {
        "ffmpeg": _prefer_layout_or_path(
            media_root / "ffmpeg" if media_root is not None else Path("ffmpeg"),
            "ffmpeg",
        ),
        "ffprobe": _prefer_layout_or_path(
            media_root / "ffprobe" if media_root is not None else Path("ffprobe"),
            "ffprobe",
        ),
        "colmap": shutil.which("colmap") or "colmap",
        "backend_python": (
            backend_override
            or _prefer_layout_or_path(
                longsplat_root / "python" if longsplat_root is not None else Path("python3"),
                "python3",
            )
        ),
        "route_python": sys.executable,
    }


def load_provider_config(path: str | Path) -> dict[str, str]:
    """Load a small ignored JSON provider override file."""

    config_path = Path(path)
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineBlocked(f"provider config is not valid JSON: {config_path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PipelineBlocked("provider config must be a JSON object")
    unknown = set(value) - PROVIDER_CONFIG_KEYS
    if unknown:
        raise PipelineBlocked(f"unsupported provider config keys: {sorted(unknown)}")
    if any(not isinstance(item, str) for item in value.values()):
        raise PipelineBlocked("provider config values must be strings")
    return {str(key): str(item) for key, item in value.items()}


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
    unknown = set(requested) - PROVIDER_CONFIG_KEYS
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
        "source": "explicit" if overrides else "discovered_layout_or_path",
        "backend_env": None if backend_env is None else str(Path(backend_env).absolute()),
        "workspace_root": (
            str(_discover_workspace_root(route_root))
            if _discover_workspace_root(route_root) is not None
            else None
        ),
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
