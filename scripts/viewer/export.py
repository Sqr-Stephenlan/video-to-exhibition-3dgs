from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import uuid
from pathlib import Path
from urllib.parse import urlencode, urlparse
from typing import Any, Mapping

from .ply_contract import GaussianPlyError, validate_gaussian_ply


class ModelExportError(RuntimeError):
    """Raised when a viewer model cannot be safely produced."""


def _identity(path: Path) -> dict[str, Any]:
    try:
        identity = validate_gaussian_ply(path)
    except GaussianPlyError as exc:
        raise ModelExportError(f"Gaussian PLY validation failed: {exc}") from exc
    return {
        "filename": identity.filename,
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
        "vertices": identity.vertices,
        "properties": list(identity.properties),
        "schema_version": identity.schema_version,
    }


def _coordinate_system(value: Mapping[str, Any]) -> dict[str, Any]:
    name = value.get("name")
    version = value.get("version")
    if not isinstance(name, str) or not name:
        raise ModelExportError("coordinate system name is required")
    if not isinstance(version, str) or not version:
        raise ModelExportError("coordinate system version is required")
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, (str, int, float, bool))
    }


def export_web_model(
    *,
    source_ply: str | Path,
    destination_root: str | Path,
    target_format: str,
    coordinate_system: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a viewer-ready PLY and a path-free identity manifest.

    PLY is a deliberate first-version pass-through because the repository does
    not currently contain a .compressed.ply or SOG converter.
    """

    source = Path(source_ply)
    destination_root_path = Path(destination_root)
    if target_format != "ply":
        raise ModelExportError(
            f"converter for {target_format} is not configured; only PLY passthrough is available"
        )
    if source.is_symlink() or not source.is_file():
        raise ModelExportError("source PLY is missing or symlinked")
    destination_root_path.mkdir(parents=True, exist_ok=True)
    resolved_root = destination_root_path.resolve(strict=True)
    destination = destination_root_path / source.name
    try:
        destination.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise ModelExportError("viewer model destination escapes its root") from exc
    if destination.is_symlink():
        raise ModelExportError("viewer model destination is symlinked")

    source_identity = _identity(source)
    if source.resolve() != destination.resolve():
        temporary = destination_root_path / f".{destination.name}.{uuid.uuid4().hex}.part"
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    target_identity = _identity(destination)
    if (
        source_identity["sha256"] != target_identity["sha256"]
        or source_identity["size_bytes"] != target_identity["size_bytes"]
    ):
        raise ModelExportError("viewer model changed while it was being published")

    return {
        "schema_version": "viewer-model-v1",
        "source": source_identity,
        "target": {
            **target_identity,
            "format": target_format,
        },
        "coordinate_system": _coordinate_system(coordinate_system),
        "gaussian_schema_version": source_identity["schema_version"],
        "conversion": {
            "converter": "passthrough",
            "version": "viewer-ply-v1",
        },
        "format_compatible": True,
        "runtime_verified": False,
        "manual_review": False,
    }


def build_supersplat_editor_link(
    splat_url: str,
    *,
    viewer_base: str = "https://superspl.at/editor",
) -> str:
    source = urlparse(splat_url)
    viewer = urlparse(viewer_base)
    if source.scheme not in {"http", "https"} or not source.netloc:
        raise ValueError("SuperSplat load URL must be an absolute HTTP(S) URL")
    if viewer.scheme not in {"http", "https"} or not viewer.netloc:
        raise ValueError("SuperSplat viewer base must be an absolute HTTP(S) URL")
    separator = "&" if viewer.query else "?"
    return f"{viewer_base}{separator}{urlencode({'load': splat_url})}"
