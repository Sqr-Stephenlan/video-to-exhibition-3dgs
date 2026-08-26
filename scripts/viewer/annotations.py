from __future__ import annotations

import math
import re
from typing import Any, Mapping


class AnnotationContractError(ValueError):
    """Raised when annotation data is not bound to a model contract."""


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_ITEM_TYPES = {"exhibit", "zone", "path", "hotspot"}


def _vector(value: Any, *, length: int, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise AnnotationContractError(f"{field} must contain {length} numbers")
    result: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)):
            raise AnnotationContractError(f"{field} must contain finite numbers")
        result.append(float(item))
    return result


def validate_annotations(
    payload: Mapping[str, Any],
    *,
    model_sha256: str,
    coordinate_system_version: str,
) -> dict[str, Any]:
    if payload.get("schema_version") != "annotations-v1":
        raise AnnotationContractError("annotation schema_version must be annotations-v1")
    if not _SHA256.fullmatch(model_sha256):
        raise AnnotationContractError("expected model SHA-256 is invalid")
    if payload.get("model_sha256") != model_sha256:
        raise AnnotationContractError("annotation model SHA does not match the model")

    coordinate = payload.get("coordinate_system")
    if not isinstance(coordinate, Mapping):
        raise AnnotationContractError("coordinate_system is required")
    if coordinate.get("version") != coordinate_system_version:
        raise AnnotationContractError("annotation coordinate system version does not match")
    if not isinstance(coordinate.get("name"), str) or not coordinate["name"]:
        raise AnnotationContractError("coordinate system name is required")

    items = payload.get("items")
    if not isinstance(items, list):
        raise AnnotationContractError("annotation items must be a list")
    seen: set[str] = set()
    normalized_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise AnnotationContractError("annotation item must be an object")
        item_id = item.get("id")
        item_type = item.get("type")
        if not isinstance(item_id, str) or not item_id:
            raise AnnotationContractError("annotation item id is required")
        if item_id in seen:
            raise AnnotationContractError("annotation item IDs must be unique")
        seen.add(item_id)
        if item_type not in _ITEM_TYPES:
            raise AnnotationContractError("annotation item type is unsupported")
        normalized: dict[str, Any] = {
            "id": item_id,
            "type": item_type,
            "name": str(item.get("name") or item_id),
            "position": _vector(item.get("position"), length=3, field="position"),
        }
        if "rotation_quaternion" in item:
            normalized["rotation_quaternion"] = _vector(
                item["rotation_quaternion"],
                length=4,
                field="rotation_quaternion",
            )
        if "description" in item:
            normalized["description"] = str(item["description"])
        if "media" in item:
            media = item["media"]
            if not isinstance(media, list):
                raise AnnotationContractError("media must be a list")
            normalized_media: list[dict[str, str]] = []
            for entry in media:
                if not isinstance(entry, Mapping):
                    raise AnnotationContractError("media entry must be an object")
                url = entry.get("url")
                if not isinstance(url, str) or not (
                    url.startswith("https://")
                    or url.startswith("http://")
                    or url.startswith("/api/")
                ):
                    raise AnnotationContractError("media URL is not allowed")
                normalized_media.append(
                    {
                        "kind": str(entry.get("kind") or "link"),
                        "url": url,
                        "title": str(entry.get("title") or ""),
                    }
                )
            normalized["media"] = normalized_media
        normalized_items.append(normalized)

    return {
        "schema_version": "annotations-v1",
        "model_sha256": model_sha256,
        "coordinate_system": dict(coordinate),
        "items": normalized_items,
    }
