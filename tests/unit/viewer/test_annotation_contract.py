from __future__ import annotations

import pytest

from scripts.viewer.annotations import AnnotationContractError, validate_annotations


def _payload():
    return {
        "schema_version": "annotations-v1",
        "model_sha256": "a" * 64,
        "coordinate_system": {
            "name": "model-local",
            "version": "viewer-v1",
            "up_axis": "y",
            "units": "meters",
        },
        "items": [
            {
                "id": "exhibit-1",
                "type": "exhibit",
                "name": "入口展品",
                "position": [0.0, 1.0, 2.0],
                "rotation_quaternion": [0.0, 0.0, 0.0, 1.0],
                "description": "入口",
                "media": [],
            }
        ],
    }


def test_validate_annotations_requires_model_and_coordinate_binding():
    result = validate_annotations(
        _payload(),
        model_sha256="a" * 64,
        coordinate_system_version="viewer-v1",
    )

    assert result["schema_version"] == "annotations-v1"
    assert result["items"][0]["id"] == "exhibit-1"


def test_validate_annotations_rejects_mismatched_model_or_duplicate_ids():
    payload = _payload()
    with pytest.raises(AnnotationContractError, match="model"):
        validate_annotations(
            payload,
            model_sha256="b" * 64,
            coordinate_system_version="viewer-v1",
        )

    payload["items"].append(dict(payload["items"][0]))
    with pytest.raises(AnnotationContractError, match="unique"):
        validate_annotations(
            payload,
            model_sha256="a" * 64,
            coordinate_system_version="viewer-v1",
        )


def test_validate_annotations_rejects_bad_vectors_and_unknown_types():
    payload = _payload()
    payload["items"][0]["position"] = [0.0, 1.0]
    with pytest.raises(AnnotationContractError, match="position"):
        validate_annotations(
            payload,
            model_sha256="a" * 64,
            coordinate_system_version="viewer-v1",
        )

    payload = _payload()
    payload["items"][0]["type"] = "unknown"
    with pytest.raises(AnnotationContractError, match="type"):
        validate_annotations(
            payload,
            model_sha256="a" * 64,
            coordinate_system_version="viewer-v1",
        )
