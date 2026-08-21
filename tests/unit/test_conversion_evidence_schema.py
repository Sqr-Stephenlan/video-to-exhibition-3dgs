from __future__ import annotations

import pytest

from scripts.longsplat.conversion_evidence_schema import (
    ConvertedEvaluationSchemaError,
    normalize_converted_evaluation,
)


def _identity() -> dict[str, object]:
    return {
        "camera_count": 3,
        "camera_order": ["arbitrary-a.png", "arbitrary-b.png", "arbitrary-c.png"],
        "camera_dimensions": {"width": 37, "height": 23},
    }


def _legacy_postprocess() -> dict[str, object]:
    return {
        "schema_version": "longsplat-converted-eval-cpu-postprocess-v1",
        **_identity(),
        "STRUCTURAL_EVALUATION_PASS": True,
        "FULL_STREAM_VALIDATION_PASS": True,
        "SAME_CAMERA_VISUAL_PASS": "fail",
        "full_stream_validation": {
            "pass": False,
            "all_pngs_decoded": True,
            "all_pngs_finite": True,
            "all_dimensions_exact": True,
            "all_names_and_order_exact": True,
            "no_usable_converted_views": False,
        },
    }


def test_canonical_structural_and_visual_fields_are_independent() -> None:
    result = normalize_converted_evaluation(
        {
            "schema_version": "longsplat-converted-eval-cpu-postprocess-v2",
            **_identity(),
            "STRUCTURAL_EVALUATION_PASS": True,
            "FULL_STREAM_VALIDATION_PASS": True,
            "visual_quality_pass": True,
            "SAME_CAMERA_VISUAL_PASS": "needs_review",
            "full_stream_validation": {"pass": True},
        },
        expected_identity=_identity(),
    )
    assert result["STRUCTURAL_EVALUATION_PASS"] is True
    assert result["FULL_STREAM_VALIDATION_PASS"] is True
    assert result["visual_quality_pass"] is True


def test_legacy_mixed_nested_pass_requires_identity_and_structural_flags() -> None:
    result = normalize_converted_evaluation(_legacy_postprocess(), expected_identity=_identity())
    assert result["STRUCTURAL_EVALUATION_PASS"] is True
    assert result["FULL_STREAM_VALIDATION_PASS"] is True
    assert result["visual_quality_pass"] is False
    assert result["legacy_compatibility_warnings"]

    missing_identity = _legacy_postprocess()
    missing_identity.pop("camera_order")
    with pytest.raises(ConvertedEvaluationSchemaError, match="immutable identity"):
        normalize_converted_evaluation(missing_identity, expected_identity=_identity())


def test_contradictory_or_weak_legacy_evidence_is_blocked() -> None:
    contradictory = _legacy_postprocess()
    contradictory["FULL_STREAM_VALIDATION_PASS"] = False
    with pytest.raises(ConvertedEvaluationSchemaError, match="disagree"):
        normalize_converted_evaluation(contradictory, expected_identity=_identity())

    incomplete = _legacy_postprocess()
    del incomplete["STRUCTURAL_EVALUATION_PASS"]
    del incomplete["FULL_STREAM_VALIDATION_PASS"]
    del incomplete["full_stream_validation"]["all_names_and_order_exact"]  # type: ignore[index]
    with pytest.raises(ConvertedEvaluationSchemaError, match="incomplete"):
        normalize_converted_evaluation(incomplete, expected_identity=_identity())

    nested_contradiction = {
        "schema_version": "longsplat-converted-eval-cpu-postprocess-v2",
        "STRUCTURAL_EVALUATION_PASS": True,
        "FULL_STREAM_VALIDATION_PASS": True,
        "full_stream_validation": {"pass": True, "structural_pass": False},
    }
    with pytest.raises(ConvertedEvaluationSchemaError, match="structural_pass"):
        normalize_converted_evaluation(nested_contradiction)
