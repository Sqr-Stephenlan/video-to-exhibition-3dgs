"""Single interpretation boundary for converted-evaluation evidence.

The converted evaluator and the CPU postprocess historically used several
names for the same structural contract.  This module gives consumers one
strict interpretation while keeping visual quality as a separate result.  A
legacy postprocess record is accepted only when its explicit structural
fields and immutable camera identity are still present and consistent.
"""

from __future__ import annotations

from typing import Any, Mapping


LEGACY_POSTPROCESS_SCHEMA = "longsplat-converted-eval-cpu-postprocess-v1"
_STRUCTURAL_FLAGS = (
    "all_pngs_decoded",
    "all_pngs_finite",
    "all_dimensions_exact",
    "all_names_and_order_exact",
)
_LEGACY_COUNT_FIELDS = ("triples_processed", "expected_triples")


class ConvertedEvaluationSchemaError(ValueError):
    """Converted-evaluation evidence is missing, contradictory, or unsafe."""


def _bool_field(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConvertedEvaluationSchemaError(f"{field} must be a boolean")
    return value


def _legacy_structural_pass(result: Mapping[str, Any]) -> bool:
    validation = result.get("full_stream_validation")
    if not isinstance(validation, Mapping):
        raise ConvertedEvaluationSchemaError(
            "legacy converted-evaluation evidence lacks full_stream_validation structural evidence"
        )
    camera_count = result.get("camera_count")
    if isinstance(camera_count, bool) or not isinstance(camera_count, int) or camera_count <= 0:
        raise ConvertedEvaluationSchemaError(
            "legacy converted-evaluation evidence lacks a valid camera_count for structural binding"
        )
    missing = [field for field in _STRUCTURAL_FLAGS if validation.get(field) is not True]
    if missing:
        raise ConvertedEvaluationSchemaError(
            "legacy converted-evaluation structural evidence is incomplete: " + ", ".join(missing)
        )
    counts: dict[str, int] = {}
    for field in _LEGACY_COUNT_FIELDS:
        value = validation.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConvertedEvaluationSchemaError(
                f"legacy converted-evaluation structural evidence lacks valid {field}"
            )
        counts[field] = value
    if counts["expected_triples"] != camera_count or counts["triples_processed"] != counts["expected_triples"]:
        raise ConvertedEvaluationSchemaError(
            "legacy converted-evaluation structural evidence count differs from camera_count"
        )
    no_usable = validation.get("no_usable_converted_views")
    if not isinstance(no_usable, bool):
        raise ConvertedEvaluationSchemaError(
            "legacy converted-evaluation evidence lacks boolean no_usable_converted_views"
        )
    return not no_usable


def _check_identity(
    result: Mapping[str, Any],
    expected_identity: Mapping[str, Any] | None,
    *,
    required: bool,
) -> None:
    fields = ("camera_count", "camera_order", "camera_dimensions")
    identity_present = any(field in result for field in fields)
    if required or identity_present:
        count = result.get("camera_count")
        order = result.get("camera_order")
        dimensions = result.get("camera_dimensions")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ConvertedEvaluationSchemaError(
                "converted-evaluation immutable identity camera_count is missing or invalid"
            )
        if (
            not isinstance(order, list)
            or len(order) != count
            or not all(isinstance(value, str) and value for value in order)
            or len(set(order)) != len(order)
        ):
            raise ConvertedEvaluationSchemaError(
                "converted-evaluation immutable identity camera_order is missing or invalid"
            )
        if not isinstance(dimensions, Mapping):
            raise ConvertedEvaluationSchemaError(
                "converted-evaluation immutable identity camera_dimensions is missing or invalid"
            )
        for dimension in ("width", "height"):
            value = dimensions.get(dimension)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConvertedEvaluationSchemaError(
                    f"converted-evaluation immutable identity camera_dimensions is missing or invalid: {dimension}"
                )
    if expected_identity is None:
        return
    if expected_identity is None:
        return
    for field in fields:
        actual = result.get(field)
        expected = expected_identity.get(field)
        if actual is None:
            if required:
                raise ConvertedEvaluationSchemaError(
                    f"legacy converted-evaluation evidence lacks immutable identity field: {field}"
                )
            continue
        if actual != expected:
            raise ConvertedEvaluationSchemaError(
                f"converted-evaluation immutable identity differs for {field}"
            )


def normalize_converted_evaluation(
    result: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return canonical structural/visual fields for one evaluation record.

    ``STRUCTURAL_EVALUATION_PASS`` and ``FULL_STREAM_VALIDATION_PASS`` are
    aliases for the same structural contract: ordered finite render evidence
    with at least one usable converted view.  ``SAME_CAMERA_VISUAL_PASS`` and
    ``visual_quality_pass`` describe visual quality only and never weaken the
    structural safety requirement.
    """

    if not isinstance(result, Mapping):
        raise ConvertedEvaluationSchemaError("converted-evaluation result must be an object")
    schema_version = result.get("schema_version")
    legacy_v1 = schema_version == LEGACY_POSTPROCESS_SCHEMA
    structural_values: list[bool] = []
    for field in ("STRUCTURAL_EVALUATION_PASS", "FULL_STREAM_VALIDATION_PASS"):
        if field in result:
            structural_values.append(_bool_field(result[field], field))
    compatibility_warnings: list[str] = []
    if legacy_v1:
        # Schema identity must control the interpretation path.  A legacy
        # record cannot opt out of its full-stream evidence merely by adding
        # the newer top-level aliases.
        structural_pass = _legacy_structural_pass(result)
        if structural_values:
            if len(set(structural_values)) != 1:
                raise ConvertedEvaluationSchemaError(
                    "STRUCTURAL_EVALUATION_PASS and FULL_STREAM_VALIDATION_PASS disagree"
                )
            if structural_values[0] != structural_pass:
                raise ConvertedEvaluationSchemaError(
                    "legacy top-level structural aliases disagree with verified full-stream evidence"
                )
        else:
            compatibility_warnings.append("legacy_structural_pass_derived_from_verified_full_stream_flags")
    else:
        if not structural_values:
            raise ConvertedEvaluationSchemaError(
                "converted-evaluation structural pass is missing; no supported legacy schema fallback"
            )
        if len(set(structural_values)) != 1:
            raise ConvertedEvaluationSchemaError(
                "STRUCTURAL_EVALUATION_PASS and FULL_STREAM_VALIDATION_PASS disagree"
            )
        structural_pass = structural_values[0]

    validation = result.get("full_stream_validation")
    nested_pass: bool | None = None
    if validation is not None:
        if not isinstance(validation, Mapping):
            raise ConvertedEvaluationSchemaError("full_stream_validation must be an object")
        if "pass" in validation:
            nested_pass = _bool_field(validation["pass"], "full_stream_validation.pass")
        if "structural_pass" in validation:
            nested_structural = _bool_field(
                validation["structural_pass"],
                "full_stream_validation.structural_pass",
            )
            if nested_structural != structural_pass:
                raise ConvertedEvaluationSchemaError(
                    "full_stream_validation.structural_pass disagrees with canonical structural pass"
                )
        if legacy_v1:
            # The fallback above already proved these fields.  Keep the
            # explicit check here so a malformed mixed record cannot pass by
            # virtue of a single legacy boolean.
            _legacy_structural_pass(result)
            if nested_pass is not None and nested_pass != structural_pass:
                # v1 postprocess wrote the visual result into nested ``pass``
                # while its top-level structural aliases were true.  Reconcile
                # only that exact, independently verifiable legacy shape.
                legacy_visual_mixed = (
                    nested_pass is False
                    and structural_pass is True
                )
                if not legacy_visual_mixed:
                    raise ConvertedEvaluationSchemaError(
                        "full_stream_validation.pass disagrees with canonical structural pass"
                    )
                compatibility_warnings.append(
                    "legacy_nested_full_stream_pass_was_visual_mixed; canonical_structural_pass_retained"
                )
        elif nested_pass is not None and nested_pass != structural_pass:
            raise ConvertedEvaluationSchemaError(
                "full_stream_validation.pass disagrees with canonical structural pass"
            )

    same_camera = result.get("SAME_CAMERA_VISUAL_PASS")
    if same_camera is not None and same_camera not in {"pass", "fail", "needs_review"}:
        raise ConvertedEvaluationSchemaError(
            "SAME_CAMERA_VISUAL_PASS must be pass, fail, or needs_review"
        )
    derived_visual: bool | None = None
    if same_camera == "pass":
        derived_visual = True
    elif same_camera == "fail":
        derived_visual = False
    else:
        # ``needs_review`` is an explicit absence of a visual claim, not a
        # failed automatic quality result.  A producer may still provide the
        # independent boolean visual_quality_pass from numeric evidence.
        derived_visual = None
    explicit_visual = result.get("visual_quality_pass")
    if explicit_visual is not None:
        explicit_visual = _bool_field(explicit_visual, "visual_quality_pass")
        if derived_visual is not None and explicit_visual != derived_visual:
            raise ConvertedEvaluationSchemaError(
                "visual_quality_pass and SAME_CAMERA_VISUAL_PASS disagree"
            )
    visual_pass = explicit_visual if explicit_visual is not None else derived_visual
    _check_identity(
        result,
        expected_identity,
        required=legacy_v1,
    )

    normalized = dict(result)
    normalized["STRUCTURAL_EVALUATION_PASS"] = structural_pass
    normalized["FULL_STREAM_VALIDATION_PASS"] = structural_pass
    normalized["visual_quality_pass"] = visual_pass
    normalized["legacy_compatibility_warnings"] = compatibility_warnings
    return normalized
