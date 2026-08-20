from __future__ import annotations

import pytest

from scripts.longsplat.gate_schema import (
    GATE_SCHEMA_EARLY_V1,
    GATE_SCHEMA_EARLY_V2,
    GATE_SCHEMA_FORMAL_V2,
    GateSchemaBlocked,
    normalize_gate_evidence,
)


def _flat_v2_early(**overrides):
    """A well-formed current v2 flat early-gate artifact (write-end shape)."""
    doc = {
        "schema_version": GATE_SCHEMA_EARLY_V2,
        "policy_id": "automated-technical-v1",
        "automated_technical_gate": True,
        "manual_visual_review": False,
        "computed_pass": True,
        "formal_release_eligible": True,
        "formal_auto_release": True,
        "held_out": False,
        "training_views_only": True,
        "decision": "pass",
        "reasons": [],
        "warnings": [],
        "policy": {"schema_version": "automated-technical-policy-v1", "policy_id": "automated-technical-v1"},
    }
    doc.update(overrides)
    return doc


def _legacy_v1_wrapper_early(**decision_overrides):
    """A historical v1 {policy, decision} wrapper (resume/pre-fix shape)."""
    decision = {
        "schema_version": GATE_SCHEMA_EARLY_V1,
        "policy_id": "automated-technical-v1",
        "automated_technical_gate": True,
        "manual_visual_review": False,
        "computed_pass": True,
        "formal_release_eligible": True,
        "formal_auto_release": True,
        "held_out": False,
        "training_views_only": True,
        "decision": "pass",
        "reasons": [],
        "warnings": [],
    }
    decision.update(decision_overrides)
    return {
        "policy": {"schema_version": "automated-technical-policy-v1", "policy_id": "automated-technical-v1"},
        "decision": decision,
    }


def test_v2_flat_direct_roundtrip_passes_and_keeps_kind():
    gate = _flat_v2_early()
    normalized = normalize_gate_evidence(gate)
    assert normalized["schema_version"] == GATE_SCHEMA_EARLY_V2
    assert normalized.get("normalized_from_legacy_wrapper") is not True
    assert normalized["automated_technical_gate"] is True
    assert normalized["manual_visual_review"] is False
    assert normalized["computed_pass"] is True


def test_v2_flat_requires_bool_judgment_fields():
    gate = _flat_v2_early(computed_pass="yes")
    with pytest.raises(GateSchemaBlocked, match="must be a boolean"):
        normalize_gate_evidence(gate)


def test_v2_flat_rejects_broken_invariants():
    with pytest.raises(GateSchemaBlocked, match="automated_technical_gate"):
        normalize_gate_evidence(_flat_v2_early(automated_technical_gate=False))
    with pytest.raises(GateSchemaBlocked, match="manual_visual_review"):
        normalize_gate_evidence(_flat_v2_early(manual_visual_review=True))


def test_legacy_v1_wrapper_is_promoted_to_flat_and_restamped():
    gate = _legacy_v1_wrapper_early()
    normalized = normalize_gate_evidence(gate)
    assert normalized["schema_version"] == GATE_SCHEMA_EARLY_V2
    assert normalized["normalized_from_legacy_wrapper"] is True
    # decision fields are at the top level, policy preserved
    assert normalized["computed_pass"] is True
    assert normalized["automated_technical_gate"] is True
    assert normalized["policy"]["policy_id"] == "automated-technical-v1"
    # the persisted wrapper sub-object is gone; the decision copy is the string
    # judgment marker at the top level, not a wrapped object
    assert normalized["decision"] == "pass"
    assert isinstance(normalized["decision"], str)


def test_legacy_wrapper_preserves_gate_kind_by_schema():
    gate = _legacy_v1_wrapper_early()
    normalized = normalize_gate_evidence(gate)
    assert normalized["schema_version"] == GATE_SCHEMA_EARLY_V2  # early, not formal


def test_legacy_formal_wrapper_restamps_to_formal_v2():
    gate = _legacy_v1_wrapper_early(schema_version="automated-formal-gate-v1")
    normalized = normalize_gate_evidence(gate)
    assert normalized["schema_version"] == GATE_SCHEMA_FORMAL_V2
    assert normalized["normalized_from_legacy_wrapper"] is True


def test_illegal_shape_without_schema_and_without_decision_is_rejected():
    with pytest.raises(GateSchemaBlocked, match="not a supported flat or legacy wrapper"):
        normalize_gate_evidence({"automated_technical_gate": True, "computed_pass": True})


def test_legacy_wrapper_with_unknown_decision_schema_is_rejected():
    gate = _legacy_v1_wrapper_early(schema_version="automated-early-gate-v9")
    with pytest.raises(GateSchemaBlocked, match="not a supported flat or legacy decision schema"):
        normalize_gate_evidence(gate)


def test_wrapper_with_non_object_decision_is_rejected():
    # A wrapper whose decision is not an object has nowhere to promote from;
    # it is rejected as an unsupported shape (never silently treated as a pass).
    with pytest.raises(GateSchemaBlocked, match="not a supported flat or legacy wrapper"):
        normalize_gate_evidence({"policy": {}, "decision": "not-an-object"})


def test_wrapper_without_policy_is_rejected():
    gate = _legacy_v1_wrapper_early()
    del gate["policy"]
    with pytest.raises(GateSchemaBlocked, match="no policy descriptor"):
        normalize_gate_evidence(gate)


def test_non_object_document_is_rejected():
    with pytest.raises(GateSchemaBlocked, match="must be an object"):
        normalize_gate_evidence([1, 2, 3])  # type: ignore[arg-type]

