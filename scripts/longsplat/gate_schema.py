"""Shared schema + single validator for persisted automated gate evidence.

This module is the ONE place that defines, versions, and normalizes the
automated-gate artifact that crosses the producer/consumer boundary:

* producer: ``reconstruct_pipeline._automated_gate_stage`` writes it;
* consumers: ``formal_executor.plan_automated_formal_training`` and
  ``smoke_executor`` read it.

Layout contracts
----------------
The persisted artifact is a flat, top-level JSON object whose judgment fields
live at the top level (never wrapped under a ``decision`` sub-object).  Two
on-disk generations are accepted:

* v2 (current): ``schema_version`` is ``automated-early-gate-v2`` (early) or
  ``automated-formal-gate-v2`` (formal).  Judgment fields are already at the
  top level, alongside a nested ``policy`` descriptor.
* v1 (legacy, resume compatibility): the historical ``{policy, decision}``
  wrapper whose nested ``decision.schema_version`` is a recognized v1 decision
  schema.  ``decision`` is promoted to the top level and re-stamped to the
  matching v2 schema.

Nothing else is accepted; an ambiguous or mutated artifact is a structured
block, never a silent second-file read or wrapper-specific fallback.

Required gate judgment field contract (v2)
------------------------------------------
Versioned document contract for the flattened v2 gate.  This is the canonical
list of the fields a well-formed automated gate carries for downstream
judgment (consumed by formal_executor / smoke_executor and emitted by
automated_policy.evaluate_early_gate / evaluate_formal_gate).  Each row gives
field/type/required/invariant.

* schema_version -- str -- required -- must be a recognized v2 flat schema
  (automated-early-gate-v2 or automated-formal-gate-v2) after normalization;
  also encodes the gate stage (early vs formal).  A legacy v1 {policy,
  decision} wrapper is promoted and re-stamped to the matching v2 schema.  Not
  a boolean.
* computed_pass -- bool -- required -- must be a boolean (type-validated);
  carries the technical pass/fail outcome of the gate evaluation.
* decision -- str -- required -- the gate status marker, "pass" or "blocked"
  (a blocked gate is still well-formed evidence and must parse).
* automated_technical_gate -- bool -- required -- invariant must be exactly
  True (enforced literally).
* formal_auto_release -- bool -- required -- must be a boolean
  (type-validated); whether the gate would auto-release the formal stage.
* formal_release_eligible -- bool -- required -- must be a boolean
  (type-validated); the release-eligibility judgment consumed by
  formal_executor.
* manual_visual_review -- bool -- required -- invariant must be exactly False
  (enforced literally): the automated technical gate never manufactures a
  human/supervisor visual decision.
* held_out -- bool -- required -- must be a boolean (type-validated); an
  automated gate binds no held-out evidence, so a well-formed automated gate
  carries False.
* training_views_only -- bool -- required -- must be a boolean
  (type-validated); an automated gate judges training views only, so a
  well-formed automated gate carries True.
* structural -- Mapping -- required -- the render/training structural evidence
  block (count/order/finite contract).  Required and non-null.
* metrics -- Mapping with warning_only -- required -- the metrics gate block;
  carries a warning_only: True flag marking the advisory (non-blocking)
  metric-threshold semantics.
* contact_sheets + PNG hashes -- Mapping -- required -- hash-bound
  contact-sheet evidence: each sheet key maps to a record with path and
  sha256 strings.
* source / camera / pose / model / evidence paths + SHAs -- str -- required --
  provenance bindings: file paths and their hex sha256 digests so the judged
  artifact is hash-anchored to its inputs.

Bools in this contract are validated for type only by
:func:`normalize_gate_evidence` (their concrete value is meaningful but a
blocked gate is still well-formed and parses); the two invariant fields
automated_technical_gate (True) and manual_visual_review (False) are enforced
literally.
"""

from __future__ import annotations

from typing import Any, Mapping

from .pipeline_contract import PipelineBlocked

# Decision / flat schema versions (old v1 historical decision schemas and the
# current v2 flat schemas).
GATE_SCHEMA_EARLY_V1 = "automated-early-gate-v1"
GATE_SCHEMA_EARLY_V2 = "automated-early-gate-v2"
GATE_SCHEMA_FORMAL_V1 = "automated-formal-gate-v1"
GATE_SCHEMA_FORMAL_V2 = "automated-formal-gate-v2"

# Wrapper schemas: a v1 document is always the legacy ``{policy, decision}``
# wrapper; these decision schemas are promoted by :func:`normalize_gate_evidence`.
LEGACY_GATE_DECISION_SCHEMAS = {GATE_SCHEMA_EARLY_V1, GATE_SCHEMA_FORMAL_V1}
# Flat schemas: a v2 document is always the current flat shape read directly.
NEW_GATE_FLAT_SCHEMAS = {GATE_SCHEMA_EARLY_V2, GATE_SCHEMA_FORMAL_V2}

# Schema -> gate kind.  Stays identical across v1/v2 so a promoted legacy
# wrapper keeps the same gate kind it was written with.
GATE_KIND_BY_SCHEMA = {
    GATE_SCHEMA_EARLY_V1: "early",
    GATE_SCHEMA_EARLY_V2: "early",
    GATE_SCHEMA_FORMAL_V1: "formal",
    GATE_SCHEMA_FORMAL_V2: "formal",
}
# gate kind -> current v2 flat schema.
GATE_V2_SCHEMA_BY_KIND = {
    "early": GATE_SCHEMA_EARLY_V2,
    "formal": GATE_SCHEMA_FORMAL_V2,
}
GATE_KINDS = frozenset(GATE_V2_SCHEMA_BY_KIND)

# Invariant fields that are always literally this value in a well-formed gate
# (both a pass and a blocked gate carry them unchanged).
_GATE_INVARIANT_FIELDS = {
    "automated_technical_gate": True,
    "manual_visual_review": False,
}
# Judgment booleans validated for type only (their concrete value is meaningful
# but a blocked gate is still well-formed evidence and must parse).
_GATE_BOOL_FIELDS = (
    "computed_pass",
    "formal_release_eligible",
    "formal_auto_release",
    "held_out",
    "training_views_only",
)


class GateSchemaBlocked(PipelineBlocked):
    """The persisted automated gate artifact is not a supported envelope shape."""


def normalize_gate_evidence(doc):
    """Return the single canonical flat automated gate evidence shape.

    Three states are accepted:

    1. a current v2 flat artifact (top-level judgment fields with a v2 flat
       ``schema_version``) -- validated directly;
    2. a historical v1 ``{policy, decision}`` wrapper whose nested
       ``decision.schema_version`` is a recognized v1 decision schema -- the
       decision is strictly promoted to the top level and re-stamped to the
       matching v2 schema;
    3. anything else -- a structured ``GateSchemaBlocked``.
    """

    if not isinstance(doc, dict):
        raise GateSchemaBlocked("automated gate evidence must be an object")

    schema = doc.get("schema_version")
    if schema in NEW_GATE_FLAT_SCHEMAS:
        candidate = dict(doc)
        promoted = False
    elif schema is None and isinstance(doc.get("decision"), dict):
        candidate = _promote_legacy_wrapper(doc)
        promoted = True
    else:
        raise GateSchemaBlocked(
            "automated gate evidence schema {!r} is not a supported flat or legacy wrapper shape".format(schema)
        )

    kind = GATE_KIND_BY_SCHEMA.get(candidate.get("schema_version"))
    if kind not in GATE_KINDS:
        raise GateSchemaBlocked("automated gate evidence gate kind {!r} is invalid".format(kind))

    for key, expected in _GATE_INVARIANT_FIELDS.items():
        if candidate.get(key) is not expected:
            raise GateSchemaBlocked("automated gate evidence invariant field {!r} must be {!r}".format(key, expected))
    for key in _GATE_BOOL_FIELDS:
        if not isinstance(candidate.get(key), bool):
            raise GateSchemaBlocked("automated gate evidence field {!r} must be a boolean".format(key))

    normalized = dict(candidate)
    normalized["schema_version"] = GATE_V2_SCHEMA_BY_KIND[kind]
    if promoted:
        normalized["normalized_from_legacy_wrapper"] = True
    return normalized


def _promote_legacy_wrapper(gate):
    """Promote the historical ``{policy, decision}`` wrapper to the flat shape.

    This is the only compatibility boundary for the pre-v2 gate artifact.  It
    requires the wrapper to contain a nested ``decision`` whose ``schema_version``
    is a recognized legacy decision schema and a nested ``policy`` descriptor;
    anything else is rejected so an ambiguous or mutated wrapper can never be
    silently interpreted as a technical pass.
    """

    decision = gate.get("decision")
    if not isinstance(decision, dict):
        raise GateSchemaBlocked("legacy automated gate wrapper has no decision object")
    scheme = decision.get("schema_version")
    if scheme not in LEGACY_GATE_DECISION_SCHEMAS:
        raise GateSchemaBlocked(
            "automated gate evidence schema {!r} is not a supported flat or legacy decision schema".format(scheme)
        )
    if not isinstance(gate.get("policy"), dict):
        raise GateSchemaBlocked("legacy automated gate wrapper has no policy descriptor")
    promoted = {key: value for key, value in decision.items()}
    promoted["policy"] = dict(gate["policy"])
    return promoted

