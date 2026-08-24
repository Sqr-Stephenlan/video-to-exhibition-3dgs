"""Versioned, CPU-verifiable technical gates for the default one-click chain.

These gates are deliberately narrower than application or human visual
acceptance.  They only decide whether the next frozen technical stage may be
run, and bind every decision to the stage evidence supplied by the caller.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .convergence_smoke import ConvergenceSmokeBlocked, predicted_exposure
from .pipeline_contract import PipelineBlocked
from .render_postcheck import normalize_postcheck_evidence
from .gate_schema import normalize_gate_evidence  # re-export shared producer/consumer validator


POLICY_ID = "automated-technical-v1"
POLICY_SCHEMA = "automated-technical-policy-v1"
# The persisted automated gate evidence artifact and its single
# producer/consumer validator live in :mod:`scripts.longsplat.gate_schema`.
# This module only decides the gate and sets the flat v2 decision
# ``schema_version``; it no longer defines or normalizes the envelope shape.
# ``normalize_gate_evidence`` is re-exported here so existing call sites that
# import it from this module keep working unchanged.
EARLY_THRESHOLDS = {
    "psnr_mean_db_min": 15.0,
    "psnr_min_db_min": 10.0,
    "ssim_mean_min": 0.60,
    "ssim_min_min": 0.45,
}
FORMAL_THRESHOLDS = {
    "psnr_mean_db_min": 20.0,
    "psnr_min_db_min": 15.0,
    "ssim_mean_min": 0.75,
    "ssim_min_min": 0.60,
}


class AutomatedPolicyBlocked(PipelineBlocked):
    """Technical evidence cannot release the next frozen stage."""


def _fail(message: str) -> None:
    raise AutomatedPolicyBlocked(message)


def policy_descriptor() -> dict[str, Any]:
    return {
        "schema_version": POLICY_SCHEMA,
        "policy_id": POLICY_ID,
        "classification": "local_evidence_driven_automated_policy",
        "manual_visual_review": False,
        "held_out": False,
        "official_formula": False,
        "human_visual_claim": False,
        "cross_gap_policy": "record_warning_unless_structural_failure",
        "automatic_iteration_search": False,
        "quality_thresholds_are_warning_only": True,
        "early_thresholds": dict(EARLY_THRESHOLDS),
        "formal_thresholds": dict(FORMAL_THRESHOLDS),
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    return value


def _training_structural(training: Mapping[str, Any]) -> Mapping[str, Any]:
    if training.get("computed_pass") is not True:
        _fail("training stage is not a computed structural pass")
    executor = training.get("executor_result")
    if isinstance(executor, Mapping):
        structural = executor.get("structural")
        if isinstance(structural, Mapping):
            if structural.get("structural_pass") is not True:
                _fail("training executor structural pass is false")
            return structural
    structural = training.get("structural")
    if isinstance(structural, Mapping) and structural.get("structural_pass", True) is not False:
        return structural
    _fail("training structural evidence is missing")


def _telemetry_gate(
    structural: Mapping[str, Any],
    *,
    iterations: int,
    require_two_rounds: bool,
) -> dict[str, Any]:
    telemetry = _mapping(structural.get("camera_sampling_telemetry"), "camera sampling telemetry")
    count = structural.get("active_camera_count")
    if not isinstance(count, int) or count <= 0:
        count = telemetry.get("active_camera_count")
    if not isinstance(count, int) or count <= 0:
        _fail("active camera count is missing from training structural evidence")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        _fail("automated gate iteration count is invalid")
    coverage_warnings: list[str] = []
    if require_two_rounds and iterations < 2 * count:
        coverage_warnings.append("camera sampling has fewer than two complete coverage rounds; advisory only")
    if telemetry.get("iterations") != iterations:
        _fail("camera sampling telemetry iteration count differs from policy")
    if telemetry.get("active_camera_count") != count:
        _fail("camera sampling telemetry active camera count differs from policy")
    unique_count = telemetry.get("unique_camera_count")
    if isinstance(unique_count, bool) or not isinstance(unique_count, int) or not 0 <= unique_count <= count:
        coverage_warnings.append("camera sampling unique-camera telemetry is unavailable or out of range; advisory only")
        unique_count = None
    elif unique_count != count:
        coverage_warnings.append("camera sampling did not expose every active camera; advisory only")
    exposures = _mapping(telemetry.get("exposure_counts"), "exposure counts")
    if len(exposures) != count:
        _fail("exposure count identity set differs from active camera count")
    try:
        prediction = predicted_exposure(count, iterations)
    except ConvergenceSmokeBlocked as exc:
        _fail(str(exc))
    values = [value for value in exposures.values()]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        _fail("exposure counts are not integers")
    if any(value not in (prediction["min_exposure_count"], prediction["max_exposure_count"]) for value in values):
        _fail("exposure counts are outside the floor/ceil distribution")
    if prediction["min_exposure_count"] == prediction["max_exposure_count"]:
        if values.count(prediction["min_exposure_count"]) != count:
            _fail("uniform exposure count differs from the deterministic policy")
    else:
        if values.count(prediction["min_exposure_count"]) != prediction["camera_count_at_min_exposure"]:
            _fail("floor-exposure camera count differs from the deterministic policy")
        if values.count(prediction["max_exposure_count"]) != prediction["camera_count_at_max_exposure"]:
            _fail("ceil-exposure camera count differs from the deterministic policy")
    if sum(values) != iterations:
        _fail("exposure count sum differs from iteration count")
    zero = sum(value == 0 for value in values)
    if zero:
        coverage_warnings.append(f"{zero} active camera(s) have zero exposure in this fixed iteration budget; advisory only")
    anchor = _mapping(structural.get("anchor_schedule"), "anchor runtime schedule")
    if anchor.get("observed_from_runtime") is not True:
        _fail("anchor schedule was not observed from runtime")
    if not isinstance(anchor.get("observed_events"), list):
        _fail("anchor runtime events are missing")
    checkpoint = _mapping(structural.get("checkpoint"), "training checkpoint")
    if checkpoint.get("finite") is not True:
        _fail("training checkpoint is not explicitly finite")
    return {
        "active_camera_count": count,
        "iterations": iterations,
        "exposure_counts": dict(exposures),
        "exposure_prediction": prediction,
        "coverage_advisory": {
            "policy": "advisory_only_no_complete_round_or_full_camera_coverage_gate",
            "unique_camera_count": unique_count,
            "zero_exposure_camera_count": zero,
            "complete_rounds": prediction["complete_rounds"],
            "predicted_coverage_fraction": prediction["predicted_coverage_fraction"],
            "warnings": coverage_warnings,
        },
        "zero_exposure_camera_count": zero,
        "anchor_schedule": dict(anchor),
        "checkpoint": dict(checkpoint),
    }


def _postcheck_evidence(postcheck: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        evidence = normalize_postcheck_evidence(postcheck)
    except (OSError, TypeError, ValueError) as exc:
        _fail(f"render postcheck evidence normalization failed: {exc}")
    if evidence.get("computed_pass") is not True:
        _fail("render postcheck is not a computed pass")
    structural = _mapping(evidence.get("structural"), "render postcheck structural evidence")
    if structural.get("render_exit_code") != 0:
        _fail("render postcheck render exit is not zero")
    required = (
        "fixed_render_count_exact",
        "training_gt_count_exact",
        "nvs_count_from_actual_pose_contract",
        "all_pngs_decoded_finite_and_dimensions_exact",
        "camera_order_exact",
    )
    if any(structural.get(key) is not True for key in required):
        _fail("render postcheck structural count/order/finite contract is incomplete")
    warnings: list[str] = []
    contact_sheets = evidence.get("contact_sheets")
    if not isinstance(contact_sheets, Mapping):
        warnings.append("contact-sheet evidence is unavailable; technical gate does not depend on presentation sheets")
    else:
        for key in ("fixed_vs_gt", "on_path"):
            record = contact_sheets.get(key)
            if not isinstance(record, Mapping) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
                warnings.append(f"{key} contact sheet is unavailable or not hash-bound")
    for key in ("metrics", "png_hashes"):
        record = evidence.get(f"{key}_artifact")
        if not isinstance(record, Mapping):
            record = evidence.get(key)
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
            warnings.append(f"{key} artifact is unavailable or not hash-bound")
    counts = _mapping(evidence.get("counts"), "render postcheck counts")
    if not all(isinstance(counts.get(key), int) and counts.get(key) > 0 for key in ("fixed_render", "training_gt", "nvs_on_path")):
        _fail("render postcheck counts are incomplete")
    rough = _mapping(evidence.get("rough_visual"), "rough render health")
    if rough.get("automatic_health") == "fail":
        _fail("render postcheck reports black/unusable or non-finite fixed views")
    if rough.get("automatic_health") in {"needs_review", "unclassified"}:
        warnings.append(f"rough visual health is {rough.get('automatic_health')}; no human visual claim is made")
    metrics = _mapping(evidence.get("fixed_view_metrics"), "training-view metrics")
    values = {
        "psnr_mean_db": metrics.get("mean_psnr_db"),
        "psnr_min_db": metrics.get("min_psnr_db"),
        "ssim_mean": metrics.get("mean_ssim"),
        "ssim_min": metrics.get("min_ssim"),
    }
    if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values.values()):
        _fail("training-view metrics are missing or non-finite")
    normalized = dict(evidence)
    normalized["technical_warnings"] = warnings
    return normalized


def _thresholds_gate(metrics: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    observed = {
        "psnr_mean_db": float(metrics["mean_psnr_db"]),
        "psnr_min_db": float(metrics["min_psnr_db"]),
        "ssim_mean": float(metrics["mean_ssim"]),
        "ssim_min": float(metrics["min_ssim"]),
    }
    checks = {
        "psnr_mean_db": observed["psnr_mean_db"] >= float(thresholds["psnr_mean_db_min"]),
        "psnr_min_db": observed["psnr_min_db"] >= float(thresholds["psnr_min_db_min"]),
        "ssim_mean": observed["ssim_mean"] >= float(thresholds["ssim_mean_min"]),
        "ssim_min": observed["ssim_min"] >= float(thresholds["ssim_min_min"]),
    }
    return {
        "observed": observed,
        "thresholds": dict(thresholds),
        "checks": checks,
        "pass": all(checks.values()),
        "warning_only": True,
    }


def _cross_gap_gate(evidence: Mapping[str, Any]) -> dict[str, Any]:
    cross_gap = evidence.get("cross_gap")
    if not isinstance(cross_gap, Mapping):
        return {"present": False, "policy": "record_warning_unless_structural_failure", "pass": True, "warning": False}
    present = cross_gap.get("cross_gap") is True
    return {
        "present": present,
        "policy": "record_warning_unless_structural_failure",
        "record": dict(cross_gap),
        "pass": True,
        "warning": present,
    }


def evaluate_early_gate(*, training: Mapping[str, Any], postcheck: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the one-shot 1000-it technical release boundary."""

    reasons: list[str] = []
    warnings: list[str] = []
    try:
        structural = _training_structural(training)
        runtime = _telemetry_gate(structural, iterations=1000, require_two_rounds=False)
        evidence = _postcheck_evidence(postcheck)
        metric_gate = _thresholds_gate(_mapping(evidence["fixed_view_metrics"], "training-view metrics"), EARLY_THRESHOLDS)
        cross_gap = _cross_gap_gate(evidence)
        if not metric_gate["pass"]:
            warnings.append("early metric thresholds are below the local advisory values")
        if cross_gap.get("warning"):
            warnings.append("cross-gap is explicitly present and retained as provenance")
        warnings.extend(str(item) for item in runtime.get("coverage_advisory", {}).get("warnings", []))
        warnings.extend(str(item) for item in evidence.get("technical_warnings", []))
        passed = not reasons
    except AutomatedPolicyBlocked as exc:
        evidence = postcheck.get("postcheck_result") if isinstance(postcheck.get("postcheck_result"), Mapping) else {}
        runtime = {"available": False}
        metric_gate = {"pass": False, "reason": str(exc)}
        cross_gap = _cross_gap_gate(evidence)
        reasons.append(str(exc))
        passed = False
    return {
        "schema_version": "automated-early-gate-v2",
        "policy_id": POLICY_ID,
        "automated_technical_gate": True,
        "manual_visual_review": False,
        "formal_release_eligible": passed,
        "formal_auto_release": passed,
        "computed_pass": passed,
        "decision": "pass" if passed else "blocked",
        "reasons": reasons,
        "warnings": warnings,
        "runtime": runtime,
        "metrics": metric_gate,
        "cross_gap": cross_gap,
        "evidence_bound": postcheck.get("postcheck_result_path"),
        "held_out": False,
        "training_views_only": True,
        "human_visual_claim": False,
    }


def evaluate_formal_gate(*, training: Mapping[str, Any], postcheck: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the 30000-it native render technical boundary."""

    reasons: list[str] = []
    warnings: list[str] = []
    try:
        structural = _training_structural(training)
        runtime = _telemetry_gate(structural, iterations=30000, require_two_rounds=False)
        evidence = _postcheck_evidence(postcheck)
        metric_gate = _thresholds_gate(_mapping(evidence["fixed_view_metrics"], "training-view metrics"), FORMAL_THRESHOLDS)
        cross_gap = _cross_gap_gate(evidence)
        if not metric_gate["pass"]:
            warnings.append("formal metric thresholds are below the local advisory values")
        if cross_gap.get("warning"):
            warnings.append("cross-gap is explicitly present and retained as provenance")
        warnings.extend(str(item) for item in runtime.get("coverage_advisory", {}).get("warnings", []))
        warnings.extend(str(item) for item in evidence.get("technical_warnings", []))
        passed = not reasons
    except AutomatedPolicyBlocked as exc:
        evidence = postcheck.get("postcheck_result") if isinstance(postcheck.get("postcheck_result"), Mapping) else {}
        runtime = {"available": False}
        metric_gate = {"pass": False, "reason": str(exc)}
        cross_gap = _cross_gap_gate(evidence)
        reasons.append(str(exc))
        passed = False
    return {
        "schema_version": "automated-formal-gate-v2",
        "policy_id": POLICY_ID,
        "automated_technical_gate": True,
        "manual_visual_review": False,
        "formal_release_eligible": passed,
        "formal_auto_release": passed,
        "computed_pass": passed,
        "decision": "pass" if passed else "blocked",
        "reasons": reasons,
        "warnings": warnings,
        "runtime": runtime,
        "metrics": metric_gate,
        "cross_gap": cross_gap,
        "evidence_bound": postcheck.get("postcheck_result_path"),
        "held_out": False,
        "training_views_only": True,
        "delivery_quality": False,
        "human_visual_claim": False,
    }
