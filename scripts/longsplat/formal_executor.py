"""CPU authority adapter for one fresh local formal30000-v1 attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .convergence_smoke import expected_anchor_adjust_iterations, predicted_exposure, write_json_once
from .longsplat_input import _nested_backend_code_identity, future_workload_profile, validate_future_smoke_plan
from .pipeline_contract import PipelineBlocked, code_identity, sha256_file
from .smoke_executor import SmokeExecutorBlocked, _load_json, _output_path, validate_plan_and_static
from .gate_schema import GateSchemaBlocked, normalize_gate_evidence


PROFILE = "formal30000-v1"
ITERATIONS = 30000
PLAN_SCHEMA = "longsplat-formal-training-plan-v1"
POLICY_SCHEMA = "formal30000-policy-v1"
AUTOMATED_POLICY_SCHEMA = "formal30000-automated-policy-v1"
AUTOMATED_POLICY_MODE = "automated_technical_v1"
TELEMETRY_POLICY = {
    "schema_version": "camera-sampling-telemetry-v1",
    "required_for_new_execution": True,
    "scope": "external_colmap_pose_only",
    "observer": "record selected camera after executor selection; no random calls or math mutation",
    "legacy_evidence_compatibility": "absent_allowed",
    "resume_semantics": "artifact inspection only; checkpoint is not bit-exact training resume",
}


class FormalAdapterBlocked(PipelineBlocked):
    """The local formal plan could not be bound to the released input."""


def _fail(message: str) -> None:
    raise FormalAdapterBlocked(message)


def _flag_index(argv: Sequence[str], flag: str) -> int:
    indexes = [index for index, value in enumerate(argv) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(argv):
        _fail(f"formal parent argv must contain exactly one {flag} with a value")
    return indexes[0]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is invalid: {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"formal output must be fresh and absent: {path}")
    write_json_once(path, value)


def _camera_identity(source: Path, static_path: Path) -> tuple[list[str], dict[str, Any], Path]:
    camera_path = source / "camera_contract-v1.json"
    camera_contract = _read_json(camera_path, "formal camera contract")
    names = camera_contract.get("frame_names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        _fail("formal camera contract order must be unique and non-empty")
    static = _read_json(static_path, "formal static contract")
    if static.get("image_count") != len(names):
        _fail("formal camera count differs from static contract")
    camera = static.get("camera")
    if not isinstance(camera, Mapping):
        _fail("formal static camera record is missing")
    return [str(name) for name in names], dict(camera), camera_path


def plan_formal_training(
    *,
    source: Path,
    static_path: Path,
    release_path: Path,
    camera_names: Sequence[str],
    camera: Mapping[str, Any],
    release_sha256: str,
) -> dict[str, Any]:
    release = _read_json(release_path, "supervisor formal release decision")
    decision = release.get("decision")
    profile = release.get("formal_profile")
    if not isinstance(decision, Mapping) or decision.get("EARLY_VISUAL_RECOGNIZABLE") is not True:
        _fail("supervisor release must explicitly set EARLY_VISUAL_RECOGNIZABLE=true")
    if decision.get("ROUGH_VISUAL_FOR_FORMAL_GATE") not in {"pass", "needs_review"}:
        _fail("supervisor release rough visual formal gate must be pass or needs_review")
    if decision.get("DELIVERY_QUALITY") is not False or decision.get("held_out") is not False:
        _fail("supervisor release must keep delivery quality false and held_out false")
    if not isinstance(profile, Mapping) or profile.get("profile_id") != PROFILE or profile.get("iterations") != ITERATIONS:
        _fail("supervisor release formal profile is not formal30000-v1")
    prediction = predicted_exposure(len(camera_names), ITERATIONS)
    return {
        "schema_version": POLICY_SCHEMA,
        "profile": PROFILE,
        "profile_class": "local_versioned_formal_policy",
        "computed_pass": True,
        "plan_status": "plan_validated",
        "gpu_invoked": False,
        "training_invoked": False,
        "render_invoked": False,
        "conversion_invoked": False,
        "formal_auto_release": False,
        "delivery_quality": False,
        "held_out": False,
        "requested_iterations": ITERATIONS,
        "render_iteration": ITERATIONS,
        "identity_binding": {
            "source_path": str(source.resolve()),
            "source_video_sha256": _read_json(static_path, "formal static contract").get("source_video_sha256"),
            "static_contract_path": str(static_path.resolve()),
            "static_contract_sha256": sha256_file(static_path),
            "camera_contract_path": str((source / "camera_contract-v1.json").resolve()),
            "camera_contract_sha256": sha256_file(source / "camera_contract-v1.json"),
            "camera_count": len(camera_names),
            "camera_order": list(camera_names),
            "camera": dict(camera),
            "external_colmap_pose": True,
            "training_mode": "fixed_pose_rgb_only",
            "depth_source": "disabled",
            "pose_contract_semantics": "same immutable external fixed-pose contract; runtime output must re-prove exact residual/order",
        },
        "supervisor_release": {
            "path": str(release_path.resolve()),
            "sha256": release_sha256,
            "decision_source": release.get("decision_source"),
            "EARLY_VISUAL_RECOGNIZABLE": True,
            "ROUGH_VISUAL_FOR_FORMAL_GATE": decision.get("ROUGH_VISUAL_FOR_FORMAL_GATE"),
            "DELIVERY_QUALITY": False,
        },
        "sampling": {
            "confirmed_semantics": "one camera per iteration; phase-local random pop without replacement; stack refill",
            "exposure_rule": "floor(T/N) or ceil(T/N) per camera",
            "prediction": prediction,
            "telemetry_required": True,
            "expected_exposure_distribution": {
                "96_cameras": prediction["min_exposure_count"],
                "48_cameras": prediction["max_exposure_count"],
            },
        },
        "densification": {
            "fork_semantics": "anchor_growing via adjust_anchor/anchor_growing; not classic densify_and_clone/split/reset_opacity",
            "planned_adjust_anchor_iterations": expected_anchor_adjust_iterations(ITERATIONS),
            "update_interval": 100,
            "update_until": ITERATIONS,
            "require_purning": False,
            "runtime_counts_required": True,
        },
        "policy_boundary": {
            "official_code_baseline": "30000",
            "official_final_schedule_claimed": False,
            "official_post_iter_20000_or_native_around_50000_run": False,
            "adaptive_formula_claimed": False,
            "local_policy": "formal30000-v1; one authorized fresh formal attempt",
            "conversion": "forbidden in this task",
        },
        "training_state_resume": {
            "bit_exact_resume": False,
            "fresh_model_required": True,
            "meaning": "checkpoint supports inspection/render/conversion but cannot restore training exactly",
        },
    }


def plan_automated_formal_training(
    *,
    source: Path,
    static_path: Path,
    gate_path: Path,
    gate_sha256: str,
    camera_names: Sequence[str],
    camera: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the default formal policy from the automated early gate.

    The default one-click chain must not manufacture a human/supervisor visual
    decision.  This policy is therefore a separate schema and binds the
    already-passed automated early gate as its release authority.
    """

    gate = _read_json(gate_path, "automated early gate")
    try:
        gate = normalize_gate_evidence(gate)
    except GateSchemaBlocked as exc:
        _fail(f"automated early gate normalization failed: {exc}")
    if gate.get("automated_technical_gate") is not True or gate.get("computed_pass") is not True:
        _fail("automated early gate is not a computed technical pass")
    if gate.get("formal_release_eligible") is not True:
        _fail("automated early gate is not formal-release eligible")
    if gate.get("manual_visual_review") is not False:
        _fail("automated formal policy cannot bind manual visual review")
    if gate.get("held_out") is not False:
        _fail("automated formal policy cannot bind held-out evidence")
    prediction = predicted_exposure(len(camera_names), ITERATIONS)
    return {
        "schema_version": AUTOMATED_POLICY_SCHEMA,
        "profile": PROFILE,
        "policy_mode": AUTOMATED_POLICY_MODE,
        "profile_class": "local_versioned_formal_policy",
        "computed_pass": True,
        "plan_status": "plan_validated",
        "gpu_invoked": False,
        "training_invoked": False,
        "render_invoked": False,
        "conversion_invoked": False,
        "formal_auto_release": True,
        "automated_technical_gate": True,
        "formal_release_eligible": True,
        "manual_visual_review": False,
        "human_visual_claim": False,
        "delivery_quality": False,
        "held_out": False,
        "requested_iterations": ITERATIONS,
        "render_iteration": ITERATIONS,
        "identity_binding": {
            "source_path": str(source.resolve()),
            "source_video_sha256": _read_json(static_path, "formal static contract").get("source_video_sha256"),
            "static_contract_path": str(static_path.resolve()),
            "static_contract_sha256": sha256_file(static_path),
            "camera_contract_path": str((source / "camera_contract-v1.json").resolve()),
            "camera_contract_sha256": sha256_file(source / "camera_contract-v1.json"),
            "camera_count": len(camera_names),
            "camera_order": list(camera_names),
            "camera": dict(camera),
            "external_colmap_pose": True,
            "training_mode": "fixed_pose_rgb_only",
            "depth_source": "disabled",
            "pose_contract_semantics": "same immutable external fixed-pose contract; runtime output must re-prove exact residual/order",
        },
        "automated_release": {
            "kind": "automated_technical_gate",
            "path": str(gate_path.resolve()),
            "sha256": gate_sha256,
            "policy_id": "automated-technical-v1",
            "computed_pass": True,
            "formal_release_eligible": True,
            "manual_visual_review": False,
            "held_out": False,
        },
        "sampling": {
            "confirmed_semantics": "one camera per iteration; phase-local random pop without replacement; stack refill",
            "exposure_rule": "floor(T/N) or ceil(T/N) per camera",
            "prediction": prediction,
            "telemetry_required": True,
            "expected_exposure_distribution": {
                "camera_count_at_min_exposure": prediction["camera_count_at_min_exposure"],
                "camera_count_at_max_exposure": prediction["camera_count_at_max_exposure"],
            },
        },
        "densification": {
            "fork_semantics": "anchor_growing via adjust_anchor/anchor_growing; not classic densify_and_clone/split/reset_opacity",
            "planned_adjust_anchor_iterations": expected_anchor_adjust_iterations(ITERATIONS),
            "update_interval": 100,
            "update_until": ITERATIONS,
            "require_purning": False,
            "runtime_counts_required": True,
        },
        "policy_boundary": {
            "official_code_baseline": "30000",
            "official_final_schedule_claimed": False,
            "official_post_iter_20000_or_native_around_50000_run": False,
            "adaptive_formula_claimed": False,
            "local_policy": "formal30000-v1; automated technical early-gate release; no human visual claim",
            "conversion": "not part of the training policy",
        },
        "training_state_resume": {
            "bit_exact_resume": False,
            "fresh_model_required": True,
            "meaning": "checkpoint supports inspection/render/conversion but cannot restore training exactly",
        },
    }


def validate_formal_training_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    policy_mode = plan.get("policy_mode", "supervisor_manual")
    expected_schema = AUTOMATED_POLICY_SCHEMA if policy_mode == AUTOMATED_POLICY_MODE else POLICY_SCHEMA
    if plan.get("schema_version") != expected_schema or plan.get("profile") != PROFILE:
        _fail("formal CPU policy schema/profile mismatch")
    if plan.get("requested_iterations") != ITERATIONS or plan.get("render_iteration") != ITERATIONS:
        _fail("formal CPU policy must be exactly 30000 iterations")
    if plan.get("computed_pass") is not True or plan.get("gpu_invoked") is not False:
        _fail("formal CPU policy boundary is invalid")
    if policy_mode == AUTOMATED_POLICY_MODE:
        if (
            plan.get("formal_auto_release") is not True
            or plan.get("automated_technical_gate") is not True
            or plan.get("formal_release_eligible") is not True
            or plan.get("manual_visual_review") is not False
            or plan.get("human_visual_claim") is not False
        ):
            _fail("automated formal policy boundary is invalid")
    elif plan.get("formal_auto_release") is not False:
        _fail("supervisor formal policy cannot auto-release")
    if plan.get("conversion_invoked") is not False or plan.get("delivery_quality") is not False or plan.get("held_out") is not False:
        _fail("formal CPU policy cannot reach conversion/delivery or claim held-out")
    identity = plan.get("identity_binding")
    if not isinstance(identity, Mapping) or identity.get("depth_source") != "disabled" or identity.get("external_colmap_pose") is not True:
        _fail("formal CPU identity binding must be fixed-pose and depth-disabled")
    if policy_mode == AUTOMATED_POLICY_MODE:
        release = plan.get("automated_release")
        if (
            not isinstance(release, Mapping)
            or release.get("kind") != "automated_technical_gate"
            or release.get("computed_pass") is not True
            or release.get("formal_release_eligible") is not True
            or release.get("manual_visual_review") is not False
            or release.get("held_out") is not False
        ):
            _fail("automated formal policy must bind the automated early gate")
    else:
        release = plan.get("supervisor_release")
        if not isinstance(release, Mapping) or release.get("EARLY_VISUAL_RECOGNIZABLE") is not True or release.get("DELIVERY_QUALITY") is not False:
            _fail("formal CPU policy must bind the supervisor release decision")
    prediction = predicted_exposure(int(identity["camera_count"]), ITERATIONS)
    sampling = plan.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("prediction") != prediction:
        _fail("formal exposure prediction is inconsistent")
    expected = expected_anchor_adjust_iterations(ITERATIONS)
    if plan.get("densification", {}).get("planned_adjust_anchor_iterations") != expected:
        _fail("formal anchor schedule is inconsistent with the fork")
    return {
        "schema_version": "formal30000-policy-validation-v1",
        "computed_pass": True,
        "profile": PROFILE,
        "iterations": ITERATIONS,
        "camera_count": identity["camera_count"],
        "prediction": prediction,
        "planned_anchor_event_count": len(expected),
        "formal_auto_release": policy_mode == AUTOMATED_POLICY_MODE,
        "conversion_invoked": False,
    }


def derive_automated_formal_training_plan(
    *,
    parent_plan_path: str | Path,
    static_contract_path: str | Path,
    automated_gate_path: str | Path,
    automated_gate_sha256: str,
    formal_policy_path: str | Path,
    output_plan_path: str | Path,
    model_path: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    allow_existing_model: bool = False,
    reason: str = "automated technical early gate released one local formal30000-v1 attempt",
) -> dict[str, Any]:
    """Derive a formal plan for the default automated chain.

    This is a consumer adapter over the immutable LongSplat input plan.  It
    adds the formal policy and explicit anchor schedule binding without
    changing source images, camera poses, or training argv.  Existing model
    directories are accepted only for an explicit CPU evidence-recovery path;
    normal fresh execution still requires an absent model.
    """

    route = Path(route_root).resolve()
    parent_path = _output_path(parent_plan_path, route, "formal parent plan", must_exist=True, containment_root=containment_root)
    static_path = _output_path(static_contract_path, route, "formal static contract", must_exist=True, containment_root=containment_root)
    gate_path = _output_path(automated_gate_path, route, "automated early gate", must_exist=True, containment_root=containment_root)
    policy_path = _output_path(formal_policy_path, route, "automated formal policy", must_exist=False, containment_root=containment_root)
    output_path = _output_path(output_plan_path, route, "automated formal plan", must_exist=False, containment_root=containment_root)
    model = _output_path(model_path, route, "formal model path", must_exist=False, containment_root=containment_root)
    if output_path.exists() or output_path.is_symlink() or policy_path.exists() or policy_path.is_symlink():
        _fail("automated formal plan/policy must be fresh and absent")
    if not allow_existing_model and (model.exists() or model.is_symlink()):
        _fail(f"fresh formal model_path is required and must be absent: {model}")
    if allow_existing_model and (not model.is_dir() or model.is_symlink()):
        _fail(f"preserved formal model_path is missing or symlinked: {model}")
    if not isinstance(automated_gate_sha256, str) or sha256_file(gate_path) != automated_gate_sha256:
        _fail("automated early gate SHA differs from its binding")

    parent = _load_json(parent_path, "formal parent plan")
    static = _load_json(static_path, "formal static contract")
    source_value = parent.get("source_path")
    if not isinstance(source_value, str):
        _fail("formal parent source_path is missing")
    source = Path(source_value).resolve()
    if static_path != (source / "contract" / "static_contract.json").resolve():
        _fail("formal static contract is not the immutable source contract")
    names, camera, camera_contract_path = _camera_identity(source, static_path)
    policy = plan_automated_formal_training(
        source=source,
        static_path=static_path,
        gate_path=gate_path,
        gate_sha256=automated_gate_sha256,
        camera_names=names,
        camera=camera,
    )
    _write_once(policy_path, policy)
    policy = _read_json(policy_path, "automated formal policy")
    validate_formal_training_plan(policy)
    try:
        validate_future_smoke_plan(
            parent,
            route_root=route,
            containment_root=containment_root,
            training_root=source,
            allow_producer_code_drift=True,
        )
    except Exception as exc:
        _fail(f"formal parent plan validation failed: {type(exc).__name__}: {exc}")

    profile = future_workload_profile(PROFILE)
    argv = list(parent.get("argv", []))
    iteration_index = _flag_index(argv, "--iterations")
    model_index = _flag_index(argv, "--model_path")
    argv[iteration_index + 1] = str(profile["iterations"])
    argv[model_index + 1] = str(model)
    current_code = code_identity(route)
    train_script = (route / "third_party" / "LongSplat" / "train.py").resolve()
    nested_identity = _nested_backend_code_identity(route, train_script)
    frozen = dict(parent.get("frozen_contract", {}))
    frozen.update(
        {
            "workload_profile": PROFILE,
            "iterations": profile["iterations"],
            "render_iteration": profile["render_iteration"],
            "profile_semantics": "local formal30000-v1 policy; automated technical early-gate release; not official complete final schedule",
            "camera_sampling_telemetry": dict(TELEMETRY_POLICY),
            "external_route_semantics": "fork-specific external COLMAP fixed-pose RGB-only, depth disabled; not official unposed path",
            "automated_release": {"path": str(gate_path), "sha256": automated_gate_sha256},
            "delivery_quality": False,
            "held_out": False,
            "conversion": False,
        }
    )
    seed = frozen.get("seed")
    if isinstance(seed, Mapping):
        seed = dict(seed)
        seed["nested_backend_code_identity_sha256"] = nested_identity["identity_sha256"]
        frozen["seed"] = seed
    formal_binding = {
        "schema_version": "longsplat-formal-plan-binding-v2",
        "path": str(policy_path),
        "sha256": sha256_file(policy_path),
        "profile": PROFILE,
        "iterations": ITERATIONS,
        "render_iteration": ITERATIONS,
        "active_camera_count": len(names),
        "active_camera_order": names,
        "camera_contract_path": str(camera_contract_path),
        "camera_contract_sha256": sha256_file(camera_contract_path),
        "planned_anchor_adjust_iterations": list(policy["densification"]["planned_adjust_anchor_iterations"]),
        "release_kind": "automated_technical_gate",
        "automated_gate_path": str(gate_path),
        "automated_gate_sha256": automated_gate_sha256,
        "identity_binding": dict(policy["identity_binding"]),
    }
    retry = {
        "schema_version": "longsplat-formal-plan-adapter-v2",
        "reason": reason,
        "parent_plan_path": str(parent_path),
        "parent_plan_sha256": sha256_file(parent_path),
        "parent_static_contract_path": str(static_path),
        "producer_code_identity_sha256": current_code.get("code_identity_sha256"),
        "fresh_model_required": not allow_existing_model,
        "checkpoint_resume": "artifact inspection only; not bit-exact training resume",
        "recovery_mode": "existing_gpu_artifact_revalidation" if allow_existing_model else "fresh_gpu_execution",
    }
    plan: dict[str, Any] = dict(parent)
    plan.update(
        {
            "schema_version": "longsplat-future-smoke-plan-v2",
            "formal_plan_schema": PLAN_SCHEMA,
            "workload_profile": PROFILE,
            "execution_status": "plan_only_derived_automated_formal",
            "accepted": False,
            "delivery_reachable": False,
            "gpu_invoked": False,
            "training_invoked": False,
            "render_invoked": False,
            "conversion_invoked": False,
            "source_path": str(source),
            "model_path": str(model),
            "model_path_exists_at_plan_time": False,
            "argv": argv,
            "frozen_contract": frozen,
            "formal_plan": formal_binding,
            "nested_backend_code_identity": nested_identity,
            "route_code_identity_sha256": current_code.get("code_identity_sha256"),
            "route_code_identity": dict(current_code),
            "train_script_sha256": sha256_file(train_script),
            "derived_retry": retry,
        }
    )
    _write_once(output_path, plan)
    try:
        validate_plan_and_static(
            output_path,
            static_path,
            route_root=route,
            containment_root=containment_root,
            allow_existing_model=allow_existing_model,
            allow_producer_code_drift=True,
        )
    except (SmokeExecutorBlocked, FormalAdapterBlocked, OSError, ValueError) as exc:
        _fail(f"derived automated formal plan failed executor validation: {exc}")
    plan["derived_plan_path"] = str(output_path)
    plan["derived_plan_sha256"] = sha256_file(output_path)
    return plan


def derive_formal_training_plan(
    *,
    parent_plan_path: str | Path,
    static_contract_path: str | Path,
    release_decision_path: str | Path,
    formal_policy_path: str | Path,
    output_plan_path: str | Path,
    model_path: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    reason: str = "supervisor released one local formal30000-v1 training attempt",
) -> dict[str, Any]:
    route = Path(route_root).resolve()
    parent_path = _output_path(parent_plan_path, route, "formal parent plan", must_exist=True, containment_root=containment_root)
    static_path = _output_path(static_contract_path, route, "formal static contract", must_exist=True, containment_root=containment_root)
    release_path = _output_path(release_decision_path, route, "formal supervisor decision", must_exist=True, containment_root=containment_root)
    policy_path = _output_path(formal_policy_path, route, "formal policy plan", must_exist=False, containment_root=containment_root)
    output_path = _output_path(output_plan_path, route, "formal derived plan", must_exist=False, containment_root=containment_root)
    model = _output_path(model_path, route, "formal model path", must_exist=False, containment_root=containment_root)
    if output_path.exists() or output_path.is_symlink() or model.exists() or model.is_symlink():
        _fail("formal derived plan/model must be fresh and absent")
    parent = _load_json(parent_path, "formal parent plan")
    static = _load_json(static_path, "formal static contract")
    source_value = parent.get("source_path")
    if not isinstance(source_value, str):
        _fail("formal parent source_path is missing")
    source = Path(source_value).resolve()
    if static_path != (source / "contract" / "static_contract.json").resolve():
        _fail("formal static contract is not the immutable source contract")
    names, camera, camera_contract_path = _camera_identity(source, static_path)
    release = _read_json(release_path, "formal supervisor decision")
    release_sha = sha256_file(release_path)
    policy = plan_formal_training(
        source=source,
        static_path=static_path,
        release_path=release_path,
        camera_names=names,
        camera=camera,
        release_sha256=release_sha,
    )
    if policy_path.exists():
        supplied_policy = _read_json(policy_path, "formal policy plan")
    else:
        _write_once(policy_path, policy)
        supplied_policy = policy
    validate_formal_training_plan(supplied_policy)
    if supplied_policy.get("identity_binding", {}).get("camera_order") != names:
        _fail("formal policy camera order differs from immutable camera contract")
    if supplied_policy.get("supervisor_release", {}).get("sha256") != release_sha:
        _fail("formal policy supervisor decision SHA differs")
    try:
        validate_future_smoke_plan(
            parent,
            route_root=route,
            containment_root=containment_root,
            training_root=source,
            allow_producer_code_drift=True,
        )
    except Exception as exc:
        _fail(f"formal parent plan validation failed: {type(exc).__name__}: {exc}")
    profile = future_workload_profile(PROFILE)
    argv = list(parent.get("argv", []))
    iteration_index = _flag_index(argv, "--iterations")
    model_index = _flag_index(argv, "--model_path")
    argv[iteration_index + 1] = str(profile["iterations"])
    argv[model_index + 1] = str(model)
    current_code = code_identity(route)
    train_script = (route / "third_party" / "LongSplat" / "train.py").resolve()
    nested_identity = _nested_backend_code_identity(route, train_script)
    frozen = dict(parent.get("frozen_contract", {}))
    frozen.update(
        {
            "workload_profile": PROFILE,
            "iterations": profile["iterations"],
            "render_iteration": profile["render_iteration"],
            "profile_semantics": "local formal30000-v1 policy; official code baseline 30000 but not official complete final schedule; no adaptive formula",
            "camera_sampling_telemetry": dict(TELEMETRY_POLICY),
            "external_route_semantics": "fork-specific external COLMAP fixed-pose RGB-only, depth disabled; not official unposed path",
            "supervisor_release_decision": {"path": str(release_path), "sha256": release_sha},
            "delivery_quality": False,
            "held_out": False,
            "conversion": False,
        }
    )
    seed = frozen.get("seed")
    if isinstance(seed, Mapping):
        seed = dict(seed)
        seed["nested_backend_code_identity_sha256"] = nested_identity["identity_sha256"]
        frozen["seed"] = seed
    formal_binding = {
        "schema_version": "longsplat-formal-plan-binding-v1",
        "path": str(policy_path),
        "sha256": sha256_file(policy_path),
        "profile": PROFILE,
        "iterations": ITERATIONS,
        "render_iteration": ITERATIONS,
        "active_camera_count": len(names),
        "active_camera_order": names,
        "camera_contract_path": str(camera_contract_path),
        "camera_contract_sha256": sha256_file(camera_contract_path),
        "planned_anchor_adjust_iterations": list(supplied_policy["densification"]["planned_adjust_anchor_iterations"]),
        "supervisor_release_path": str(release_path),
        "supervisor_release_sha256": release_sha,
        "identity_binding": dict(supplied_policy["identity_binding"]),
    }
    immutable_parent = (source / "future_smoke_plan-v2.json").resolve()
    if not immutable_parent.is_file() or immutable_parent.is_symlink():
        immutable_parent = parent_path
    plan: dict[str, Any] = dict(parent)
    plan.update(
        {
            "schema_version": "longsplat-future-smoke-plan-v2",
            "formal_plan_schema": PLAN_SCHEMA,
            "workload_profile": PROFILE,
            "execution_status": "plan_only_derived_formal",
            "accepted": False,
            "delivery_reachable": False,
            "gpu_invoked": False,
            "training_invoked": False,
            "render_invoked": False,
            "conversion_invoked": False,
            "source_path": str(source),
            "model_path": str(model),
            "model_path_exists_at_plan_time": False,
            "argv": argv,
            "frozen_contract": frozen,
            "formal_plan": formal_binding,
            "nested_backend_code_identity": nested_identity,
            "route_code_identity_sha256": current_code.get("code_identity_sha256"),
            "route_code_identity": dict(current_code),
            "train_script_sha256": sha256_file(train_script),
            "derived_retry": {
                "schema_version": "longsplat-formal-plan-adapter-v1",
                "reason": reason,
                "parent_plan_path": str(immutable_parent),
                "parent_plan_sha256": sha256_file(immutable_parent),
                "parent_static_contract_path": str(static_path),
                "producer_code_identity_sha256": current_code.get("code_identity_sha256"),
                "fresh_model_required": True,
                "checkpoint_resume": "forbidden_not_bit_exact",
            },
        }
    )
    _write_once(output_path, plan)
    try:
        validate_plan_and_static(
            output_path,
            static_path,
            route_root=route,
            containment_root=containment_root,
            verify_static=True,
            allow_producer_code_drift=True,
        )
    except (SmokeExecutorBlocked, FormalAdapterBlocked, OSError, ValueError) as exc:
        _fail(f"derived formal plan failed executor validation: {exc}")
    plan["derived_plan_path"] = str(output_path)
    plan["derived_plan_sha256"] = sha256_file(output_path)
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind formal30000-v1 to one fresh LongSplat executor plan")
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--static-contract", required=True, type=Path)
    parser.add_argument("--release-decision", required=True, type=Path)
    parser.add_argument("--formal-policy", required=True, type=Path)
    parser.add_argument("--output-plan", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--route-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--containment-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = derive_formal_training_plan(
            parent_plan_path=args.parent_plan,
            static_contract_path=args.static_contract,
            release_decision_path=args.release_decision,
            formal_policy_path=args.formal_policy,
            output_plan_path=args.output_plan,
            model_path=args.model_path,
            route_root=args.route_root,
            containment_root=args.containment_root,
        )
        output = Path(result["derived_plan_path"])
        print(json.dumps({
            "status": "passed",
            "computed_pass": True,
            "gpu_invoked": False,
            "plan_path": str(output),
            "plan_sha256": sha256_file(output),
            "model_path": result["model_path"],
            "workload_profile": result["workload_profile"],
            "iterations": result["frozen_contract"]["iterations"],
            "camera_contract_sha256": result["formal_plan"]["camera_contract_sha256"],
            "supervisor_release_sha256": result["formal_plan"]["supervisor_release_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    except (FormalAdapterBlocked, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
