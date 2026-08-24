"""CPU adapter for one fresh ``convergence1000-v1`` executor attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .convergence_smoke import (
    CONVERGENCE_SMOKE_SCHEMA,
    ConvergenceSmokeBlocked,
    plan_convergence_smoke,
    validate_convergence_smoke_plan,
    write_json_once,
)
from .longsplat_input import _nested_backend_code_identity, future_workload_profile, validate_future_smoke_plan
from .pipeline_contract import PipelineBlocked, code_identity, sha256_file
from .smoke_executor import SmokeExecutorBlocked, _load_json, _output_path, validate_plan_and_static


PROFILE = "convergence1000-v1"
PLAN_SCHEMA = "longsplat-convergence-training-plan-v1"
TELEMETRY_POLICY = {
    "schema_version": "camera-sampling-telemetry-v1",
    "required_for_new_execution": True,
    "scope": "external_colmap_pose_only",
    "observer": "record selected camera after executor selection; no random calls or math mutation",
    "legacy_evidence_compatibility": "absent_allowed",
    "resume_semantics": "artifact inspection only; checkpoint is not bit-exact training resume",
}


class ConvergenceAdapterBlocked(PipelineBlocked):
    """The convergence adapter could not preserve the immutable input identity."""


def _fail(message: str) -> None:
    raise ConvergenceAdapterBlocked(message)


def _flag_index(argv: Sequence[str], flag: str) -> int:
    indexes = [index for index, value in enumerate(argv) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(argv):
        _fail(f"parent argv must contain exactly one {flag} with a value")
    return indexes[0]


def _read_camera_contract(path: Path) -> tuple[list[str], dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        _fail(f"camera contract is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"camera contract is invalid: {path}: {exc}")
    if not isinstance(value, Mapping):
        _fail("camera contract must be an object")
    names = value.get("frame_names")
    camera = value.get("camera")
    if not isinstance(names, list) or not names or len(names) != len(set(names)) or not isinstance(camera, Mapping):
        _fail("camera contract order/camera record is invalid")
    return [str(name) for name in names], dict(camera)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"derived plan must be fresh and absent: {path}")
    write_json_once(path, value)


def derive_convergence_training_plan(
    *,
    parent_plan_path: str | Path,
    static_contract_path: str | Path,
    output_plan_path: str | Path,
    convergence_plan_path: str | Path,
    model_path: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    pose_reference_path: str | Path | None = None,
    reason: str = "one authorized convergence1000-v1 intermediate diagnostic",
) -> dict[str, Any]:
    """Create a fresh training plan; no CPU preparation or checkpoint resume occurs."""

    route = Path(route_root).resolve()
    parent_path = _output_path(parent_plan_path, route, "convergence parent plan", must_exist=True, containment_root=containment_root)
    static_path = _output_path(static_contract_path, route, "convergence static contract", must_exist=True, containment_root=containment_root)
    output_path = _output_path(output_plan_path, route, "convergence derived plan", must_exist=False, containment_root=containment_root)
    cpu_plan_path = _output_path(convergence_plan_path, route, "convergence CPU plan", must_exist=True, containment_root=containment_root)
    model = _output_path(model_path, route, "convergence model path", must_exist=False, containment_root=containment_root)
    if output_path.exists() or output_path.is_symlink() or model.exists() or model.is_symlink():
        _fail("convergence derived plan/model must be fresh and absent")

    parent = _load_json(parent_path, "convergence parent plan")
    static = _load_json(static_path, "convergence static contract")
    source_value = parent.get("source_path")
    if not isinstance(source_value, str):
        _fail("parent plan source_path is missing")
    source = Path(source_value).resolve()
    if static_path != (source / "contract" / "static_contract.json").resolve():
        _fail("convergence static contract is not the immutable parent contract")
    immutable_parent_path = (source / "future_smoke_plan-v2.json").resolve()
    if not immutable_parent_path.is_file() or immutable_parent_path.is_symlink():
        immutable_parent_path = parent_path
    camera_contract_path = source / "camera_contract-v1.json"
    names, camera = _read_camera_contract(camera_contract_path)
    if static.get("image_count") != len(names):
        _fail("static image count differs from camera contract")
    try:
        validate_future_smoke_plan(
            parent,
            route_root=route,
            containment_root=containment_root,
            training_root=source,
            allow_producer_code_drift=True,
        )
    except Exception as exc:
        _fail(f"convergence parent plan validation failed: {type(exc).__name__}: {exc}")

    pose_reference = None if pose_reference_path is None else Path(pose_reference_path).resolve()
    if pose_reference is not None and (pose_reference.is_symlink() or not pose_reference.is_file()):
        _fail(f"pose reference is missing or symlinked: {pose_reference}")
    binding_plan = plan_convergence_smoke(
        active_camera_count=len(names),
        camera_names=names,
        source_video_sha256=static.get("source_video_sha256"),
        source_path=source,
        static_contract_path=static_path,
        static_contract_sha256=sha256_file(static_path),
        camera_contract_path=camera_contract_path,
        camera_contract_sha256=sha256_file(camera_contract_path),
        pose_reference_path=pose_reference,
        pose_reference_sha256=None if pose_reference is None else sha256_file(pose_reference),
        camera=camera,
    )
    validate_convergence_smoke_plan(binding_plan)
    _write_once(cpu_plan_path, binding_plan) if not cpu_plan_path.exists() else None
    # The CPU plan is normally written by the caller.  Re-read its exact bytes
    # so the executor binds the artifact supplied to this attempt.
    try:
        supplied_cpu_plan = json.loads(cpu_plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"convergence CPU plan is invalid: {cpu_plan_path}: {exc}")
    if not isinstance(supplied_cpu_plan, Mapping):
        _fail("convergence CPU plan must be an object")
    try:
        validate_convergence_smoke_plan(supplied_cpu_plan)
    except ConvergenceSmokeBlocked as exc:
        _fail(f"convergence CPU plan validation failed: {exc}")
    if supplied_cpu_plan.get("identity_binding", {}).get("camera_order") != names:
        _fail("convergence CPU plan camera order differs from immutable contract")

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
            "profile_semantics": "local experimental convergence diagnostic; fixed 1000 iterations; not official/formal/acceptance; no automatic higher-iteration search",
            "camera_sampling_telemetry": dict(TELEMETRY_POLICY),
            "external_route_semantics": "fork-specific external COLMAP fixed-pose RGB-only, depth disabled; not official unposed path",
        }
    )
    seed = frozen.get("seed")
    if isinstance(seed, Mapping):
        seed = dict(seed)
        seed["nested_backend_code_identity_sha256"] = nested_identity["identity_sha256"]
        frozen["seed"] = seed
    convergence_binding = {
        "schema_version": "longsplat-convergence-plan-binding-v1",
        "path": str(cpu_plan_path),
        "sha256": sha256_file(cpu_plan_path),
        "profile": PROFILE,
        "iterations": profile["iterations"],
        "render_iteration": profile["render_iteration"],
        "active_camera_count": len(names),
        "camera_contract_path": str(camera_contract_path),
        "camera_contract_sha256": sha256_file(camera_contract_path),
        "active_camera_order": names,
        "planned_anchor_adjust_iterations": list(
            supplied_cpu_plan["densification"]["planned_adjust_anchor_iterations"]
        ),
        "identity_binding": dict(supplied_cpu_plan["identity_binding"]),
    }
    plan: dict[str, Any] = dict(parent)
    plan.update(
        {
            "schema_version": "longsplat-future-smoke-plan-v2",
            "convergence_plan_schema": PLAN_SCHEMA,
            "workload_profile": PROFILE,
            "execution_status": "plan_only_derived_convergence",
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
            "convergence_plan": convergence_binding,
            "nested_backend_code_identity": nested_identity,
            "route_code_identity_sha256": current_code.get("code_identity_sha256"),
            "route_code_identity": dict(current_code),
            "train_script_sha256": sha256_file(train_script),
            "derived_retry": {
                "schema_version": "longsplat-convergence-plan-adapter-v1",
                "reason": reason,
                "parent_plan_path": str(immutable_parent_path),
                "parent_plan_sha256": sha256_file(immutable_parent_path),
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
    except (SmokeExecutorBlocked, ConvergenceSmokeBlocked, OSError, ValueError) as exc:
        _fail(f"derived convergence plan failed executor validation: {exc}")
    plan["derived_plan_path"] = str(output_path)
    plan["derived_plan_sha256"] = sha256_file(output_path)
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind convergence1000-v1 to one fresh LongSplat executor plan")
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--static-contract", required=True, type=Path)
    parser.add_argument("--convergence-plan", required=True, type=Path)
    parser.add_argument("--output-plan", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--pose-reference", type=Path)
    parser.add_argument("--route-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--containment-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = derive_convergence_training_plan(
            parent_plan_path=args.parent_plan,
            static_contract_path=args.static_contract,
            convergence_plan_path=args.convergence_plan,
            output_plan_path=args.output_plan,
            model_path=args.model_path,
            route_root=args.route_root,
            containment_root=args.containment_root,
            pose_reference_path=args.pose_reference,
        )
        output = Path(result["derived_plan_path"])
        print(
            json.dumps(
                {
                    "status": "passed",
                    "computed_pass": True,
                    "gpu_invoked": False,
                    "plan_path": str(output),
                    "plan_sha256": sha256_file(output),
                    "model_path": result["model_path"],
                    "workload_profile": result["workload_profile"],
                    "iterations": result["frozen_contract"]["iterations"],
                    "camera_contract_sha256": result["convergence_plan"]["camera_contract_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (ConvergenceAdapterBlocked, ConvergenceSmokeBlocked, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
