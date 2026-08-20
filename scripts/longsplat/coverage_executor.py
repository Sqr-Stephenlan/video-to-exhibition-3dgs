"""CPU adapter that binds one audited coverage-smoke plan to the GPU executor.

This adapter does not run a child process.  It derives a fresh append-only
coverage training plan from the immutable smoke input plan, preserving the
source/static/backend/camera contract and changing only the versioned
coverage profile, iteration, model/evidence paths, and telemetry requirement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .coverage_smoke import (
    MIN_COMPLETE_COVERAGE_ROUNDS,
    CoverageSmokeBlocked,
    load_frame_selection,
    validate_coverage_smoke_plan,
)
from .longsplat_input import (
    _nested_backend_code_identity,
    future_workload_profile,
    validate_future_smoke_plan,
)
from .pipeline_contract import PipelineBlocked, code_identity, sha256_file
from .smoke_executor import (
    SmokeExecutorBlocked,
    _load_json,
    _output_path,
    validate_plan_and_static,
)


PROFILE = "coverage-smoke-v1"
PLAN_SCHEMA = "longsplat-coverage-training-plan-v1"
TELEMETRY_POLICY = {
    "schema_version": "camera-sampling-telemetry-v1",
    "required_for_new_execution": True,
    "scope": "external_colmap_pose_only",
    "legacy_evidence_compatibility": "absent_allowed",
    "resume_semantics": "artifact inspection only; checkpoint is not bit-exact training resume",
}


class CoverageAdapterBlocked(PipelineBlocked):
    """Coverage plan could not be identity-bound to the immutable input."""


def _fail(message: str) -> None:
    raise CoverageAdapterBlocked(message)


def _flag_index(argv: Sequence[str], flag: str) -> int:
    indexes = [index for index, value in enumerate(argv) if value == flag]
    if len(indexes) != 1 or indexes[0] + 1 >= len(argv):
        _fail(f"coverage parent argv must contain exactly one {flag} with a value")
    return indexes[0]


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"coverage derived plan must be absent: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _coverage_binding(
    *,
    coverage_path: Path,
    coverage_plan: Mapping[str, Any],
    static: Mapping[str, Any],
    source: Path,
) -> dict[str, Any]:
    camera_contract_path = source / "camera_contract-v1.json"
    if camera_contract_path.is_symlink() or not camera_contract_path.is_file():
        _fail(f"immutable camera contract is missing: {camera_contract_path}")
    try:
        camera_contract = json.loads(camera_contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"immutable camera contract is invalid: {camera_contract_path}: {exc}")
    if not isinstance(camera_contract, Mapping):
        _fail("immutable camera contract must be an object")
    ordered_names = camera_contract.get("frame_names")
    if not isinstance(ordered_names, list) or coverage_plan.get("active_camera_order") != ordered_names:
        _fail("coverage camera order differs from the immutable camera contract")
    camera_sha = sha256_file(camera_contract_path)
    if coverage_plan.get("camera_contract_sha256") != camera_sha:
        _fail("coverage plan camera contract SHA differs")
    parent_binding = static.get("parent_binding")
    static_file_sha = parent_binding.get("camera_contract_file_sha256") if isinstance(parent_binding, Mapping) else None
    if static_file_sha != camera_sha:
        _fail("static camera contract SHA differs from coverage plan")
    stable_sha = coverage_plan.get("camera_contract_stable_sha256")
    if stable_sha is not None and static.get("camera_contract_sha256") != stable_sha:
        _fail("static camera contract stable SHA differs from coverage plan")
    if static.get("image_count") != coverage_plan.get("active_camera_count"):
        _fail("coverage camera count differs from immutable static input")
    source_sha = static.get("source_video_sha256")
    if coverage_plan.get("source_video_sha256") not in {None, source_sha}:
        _fail("coverage source video SHA differs from immutable static input")
    return {
        "schema_version": "longsplat-coverage-plan-binding-v1",
        "path": str(coverage_path),
        "sha256": sha256_file(coverage_path),
        "active_camera_count": coverage_plan.get("active_camera_count"),
        "camera_contract_path": str(camera_contract_path),
        "camera_contract_sha256": camera_sha,
        "camera_order_sha256": sha256_file(camera_contract_path),
        "iterations": coverage_plan.get("requested_iterations"),
        "planned_anchor_adjust_iterations": list(
            coverage_plan.get("densification", {}).get("adjust_anchor_iterations", [])
        )
        if isinstance(coverage_plan.get("densification"), Mapping)
        else [],
        "source_video_sha256": source_sha,
    }


def _coverage_training_policy(
    *,
    coverage_plan: Mapping[str, Any],
    route: Path,
    containment_root: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute the dynamic workload boundary before deriving a GPU plan."""

    count = coverage_plan.get("active_camera_count")
    iterations = coverage_plan.get("requested_iterations")
    selector_cap = coverage_plan.get("frame_selector_max_frames")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        _fail("coverage plan active camera count is invalid")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        _fail("coverage plan requested iterations are invalid")
    if isinstance(selector_cap, bool) or not isinstance(selector_cap, int) or selector_cap <= 0:
        _fail("coverage plan selector cap is missing or invalid")
    selection_value = coverage_plan.get("frame_selection_path")
    selection_sha = coverage_plan.get("frame_selection_sha256")
    if not isinstance(selection_value, str) or not isinstance(selection_sha, str):
        _fail("coverage plan frame-selection binding is incomplete")
    selection_path = _output_path(selection_value, route, "coverage frame selection", must_exist=True, containment_root=containment_root)
    if selection_path.is_symlink() or sha256_file(selection_path) != selection_sha:
        _fail("coverage frame-selection SHA binding failed")
    selection = load_frame_selection(selection_path)
    if selection["max_frames"] != selector_cap:
        _fail("coverage plan selector cap differs from immutable frame selection")
    minimum = MIN_COMPLETE_COVERAGE_ROUNDS * count
    upper = MIN_COMPLETE_COVERAGE_ROUNDS * selector_cap
    formula = coverage_plan.get("formula")
    if not isinstance(formula, Mapping):
        _fail("coverage plan formula is missing")
    if formula.get("local_policy_minimum_iterations") != minimum:
        _fail("coverage plan local minimum is not derived from active camera count")
    if formula.get("selector_derived_upper_bound") != upper:
        _fail("coverage plan upper bound is not derived from immutable selector cap")
    if count > selector_cap or iterations < minimum or iterations > upper:
        _fail("coverage plan requested iterations are outside the versioned dynamic policy")
    stop = coverage_plan.get("stop")
    if not isinstance(stop, Mapping) or stop.get("reasons") != [] or coverage_plan.get("computed_pass") is not True:
        _fail("coverage plan is not a passed dynamic policy result")
    estimate = coverage_plan.get("gpu_workload_estimate")
    if not isinstance(estimate, Mapping) or estimate.get("iterations") != iterations:
        _fail("coverage plan workload estimate does not bind requested iterations")
    return {
        "iterations": iterations,
        "render_iteration": iterations,
        "active_camera_count": count,
        "frame_selector_max_frames": selector_cap,
        "minimum_iterations": minimum,
        "selector_derived_upper_bound": upper,
        "policy": "coverage-smoke-v1-dynamic-plan-bound",
    }


def derive_coverage_training_plan(
    *,
    parent_plan_path: str | Path,
    static_contract_path: str | Path,
    coverage_plan_path: str | Path,
    output_plan_path: str | Path,
    model_path: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    reason: str = "coverage-smoke-v1 full camera coverage adapter",
) -> dict[str, Any]:
    """Create one fresh plan bound to the immutable dynamic coverage workload."""

    route = Path(route_root).resolve()
    parent_path = _output_path(parent_plan_path, route, "coverage parent plan", must_exist=True, containment_root=containment_root)
    static_path = _output_path(static_contract_path, route, "coverage static contract", must_exist=True, containment_root=containment_root)
    coverage_path = _output_path(coverage_plan_path, route, "coverage CPU plan", must_exist=True, containment_root=containment_root)
    output_path = _output_path(output_plan_path, route, "coverage derived plan", must_exist=False, containment_root=containment_root)
    model = _output_path(model_path, route, "coverage model path", must_exist=False, containment_root=containment_root)
    if output_path.exists() or output_path.is_symlink() or model.exists() or model.is_symlink():
        _fail("coverage derived plan/model must be fresh and absent")

    parent = _load_json(parent_path, "coverage parent plan")
    static = _load_json(static_path, "coverage static contract")
    coverage = _load_json(coverage_path, "coverage CPU plan")
    try:
        coverage_validation = validate_coverage_smoke_plan(coverage)
    except (CoverageSmokeBlocked, ValueError) as exc:
        _fail(f"coverage CPU plan validation failed: {exc}")
    if coverage_validation.get("computed_pass") is not True:
        _fail("coverage CPU plan is not computed_pass")
    profile = future_workload_profile(PROFILE)
    workload = _coverage_training_policy(coverage_plan=coverage, route=route, containment_root=containment_root)
    source_value = parent.get("source_path")
    if not isinstance(source_value, str):
        _fail("coverage parent source_path is missing")
    source = Path(source_value).resolve()
    expected_static = (source / "contract" / "static_contract.json").resolve()
    if static_path != expected_static:
        _fail("coverage static contract is not the immutable parent contract")
    try:
        validate_future_smoke_plan(
            parent,
            route_root=route,
            containment_root=containment_root,
            training_root=source,
            allow_producer_code_drift=True,
        )
    except Exception as exc:
        _fail(f"coverage parent plan validation failed: {type(exc).__name__}: {exc}")
    binding = _coverage_binding(
        coverage_path=coverage_path,
        coverage_plan=coverage,
        static=static,
        source=source,
    )

    argv = list(parent.get("argv", []))
    iteration_index = _flag_index(argv, "--iterations")
    model_index = _flag_index(argv, "--model_path")
    argv[iteration_index + 1] = str(workload["iterations"])
    argv[model_index + 1] = str(model)
    current_code = code_identity(route)
    train_script = (route / "third_party" / "LongSplat" / "train.py").resolve()
    nested_identity = _nested_backend_code_identity(route, train_script)
    frozen = dict(parent.get("frozen_contract", {}))
    frozen.update(
        {
            "workload_profile": PROFILE,
            "iterations": workload["iterations"],
            "render_iteration": workload["render_iteration"],
            "profile_semantics": "local evidence-driven coverage smoke; not formal training or acceptance",
            "camera_sampling_telemetry": dict(TELEMETRY_POLICY),
        }
    )
    seed = frozen.get("seed")
    if isinstance(seed, Mapping):
        seed = dict(seed)
        seed["nested_backend_code_identity_sha256"] = nested_identity["identity_sha256"]
        frozen["seed"] = seed
    plan: dict[str, Any] = dict(parent)
    plan.update(
        {
            "schema_version": "longsplat-future-smoke-plan-v2",
            "coverage_plan_schema": PLAN_SCHEMA,
            "workload_profile": PROFILE,
            "execution_status": "plan_only_derived_coverage",
            "accepted": False,
            "delivery_reachable": False,
            "gpu_invoked": False,
            "training_invoked": False,
            "conversion_invoked": False,
            "source_path": str(source),
            "model_path": str(model),
            "model_path_exists_at_plan_time": False,
            "argv": argv,
            "frozen_contract": frozen,
            "coverage_plan": binding,
            "coverage_workload": workload,
            "nested_backend_code_identity": nested_identity,
            "route_code_identity_sha256": current_code.get("code_identity_sha256"),
            "route_code_identity": dict(current_code),
            "train_script_sha256": sha256_file(train_script),
            "derived_retry": {
                "schema_version": "longsplat-coverage-plan-adapter-v1",
                "reason": reason,
                "parent_plan_path": str(parent_path),
                "parent_plan_sha256": sha256_file(parent_path),
                "parent_static_contract_path": str(static_path),
                "producer_code_identity_sha256": current_code.get("code_identity_sha256"),
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
    except (SmokeExecutorBlocked, CoverageSmokeBlocked, OSError, ValueError) as exc:
        _fail(f"derived coverage plan failed executor validation: {exc}")
    plan["derived_plan_path"] = str(output_path)
    plan["derived_plan_sha256"] = sha256_file(output_path)
    return plan


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind coverage-smoke-v1 to one fresh LongSplat executor plan")
    parser.add_argument("--parent-plan", required=True, type=Path)
    parser.add_argument("--static-contract", required=True, type=Path)
    parser.add_argument("--coverage-plan", required=True, type=Path)
    parser.add_argument("--output-plan", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--route-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--containment-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = derive_coverage_training_plan(
            parent_plan_path=args.parent_plan,
            static_contract_path=args.static_contract,
            coverage_plan_path=args.coverage_plan,
            output_plan_path=args.output_plan,
            model_path=args.model_path,
            route_root=args.route_root,
            containment_root=args.containment_root,
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
                    "camera_contract_sha256": result["coverage_plan"]["camera_contract_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (CoverageAdapterBlocked, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
