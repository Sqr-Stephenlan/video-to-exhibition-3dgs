"""Structured, single-stage executor for the frozen LongSplat GPU smoke.

The executor is intentionally a route-side boundary.  It validates the
immutable CPU input and future plan before constructing either workload, then
invokes the existing ``dev.sh`` wrapper with an argv list and ``shell=False``.
It does not probe CUDA, install dependencies, train beyond the supplied plan,
convert models, or delete partial output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from .longsplat_input import (
        _nested_backend_code_identity,
        resolve_future_workload_profile,
        validate_future_smoke_plan,
    )
    from .coverage_smoke import CoverageSmokeBlocked, validate_coverage_smoke_plan, validate_sampling_telemetry
    from .pipeline_contract import (
        PipelineBlocked,
        code_identity,
        resolve_contained_path,
        resolve_containment_root,
        sha256_file,
        stable_sha256,
    )
    from .gate_schema import GateSchemaBlocked, normalize_gate_evidence
except ImportError:  # direct ``dev.sh python scripts/...`` invocation
    from scripts.longsplat.longsplat_input import (  # type: ignore
        _nested_backend_code_identity,
        resolve_future_workload_profile,
        validate_future_smoke_plan,
    )
    from scripts.longsplat.coverage_smoke import CoverageSmokeBlocked, validate_coverage_smoke_plan, validate_sampling_telemetry  # type: ignore
    from scripts.longsplat.pipeline_contract import (  # type: ignore
        PipelineBlocked,
        code_identity,
        resolve_contained_path,
        resolve_containment_root,
        sha256_file,
        stable_sha256,
    )
    from scripts.longsplat.gate_schema import GateSchemaBlocked, normalize_gate_evidence  # type: ignore


class SmokeExecutorBlocked(RuntimeError):
    """A frozen-plan or append-only execution contract failed."""


_SCHEMA = "longsplat-future-smoke-plan-v2"
_TRAIN_REL = Path("third_party/LongSplat/train.py")
_RENDER_REL = Path("third_party/LongSplat/render.py")
_DATASET_VALUE_FLAGS = (
    "--source_path",
    "--images",
    "--mode",
    "--resolution",
    "--depth_source",
    "--model_path",
)
_DATASET_BOOL_FLAGS = ("--external_colmap_pose", "--disable_resize")

_RENDER_CONSUMER_FILES = {
    "nested/render.py": Path("third_party/LongSplat/render.py"),
    "nested/utils/colmap_utils.py": Path("third_party/LongSplat/utils/colmap_utils.py"),
}
_SNAPSHOT_RELATIVE_FILES = (
    "cfg_args",
    "cameras_all_train.json",
    "cameras_all_test.json",
    "external_colmap_pose_contract.json",
    "point_cloud/iteration_{iteration}/point_cloud.ply",
    "point_cloud/iteration_{iteration}/color_mlp.pt",
    "point_cloud/iteration_{iteration}/cov_mlp.pt",
    "point_cloud/iteration_{iteration}/opacity_mlp.pt",
)

# Conversion/evaluation authority is per-run and lives in
# ``authority_manifest.py``.  Keep only the render snapshot allowlist here;
# its iteration placeholder is filled from the verified manifest.


def _fail(message: str) -> None:
    raise SmokeExecutorBlocked(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not valid JSON: {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object: {path}")
    return value


def derive_training_retry_plan(
    *,
    parent_plan_path: str | Path,
    output_plan_path: str | Path,
    model_path: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    reason: str,
) -> dict[str, Any]:
    """Derive one append-only training plan after an executor-only fix.

    The CPU-produced input, parent binding, profile, seed value, and argv
    semantics remain frozen.  Only the producer identity and fresh model/evidence
    paths move forward, so a failed GPU attempt cannot be overwritten or reused.
    """

    route = Path(route_root).resolve()
    parent_path = _output_path(parent_plan_path, route, "retry parent plan", must_exist=True, containment_root=containment_root)
    output_path = _output_path(output_plan_path, route, "retry plan", must_exist=False, containment_root=containment_root)
    if output_path.exists() or output_path.is_symlink():
        _fail(f"retry plan must be absent: {output_path}")
    model = _output_path(model_path, route, "retry model path", must_exist=False, containment_root=containment_root)
    if model.exists() or model.is_symlink():
        _fail(f"retry model path must be absent: {model}")
    parent = _load_json(parent_path, "retry parent plan")
    current_code = code_identity(route)
    train_script = (route / _TRAIN_REL).resolve()
    nested_identity = _nested_backend_code_identity(route, train_script)
    argv = list(parent.get("argv", []))
    indices = _flag_indices(argv, "--model_path")
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        _fail("retry parent plan has no unique --model_path")
    argv[indices[0] + 1] = str(model)
    result = dict(parent)
    result.update(
        {
            "execution_status": "plan_only_derived_retry",
            "accepted": False,
            "delivery_reachable": False,
            "gpu_invoked": False,
            "training_invoked": False,
            "conversion_invoked": False,
            "model_path": str(model),
            "model_path_exists_at_plan_time": False,
            "argv": argv,
            "nested_backend_code_identity": nested_identity,
            "route_code_identity_sha256": current_code.get("code_identity_sha256"),
            "route_code_identity": dict(current_code),
            "train_script_sha256": sha256_file(train_script),
            "derived_retry": {
                "schema_version": "longsplat-training-plan-retry-v1",
                "reason": reason,
                "parent_plan_path": str(parent_path),
                "parent_plan_sha256": sha256_file(parent_path),
                "parent_static_contract_path": str((Path(str(parent["source_path"])) / "contract" / "static_contract.json").resolve()),
                "producer_code_identity_sha256": current_code.get("code_identity_sha256"),
            },
        }
    )
    frozen = dict(result.get("frozen_contract", {}))
    seed = dict(frozen.get("seed", {}))
    seed["nested_backend_code_identity_sha256"] = nested_identity["identity_sha256"]
    frozen["seed"] = seed
    frozen.setdefault(
        "camera_sampling_telemetry",
        {
            "schema_version": "camera-sampling-telemetry-v1",
            "required_for_new_execution": True,
            "scope": "external_colmap_pose_only",
            "legacy_evidence_compatibility": "absent_allowed",
            "resume_semantics": "artifact inspection only; checkpoint is not bit-exact training resume",
        },
    )
    result["frozen_contract"] = frozen
    _write_stage_output(output_path, result)
    return result


def _inside(root: Path, value: str | Path, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        _fail(f"{label} escapes {root}: {resolved}")
    return resolved


def _route_outputs(
    route_root: str | Path,
    *,
    containment_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return code root plus the verified dynamic data containment root.

    The optional argument is required by production callers.  Omitting it
    preserves the narrow legacy helper contract used by old unit fixtures.
    """

    route = Path(route_root).resolve()
    if containment_root is not None:
        try:
            return route, resolve_containment_root(containment_root)
        except PipelineBlocked as exc:
            _fail(str(exc))
    outputs = route / "outputs"
    if outputs.is_symlink() or not outputs.is_dir():
        _fail(f"route outputs directory is missing or symlinked: {outputs}")
    return route, outputs.resolve()


def _output_path(
    value: str | Path,
    route_root: str | Path,
    label: str,
    *,
    must_exist: bool = False,
    containment_root: str | Path | None = None,
) -> Path:
    """Resolve an evidence path strictly below the verified data root.

    The lexical check rejects ``..`` and the component walk rejects symlink
    traversal before the final real-path containment check.  This makes a
    fresh evidence path safe even when a caller supplied an absolute path.
    """

    route, outputs = _route_outputs(route_root, containment_root=containment_root)
    if containment_root is not None:
        try:
            return resolve_contained_path(
                value,
                root=outputs,
                label=label,
                must_exist=must_exist,
            )
        except PipelineBlocked as exc:
            _fail(str(exc))
    raw = Path(value)
    if not raw.is_absolute():
        _fail(f"{label} must be an absolute path: {raw}")
    project_name = route.parents[1].name if len(route.parents) > 1 else route.name
    if raw.parts.count(project_name) > 1:
        _fail(f"{label} contains a repeated worktree/project prefix: {raw}")
    try:
        lexical_relative = raw.relative_to(outputs)
    except ValueError:
        _fail(f"{label} must be below route outputs: {raw}")
    if not lexical_relative.parts or any(part in {".", ".."} for part in lexical_relative.parts):
        _fail(f"{label} must be a strict descendant of route outputs: {raw}")
    probe = outputs
    if probe.is_symlink():
        _fail(f"{label} traverses a symlinked outputs root: {outputs}")
    for part in lexical_relative.parts:
        probe = probe / part
        if probe.is_symlink():
            _fail(f"{label} traverses a symlink: {probe}")
    resolved = raw.resolve(strict=False)
    try:
        resolved.relative_to(outputs)
    except ValueError:
        _fail(f"{label} resolves outside route outputs: {resolved}")
    if resolved == outputs:
        _fail(f"{label} cannot equal route outputs: {resolved}")
    if must_exist and not resolved.exists():
        _fail(f"{label} is missing: {resolved}")
    return resolved


def _authority_path(
    value: str | Path,
    *,
    containment_root: str | Path | None,
    label: str,
    must_exist: bool = False,
    directory: bool | None = None,
) -> Path:
    """Resolve an artifact under the stage's dynamic containment authority.

    ``route_root`` identifies code.  Production callers additionally provide
    the run-local containment root; every render source and destination must
    then resolve below that same root.  ``None`` preserves the narrow legacy
    helper behavior for direct old callers.
    """

    if containment_root is None:
        return Path(value).resolve()
    try:
        return resolve_contained_path(
            value,
            root=resolve_containment_root(containment_root),
            label=label,
            must_exist=must_exist,
            directory=directory,
        )
    except PipelineBlocked as exc:
        _fail(str(exc))


def _flag_indices(argv: Sequence[str], flag: str) -> list[int]:
    return [index for index, token in enumerate(argv) if token == flag]


def _value(argv: Sequence[str], flag: str) -> str:
    indices = _flag_indices(argv, flag)
    if len(indices) != 1:
        _fail(f"plan argv must contain exactly one {flag}")
    index = indices[0]
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        _fail(f"plan argv {flag} has no value")
    return argv[index + 1]


def _boolean(argv: Sequence[str], flag: str) -> None:
    if len(_flag_indices(argv, flag)) != 1:
        _fail(f"plan argv must contain exactly one {flag}")


def _workload_spec(plan: Mapping[str, Any]) -> dict[str, Any]:
    profile = plan.get("workload_profile")
    if not isinstance(profile, str):
        _fail("future smoke workload_profile is required")
    try:
        spec = resolve_future_workload_profile(plan)
    except Exception as exc:
        _fail(f"future smoke workload profile is invalid: {exc}")
    frozen = plan.get("frozen_contract")
    if not isinstance(frozen, Mapping) or frozen.get("workload_profile") != profile:
        _fail("future smoke frozen workload_profile differs from plan")
    if frozen.get("iterations") != spec["iterations"] or frozen.get("render_iteration") != spec["render_iteration"]:
        _fail(f"future smoke {profile} iteration contract is not fixed")
    return spec


def _contract_iteration(contract: Mapping[str, Any]) -> int:
    value = contract.get("render_iteration")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail("verified plan render_iteration is required")
    return value


def _model_required_files(iteration: int) -> tuple[str, ...]:
    return (
        "cameras_all_train.json",
        "cameras_all_test.json",
        "external_colmap_pose_contract.json",
        "cfg_args",
        f"point_cloud/iteration_{iteration}/point_cloud.ply",
        f"point_cloud/iteration_{iteration}/color_mlp.pt",
        f"point_cloud/iteration_{iteration}/cov_mlp.pt",
        f"point_cloud/iteration_{iteration}/opacity_mlp.pt",
    )


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return True


def _image_names(manifest: Mapping[str, Any], static: Mapping[str, Any]) -> list[str]:
    records = manifest.get("image_records") or manifest.get("frames")
    if not isinstance(records, list) or not records:
        _fail("staging manifest has no dynamic image records")
    filenames: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            _fail("staging manifest image record is malformed")
        name = record.get("name", record.get("staged_name", record.get("frame_id")))
        if not isinstance(name, str) or not name or Path(name).name != name:
            _fail(f"staging image name is not a safe basename: {name!r}")
        filenames.append(name)
    if len(filenames) != len(set(filenames)):
        _fail("staging image names are not unique")
    names = [Path(name).stem for name in filenames]
    if len(names) != len(set(names)):
        _fail("staging image names have duplicate stems across extensions")
    if int(static.get("image_count", 0)) != len(filenames):
        _fail("static contract image_count differs from dynamic staging records")
    return names


def _verify_static_input(
    *,
    plan: Mapping[str, Any],
    static: Mapping[str, Any],
    static_path: Path,
    source: Path,
    route: Path,
    containment_root: Path | None = None,
    enforce_route_identity: bool = True,
    allow_producer_code_drift: bool = False,
) -> dict[str, Any]:
    spec = _workload_spec(plan)
    if static.get("computed_pass") is not True:
        _fail("static contract is not computed_pass")
    for key in ("accepted", "delivery_reachable", "gpu_invoked", "training_invoked", "render_invoked"):
        if static.get(key) is not False:
            _fail(f"static contract {key} must remain false")
    if static.get("route_code_identity_sha256") != plan.get("route_code_identity_sha256"):
        if not allow_producer_code_drift or not isinstance(plan.get("derived_retry"), Mapping):
            _fail("static/plan route code identity differs")
    workload_profile = plan.get("workload_profile")
    coverage_adapter = plan.get("coverage_plan") if workload_profile == "coverage-smoke-v1" else None
    convergence_adapter = plan.get("convergence_plan") if workload_profile == "convergence1000-v1" else None
    formal_adapter = plan.get("formal_plan") if workload_profile == "formal30000-v1" else None
    profile_adapter = next(
        (
            adapter
            for adapter in (coverage_adapter, convergence_adapter, formal_adapter)
            if isinstance(adapter, Mapping)
        ),
        None,
    )
    if static.get("future_smoke_profile") != workload_profile:
        if not isinstance(profile_adapter, Mapping):
            _fail("static/plan future workload profile differs")
    if static.get("future_smoke_iterations") != spec["iterations"]:
        if not isinstance(profile_adapter, Mapping):
            _fail("static/plan future workload iterations differ")
    if isinstance(coverage_adapter, Mapping):
        coverage_path_value = coverage_adapter.get("path")
        coverage_sha = coverage_adapter.get("sha256")
        if not isinstance(coverage_path_value, str) or not isinstance(coverage_sha, str):
            _fail("coverage-smoke plan binding is incomplete")
        coverage_path = Path(coverage_path_value).resolve()
        if coverage_path.is_symlink() or not coverage_path.is_file() or sha256_file(coverage_path) != coverage_sha:
            _fail("coverage-smoke plan binding SHA failed")
        try:
            coverage_plan = _load_json(coverage_path, "coverage-smoke plan")
            coverage_validation = validate_coverage_smoke_plan(coverage_plan)
        except (SmokeExecutorBlocked, CoverageSmokeBlocked) as exc:
            _fail(f"coverage-smoke plan validation failed: {exc}")
        if coverage_validation.get("computed_pass") is not True:
            _fail("coverage-smoke plan is not computed_pass")
        if coverage_plan.get("requested_iterations") != spec["iterations"] or coverage_adapter.get("iterations") != spec["iterations"]:
            _fail("coverage-smoke plan iterations differ from the versioned profile")
        if coverage_plan.get("active_camera_count") != int(static.get("image_count", 0)):
            _fail("coverage-smoke active camera count differs from immutable static input")
    if isinstance(convergence_adapter, Mapping):
        convergence_path_value = convergence_adapter.get("path")
        convergence_sha = convergence_adapter.get("sha256")
        if not isinstance(convergence_path_value, str) or not isinstance(convergence_sha, str):
            _fail("convergence plan binding is incomplete")
        convergence_path = Path(convergence_path_value).resolve()
        if convergence_path.is_symlink() or not convergence_path.is_file() or sha256_file(convergence_path) != convergence_sha:
            _fail("convergence plan binding SHA failed")
        try:
            from .convergence_smoke import ConvergenceSmokeBlocked, validate_convergence_smoke_plan
        except ImportError:
            from scripts.longsplat.convergence_smoke import ConvergenceSmokeBlocked, validate_convergence_smoke_plan  # type: ignore
        convergence_plan = _load_json(convergence_path, "convergence CPU plan")
        try:
            convergence_validation = validate_convergence_smoke_plan(convergence_plan)
        except ConvergenceSmokeBlocked as exc:
            _fail(f"convergence plan validation failed: {exc}")
        if convergence_validation.get("computed_pass") is not True:
            _fail("convergence CPU plan is not computed_pass")
        if convergence_plan.get("requested_iterations") != spec["iterations"] or convergence_adapter.get("iterations") != spec["iterations"]:
            _fail("convergence plan iterations differ from the versioned profile")
        if convergence_plan.get("active_camera_count") != int(static.get("image_count", 0)):
            _fail("convergence active camera count differs from immutable static input")
        if convergence_adapter.get("active_camera_order") != convergence_plan.get("active_camera_order"):
            _fail("convergence binding camera order differs from CPU plan")
    if isinstance(formal_adapter, Mapping):
        policy_path_value = formal_adapter.get("path")
        policy_sha = formal_adapter.get("sha256")
        if not isinstance(policy_path_value, str) or not isinstance(policy_sha, str):
            _fail("formal30000 plan binding is incomplete")
        policy_path = _output_path(policy_path_value, route, "formal30000 policy", must_exist=True, containment_root=containment_root)
        if policy_path.is_symlink() or sha256_file(policy_path) != policy_sha:
            _fail("formal30000 policy binding SHA failed")
        try:
            from .formal_executor import FormalAdapterBlocked, validate_formal_training_plan
        except ImportError:
            from scripts.longsplat.formal_executor import FormalAdapterBlocked, validate_formal_training_plan  # type: ignore
        try:
            formal_policy = _load_json(policy_path, "formal30000 policy")
            formal_validation = validate_formal_training_plan(formal_policy)
        except (SmokeExecutorBlocked, FormalAdapterBlocked) as exc:
            _fail(f"formal30000 policy validation failed: {exc}")
        if formal_validation.get("computed_pass") is not True:
            _fail("formal30000 policy is not computed_pass")
        if formal_policy.get("requested_iterations") != spec["iterations"] or formal_adapter.get("iterations") != spec["iterations"]:
            _fail("formal30000 policy iterations differ from the versioned profile")
        identity = formal_policy.get("identity_binding")
        if not isinstance(identity, Mapping):
            _fail("formal30000 policy identity binding is missing")
        if identity.get("static_contract_sha256") != sha256_file(static_path):
            _fail("formal30000 policy static contract SHA differs from immutable input")
        if identity.get("camera_count") != int(static.get("image_count", 0)):
            _fail("formal30000 policy camera count differs from immutable static input")
        camera_contract_path = source / "camera_contract-v1.json"
        camera_contract = _load_json(camera_contract_path, "formal camera contract")
        camera_order = camera_contract.get("frame_names")
        if not isinstance(camera_order, list) or not camera_order or len(camera_order) != len(set(camera_order)):
            _fail("formal camera contract order is missing or non-unique")
        if identity.get("camera_order") != camera_order or formal_adapter.get("active_camera_order") != camera_order:
            _fail("formal policy camera order differs from the immutable explicit contract")
        if formal_adapter.get("camera_contract_sha256") != sha256_file(camera_contract_path):
            _fail("formal plan camera contract SHA differs from immutable input")
        automated_policy = formal_policy.get("policy_mode") == "automated_technical_v1"
        if automated_policy:
            if formal_adapter.get("release_kind") != "automated_technical_gate":
                _fail("automated formal plan release kind is invalid")
            release_path_value = formal_adapter.get("automated_gate_path")
            release_sha = formal_adapter.get("automated_gate_sha256")
        else:
            release_path_value = formal_adapter.get("supervisor_release_path")
            release_sha = formal_adapter.get("supervisor_release_sha256")
        if not isinstance(release_path_value, str) or not isinstance(release_sha, str):
            _fail("formal plan release binding is incomplete")
        release_path = _output_path(
            release_path_value,
            route,
            "automated early gate" if automated_policy else "formal supervisor release",
            must_exist=True,
            containment_root=containment_root,
        )
        if release_path.is_symlink() or sha256_file(release_path) != release_sha:
            _fail("formal release binding SHA failed")
        release = _load_json(release_path, "formal supervisor release")
        if automated_policy:
            try:
                release = normalize_gate_evidence(release)
            except GateSchemaBlocked as exc:
                _fail(f"automated formal release normalization failed: {exc}")
            if (
                release.get("automated_technical_gate") is not True
                or release.get("computed_pass") is not True
                or release.get("formal_release_eligible") is not True
                or release.get("manual_visual_review") is not False
                or release.get("held_out") is not False
            ):
                _fail("automated formal release is not a valid early technical gate")
            policy_release = formal_policy.get("automated_release")
            if not isinstance(policy_release, Mapping) or policy_release.get("sha256") != release_sha:
                _fail("automated formal policy release SHA differs from bound early gate")
        else:
            decision = release.get("decision")
            if not isinstance(decision, Mapping):
                _fail("formal supervisor release decision is missing")
            if decision.get("EARLY_VISUAL_RECOGNIZABLE") is not True:
                _fail("formal supervisor release did not explicitly recognize early visual structure")
            if decision.get("ROUGH_VISUAL_FOR_FORMAL_GATE") not in {"pass", "needs_review"}:
                _fail("formal supervisor release rough visual gate is not pass/needs_review")
            if decision.get("DELIVERY_QUALITY") is not False or decision.get("held_out") is not False:
                _fail("formal supervisor release must not claim delivery quality or held-out validation")
            policy_release = formal_policy.get("supervisor_release")
            if not isinstance(policy_release, Mapping) or policy_release.get("sha256") != release_sha:
                _fail("formal policy supervisor release SHA differs from bound decision")
    if static_path != (source / "contract" / "static_contract.json").resolve():
        _fail("static contract must be source/contract/static_contract.json")
    manifest_path = source / "staging_manifest.json"
    manifest = _load_json(manifest_path, "staging manifest")
    manifest_plan_value = manifest.get("future_smoke_plan_path")
    if not isinstance(manifest_plan_value, str):
        _fail("staging manifest future smoke plan path is missing")
    manifest_plan = _inside(source, manifest_plan_value, "future smoke plan")
    requested_plan = Path(plan.get("_plan_path", "")).resolve()
    if manifest_plan != requested_plan:
        retry = plan.get("derived_retry")
        if not allow_producer_code_drift or not isinstance(retry, Mapping):
            _fail("staging manifest does not bind the requested future plan")
        parent_path = Path(str(retry.get("parent_plan_path", ""))).resolve()
        if manifest_plan != parent_path or retry.get("parent_plan_sha256") != sha256_file(parent_path):
            _fail("derived retry plan is not bound to the immutable parent plan")
    names = _image_names(manifest, static)
    records = manifest.get("image_records") or manifest.get("frames")
    if not isinstance(records, list) or not records:
        _fail("staging manifest has no dynamic image records")
    image_root = source / "images"
    for record in records:
        if not isinstance(record, Mapping):
            _fail("staging manifest image record is malformed")
        filename = record.get("name", record.get("staged_name", record.get("frame_id")))
        if not isinstance(filename, str) or Path(filename).name != filename:
            _fail(f"staging image name is not a safe basename: {filename!r}")
        image = image_root / filename
        if not image.is_file() or image.is_symlink():
            _fail(f"training image is missing or not an independent file: {image}")
    sparse = source / "sparse" / "0"
    for filename in ("cameras.txt", "images.txt", "points3D.txt", "cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse / filename).is_file():
            _fail(f"training sparse model artifact is missing: {sparse / filename}")
    camera = static.get("camera")
    if not isinstance(camera, Mapping) or camera.get("model") != "PINHOLE":
        _fail("static contract must contain a PINHOLE camera")
    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        value = camera.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
            _fail(f"static camera field is invalid: {key}")
    # This read-only validator is the authoritative reference/track/pixel
    # checker.  It runs in the route environment and never imports torch.
    try:
        try:
            from .validate_external_colmap_contract import validate_training_input
        except ImportError:
            from scripts.longsplat.validate_external_colmap_contract import validate_training_input  # type: ignore

        validated = validate_training_input(
            source,
            containment_root=containment_root,
            allow_producer_code_drift=allow_producer_code_drift,
        )
    except Exception as exc:
        _fail(f"static training-input validation failed: {type(exc).__name__}: {exc}")
    if validated != dict(static):
        _fail("static contract on disk differs from a fresh read-only validation")
    current_code = code_identity(route)
    if enforce_route_identity and current_code.get("code_identity_sha256") != plan.get("route_code_identity_sha256"):
        _fail("route source identity differs from the frozen plan")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "image_names": names,
        "image_count": len(names),
        "camera": dict(camera),
        "workload_profile": plan["workload_profile"],
        "iterations": spec["iterations"],
        "render_iteration": spec["render_iteration"],
        "route_code_identity": current_code,
    }


def _verify_render_consumer_identity(plan: Mapping[str, Any], route: Path) -> dict[str, Any]:
    """Permit executor-only drift while pinning the render math consumers."""

    declared = plan.get("route_code_identity")
    files = declared.get("files") if isinstance(declared, Mapping) else None
    if not isinstance(files, list):
        _fail("frozen plan route code identity has no file records")
    expected_by_path = {
        str(item.get("path")): item
        for item in files
        if isinstance(item, Mapping) and isinstance(item.get("path"), str)
    }
    records: dict[str, Any] = {}
    for producer_path, current_relative in _RENDER_CONSUMER_FILES.items():
        declared_file = expected_by_path.get(producer_path)
        if not isinstance(declared_file, Mapping) or not isinstance(declared_file.get("sha256"), str):
            _fail(f"frozen plan is missing render consumer identity: {producer_path}")
        current_path = (route / current_relative).resolve()
        if not current_path.is_file() or current_path.is_symlink():
            _fail(f"render consumer is missing or symlinked: {current_path}")
        actual_sha = sha256_file(current_path)
        if actual_sha != declared_file["sha256"]:
            _fail(f"render consumer identity drifted: {current_path}")
        records[producer_path] = {
            "producer_path": producer_path,
            "current_path": str(current_path),
            "sha256": actual_sha,
            "size_bytes": current_path.stat().st_size,
        }
    return {
        "producer_route_code_identity_sha256": plan.get("route_code_identity_sha256"),
        "files": records,
    }


def validate_plan_and_static(
    plan_path: str | Path,
    static_contract_path: str | Path,
    *,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    verify_static: bool = True,
    allow_existing_model: bool = False,
    allow_producer_code_drift: bool = False,
) -> dict[str, Any]:
    """Read and validate the plan/static contract without creating outputs."""

    route = Path(route_root).resolve()
    wrapper = route / "dev.sh"
    if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
        _fail(f"route dev.sh wrapper is missing or not executable: {wrapper}")
    plan_file = Path(plan_path).resolve()
    static_file = Path(static_contract_path).resolve()
    if containment_root is not None:
        plan_file = _output_path(plan_file, route, "future smoke plan", must_exist=True, containment_root=containment_root)
        static_file = _output_path(static_file, route, "static contract", must_exist=True, containment_root=containment_root)
    plan = _load_json(plan_file, "future smoke plan")
    plan["_plan_path"] = str(plan_file)
    static = _load_json(static_file, "static contract")
    if plan.get("schema_version") != _SCHEMA:
        _fail(f"future smoke plan schema must be {_SCHEMA}")
    spec = _workload_spec(plan)
    argv = plan.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(token, str) for token in argv):
        _fail("future smoke plan argv must be a non-empty string list")
    if any("\x00" in token for token in argv):
        _fail("plan argv contains a NUL token")
    if "--seed" in argv or "--skip_train" in argv or "--skip_test" in argv:
        _fail("training plan contains a render/unsupported flag")
    source_value = plan.get("source_path")
    model_value = plan.get("model_path")
    if not isinstance(source_value, str) or not Path(source_value).is_absolute():
        _fail("plan source_path must be an absolute path")
    if not isinstance(model_value, str) or not Path(model_value).is_absolute():
        _fail("plan model_path must be an absolute path")
    source = Path(source_value).resolve()
    model = Path(model_value).resolve()
    if containment_root is not None:
        source = _output_path(source, route, "training source directory", must_exist=True, containment_root=containment_root)
        model = _output_path(model, route, "training model path", must_exist=False, containment_root=containment_root)
    if not source.is_dir():
        _fail(f"plan source_path is not a directory: {source}")
    if plan.get("model_path_exists_at_plan_time") is not False:
        _fail(f"plan must declare an absent model_path at plan time: {model}")
    if not allow_existing_model and model.exists():
        _fail(f"fresh model_path is required and must be absent: {model}")
    if allow_existing_model and not model.is_dir():
        _fail(f"render model_path is missing: {model}")
    backend = Path(argv[0]).resolve()
    if not backend.is_file() or not os.access(backend, os.X_OK):
        _fail(f"plan backend executable is missing/not executable: {backend}")
    backend_declared = plan.get("backend_identity")
    if isinstance(backend_declared, Mapping) and backend_declared.get("resolved_path"):
        if backend != Path(str(backend_declared["resolved_path"])).resolve():
            _fail("plan backend resolved path differs from argv")
    train = Path(argv[1]).resolve() if len(argv) > 1 else Path()
    if train != (route / _TRAIN_REL).resolve():
        _fail("plan train entrypoint is not the route nested train.py")
    if _value(argv, "--source_path") != str(source):
        _fail("argv source_path differs from plan source_path")
    if _value(argv, "--model_path") != str(model):
        _fail("argv model_path differs from plan model_path")
    for flag, expected in {
        "--images": "images",
        "--mode": "custom",
        "--resolution": "1",
        "--iterations": str(spec["iterations"]),
        "--depth_source": "disabled",
        "--loss_2d_correspondence_weight": "0",
        "--depth_loss_weight": "0",
        "--rotation_lr_init": "0",
        "--translation_lr_init": "0",
    }.items():
        if _value(argv, flag) != expected:
            _fail(f"argv {flag} must be {expected}")
    for flag in _DATASET_BOOL_FLAGS:
        _boolean(argv, flag)
    if "--external_colmap_pose" not in argv or "--disable_resize" not in argv:
        _fail("external fixed-pose flags are required")
    if static_file != (source / "contract" / "static_contract.json").resolve():
        _fail("static contract path is outside the plan source contract")
    retry = plan.get("derived_retry")
    derived_retry = isinstance(retry, Mapping)
    if static.get("route_code_identity_sha256") != plan.get("route_code_identity_sha256"):
        if not derived_retry:
            _fail("static/plan route code identity differs")
        parent_path = Path(str(retry.get("parent_plan_path", ""))).resolve()
        parent = _load_json(parent_path, "derived retry parent plan")
        if retry.get("parent_plan_sha256") != sha256_file(parent_path):
            _fail("derived retry parent plan SHA differs")
        if parent.get("route_code_identity_sha256") != static.get("route_code_identity_sha256"):
            _fail("derived retry parent plan is not bound to the static contract")
    plan_future_value = static.get("future_smoke_plan_path")
    if not isinstance(plan_future_value, str):
        _fail("static contract future plan binding is missing")
    static_plan = _inside(source, plan_future_value, "static future plan")
    if static_plan != plan_file:
        if not derived_retry or static_plan != Path(str(retry.get("parent_plan_path", ""))).resolve():
            _fail("static contract future plan binding differs from requested plan")
    # Reuse the existing seed/parent-binding validator; this also proves the
    # nested safe_state/train ordering and refuses a stale code identity.
    try:
        validate_future_smoke_plan(
            plan,
            route_root=route,
            containment_root=containment_root,
            training_root=source,
            allow_producer_code_drift=derived_retry or allow_producer_code_drift,
        )
    except Exception as exc:
        _fail(f"future smoke plan validation failed: {type(exc).__name__}: {exc}")
    current_code = code_identity(route)
    if not allow_producer_code_drift and current_code.get("code_identity_sha256") != plan.get("route_code_identity_sha256"):
        _fail("route source identity differs from the frozen plan")
    details = {
        "plan_path": str(plan_file),
        "plan_sha256": sha256_file(plan_file),
        "static_contract_path": str(static_file),
        "static_contract_sha256": sha256_file(static_file),
        "workload_profile": plan["workload_profile"],
        "iterations": spec["iterations"],
        "render_iteration": spec["render_iteration"],
        "source_path": str(source),
        "model_path": str(model),
        "backend_path": str(backend),
        "train_path": str(train),
        "argv": list(argv),
        "plan": plan,
        "static": static,
        "backend_env": str(backend.parent.parent),
        "current_route_code_identity": current_code,
        "producer_route_code_identity": plan.get("route_code_identity"),
    }
    if verify_static:
        details["static_verification"] = _verify_static_input(
            plan=plan,
            static=static,
            static_path=static_file,
            source=source,
            route=route,
            containment_root=None if containment_root is None else resolve_containment_root(containment_root),
            enforce_route_identity=not allow_producer_code_drift,
            allow_producer_code_drift=derived_retry or allow_producer_code_drift,
        )
    else:
        details["static_verification"] = {
            "image_count": int(static.get("image_count", 0)),
            "camera": dict(static.get("camera", {})),
        }
    if allow_producer_code_drift:
        details["render_consumer_identity"] = _verify_render_consumer_identity(plan, route)
    return details


def build_training_command(contract: Mapping[str, Any], *, route_root: str | Path) -> list[str]:
    """Build the wrapper command without shell interpolation or path joining."""

    route = Path(route_root).resolve()
    argv = contract["argv"]
    return [str(route / "dev.sh"), "python", *list(argv[1:])]


def build_render_command(contract: Mapping[str, Any], *, route_root: str | Path) -> list[str]:
    """Derive render dataset args from the verified plan and add fixed flags."""

    route = Path(route_root).resolve()
    argv = list(contract["argv"])
    iteration = _contract_iteration(contract)
    result = [str(route / "dev.sh"), "python", str(route / _RENDER_REL)]
    for flag in _DATASET_VALUE_FLAGS:
        result.extend((flag, _value(argv, flag)))
    for flag in _DATASET_BOOL_FLAGS:
        if flag in argv:
            result.append(flag)
    result.extend(("--iteration", str(iteration), "--eval", "--skip_test", "--nvs_pose_mode", "adjacent_midpoint_slerp"))
    if "--skip_train" in result:
        _fail("render command must not contain --skip_train")
    return result


def _environment(contract: Mapping[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["VENV_DIR"] = str(contract["backend_env"])
    return env


def _redacted_environment(env: Mapping[str, str]) -> dict[str, str]:
    sensitive = ("TOKEN", "SECRET", "PASSWORD", "PRIVATE", "CREDENTIAL", "AUTH")
    return {
        key: ("<redacted>" if any(part in key.upper() for part in sensitive) else value)
        for key, value in sorted(env.items())
    }


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _file_identity(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} is missing or symlinked: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _snapshot_render_model(
    *,
    contract: Mapping[str, Any],
    training_evidence: Mapping[str, Any],
    evidence_root: Path,
    route: Path,
    containment_root: str | Path | None = None,
) -> tuple[Path, dict[str, Any], Path]:
    """Copy only the immutable render inputs into an isolated model root."""

    original = _authority_path(
        contract["model_path"],
        containment_root=containment_root,
        label="render snapshot source model",
        must_exist=True,
        directory=True,
    )
    evidence_root = _authority_path(
        evidence_root,
        containment_root=containment_root,
        label="render snapshot evidence root",
        must_exist=True,
        directory=True,
    )
    iteration = _contract_iteration(contract)
    snapshot = evidence_root / "render_model_snapshot"
    snapshot = _authority_path(
        snapshot,
        containment_root=containment_root,
        label="isolated render snapshot",
    )
    if snapshot.exists() or snapshot.is_symlink():
        _fail(f"isolated render snapshot must be absent: {snapshot}")
    snapshot.mkdir(parents=True, exist_ok=False)
    source_records: list[dict[str, Any]] = []
    for template in _SNAPSHOT_RELATIVE_FILES:
        relative = template.format(iteration=iteration)
        source = _authority_path(
            original / relative,
            containment_root=containment_root,
            label=f"render snapshot source {relative}",
            must_exist=True,
            directory=False,
        )
        destination = _authority_path(
            snapshot / relative,
            containment_root=containment_root,
            label=f"render snapshot destination {relative}",
            directory=False,
        )
        source_identity = _file_identity(source, "render snapshot source")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination_identity = _file_identity(destination, "render snapshot destination")
        if source_identity["sha256"] != destination_identity["sha256"] or source_identity["size_bytes"] != destination_identity["size_bytes"]:
            _fail(f"render snapshot copy identity differs: {relative}")
        source_records.append(
            {
                "relative_path": relative,
                "source_path": str(source),
                "snapshot_path": str(destination),
                "source_sha256": source_identity["sha256"],
                "snapshot_sha256": destination_identity["sha256"],
                "source_size_bytes": source_identity["size_bytes"],
                "snapshot_size_bytes": destination_identity["size_bytes"],
            }
        )
    before = {record["relative_path"]: (record["source_sha256"], record["source_size_bytes"]) for record in source_records}
    after = {
        relative: (
            _file_identity(
                _authority_path(
                    original / relative,
                    containment_root=containment_root,
                    label=f"render snapshot source recheck {relative}",
                    must_exist=True,
                    directory=False,
                ),
                "render snapshot source recheck",
            )["sha256"],
            _file_identity(
                _authority_path(
                    original / relative,
                    containment_root=containment_root,
                    label=f"render snapshot source recheck {relative}",
                    must_exist=True,
                    directory=False,
                ),
                "render snapshot source recheck",
            )["size_bytes"],
        )
        for relative in before
    }
    if before != after:
        _fail("original training model changed during isolated snapshot creation")
    snapshot_aggregate = stable_sha256(source_records)
    consumer = contract.get("render_consumer_identity", {})
    consumer_files = consumer.get("files", {}) if isinstance(consumer, Mapping) else {}
    render_file = consumer_files.get("nested/render.py", {}) if isinstance(consumer_files, Mapping) else {}
    colmap_file = consumer_files.get("nested/utils/colmap_utils.py", {}) if isinstance(consumer_files, Mapping) else {}
    manifest: dict[str, Any] = {
        "schema": "longsplat-isolated-render-snapshot-v1",
        "reason": "preserve_partial_render_without_overwrite",
        "original_training_model_path": str(original),
        "isolated_render_model_path": str(snapshot),
        "training_evidence_root": training_evidence.get("root"),
        "training_result_path": training_evidence.get("result_path"),
        "training_result_sha256": training_evidence.get("result_sha256"),
        "training_request_path": training_evidence.get("request_path"),
        "training_request_sha256": training_evidence.get("request_sha256"),
        "plan_path": contract["plan_path"],
        "plan_sha256": contract["plan_sha256"],
        "static_contract_path": contract["static_contract_path"],
        "static_contract_sha256": contract["static_contract_sha256"],
        "producer_route_code_identity_sha256": contract["plan"].get("route_code_identity_sha256"),
        "consumer_executor_sha256": sha256_file(Path(__file__).resolve()),
        "consumer_render_py_sha256": render_file.get("sha256"),
        "consumer_colmap_utils_sha256": colmap_file.get("sha256"),
        "iteration": iteration,
        "files": source_records,
        "snapshot_aggregate_sha256": snapshot_aggregate,
        "source_model_unchanged_after_copy": True,
        "route_root": str(route),
    }
    manifest_path = snapshot / "snapshot_manifest.json"
    _write_json_once(manifest_path, manifest)
    return snapshot, manifest, manifest_path


def _verify_snapshot_manifest(
    manifest_path: Path,
    *,
    containment_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = _authority_path(
        manifest_path,
        containment_root=containment_root,
        label="isolated render snapshot manifest",
        must_exist=True,
        directory=False,
    )
    manifest = _load_json(manifest_path, "isolated render snapshot manifest")
    snapshot = manifest_path.parent
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        _fail("isolated render snapshot has no file records")
    expected_paths: set[Path] = set()
    for record in files:
        if not isinstance(record, Mapping):
            _fail("isolated render snapshot file record is malformed")
        relative = record.get("relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            _fail("isolated render snapshot relative path is unsafe")
        source = _authority_path(
            str(record.get("source_path", "")),
            containment_root=containment_root,
            label=f"isolated render snapshot source {relative}",
            must_exist=True,
            directory=False,
        )
        destination = _authority_path(
            snapshot / relative,
            containment_root=containment_root,
            label=f"isolated render snapshot destination {relative}",
            must_exist=True,
            directory=False,
        )
        if destination != Path(str(record.get("snapshot_path", ""))).resolve():
            _fail("isolated render snapshot destination binding differs")
        source_identity = _file_identity(source, "isolated render snapshot source")
        destination_identity = _file_identity(destination, "isolated render snapshot destination")
        for prefix, identity in (("source", source_identity), ("snapshot", destination_identity)):
            if record.get(f"{prefix}_sha256") != identity["sha256"] or record.get(f"{prefix}_size_bytes") != identity["size_bytes"]:
                _fail(f"isolated render snapshot {prefix} identity drifted: {relative}")
        expected_paths.add(destination)
    actual_paths = {path for path in snapshot.rglob("*") if path.is_file() and path.name != manifest_path.name}
    if containment_root is not None:
        for path in actual_paths:
            _authority_path(
                path,
                containment_root=containment_root,
                label=f"isolated render snapshot file {path.name}",
                must_exist=True,
                directory=False,
            )
    unexpected = actual_paths - expected_paths
    unexpected_non_render = {
        path for path in unexpected if path.relative_to(snapshot).parts[:1] != ("train",)
    }
    if unexpected_non_render:
        _fail("isolated render snapshot contains files outside the allowlist/render output")
    if manifest.get("snapshot_aggregate_sha256") != stable_sha256(files):
        _fail("isolated render snapshot aggregate SHA failed")
    return manifest


def _replace_model_path(contract: Mapping[str, Any], model_path: Path) -> dict[str, Any]:
    result = dict(contract)
    argv = list(contract["argv"])
    indices = _flag_indices(argv, "--model_path")
    if len(indices) != 1 or indices[0] + 1 >= len(argv):
        _fail("verified plan has no unique model_path token")
    argv[indices[0] + 1] = str(model_path.resolve())
    result["argv"] = argv
    result["model_path"] = str(model_path.resolve())
    return result


def _verify_ply_finite(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    marker = b"end_header\n"
    header_end = data.find(marker)
    if header_end < 0:
        _fail(f"checkpoint PLY header is malformed: {path}")
    header = data[:header_end].decode("ascii", errors="strict").splitlines()
    if "format binary_little_endian 1.0" not in header:
        _fail(f"checkpoint PLY is not binary little-endian: {path}")
    vertex_count = 0
    properties: list[tuple[str, str]] = []
    in_vertex = False
    sizes = {"char": 1, "uchar": 1, "short": 2, "ushort": 2, "int": 4, "uint": 4, "float": 4, "double": 8}
    for line in header:
        fields = line.split()
        if fields[:1] == ["element"]:
            in_vertex = len(fields) == 3 and fields[1] == "vertex"
            if in_vertex:
                vertex_count = int(fields[2])
        elif in_vertex and fields[:1] == ["property"] and len(fields) == 3:
            if fields[1] not in sizes:
                _fail(f"unsupported checkpoint PLY property type: {fields[1]}")
            properties.append((fields[1], fields[2]))
    if vertex_count <= 0 or not properties:
        _fail(f"checkpoint PLY has no positive vertex schema: {path}")
    stride = sum(sizes[type_name] for type_name, _ in properties)
    payload = data[header_end + len(marker):]
    if len(payload) < vertex_count * stride:
        _fail(f"checkpoint PLY payload is truncated: {path}")
    offset = 0
    format_codes = {"char": "b", "uchar": "B", "short": "h", "ushort": "H", "int": "i", "uint": "I", "float": "f", "double": "d"}
    for _ in range(vertex_count):
        for type_name, _ in properties:
            size = sizes[type_name]
            value = struct.unpack_from("<" + format_codes[type_name], payload, offset)[0]
            if type_name in ("float", "double") and not math.isfinite(float(value)):
                _fail(f"checkpoint PLY contains non-finite vertex data: {path}")
            offset += size
    return {"path": str(path), "sha256": sha256_file(path), "vertex_count": vertex_count, "finite": True}


def _verify_training_outputs(
    contract: Mapping[str, Any],
    *,
    containment_root: str | Path | None = None,
) -> dict[str, Any]:
    model = _authority_path(
        contract["model_path"],
        containment_root=containment_root,
        label="training model",
        must_exist=True,
        directory=True,
    )
    static = contract["static"]
    camera = static["camera"]
    expected_names = list(contract["static_verification"]["image_names"])
    iteration = _contract_iteration(contract)
    required_files = _model_required_files(iteration)
    if containment_root is not None:
        for relative in required_files:
            _authority_path(
                model / relative,
                containment_root=containment_root,
                label=f"training model artifact {relative}",
                must_exist=True,
                directory=False,
            )
    missing = [str(model / relative) for relative in required_files if not (model / relative).is_file()]
    if missing:
        _fail("training structural artifacts are missing: " + ", ".join(missing))
    # The native files are JSON arrays, not object contracts.
    train_value = json.loads((model / "cameras_all_train.json").read_text(encoding="utf-8"))
    test_value = json.loads((model / "cameras_all_test.json").read_text(encoding="utf-8"))
    if not isinstance(train_value, list) or len(train_value) != len(expected_names):
        _fail("cameras_all_train.json does not contain the dynamic active camera count")
    if not isinstance(test_value, list):
        _fail("cameras_all_test.json must be a JSON list")
    names: list[str] = []
    for item in train_value:
        if not isinstance(item, Mapping):
            _fail("active camera record is malformed")
        name = item.get("image_name")
        if not isinstance(name, str) or not name:
            _fail("active camera image_name is missing")
        names.append(Path(name).stem)
        if int(item.get("width", -1)) != int(camera["width"]) or int(item.get("height", -1)) != int(camera["height"]):
            _fail("active camera dimensions differ from static PINHOLE contract")
        for field in ("Focalx", "Focaly"):
            if not math.isclose(float(item.get(field, "nan")), float(camera["fx"] if field == "Focalx" else camera["fy"]), rel_tol=0.0, abs_tol=1e-6):
                _fail(f"active camera {field} differs from static PINHOLE contract")
        if not _all_finite(item):
            _fail("active camera contains non-finite pose data")
    if names != expected_names:
        _fail("active camera names differ from dynamic staged order")
    pose_contract = _load_json(model / "external_colmap_pose_contract.json", "external pose contract")
    for key, expected in (("external_colmap_pose", True), ("mast3r_global_align_skipped", True), ("pose_frozen", True), ("vda_external_depth_disabled", True), ("depth_source", "disabled")):
        if pose_contract.get(key) != expected:
            _fail(f"external pose contract mismatch at {key}")
    if pose_contract.get("active_camera_count") != len(expected_names) or pose_contract.get("reference_camera_count") != len(expected_names):
        _fail("external pose contract camera counts differ from static input")
    for key in ("active_vs_colmap_normalized_residual_p90", "active_vs_colmap_center_residual_p90", "active_vs_colmap_rotation_max_abs", "active_vs_colmap_translation_residual_p90"):
        if not isinstance(pose_contract.get(key), (int, float)) or not math.isfinite(float(pose_contract[key])) or float(pose_contract[key]) > 1e-5:
            _fail(f"external pose residual exceeds strict smoke tolerance: {key}")
    frozen = contract.get("plan", {}).get("frozen_contract", {}) if isinstance(contract.get("plan"), Mapping) else {}
    workload_profile = contract.get("plan", {}).get("workload_profile") if isinstance(contract.get("plan"), Mapping) else None
    telemetry_policy = frozen.get("camera_sampling_telemetry") if isinstance(frozen, Mapping) else None
    telemetry_required = isinstance(telemetry_policy, Mapping) and telemetry_policy.get("required_for_new_execution") is True
    telemetry_files = (model / "camera_sampling_telemetry-v1.jsonl", model / "camera_sampling_telemetry-v1.json")
    if telemetry_required or any(path.exists() for path in telemetry_files):
        identity = pose_contract.get("camera_identity")
        expected_contract_sha = identity.get("contract_file_sha256") if isinstance(identity, Mapping) else None
        internal_names = pose_contract.get("image_names")
        if not isinstance(internal_names, list):
            _fail("external pose contract image_names are required for sampling telemetry")
        telemetry = validate_sampling_telemetry(
            model_path=model,
            expected_iterations=iteration,
            expected_internal_names=[str(value) for value in internal_names],
            expected_contract_sha256=expected_contract_sha if isinstance(expected_contract_sha, str) else None,
        )
        bound = pose_contract.get("camera_sampling_telemetry")
        if telemetry_required and not isinstance(bound, Mapping):
            _fail("new external training contract must bind camera sampling telemetry")
        if isinstance(bound, Mapping):
            if bound.get("summary_sha256") != telemetry.get("summary_sha256"):
                _fail("external pose contract telemetry summary SHA differs")
            if bound.get("events_sha256") != telemetry.get("events_sha256"):
                _fail("external pose contract telemetry event SHA differs")
        if workload_profile == "coverage-smoke-v1":
            expected_count = len(expected_names)
            if telemetry.get("active_camera_count") != expected_count:
                _fail("coverage telemetry active camera count differs from static input")
            if telemetry.get("iterations") != 2 * expected_count:
                _fail("coverage telemetry must contain exactly two complete camera rounds")
            if telemetry.get("unique_camera_count") != expected_count:
                _fail("coverage telemetry did not expose every active camera")
            exposure_counts = telemetry.get("exposure_counts")
            if not isinstance(exposure_counts, Mapping) or set(exposure_counts) != set(internal_names):
                _fail("coverage telemetry exposure identity set differs from pose contract")
            if any(value != 2 for value in exposure_counts.values()):
                _fail("coverage telemetry must expose every active camera exactly twice")
            anchor_schedule = pose_contract.get("anchor_schedule")
            if not isinstance(anchor_schedule, Mapping) or anchor_schedule.get("observed_from_runtime") is not True:
                _fail("coverage training must record runtime anchor schedule events")
            observed_anchor = anchor_schedule.get("observed_iterations")
            if not isinstance(observed_anchor, list) or any(
                isinstance(value, bool) or not isinstance(value, int) for value in observed_anchor
            ):
                _fail("coverage runtime anchor schedule is malformed")
            coverage_binding = contract.get("plan", {}).get("coverage_plan") if isinstance(contract.get("plan"), Mapping) else None
            planned_anchor = coverage_binding.get("planned_anchor_adjust_iterations") if isinstance(coverage_binding, Mapping) else None
            if not isinstance(planned_anchor, list):
                _fail("coverage plan must record planned anchor schedule iterations")
            if observed_anchor != planned_anchor:
                _fail("runtime anchor schedule differs from the versioned coverage plan")
        elif workload_profile == "convergence1000-v1":
            try:
                from .convergence_smoke import (
                    ConvergenceSmokeBlocked,
                    expected_anchor_adjust_iterations,
                    predicted_exposure,
                    validate_anchor_events,
                )
            except ImportError:
                from scripts.longsplat.convergence_smoke import (  # type: ignore
                    ConvergenceSmokeBlocked,
                    expected_anchor_adjust_iterations,
                    predicted_exposure,
                    validate_anchor_events,
                )
            expected_prediction = predicted_exposure(len(expected_names), iteration)
            if telemetry.get("iterations") != iteration or iteration != 1000:
                _fail("convergence telemetry must contain exactly 1000 iterations")
            if telemetry.get("active_camera_count") != len(expected_names) or telemetry.get("unique_camera_count") != len(expected_names):
                _fail("convergence telemetry must expose every active camera")
            exposure_counts = telemetry.get("exposure_counts")
            if not isinstance(exposure_counts, Mapping) or set(exposure_counts) != set(internal_names):
                _fail("convergence telemetry exposure identity set differs from pose contract")
            values = list(exposure_counts.values())
            if any(value not in (expected_prediction["min_exposure_count"], expected_prediction["max_exposure_count"]) for value in values):
                _fail("convergence telemetry exposure counts are outside the floor/ceil distribution")
            if values.count(expected_prediction["min_exposure_count"]) != expected_prediction["camera_count_at_min_exposure"]:
                _fail("convergence telemetry floor-exposure camera count is incorrect")
            if values.count(expected_prediction["max_exposure_count"]) != expected_prediction["camera_count_at_max_exposure"]:
                _fail("convergence telemetry ceil-exposure camera count is incorrect")
            anchor_schedule = pose_contract.get("anchor_schedule")
            if not isinstance(anchor_schedule, Mapping) or anchor_schedule.get("observed_from_runtime") is not True:
                _fail("convergence training must record runtime anchor schedule events")
            observed_anchor = anchor_schedule.get("observed_iterations")
            observed_events = anchor_schedule.get("observed_events")
            if not isinstance(observed_anchor, list) or not isinstance(observed_events, list):
                _fail("convergence runtime anchor events are missing")
            convergence_binding = contract.get("plan", {}).get("convergence_plan") if isinstance(contract.get("plan"), Mapping) else None
            planned_anchor = convergence_binding.get("planned_anchor_adjust_iterations") if isinstance(convergence_binding, Mapping) else None
            if not isinstance(planned_anchor, list):
                _fail("convergence plan must record planned anchor schedule iterations")
            if observed_anchor != planned_anchor or observed_anchor != expected_anchor_adjust_iterations(iteration):
                _fail("runtime anchor schedule differs from the versioned convergence plan")
            try:
                anchor_validation = validate_anchor_events(observed_events, planned_anchor)
            except ConvergenceSmokeBlocked as exc:
                _fail(f"convergence anchor runtime telemetry is invalid: {exc}")
            telemetry = dict(telemetry)
            telemetry["exposure_distribution"] = expected_prediction
            telemetry["anchor_runtime_validation"] = anchor_validation
        elif workload_profile == "formal30000-v1":
            try:
                from .convergence_smoke import (
                    ConvergenceSmokeBlocked,
                    expected_anchor_adjust_iterations,
                    predicted_exposure,
                    validate_anchor_events,
                )
            except ImportError:
                from scripts.longsplat.convergence_smoke import (  # type: ignore
                    ConvergenceSmokeBlocked,
                    expected_anchor_adjust_iterations,
                    predicted_exposure,
                    validate_anchor_events,
                )
            if iteration != 30000 or telemetry.get("iterations") != iteration:
                _fail("formal30000 telemetry must contain exactly 30000 iterations")
            if telemetry.get("active_camera_count") != len(expected_names) or telemetry.get("unique_camera_count") != len(expected_names):
                _fail("formal30000 telemetry must expose every active camera")
            exposure_counts = telemetry.get("exposure_counts")
            if not isinstance(exposure_counts, Mapping) or set(exposure_counts) != set(internal_names):
                _fail("formal30000 telemetry exposure identity set differs from pose contract")
            expected_prediction = predicted_exposure(len(expected_names), iteration)
            values = list(exposure_counts.values())
            if any(value not in (expected_prediction["min_exposure_count"], expected_prediction["max_exposure_count"]) for value in values):
                _fail("formal30000 telemetry exposure counts are outside the floor/ceil distribution")
            if values.count(expected_prediction["min_exposure_count"]) != expected_prediction["camera_count_at_min_exposure"]:
                _fail("formal30000 floor-exposure camera count is incorrect")
            if values.count(expected_prediction["max_exposure_count"]) != expected_prediction["camera_count_at_max_exposure"]:
                _fail("formal30000 ceil-exposure camera count is incorrect")
            formal_binding = contract.get("plan", {}).get("formal_plan") if isinstance(contract.get("plan"), Mapping) else None
            planned_anchor = formal_binding.get("planned_anchor_adjust_iterations") if isinstance(formal_binding, Mapping) else None
            expected_anchor = expected_anchor_adjust_iterations(iteration)
            if not isinstance(planned_anchor, list) or planned_anchor != expected_anchor:
                _fail("formal30000 plan anchor schedule is not the exact local fork schedule")
            anchor_schedule = pose_contract.get("anchor_schedule")
            if not isinstance(anchor_schedule, Mapping) or anchor_schedule.get("observed_from_runtime") is not True:
                _fail("formal30000 training must record runtime anchor schedule events")
            observed_anchor = anchor_schedule.get("observed_iterations")
            observed_events = anchor_schedule.get("observed_events")
            if not isinstance(observed_anchor, list) or not isinstance(observed_events, list):
                _fail("formal30000 runtime anchor events are missing")
            if observed_anchor != planned_anchor:
                _fail("formal30000 runtime anchor schedule differs from the versioned plan")
            try:
                anchor_validation = validate_anchor_events(observed_events, planned_anchor)
            except ConvergenceSmokeBlocked as exc:
                _fail(f"formal30000 anchor runtime telemetry is invalid: {exc}")
            telemetry = dict(telemetry)
            telemetry["exposure_distribution"] = expected_prediction
            telemetry["anchor_runtime_validation"] = anchor_validation
    else:
        telemetry = {
            "schema_version": "camera-sampling-telemetry-v1",
            "status": "legacy_absent_allowed",
            "required_for_new_execution": False,
            "bit_exact_resume": False,
        }
    ply = _verify_ply_finite(model / f"point_cloud/iteration_{iteration}/point_cloud.ply")
    return {
        "structural_pass": True,
        "model_path": str(model),
        "active_camera_count": len(train_value),
        "test_camera_count": len(test_value),
        "camera": dict(camera),
        "external_pose_contract_sha256": sha256_file(model / "external_colmap_pose_contract.json"),
        "checkpoint": ply,
        "mlp_files": {relative: {"sha256": sha256_file(model / relative), "size_bytes": (model / relative).stat().st_size} for relative in required_files if relative.endswith(".pt")},
        "cameras_all_train_sha256": sha256_file(model / "cameras_all_train.json"),
        "cameras_all_test_sha256": sha256_file(model / "cameras_all_test.json"),
        "camera_sampling_telemetry": telemetry,
        "anchor_schedule": pose_contract.get("anchor_schedule") if workload_profile in {"coverage-smoke-v1", "convergence1000-v1", "formal30000-v1"} else {
            "status": "legacy_or_noncoverage_not_required"
        },
    }


def _check_pngs(paths: Sequence[Path], width: int, height: int) -> None:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        _fail(f"render PNG verification requires OpenCV/NumPy: {exc}")
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim < 2 or image.shape[1] != width or image.shape[0] != height:
            _fail(f"render PNG dimensions/decode failed: {path}")
        if not np.isfinite(image).all():
            _fail(f"render PNG contains non-finite data: {path}")


def _verify_render_outputs(
    contract: Mapping[str, Any],
    *,
    containment_root: str | Path | None = None,
) -> dict[str, Any]:
    model = _authority_path(
        contract["model_path"],
        containment_root=containment_root,
        label="render model",
        must_exist=True,
        directory=True,
    )
    static = contract["static"]
    camera = static["camera"]
    count = int(contract["static_verification"]["image_count"])
    iteration = _contract_iteration(contract)
    root = _authority_path(
        model / "train" / f"ours_{iteration}",
        containment_root=containment_root,
        label="render output root",
        must_exist=True,
        directory=True,
    )
    renders = sorted((root / "renders").glob("*.png"))
    gts = sorted((root / "gt").glob("*.png"))
    nvs = sorted((root / "nvs").glob("*.png"))
    if containment_root is not None:
        renders = [
            _authority_path(
                path,
                containment_root=containment_root,
                label=f"fixed render PNG {path.name}",
                must_exist=True,
                directory=False,
            )
            for path in renders
        ]
        gts = [
            _authority_path(
                path,
                containment_root=containment_root,
                label=f"GT render PNG {path.name}",
                must_exist=True,
                directory=False,
            )
            for path in gts
        ]
        nvs = [
            _authority_path(
                path,
                containment_root=containment_root,
                label=f"NVS render PNG {path.name}",
                must_exist=True,
                directory=False,
            )
            for path in nvs
        ]
    if len(renders) != count or len(gts) != count or len(nvs) != (2 * count - 1):
        _fail(f"render counts failed: fixed={len(renders)}, gt={len(gts)}, nvs={len(nvs)}, expected={count}/{count}/{2 * count - 1}")
    width, height = int(camera["width"]), int(camera["height"])
    _check_pngs([*renders, *gts, *nvs], width, height)
    pose_path = _authority_path(
        root / "videos" / "nvs_camera_poses.json",
        containment_root=containment_root,
        label="NVS pose JSON",
        must_exist=True,
        directory=False,
    )
    pose = _load_json(pose_path, "NVS pose JSON")
    if pose.get("pose_mode") != "adjacent_midpoint_slerp" or pose.get("input_count") != count or pose.get("nvs_count") != 2 * count - 1:
        _fail("NVS pose contract count/mode mismatch")
    if not _all_finite(pose):
        _fail("NVS pose contract contains non-finite values")
    endpoints_decoded = 0
    endpoints_bytes = 0
    for index in range(count):
        fixed = renders[index].read_bytes()
        endpoint = nvs[2 * index].read_bytes()
        if fixed == endpoint:
            endpoints_bytes += 1
        try:
            import cv2

            if cv2.imread(str(renders[index]), cv2.IMREAD_UNCHANGED).tobytes() == cv2.imread(str(nvs[2 * index]), cv2.IMREAD_UNCHANGED).tobytes():
                endpoints_decoded += 1
        except ImportError as exc:
            _fail(f"endpoint verification requires OpenCV: {exc}")
    if endpoints_decoded != count:
        _fail(f"fixed/NVS endpoint decoded equality failed: {endpoints_decoded}/{count}")
    return {
        "structural_pass": True,
        "fixed_render_count": len(renders),
        "fixed_gt_count": len(gts),
        "nvs_png_count": len(nvs),
        "nvs_pose_json": str(pose_path),
        "nvs_pose_sha256": sha256_file(pose_path),
        "endpoint_decoded_pixel_equal_count": endpoints_decoded,
        "endpoint_file_byte_equal_count": endpoints_bytes,
        "training_views_only": True,
        "render_root": str(root),
    }


# Conversion/evaluation paths are resolved from the per-run authority manifest.








def _write_stage_output(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )










def _training_result_path(root: Path) -> Path:
    return root / "result.json"


def _verify_training_evidence(
    path: Path,
    contract: Mapping[str, Any],
    route: Path,
    *,
    containment_root: str | Path | None = None,
) -> dict[str, Any]:
    evidence_root = _output_path(
        path,
        route,
        "training evidence root",
        must_exist=True,
        containment_root=containment_root,
    )
    result_path = _training_result_path(evidence_root)
    request_path = evidence_root / "request.json"
    argv_path = evidence_root / "argv.json"
    result = _load_json(result_path, "training executor result")
    request = _load_json(request_path, "training executor request")
    argv_record = _load_json(argv_path, "training executor argv")
    if result.get("stage") != "training" or result.get("exit_code") != 0 or result.get("structural", {}).get("structural_pass") is not True:
        _fail("render requires a successful structural training executor result")
    if result.get("model_path") != contract.get("model_path"):
        _fail("training result model_path differs from render plan")
    for key in ("plan_path", "plan_sha256", "static_contract_path", "static_contract_sha256", "source_path", "workload_profile", "iterations", "render_iteration"):
        if request.get(key) != contract.get(key):
            _fail(f"training request {key} differs from the verified plan")
    if request.get("stage") != "training" or request.get("model_path") != contract.get("model_path") or request.get("shell") is not False:
        _fail("training request identity is invalid")
    request_argv = request.get("argv")
    stored_argv = argv_record.get("argv")
    if not isinstance(request_argv, list) or request_argv != stored_argv or argv_record.get("shell") is not False:
        _fail("training request/argv identity is invalid")
    if result.get("request_path") != str(request_path) or result.get("argv_path") != str(argv_path):
        _fail("training result does not bind its request and argv artifacts")
    reference = result.get("static_input_reference")
    expected_reference = {
        "source_path": contract["source_path"],
        "plan_sha256": contract["plan_sha256"],
        "static_contract_sha256": contract["static_contract_sha256"],
    }
    if reference != expected_reference:
        _fail("training result static input reference differs from the verified plan")
    if request.get("route_code_identity_sha256") != contract["plan"].get("route_code_identity_sha256"):
        _fail("training request producer code identity differs from the plan")
    structural = _verify_training_outputs(contract, containment_root=containment_root)
    return {
        "root": str(evidence_root),
        "result": result,
        "request": request,
        "result_path": str(result_path),
        "request_path": str(request_path),
        "argv_path": str(argv_path),
        "result_sha256": sha256_file(result_path),
        "request_sha256": sha256_file(request_path),
        "argv_sha256": sha256_file(argv_path),
        "structural": structural,
    }


def execute_stage(
    *,
    plan_path: str | Path,
    static_contract_path: str | Path,
    evidence_root: str | Path,
    stage: str,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    training_evidence_root: str | Path | None = None,
    native_evidence_root: str | Path | None = None,
    conversion_evidence_root: str | Path | None = None,
    authority_manifest_path: str | Path | None = None,
    isolated_render_snapshot: bool = False,
    dry_run: bool = False,
    validate_only: bool = False,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    if stage == "conversion":
        if isolated_render_snapshot or training_evidence_root is not None:
            _fail("conversion stage does not accept render/training recovery options")
        return _execute_conversion_stage(
            plan_path=plan_path,
            static_contract_path=static_contract_path,
            evidence_root=evidence_root,
            route_root=route_root,
            containment_root=containment_root,
            authority_manifest_path=authority_manifest_path,
            dry_run=dry_run,
            validate_only=validate_only,
            runner=runner,
        )
    if stage == "converted-eval":
        if isolated_render_snapshot or training_evidence_root is not None:
            _fail("converted-eval stage does not accept native-training recovery options")
        return _execute_converted_evaluation_stage(
            plan_path=plan_path,
            static_contract_path=static_contract_path,
            evidence_root=evidence_root,
            route_root=route_root,
            containment_root=containment_root,
            authority_manifest_path=authority_manifest_path,
            native_evidence_root=native_evidence_root,
            conversion_evidence_root=conversion_evidence_root,
            dry_run=dry_run,
            validate_only=validate_only,
            runner=runner,
        )
    if stage == "converted-eval-postprocess":
        if isolated_render_snapshot is not False or training_evidence_root is not None:
            _fail("converted-eval-postprocess does not accept native-training recovery options")
        return _execute_converted_evaluation_postprocess_stage(
            plan_path=plan_path,
            static_contract_path=static_contract_path,
            evidence_root=evidence_root,
            route_root=route_root,
            containment_root=containment_root,
            authority_manifest_path=authority_manifest_path,
            native_evidence_root=native_evidence_root,
            dry_run=dry_run,
            validate_only=validate_only,
        )
    if stage not in {"training", "render"}:
        _fail("stage must be training or render")
    if isolated_render_snapshot and stage != "render":
        _fail("--isolated-render-snapshot is valid only for the render stage")
    route = Path(route_root).resolve()
    root = _output_path(evidence_root, route, "evidence root", containment_root=containment_root)
    if root.exists():
        _fail(f"evidence root must be absent before execution: {root}")
    recovery = stage == "render" and isolated_render_snapshot
    contract = validate_plan_and_static(
        plan_path,
        static_contract_path,
        route_root=route,
        containment_root=containment_root,
        allow_existing_model=stage == "render",
        allow_producer_code_drift=recovery,
    )
    training_evidence: dict[str, Any] | None = None
    effective_contract: dict[str, Any] = dict(contract)
    snapshot_path: Path | None = None
    snapshot_manifest_path: Path | None = None
    if stage == "render":
        if training_evidence_root is None:
            _fail("render requires --training-evidence-root")
        training_evidence = _verify_training_evidence(
            Path(training_evidence_root),
            contract,
            route,
            containment_root=containment_root,
        )
        model = Path(contract["model_path"])
        iteration = _contract_iteration(contract)
        if not recovery:
            for relative in (f"train/ours_{iteration}", f"test/ours_{iteration}"):
                if (model / relative).exists():
                    _fail(f"render output directory already exists and cannot be overwritten: {model / relative}")
        else:
            snapshot_path = root / "render_model_snapshot"
            snapshot_manifest_path = snapshot_path / "snapshot_manifest.json"
            if snapshot_path.exists() or snapshot_path.is_symlink():
                _fail(f"isolated render snapshot must be absent: {snapshot_path}")
            effective_contract = _replace_model_path(contract, snapshot_path)
    command = build_training_command(effective_contract, route_root=route) if stage == "training" else build_render_command(effective_contract, route_root=route)
    env = _environment(effective_contract)
    request = {
        "schema": "longsplat-smoke-executor-request-v1",
        "stage": stage,
        "plan_path": contract["plan_path"],
        "plan_sha256": contract["plan_sha256"],
        "static_contract_path": contract["static_contract_path"],
        "static_contract_sha256": contract["static_contract_sha256"],
        "source_path": effective_contract["source_path"],
        "model_path": effective_contract["model_path"],
        "workload_profile": effective_contract["workload_profile"],
        "iterations": effective_contract["iterations"],
        "render_iteration": effective_contract["render_iteration"],
        "route_root": str(route),
        "route_code_identity_sha256": contract["plan"].get("route_code_identity_sha256"),
        "producer_route_code_identity_sha256": contract["plan"].get("route_code_identity_sha256"),
        "consumer_executor_sha256": sha256_file(Path(__file__).resolve()),
        "consumer_render_identity": contract.get("render_consumer_identity"),
        "backend_path": effective_contract["backend_path"],
        "backend_env": effective_contract["backend_env"],
        "argv": command,
        "shell_escaped_command_display": shlex.join(command),
        "shell": False,
        "cwd": str(route),
        "environment": _redacted_environment(env),
        "training_evidence_root": None if training_evidence is None else training_evidence["root"],
        "isolated_render_snapshot": recovery,
        "original_training_model_path": None if stage != "render" else contract["model_path"],
        "isolated_render_model_path": None if snapshot_path is None else str(snapshot_path),
        "snapshot_reason": "preserve_partial_render_without_overwrite" if recovery else None,
        "snapshot_manifest_path": None if snapshot_manifest_path is None else str(snapshot_manifest_path),
    }
    if dry_run or validate_only:
        return {"validated": True, "stage": stage, "request": request, "command": command}
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        _fail(f"evidence root was created concurrently: {root}")
    if recovery:
        if training_evidence is None or snapshot_path is None:
            _fail("isolated render snapshot metadata is incomplete")
        snapshot_path, _, snapshot_manifest_path = _snapshot_render_model(
            contract=contract,
            training_evidence=training_evidence,
            evidence_root=root,
            route=route,
            containment_root=containment_root,
        )
        effective_contract = _replace_model_path(contract, snapshot_path)
        command = build_render_command(effective_contract, route_root=route)
        request["model_path"] = str(snapshot_path)
        request["argv"] = command
        request["shell_escaped_command_display"] = shlex.join(command)
        request["isolated_render_model_path"] = str(snapshot_path)
        request["snapshot_manifest_path"] = str(snapshot_manifest_path)
        request["snapshot_manifest_sha256"] = sha256_file(snapshot_manifest_path)
    _write_json_once(root / "request.json", request)
    _write_json_once(root / "argv.json", {"argv": command, "shell_escaped_command_display": shlex.join(command), "shell": False})
    completed = runner(
        command,
        cwd=str(route),
        env=env,
        shell=False,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = str(getattr(completed, "stdout", "") or "")
    stderr = str(getattr(completed, "stderr", "") or "")
    exit_code = int(getattr(completed, "returncode", 1))
    (root / "stdout.log").write_text(stdout, encoding="utf-8")
    (root / "stderr.log").write_text(stderr, encoding="utf-8")
    result: dict[str, Any] = {
        "schema": "longsplat-smoke-executor-result-v1",
        "stage": stage,
        "exit_code": exit_code,
        "model_path": effective_contract["model_path"],
        "original_training_model_path": None if stage != "render" else contract["model_path"],
        "isolated_render_model_path": None if snapshot_path is None else str(snapshot_path),
        "snapshot_manifest_path": None if snapshot_manifest_path is None else str(snapshot_manifest_path),
        "snapshot_manifest_sha256": None if snapshot_manifest_path is None else sha256_file(snapshot_manifest_path),
        "producer_route_code_identity_sha256": contract["plan"].get("route_code_identity_sha256"),
        "consumer_executor_sha256": sha256_file(Path(__file__).resolve()),
        "consumer_render_identity": contract.get("render_consumer_identity"),
        "workload_profile": effective_contract["workload_profile"],
        "iterations": effective_contract["iterations"],
        "render_iteration": effective_contract["render_iteration"],
        "stdout_path": str(root / "stdout.log"),
        "stderr_path": str(root / "stderr.log"),
        "stdout_sha256": sha256_file(root / "stdout.log"),
        "stderr_sha256": sha256_file(root / "stderr.log"),
        "request_path": str(root / "request.json"),
        "argv_path": str(root / "argv.json"),
        "gpu_invoked": True,
        "static_input_reference": {
            "source_path": effective_contract["source_path"],
            "plan_sha256": effective_contract["plan_sha256"],
            "static_contract_sha256": effective_contract["static_contract_sha256"],
        },
    }
    if exit_code == 0:
        try:
            result["structural"] = (
                _verify_training_outputs(effective_contract, containment_root=containment_root)
                if stage == "training"
                else _verify_render_outputs(effective_contract, containment_root=containment_root)
            )
            if stage == "render" and snapshot_manifest_path is not None:
                _verify_snapshot_manifest(snapshot_manifest_path, containment_root=containment_root)
        except (SmokeExecutorBlocked, CoverageSmokeBlocked) as exc:
            result["structural"] = {"structural_pass": False, "reason": str(exc)}
    else:
        result["structural"] = {"structural_pass": False, "reason": "child process returned nonzero"}
    result["structural_pass"] = bool(result["structural"].get("structural_pass"))
    _write_json_once(root / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute one frozen LongSplat GPU smoke stage")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--static-contract", type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("training", "render", "conversion", "converted-eval", "converted-eval-postprocess"))
    parser.add_argument("--route-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--containment-root", type=Path, help="verified dynamic run/evidence root; never defaults to route/outputs in production")
    parser.add_argument("--authority-manifest", type=Path)
    parser.add_argument("--training-evidence-root", type=Path)
    parser.add_argument("--native-evidence-root", type=Path)
    parser.add_argument("--isolated-render-snapshot", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.stage in {"training", "render"} and (args.plan is None or args.static_contract is None):
        parser.error("--plan and --static-contract are required for training/render")
    if args.stage in {"conversion", "converted-eval", "converted-eval-postprocess"} and args.authority_manifest is None:
        parser.error("--authority-manifest is required for conversion/evaluation stages")
    try:
        result = execute_stage(
            plan_path=args.plan,
            static_contract_path=args.static_contract,
            evidence_root=args.evidence_root,
            stage=args.stage,
            route_root=args.route_root,
            containment_root=args.containment_root,
            training_evidence_root=args.training_evidence_root,
            native_evidence_root=args.native_evidence_root,
            authority_manifest_path=args.authority_manifest,
            isolated_render_snapshot=args.isolated_render_snapshot,
            dry_run=args.dry_run,
            validate_only=args.validate_only,
        )
    except SmokeExecutorBlocked as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if result.get("validated"):
        return 0
    if result.get("stage") in {"converted-eval", "converted-eval-postprocess"}:
        return 0 if result.get("SAME_CAMERA_VISUAL_PASS") in {"pass", "needs_review"} else 1
    if result.get("stage") == "conversion":
        return 0 if result.get("exit_code") == 0 and result.get("structural_pass") else 1
    return 0 if result.get("exit_code") == 0 and result.get("structural_pass") else 1


def _conversion_compat_authority(paths: Mapping[str, Path], authority: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build the small in-memory adapter used by legacy helper tests."""

    if authority is not None and "manifest" in authority:
        return dict(authority)
    from .authority_manifest import conversion_profile

    profile = conversion_profile("standard30000-v1")
    return {
        "manifest": {
            "training_model": {"path": str(Path(paths["model"]).resolve())},
            "training_input": {"path": str(Path(paths["source"]).resolve())},
        },
        "profile": profile,
        "profile_id": "standard30000-v1",
        # Compatibility only: legacy helper tests must provide their explicit
        # regression metadata; production execution always loads the manifest.
        "camera_count": int(authority["camera_count"]),
        "camera_order": [],
        "camera_dimensions": {"width": 1, "height": 1},
        "source_video_sha256": "",
        "status": {"held_out": False, "training_views_only": True},
    }


def _conversion_source_relatives(iteration: int | None = None) -> tuple[str, ...]:
    from .conversion_executor import source_relatives
    from .authority_manifest import conversion_profile

    if iteration is None:
        iteration = conversion_profile("standard30000-v1")["checkpoint_iteration"]
    return source_relatives(iteration=iteration)


def _conversion_paths(
    route_root: str | Path,
    authority_manifest_path: str | Path | None = None,
    containment_root: str | Path | None = None,
) -> dict[str, Path]:
    from .authority_manifest import load_authority_manifest

    route = Path(route_root).resolve()
    if authority_manifest_path is None:
        raise SmokeExecutorBlocked("conversion/evaluation requires --authority-manifest")
    authority = load_authority_manifest(
        authority_manifest_path,
        route_root=route,
        containment_root=containment_root,
    )
    model = Path(str(authority["manifest"]["training_model"]["path"])).resolve()
    source = Path(str(authority["manifest"]["training_input"]["path"])).resolve()
    nested = route / "third_party/LongSplat"
    if nested.is_symlink():
        raise SmokeExecutorBlocked(f"nested LongSplat is symlinked: {nested}")
    backend_value = os.environ.get("LONGSPLAT_BACKEND_PYTHON")
    backend = Path(backend_value) if backend_value else route.parent.parent / "backend-envs/longsplat-cu128/bin/python"
    return {
        "route": route,
        "source": source,
        "model": model,
        "nested": nested.resolve(),
        "backend_env": backend.resolve().parent.parent,
        "backend_python": backend.resolve(),
        "plan": Path(str(authority["manifest"]["plan"]["path"])),
        "static": Path(str(authority["manifest"]["static_contract"]["path"])),
    }


def _build_conversion_argv(
    *,
    paths: Mapping[str, Path],
    evidence_root: Path,
    snapshot: Path,
    authority: Mapping[str, Any] | None = None,
    authority_manifest_path: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    from .conversion_executor import build_conversion_argv_for_paths

    profile_id = "standard30000-v1" if authority is None else str(authority["profile_id"])
    return build_conversion_argv_for_paths(
        route=Path(paths["route"]),
        source=Path(paths["source"]),
        model=Path(paths["model"]),
        nested=Path(paths["nested"]),
        backend_python=Path(paths["backend_python"]),
        evidence_root=Path(evidence_root),
        snapshot=Path(snapshot),
        profile_id=profile_id,
    )


def _create_conversion_snapshot(
    *,
    paths: Mapping[str, Path],
    evidence_root: Path,
    authority: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    from .conversion_executor import _snapshot

    return _snapshot(authority=_conversion_compat_authority(paths, authority), evidence_root=Path(evidence_root))


def _verify_conversion_snapshot(
    *,
    snapshot: Path,
    manifest_path: Path,
    allow_converted_output: bool,
) -> dict[str, Any]:
    from .conversion_executor import ConversionExecutorBlocked, verify_snapshot

    try:
        return verify_snapshot(Path(snapshot), Path(manifest_path), allow_converted_output=allow_converted_output)
    except ConversionExecutorBlocked as exc:
        _fail(str(exc))


def _verify_conversion_authority(
    *,
    authority_manifest_path: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
) -> dict[str, Any]:
    from .authority_manifest import load_authority_manifest

    return load_authority_manifest(
        authority_manifest_path,
        route_root=route_root,
        containment_root=containment_root,
    )


def _execute_conversion_stage(
    *,
    plan_path: str | Path | None = None,
    static_contract_path: str | Path | None = None,
    authority_manifest_path: str | Path | None = None,
    evidence_root: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    dry_run: bool,
    validate_only: bool,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    if authority_manifest_path is None:
        _fail("conversion stage requires --authority-manifest; plan/static are run inputs, not authority")
    from .conversion_executor import ConversionExecutorBlocked, execute_conversion

    try:
        return execute_conversion(
            authority_manifest_path=authority_manifest_path,
            evidence_root=evidence_root,
            route_root=route_root,
            containment_root=containment_root,
            dry_run=dry_run,
            validate_only=validate_only,
            runner=runner,
        )
    except ConversionExecutorBlocked as exc:
        _fail(str(exc))


def _execute_converted_evaluation_stage(
    *,
    plan_path: str | Path | None = None,
    static_contract_path: str | Path | None = None,
    authority_manifest_path: str | Path | None = None,
    evidence_root: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    native_evidence_root: str | Path | None = None,
    conversion_evidence_root: str | Path | None = None,
    dry_run: bool,
    validate_only: bool,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    if authority_manifest_path is None:
        _fail("converted-eval stage requires --authority-manifest")
    from .conversion_executor import ConversionExecutorBlocked, execute_evaluation

    try:
        return execute_evaluation(
            authority_manifest_path=authority_manifest_path,
            evidence_root=evidence_root,
            route_root=route_root,
            containment_root=containment_root,
            conversion_evidence_root=conversion_evidence_root,
            dry_run=dry_run,
            validate_only=validate_only,
            runner=runner,
        )
    except ConversionExecutorBlocked as exc:
        _fail(str(exc))


def _execute_converted_evaluation_postprocess_stage(
    *,
    plan_path: str | Path | None = None,
    static_contract_path: str | Path | None = None,
    authority_manifest_path: str | Path | None = None,
    evidence_root: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    native_evidence_root: str | Path | None = None,
    dry_run: bool,
    validate_only: bool,
) -> dict[str, Any]:
    if authority_manifest_path is None:
        _fail("converted-eval-postprocess requires --authority-manifest")
    from .authority_manifest import load_authority_manifest
    from .conversion_executor import ConversionExecutorBlocked, output_path, postprocess_evaluation

    try:
        authority = load_authority_manifest(
            authority_manifest_path,
            route_root=route_root,
            containment_root=containment_root,
        )
        root = output_path(evidence_root, route_root, "conversion postprocess evidence root", must_exist=True, containment_root=containment_root)
        snapshot = root / "conversion_model_snapshot"
        result = _load_json(root / "conversion_result.json", "conversion result")
        converted = Path(str(result.get("structural", {}).get("converted_ply_path", snapshot / "converted_3dgs/point_cloud.ply"))).resolve()
        eval_root = root / "same_camera_eval"
        if dry_run or validate_only:
            return {"validated": True, "stage": "converted-eval-postprocess", "camera_count": authority["camera_count"], "held_out": authority["status"]["held_out"]}
        if not eval_root.is_dir() or eval_root.is_symlink():
            _fail(f"completed same-camera evaluator evidence is missing: {eval_root}")
        ab = postprocess_evaluation(authority=authority, eval_root=eval_root, snapshot=snapshot, converted_ply=converted)
        _write_stage_output(eval_root / "ab_metrics.json", ab)
        final = {"stage": "converted-eval-postprocess", "gpu_invoked": False, **ab}
        _write_stage_output(root / "evaluation_result.json", final)
        return final
    except ConversionExecutorBlocked as exc:
        _fail(str(exc))


if __name__ == "__main__":
    if __package__ in (None, ""):
        # ``./dev.sh python scripts/longsplat/smoke_executor.py`` has no
        # package context; rerun the import through the route package path.
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.longsplat.smoke_executor import main as _main
    else:
        _main = main
    raise SystemExit(_main())
