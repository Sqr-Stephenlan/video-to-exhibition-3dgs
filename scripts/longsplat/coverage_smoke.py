"""CPU-only planning and validation for the LongSplat coverage smoke.

The nested trainer owns camera selection and remains the only producer of
sampling events.  This module only turns the audited stack semantics into an
immutable plan and validates the resulting append-only telemetry.  It has no
torch, CUDA, or random dependency and never starts a training process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .frame_contract import FrameSelectionConfig
from .pipeline_contract import PipelineBlocked, resolve_contained_path, resolve_containment_root, sha256_file


COVERAGE_SMOKE_SCHEMA = "coverage-smoke-v1"
MIN_COMPLETE_COVERAGE_ROUNDS = 2
TELEMETRY_SCHEMA = "camera-sampling-telemetry-v1"
TELEMETRY_EVENTS_NAME = "camera_sampling_telemetry-v1.jsonl"
TELEMETRY_SUMMARY_NAME = "camera_sampling_telemetry-v1.json"
_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class CoverageSmokeBlocked(PipelineBlocked):
    """The coverage plan or immutable sampling evidence is not admissible."""


def _fail(message: str) -> None:
    raise CoverageSmokeBlocked(message)


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        _fail(f"{label} is missing or symlinked: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not valid JSON: {source}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object: {source}")
    return value


def _basename(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        _fail(f"{field} must be a safe basename")
    if Path(value).suffix.lower() not in _IMAGE_EXTENSIONS:
        _fail(f"{field} must have a supported image extension")
    return value


def load_camera_contract(path: str | Path) -> dict[str, Any]:
    """Read the explicit, ordered training-camera identity contract."""

    source = Path(path).resolve()
    contract = _load_json(source, "camera contract")
    names_value = contract.get("frame_names")
    if not isinstance(names_value, list) or not names_value:
        _fail("camera contract frame_names are required")
    names = [_basename(value, "camera contract frame name") for value in names_value]
    if len(names) != len(set(names)):
        _fail("camera contract frame_names must be unique")
    if contract.get("frame_count") != len(names):
        _fail("camera contract frame_count differs from ordered frame_names")
    camera = contract.get("camera")
    if not isinstance(camera, Mapping):
        _fail("camera contract camera record is required")
    width = camera.get("width")
    height = camera.get("height")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        _fail("camera contract dimensions must be positive integers")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "contract_sha256": contract.get("contract_sha256"),
        "camera_names": names,
        "camera_count": len(names),
        "camera": {"width": width, "height": height, "model": camera.get("model")},
        "source_video_sha256": contract.get("source_video_sha256"),
    }


def load_frame_selection(path: str | Path) -> dict[str, Any]:
    """Read only the immutable selector cap and selected-frame provenance."""

    source = Path(path).resolve()
    selection = _load_json(source, "frame-selection contract")
    config = selection.get("config")
    if not isinstance(config, Mapping):
        _fail("frame-selection config is required")
    max_frames = config.get("max_frames")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
        _fail("frame-selection config.max_frames must be a positive integer")
    frames = selection.get("frames")
    if not isinstance(frames, list) or not frames:
        _fail("frame-selection frames are required")
    selected = [item for item in frames if isinstance(item, Mapping) and item.get("selected") is True]
    duration = selection.get("duration_sec")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(float(duration)) or float(duration) <= 0:
        _fail("frame-selection duration_sec must be positive and finite")
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "max_frames": max_frames,
        "selected_count": len(selected),
        "frame_count": len(frames),
        "duration_sec": float(duration),
        "binding": dict(selection.get("binding", {})) if isinstance(selection.get("binding"), Mapping) else {},
    }


def _camera_names(camera_names: Sequence[str] | None, count: int) -> list[str]:
    if camera_names is None:
        return [f"camera_{index:06d}" for index in range(count)]
    names = [str(value) for value in camera_names]
    if len(names) != count:
        _fail("camera_names length must equal active_camera_count")
    if not all(names):
        _fail("camera_names must be non-empty")
    if len(names) != len(set(names)):
        _fail("camera_names must be unique")
    return names


def _predicted_exposure(count: int, iterations: int) -> dict[str, int | float]:
    rounds, partial = divmod(iterations, count)
    low = rounds
    high = rounds + (1 if partial else 0)
    return {
        "complete_rounds": rounds,
        "partial_round_size": partial,
        "min_exposure_count": low,
        "max_exposure_count": high,
        "predicted_unique_camera_count": count if iterations >= count else iterations,
        "predicted_zero_exposure_camera_count": max(0, count - iterations),
        "predicted_coverage_fraction": min(1.0, iterations / count),
    }


def _anchor_schedule(iterations: int) -> list[int]:
    # This is the external fixed-pose fork's actual condition:
    # iteration < update_until, iteration > update_from, and modulo interval.
    return [iteration for iteration in range(1, iterations + 1) if iteration < iterations and iteration > 0 and iteration % 100 == 0]


def _coverage_policy_values(
    *,
    active_camera_count: int,
    frame_selector_max_frames: int,
    requested_iterations: int | None,
) -> dict[str, Any]:
    """Derive the versioned local policy from the current camera/selector inputs."""

    minimum = MIN_COMPLETE_COVERAGE_ROUNDS * active_camera_count
    upper = MIN_COMPLETE_COVERAGE_ROUNDS * frame_selector_max_frames
    iterations = minimum if requested_iterations is None else requested_iterations
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        _fail("requested_iterations must be a positive integer")
    reasons: list[str] = []
    if active_camera_count > frame_selector_max_frames:
        reasons.append("active_camera_count_exceeds_frame_selector_max_frames")
    if iterations < minimum:
        reasons.append("requested_iterations_below_two_complete_camera_rounds")
    if iterations > upper:
        reasons.append("requested_iterations_exceeds_selector_derived_upper_bound")
    return {
        "minimum_iterations": minimum,
        "selector_derived_upper_bound": upper,
        "requested_iterations": iterations,
        "reasons": reasons,
        "computed_pass": not reasons,
    }


def plan_coverage_smoke(
    *,
    active_camera_count: int,
    camera_names: Sequence[str] | None = None,
    frame_selector_max_frames: int = FrameSelectionConfig().max_frames,
    requested_iterations: int | None = None,
    camera_contract_path: str | Path | None = None,
    frame_selection_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the local two-round diagnostic plan from dynamic camera count.

    The two-round target and the selector-derived ceiling are local policy.
    The stack refill and floor/ceil exposure facts are the audited confirmed
    semantics; no official adaptive iteration formula is asserted here.
    """

    if isinstance(active_camera_count, bool) or not isinstance(active_camera_count, int) or active_camera_count <= 0:
        _fail("active_camera_count must be a positive integer")
    if isinstance(frame_selector_max_frames, bool) or not isinstance(frame_selector_max_frames, int) or frame_selector_max_frames <= 0:
        _fail("frame_selector_max_frames must be a positive integer")
    names = _camera_names(camera_names, active_camera_count)
    policy = _coverage_policy_values(
        active_camera_count=active_camera_count,
        frame_selector_max_frames=frame_selector_max_frames,
        requested_iterations=requested_iterations,
    )
    minimum = int(policy["minimum_iterations"])
    upper = int(policy["selector_derived_upper_bound"])
    iterations = int(policy["requested_iterations"])
    reasons = list(policy["reasons"])
    prediction = _predicted_exposure(active_camera_count, iterations)
    anchors = _anchor_schedule(iterations)
    computed_pass = not reasons
    return {
        "schema_version": COVERAGE_SMOKE_SCHEMA,
        "plan_status": "plan_validated" if computed_pass else "structured_stop",
        "computed_pass": computed_pass,
        "gpu_invoked": False,
        "training_invoked": False,
        "coverage_smoke_visual_decision": "not_run",
        "formal_auto_release": False,
        "active_camera_count": active_camera_count,
        "active_camera_order": names,
        "requested_iterations": iterations,
        "frame_selector_max_frames": frame_selector_max_frames,
        "formula": {
            "confirmed": "one camera per iteration; phase-local random pop without replacement; stack refill",
            "exposure_rule": "floor(T/N) or ceil(T/N) per camera for fixed N and T",
            "local_policy_target": "at least two complete camera coverage rounds",
            "local_policy_minimum_iterations": minimum,
            "selector_derived_upper_bound": upper,
            "official_formula_status": "not_specified",
        },
        "confirmed_facts": {
            "sampling_semantics": "phase-local random pop without replacement with stack refill",
            "exposure_distribution": "floor(T/N) or ceil(T/N)",
            "telemetry_scope": "observer only; no random calls and no selection/math mutation",
            "historical_log_camera_set": "unavailable unless camera-sampling telemetry was emitted",
        },
        "policy_class": "local_evidence_driven_early_diagnostic",
        "policy_status": "local_policy",
        "local_policy": {
            "minimum_complete_rounds": 2,
            "selector_cap_source": "frame-selection-v1.config.max_frames",
            "purpose": "coverage smoke is an early visual diagnostic, not formal training or acceptance",
            "not_official_formula": True,
        },
        "experiment_required": [
            "official adaptive iteration formula by camera count/video duration/GPU memory is not specified",
            "visual thresholds and workload scaling require a future versioned experiment",
            "coverage smoke does not replace formal training or manual visual review",
        ],
        "coverage_prediction": prediction,
        "densification": {
            "fork_semantics": "anchor_growing via adjust_anchor; external fixed-pose path does not call classic densify_and_clone/split/reset_opacity",
            "start_stat": 0,
            "update_from": 0,
            "update_interval": 100,
            "update_until": iterations,
            "adjust_anchor_iterations": anchors,
            "require_purning": False,
            "densify_occlusion_called": False,
            "opacity_reset": "not_called_in_external_fixed_pose",
            "interpretation": "expected schedule from current fork code, not a quality claim",
        },
        "gpu_workload_estimate": {
            "active_camera_count": active_camera_count,
            "iterations": iterations,
            "camera_exposures": iterations,
            "estimated_complete_rounds": prediction["complete_rounds"],
            "gpu_invoked": False,
            "estimate_status": "planning_only_no_runtime_measurement",
        },
        "training_state_resume": "save/checkpoint artifacts support quality/render/conversion inspection but not bit-exact training resume",
        "profile_boundary": {
            "smoke100-v1": "local structural/numeric smoke only; rough visual failure cannot reject video/SfM by itself",
            "formal30000-v1": "local versioned policy, not the official final training standard",
            "standard30000-v1": "local conversion policy validated by the prior accepted B chain",
            "official_baseline": "official code baseline 30000, default post_iter 20000, native save commonly around 50000",
            "official_converter_defaults": "optimization 10000 and prune 0.6",
        },
        "external_route_semantics": "fork-specific external COLMAP fixed-pose RGB-only, depth disabled, lazy MASt3R, dynamic camera identity; not official unposed path",
        "camera_contract_path": None if camera_contract_path is None else str(Path(camera_contract_path).resolve()),
        "frame_selection_path": None if frame_selection_path is None else str(Path(frame_selection_path).resolve()),
        "stop": {
            "required": bool(reasons),
            "reasons": reasons,
            "next_action": "record a new append-only attempt or revise only through a future versioned policy" if reasons else "coverage plan may be validated; GPU execution remains separately authorized",
        },
    }


def validate_coverage_smoke_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate plan arithmetic and the no-GPU/formal boundary."""

    if plan.get("schema_version") != COVERAGE_SMOKE_SCHEMA:
        _fail(f"coverage plan schema must be {COVERAGE_SMOKE_SCHEMA}")
    count = plan.get("active_camera_count")
    iterations = plan.get("requested_iterations")
    if not isinstance(count, int) or count <= 0 or not isinstance(iterations, int) or iterations <= 0:
        _fail("coverage plan camera count and iterations must be positive integers")
    order = plan.get("active_camera_order")
    if not isinstance(order, list) or len(order) != count or len(set(order)) != count:
        _fail("coverage plan camera order must be unique and exact")
    formula = plan.get("formula")
    if not isinstance(formula, Mapping) or formula.get("official_formula_status") != "not_specified":
        _fail("coverage plan must distinguish local policy from unspecified official formula")
    selector_cap = plan.get("frame_selector_max_frames")
    if selector_cap is None:
        # Historical v1/v2 evidence did not expose the cap separately; retain
        # read-only compatibility while new executor plans bind it explicitly.
        upper = formula.get("selector_derived_upper_bound")
        if isinstance(upper, bool) or not isinstance(upper, int) or upper <= 0 or upper % MIN_COMPLETE_COVERAGE_ROUNDS:
            _fail("coverage plan selector cap is missing or invalid")
        selector_cap = upper // MIN_COMPLETE_COVERAGE_ROUNDS
    if isinstance(selector_cap, bool) or not isinstance(selector_cap, int) or selector_cap <= 0:
        _fail("coverage plan frame selector cap must be a positive integer")
    policy = _coverage_policy_values(
        active_camera_count=count,
        frame_selector_max_frames=selector_cap,
        requested_iterations=iterations,
    )
    if formula.get("local_policy_minimum_iterations") != policy["minimum_iterations"]:
        _fail("coverage plan local minimum is inconsistent with dynamic camera count")
    if formula.get("selector_derived_upper_bound") != policy["selector_derived_upper_bound"]:
        _fail("coverage plan selector upper bound is inconsistent with immutable selector cap")
    if plan.get("frame_selector_max_frames") is not None and plan.get("frame_selector_max_frames") != selector_cap:
        _fail("coverage plan selector cap binding is inconsistent")
    prediction = _predicted_exposure(count, iterations)
    if plan.get("coverage_prediction") != prediction:
        _fail("coverage plan exposure prediction is inconsistent")
    if plan.get("gpu_invoked") is not False or plan.get("training_invoked") is not False:
        _fail("coverage plan must remain CPU-only")
    if plan.get("formal_auto_release") is not False:
        _fail("coverage plan cannot automatically release formal training")
    expected_pass = not policy["reasons"]
    if plan.get("stop", {}).get("reasons", []) != policy["reasons"]:
        _fail("coverage plan structured stop is inconsistent with dynamic policy")
    if plan.get("computed_pass") is not expected_pass:
        _fail("coverage plan computed_pass disagrees with structured stop")
    estimate = plan.get("gpu_workload_estimate")
    if isinstance(estimate, Mapping) and estimate.get("iterations") != iterations:
        _fail("coverage plan workload estimate iterations differ")
    densification = plan.get("densification")
    if isinstance(densification, Mapping) and densification.get("update_until") != iterations:
        _fail("coverage plan anchor schedule bound differs from iterations")
    return {
        "schema_version": "coverage-smoke-validation-v1",
        "computed_pass": bool(plan["computed_pass"]),
        "gpu_invoked": False,
        "formal_auto_release": False,
        "active_camera_count": count,
        "iterations": iterations,
        "frame_selector_max_frames": selector_cap,
        "policy_minimum_iterations": policy["minimum_iterations"],
        "policy_upper_bound": policy["selector_derived_upper_bound"],
        "coverage_prediction": prediction,
        "policy_status": plan.get("policy_status"),
        "official_formula_status": formula.get("official_formula_status"),
    }


def validate_sampling_telemetry(
    *,
    model_path: str | Path,
    expected_iterations: int,
    expected_internal_names: Sequence[str],
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate emitted sampling events against the immutable pose contract."""

    model = Path(model_path).resolve()
    if model.is_symlink() or not model.is_dir():
        _fail(f"telemetry model path is missing or symlinked: {model}")
    summary_path = model / TELEMETRY_SUMMARY_NAME
    events_path = model / TELEMETRY_EVENTS_NAME
    if not summary_path.is_file() or summary_path.is_symlink() or not events_path.is_file() or events_path.is_symlink():
        _fail("camera sampling telemetry summary/events are required")
    summary = _load_json(summary_path, "camera sampling telemetry summary")
    if summary.get("schema_version") != TELEMETRY_SCHEMA:
        _fail("camera sampling telemetry schema mismatch")
    if summary.get("selection_policy") != "one_camera_per_iteration_random_pop_without_replacement_per_full_stack":
        _fail("camera sampling telemetry selection policy is not the audited stack policy")
    internal_names = [str(value) for value in expected_internal_names]
    if len(internal_names) != len(set(internal_names)) or not internal_names:
        _fail("expected internal camera names must be unique")
    identity_order = summary.get("active_camera_internal_order")
    if identity_order != internal_names:
        _fail("telemetry internal camera order differs from immutable pose contract")
    if summary.get("iterations") != expected_iterations or summary.get("sampled_iteration_count") != expected_iterations:
        _fail("telemetry iteration count differs from checkpoint contract")
    if expected_contract_sha256 is not None and summary.get("camera_contract_file_sha256") != expected_contract_sha256:
        _fail("telemetry camera contract SHA differs from immutable input")
    records = summary.get("exposure_counts")
    if not isinstance(records, list) or len(records) != len(internal_names):
        _fail("telemetry exposure_counts are incomplete")
    counts: dict[str, int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("camera_internal_name") != internal_names[index] or record.get("camera_contract_index") != index:
            _fail("telemetry exposure record order/identity differs from pose contract")
        count = record.get("exposure_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _fail("telemetry exposure count is invalid")
        counts[internal_names[index]] = count
    if sum(counts.values()) != expected_iterations:
        _fail("telemetry exposure counts do not sum to iterations")
    raw_events = events_path.read_bytes()
    if summary.get("events_sha256") != hashlib.sha256(raw_events).hexdigest() or summary.get("events_size_bytes") != len(raw_events):
        _fail("telemetry event file identity differs from summary")
    observed: dict[str, int] = {name: 0 for name in internal_names}
    lines = raw_events.decode("utf-8").splitlines()
    if len(lines) != expected_iterations:
        _fail("telemetry event line count differs from iterations")
    for expected_iteration, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"telemetry event {expected_iteration} is invalid: {exc}")
        if not isinstance(event, Mapping) or event.get("schema_version") != TELEMETRY_SCHEMA or event.get("iteration") != expected_iteration:
            _fail("telemetry event sequence/schema is invalid")
        name = event.get("camera_internal_name")
        index = event.get("camera_contract_index")
        if not isinstance(name, str) or name not in observed or index != internal_names.index(name):
            _fail("telemetry event camera identity is outside the exact contract")
        observed[name] += 1
        if event.get("exposure_count") != observed[name] or event.get("cumulative_unique_count") != sum(value > 0 for value in observed.values()):
            _fail("telemetry event cumulative exposure is inconsistent")
    if observed != counts:
        _fail("telemetry event exposures differ from summary exposures")
    return {
        "schema_version": TELEMETRY_SCHEMA,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "summary_size_bytes": summary_path.stat().st_size,
        "events_path": str(events_path),
        "events_sha256": hashlib.sha256(raw_events).hexdigest(),
        "events_size_bytes": len(raw_events),
        "active_camera_count": len(internal_names),
        "iterations": expected_iterations,
        "unique_camera_count": sum(value > 0 for value in observed.values()),
        "coverage_fraction": sum(value > 0 for value in observed.values()) / len(observed),
        "exposure_counts": counts,
        "bit_exact_resume": False,
    }


def formal_release_decision(
    *,
    smoke_structural_pass: bool,
    coverage_visual_decision: str | None,
    explicit_visual_review: bool = False,
) -> dict[str, Any]:
    """Keep structural smoke, coverage judgment, and formal release separate."""

    reasons: list[str] = []
    if not smoke_structural_pass:
        reasons.append("smoke100_structural_pass_required")
    if coverage_visual_decision != "pass":
        reasons.append("explicit_coverage_smoke_visual_pass_required")
    if not explicit_visual_review:
        reasons.append("explicit_visual_review_token_required")
    eligible = not reasons
    return {
        "schema_version": "coverage-smoke-formal-gate-v1",
        "smoke100_structural_pass": bool(smoke_structural_pass),
        "coverage_smoke_visual_decision": coverage_visual_decision or "not_run",
        "explicit_visual_review": bool(explicit_visual_review),
        "formal_auto_release": False,
        "formal_release_eligible": eligible,
        "computed_pass": eligible,
        "reasons": reasons,
    }


def _output_dir(
    path: str | Path,
    route_root: Path,
    containment_root: str | Path | None = None,
) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = (Path.cwd() / value).absolute()
    outputs = resolve_containment_root(containment_root) if containment_root is not None else route_root / "outputs"
    if outputs.is_symlink():
        _fail(f"route outputs is symlinked: {outputs}")
    if containment_root is not None:
        try:
            resolved = resolve_contained_path(value, root=outputs, label="coverage output", must_exist=False)
        except PipelineBlocked as exc:
            raise CoverageSmokeBlocked(str(exc)) from exc
        if resolved.exists() or resolved.is_symlink():
            _fail(f"coverage output must be absent for append-only evidence: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.mkdir(parents=False, exist_ok=False)
        return resolved
    resolved = value.resolve(strict=False)
    try:
        relative = resolved.relative_to(outputs.resolve())
    except ValueError as exc:
        raise CoverageSmokeBlocked(f"coverage output must be below route outputs: {resolved}") from exc
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        _fail("coverage output must be a strict descendant of route outputs")
    if value.exists() or value.is_symlink():
        _fail(f"coverage output must be absent for append-only evidence: {value}")
    value.parent.mkdir(parents=True, exist_ok=True)
    value.mkdir(parents=False, exist_ok=False)
    return value.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and validate LongSplat coverage-smoke-v1 without GPU execution")
    parser.add_argument("--camera-contract", required=True, type=Path)
    parser.add_argument("--frame-selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--route-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--containment-root", type=Path, help="verified dynamic run root for output evidence")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        route = args.route_root.resolve()
        camera = load_camera_contract(args.camera_contract)
        selection = load_frame_selection(args.frame_selection)
        plan = plan_coverage_smoke(
            active_camera_count=int(camera["camera_count"]),
            camera_names=camera["camera_names"],
            frame_selector_max_frames=int(selection["max_frames"]),
            requested_iterations=args.iterations,
            camera_contract_path=camera["path"],
            frame_selection_path=selection["path"],
        )
        plan.update(
            {
                "camera_contract_sha256": camera["sha256"],
                "camera_contract_stable_sha256": camera.get("contract_sha256"),
                "frame_selection_sha256": selection["sha256"],
                "frame_selection_selected_count": selection["selected_count"],
                "frame_selection_duration_sec": selection["duration_sec"],
                "camera": camera["camera"],
                "source_video_sha256": camera.get("source_video_sha256") or selection["binding"].get("source_video_sha256"),
            }
        )
        validation = validate_coverage_smoke_plan(plan)
        output = _output_dir(args.output_dir, route, args.containment_root)
        plan_path = output / "coverage-smoke-plan-v1.json"
        validation_path = output / "coverage-smoke-validation-v1.json"
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        validation.update(
            {
                "plan_path": str(plan_path),
                "plan_sha256": sha256_file(plan_path),
                "camera_contract_path": camera["path"],
                "camera_contract_sha256": camera["sha256"],
                "frame_selection_path": selection["path"],
                "frame_selection_sha256": selection["sha256"],
                "gpu_invoked": False,
            }
        )
        validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        result = {
            "schema_version": "coverage-smoke-result-v1",
            "status": "passed" if validation["computed_pass"] else "blocked",
            "computed_pass": bool(validation["computed_pass"]),
            "gpu_invoked": False,
            "training_invoked": False,
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "validation_path": str(validation_path),
            "validation_sha256": sha256_file(validation_path),
            "active_camera_count": camera["camera_count"],
            "iterations": plan["requested_iterations"],
            "coverage_prediction": plan["coverage_prediction"],
            "stop": plan["stop"],
        }
        result_path = output / "result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({**result, "result_path": str(result_path), "result_sha256": sha256_file(result_path)}, indent=2, sort_keys=True))
        return 0 if validation["computed_pass"] else 2
    except (CoverageSmokeBlocked, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
