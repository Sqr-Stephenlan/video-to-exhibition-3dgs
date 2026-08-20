"""CPU-only plan, audit, and validation for the one-shot convergence smoke.

``convergence1000-v1`` is a local diagnostic profile.  It is deliberately
fixed at 1000 iterations: this module never searches for a larger schedule and
never makes a formal-training or acceptance decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .sampling_telemetry import validate_sampling_telemetry
from .pipeline_contract import PipelineBlocked, sha256_file


CONVERGENCE_SMOKE_SCHEMA = "convergence-smoke-v1"
PROFILE = "convergence1000-v1"
ITERATIONS = 1000
MAX_ACTIVE_CAMERA_COUNT = 500
TELEMETRY_SCHEMA = "camera-sampling-telemetry-v1"
_LOSS_RE = re.compile(r"Loss=([-+0-9.eE]+)")


class ConvergenceSmokeBlocked(PipelineBlocked):
    """The fixed convergence diagnostic is not identity-safe."""


def _fail(message: str) -> None:
    raise ConvergenceSmokeBlocked(message)


def _sha(path: Path) -> str:
    return sha256_file(path)


def predicted_exposure(active_camera_count: int, iterations: int = ITERATIONS) -> dict[str, Any]:
    """Return the exact floor/ceil distribution implied by stack refill."""

    if isinstance(active_camera_count, bool) or not isinstance(active_camera_count, int) or active_camera_count <= 0:
        _fail("active_camera_count must be a positive integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        _fail("iterations must be a positive integer")
    complete_rounds, partial_round_size = divmod(iterations, active_camera_count)
    high_count = partial_round_size
    low_count = active_camera_count - high_count
    return {
        "complete_rounds": complete_rounds,
        "partial_round_size": partial_round_size,
        "min_exposure_count": complete_rounds,
        "max_exposure_count": complete_rounds + (1 if partial_round_size else 0),
        "camera_count_at_min_exposure": low_count,
        "camera_count_at_max_exposure": high_count,
        "predicted_unique_camera_count": min(active_camera_count, iterations),
        "predicted_zero_exposure_camera_count": max(0, active_camera_count - iterations),
        "predicted_coverage_fraction": min(1.0, iterations / active_camera_count),
        "total_exposure_count": iterations,
    }


def expected_anchor_adjust_iterations(iterations: int = ITERATIONS) -> list[int]:
    """Mirror the external fork's actual ``iteration < update_until`` gate."""

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        _fail("iterations must be a positive integer")
    return [
        iteration
        for iteration in range(1, iterations + 1)
        if iteration < iterations and iteration > 0 and iteration % 100 == 0
    ]


def _names(values: Sequence[str]) -> list[str]:
    result = [str(value) for value in values]
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        _fail("active camera order must be non-empty and unique")
    return result


def plan_convergence_smoke(
    *,
    active_camera_count: int,
    camera_names: Sequence[str],
    source_video_sha256: str | None = None,
    source_path: str | Path | None = None,
    static_contract_path: str | Path | None = None,
    static_contract_sha256: str | None = None,
    camera_contract_path: str | Path | None = None,
    camera_contract_sha256: str | None = None,
    pose_reference_path: str | Path | None = None,
    pose_reference_sha256: str | None = None,
    camera: Mapping[str, Any] | None = None,
    requested_iterations: int = ITERATIONS,
) -> dict[str, Any]:
    """Build the single fixed local convergence diagnostic plan."""

    if isinstance(active_camera_count, bool) or not isinstance(active_camera_count, int) or active_camera_count <= 0:
        _fail("active_camera_count must be a positive integer")
    if requested_iterations != ITERATIONS:
        _fail(f"{PROFILE} is fixed at exactly {ITERATIONS} iterations")
    if active_camera_count > MAX_ACTIVE_CAMERA_COUNT:
        _fail(
            f"{PROFILE} supports at most {MAX_ACTIVE_CAMERA_COUNT} active cameras; "
            "profile insufficient and automatic iteration search is forbidden"
        )
    if requested_iterations < 2 * active_camera_count:
        _fail(f"{PROFILE} requires at least two complete camera coverage rounds: T={requested_iterations}, N={active_camera_count}")
    names = _names(camera_names)
    if len(names) != active_camera_count:
        _fail("camera order length differs from active_camera_count")
    prediction = predicted_exposure(active_camera_count, requested_iterations)
    anchors = expected_anchor_adjust_iterations(requested_iterations)
    identity = {
        "source_video_sha256": source_video_sha256,
        "source_path": None if source_path is None else str(Path(source_path).resolve()),
        "static_contract_path": None if static_contract_path is None else str(Path(static_contract_path).resolve()),
        "static_contract_sha256": static_contract_sha256,
        "camera_contract_path": None if camera_contract_path is None else str(Path(camera_contract_path).resolve()),
        "camera_contract_sha256": camera_contract_sha256,
        "camera_order_sha256": hashlib.sha256(
            json.dumps(names, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "pose_reference_path": None if pose_reference_path is None else str(Path(pose_reference_path).resolve()),
        "pose_reference_sha256": pose_reference_sha256,
        "camera_count": active_camera_count,
        "camera_order": names,
        "camera": dict(camera or {}),
        "external_colmap_pose": True,
        "training_mode": "fixed_pose_rgb_only",
        "depth_source": "disabled",
    }
    return {
        "schema_version": CONVERGENCE_SMOKE_SCHEMA,
        "profile": PROFILE,
        "profile_class": "local_experimental_intermediate_diagnostic",
        "computed_pass": True,
        "plan_status": "plan_validated",
        "gpu_invoked": False,
        "training_invoked": False,
        "render_invoked": False,
        "formal_auto_release": False,
        "formal_gate": "closed",
        "requested_iterations": requested_iterations,
        "render_iteration": requested_iterations,
        "active_camera_count": active_camera_count,
        "active_camera_order": names,
        "identity_binding": identity,
        "sampling": {
            "confirmed_semantics": "one camera per iteration; phase-local random pop without replacement; stack refill",
            "exposure_rule": "floor(T/N) or ceil(T/N) per camera",
            "prediction": prediction,
            "telemetry_required": True,
            "telemetry_observer_has_no_random_side_effect": True,
        },
        "densification": {
            "fork_semantics": "anchor_growing via adjust_anchor/anchor_growing; external route does not use classic densify_and_clone/split/reset_opacity",
            "planned_adjust_anchor_iterations": anchors,
            "update_interval": 100,
            "update_until": requested_iterations,
            "require_purning": False,
            "runtime_counts_required": True,
            "runtime_counts_are_observed_not_predicted": True,
        },
        "policy_boundary": {
            "official_formula_status": "not_specified",
            "official_training_standard": "not_claimed",
            "local_policy": "one authorized 1000-it diagnostic only",
            "automatic_higher_iteration_search": False,
            "forbidden_next_automatic_schedules": [2000, 5000],
            "formal_training": "not formal; no automatic release",
            "visual_quality": "diagnostic only; human review remains separate",
        },
        "training_state_resume": {
            "bit_exact_resume": False,
            "meaning": "checkpoint/save supports inspection, render, and conversion but cannot restore training exactly",
            "fresh_model_required": True,
        },
        "gpu_workload_estimate": {
            "camera_exposures": requested_iterations,
            "predicted_complete_rounds": prediction["complete_rounds"],
            "anchor_adjust_event_count": len(anchors),
            "gpu_invoked": False,
            "estimate_status": "planning_only_no_runtime_measurement",
        },
        "experiment_required": [
            "whether 1000 iterations produces recognizable geometry requires this one runtime diagnostic",
            "initialization/anchor growth must be diagnosed from runtime counts if visual output remains low-frequency",
            "no adaptive iteration formula is asserted by this local profile",
        ],
    }


def validate_convergence_smoke_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema_version") != CONVERGENCE_SMOKE_SCHEMA:
        _fail(f"convergence plan schema must be {CONVERGENCE_SMOKE_SCHEMA}")
    if plan.get("profile") != PROFILE or plan.get("requested_iterations") != ITERATIONS:
        _fail(f"convergence profile must be fixed {PROFILE}/{ITERATIONS}")
    count = plan.get("active_camera_count")
    names = plan.get("active_camera_order")
    if not isinstance(count, int) or count <= 0 or not isinstance(names, list) or len(names) != count:
        _fail("convergence camera count/order is invalid")
    if count > MAX_ACTIVE_CAMERA_COUNT:
        _fail(f"convergence camera count exceeds the {MAX_ACTIVE_CAMERA_COUNT}-camera profile cap")
    if ITERATIONS < 2 * count:
        _fail("convergence plan does not provide two complete camera coverage rounds")
    _names([str(value) for value in names])
    if plan.get("computed_pass") is not True or plan.get("gpu_invoked") is not False or plan.get("training_invoked") is not False:
        _fail("convergence CPU plan must be computed_pass and not invoked")
    if plan.get("formal_auto_release") is not False or plan.get("formal_gate") != "closed":
        _fail("convergence plan must keep the formal gate closed")
    prediction = predicted_exposure(count, ITERATIONS)
    sampling = plan.get("sampling")
    if not isinstance(sampling, Mapping) or sampling.get("prediction") != prediction:
        _fail("convergence exposure prediction is inconsistent")
    densification = plan.get("densification")
    if not isinstance(densification, Mapping) or densification.get("planned_adjust_anchor_iterations") != expected_anchor_adjust_iterations(ITERATIONS):
        _fail("convergence anchor schedule is inconsistent with the fork gate")
    boundary = plan.get("policy_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("automatic_higher_iteration_search") is not False:
        _fail("convergence plan must forbid automatic higher-iteration search")
    return {
        "schema_version": "convergence-smoke-validation-v1",
        "computed_pass": True,
        "profile": PROFILE,
        "iterations": ITERATIONS,
        "active_camera_count": count,
        "prediction": prediction,
        "planned_anchor_adjust_iterations": list(densification["planned_adjust_anchor_iterations"]),
        "formal_auto_release": False,
        "gpu_invoked": False,
    }


def validate_anchor_events(
    events: Sequence[Mapping[str, Any]],
    planned_iterations: Sequence[int],
) -> dict[str, Any]:
    """Check observed before/after/add/delete arithmetic without judging growth."""

    planned = [int(value) for value in planned_iterations]
    if len(events) != len(planned):
        _fail("runtime anchor event count differs from planned schedule")
    observed_iterations: list[int] = []
    normalized: list[dict[str, int]] = []
    for event, planned_iteration in zip(events, planned):
        if not isinstance(event, Mapping) or event.get("iteration") != planned_iteration:
            _fail("runtime anchor event iteration differs from planned schedule")
        values: dict[str, int] = {"iteration": planned_iteration}
        for field in ("before_anchor_count", "after_anchor_count", "added_anchor_count", "deleted_anchor_count", "net_anchor_delta"):
            value = event.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 and field != "net_anchor_delta":
                _fail(f"runtime anchor event field is invalid: {field}")
            values[field] = int(value)
        if values["after_anchor_count"] != values["before_anchor_count"] + values["added_anchor_count"] - values["deleted_anchor_count"]:
            _fail("runtime anchor event before/after/add/delete arithmetic is inconsistent")
        if values["net_anchor_delta"] != values["after_anchor_count"] - values["before_anchor_count"]:
            _fail("runtime anchor event net delta is inconsistent")
        observed_iterations.append(planned_iteration)
        normalized.append(values)
    return {"observed_iterations": observed_iterations, "events": normalized, "event_count": len(normalized)}


def _parse_loss_evidence(stdout_path: Path | None, stderr_path: Path | None) -> dict[str, Any]:
    telemetry: list[dict[str, Any]] = []
    sparse: list[dict[str, Any]] = []
    for path in (stdout_path,):
        if path is None or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "LOSS_TELEMETRY " not in line:
                continue
            try:
                payload = line.split("LOSS_TELEMETRY ", 1)[1]
                value, _ = json.JSONDecoder().raw_decode(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                telemetry.append(value)
    if stderr_path is not None and stderr_path.is_file():
        for line in stderr_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _LOSS_RE.search(line)
            if match:
                try:
                    sparse.append({"raw": line.strip(), "loss": float(match.group(1))})
                except ValueError:
                    pass
    return {
        "loss_telemetry_points": telemetry,
        "loss_telemetry_count": len(telemetry),
        "sparse_tqdm_loss_points": sparse,
        "sparse_tqdm_loss_count": len(sparse),
        "per_iteration_loss_series_available": False,
        "trend_status": "insufficient_complete_series",
        "interpretation": "LOSS_TELEMETRY is sparse by design; tqdm values are sparse EMA snapshots and do not prove a complete monotonic trend",
    }


def _checkpoint_summary(result_path: Path | None, pose_path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if result_path is not None and result_path.is_file():
        try:
            loaded = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                structural = loaded.get("structural")
                if isinstance(structural, dict):
                    checkpoint = structural.get("checkpoint")
                    if isinstance(checkpoint, dict):
                        result["checkpoint"] = dict(checkpoint)
                    result["anchor_schedule"] = structural.get("anchor_schedule")
        except (OSError, json.JSONDecodeError):
            result["result_parse"] = "unavailable"
    if pose_path is not None and pose_path.is_file():
        try:
            loaded = json.loads(pose_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                result["pose_anchor_schedule"] = loaded.get("anchor_schedule")
        except (OSError, json.JSONDecodeError):
            result["pose_parse"] = "unavailable"
    return result


def audit_prior_training_evidence(
    *,
    smoke_result_path: str | Path,
    coverage_result_path: str | Path,
    smoke_stdout_path: str | Path,
    smoke_stderr_path: str | Path,
    coverage_stdout_path: str | Path,
    coverage_stderr_path: str | Path,
    smoke_pose_path: str | Path | None = None,
    coverage_pose_path: str | Path | None = None,
) -> dict[str, Any]:
    """Record only what the legacy attempts actually prove."""

    smoke_result = Path(smoke_result_path).resolve()
    coverage_result = Path(coverage_result_path).resolve()
    smoke_pose = None if smoke_pose_path is None else Path(smoke_pose_path).resolve()
    coverage_pose = None if coverage_pose_path is None else Path(coverage_pose_path).resolve()
    smoke = _checkpoint_summary(smoke_result, smoke_pose)
    coverage = _checkpoint_summary(coverage_result, coverage_pose)
    smoke_count = smoke.get("checkpoint", {}).get("vertex_count")
    coverage_count = coverage.get("checkpoint", {}).get("vertex_count")
    same_count = isinstance(smoke_count, int) and isinstance(coverage_count, int) and smoke_count == coverage_count
    return {
        "schema_version": "convergence-training-audit-v1",
        "scope": "CPU-only audit of preserved smoke100 and coverage288 evidence",
        "gpu_invoked_by_audit": False,
        "smoke100": {
            "result": str(smoke_result),
            "result_sha256": _sha(smoke_result) if smoke_result.is_file() else None,
            "checkpoint": smoke.get("checkpoint", {}),
            "anchor_runtime": smoke.get("anchor_schedule", "unavailable_legacy_contract"),
            "loss": _parse_loss_evidence(Path(smoke_stdout_path).resolve(), Path(smoke_stderr_path).resolve()),
        },
        "coverage288": {
            "result": str(coverage_result),
            "result_sha256": _sha(coverage_result) if coverage_result.is_file() else None,
            "checkpoint": coverage.get("checkpoint", {}),
            "anchor_runtime": coverage.get("anchor_schedule", coverage.get("pose_anchor_schedule", "unavailable")),
            "loss": _parse_loss_evidence(Path(coverage_stdout_path).resolve(), Path(coverage_stderr_path).resolve()),
        },
        "checkpoint_comparison": {
            "smoke100_vertex_count": smoke_count,
            "coverage288_vertex_count": coverage_count,
            "same_vertex_count_observed": same_count,
            "net_anchor_growth_proven": False,
            "reason": "legacy coverage anchor telemetry recorded iterations only; before/after/add/delete counts were not emitted",
        },
        "sampling_audit": {
            "confirmed": "phase-local random pop without replacement with stack refill",
            "n144_t1000_prediction": predicted_exposure(144, ITERATIONS),
            "runtime_n1000_not_yet_observed": True,
        },
        "limitations": [
            "no complete per-iteration loss series is available from the preserved logs",
            "legacy smoke100 has no anchor runtime contract",
            "coverage288 has anchor iterations [100, 200] but no per-call counts",
            "equal 100-it and 288-it checkpoint vertex counts do not prove zero runtime anchor growth",
        ],
    }


def write_json_once(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path).resolve()
    if destination.exists() or destination.is_symlink():
        _fail(f"audit/plan output must be fresh and absent: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _cli_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConvergenceSmokeBlocked(f"CLI input is missing or symlinked: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConvergenceSmokeBlocked(f"CLI JSON input must be an object: {path}")
    return value


def _build_cli() -> Any:
    parser = __import__("argparse").ArgumentParser(description="CPU-only convergence1000-v1 plan and audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan", help="write one fixed 1000-iteration CPU plan")
    plan_parser.add_argument("--camera-contract", type=Path, required=True)
    plan_parser.add_argument("--static-contract", type=Path, required=True)
    plan_parser.add_argument("--pose-reference", type=Path)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--iterations", type=int, default=ITERATIONS)
    audit_parser = subparsers.add_parser("audit", help="audit preserved smoke100/coverage288 evidence")
    for name in ("smoke-result", "coverage-result", "smoke-stdout", "smoke-stderr", "coverage-stdout", "coverage-stderr"):
        audit_parser.add_argument(f"--{name}", type=Path, required=True)
    audit_parser.add_argument("--smoke-pose", type=Path)
    audit_parser.add_argument("--coverage-pose", type=Path)
    audit_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    try:
        if args.command == "plan":
            camera_contract = _cli_json(args.camera_contract)
            static = _cli_json(args.static_contract)
            names = camera_contract.get("frame_names")
            if not isinstance(names, list):
                _fail("camera contract frame_names are required")
            result = plan_convergence_smoke(
                active_camera_count=len(names),
                camera_names=[str(value) for value in names],
                source_video_sha256=static.get("source_video_sha256"),
                source_path=args.static_contract.parent.parent,
                static_contract_path=args.static_contract,
                static_contract_sha256=sha256_file(args.static_contract),
                camera_contract_path=args.camera_contract,
                camera_contract_sha256=sha256_file(args.camera_contract),
                pose_reference_path=args.pose_reference,
                pose_reference_sha256=None if args.pose_reference is None else sha256_file(args.pose_reference),
                camera=static.get("camera") if isinstance(static.get("camera"), Mapping) else camera_contract.get("camera"),
                requested_iterations=args.iterations,
            )
            validate_convergence_smoke_plan(result)
            write_json_once(args.output, result)
        else:
            result = audit_prior_training_evidence(
                smoke_result_path=args.smoke_result,
                coverage_result_path=args.coverage_result,
                smoke_stdout_path=args.smoke_stdout,
                smoke_stderr_path=args.smoke_stderr,
                coverage_stdout_path=args.coverage_stdout,
                coverage_stderr_path=args.coverage_stderr,
                smoke_pose_path=args.smoke_pose,
                coverage_pose_path=args.coverage_pose,
            )
            write_json_once(args.output, result)
        print(json.dumps({"status": "passed", "output": str(args.output.resolve()), "sha256": sha256_file(args.output)}, indent=2, sort_keys=True))
        return 0
    except (ConvergenceSmokeBlocked, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
