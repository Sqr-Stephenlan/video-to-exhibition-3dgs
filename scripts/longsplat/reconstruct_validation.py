"""CPU-only validation of an existing LongSplat reconstruction run.

This module deliberately does not create a :class:`RunLedger`, launch an
executor, decode images, or infer a visual decision.  It reads the immutable
artifacts already present in one run and writes one append-only validation
record so a failed GPU evaluator can be closed by the resident CPU
postprocess evidence without rerunning CUDA.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .authority_manifest import AuthorityManifestError, load_authority_manifest
from .converted_eval_postprocess import _expected_converted_names
from .conversion_evidence_schema import (
    ConvertedEvaluationSchemaError,
    normalize_converted_evaluation,
)
from .pipeline_contract import PipelineBlocked, sha256_file, stable_sha256


class ExistingRunValidationError(PipelineBlocked):
    """The existing run cannot be proven reusable from its immutable evidence."""


_ATTEMPT_RE = re.compile(r"attempt-(?P<number>[0-9]+)$")


def _fail(message: str) -> None:
    raise ExistingRunValidationError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not valid JSON: {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object: {path}")
    return value


def _reject_symlink_components(path: Path, label: str) -> None:
    probe = Path(path.anchor)
    for component in path.parts[1:]:
        probe /= component
        if probe.is_symlink():
            _fail(f"{label} traverses a symlink: {probe}")


def _identity(path: Path, label: str, *, directory: bool = False) -> dict[str, Any]:
    if not path.is_absolute():
        _fail(f"{label} must be absolute: {path}")
    _reject_symlink_components(path, label)
    if directory:
        if not path.is_dir():
            _fail(f"{label} directory is missing: {path}")
        return {"path": str(path.resolve())}
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} is missing or symlinked: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _attempt_number(path: Path) -> int:
    match = _ATTEMPT_RE.fullmatch(path.name)
    return int(match.group("number")) if match else -1


def _attempts(run_root: Path, stage: str) -> list[Path]:
    root = run_root / "stages" / stage
    if root.is_symlink() or not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.is_symlink() and _attempt_number(path) >= 0),
        key=_attempt_number,
    )


def _latest_file(run_root: Path, stage: str, relative: str) -> Path | None:
    for attempt in reversed(_attempts(run_root, stage)):
        candidate = attempt / relative
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    return None


def _latest_declared_artifact(run_root: Path, stage: str, field: str) -> Path | None:
    """Read one artifact only through the latest passed ledger declaration.

    A stage helper may write evidence below its pre-created attempt directory,
    so validate-only must not infer a path from a historical filename.  The
    canonical stage result declares the exact path and its artifact SHA; a
    blocked/failed latest attempt is never reused.
    """

    summary_path = run_root / "run.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        return None
    summary = _load_json(summary_path, "existing run ledger")
    entries = summary.get("stages", {}).get(stage, [])
    if not isinstance(entries, list) or not entries:
        return None
    entry = entries[-1]
    if not isinstance(entry, Mapping) or not isinstance(entry.get("result_path"), str):
        _fail(f"{stage} ledger entry does not declare its canonical result path")
    if entry.get("status") != "passed":
        return None
    canonical_path = Path(str(entry["result_path"]))
    if not canonical_path.is_absolute():
        canonical_path = run_root / canonical_path
    _reject_symlink_components(canonical_path, f"{stage} canonical result")
    try:
        canonical_path.resolve(strict=False).relative_to(run_root.resolve())
    except ValueError:
        _fail(f"{stage} canonical result escapes the run: {canonical_path}")
    attempt = canonical_path.parent
    if attempt.parent.name != stage or _attempt_number(attempt) < 0:
        _fail(f"{stage} canonical result is not below an attempt directory: {canonical_path}")
    if entry.get("attempt") != attempt.name:
        _fail(f"{stage} ledger attempt identity differs from its result path")
    if canonical_path.is_symlink() or not canonical_path.is_file():
        return None
    canonical = _load_json(canonical_path, f"{stage} canonical result")
    if canonical.get("stage") != stage:
        _fail(f"{stage} canonical result declares a different stage")
    if canonical.get("status") != "passed":
        return None
    result = canonical.get("result")
    if not isinstance(result, Mapping) or result.get("computed_pass") is not True:
        _fail(f"{stage} canonical result is not a passed reusable record")
    value = result.get(field)
    if not isinstance(value, str):
        _fail(f"{stage} canonical result does not declare {field}")
    candidate = Path(value)
    if not candidate.is_absolute():
        _fail(f"{stage} declared {field} must be absolute: {candidate}")
    _reject_symlink_components(candidate, f"{stage} declared {field}")
    try:
        candidate.resolve(strict=False).relative_to(attempt.resolve())
    except ValueError:
        _fail(f"{stage} declared {field} escapes its attempt: {candidate}")
    if candidate.is_symlink() or not candidate.is_file():
        _fail(f"{stage} declared {field} is missing or symlinked: {candidate}")
    actual_sha = sha256_file(candidate)
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        _fail(f"{stage} canonical result has no artifact inventory")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        artifact_path = artifact.get("path")
        artifact_sha = artifact.get("sha256")
        if not isinstance(artifact_path, str) or not isinstance(artifact_sha, str):
            continue
        declared = Path(artifact_path)
        _reject_symlink_components(declared, f"{stage} artifact inventory")
        if declared.resolve(strict=False) == candidate.resolve(strict=False) and artifact_sha == actual_sha:
            return candidate
    _fail(f"{stage} declared {field} is not bound to a matching artifact SHA")


def _stage(
    name: str,
    status: str,
    computed_pass: bool,
    reason: str,
    *,
    artifacts: list[Mapping[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "longsplat-reconstruct-validation-stage-v1",
        "stage": name,
        "status": status,
        "computed_pass": bool(computed_pass),
        "reason": reason,
        "artifacts": [dict(value) for value in (artifacts or [])],
        **extra,
    }


def _artifact_list(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        identity = _identity(path, "validation artifact")
        if identity["path"] not in seen:
            records.append(identity)
            seen.add(identity["path"])
    return records


def _ply_vertex_count(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            for raw in handle:
                line = raw.decode("ascii", errors="strict").strip()
                fields = line.split()
                if fields[:2] == ["element", "vertex"] and len(fields) == 3:
                    count = int(fields[2])
                    if count <= 0:
                        _fail(f"technical PLY vertex count is non-positive: {path}")
                    return count
                if line == "end_header":
                    break
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        _fail(f"technical PLY header cannot be read: {path}: {exc}")
    _fail(f"technical PLY vertex element is missing: {path}")


def _raw_stage_status(run_root: Path, stage: str) -> dict[str, Any] | None:
    ledger_path = run_root / "run.json"
    if not ledger_path.is_file() or ledger_path.is_symlink():
        return None
    ledger = _load_json(ledger_path, "existing raw run ledger")
    records = ledger.get("stages", {}).get(stage, [])
    if not isinstance(records, list) or not records:
        return None
    record = records[-1]
    if not isinstance(record, Mapping) or not isinstance(record.get("result_path"), str):
        _fail(f"raw ledger record is malformed for {stage}")
    result_path = run_root / str(record["result_path"])
    return {
        "status": record.get("status"),
        "result_path": str(result_path),
        "result": _load_json(result_path, f"raw {stage} result"),
    }


def _raw_stage(
    run_root: Path,
    stage: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    record = _raw_stage_status(run_root, stage)
    if record is None:
        if required:
            _fail(f"existing run has no raw {stage} ledger record")
        return _stage(stage, "unavailable", False, "no raw stage ledger record was found")
    payload = record["result"].get("result")
    if not isinstance(payload, Mapping):
        payload = record["result"]
    passed = record.get("status") == "passed" and payload.get("computed_pass") is not False
    return _stage(
        stage,
        "passed" if passed else str(record.get("status", "unknown")),
        passed,
        "existing raw stage record and artifacts are present" if passed else "existing raw stage is not passed",
        artifacts=_artifact_list([Path(str(record["result_path"]))]),
        reused=True,
        result_path=record["result_path"],
    )


def _validate_png_set(root: Path, expected_names: list[str], label: str) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        _fail(f"{label} directory is missing or symlinked: {root}")
    entries = list(root.iterdir())
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        _fail(f"{label} contains a symlink or non-file entry")
    actual = sorted(entry.name for entry in entries)
    if actual != sorted(expected_names) or len(actual) != len(expected_names):
        _fail(f"{label} names/count differ from authority: {len(actual)} != {len(expected_names)}")
    return {
        "root": str(root.resolve()),
        "count": len(actual),
        "names": expected_names,
        "artifacts": _artifact_list([root / name for name in expected_names]),
    }


def _find_authority(run_root: Path, supplied: str | Path | None) -> Path:
    if supplied is not None:
        path = Path(supplied).resolve()
        if path.is_file() and not path.is_symlink():
            return path
        _fail(f"supplied authority manifest is missing or symlinked: {path}")
    paths: list[Path] = []
    for attempt in _attempts(run_root, "authority-manifest"):
        candidate = attempt / "authority_manifest-v1.json"
        if candidate.is_file() and not candidate.is_symlink():
            paths.append(candidate)
    if not paths:
        _fail("existing run has no authority manifest")
    return paths[-1]


def _accepted_package(run_root: Path) -> Path | None:
    candidates = [
        path
        for path in run_root.glob("accepted_delivery_*")
        if path.is_dir() and not path.is_symlink() and (path / "SUPERSPLAT_ACCEPTANCE.json").is_file()
    ]
    return sorted(candidates)[-1] if candidates else None


def _candidate_package(run_root: Path) -> Path | None:
    candidates: list[Path] = []
    for attempt in _attempts(run_root, "candidate-delivery"):
        for name in ("candidate_delivery", "candidate"):
            path = attempt / name
            if path.is_dir() and not path.is_symlink() and (path / "candidate_manifest.json").is_file():
                candidates.append(path)
    return sorted(candidates)[-1] if candidates else None


def _validate_accepted(
    package: Path,
    *,
    candidate_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    acceptance_path = package / "SUPERSPLAT_ACCEPTANCE.json"
    record = _load_json(acceptance_path, "accepted SuperSplat record")
    point = record.get("point_cloud")
    if not isinstance(point, Mapping):
        _fail("accepted SuperSplat record has no point_cloud identity")
    point_path = package / "point_cloud.ply"
    point_identity = _identity(point_path, "accepted point cloud")
    if point.get("sha256") != point_identity["sha256"] or point.get("size_bytes") != point_identity["size_bytes"]:
        _fail("accepted point cloud identity differs from its record")
    if candidate_identity is not None and (
        point_identity["sha256"] != candidate_identity.get("sha256")
        or point_identity["size_bytes"] != candidate_identity.get("size_bytes")
    ):
        _fail("accepted point cloud differs from technical candidate identity")
    screenshot_evidence = record.get("screenshot_file_evidence")
    if screenshot_evidence not in {"missing", "complete"}:
        _fail("accepted record must distinguish screenshot_file_evidence")
    if screenshot_evidence == "missing":
        if record.get("screenshot_evidence_complete") is not False or record.get("screenshot_records") != [] or record.get("contact_sheet") is not None:
            _fail("missing screenshot evidence has contradictory archive fields")
        status = "ACCEPTED_BY_USER_PENDING_SCREENSHOT_ARCHIVE"
    else:
        status = str(record.get("status", "ACCEPTED_DELIVERY"))
    if record.get("accepted") is not True or record.get("supersplat") is not True or record.get("three_view_manual_acceptance") is not True:
        _fail("accepted package lacks explicit user/SuperSplat acceptance flags")
    return _stage(
        "accepted-delivery",
        status,
        True,
        "explicit user acceptance imported; screenshot archive state is recorded separately",
        artifacts=_artifact_list([acceptance_path, point_path, package / "PROVENANCE.json", package / "SHA256SUMS.txt"]),
        accepted=True,
        supersplat=True,
        user_asserted_manual_acceptance=bool(record.get("user_asserted_manual_acceptance", True)),
        screenshot_file_evidence=screenshot_evidence,
        screenshot_evidence_complete=record.get("screenshot_evidence_complete"),
        package=str(package.resolve()),
        point_cloud=point_identity,
    )


def validate_existing_run(
    *,
    route_root: str | Path,
    run_root: str | Path,
    input_video: str | Path,
    authority_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an existing run without executing or mutating any algorithm stage."""

    route = Path(route_root).resolve()
    run = Path(run_root).resolve()
    outputs = run.parent
    if outputs == run or outputs == Path(outputs.anchor) or outputs.is_symlink() or not outputs.is_dir():
        _fail(f"dynamic output root is missing, symlinked, or unsafe: {outputs}")
    if run.parent != outputs:
        _fail(f"existing run must be a direct child of its dynamic output root: {run}")
    _reject_symlink_components(run, "existing run")
    if not run.is_dir() or run.is_symlink():
        _fail(f"existing run directory is missing or symlinked: {run}")

    source = Path(input_video).resolve()
    source_identity = _identity(source, "input video")
    authority_path = _find_authority(run, authority_manifest)
    try:
        authority = load_authority_manifest(authority_path, route_root=route, containment_root=run)
    except AuthorityManifestError as exc:
        _fail(str(exc))
    manifest = authority["manifest"]
    declared_source = manifest["source_video"]
    if declared_source["path"] != source_identity["path"] or declared_source["sha256"] != source_identity["sha256"]:
        _fail("input video identity differs from the authority manifest")

    stage_results: dict[str, dict[str, Any]] = {}
    for name in ("preflight", "probe", "frames", "colmap", "camera-staging", "longsplat-input"):
        stage_results[name] = _raw_stage(run, name, required=False)

    smoke_render_path = _latest_file(run, "smoke100-render", "executor/result.json")
    if smoke_render_path is not None:
        smoke = _load_json(smoke_render_path, "smoke100 render result")
        structural = smoke.get("structural")
        smoke_pass = smoke.get("exit_code") == 0 and smoke.get("structural_pass") is True
        stage_results["smoke100-render"] = _stage(
            "smoke100-render",
            "passed" if smoke_pass else "failed",
            smoke_pass,
            "smoke100 structural render evidence is reusable; rough visual is not an acceptance gate" if smoke_pass else "smoke100 structural render evidence did not pass",
            artifacts=_artifact_list([smoke_render_path]),
            reused=True,
            rough_visual_decision="separate_manual_or_coverage_decision",
            structural=structural,
        )
    else:
        stage_results["smoke100-render"] = _stage("smoke100-render", "unavailable", False, "no smoke100 render result was found")

    coverage_plan_path = _latest_file(run, "coverage-smoke", "coverage-smoke-plan-v1.json")
    coverage_validation_path = _latest_file(run, "coverage-smoke", "coverage-smoke-validation-v1.json")
    coverage_result_path = _latest_file(run, "coverage-smoke", "result.json")
    if coverage_plan_path and coverage_validation_path and coverage_result_path:
        coverage_result = _load_json(coverage_result_path, "coverage smoke result")
        stage_results["coverage-smoke-plan"] = _stage(
            "coverage-smoke-plan",
            "passed" if coverage_result.get("computed_pass") is True else "blocked",
            coverage_result.get("computed_pass") is True,
            "dynamic coverage plan/validation is reusable and CPU-only" if coverage_result.get("computed_pass") is True else "coverage plan is blocked",
            artifacts=_artifact_list([coverage_plan_path, coverage_validation_path, coverage_result_path]),
            reused=True,
            gpu_invoked=False,
            formal_auto_release=False,
            active_camera_count=authority["camera_count"],
            active_camera_order=authority["camera_order"],
        )
    else:
        stage_results["coverage-smoke-plan"] = _stage("coverage-smoke-plan", "unavailable", False, "coverage plan artifacts were not found")

    for stage in ("coverage-smoke-training", "coverage-smoke-render", "convergence-smoke-training", "convergence-smoke-render", "formal-training", "native-render"):
        source_stage = "formal-native-render" if stage == "native-render" and not _attempts(run, stage) else stage
        executor_result = _latest_file(run, source_stage, "executor/result.json")
        if executor_result is None:
            stage_results[stage] = _stage(stage, "unavailable", False, "no executor result was found")
            continue
        value = _load_json(executor_result, f"{stage} executor result")
        passed = value.get("exit_code") == 0 and value.get("structural_pass") is True
        stage_results[stage] = _stage(
            stage,
            "passed" if passed else "failed",
            passed,
            "existing executor result passed structural validation" if passed else "existing executor result did not pass structural validation",
            artifacts=_artifact_list([executor_result]),
            reused=True,
            gpu_invoked=True,
            source_stage=source_stage,
            structural=value.get("structural"),
        )

    authority_identity = _identity(authority_path, "authority manifest")
    stage_results["authority-manifest"] = _stage(
        "authority-manifest",
        "passed",
        True,
        "authority manifest revalidated from this run and dynamic camera contract",
        artifacts=[authority_identity],
        reused=True,
        authority_manifest=authority_identity,
        camera_count=authority["camera_count"],
        camera_order=authority["camera_order"],
        camera_dimensions=authority["camera_dimensions"],
        depth_source=manifest.get("run_scope", {}).get("depth_source"),
    )

    conversion_path = _latest_file(run, "conversion", "executor/conversion_result.json")
    candidate_identity: dict[str, Any] | None = None
    if conversion_path is not None:
        conversion = _load_json(conversion_path, "conversion result")
        structural = conversion.get("structural")
        if not isinstance(structural, Mapping) or not isinstance(structural.get("path"), str):
            _fail("conversion result has no technical PLY identity")
        technical_ply = Path(str(structural["path"])).resolve()
        identity = _identity(technical_ply, "technical converted PLY")
        if structural.get("sha256") != identity["sha256"] or int(structural.get("file_size", identity["size_bytes"])) != identity["size_bytes"]:
            _fail("technical converted PLY identity drifted from conversion result")
        candidate_identity = {**identity, "vertices": _ply_vertex_count(technical_ply)}
        conversion_pass = conversion.get("STRUCTURAL_CONVERSION_PASS") is True and conversion.get("structural_pass") is True
        stage_results["conversion"] = _stage(
            "conversion",
            "passed" if conversion_pass else "failed",
            conversion_pass,
            "unique standard conversion technical pass is reusable" if conversion_pass else "conversion technical pass is not reusable",
            artifacts=_artifact_list([conversion_path, technical_ply]),
            reused=True,
            gpu_invoked=True,
            technical_ply=candidate_identity,
        )
    else:
        stage_results["conversion"] = _stage("conversion", "unavailable", False, "no conversion result was found")

    eval_path = _latest_file(run, "converted-eval", "executor/evaluation_result.json") or _latest_file(run, "conversion", "executor/evaluation_result.json")
    if eval_path is not None:
        evaluation = _load_json(eval_path, "converted evaluation result")
        eval_attempt_root = eval_path.parent
        eval_root = eval_attempt_root / "same_camera_eval"
        render_available = False
        if eval_root.is_dir() and candidate_identity is not None:
            try:
                expected = _expected_converted_names(authority["camera_order"], manifest)
                _validate_png_set(eval_root / "evaluator_result_gt", expected, "converted-eval GT")
                _validate_png_set(eval_root / "evaluator_result_renders", expected, "converted-eval converted")
                render_available = True
            except ExistingRunValidationError:
                render_available = False
        stage_results["converted-eval"] = _stage(
            "converted-eval",
            "render_artifacts_available" if render_available else "failed",
            render_available,
            "GPU evaluator output is complete and can be resumed by CPU postprocess" if render_available else "converted evaluation output is incomplete",
            artifacts=_artifact_list([eval_path]),
            reused=True,
            gpu_invoked=True,
            render_artifacts_available=render_available,
            original_evaluator_exit_code=evaluation.get("exit_code"),
            original_evaluator_failure_reason="unknown" if evaluation.get("exit_code") == -9 else None,
        )
    else:
        stage_results["converted-eval"] = _stage("converted-eval", "unavailable", False, "no converted evaluator result was found")

    postprocess_path = _latest_declared_artifact(
        run,
        "converted-eval-postprocess",
        "postprocess_result_path",
    )
    if postprocess_path is not None:
        postprocess = _load_json(postprocess_path, "CPU postprocess result")
        # CPU recovery is a technical contract.  Manual/rough visual review
        # remains separate and must not be reintroduced into validate-only or
        # automated delivery reachability.
        try:
            normalized_postprocess = normalize_converted_evaluation(
                postprocess,
                expected_identity={
                    "camera_count": authority["camera_count"],
                    "camera_order": authority["camera_order"],
                    "camera_dimensions": authority["camera_dimensions"],
                },
            )
        except ConvertedEvaluationSchemaError as exc:
            _fail(str(exc))
        post_pass = normalized_postprocess["STRUCTURAL_EVALUATION_PASS"] is True
        stage_results["converted-eval-postprocess"] = _stage(
            "converted-eval-postprocess",
            "passed" if post_pass else "needs_review",
            post_pass,
            "resident CPU streaming postprocess closes the evaluator recovery" if post_pass else "CPU postprocess is not a complete pass",
            artifacts=_artifact_list([postprocess_path, Path(str(postprocess.get("metrics", {}).get("path", ""))), Path(str(postprocess.get("contact_sheet", {}).get("path", "")))]) if isinstance(postprocess.get("metrics"), Mapping) and isinstance(postprocess.get("contact_sheet"), Mapping) else _artifact_list([postprocess_path]),
            reused=True,
            gpu_invoked=False,
            render_reused=postprocess.get("render_reused"),
            cuda_rerun=postprocess.get("cuda_rerun"),
            visual_quality_pass=normalized_postprocess["visual_quality_pass"],
            legacy_compatibility_warnings=normalized_postprocess["legacy_compatibility_warnings"],
            resident_full_resolution_frame_max=postprocess.get("full_stream_validation", {}).get("resident_full_resolution_frame_max") if isinstance(postprocess.get("full_stream_validation"), Mapping) else None,
        )
    else:
        stage_results["converted-eval-postprocess"] = _stage("converted-eval-postprocess", "unavailable", False, "no CPU postprocess result was found")

    candidate_package = _candidate_package(run)
    if candidate_package is not None:
        candidate_manifest_path = candidate_package / "candidate_manifest.json"
        candidate_manifest = _load_json(candidate_manifest_path, "candidate manifest")
        candidate_ply = candidate_package / "point_cloud.ply"
        candidate_file_identity = _identity(candidate_ply, "candidate point cloud")
        if candidate_identity is not None and candidate_file_identity["sha256"] != candidate_identity["sha256"]:
            _fail("candidate PLY differs from technical conversion PLY")
        if candidate_manifest.get("accepted") is not False or candidate_manifest.get("supersplat") is not False:
            _fail("technical candidate must remain separate from acceptance")
        stage_results["candidate-delivery"] = _stage(
            "candidate-delivery",
            "passed",
            True,
            "technical candidate package is reusable and acceptance remains separate",
            artifacts=_artifact_list([candidate_manifest_path, candidate_ply, candidate_package / "SHA256SUMS.txt"]),
            reused=True,
            accepted=False,
            supersplat=False,
            candidate_package=str(candidate_package.resolve()),
            point_cloud=candidate_file_identity,
        )
        candidate_identity = {**candidate_identity, **candidate_file_identity}
    else:
        stage_results["candidate-delivery"] = _stage("candidate-delivery", "unavailable", False, "no technical candidate package was found")

    accepted_package = _accepted_package(run)
    if accepted_package is not None:
        stage_results["accepted-delivery"] = _validate_accepted(accepted_package, candidate_identity=candidate_identity)
    else:
        stage_results["accepted-delivery"] = _stage(
            "accepted-delivery",
            "blocked",
            False,
            "no accepted delivery package was found; explicit SuperSplat import remains required",
            accepted=False,
            supersplat=False,
        )

    identity_path = run / "identity.json"
    historical_gpu = False
    if identity_path.is_file() and not identity_path.is_symlink():
        identity = _load_json(identity_path, "existing run identity")
        historical_gpu = bool(_load_json(run / "run.json", "existing run ledger").get("gpu_invoked", False)) if (run / "run.json").is_file() else False
    else:
        identity = {"source_video_sha256": source_identity["sha256"]}
    reusable = {name: value.get("computed_pass") is True for name, value in stage_results.items()}
    result = {
        "schema_version": "longsplat-reconstruct-validate-only-v1",
        "status": "stopped",
        "computed_pass": all(
            stage_results[name].get("computed_pass") is True
            for name in ("authority-manifest", "conversion", "converted-eval-postprocess", "candidate-delivery", "accepted-delivery")
        ),
        "validate_only": True,
        "gpu_invoked": False,
        "historical_gpu_invoked": historical_gpu,
        "cuda_rerun": False,
        "algorithm_invoked": False,
        "source_video": source_identity,
        "run_root": str(run),
        "run_identity": identity,
        "run_identity_sha256": stable_sha256(identity),
        "authority_manifest": authority_identity,
        "camera_contract": {
            "count": authority["camera_count"],
            "order": authority["camera_order"],
            "dimensions": authority["camera_dimensions"],
            "held_out": authority["status"]["held_out"],
        },
        "stage_results": stage_results,
        "reusable_stage_map": reusable,
        "recovery": {
            "failed_gpu_evaluator_exit_code": stage_results["converted-eval"].get("original_evaluator_exit_code"),
            "failed_gpu_evaluator_reason": stage_results["converted-eval"].get("original_evaluator_failure_reason"),
            "render_reused": stage_results["converted-eval-postprocess"].get("render_reused"),
            "cpu_postprocess_pass": stage_results["converted-eval-postprocess"].get("computed_pass") is True,
            "cuda_rerun": False,
        },
        "acceptance_state": {
            "accepted": stage_results["accepted-delivery"].get("accepted", False),
            "supersplat": stage_results["accepted-delivery"].get("supersplat", False),
            "screenshot_file_evidence": stage_results["accepted-delivery"].get("screenshot_file_evidence", "missing"),
        },
        "next_slice": "archive the current candidate's three SuperSplat screenshots if a file-complete package is required; fresh-video E2E remains unexecuted",
    }

    validation_root = run / "stages" / "reconstruct-validate-only"
    validation_root.mkdir(parents=True, exist_ok=True)
    existing = [_attempt_number(path) for path in validation_root.iterdir() if path.is_dir() and _attempt_number(path) >= 0]
    attempt = validation_root / f"attempt-{(max(existing) + 1 if existing else 1):04d}"
    if attempt.exists() or attempt.is_symlink():
        _fail(f"validate-only attempt is not fresh: {attempt}")
    attempt.mkdir()
    request = {
        "schema_version": "reconstruct-validate-only-request-v1",
        "input_video": source_identity,
        "authority_manifest": authority_identity,
        "algorithm_invoked": False,
        "gpu_invoked": False,
    }
    (attempt / "request.json").write_text(json.dumps(request, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    result["validation_attempt"] = str(attempt.resolve())
    result_path = attempt / "validate_only_result-v1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path.resolve())
    result["result_sha256"] = sha256_file(result_path)
    (attempt / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result
