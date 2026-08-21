"""Generic one-click LongSplat reconstruction orchestration.

This module owns only orchestration: ordering, append-only attempts, resume
identity, stage gates, and delivery boundaries.  Raw video preparation,
training/render execution, conversion, and evaluation remain implemented by
their existing contract-bound executors.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .pipeline_contract import (
    PipelineBlocked,
    ResumeMismatchError,
    RunLedger,
    build_run_identity,
    code_identity,
    resolve_output_root,
    sha256_file,
    stable_sha256,
    validate_run_id,
    write_json,
)
from .tool_provider import resolve_tool_provider
from .publisher import PublishError, publish_ply, verify_public_delivery, write_json_once_atomic
from .reconstruct_validation import validate_existing_run
from .automated_policy import (
    POLICY_ID as AUTOMATED_POLICY_ID,
    evaluate_early_gate,
    evaluate_formal_gate,
    policy_descriptor,
)
from .colmap_contract import SUPPORTED_CAMERA_MODELS, SUPPORTED_MATCHING_MODES
from .gate_schema import normalize_gate_evidence
from .render_postcheck import normalize_postcheck_evidence


STAGES = (
    "preflight",
    "probe",
    "frames",
    "colmap",
    "camera-staging",
    "longsplat-input",
    "smoke100-training",
    "smoke100-render",
    "coverage-smoke-plan",
    "coverage-smoke-training",
    "coverage-smoke-render",
    "coverage-smoke-render-postcheck",
    "coverage-smoke-visual-gate",
    "convergence-smoke-plan",
    "convergence-smoke-training",
    "convergence-smoke-render",
    "convergence-smoke-render-postcheck",
    "convergence-smoke-visual-gate",
    "automated-early-gate",
    "formal-training",
    "native-render",
    "formal-native-render-postcheck",
    "automated-formal-gate",
    "authority-manifest",
    "conversion",
    "converted-eval",
    "converted-eval-postprocess",
    "candidate-delivery",
    "automated-technical-delivery",
    "accepted-delivery",
)
STAGE_INDEX = {name: index for index, name in enumerate(STAGES)}
CPU_STAGES = STAGES[:6]
DEFAULT_PIPELINE_PROFILE = "single-convergence1000-v1"
LEGACY_PIPELINE_PROFILE = "legacy-smoke-coverage-v1"
DEFAULT_STAGES = (
    "preflight",
    "probe",
    "frames",
    "colmap",
    "camera-staging",
    "longsplat-input",
    "convergence-smoke-plan",
    "convergence-smoke-training",
    "convergence-smoke-render",
    "convergence-smoke-render-postcheck",
    "automated-early-gate",
    "formal-training",
    "native-render",
    "formal-native-render-postcheck",
    "automated-formal-gate",
    "authority-manifest",
    "conversion",
    "converted-eval",
    "converted-eval-postprocess",
    "automated-technical-delivery",
)
GPU_STAGES = {
    "smoke100-training",
    "smoke100-render",
    "coverage-smoke-training",
    "coverage-smoke-render",
    "convergence-smoke-training",
    "convergence-smoke-render",
    "formal-training",
    "native-render",
    "conversion",
    "converted-eval",
}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SMOKE_PROFILE = "smoke100-v1"
_FORMAL_PROFILE = "formal30000-v1"
_CONTACT_SHEET_LAYOUT_POLICY = "aspect-preserving-letterbox-v1"
_CONVERGENCE_CONSUMER_CODE_PATHS = frozenset(
    {
        "scripts/longsplat/reconstruct_pipeline.py",
        "scripts/longsplat/pipeline_contract.py",
        "scripts/longsplat/automated_policy.py",
        "scripts/longsplat/acceptance_delivery.py",
        "scripts/longsplat/publisher.py",
        "scripts/longsplat/authority_manifest.py",
        "scripts/longsplat/converted_eval_postprocess.py",
        "scripts/longsplat/convergence_smoke.py",
        "scripts/longsplat/convergence_executor.py",
        "scripts/longsplat/smoke_executor.py",
        "scripts/longsplat/render_postcheck.py",
    }
)
_PRODUCER_CODE_PATHS = frozenset(
    {
        "scripts/longsplat/raw_pipeline.py",
        "scripts/longsplat/video_contract.py",
        "scripts/longsplat/frame_contract.py",
        "scripts/longsplat/colmap_contract.py",
        "scripts/longsplat/camera_staging.py",
        "scripts/longsplat/longsplat_input.py",
        "scripts/longsplat/pipeline_contract.py",
        "scripts/longsplat/tool_provider.py",
        "scripts/longsplat/validate_external_colmap_contract.py",
        "nested/train.py",
        "nested/render.py",
        "nested/arguments/__init__.py",
        "nested/utils/external_colmap_pose.py",
        "nested/utils/colmap_utils.py",
        "nested/utils/general_utils.py",
        "nested/utils/graphics_utils.py",
    }
)
_PRODUCER_CODE_PREFIXES = ("nested/scene/",)


def _safe_run_id(value: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or not _RUN_ID_RE.fullmatch(value):
        raise ValueError("run-id must contain only letters, numbers, '.', '_' and '-'")
    return validate_run_id(value)


def _route_output_root(value: str | Path, route: Path) -> Path:
    """Resolve a user-selected output root; ``route`` is only a code-root hint.

    The historical name is retained for callers and compatibility, but the
    output root is intentionally not constrained to ``route / outputs``.
    """

    del route
    return resolve_output_root(value)


def _stage_result(
    *,
    stage: str,
    status: str,
    computed_pass: bool,
    reason: str,
    plan: bool,
    artifacts: Sequence[Mapping[str, Any]] | None = None,
    payload: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    base = {
        "schema_version": "longsplat-reconstruct-stage-v2",
        "stage": stage,
        "status": status,
        "computed_pass": bool(computed_pass),
        "plan": bool(plan),
        "reason": reason,
        "artifacts": [dict(item) for item in (artifacts or [])],
    }
    merged = dict(base)
    merged.update(extra)
    conflicts: dict[str, dict[str, Any]] = {}
    for key, value in dict(payload or {}).items():
        if key in merged and merged[key] != value:
            conflicts[key] = {"builder_value": merged[key], "payload_value": value}
        else:
            merged[key] = value
    if conflicts:
        # Return a structured blocked result instead of allowing duplicate
        # kwargs or a silent overwrite to escape before ledger.finish_attempt.
        return {
            **base,
            "status": "blocked",
            "computed_pass": False,
            "reason": "stage-result reserved field conflict",
            "stage_result_conflict": conflicts,
        }
    return merged


def _stage_reusable(ledger: RunLedger, stage: str) -> dict[str, Any] | None:
    latest = ledger.latest_attempt(stage)
    if latest is None or not latest["reusable"]:
        return None
    result = latest.get("result")
    if not isinstance(result, Mapping):
        return None
    if stage in {"coverage-smoke-render-postcheck", "convergence-smoke-render-postcheck", "formal-native-render-postcheck"}:
        layout = result.get("contact_sheet_layout")
        if not isinstance(layout, Mapping) or layout.get("policy") != _CONTACT_SHEET_LAYOUT_POLICY:
            # A prior postcheck may be structurally valid but its contact sheet
            # was generated with the historical non-aspect-preserving layout.
            # Preserve it and force a fresh CPU-only evidence attempt.
            return None
    return dict(result)


def _record_stage(
    ledger: RunLedger,
    stage: str,
    result: Mapping[str, Any],
    *,
    status: str,
    request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attempt = ledger.begin_attempt(stage, request or {"stage": stage, "contract": dict(result)})
    ledger.finish_attempt(stage=stage, attempt=attempt, status=status, result=result)
    return dict(result)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineBlocked(f"immutable JSON artifact is unreadable: {path}: {exc}") from exc
        if existing != dict(value):
            raise PipelineBlocked(f"immutable JSON artifact differs on resume: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _verify_existing_public_delivery(
    *,
    ledger: RunLedger,
    publish_dir: str | Path,
    delivery_name: str | None,
) -> None:
    """Revalidate an already-published PLY before any resume reuse.

    Public delivery is deliberately outside the internal artifact containment
    root. Its receipt therefore binds the exact configured directory/name and
    is checked independently on every product resume.
    """

    receipt_path = ledger.run_dir / "published_ply.json"
    summary_receipt = ledger.summary.get("published_ply_receipt")
    summary_published = ledger.summary.get("published_ply")
    has_summary_delivery = summary_receipt is not None or summary_published is not None
    if not receipt_path.exists() and not receipt_path.is_symlink():
        if has_summary_delivery or ledger.summary.get("delivery_reachable") is True:
            reason = "public delivery receipt is missing while run.json claims delivery"
            ledger.summary["published_ply"] = None
            ledger.summary["published_ply_receipt"] = None
            ledger.mark_blocked(stage="automated-technical-delivery", error=reason, exit_code=2)
            raise PublishError(reason)
        return
    if receipt_path.is_symlink() or not receipt_path.is_file():
        reason = f"public delivery receipt is not a regular file: {receipt_path}"
        ledger.summary["published_ply"] = None
        ledger.summary["published_ply_receipt"] = None
        ledger.mark_blocked(stage="automated-technical-delivery", error=reason, exit_code=2)
        raise PublishError(reason)
    try:
        receipt = _load_json(receipt_path, "published PLY receipt")
        verify_public_delivery(receipt, output_dir=publish_dir, name=delivery_name)
    except (
        PipelineBlocked,
        PublishError,
        OSError,
        TypeError,
        AttributeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        reason = f"public delivery verification failed on resume: {exc}"
        ledger.summary["published_ply"] = None
        ledger.summary["published_ply_receipt"] = None
        ledger.mark_blocked(stage="automated-technical-delivery", error=reason, exit_code=2)
        raise PublishError(reason) from exc
    if summary_receipt is not None and summary_receipt != str(receipt_path):
        reason = "run.json public receipt path does not bind published_ply.json"
        ledger.summary["published_ply"] = None
        ledger.summary["published_ply_receipt"] = None
        ledger.mark_blocked(stage="automated-technical-delivery", error=reason, exit_code=2)
        raise PublishError(reason)
    if summary_published is not None and summary_published != receipt:
        reason = "run.json public receipt payload differs from published_ply.json"
        ledger.summary["published_ply"] = None
        ledger.summary["published_ply_receipt"] = None
        ledger.mark_blocked(stage="automated-technical-delivery", error=reason, exit_code=2)
        raise PublishError(reason)


def _artifact(path: str | Path) -> dict[str, Any] | None:
    value = Path(path).resolve()
    if value.is_symlink() or not value.is_file():
        return None
    return {"path": str(value), "sha256": sha256_file(value), "size_bytes": value.stat().st_size}


def _artifacts(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in paths:
        record = _artifact(value)
        if record is None or record["path"] in seen:
            continue
        seen.add(record["path"])
        records.append(record)
    return records


def _payload_artifacts(payload: Mapping[str, Any], *, result_path: Path) -> list[dict[str, Any]]:
    paths: list[str | Path] = [result_path]
    declared = payload.get("artifacts")
    if isinstance(declared, list):
        for item in declared:
            if isinstance(item, Mapping) and isinstance(item.get("path"), str):
                paths.append(item["path"])
    for key in ("probe_path", "selection_path", "canonical_media_path", "contract_path", "staging_manifest_path", "static_contract_path", "future_smoke_plan_path"):
        if isinstance(payload.get(key), str):
            paths.append(payload[key])
    return _artifacts(paths)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PipelineBlocked(f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineBlocked(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineBlocked(f"{label} must be a JSON object: {path}")
    return value


def _raw_stage_payload(raw_run_dir: Path, stage: str) -> tuple[str, dict[str, Any], Path, dict[str, Any]] | None:
    summary = _load_json(raw_run_dir / "run.json", "raw pipeline run ledger")
    entries = summary.get("stages", {}).get(stage, [])
    if not isinstance(entries, list) or not entries:
        return None
    entry = entries[-1]
    if not isinstance(entry, Mapping) or not isinstance(entry.get("result_path"), str):
        raise PipelineBlocked(f"raw pipeline stage record is malformed: {raw_run_dir}/{stage}")
    result_path = raw_run_dir / str(entry["result_path"])
    envelope = _load_json(result_path, f"raw pipeline {stage} result")
    payload = envelope.get("result")
    if not isinstance(payload, Mapping):
        raise PipelineBlocked(f"raw pipeline {stage} result payload is malformed: {result_path}")
    return str(entry.get("status", envelope.get("status", "blocked"))), dict(payload), result_path, summary


def _adapt_raw_stage(
    *,
    stage: str,
    raw_run_dir: Path,
    plan: bool,
) -> dict[str, Any] | None:
    loaded = _raw_stage_payload(raw_run_dir, stage)
    if loaded is None:
        return None
    raw_status, payload, result_path, summary = loaded
    passed = raw_status == "passed"
    if stage == "camera-staging":
        passed = passed and isinstance(payload.get("contract"), Mapping)
    elif stage == "longsplat-input":
        passed = passed and payload.get("status") == "computed_pass"
    elif stage in {"probe", "frames", "colmap"}:
        if stage == "probe":
            passed = passed and isinstance(payload.get("probe"), Mapping)
        elif stage == "frames":
            passed = passed and isinstance(payload.get("selection"), Mapping) and isinstance(payload.get("canonical_media"), Mapping)
        else:
            passed = passed and payload.get("computed_pass") is True
    status = "passed" if passed else raw_status
    if raw_status == "planned":
        status = "planned"
    reason = {
        "passed": "raw_pipeline stage contract passed",
        "planned": "raw_pipeline stage was planned only",
        "blocked": "raw_pipeline stage blocked",
        "failed": "raw_pipeline stage failed",
    }.get(status, f"raw_pipeline returned {raw_status}")
    raw_run_json = raw_run_dir / "run.json"
    records = _payload_artifacts(payload, result_path=result_path)
    records.extend(_artifacts([raw_run_json, raw_run_dir / "identity.json", raw_run_dir / "config.json"]))
    return _stage_result(
        stage=stage,
        status=status,
        computed_pass=passed,
        reason=reason,
        plan=plan,
        artifacts=records,
        algorithm_owner="scripts.longsplat.raw_pipeline",
        raw_run_dir=str(raw_run_dir),
        raw_stage_status=raw_status,
        raw_result_path=str(result_path),
        raw_result=payload,
        raw_summary_status=summary.get("status"),
        depth_source="disabled",
    )


def _manual_acceptance_valid(token: str | Path) -> dict[str, Any]:
    path = Path(token).resolve()
    if path.is_symlink() or not path.is_file():
        raise PipelineBlocked(f"manual acceptance token is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineBlocked(f"manual acceptance token is invalid: {path}: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("accepted") is not True or value.get("supersplat") is not True:
        raise PipelineBlocked("manual acceptance requires explicit accepted=true and supersplat=true")
    if value.get("three_view_manual_acceptance") is not True:
        raise PipelineBlocked("manual acceptance token does not prove three-view review")
    if value.get("screenshot_embeds_ply_sha") is True:
        raise PipelineBlocked("manual acceptance identity must not claim screenshots embed a PLY SHA")
    screenshot_state = value.get("screenshot_file_evidence", "complete")
    if screenshot_state not in {"missing", "complete"}:
        raise PipelineBlocked("manual acceptance must distinguish screenshot_file_evidence=missing|complete")
    if screenshot_state == "missing":
        if value.get("screenshot_evidence_complete") is not False:
            raise PipelineBlocked("missing screenshot evidence must set screenshot_evidence_complete=false")
        if value.get("status") != "ACCEPTED_BY_USER_PENDING_SCREENSHOT_ARCHIVE":
            raise PipelineBlocked("missing screenshot evidence requires the pending archive status")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "record": dict(value),
        "accepted": True,
        "supersplat": True,
        "screenshot_file_evidence": screenshot_state,
        "screenshot_evidence_complete": value.get("screenshot_evidence_complete", True),
    }


def _visual_decision_valid(token: str | Path, *, label: str) -> dict[str, Any]:
    """Validate an explicit human visual decision without accepting it as delivery."""

    path = Path(token).resolve()
    if path.is_symlink() or not path.is_file():
        raise PipelineBlocked(f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineBlocked(f"{label} is invalid: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise PipelineBlocked(f"{label} must be a JSON object")
    decision = value.get("decision", value.get("coverage_smoke_visual_decision"))
    if decision not in {"pass", "needs_review", "fail"}:
        raise PipelineBlocked(f"{label} decision must be pass, needs_review, or fail")
    if value.get("explicit_visual_review") is not True and value.get("supervisor_manual_decision") is not True:
        raise PipelineBlocked(f"{label} must explicitly identify a human visual review")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "record": dict(value),
        "decision": decision,
        "explicit_visual_review": True,
    }


def _visual_evidence_binding(postcheck: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Return the exact postcheck artifacts a human decision must bind."""

    postcheck_path = postcheck.get("postcheck_result_path")
    quality_path = postcheck.get("quality_evidence_path")
    postcheck_result = postcheck.get("postcheck_result")
    if not isinstance(postcheck_path, str) or not isinstance(quality_path, str) or not isinstance(postcheck_result, Mapping):
        raise PipelineBlocked(f"{label} requires a complete CPU postcheck evidence record")

    def require_artifact(value: Any, name: str) -> dict[str, Any]:
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
            raise PipelineBlocked(f"{label} is missing bound {name} evidence")
        path = Path(str(value["path"])).resolve()
        if path.is_symlink() or not path.is_file() or sha256_file(path) != value["sha256"]:
            raise PipelineBlocked(f"{label} {name} evidence is missing or mutated")
        return {"path": str(path), "sha256": str(value["sha256"]), "size_bytes": int(value.get("size_bytes", path.stat().st_size))}

    from .render_postcheck import adapt_existing_postcheck_return

    try:
        postcheck_result = adapt_existing_postcheck_return(
            postcheck_path=Path(postcheck_path),
            legacy_return=postcheck_result,
        )
    except (OSError, ValueError, TypeError) as exc:
        raise PipelineBlocked(f"{label} postcheck return contract is invalid: {exc}") from exc
    fixed = postcheck_result.get("contact_sheets", {})
    if not isinstance(fixed, Mapping):
        raise PipelineBlocked(f"{label} postcheck has no contact-sheet evidence")
    return {
        "postcheck": require_artifact({"path": postcheck_path, "sha256": sha256_file(Path(postcheck_path))}, "postcheck"),
        "quality_evidence": require_artifact({"path": quality_path, "sha256": sha256_file(Path(quality_path))}, "quality_evidence"),
        "fixed_vs_gt": require_artifact(fixed.get("fixed_vs_gt"), "fixed-vs-GT contact sheet"),
        "on_path": require_artifact(fixed.get("on_path"), "on-path contact sheet"),
        "metrics": require_artifact(postcheck_result.get("metrics_artifact"), "metrics"),
        "png_hashes": require_artifact(postcheck_result.get("png_hashes_artifact"), "PNG hash manifest"),
    }


def _decision_binds_evidence(decision: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    supplied = decision.get("evidence")
    if not isinstance(supplied, Mapping):
        raise PipelineBlocked(f"{label} must bind postcheck, contact-sheet, metrics, and PNG-hash evidence")
    for key, record in expected.items():
        value = supplied.get(key)
        if not isinstance(value, Mapping) or value.get("path") != record.get("path") or value.get("sha256") != record.get("sha256"):
            raise PipelineBlocked(f"{label} evidence binding differs for {key}")


def _first_payload_artifact(payload: Mapping[str, Any], suffix: str) -> Path | None:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    for item in artifacts:
        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
            candidate = Path(item["path"])
            if candidate.name == suffix and candidate.is_file() and not candidate.is_symlink():
                return candidate.resolve()
    return None


def _strict_path(value: str | Path, *, label: str, kind: str, root: Path | None = None) -> Path:
    """Resolve a contract path without accepting symlink or containment aliases."""

    raw = Path(value)
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    raw = raw.absolute()
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PipelineBlocked(f"{label} contains a symlink component: {raw}")
    resolved = raw.resolve(strict=False)
    if root is not None:
        root_resolved = Path(root).resolve(strict=True)
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise PipelineBlocked(f"{label} escapes containment root {root_resolved}: {resolved}") from exc
    if kind == "file" and not resolved.is_file():
        raise PipelineBlocked(f"{label} must be a regular file: {resolved}")
    if kind == "directory" and not resolved.is_dir():
        raise PipelineBlocked(f"{label} must be a regular directory: {resolved}")
    if kind not in {"file", "directory"}:
        raise ValueError(f"unsupported strict path kind: {kind}")
    return resolved


def _strict_file_record(value: str | Path, *, label: str, root: Path | None = None) -> dict[str, Any]:
    path = _strict_path(value, label=label, kind="file", root=root)
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _strict_directory_record(value: str | Path, *, label: str, root: Path | None = None) -> dict[str, Any]:
    path = _strict_path(value, label=label, kind="directory", root=root)
    return {"path": str(path), "containment_root": None if root is None else str(Path(root).resolve())}


def _explicit_sparse_model_identity(path_value: str | Path, *, label: str, training_root: Path) -> dict[str, Any]:
    directory = _strict_path(path_value, label=label, kind="directory", root=training_root)
    names = ("cameras.bin", "images.bin", "points3D.bin") if directory.name.endswith("bin") else ("cameras.txt", "images.txt", "points3D.txt")
    files = [_strict_file_record(directory / name, label=f"{label}/{name}", root=training_root) for name in names]
    return {
        "path": str(directory),
        "files": files,
        "files_sha256": stable_sha256(files),
        "hash_scope": "explicit COLMAP sparse model files only; no recursive model/render hashing",
    }


def _input_video_identity(source: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    record = _strict_file_record(source, label="input video")
    expected_sha = identity.get("source_video_sha256")
    if record["sha256"] != expected_sha:
        raise PipelineBlocked("input video SHA differs from the immutable run identity")
    expected_path = identity.get("source_video_path")
    if isinstance(expected_path, str) and str(Path(expected_path).resolve()) != record["path"]:
        raise PipelineBlocked("input video path differs from the immutable run identity")
    expected_size = identity.get("source_video_size_bytes")
    if isinstance(expected_size, int) and expected_size != record["size_bytes"]:
        raise PipelineBlocked("input video size differs from the immutable run identity")
    run_sha = identity.get("run_identity_sha256")
    if not isinstance(run_sha, str) or not run_sha:
        raise PipelineBlocked("run identity has no run_identity_sha256 for input video binding")
    return {
        "schema_version": "input-video-identity-v1",
        "video_file": record,
        "source_video_sha256": record["sha256"],
        "source_video_path": record["path"],
        "source_video_size_bytes": record["size_bytes"],
        "run_identity_sha256": run_sha,
    }


def _scoped_code_identity(code_identity_value: Mapping[str, Any], scope: str) -> dict[str, Any]:
    """Derive a versioned production-only identity for a resume stage.

    The broad run identity remains the producer guard.  Diagnostic consumer
    bindings use this narrower record so documentation/tests do not churn a
    plan, while nested LongSplat and non-consumer production code remain in
    the independent producer scope.
    """

    if scope not in {"producer", "convergence-consumer"}:
        raise ValueError(f"unsupported scoped code identity: {scope}")
    records = code_identity_value.get("files")
    selected: list[dict[str, Any]] = []
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                continue
            path = str(record["path"])
            if scope == "convergence-consumer":
                include = path in _CONVERGENCE_CONSUMER_CODE_PATHS
            else:
                include = path in _PRODUCER_CODE_PATHS or any(path.startswith(prefix) for prefix in _PRODUCER_CODE_PREFIXES)
            if include:
                selected.append({
                    "path": path,
                    "sha256": record.get("sha256"),
                    "size_bytes": record.get("size_bytes"),
                })
    selected.sort(key=lambda item: str(item["path"]))
    payload = {
        "schema_version": "stage-scoped-code-identity-v1",
        "scope": scope,
        "files": selected,
    }
    digest = stable_sha256(payload)
    return {**payload, "identity_sha256": digest, "code_identity_sha256": digest}


def _convergence_consumer_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    code = identity.get("code_identity")
    if not isinstance(code, Mapping):
        raise PipelineBlocked("run identity has no code identity for convergence consumer")
    consumer_code = _scoped_code_identity(code, "convergence-consumer")
    producer_code = _scoped_code_identity(code, "producer")
    return {
        **dict(identity),
        "code_identity": consumer_code,
        "producer_code_identity_sha256": producer_code["identity_sha256"],
    }


def _legacy_video_identity(run_dir: Path) -> dict[str, Any] | None:
    """Read the immutable raw-media source record for pre-v2 run identities."""

    candidate = run_dir / "raw-camera" / "camera" / "canonical_media-v1.json"
    if not candidate.is_file() or candidate.is_symlink():
        return None
    document = _load_json(candidate, "legacy canonical media identity")
    source = document.get("source")
    if not isinstance(source, Mapping):
        return None
    path = source.get("path")
    sha = source.get("sha256")
    size = source.get("size_bytes")
    if not isinstance(path, str) or not isinstance(sha, str) or not isinstance(size, int):
        return None
    return {"path": str(Path(path).resolve()), "sha256": sha, "size_bytes": size}


def _training_source_identity(
    *,
    input_result: Mapping[str, Any],
    input_video_identity: Mapping[str, Any],
    run_dir: Path,
    route: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Bind the immutable LongSplat input directory separately from the video."""

    payload = input_result.get("raw_result", input_result)
    if not isinstance(payload, Mapping):
        raise PipelineBlocked("LongSplat input payload is missing for training-source identity")
    training_value = input_result.get("source_path", payload.get("source_path"))
    staging_value = input_result.get("staging_manifest_path", payload.get("staging_manifest_path"))
    static_value = input_result.get("static_contract_path", payload.get("static_contract_path"))
    camera_value = input_result.get("camera_contract_path", payload.get("camera_contract_path"))
    if not all(isinstance(value, str) for value in (training_value, staging_value, static_value, camera_value)):
        raise PipelineBlocked("LongSplat input lacks training directory, staging manifest, static, or camera contract")
    run_root = _strict_path(run_dir, label="reconstruction run root", kind="directory")
    dynamic_output_root = _strict_path(
        output_root if output_root is not None else run_root.parent,
        label="dynamic output root",
        kind="directory",
    )
    if run_root == dynamic_output_root or run_root.parent != dynamic_output_root:
        raise PipelineBlocked("reconstruction run root must be a direct child of the dynamic output root")
    training_root = _strict_path(training_value, label="training source directory", kind="directory", root=run_root)
    video_sha = input_video_identity.get("source_video_sha256")
    if not isinstance(video_sha, str):
        raise PipelineBlocked("input video identity has no source_video_sha256")

    def contract_path(value: str, label: str) -> Path:
        return _strict_path(value, label=label, kind="file", root=training_root)

    staging_path = contract_path(staging_value, "staging manifest")
    static_path = contract_path(static_value, "static contract")
    camera_path = contract_path(camera_value, "camera contract")
    if staging_path == static_path or staging_path == camera_path or static_path == camera_path:
        raise PipelineBlocked("training source contract paths must be distinct")
    staging = _load_json(staging_path, "training staging manifest")
    static = _load_json(static_path, "training static contract")
    camera = _load_json(camera_path, "training camera contract")
    for label, document in (("staging manifest", staging), ("static contract", static), ("camera contract", camera)):
        declared = document.get("source_video_sha256")
        if declared is not None and declared != video_sha:
            raise PipelineBlocked(f"{label} source video SHA differs from input video identity")
    declared_training = staging.get("source_path")
    if isinstance(declared_training, str) and _strict_path(declared_training, label="staging source path", kind="directory", root=run_root) != training_root:
        raise PipelineBlocked("staging manifest training source directory differs from input result")
    declared_static_sha = payload.get("static_contract_sha256")
    if isinstance(declared_static_sha, str) and declared_static_sha != sha256_file(static_path):
        raise PipelineBlocked("static contract file SHA differs from LongSplat input payload")
    declared_camera_contract_sha = camera.get("contract_sha256")
    if not isinstance(declared_camera_contract_sha, str) or not declared_camera_contract_sha:
        raise PipelineBlocked("camera contract has no immutable contract SHA")
    pixel_keys = (
        "canonical_media_binding_sha256",
        "canonical_media_pixels_sha256",
        "undistorted_input_pixel_aggregate_sha256",
        "final_staged_pixel_aggregate_sha256",
    )
    pixel_aggregates: dict[str, str] = {}
    for key in pixel_keys:
        values = {
            str(value)
            for value in (payload.get(key), staging.get(key), static.get(key), camera.get(key))
            if isinstance(value, str) and value
        }
        if len(values) > 1:
            raise PipelineBlocked(f"training pixel aggregate drift for {key}")
        if values:
            pixel_aggregates[key] = next(iter(values))
    sparse_binary_value = staging.get("sparse_binary_evidence_path")
    sparse_text_value = staging.get("sparse_txt_evidence_path")
    if not isinstance(sparse_binary_value, str) or not isinstance(sparse_text_value, str):
        raise PipelineBlocked("staging manifest lacks immutable sparse model evidence paths")
    return {
        "schema_version": "training-source-identity-v2",
        "training_source_directory": str(training_root),
        "route_root": str(Path(route).resolve()),
        "output_root": str(dynamic_output_root),
        "run_root": str(run_root),
        "route_contained": True,
        "run_contained": True,
        "symlink_free": True,
        "source_video_sha256": video_sha,
        "immutable_contracts": {
            "staging_manifest": _strict_file_record(staging_path, label="staging manifest", root=training_root),
            "static_contract": {
                **_strict_file_record(static_path, label="static contract", root=training_root),
                "contract_sha256": static.get("static_contract_sha256", sha256_file(static_path)),
            },
            "camera_contract": {
                **_strict_file_record(camera_path, label="camera contract", root=training_root),
                "contract_sha256": declared_camera_contract_sha,
            },
        },
        "pixel_aggregates": pixel_aggregates,
        "sparse_model": {
            "binary": _explicit_sparse_model_identity(sparse_binary_value, label="sparse binary evidence", training_root=training_root),
            "text": _explicit_sparse_model_identity(sparse_text_value, label="sparse text evidence", training_root=training_root),
        },
    }


def _convergence_consumer_binding(
    *,
    consumer_identity: Mapping[str, Any],
    input_video_identity: Mapping[str, Any],
    training_source_identity: Mapping[str, Any],
    convergence_plan: Mapping[str, Any],
    coverage_visual_token: str | Path | None,
    coverage_decision: str,
) -> dict[str, Any]:
    """Build the immutable identity slice owned by the diagnostic consumer."""

    token: Mapping[str, Any] | None = None
    if coverage_decision in {"fail", "needs_review"}:
        if coverage_visual_token is None:
            raise PipelineBlocked("convergence diagnostic requires the coverage visual decision token")
        token = _visual_decision_valid(coverage_visual_token, label="coverage visual decision token")
        if token["decision"] != coverage_decision:
            raise PipelineBlocked("coverage visual token decision differs from the consumed coverage gate decision")
    elif coverage_decision != "not_applicable":
        raise PipelineBlocked("convergence consumer requires fail/needs_review evidence or the default not_applicable decision")
    code = consumer_identity.get("code_identity")
    code_sha = code.get("code_identity_sha256") if isinstance(code, Mapping) else None
    producer_code_sha = consumer_identity.get("producer_code_identity_sha256")
    source_sha = consumer_identity.get("source_video_sha256")
    run_sha = consumer_identity.get("run_identity_sha256")
    if not all(isinstance(value, str) and value for value in (source_sha, run_sha, code_sha, producer_code_sha)):
        raise PipelineBlocked("consumer identity is missing source, run, consumer code, or producer code identity")
    if input_video_identity.get("source_video_sha256") != source_sha:
        raise PipelineBlocked("input video identity differs from consumer run source identity")
    if input_video_identity.get("run_identity_sha256") != run_sha:
        raise PipelineBlocked("input video identity differs from consumer run identity")
    identity_binding = convergence_plan.get("identity_binding")
    if not isinstance(identity_binding, Mapping):
        raise PipelineBlocked("convergence plan has no identity binding")
    order_sha = identity_binding.get("camera_order_sha256")
    if not isinstance(order_sha, str) or not order_sha:
        raise PipelineBlocked("convergence plan has no camera order identity")
    count = convergence_plan.get("active_camera_count")
    order = convergence_plan.get("active_camera_order")
    iterations = convergence_plan.get("requested_iterations")
    if not isinstance(count, int) or count <= 0 or not isinstance(order, list) or len(order) != count or len(order) != len(set(order)):
        raise PipelineBlocked("convergence plan camera order is not unique and exact")
    if iterations != 1000:
        raise PipelineBlocked("convergence diagnostic plan must use convergence1000-v1 with 1000 iterations")
    return {
        "schema_version": "convergence-consumer-binding-v3",
        "profile_id": "convergence1000-v1",
        "consumer_run_identity_sha256": run_sha,
        "consumer_code_identity_sha256": code_sha,
        "producer_code_identity_sha256": producer_code_sha,
        "input_video_identity": dict(input_video_identity),
        "training_source_identity": dict(training_source_identity),
        "camera_order_sha256": order_sha,
        "active_camera_count": count,
        "active_camera_order": list(order),
        "requested_iterations": iterations,
        "coverage_visual_token_path": None if token is None else token["path"],
        "coverage_visual_token_sha256": None if token is None else token["sha256"],
        "coverage_visual_decision": coverage_decision,
        "coverage_decision_source": "default_pipeline_not_applicable" if token is None else "explicit_supervisor_token",
        "depth_source": "disabled",
        "external_colmap_pose": True,
        "training_mode": "fixed_pose_rgb_only",
    }


def _convergence_plan_comparison(
    classification: str,
    reason: str,
    diff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a machine-readable plan/consumer comparison outcome."""

    if classification not in {"reusable_exact", "stale_rebuildable", "unsafe_drift"}:
        raise ValueError(f"unsupported convergence plan comparison: {classification}")
    return {
        "schema_version": "convergence-plan-consumer-comparison-v1",
        "classification": classification,
        "reusable_exact": classification == "reusable_exact",
        "stale_rebuildable": classification == "stale_rebuildable",
        "unsafe_drift": classification == "unsafe_drift",
        "reason": reason,
        "diff": {} if diff is None else dict(diff),
    }


def _compare_convergence_binding(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare immutable binding fields without hiding migration reasons."""

    stale: list[str] = []
    unsafe: list[str] = []
    volatile_top_level = {
        "schema_version",
        "consumer_code_identity_sha256",
        # This digest contains broad run/code identity.  Its producer and
        # consumer scopes are compared independently below; a scope-only
        # change must be rebuildable, not an immutable-data drift.
        "consumer_run_identity_sha256",
    }
    for key, expected_value in expected.items():
        if key in volatile_top_level:
            if key not in observed:
                stale.append(key)
            elif observed.get(key) != expected_value:
                stale.append(key)
            continue
        if key not in observed:
            stale.append(key)
            continue
        observed_value = observed.get(key)
        if key == "input_video_identity" and isinstance(observed_value, Mapping) and isinstance(expected_value, Mapping):
            for nested_key, nested_expected in expected_value.items():
                if nested_key == "run_identity_sha256":
                    continue
                if nested_key not in observed_value:
                    stale.append(f"{key}.{nested_key}")
                elif observed_value.get(nested_key) != nested_expected:
                    unsafe.append(f"{key}.{nested_key}")
        elif key == "training_source_identity" and isinstance(observed_value, Mapping) and isinstance(expected_value, Mapping):
            for nested_key, nested_expected in expected_value.items():
                if nested_key not in observed_value:
                    stale.append(f"{key}.{nested_key}")
                elif observed_value.get(nested_key) != nested_expected:
                    unsafe.append(f"{key}.{nested_key}")
        elif observed_value != expected_value:
            unsafe.append(key)
    if unsafe:
        return _convergence_plan_comparison(
            "unsafe_drift",
            "immutable convergence consumer binding drift",
            {"unsafe_fields": sorted(unsafe), "stale_fields": sorted(stale)},
        )
    extra = sorted(set(observed) - set(expected))
    if stale or extra or dict(observed) != dict(expected):
        return _convergence_plan_comparison(
            "stale_rebuildable",
            "plan uses an older consumer binding or stage-scoped code identity",
            {"stale_fields": sorted(stale), "legacy_extra_fields": extra},
        )
    return _convergence_plan_comparison("reusable_exact", "plan consumer binding is exact")


def _convergence_plan_matches_consumer(
    *,
    plan_result: Mapping[str, Any],
    input_result: Mapping[str, Any],
    input_video_identity: Mapping[str, Any],
    consumer_identity: Mapping[str, Any],
    run_dir: Path,
    route: Path,
    output_root: Path | None = None,
    coverage_visual_token: str | Path | None,
    coverage_decision: str | None,
) -> dict[str, Any]:
    """Classify a passed CPU plan against the current diagnostic consumer.

    ``stale_rebuildable`` is deliberately distinct from ``unsafe_drift``:
    migration/schema or consumer-scope changes create a new append-only plan,
    while source, camera, token, profile, pose, or producer changes stop the
    run without silently accepting the old plan.
    """

    binding = plan_result.get("consumer_binding")
    if binding is None:
        return _convergence_plan_comparison(
            "stale_rebuildable",
            "plan has no consumer binding; append-only consumer plan rebuild required",
            {"missing": ["consumer_binding"]},
        )
    if not isinstance(binding, Mapping):
        return _convergence_plan_comparison(
            "unsafe_drift",
            "convergence plan consumer binding is malformed",
            {"binding_type": type(binding).__name__},
        )
    convergence_plan = plan_result.get("convergence_plan")
    if not isinstance(convergence_plan, Mapping):
        return _convergence_plan_comparison(
            "unsafe_drift",
            "convergence plan result is missing its plan payload",
            {"missing": ["convergence_plan"]},
        )
    if coverage_visual_token is None or coverage_decision not in {"fail", "needs_review"}:
        return _convergence_plan_comparison(
            "unsafe_drift",
            "convergence resume requires the explicit coverage visual decision token",
            {"coverage_decision": coverage_decision},
        )
    if plan_result.get("profile_id") != "convergence1000-v1" or convergence_plan.get("requested_iterations") != 1000:
        return _convergence_plan_comparison(
            "unsafe_drift",
            "convergence plan profile or iteration identity differs on resume",
            {
                "profile_id": plan_result.get("profile_id"),
                "requested_iterations": convergence_plan.get("requested_iterations"),
            },
        )
    try:
        training_source_identity = _training_source_identity(
            input_result=input_result,
            input_video_identity=input_video_identity,
            run_dir=run_dir,
            route=route,
            output_root=output_root,
        )
        camera_path = Path(training_source_identity["immutable_contracts"]["camera_contract"]["path"])
        camera_document = _load_json(camera_path, "convergence camera contract")
        current_order = camera_document.get("frame_names", camera_document.get("camera_names"))
        plan_order = convergence_plan.get("active_camera_order")
        if (
            not isinstance(current_order, list)
            or not current_order
            or len(current_order) != len(set(current_order))
            or plan_order != current_order
            or convergence_plan.get("active_camera_count") != len(current_order)
        ):
            return _convergence_plan_comparison(
                "unsafe_drift",
                "convergence plan camera order/count differs from current camera contract",
                {
                    "plan_count": convergence_plan.get("active_camera_count"),
                    "current_count": len(current_order) if isinstance(current_order, list) else None,
                },
            )
        expected = _convergence_consumer_binding(
            consumer_identity=consumer_identity,
            input_video_identity=input_video_identity,
            training_source_identity=training_source_identity,
            convergence_plan=convergence_plan,
            coverage_visual_token=coverage_visual_token,
            coverage_decision=coverage_decision,
        )
    except (PipelineBlocked, OSError, ValueError, KeyError, TypeError) as exc:
        return _convergence_plan_comparison(
            "unsafe_drift",
            "current immutable convergence consumer evidence cannot be verified",
            {"error": str(exc)},
        )
    binding_comparison = _compare_convergence_binding(binding, expected)
    if binding_comparison["classification"] == "unsafe_drift":
        return binding_comparison
    plan_path_value = plan_result.get("convergence_plan_path")
    validation_path_value = plan_result.get("convergence_validation_path")
    if not isinstance(plan_path_value, str) or not isinstance(validation_path_value, str):
        return _convergence_plan_comparison(
            "unsafe_drift",
            "convergence plan result is missing immutable plan/validation paths",
            {"missing": ["convergence_plan_path", "convergence_validation_path"]},
        )
    try:
        plan_path = _strict_path(plan_path_value, label="convergence plan", kind="file", root=run_dir)
        validation_path = _strict_path(validation_path_value, label="convergence validation", kind="file", root=run_dir)
        _load_json(plan_path, "convergence plan")
        validation = _load_json(validation_path, "convergence validation")
    except (PipelineBlocked, OSError, ValueError) as exc:
        return _convergence_plan_comparison(
            "unsafe_drift",
            "convergence plan or validation artifact is missing, escaped, or mutated",
            {"error": str(exc)},
        )
    validation_binding = validation.get("consumer_binding")
    if not isinstance(validation_binding, Mapping):
        return _convergence_plan_comparison(
            "unsafe_drift",
            "convergence validation has no consumer binding",
            {"missing": ["validation.consumer_binding"]},
        )
    validation_comparison = _compare_convergence_binding(validation_binding, expected)
    if validation_comparison["classification"] == "unsafe_drift":
        return validation_comparison
    if binding_comparison["classification"] == "stale_rebuildable" or validation_comparison["classification"] == "stale_rebuildable":
        return _convergence_plan_comparison(
            "stale_rebuildable",
            "plan or validation uses an older consumer binding; append-only rebuild required",
            {
                "plan": binding_comparison.get("diff", {}),
                "validation": validation_comparison.get("diff", {}),
            },
        )
    return _convergence_plan_comparison("reusable_exact", "plan and validation consumer bindings are exact")


def _coverage_plan_stage(
    *,
    ledger: RunLedger,
    input_result: Mapping[str, Any],
    frames_result: Mapping[str, Any] | None,
    route: Path,
    plan: bool,
) -> tuple[dict[str, Any], str]:
    """Create the dynamic CPU coverage plan inside the same run ledger."""

    if plan:
        return _stage_result(
            stage="coverage-smoke-plan",
            status="planned",
            computed_pass=False,
            reason="coverage-smoke-v1 dynamic CPU plan planned",
            plan=True,
            gpu_invoked=False,
            formal_auto_release=False,
        ), "planned"
    payload = input_result.get("raw_result", input_result)
    if not isinstance(payload, Mapping):
        result = _stage_result(stage="coverage-smoke-plan", status="blocked", computed_pass=False, reason="LongSplat input payload is missing", plan=False, gpu_invoked=False, formal_auto_release=False)
        _record_stage(ledger, "coverage-smoke-plan", result, status="blocked")
        return result, "blocked"
    camera_contract_value = payload.get("camera_contract_path")
    if not isinstance(camera_contract_value, str):
        result = _stage_result(stage="coverage-smoke-plan", status="blocked", computed_pass=False, reason="LongSplat input has no immutable camera contract", plan=False, gpu_invoked=False, formal_auto_release=False)
        _record_stage(ledger, "coverage-smoke-plan", result, status="blocked")
        return result, "blocked"
    frame_payload = frames_result.get("raw_result", frames_result) if isinstance(frames_result, Mapping) else {}
    frame_selection = _first_payload_artifact(frame_payload, "frame_selection-v1.json")
    if frame_selection is None:
        raw_run_dir = frames_result.get("raw_run_dir") if isinstance(frames_result, Mapping) else None
        if isinstance(raw_run_dir, str):
            candidate = Path(raw_run_dir) / "frame_selection-v1.json"
            if candidate.is_file() and not candidate.is_symlink():
                frame_selection = candidate.resolve()
    if frame_selection is None:
        result = _stage_result(stage="coverage-smoke-plan", status="blocked", computed_pass=False, reason="frame-selection-v1 evidence is required for a dynamic coverage cap", plan=False, gpu_invoked=False, formal_auto_release=False)
        _record_stage(ledger, "coverage-smoke-plan", result, status="blocked")
        return result, "blocked"
    attempt = ledger.begin_attempt("coverage-smoke-plan", {"camera_contract_path": camera_contract_value, "frame_selection_path": str(frame_selection), "policy": "coverage-smoke-v1"})
    try:
        from .coverage_smoke import load_camera_contract, load_frame_selection, plan_coverage_smoke, validate_coverage_smoke_plan

        camera = load_camera_contract(camera_contract_value)
        selection = load_frame_selection(frame_selection)
        coverage_plan = plan_coverage_smoke(
            active_camera_count=int(camera["camera_count"]),
            camera_names=camera["camera_names"],
            frame_selector_max_frames=int(selection["max_frames"]),
            camera_contract_path=camera["path"],
            frame_selection_path=selection["path"],
        )
        coverage_plan.update(
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
        validation = validate_coverage_smoke_plan(coverage_plan)
        plan_path = attempt / "coverage-smoke-plan-v1.json"
        validation_path = attempt / "coverage-smoke-validation-v1.json"
        _write_json_once(plan_path, coverage_plan)
        validation = {
            **validation,
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "camera_contract_path": camera["path"],
            "camera_contract_sha256": camera["sha256"],
            "frame_selection_path": selection["path"],
            "frame_selection_sha256": selection["sha256"],
            "gpu_invoked": False,
            "formal_auto_release": False,
        }
        _write_json_once(validation_path, validation)
        helper_result_path = attempt / "coverage-smoke-result-v1.json"
        helper_result = {
            "schema_version": "coverage-smoke-result-v1",
            "stage": "coverage-smoke-plan",
            "status": "passed" if validation["computed_pass"] else "blocked",
            "computed_pass": bool(validation["computed_pass"]),
            "gpu_invoked": False,
            "training_invoked": False,
            "formal_auto_release": False,
            "coverage_plan": {
                "path": str(plan_path),
                "sha256": sha256_file(plan_path),
            },
            "coverage_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
        }
        _write_json_once(helper_result_path, helper_result)
        result = _stage_result(
            stage="coverage-smoke-plan",
            status="passed" if validation["computed_pass"] else "blocked",
            computed_pass=bool(validation["computed_pass"]),
            reason="dynamic coverage-smoke-v1 plan validated" if validation["computed_pass"] else "coverage plan reached a structured stop",
            plan=False,
            artifacts=_artifacts([plan_path, validation_path, helper_result_path]),
            gpu_invoked=False,
            training_invoked=False,
            formal_auto_release=False,
            coverage_plan_path=str(plan_path),
            coverage_validation_path=str(validation_path),
            coverage_plan=coverage_plan,
            coverage_validation=validation,
        )
        status = "passed" if result["computed_pass"] else "blocked"
        ledger.finish_attempt(stage="coverage-smoke-plan", attempt=attempt, status=status, result=result)
        return result, status
    except Exception as exc:
        result = _stage_result(stage="coverage-smoke-plan", status="blocked", computed_pass=False, reason=str(exc), plan=False, gpu_invoked=False, formal_auto_release=False)
        ledger.finish_attempt(stage="coverage-smoke-plan", attempt=attempt, status="blocked", result=result)
        return result, "blocked"


def _convergence_plan_stage(
    *,
    ledger: RunLedger,
    input_result: Mapping[str, Any],
    route: Path,
    output_root: Path | None = None,
    plan: bool,
    input_video_identity: Mapping[str, Any] | None = None,
    consumer_identity: Mapping[str, Any] | None = None,
    coverage_visual_token: str | Path | None = None,
    coverage_decision: str | None = None,
    rebuild_reason: Mapping[str, Any] | None = None,
    default_pipeline: bool = False,
) -> tuple[dict[str, Any], str]:
    """Build the fixed, explicit convergence1000-v1 CPU diagnostic plan."""

    if plan:
        return _stage_result(
            stage="convergence-smoke-plan",
            status="planned",
            computed_pass=False,
            reason="convergence1000-v1 one-shot diagnostic plan planned",
            plan=True,
            gpu_invoked=False,
            diagnostic_only=True,
            formal_auto_release=False,
        ), "planned"
    payload = input_result.get("raw_result", input_result)
    if not isinstance(payload, Mapping):
        result = _stage_result(stage="convergence-smoke-plan", status="blocked", computed_pass=False, reason="LongSplat input payload is missing", plan=False, gpu_invoked=False, diagnostic_only=True, formal_auto_release=False)
        _record_stage(ledger, "convergence-smoke-plan", result, status="blocked")
        return result, "blocked"
    training_source_value = input_result.get("source_path", payload.get("source_path"))
    parent_value = input_result.get("plan_path", payload.get("future_smoke_plan_path"))
    static_value = input_result.get("static_contract_path", payload.get("static_contract_path"))
    if not all(isinstance(value, str) for value in (training_source_value, parent_value, static_value)):
        result = _stage_result(stage="convergence-smoke-plan", status="blocked", computed_pass=False, reason="convergence input lacks training-source-directory/parent/static contract paths", plan=False, gpu_invoked=False, diagnostic_only=True, formal_auto_release=False)
        _record_stage(ledger, "convergence-smoke-plan", result, status="blocked")
        return result, "blocked"
    training_source = Path(str(training_source_value)).absolute()
    parent_plan = Path(str(parent_value)).absolute()
    static_contract = Path(str(static_value)).absolute()
    attempt = ledger.begin_attempt(
        "convergence-smoke-plan",
        {
            "profile": "convergence1000-v1",
            "training_source_directory": str(training_source),
            "input_video_identity": None if input_video_identity is None else dict(input_video_identity),
            "parent_plan": _artifact(parent_plan),
            "static_contract": _artifact(static_contract),
            "gpu_invoked": False,
            "diagnostic_only": True,
            "rebuild_reason": None if rebuild_reason is None else dict(rebuild_reason),
        },
    )
    try:
        from .convergence_smoke import plan_convergence_smoke, validate_convergence_smoke_plan

        if input_video_identity is None:
            raise PipelineBlocked("convergence diagnostic plan requires an input video identity")
        training_source_identity = _training_source_identity(
            input_result=input_result,
            input_video_identity=input_video_identity,
            run_dir=ledger.run_dir,
            route=route,
            output_root=output_root,
        )
        training_source = Path(training_source_identity["training_source_directory"])
        static_contract = Path(training_source_identity["immutable_contracts"]["static_contract"]["path"])
        camera_contract = Path(training_source_identity["immutable_contracts"]["camera_contract"]["path"])
        parent_plan = _strict_path(parent_plan, label="convergence parent plan", kind="file", root=ledger.run_dir)
        source_document = _load_json(static_contract, "convergence static contract")
        camera_document = _load_json(camera_contract, "convergence camera contract")
        names = camera_document.get("frame_names")
        camera = camera_document.get("camera")
        if not isinstance(names, list) or not names or len(names) != len(set(names)) or not isinstance(camera, Mapping):
            raise PipelineBlocked("convergence camera contract has no exact unique order")
        pose_reference = None
        for key in ("pose_reference_path", "external_pose_contract_path", "external_colmap_pose_contract_path", "pose_contract_path"):
            value = payload.get(key)
            if isinstance(value, str):
                candidate = Path(value).resolve()
                if candidate.is_file() and not candidate.is_symlink():
                    pose_reference = candidate
                    break
        convergence_plan = plan_convergence_smoke(
            active_camera_count=len(names),
            camera_names=[str(name) for name in names],
            source_video_sha256=input_video_identity["source_video_sha256"],
            source_path=training_source,
            static_contract_path=static_contract,
            static_contract_sha256=sha256_file(static_contract),
            camera_contract_path=camera_contract,
            camera_contract_sha256=sha256_file(camera_contract),
            pose_reference_path=pose_reference,
            pose_reference_sha256=None if pose_reference is None else sha256_file(pose_reference),
            camera=camera,
        )
        validation = validate_convergence_smoke_plan(convergence_plan)
        if consumer_identity is None:
            raise PipelineBlocked("convergence diagnostic plan requires a consumer identity")
        if default_pipeline:
            coverage_visual_token = None
            coverage_decision = "not_applicable"
        else:
            if coverage_visual_token is None:
                raise PipelineBlocked("convergence diagnostic plan requires the coverage visual token")
            if coverage_decision not in {"fail", "needs_review"}:
                raise PipelineBlocked("convergence diagnostic plan requires a fail/needs_review coverage decision")
        plan_identity = dict(convergence_plan["identity_binding"])
        plan_identity.pop("source_path", None)
        plan_identity.update(
            {
                "input_video_file": input_video_identity["video_file"]["path"],
                "input_video_sha256": input_video_identity["source_video_sha256"],
                "training_source_directory": training_source_identity["training_source_directory"],
                "training_staging_manifest_sha256": training_source_identity["immutable_contracts"]["staging_manifest"]["sha256"],
                "training_static_contract_file_sha256": training_source_identity["immutable_contracts"]["static_contract"]["sha256"],
                "training_camera_contract_file_sha256": training_source_identity["immutable_contracts"]["camera_contract"]["sha256"],
            }
        )
        convergence_plan = {**convergence_plan, "identity_binding": plan_identity}
        validation = validate_convergence_smoke_plan(convergence_plan)
        consumer_binding = _convergence_consumer_binding(
            consumer_identity=consumer_identity,
            input_video_identity=input_video_identity,
            training_source_identity=training_source_identity,
            convergence_plan=convergence_plan,
            coverage_visual_token=coverage_visual_token,
            coverage_decision=str(coverage_decision),
        )
        plan_path = attempt / "convergence-smoke-plan-v1.json"
        validation_path = attempt / "convergence-smoke-validation-v1.json"
        _write_json_once(plan_path, convergence_plan)
        validation = {
            **validation,
            "plan_path": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            "parent_plan_path": str(parent_plan),
            "parent_plan_sha256": sha256_file(parent_plan),
            "static_contract_path": str(static_contract),
            "static_contract_sha256": sha256_file(static_contract),
            "camera_contract_path": str(camera_contract),
            "camera_contract_sha256": sha256_file(camera_contract),
            "gpu_invoked": False,
            "formal_auto_release": False,
            "diagnostic_only": True,
            "consumer_binding": consumer_binding,
        }
        _write_json_once(validation_path, validation)
        helper_path = attempt / "convergence-smoke-result-v1.json"
        helper = {
            "schema_version": "convergence-smoke-result-v1",
            "stage": "convergence-smoke-plan",
            "status": "passed",
            "computed_pass": True,
            "profile": "convergence1000-v1",
            "gpu_invoked": False,
            "training_invoked": False,
            "formal_auto_release": False,
            "diagnostic_only": True,
            "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
            "validation": {"path": str(validation_path), "sha256": sha256_file(validation_path)},
            "consumer_binding": consumer_binding,
        }
        _write_json_once(helper_path, helper)
        result = _stage_result(
            stage="convergence-smoke-plan",
            status="passed",
            computed_pass=True,
            reason="convergence1000-v1 dynamic camera-bound diagnostic plan validated",
            plan=False,
            artifacts=_artifacts([plan_path, validation_path, helper_path, parent_plan, static_contract, camera_contract]),
            gpu_invoked=False,
            training_invoked=False,
            diagnostic_only=True,
            default_pipeline=default_pipeline,
            formal_auto_release=False,
            profile_id="convergence1000-v1",
            convergence_plan_path=str(plan_path),
            convergence_validation_path=str(validation_path),
            convergence_plan=convergence_plan,
            convergence_validation=validation,
            consumer_binding=consumer_binding,
            parent_plan_path=str(parent_plan),
            static_contract_path=str(static_contract),
            input_video_identity=input_video_identity,
            training_source_identity=training_source_identity,
            training_source_directory=str(training_source),
            plan_rebuild_reason=None if rebuild_reason is None else dict(rebuild_reason),
        )
        ledger.finish_attempt(stage="convergence-smoke-plan", attempt=attempt, status="passed", result=result)
        return result, "passed"
    except Exception as exc:
        result = _stage_result(stage="convergence-smoke-plan", status="blocked", computed_pass=False, reason=str(exc), plan=False, gpu_invoked=False, diagnostic_only=True, formal_auto_release=False)
        ledger.finish_attempt(stage="convergence-smoke-plan", attempt=attempt, status="blocked", result=result)
        return result, "blocked"


def _coverage_visual_gate(
    *,
    ledger: RunLedger,
    smoke: Mapping[str, Any],
    coverage_plan: Mapping[str, Any],
    coverage_render: Mapping[str, Any],
    coverage_postcheck: Mapping[str, Any],
    decision_token: str | Path | None,
    diagnostic_profile: str | None = None,
) -> tuple[dict[str, Any], str]:
    if diagnostic_profile not in {None, "convergence1000-v1"}:
        raise PipelineBlocked("coverage visual gate accepts only the versioned convergence1000-v1 diagnostic profile")
    evidence_artifacts = [dict(item) for item in coverage_postcheck.get("artifacts", []) if isinstance(item, Mapping)]
    evidence_binding = {
        "postcheck_result_path": coverage_postcheck.get("postcheck_result_path"),
        "quality_evidence_path": coverage_postcheck.get("quality_evidence_path"),
        "postcheck_result": coverage_postcheck.get("postcheck_result"),
        "quality_evidence": coverage_postcheck.get("quality_evidence"),
    }
    required_evidence = (
        coverage_postcheck.get("computed_pass") is True
        and isinstance(coverage_postcheck.get("postcheck_result_path"), str)
        and isinstance(coverage_postcheck.get("quality_evidence_path"), str)
        and isinstance(coverage_postcheck.get("postcheck_result"), Mapping)
    )
    if not required_evidence:
        result = _stage_result(
            stage="coverage-smoke-visual-gate",
            status="blocked",
            computed_pass=False,
            reason="coverage render CPU postcheck evidence is required before visual decision",
            plan=False,
            artifacts=evidence_artifacts,
            gpu_invoked=False,
            coverage_smoke_visual_decision="not_run",
            formal_release_eligible=False,
            diagnostic_release_eligible=False,
            diagnostic_profile=diagnostic_profile,
            formal_auto_release=False,
            evidence=evidence_binding,
        )
        _record_stage(ledger, "coverage-smoke-visual-gate", result, status="blocked")
        return result, "blocked"
    if decision_token is None:
        result = _stage_result(
            stage="coverage-smoke-visual-gate",
            status="blocked",
            computed_pass=False,
            reason="explicit coverage-smoke visual decision token is required; smoke100 remains structural-only",
            plan=False,
            artifacts=evidence_artifacts,
            gpu_invoked=False,
            coverage_smoke_visual_decision="not_run",
            formal_release_eligible=False,
            diagnostic_release_eligible=False,
            diagnostic_profile=diagnostic_profile,
            formal_auto_release=False,
            evidence=evidence_binding,
        )
        _record_stage(ledger, "coverage-smoke-visual-gate", result, status="blocked")
        return result, "blocked"
    try:
        decision = _visual_decision_valid(decision_token, label="coverage visual decision token")
        expected_evidence = _visual_evidence_binding(coverage_postcheck, label="coverage visual decision token")
        _decision_binds_evidence(decision["record"], expected_evidence, label="coverage visual decision token")
        from .coverage_smoke import formal_release_decision

        gate = formal_release_decision(
            smoke_structural_pass=smoke.get("computed_pass") is True or smoke.get("smoke100_structural_pass") is True,
            coverage_visual_decision=decision["decision"],
            explicit_visual_review=True,
        )
        eligible = bool(gate["formal_release_eligible"] and coverage_render.get("computed_pass") is True and coverage_plan.get("computed_pass") is True)
        diagnostic_eligible = bool(
            not eligible
            and diagnostic_profile == "convergence1000-v1"
            and decision["decision"] in {"needs_review", "fail"}
            and coverage_render.get("computed_pass") is True
            and coverage_plan.get("computed_pass") is True
        )
        result = _stage_result(
            stage="coverage-smoke-visual-gate",
            status="passed" if eligible else "blocked",
            computed_pass=eligible,
            reason="explicit coverage visual decision released the formal gate" if eligible else "coverage visual decision did not release formal training",
            plan=False,
            artifacts=[*evidence_artifacts, decision],
            gpu_invoked=False,
            coverage_smoke_visual_decision=decision["decision"],
            formal_release_eligible=eligible,
            diagnostic_release_eligible=diagnostic_eligible,
            diagnostic_profile=diagnostic_profile,
            formal_auto_release=False,
            gate=gate,
            evidence={"postcheck": evidence_binding, "decision_binding": expected_evidence},
        )
    except PipelineBlocked as exc:
        result = _stage_result(
            stage="coverage-smoke-visual-gate",
            status="blocked",
            computed_pass=False,
            reason=str(exc),
            plan=False,
            artifacts=evidence_artifacts,
            gpu_invoked=False,
            formal_auto_release=False,
            formal_release_eligible=False,
            diagnostic_release_eligible=False,
            diagnostic_profile=diagnostic_profile,
            evidence=evidence_binding,
        )
    _record_stage(ledger, "coverage-smoke-visual-gate", result, status="passed" if result["computed_pass"] else "blocked")
    return result, "passed" if result["computed_pass"] else "blocked"


def _convergence_visual_gate(
    *,
    ledger: RunLedger,
    convergence_postcheck: Mapping[str, Any],
    decision_token: str | Path | None,
) -> tuple[dict[str, Any], str]:
    """Stop at an explicit diagnostic visual gate; never release formal."""

    evidence_artifacts = [dict(item) for item in convergence_postcheck.get("artifacts", []) if isinstance(item, Mapping)]
    evidence_binding: dict[str, Any] = {
        "postcheck_result_path": convergence_postcheck.get("postcheck_result_path"),
        "quality_evidence_path": convergence_postcheck.get("quality_evidence_path"),
        "postcheck_result": convergence_postcheck.get("postcheck_result"),
        "quality_evidence": convergence_postcheck.get("quality_evidence"),
    }
    if convergence_postcheck.get("computed_pass") is not True:
        result = _stage_result(
            stage="convergence-smoke-visual-gate",
            status="blocked",
            computed_pass=False,
            reason="convergence CPU postcheck evidence is required before diagnostic visual review",
            plan=False,
            artifacts=evidence_artifacts,
            gpu_invoked=False,
            convergence_visual_decision="not_run",
            diagnostic_only=True,
            formal_release_eligible=False,
            formal_auto_release=False,
            evidence=evidence_binding,
        )
        _record_stage(ledger, "convergence-smoke-visual-gate", result, status="blocked")
        return result, "blocked"
    if decision_token is None:
        result = _stage_result(
            stage="convergence-smoke-visual-gate",
            status="blocked",
            computed_pass=False,
            reason="explicit convergence diagnostic visual decision token is required; formal gate remains closed",
            plan=False,
            artifacts=evidence_artifacts,
            gpu_invoked=False,
            convergence_visual_decision="not_run",
            diagnostic_only=True,
            formal_release_eligible=False,
            formal_auto_release=False,
            evidence=evidence_binding,
        )
        _record_stage(ledger, "convergence-smoke-visual-gate", result, status="blocked")
        return result, "blocked"
    try:
        decision = _visual_decision_valid(decision_token, label="convergence visual decision token")
        expected_evidence = _visual_evidence_binding(convergence_postcheck, label="convergence visual decision token")
        _decision_binds_evidence(decision["record"], expected_evidence, label="convergence visual decision token")
        result = _stage_result(
            stage="convergence-smoke-visual-gate",
            status="passed",
            computed_pass=True,
            reason="explicit diagnostic visual decision recorded; formal release remains closed",
            plan=False,
            artifacts=[*evidence_artifacts, decision],
            gpu_invoked=False,
            convergence_visual_decision=decision["decision"],
            diagnostic_only=True,
            formal_release_eligible=False,
            formal_auto_release=False,
            formal_gate="closed",
            evidence={"postcheck": evidence_binding, "decision_binding": expected_evidence},
            decision=decision,
        )
    except PipelineBlocked as exc:
        result = _stage_result(
            stage="convergence-smoke-visual-gate",
            status="blocked",
            computed_pass=False,
            reason=str(exc),
            plan=False,
            artifacts=evidence_artifacts,
            gpu_invoked=False,
            convergence_visual_decision="not_run",
            diagnostic_only=True,
            formal_release_eligible=False,
            formal_auto_release=False,
            evidence=evidence_binding,
        )
    _record_stage(ledger, "convergence-smoke-visual-gate", result, status="passed" if result["computed_pass"] else "blocked")
    return result, "passed" if result["computed_pass"] else "blocked"


def _run_default_single_convergence(
    *,
    ledger: RunLedger,
    route: Path,
    run_output_root: Path,
    input_result: Mapping[str, Any],
    input_video_identity: Mapping[str, Any],
    consumer_identity: Mapping[str, Any],
    frames_result: Mapping[str, Any] | None,
    colmap_result: Mapping[str, Any] | None,
    stop_after: str,
    execute_gpu: bool,
) -> tuple[dict[str, Any], str | None]:
    """Run the default's single 1000-it pass and automated early gate."""

    results: dict[str, Any] = {}
    plan_result = _stage_reusable(ledger, "convergence-smoke-plan")
    if plan_result is None:
        plan_result, status = _convergence_plan_stage(
            ledger=ledger,
            input_result=input_result,
            route=route,
            output_root=run_output_root,
            plan=False,
            input_video_identity=input_video_identity,
            consumer_identity=consumer_identity,
            coverage_visual_token=None,
            coverage_decision="not_applicable",
            default_pipeline=True,
        )
        results["convergence-smoke-plan"] = plan_result
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=plan_result["reason"])
            return results, plan_result["reason"]
    else:
        plan_result = {**plan_result, "reused": True}
        results["convergence-smoke-plan"] = plan_result
    if stop_after == "convergence-smoke-plan":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after convergence-smoke-plan")
        return results, None

    base_input = _profile_input(
        input_result.get("raw_result", input_result),
        "convergence1000-v1",
    )
    base_input.update(
        {
            "profile": "convergence1000-v1",
            "parent_plan_path": base_input["plan_path"],
            "convergence_plan_path": plan_result["convergence_plan_path"],
        }
    )
    training = _stage_reusable(ledger, "convergence-smoke-training")
    if training is None:
        training, status = _execute_training_or_render(
            ledger=ledger,
            stage="convergence-smoke-training",
            profile_input=base_input,
            plan=False,
            execute_gpu=execute_gpu,
            route=route,
            prepare=lambda attempt, profile: _prepare_convergence_training_plan(attempt, profile, route),
            extra={"diagnostic_only": True, "formal_auto_release": False, "coverage_visual_decision": "not_applicable"},
        )
        results["convergence-smoke-training"] = training
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=training["reason"])
            return results, training["reason"]
    else:
        training = {**training, "reused": True}
        results["convergence-smoke-training"] = training
    if stop_after == "convergence-smoke-training":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after convergence-smoke-training")
        return results, None

    render_input = dict(base_input)
    if isinstance(training.get("plan_path"), str):
        render_input["plan_path"] = training["plan_path"]
    render = _stage_reusable(ledger, "convergence-smoke-render")
    if render is None:
        render, status = _execute_training_or_render(
            ledger=ledger,
            stage="convergence-smoke-render",
            profile_input=render_input,
            plan=False,
            execute_gpu=execute_gpu,
            route=route,
            training_evidence_root=Path(str(training.get("executor_root", ""))),
            isolated_render_snapshot=True,
            extra={"diagnostic_only": True, "formal_auto_release": False},
        )
        results["convergence-smoke-render"] = render
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=render["reason"])
            return results, render["reason"]
    else:
        render = {**render, "reused": True}
        results["convergence-smoke-render"] = render
    if stop_after == "convergence-smoke-render":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after convergence-smoke-render")
        return results, None

    postcheck = _stage_reusable(ledger, "convergence-smoke-render-postcheck")
    if postcheck is None:
        postcheck, status = _convergence_render_postcheck_stage(
            ledger=ledger,
            route=route,
            convergence_render=render,
            convergence_training=training,
            training_input=base_input,
            frames_result=frames_result,
            colmap_result=colmap_result,
            baseline_render_attempt=None,
            comparison_render_attempt=None,
            plan=False,
        )
        results["convergence-smoke-render-postcheck"] = postcheck
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=postcheck["reason"])
            return results, postcheck["reason"]
    else:
        postcheck = {**postcheck, "reused": True}
        results["convergence-smoke-render-postcheck"] = postcheck
    if stop_after == "convergence-smoke-render-postcheck":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after convergence-smoke-render-postcheck")
        return results, None

    gate = _stage_reusable(ledger, "automated-early-gate")
    if gate is None:
        gate, status = _automated_gate_stage(
            ledger=ledger,
            stage="automated-early-gate",
            training=training,
            postcheck=postcheck,
            formal=False,
            plan=False,
        )
        results["automated-early-gate"] = gate
    else:
        gate = {**gate, "reused": True}
        results["automated-early-gate"] = gate
        status = "passed" if gate.get("computed_pass") is True else "blocked"
    if stop_after == "automated-early-gate":
        _finish_summary(ledger, results, status="stopped" if status == "passed" else "blocked", reason="requested stop-after automated-early-gate")
        return results, status
    if status != "passed":
        _finish_summary(ledger, results, status="blocked", reason=gate.get("reason", "automated early gate blocked"))
        return results, status
    return results, status


def _prepare_coverage_training_plan(attempt: Path, profile: Mapping[str, Any], route: Path) -> Mapping[str, Any]:
    """Derive one fresh coverage plan/model only after GPU permission is given."""

    parent_plan = profile.get("parent_plan_path")
    static_contract = profile.get("static_contract_path")
    coverage_plan = profile.get("coverage_plan_path")
    if not all(isinstance(value, str) for value in (parent_plan, static_contract, coverage_plan)):
        raise PipelineBlocked("coverage training input lacks parent/static/CPU coverage plan paths")
    from .coverage_executor import derive_coverage_training_plan

    output_plan = attempt / "coverage-training-plan-v1.json"
    model_path = attempt / "model"
    derived = derive_coverage_training_plan(
        parent_plan_path=parent_plan,
        static_contract_path=static_contract,
        coverage_plan_path=coverage_plan,
        output_plan_path=output_plan,
        model_path=model_path,
        route_root=route,
        containment_root=attempt.parents[2],
    )
    return {
        "profile": "coverage-smoke-v1",
        "plan_path": str(output_plan),
        "model_path": str(model_path),
        "coverage_training_plan": derived,
    }


def _prepare_convergence_training_plan(attempt: Path, profile: Mapping[str, Any], route: Path) -> Mapping[str, Any]:
    """Derive one fresh convergence model/plan; checkpoints are never resumed."""

    parent_plan = profile.get("parent_plan_path")
    static_contract = profile.get("static_contract_path")
    convergence_plan = profile.get("convergence_plan_path")
    if not all(isinstance(value, str) for value in (parent_plan, static_contract, convergence_plan)):
        raise PipelineBlocked("convergence training input lacks parent/static/CPU diagnostic plan paths")
    from .convergence_executor import derive_convergence_training_plan

    output_plan = attempt / "convergence-training-plan-v1.json"
    model_path = attempt / "model"
    derived = derive_convergence_training_plan(
        parent_plan_path=parent_plan,
        static_contract_path=static_contract,
        output_plan_path=output_plan,
        convergence_plan_path=convergence_plan,
        model_path=model_path,
        route_root=route,
        containment_root=attempt.parents[2],
        pose_reference_path=profile.get("pose_reference_path") if isinstance(profile.get("pose_reference_path"), str) else None,
    )
    return {
        "profile": "convergence1000-v1",
        "plan_path": str(output_plan),
        "model_path": str(model_path),
        "convergence_training_plan": derived,
    }


def _render_quality(result: Mapping[str, Any]) -> dict[str, Any]:
    """Apply conservative CPU render health checks without claiming visual acceptance."""

    structural = result.get("structural")
    if not isinstance(structural, Mapping) or structural.get("structural_pass") is not True:
        return {"quality_status": "blocked", "computed_pass": False, "reason": "render structural contract failed"}
    render_root_value = structural.get("render_root")
    if not isinstance(render_root_value, str):
        return {"quality_status": "blocked", "computed_pass": False, "reason": "render root is missing"}
    render_root = Path(render_root_value).resolve() / "renders"
    renders = sorted(path for path in render_root.glob("*.png") if path.is_file() and not path.is_symlink())
    if not renders:
        return {"quality_status": "blocked", "computed_pass": False, "reason": "no fixed render PNGs"}
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        return {"quality_status": "needs_review", "computed_pass": False, "reason": f"CPU visual health dependency unavailable: {exc}"}
    images = []
    for path in renders:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0 or not np.isfinite(image).all():
            return {"quality_status": "blocked", "computed_pass": False, "reason": f"render decode/non-finite failure: {path}"}
        images.append(image)
    means = [float(image.mean()) for image in images]
    stds = [float(image.std()) for image in images]
    if max(means) < 1.0e-4:
        return {"quality_status": "blocked", "computed_pass": False, "reason": "all fixed renders are black"}
    fingerprints = {sha256_file(path) for path in renders}
    if len(fingerprints) == 1:
        return {"quality_status": "blocked", "computed_pass": False, "reason": "all fixed renders are identical"}
    if max(stds) < 1.0e-5:
        return {"quality_status": "blocked", "computed_pass": False, "reason": "all fixed renders are uniform"}
    giant_like = sum(std < 1.0 for std in stds) == len(stds)
    smoke_profile = result.get("workload_profile") in {"smoke100-v1", "smoke100"}
    coverage_profile = result.get("workload_profile") == "coverage-smoke-v1"
    base_reason = "finite, non-black, non-identical fixed views with verified endpoint/on-path structure" if not giant_like else "uniform-looking fixed views require visual review"
    if smoke_profile or coverage_profile:
        return {
            "quality_status": "needs_review",
            "computed_pass": False,
            "reason": "smoke100/coverage render quality is separate from the explicit visual decision gate",
            "smoke100_structural_pass": smoke_profile,
            "coverage_smoke_structural_pass": coverage_profile,
            "coverage_smoke_visual_decision": "not_run" if coverage_profile else None,
            "formal_auto_release": False,
            "rough_visual_health": "needs_review" if giant_like else "unclassified",
            "render_count": len(renders),
            "mean_pixel_range": [min(means), max(means)],
            "std_pixel_range": [min(stds), max(stds)],
            "structural_health_reason": base_reason,
            "visual_acceptance_claimed": False,
        }
    return {
        "quality_status": "pass" if not giant_like else "needs_review",
        "computed_pass": not giant_like,
        "reason": base_reason,
        "render_count": len(renders),
        "mean_pixel_range": [min(means), max(means)],
        "std_pixel_range": [min(stds), max(stds)],
        "visual_acceptance_claimed": False,
    }


def _stage_path(stage_result: Mapping[str, Any], *keys: str) -> Path | None:
    """Resolve a declared path from an adapted stage or its raw payload."""

    payloads: list[Mapping[str, Any]] = [stage_result]
    raw = stage_result.get("raw_result")
    if isinstance(raw, Mapping):
        payloads.append(raw)
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                candidate = Path(value).resolve()
                if candidate.is_file() and not candidate.is_symlink():
                    return candidate
    return None


def _coverage_render_postcheck_stage(
    *,
    ledger: RunLedger,
    route: Path,
    coverage_render: Mapping[str, Any],
    coverage_training: Mapping[str, Any],
    training_input: Mapping[str, Any],
    frames_result: Mapping[str, Any] | None,
    colmap_result: Mapping[str, Any] | None,
    plan: bool,
) -> tuple[dict[str, Any], str]:
    """Run only the CPU evidence builder after a passed coverage render.

    The render attempt remains immutable and reusable.  A failed postcheck is
    recorded in its own append-only stage so resume can create another CPU
    attempt without invoking the GPU render again.
    """

    if plan:
        result = _stage_result(
            stage="coverage-smoke-render-postcheck",
            status="planned",
            computed_pass=False,
            reason="resident CPU streaming coverage render postcheck planned",
            plan=True,
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
        )
        return result, "planned"

    render_executor_root = coverage_render.get("executor_root")
    training_executor_root = coverage_training.get("executor_root")
    source_path = coverage_render.get("training_input") or coverage_training.get("training_input") or training_input.get("source_path")
    frame_selection = _stage_path(frames_result or {}, "selection_path", "frame_selection_path")
    segment_provenance = _stage_path(colmap_result or {}, "raw_result_path", "segment_provenance_path")
    if not isinstance(render_executor_root, str) or not isinstance(training_executor_root, str) or not isinstance(source_path, str):
        result = _stage_result(
            stage="coverage-smoke-render-postcheck",
            status="blocked",
            computed_pass=False,
            reason="coverage postcheck lacks render/training/input roots",
            plan=False,
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
        )
        _record_stage(ledger, "coverage-smoke-render-postcheck", result, status="blocked")
        return result, "blocked"
    if frame_selection is None or segment_provenance is None:
        result = _stage_result(
            stage="coverage-smoke-render-postcheck",
            status="blocked",
            computed_pass=False,
            reason="coverage postcheck requires immutable frame-selection and COLMAP segment provenance",
            plan=False,
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
        )
        _record_stage(ledger, "coverage-smoke-render-postcheck", result, status="blocked")
        return result, "blocked"

    render_attempt = Path(render_executor_root).resolve().parent
    training_attempt = Path(training_executor_root).resolve().parent
    training_input_path = Path(source_path).resolve()
    attempt = ledger.begin_attempt(
        "coverage-smoke-render-postcheck",
        {
            "profile": "coverage-smoke-v1",
            "render_attempt": str(render_attempt),
            "training_attempt": str(training_attempt),
            "training_input": str(training_input_path),
            "frame_selection": _artifact(frame_selection),
            "segment_provenance": _artifact(segment_provenance),
            "gpu_invoked": False,
            "render_reused": True,
            "cuda_rerun": False,
        },
    )
    evidence_dir = attempt / "evidence"
    try:
        from .render_postcheck import run_postcheck

        evidence = run_postcheck(
            run_root=ledger.run_dir,
            render_attempt=render_attempt,
            training_attempt=training_attempt,
            training_input=training_input_path,
            frame_selection=frame_selection,
            segment_provenance=segment_provenance,
            output_dir=evidence_dir,
        )
        postcheck_path = Path(str(evidence["postcheck"])).resolve()
        quality_path = render_attempt / "quality-v1.json"
        quality_evidence_path = attempt / "quality-evidence-v1.json"
        quality_evidence = {
            "schema_version": "coverage-render-quality-evidence-v1",
            "profile": "coverage-smoke-v1",
            "computed_pass": True,
            "visual_decision": "not_run",
            "formal_auto_release": False,
            "gpu_invoked": False,
            "render_reused": True,
            "cuda_rerun": False,
            "quality_v1": _artifact(quality_path),
            "postcheck": _artifact(postcheck_path),
            "contact_sheets": {
                "fixed_vs_gt": _artifact(evidence["fixed_sheet"]),
                "on_path": _artifact(evidence["on_path_sheet"]),
            },
            "metrics": _artifact(evidence["metrics_path"]),
            "png_hashes": _artifact(evidence["png_hashes"]),
        }
        _write_json_once(quality_evidence_path, quality_evidence)
        evidence_paths = [
            evidence["postcheck"],
            evidence["metrics_path"],
            evidence["png_hashes"],
            evidence["fixed_sheet"],
            evidence["on_path_sheet"],
            quality_evidence_path,
        ]
        stage = _stage_result(
            stage="coverage-smoke-render-postcheck",
            status="passed" if evidence.get("computed_pass") is True else "blocked",
            computed_pass=evidence.get("computed_pass") is True,
            reason="CPU render postcheck produced dynamic contact sheets, metrics, and PNG hashes" if evidence.get("computed_pass") is True else "CPU render postcheck did not pass",
            plan=False,
            artifacts=_artifacts(evidence_paths),
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            postcheck_result_path=str(postcheck_path),
            postcheck_result=evidence,
            quality_evidence_path=str(quality_evidence_path),
            quality_evidence=quality_evidence,
            contact_sheet_layout=evidence.get("contact_sheet_layout"),
            visual_decision="not_run",
            formal_auto_release=False,
        )
        status = "passed" if stage["computed_pass"] else "blocked"
        ledger.finish_attempt(stage="coverage-smoke-render-postcheck", attempt=attempt, status=status, result=stage)
        return stage, status
    except Exception as exc:
        stage = _stage_result(
            stage="coverage-smoke-render-postcheck",
            status="blocked",
            computed_pass=False,
            reason=str(exc),
            plan=False,
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            preserved_render_attempt=str(render_attempt),
        )
        ledger.finish_attempt(stage="coverage-smoke-render-postcheck", attempt=attempt, status="blocked", result=stage)
        return stage, "blocked"


def _convergence_render_postcheck_stage(
    *,
    ledger: RunLedger,
    route: Path,
    convergence_render: Mapping[str, Any],
    convergence_training: Mapping[str, Any],
    training_input: Mapping[str, Any],
    frames_result: Mapping[str, Any] | None,
    colmap_result: Mapping[str, Any] | None,
    baseline_render_attempt: Path | None,
    comparison_render_attempt: Path | None,
    plan: bool,
) -> tuple[dict[str, Any], str]:
    """Build convergence evidence from one completed render, CPU-only."""

    if plan:
        result = _stage_result(
            stage="convergence-smoke-render-postcheck",
            status="planned",
            computed_pass=False,
            reason="resident CPU streaming convergence render postcheck planned",
            plan=True,
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            diagnostic_only=True,
            formal_auto_release=False,
        )
        return result, "planned"
    render_executor_root = convergence_render.get("executor_root")
    training_executor_root = convergence_training.get("executor_root")
    source_path = convergence_render.get("training_input") or convergence_training.get("training_input") or training_input.get("source_path")
    frame_selection = _stage_path(frames_result or {}, "selection_path", "frame_selection_path")
    segment_provenance = _stage_path(colmap_result or {}, "raw_result_path", "segment_provenance_path")
    if not isinstance(render_executor_root, str) or not isinstance(training_executor_root, str) or not isinstance(source_path, str):
        result = _stage_result(stage="convergence-smoke-render-postcheck", status="blocked", computed_pass=False, reason="convergence postcheck lacks render/training/input roots", plan=False, gpu_invoked=False, render_reused=True, cuda_rerun=False, diagnostic_only=True, formal_auto_release=False)
        _record_stage(ledger, "convergence-smoke-render-postcheck", result, status="blocked")
        return result, "blocked"
    if frame_selection is None or segment_provenance is None:
        result = _stage_result(stage="convergence-smoke-render-postcheck", status="blocked", computed_pass=False, reason="convergence postcheck requires immutable frame-selection and COLMAP segment provenance", plan=False, gpu_invoked=False, render_reused=True, cuda_rerun=False, diagnostic_only=True, formal_auto_release=False)
        _record_stage(ledger, "convergence-smoke-render-postcheck", result, status="blocked")
        return result, "blocked"

    render_attempt = Path(render_executor_root).resolve().parent
    training_attempt = Path(training_executor_root).resolve().parent
    training_input_path = Path(source_path).resolve()
    attempt = ledger.begin_attempt(
        "convergence-smoke-render-postcheck",
        {
            "profile": "convergence1000-v1",
            "render_attempt": str(render_attempt),
            "training_attempt": str(training_attempt),
            "training_input": str(training_input_path),
            "baseline_render_attempt": None if baseline_render_attempt is None else str(baseline_render_attempt.resolve()),
            "comparison_render_attempt": None if comparison_render_attempt is None else str(comparison_render_attempt.resolve()),
            "frame_selection": _artifact(frame_selection),
            "segment_provenance": _artifact(segment_provenance),
            "gpu_invoked": False,
            "render_reused": True,
            "cuda_rerun": False,
            "diagnostic_only": True,
        },
    )
    evidence_dir = attempt / "evidence"
    try:
        from .render_postcheck import run_postcheck

        evidence = run_postcheck(
            run_root=ledger.run_dir,
            render_attempt=render_attempt,
            training_attempt=training_attempt,
            training_input=training_input_path,
            frame_selection=frame_selection,
            segment_provenance=segment_provenance,
            output_dir=evidence_dir,
            baseline_render_attempt=None if baseline_render_attempt is None else baseline_render_attempt.resolve(),
            comparison_render_attempt=None if comparison_render_attempt is None else comparison_render_attempt.resolve(),
        )
        postcheck_path = Path(str(evidence["postcheck"])).resolve()
        quality_path = render_attempt / "quality-v1.json"
        quality_evidence_path = attempt / "quality-evidence-v1.json"
        comparison_paths: dict[str, Any] = {}
        for key in ("baseline_comparison", "three_way_comparison"):
            value = evidence.get(key)
            if isinstance(value, Mapping) and isinstance(value.get("contact_sheet"), Mapping):
                comparison_paths[key] = value["contact_sheet"]
        quality_evidence = {
            "schema_version": "convergence-render-quality-evidence-v1",
            "profile": "convergence1000-v1",
            "computed_pass": True,
            "visual_decision": "not_run",
            "diagnostic_only": True,
            "formal_auto_release": False,
            "gpu_invoked": False,
            "render_reused": True,
            "cuda_rerun": False,
            "resident_full_resolution_frame_max": evidence.get("resident_full_resolution_frame_max"),
            "image_residency": evidence.get("image_residency"),
            "quality_v1": _artifact(quality_path),
            "postcheck": _artifact(postcheck_path),
            "contact_sheets": {
                "fixed_vs_gt": _artifact(evidence["fixed_sheet"]),
                "on_path": _artifact(evidence["on_path_sheet"]),
                **comparison_paths,
            },
            "metrics": _artifact(evidence["metrics_path"]),
            "png_hashes": _artifact(evidence["png_hashes"]),
        }
        _write_json_once(quality_evidence_path, quality_evidence)
        evidence_paths: list[str | Path] = [
            evidence["postcheck"],
            evidence["metrics_path"],
            evidence["png_hashes"],
            evidence["fixed_sheet"],
            evidence["on_path_sheet"],
            quality_evidence_path,
        ]
        for value in comparison_paths.values():
            if isinstance(value, Mapping) and isinstance(value.get("path"), str):
                evidence_paths.append(value["path"])
        stage = _stage_result(
            stage="convergence-smoke-render-postcheck",
            status="passed" if evidence.get("computed_pass") is True else "blocked",
            computed_pass=evidence.get("computed_pass") is True,
            reason="CPU convergence postcheck produced aspect-preserving sheets, comparisons, metrics, and PNG hashes" if evidence.get("computed_pass") is True else "CPU convergence postcheck did not pass",
            plan=False,
            artifacts=_artifacts(evidence_paths),
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            diagnostic_only=True,
            postcheck_result_path=str(postcheck_path),
            postcheck_result=evidence,
            quality_evidence_path=str(quality_evidence_path),
            quality_evidence=quality_evidence,
            contact_sheet_layout=evidence.get("contact_sheet_layout"),
            convergence_comparison={
                "baseline": evidence.get("baseline_comparison"),
                "coverage": evidence.get("three_way_comparison"),
            },
            convergence_visual_decision="not_run",
            formal_release_eligible=False,
            formal_auto_release=False,
        )
        status = "passed" if stage["computed_pass"] else "blocked"
        ledger.finish_attempt(stage="convergence-smoke-render-postcheck", attempt=attempt, status=status, result=stage)
        return stage, status
    except Exception as exc:
        stage = _stage_result(
            stage="convergence-smoke-render-postcheck",
            status="blocked",
            computed_pass=False,
            reason=str(exc),
            plan=False,
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            diagnostic_only=True,
            preserved_render_attempt=str(render_attempt),
            formal_auto_release=False,
        )
        ledger.finish_attempt(stage="convergence-smoke-render-postcheck", attempt=attempt, status="blocked", result=stage)
        return stage, "blocked"


def _formal_native_render_postcheck_stage(
    *,
    ledger: RunLedger,
    route: Path,
    native_render: Mapping[str, Any],
    formal_training: Mapping[str, Any],
    training_input: Mapping[str, Any],
    frames_result: Mapping[str, Any] | None,
    colmap_result: Mapping[str, Any] | None,
    plan: bool,
) -> tuple[dict[str, Any], str]:
    """Build one CPU-only formal native evidence record.

    This deliberately reuses the same streaming postcheck implementation as
    the diagnostic path, but records a distinct stage/profile and never
    imports a human visual token.
    """

    stage_name = "formal-native-render-postcheck"
    if plan:
        return _stage_result(
            stage=stage_name,
            status="planned",
            computed_pass=False,
            reason="resident CPU streaming formal native postcheck planned",
            plan=True,
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            formal_auto_release=False,
        ), "planned"
    render_executor_root = native_render.get("executor_root")
    training_executor_root = formal_training.get("executor_root")
    source_path = native_render.get("training_input") or formal_training.get("training_input") or training_input.get("source_path")
    frame_selection = _stage_path(frames_result or {}, "selection_path", "frame_selection_path")
    segment_provenance = _stage_path(colmap_result or {}, "raw_result_path", "segment_provenance_path")
    if not isinstance(render_executor_root, str) or not isinstance(training_executor_root, str) or not isinstance(source_path, str):
        result = _stage_result(
            stage=stage_name,
            status="blocked",
            computed_pass=False,
            reason="formal native postcheck lacks render/training/input roots",
            plan=False,
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            formal_auto_release=False,
        )
        _record_stage(ledger, stage_name, result, status="blocked")
        return result, "blocked"
    if frame_selection is None or segment_provenance is None:
        result = _stage_result(
            stage=stage_name,
            status="blocked",
            computed_pass=False,
            reason="formal native postcheck requires immutable frame-selection and COLMAP segment provenance",
            plan=False,
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            formal_auto_release=False,
        )
        _record_stage(ledger, stage_name, result, status="blocked")
        return result, "blocked"

    render_attempt = Path(render_executor_root).resolve().parent
    training_attempt = Path(training_executor_root).resolve().parent
    training_input_path = Path(source_path).resolve()
    attempt = ledger.begin_attempt(
        stage_name,
        {
            "profile": _FORMAL_PROFILE,
            "render_attempt": str(render_attempt),
            "training_attempt": str(training_attempt),
            "training_input": str(training_input_path),
            "frame_selection": _artifact(frame_selection),
            "segment_provenance": _artifact(segment_provenance),
            "gpu_invoked": False,
            "render_reused": True,
            "cuda_rerun": False,
        },
    )
    evidence_dir = attempt / "evidence"
    try:
        from .render_postcheck import run_postcheck

        evidence = run_postcheck(
            run_root=ledger.run_dir,
            render_attempt=render_attempt,
            training_attempt=training_attempt,
            training_input=training_input_path,
            frame_selection=frame_selection,
            segment_provenance=segment_provenance,
            output_dir=evidence_dir,
        )
        postcheck_path = Path(str(evidence["postcheck"])).resolve()
        quality_path = render_attempt / "quality-v1.json"
        quality_evidence_path = attempt / "quality-evidence-v1.json"
        quality_evidence = {
            "schema_version": "formal-native-render-quality-evidence-v1",
            "profile": _FORMAL_PROFILE,
            "computed_pass": evidence.get("computed_pass") is True,
            "visual_decision": "not_run",
            "manual_visual_review": False,
            "formal_auto_release": False,
            "gpu_invoked": False,
            "render_reused": True,
            "cuda_rerun": False,
            "resident_full_resolution_frame_max": evidence.get("resident_full_resolution_frame_max"),
            "image_residency": evidence.get("image_residency"),
            "quality_v1": _artifact(quality_path),
            "postcheck": _artifact(postcheck_path),
            "contact_sheets": {
                "fixed_vs_gt": _artifact(evidence["fixed_sheet"]),
                "on_path": _artifact(evidence["on_path_sheet"]),
            },
            "metrics": _artifact(evidence["metrics_path"]),
            "png_hashes": _artifact(evidence["png_hashes"]),
        }
        _write_json_once(quality_evidence_path, quality_evidence)
        evidence_paths = [
            evidence["postcheck"],
            evidence["metrics_path"],
            evidence["png_hashes"],
            evidence["fixed_sheet"],
            evidence["on_path_sheet"],
            quality_evidence_path,
        ]
        stage = _stage_result(
            stage=stage_name,
            status="passed" if evidence.get("computed_pass") is True else "blocked",
            computed_pass=evidence.get("computed_pass") is True,
            reason="CPU formal native postcheck produced metrics, hashes, and aspect-preserving sheets" if evidence.get("computed_pass") is True else "CPU formal native postcheck did not pass",
            plan=False,
            artifacts=_artifacts(evidence_paths),
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            postcheck_result_path=str(postcheck_path),
            postcheck_result=evidence,
            quality_evidence_path=str(quality_evidence_path),
            quality_evidence=quality_evidence,
            contact_sheet_layout=evidence.get("contact_sheet_layout"),
            formal_auto_release=False,
            held_out=False,
            training_views_only=True,
        )
        status = "passed" if stage["computed_pass"] else "blocked"
        ledger.finish_attempt(stage=stage_name, attempt=attempt, status=status, result=stage)
        return stage, status
    except Exception as exc:
        stage = _stage_result(
            stage=stage_name,
            status="blocked",
            computed_pass=False,
            reason=str(exc),
            plan=False,
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            preserved_render_attempt=str(render_attempt),
            formal_auto_release=False,
        )
        ledger.finish_attempt(stage=stage_name, attempt=attempt, status="blocked", result=stage)
        return stage, "blocked"


def _automated_gate_stage(
    *,
    ledger: RunLedger,
    stage: str,
    training: Mapping[str, Any],
    postcheck: Mapping[str, Any],
    formal: bool,
    plan: bool,
) -> tuple[dict[str, Any], str]:
    """Record a versioned automated gate without manufacturing human review."""

    if plan:
        return _stage_result(
            stage=stage,
            status="planned",
            computed_pass=False,
            reason=f"{AUTOMATED_POLICY_ID} planned",
            plan=True,
            automated_technical_gate=True,
            manual_visual_review=False,
            formal_release_eligible=False,
            formal_auto_release=False,
        ), "planned"
    attempt = ledger.begin_attempt(stage, {"policy_id": AUTOMATED_POLICY_ID, "formal": formal, "postcheck": postcheck.get("postcheck_result_path")})
    try:
        normalized_postcheck = normalize_postcheck_evidence(postcheck, run_root=ledger.run_dir)
        decision = evaluate_formal_gate(training=training, postcheck=normalized_postcheck) if formal else evaluate_early_gate(training=training, postcheck=normalized_postcheck)
        gate_kind = "formal" if formal else "early"
        decision_path = attempt / ("automated-formal-gate-v1.json" if formal else "automated-early-gate-v1.json")
        # Persist one flat, versioned artifact.  The judgment fields from the
        # decision (which already carries the flat v2 ``schema_version``) are
        # written at the top level with a nested ``policy`` descriptor, and the
        # entire shape is validated through the shared ``normalize_gate_evidence``
        # before it is written so every consumer reads the same canonical shape.
        flat_doc = {**decision, "policy": policy_descriptor()}
        _write_json_once(decision_path, normalize_gate_evidence(flat_doc))
        passed = decision.get("computed_pass") is True
        result = _stage_result(
            stage=stage,
            status="passed" if passed else "blocked",
            computed_pass=passed,
            reason="automated technical policy passed" if passed else "; ".join(str(item) for item in decision.get("reasons", [])),
            plan=False,
            artifacts=_artifacts([decision_path, postcheck.get("postcheck_result_path", "")]),
            automated_technical_gate=True,
            automated_policy_id=AUTOMATED_POLICY_ID,
            automated_policy=policy_descriptor(),
            automated_gate_decision=decision,
            automated_gate_result_path=str(decision_path),
            postcheck_result_path=normalized_postcheck.get("postcheck_result_path", postcheck.get("postcheck_result_path")),
            postcheck_result=normalized_postcheck,
            manual_visual_review=False,
            formal_release_eligible=passed,
            formal_auto_release=passed,
            held_out=False,
            training_views_only=True,
            delivery_quality=False,
        )
        status = "passed" if passed else "blocked"
        ledger.finish_attempt(stage=stage, attempt=attempt, status=status, result=result)
        return result, status
    except Exception as exc:
        result = _stage_result(
            stage=stage,
            status="blocked",
            computed_pass=False,
            reason=str(exc),
            plan=False,
            automated_technical_gate=True,
            automated_policy_id=AUTOMATED_POLICY_ID,
            manual_visual_review=False,
            formal_release_eligible=False,
            formal_auto_release=False,
        )
        ledger.finish_attempt(stage=stage, attempt=attempt, status="blocked", result=result)
        return result, "blocked"


def _training_policy(profile: str, plan: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    structural = result.get("structural", {})
    camera = structural.get("camera", {}) if isinstance(structural, Mapping) else {}
    count = structural.get("active_camera_count") if isinstance(structural, Mapping) else None
    order = None
    train_sha = structural.get("cameras_all_train_sha256") if isinstance(structural, Mapping) else None
    if isinstance(structural, Mapping):
        model_path = structural.get("model_path")
        if isinstance(model_path, str):
            try:
                value = json.loads((Path(model_path) / "cameras_all_train.json").read_text(encoding="utf-8"))
                if isinstance(value, list):
                    order = [str(item.get("image_name")) for item in value if isinstance(item, Mapping)]
            except (OSError, json.JSONDecodeError):
                order = None
    width = camera.get("width") if isinstance(camera, Mapping) else None
    height = camera.get("height") if isinstance(camera, Mapping) else None
    telemetry = structural.get("camera_sampling_telemetry") if isinstance(structural, Mapping) else None
    image_residency = structural.get("image_residency") if isinstance(structural, Mapping) else None
    profile_semantics = {
        "smoke100-v1": "local structural/numeric smoke only; no automatic rough-visual formal release",
        "coverage-smoke-v1": "local dynamic coverage diagnostic; visual decision is separate and does not auto-release formal",
        "convergence1000-v1": "local one-shot intermediate visual diagnostic; never formal or acceptance",
        "formal30000-v1": "local versioned policy, not the official final training standard",
    }.get(profile, "compatibility alias for a versioned local profile")
    return {
        "schema_version": "training-policy-record-v1",
        "profile_id": profile,
        "profile_semantics": profile_semantics,
        "policy_scope": "local_versioned_policy",
        "official_schedule_audit": {
            "official_baseline_iterations": 30000,
            "official_default_post_iter": 20000,
            "official_native_save_context": "commonly around 50000",
            "official_frame_or_gpu_scaling_formula": "not specified",
        },
        "registered_camera_count": count,
        "registered_camera_order": order,
        "resolution": {"width": width, "height": height},
        "pixel_count": None if not isinstance(width, int) or not isinstance(height, int) else width * height,
        "initial_points": None,
        "camera_sampling_policy": "external fixed-pose phase-local random pop without replacement with stack refill; safe_state seed 0",
        "camera_sampling_telemetry": telemetry,
        "image_residency": image_residency,
        "checkpoint_metrics": {},
        "checkpoint_iteration": plan.get("render_iteration"),
        "stop_reason": "completed_fixed_profile" if result.get("exit_code") == 0 else "child_process_failure",
        "gpu_identity": None,
        "gpu_vram_bytes": None,
        "elapsed_time_seconds": None,
        "peak_memory_bytes": None,
        "cameras_all_train_sha256": train_sha,
        "depth_source": "disabled",
        "training_state_resume": "artifact inspection only; checkpoint is not bit-exact training resume",
        "external_route_semantics": "fork-specific external COLMAP fixed-pose RGB-only, depth disabled, lazy MASt3R, dynamic camera identity",
        "densification_semantics": "anchor_growing via adjust_anchor/anchor_growing; no classic densify_and_clone/split/reset_opacity claim",
    }


def _prepare_automated_formal_plan(
    attempt: Path,
    profile: Mapping[str, Any],
    *,
    ledger: RunLedger,
    route: Path,
    automated_gate: Mapping[str, Any],
    allow_existing_model: bool = False,
) -> dict[str, Any]:
    """Bind formal30000 to the passed automated early gate for this attempt."""

    gate_value = automated_gate.get("automated_gate_result_path")
    if not isinstance(gate_value, str):
        raise PipelineBlocked("formal automated plan requires the automated early gate artifact")
    gate_path = _strict_path(gate_value, label="formal automated gate", kind="file", root=ledger.run_dir)
    parent_plan_value = profile.get("plan_path")
    static_value = profile.get("static_contract_path")
    if not isinstance(parent_plan_value, str) or not isinstance(static_value, str):
        raise PipelineBlocked("formal automated plan lacks parent plan/static contract")
    parent_plan = _strict_path(parent_plan_value, label="formal parent plan", kind="file", root=ledger.run_dir)
    static_path = _strict_path(static_value, label="formal static contract", kind="file", root=ledger.run_dir)
    parent = _load_json(parent_plan, "formal parent plan")
    model_value = parent.get("model_path")
    if not isinstance(model_value, str):
        raise PipelineBlocked("formal parent plan lacks model_path")
    model_raw = Path(model_value)
    if not model_raw.is_absolute():
        raise PipelineBlocked("formal model path must be absolute")
    model_path = model_raw.absolute().resolve(strict=False)
    _strict_path(model_path.parent, label="formal model parent", kind="directory", root=ledger.run_dir)
    if model_path.is_symlink() or (model_path.exists() and not model_path.is_dir()):
        raise PipelineBlocked(f"formal model path is not a safe directory: {model_path}")
    if allow_existing_model and not model_path.is_dir():
        raise PipelineBlocked(f"preserved formal model path is missing: {model_path}")
    if not allow_existing_model and model_path.exists():
        raise PipelineBlocked(f"fresh formal model path is not absent: {model_path}")
    policy_path = attempt / "formal-policy-v1.json"
    derived_plan_path = attempt / "formal-plan-v1.json"
    from .formal_executor import derive_automated_formal_training_plan

    derived_plan = derive_automated_formal_training_plan(
        parent_plan_path=parent_plan,
        static_contract_path=static_path,
        automated_gate_path=gate_path,
        automated_gate_sha256=sha256_file(gate_path),
        formal_policy_path=policy_path,
        output_plan_path=derived_plan_path,
        model_path=model_path,
        route_root=route,
        containment_root=ledger.run_dir,
        allow_existing_model=allow_existing_model,
    )
    return {
        "plan_path": str(derived_plan_path),
        "formal_plan_path": str(derived_plan_path),
        "formal_policy_path": str(policy_path),
        "automated_gate_path": str(gate_path),
        "derived_plan_sha256": derived_plan.get("derived_plan_sha256", sha256_file(derived_plan_path)),
    }


def _execute_training_or_render(
    *,
    ledger: RunLedger,
    stage: str,
    profile_input: Mapping[str, Any],
    plan: bool,
    execute_gpu: bool,
    route: Path,
    training_evidence_root: Path | None = None,
    isolated_render_snapshot: bool = False,
    extra: Mapping[str, Any] | None = None,
    retry_failed_attempt: bool = False,
    prepare: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    if not execute_gpu:
        status = "planned" if plan else "requires_escalated_gpu_execution"
        result = _stage_result(
            stage=stage,
            status=status,
            computed_pass=False,
            reason="requires_escalated_gpu_execution",
            plan=plan,
            gpu_invoked=False,
            profile_id=profile_input["profile"],
            plan_path=profile_input["plan_path"],
            static_contract_path=profile_input["static_contract_path"],
            training_input=profile_input["source_path"],
        )
        return result, "planned" if plan else "blocked"
    attempt = ledger.begin_attempt(
        stage,
        {
            "profile": profile_input["profile"],
            "plan_path": profile_input["plan_path"],
            "static_contract_path": profile_input["static_contract_path"],
            "training_evidence_root": None if training_evidence_root is None else str(training_evidence_root),
            "isolated_render_snapshot": isolated_render_snapshot,
        },
    )
    executor_root = attempt / "executor"
    effective_profile = dict(profile_input)
    retry_plan_path: Path | None = None
    retry_model_path: Path | None = None
    try:
        from .smoke_executor import SmokeExecutorBlocked, derive_training_retry_plan, execute_stage

        if prepare is not None:
            prepared = prepare(attempt, effective_profile)
            if not isinstance(prepared, Mapping):
                raise PipelineBlocked("stage preparation callback did not return a mapping")
            effective_profile.update(dict(prepared))

        if retry_failed_attempt:
            if stage != "smoke100-training":
                raise PipelineBlocked("executor-only retry is supported only for smoke100-training")
            retry_plan_path = attempt / "retry-plan-v1.json"
            retry_model_path = attempt / "model"
            derive_training_retry_plan(
                parent_plan_path=profile_input["plan_path"],
                output_plan_path=retry_plan_path,
                model_path=retry_model_path,
                route_root=route,
                containment_root=ledger.run_dir,
                reason="dynamic camera identity adapter production fix after preserved attempt-0001",
            )
            effective_profile["plan_path"] = str(retry_plan_path)

        ledger.mark_gpu_invoked()
        child_stage = "training" if stage.endswith("training") else "render"
        child = execute_stage(
            plan_path=effective_profile["plan_path"],
            static_contract_path=effective_profile["static_contract_path"],
            evidence_root=executor_root,
            stage=child_stage,
            route_root=route,
            containment_root=ledger.run_dir,
            training_evidence_root=training_evidence_root,
            isolated_render_snapshot=isolated_render_snapshot,
        )
    except Exception as exc:
        from .smoke_executor import SmokeExecutorBlocked

        if not isinstance(exc, (PipelineBlocked, SmokeExecutorBlocked, OSError, ValueError)):
            raise
        result = _stage_result(
            stage=stage,
            status="blocked",
            computed_pass=False,
            reason=str(exc),
            plan=plan,
            gpu_invoked=True,
            executor_root=str(executor_root),
            profile_id=profile_input["profile"],
            plan_path=effective_profile["plan_path"],
            static_contract_path=effective_profile["static_contract_path"],
            training_input=effective_profile["source_path"],
            retry_plan_path=None if retry_plan_path is None else str(retry_plan_path),
            retry_model_path=None if retry_model_path is None else str(retry_model_path),
        )
        ledger.finish_attempt(stage=stage, attempt=attempt, status="blocked", result=result)
        return result, "blocked"
    child_result_path = executor_root / "result.json"
    child_pass = child.get("exit_code") == 0 and child.get("structural_pass") is True
    smoke100_structural_pass = bool(child_pass) if profile_input["profile"] in {_SMOKE_PROFILE, "smoke100"} and child_stage == "render" else None
    quality: dict[str, Any] | None = None
    if child_stage == "render" and child_pass:
        quality = _render_quality(child)
        _write_json_once(attempt / "quality-v1.json", quality)
        # Presentation quality and human/rough visual review are warnings, but
        # the render's hard technical health checks remain a gate.  This keeps
        # formal/native renders eligible when they are merely blurry or
        # uniform-looking, while still stopping black, empty, corrupt, or
        # all-identical output.
        child_pass = child_pass and quality.get("quality_status") != "blocked"
    child_status = "passed" if child_pass else ("failed" if child.get("exit_code") not in (0, None) else "blocked")
    records = _artifacts([child_result_path, executor_root / "request.json", executor_root / "argv.json"])
    structural = child.get("structural")
    if isinstance(structural, Mapping):
        for key in ("model_path", "checkpoint", "external_pose_contract_sha256", "nvs_pose_json"):
            value = structural.get(key)
            if isinstance(value, Mapping) and isinstance(value.get("path"), str):
                records.extend(_artifacts([value["path"]]))
            elif isinstance(value, str) and Path(value).is_file():
                records.extend(_artifacts([value]))
        model_path = structural.get("model_path")
        if isinstance(model_path, str) and child_stage == "training":
            iteration = int(child.get("iterations", profile_input.get("iterations", 0)) or 0)
            records.extend(_artifacts([Path(model_path) / f"point_cloud/iteration_{iteration}/point_cloud.ply", Path(model_path) / "cameras_all_train.json", Path(model_path) / "external_colmap_pose_contract.json"]))
        if isinstance(structural.get("render_root"), str) and child_stage == "render":
            render_root = Path(str(structural["render_root"]))
            records.extend(_artifacts([render_root / "videos" / "nvs_camera_poses.json"]))
    stage_payload = dict(extra or {})
    if "formal_auto_release" not in stage_payload:
        stage_payload["formal_auto_release"] = (
            False
            if profile_input["profile"] in {_SMOKE_PROFILE, "smoke100"}
            else (quality.get("formal_auto_release") if isinstance(quality, Mapping) else None)
        )
    result = _stage_result(
        stage=stage,
        status=child_status,
        computed_pass=child_pass,
        reason="GPU executor and stage gates passed" if child_pass else str(child.get("structural", {}).get("reason", "GPU stage did not pass")),
        plan=plan,
        artifacts=records,
        gpu_invoked=True,
        executor_root=str(executor_root),
        executor_result=child,
        profile_id=profile_input["profile"],
        plan_path=effective_profile["plan_path"],
        static_contract_path=effective_profile["static_contract_path"],
        training_input=effective_profile["source_path"],
        retry_plan_path=None if retry_plan_path is None else str(retry_plan_path),
        retry_model_path=None if retry_model_path is None else str(retry_model_path),
        training_policy=_training_policy(profile_input["profile"], child.get("request", {}), child) if child_stage == "training" else None,
        quality=quality,
        smoke100_structural_pass=smoke100_structural_pass,
        coverage_smoke_visual_decision=None if quality is None else quality.get("coverage_smoke_visual_decision", "not_run"),
        payload=stage_payload,
    )
    if result.get("stage_result_conflict"):
        child_status = "blocked"
    ledger.finish_attempt(stage=stage, attempt=attempt, status=child_status, result=result)
    return result, child_status


def _preserved_training_executor_candidate(ledger: RunLedger, stage: str) -> tuple[dict[str, Any], Path, dict[str, Any]] | None:
    """Find a blocked stage whose child actually completed and wrote a model.

    This is deliberately evidence-driven.  A blocked ledger attempt is never
    promoted by itself; the child result, request, and immutable model are
    revalidated by ``_recover_preserved_training_attempt`` before a new passed
    ledger attempt can be created.
    """

    attempts = ledger.summary.get("stages", {}).get(stage, [])
    if not isinstance(attempts, list) or not attempts:
        return None
    candidate: tuple[dict[str, Any], Path, dict[str, Any]] | None = None
    for entry in reversed(attempts):
        if not isinstance(entry, Mapping) or entry.get("status") != "blocked":
            return None
        result_path_value = entry.get("result_path")
        if not isinstance(result_path_value, str):
            return None
        try:
            result_path = _strict_path(
                ledger.run_dir / result_path_value,
                label=f"preserved {stage} stage result",
                kind="file",
                root=ledger.run_dir,
            )
            envelope = _load_json(result_path, f"preserved {stage} stage result")
            result = envelope.get("result")
            if not isinstance(result, Mapping) or not isinstance(result.get("executor_root"), str):
                continue
            executor_raw = Path(str(result["executor_root"]))
            if not executor_raw.is_absolute() or executor_raw.is_symlink():
                return None
            try:
                executor_raw.resolve(strict=False).relative_to(ledger.run_dir.resolve(strict=True))
            except (OSError, ValueError):
                return None
            # The orchestration stage allocates its attempt before running
            # preparation.  A preparation failure therefore leaves no child
            # directory at all; that is a CPU orphan, not a failed GPU run.
            if not executor_raw.exists():
                continue
            if not executor_raw.is_dir():
                return None
            executor_root = _strict_path(
                executor_raw,
                label=f"preserved {stage} executor root",
                kind="directory",
                root=ledger.run_dir,
            )
            executor_result_path = executor_root / "result.json"
            # Empty roots are CPU preparation orphans, not GPU attempts.
            if not executor_result_path.is_file() or executor_result_path.is_symlink():
                continue
            executor_result_path = _strict_path(
                executor_result_path,
                label=f"preserved {stage} executor result",
                kind="file",
                root=ledger.run_dir,
            )
            executor_result = _load_json(executor_result_path, f"preserved {stage} executor result")
            if executor_result.get("exit_code") != 0 or executor_result.get("gpu_invoked") is not True:
                return None
            model_value = executor_result.get("model_path")
            if not isinstance(model_value, str):
                return None
            _strict_path(model_value, label=f"preserved {stage} model", kind="directory", root=ledger.run_dir)
            if candidate is not None:
                return None
            candidate = (
                {
                    "attempt": entry.get("attempt"),
                    "status": entry.get("status"),
                    "result": dict(result),
                    "result_path": str(result_path),
                    "artifacts_valid": RunLedger._artifacts_valid(result, root=ledger.run_dir),
                },
                executor_root,
                executor_result,
            )
        except (PipelineBlocked, OSError, json.JSONDecodeError):
            return None
    return candidate


def _recover_preserved_training_attempt(
    *,
    ledger: RunLedger,
    stage: str,
    profile_input: Mapping[str, Any],
    route: Path,
    preserved: tuple[dict[str, Any], Path, dict[str, Any]],
    prepare: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """CPU-revalidate one completed GPU child without rerunning training."""

    # Bind the exception type before preparation.  Preparation is deliberately
    # inside the guarded region and may fail before the later executor import;
    # otherwise the except tuple itself raises UnboundLocalError and loses the
    # precise stage reason.
    from .smoke_executor import SmokeExecutorBlocked

    previous, preserved_root, preserved_child = preserved
    previous_result = previous.get("result") if isinstance(previous.get("result"), Mapping) else {}
    attempt = ledger.begin_attempt(
        stage,
        {
            "recovery_kind": "cpu-revalidated-preserved-gpu-training",
            "preserved_attempt": previous.get("attempt"),
            "preserved_executor_root": str(preserved_root),
        },
    )
    executor_root = attempt / "executor"
    effective_profile = dict(profile_input)
    try:
        prepared = prepare(attempt, effective_profile)
        if not isinstance(prepared, Mapping):
            raise PipelineBlocked("preserved training recovery preparation did not return a mapping")
        effective_profile.update(dict(prepared))
        from .smoke_executor import (
            SmokeExecutorBlocked,
            _verify_training_outputs,
            build_training_command,
            validate_plan_and_static,
        )

        contract = validate_plan_and_static(
            effective_profile["plan_path"],
            effective_profile["static_contract_path"],
            route_root=route,
            containment_root=ledger.run_dir,
            allow_existing_model=True,
            allow_producer_code_drift=True,
        )
        preserved_request_path = _strict_path(
            preserved_root / "request.json",
            label=f"preserved {stage} executor request",
            kind="file",
            root=ledger.run_dir,
        )
        preserved_argv_path = _strict_path(
            preserved_root / "argv.json",
            label=f"preserved {stage} executor argv",
            kind="file",
            root=ledger.run_dir,
        )
        preserved_request = _load_json(preserved_request_path, f"preserved {stage} executor request")
        preserved_argv = _load_json(preserved_argv_path, f"preserved {stage} executor argv")
        expected_argv = build_training_command(contract, route_root=route)
        for key in ("stage", "source_path", "static_contract_path", "static_contract_sha256", "workload_profile", "iterations", "render_iteration", "model_path"):
            expected = {
                "stage": "training",
                "source_path": contract["source_path"],
                "static_contract_path": contract["static_contract_path"],
                "static_contract_sha256": contract["static_contract_sha256"],
                "workload_profile": contract["workload_profile"],
                "iterations": contract["iterations"],
                "render_iteration": contract["render_iteration"],
                "model_path": contract["model_path"],
            }[key]
            if preserved_request.get(key) != expected:
                raise PipelineBlocked(f"preserved training request {key} differs from the derived formal contract")
        if preserved_request.get("shell") is not False or preserved_request.get("argv") != expected_argv or preserved_argv.get("argv") != expected_argv or preserved_argv.get("shell") is not False:
            raise PipelineBlocked("preserved training argv is not byte-identical to the derived formal contract")
        if preserved_child.get("model_path") != contract["model_path"]:
            raise PipelineBlocked("preserved training model_path differs from the derived formal contract")
        structural = _verify_training_outputs(contract, containment_root=ledger.run_dir)
        executor_root.mkdir(parents=True, exist_ok=False)
        request_path = executor_root / "request.json"
        argv_path = executor_root / "argv.json"
        result_path = executor_root / "result.json"
        request = {
            "schema": "longsplat-smoke-executor-request-v1",
            "stage": "training",
            "plan_path": contract["plan_path"],
            "plan_sha256": contract["plan_sha256"],
            "static_contract_path": contract["static_contract_path"],
            "static_contract_sha256": contract["static_contract_sha256"],
            "source_path": contract["source_path"],
            "model_path": contract["model_path"],
            "workload_profile": contract["workload_profile"],
            "iterations": contract["iterations"],
            "render_iteration": contract["render_iteration"],
            "route_root": str(route),
            "route_code_identity_sha256": contract["plan"].get("route_code_identity_sha256"),
            "producer_route_code_identity_sha256": contract["plan"].get("route_code_identity_sha256"),
            "consumer_executor_sha256": sha256_file(Path(__file__).resolve()),
            "argv": expected_argv,
            "shell_escaped_command_display": " ".join(expected_argv),
            "shell": False,
            "cwd": str(route),
            "training_evidence_root": None,
            "isolated_render_snapshot": False,
            "recovery_kind": "cpu-revalidated-preserved-gpu-training",
            "preserved_executor_root": str(preserved_root),
        }
        _write_json_once(request_path, request)
        _write_json_once(argv_path, {"argv": expected_argv, "shell_escaped_command_display": " ".join(expected_argv), "shell": False})
        recovered_child = {
            "schema": "longsplat-smoke-executor-result-v1",
            "stage": "training",
            "exit_code": 0,
            "model_path": contract["model_path"],
            "workload_profile": contract["workload_profile"],
            "iterations": contract["iterations"],
            "render_iteration": contract["render_iteration"],
            "request_path": str(request_path),
            "argv_path": str(argv_path),
            "gpu_invoked": False,
            "preserved_gpu_invoked": True,
            "recovery_kind": "cpu-revalidated-preserved-gpu-training",
            "preserved_executor_root": str(preserved_root),
            "preserved_executor_result_sha256": sha256_file(preserved_root / "result.json"),
            "static_input_reference": {
                "source_path": contract["source_path"],
                "plan_sha256": contract["plan_sha256"],
                "static_contract_sha256": contract["static_contract_sha256"],
            },
            "structural": structural,
            "structural_pass": True,
        }
        _write_json_once(result_path, recovered_child)
        records = _artifacts([request_path, argv_path, result_path, preserved_root / "request.json", preserved_root / "argv.json", preserved_root / "result.json"])
        model_path = structural.get("model_path") if isinstance(structural, Mapping) else None
        if isinstance(model_path, str):
            iteration = int(recovered_child["iterations"])
            records.extend(_artifacts([Path(model_path) / f"point_cloud/iteration_{iteration}/point_cloud.ply", Path(model_path) / "cameras_all_train.json", Path(model_path) / "external_colmap_pose_contract.json", Path(model_path) / "camera_sampling_telemetry-v1.json", Path(model_path) / "camera_sampling_telemetry-v1.jsonl"]))
        stage_payload = dict(extra or {})
        stage_payload.update(
            {
                "recovery_kind": "cpu-revalidated-preserved-gpu-training",
                "preserved_executor_root": str(preserved_root),
                "preserved_executor_result_sha256": sha256_file(preserved_root / "result.json"),
                "recovery_gpu_invoked": False,
            }
        )
        result = _stage_result(
            stage=stage,
            status="passed",
            computed_pass=True,
            reason="CPU revalidated preserved completed GPU training; no training rerun",
            plan=False,
            artifacts=records,
            gpu_invoked=True,
            recovery_gpu_invoked=False,
            executor_root=str(executor_root),
            executor_result=recovered_child,
            profile_id=profile_input["profile"],
            plan_path=effective_profile["plan_path"],
            static_contract_path=effective_profile["static_contract_path"],
            training_input=effective_profile["source_path"],
            training_policy=_training_policy(profile_input["profile"], request, recovered_child),
            payload=stage_payload,
        )
        ledger.finish_attempt(stage=stage, attempt=attempt, status="passed", result=result)
        return result, "passed"
    except (PipelineBlocked, OSError, ValueError, SmokeExecutorBlocked) as exc:
        result = _stage_result(
            stage=stage,
            status="blocked",
            computed_pass=False,
            reason=str(exc),
            plan=False,
            gpu_invoked=False,
            recovery_gpu_invoked=False,
            preserved_executor_root=str(preserved_root),
        )
        ledger.finish_attempt(stage=stage, attempt=attempt, status="blocked", result=result)
        return result, "blocked"


def _profile_input(payload: Mapping[str, Any], profile: str) -> dict[str, Any]:
    source = payload.get("source_path")
    plan_path = payload.get("future_smoke_plan_path")
    static_path = payload.get("static_contract_path")
    if not isinstance(source, str) or not isinstance(plan_path, str) or not isinstance(static_path, str):
        raise PipelineBlocked(f"{profile} LongSplat input lacks source/plan/static contract")
    return {
        "profile": profile,
        "source_path": source,
        "plan_path": plan_path,
        "static_contract_path": static_path,
        "payload": dict(payload),
    }


def _formal_input(
    *,
    source: Path,
    route: Path,
    ledger: RunLedger,
    camera_run_dir: Path,
    camera_model: str = "SIMPLE_RADIAL",
    matching: str = "sequential",
    tool_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Create the formal profile input only after the smoke gate passes."""

    output_root = ledger.run_dir / "raw-formal-input"
    formal_run_dir = output_root / "formal-input"
    source_sha256 = sha256_file(source)

    # A consumer-only resume must not ask the raw producer to recreate an
    # already-passed nested run.  The nested raw ledger quite correctly has
    # its own producer identity, so a later orchestration/schema change can
    # otherwise look like an unsafe raw-pipeline resume even though every
    # formal input artifact is unchanged.  Reuse is evidence-driven and
    # deliberately strict; an incomplete or mutated nested run is a block,
    # never an excuse to overwrite it.
    if formal_run_dir.exists():
        if formal_run_dir.is_symlink() or not formal_run_dir.is_dir():
            raise PipelineBlocked(f"existing formal profile input is not a safe directory: {formal_run_dir}")
        try:
            nested_identity = _load_json(formal_run_dir / "identity.json", "formal nested identity")
            if nested_identity.get("source_video_sha256") != source_sha256:
                raise PipelineBlocked("existing formal profile input source SHA differs")
            nested_source_path = nested_identity.get("source_video_path")
            if nested_source_path is not None and Path(str(nested_source_path)).resolve() != source.resolve():
                raise PipelineBlocked("existing formal profile input source path differs")
            nested_run_root = nested_identity.get("run_root_resolved")
            if nested_run_root is not None and Path(str(nested_run_root)).resolve() != formal_run_dir.resolve():
                raise PipelineBlocked("existing formal profile input run root differs")
            formal_run_dir.resolve(strict=True).relative_to(ledger.run_dir.resolve(strict=True))

            camera_loaded = _raw_stage_payload(camera_run_dir, "camera-staging")
            if camera_loaded is None or camera_loaded[0] != "passed":
                raise PipelineBlocked("formal input reuse requires the passed camera-staging producer")
            camera_payload = camera_loaded[1]
            camera_contract_value = camera_payload.get("contract_path")
            if not isinstance(camera_contract_value, str):
                raise PipelineBlocked("formal input reuse cannot locate the camera contract")
            camera_contract_path = _strict_path(
                camera_contract_value,
                label="formal reuse camera contract",
                kind="file",
                root=ledger.run_dir,
            )
            camera_identity = _load_json(camera_run_dir / "identity.json", "camera producer identity")

            loaded = _raw_stage_payload(formal_run_dir, "longsplat-input")
            if loaded is None or loaded[0] != "passed":
                raise PipelineBlocked("existing formal profile input is not passed")
            payload = loaded[1]
            if payload.get("status") != "computed_pass":
                raise PipelineBlocked("existing formal profile input is not computed_pass")
            if payload.get("future_smoke_profile") != _FORMAL_PROFILE or payload.get("future_smoke_iterations") != 30000:
                raise PipelineBlocked("existing formal profile input profile/iterations differ")
            parent_binding = payload.get("parent_binding")
            if not isinstance(parent_binding, Mapping):
                raise PipelineBlocked("existing formal profile input lacks parent binding")
            if parent_binding.get("source_video_sha256") != source_sha256:
                raise PipelineBlocked("existing formal parent binding source SHA differs")
            parent_root = parent_binding.get("parent_run_root")
            if not isinstance(parent_root, str) or Path(parent_root).resolve() != camera_run_dir.resolve():
                raise PipelineBlocked("existing formal parent binding camera root differs")
            parent_identity_sha = parent_binding.get("parent_run_identity_sha256")
            if parent_identity_sha != camera_identity.get("run_identity_sha256"):
                raise PipelineBlocked("existing formal parent binding camera identity differs")
            parent_contract = parent_binding.get("camera_contract_path")
            if not isinstance(parent_contract, str) or Path(parent_contract).resolve() != camera_contract_path.resolve():
                raise PipelineBlocked("existing formal parent binding camera contract differs")
            if parent_binding.get("camera_contract_file_sha256") != sha256_file(camera_contract_path):
                raise PipelineBlocked("existing formal parent binding camera contract SHA differs")
            if parent_binding.get("camera_staging_result_path") != str(camera_loaded[2].resolve()):
                raise PipelineBlocked("existing formal parent binding camera result differs")
            if parent_binding.get("camera_staging_result_file_sha256") != sha256_file(camera_loaded[2]):
                raise PipelineBlocked("existing formal parent binding camera result SHA differs")
            if not RunLedger._artifacts_valid(payload, root=ledger.run_dir):
                raise PipelineBlocked("existing formal profile input artifacts are missing, escaped, or mutated")
            result = _profile_input(payload, _FORMAL_PROFILE)
            result.update(
                {
                    "raw_run_dir": str(formal_run_dir.resolve()),
                    "raw_result_path": str(loaded[2].resolve()),
                    "artifacts": _payload_artifacts(payload, result_path=loaded[2]),
                    "reused": True,
                    "reuse_reason": "passed immutable formal input reused after consumer-only orchestration change",
                }
            )
            return result
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise PipelineBlocked(f"existing formal profile input cannot be safely reused: {exc}") from exc

    from .raw_pipeline import run_raw_video_pipeline

    summary = run_raw_video_pipeline(
        input_video=source,
        output_root=output_root,
        run_id="formal-input",
        parent_run_root=camera_run_dir,
        containment_root=ledger.run_dir,
        route_root=route,
        depth_source="disabled",
        camera_model=camera_model,
        matching=matching,
        tool_paths=tool_paths,
        stop_after="longsplat-input",
        future_smoke_profile=_FORMAL_PROFILE,
    )
    formal_run_dir = Path(str(summary.get("run_dir", output_root / "formal-input"))).resolve()
    loaded = _raw_stage_payload(formal_run_dir, "longsplat-input")
    if loaded is None or loaded[0] != "passed":
        reason = "formal profile input did not pass"
        if loaded is not None:
            reason = str(loaded[1].get("reason", reason))
        raise PipelineBlocked(reason)
    payload = loaded[1]
    result = _profile_input(payload, _FORMAL_PROFILE)
    result["raw_run_dir"] = str(formal_run_dir)
    result["raw_result_path"] = str(loaded[2])
    result["artifacts"] = _payload_artifacts(payload, result_path=loaded[2])
    return result


def _authority_stage(
    *,
    ledger: RunLedger,
    route: Path,
    source: Path,
    formal: Mapping[str, Any],
    native: Mapping[str, Any],
    supplied_manifest: str | Path | None,
    plan: bool,
    native_postcheck: Mapping[str, Any] | None = None,
    tool_provider: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    if plan:
        result = _stage_result(stage="authority-manifest", status="planned", computed_pass=False, reason="authority manifest planned", plan=True)
        return result, "planned"
    attempt = ledger.begin_attempt("authority-manifest", {"profile": _FORMAL_PROFILE, "source": str(source), "supplied_manifest": None if supplied_manifest is None else str(supplied_manifest)})
    try:
        from .authority_manifest import build_authority_manifest, load_authority_manifest, write_manifest_once

        if supplied_manifest is not None:
            manifest_path = Path(supplied_manifest).resolve()
            authority = load_authority_manifest(manifest_path, route_root=route, containment_root=ledger.run_dir)
        else:
            native_executor_root = Path(str(native["executor_root"]))
            native_child = native.get("executor_result", {})
            structural = native_child.get("structural", {}) if isinstance(native_child, Mapping) else {}
            render_root = Path(str(structural.get("render_root", ""))).resolve() / "renders"
            native_result_path = native_executor_root / "result.json"
            if isinstance(native_postcheck, Mapping) and isinstance(native_postcheck.get("postcheck_result_path"), str):
                postcheck_path = Path(str(native_postcheck["postcheck_result_path"])).resolve()
            else:
                postcheck_path = attempt / "native-postcheck-v1.json"
                postcheck = {
                    "schema_version": "native-render-postcheck-v1",
                    "computed_pass": True,
                    "quality": native.get("quality"),
                    "training_views_only": True,
                    "held_out": False,
                    "camera_count": (native_child.get("structural", {}) if isinstance(native_child, Mapping) else {}).get("fixed_render_count"),
                    "render_root": str(render_root),
                }
                _write_json_once(postcheck_path, postcheck)
            manifest = build_authority_manifest(
                source_video=source,
                plan=formal["plan_path"],
                static_contract=formal["static_contract_path"],
                training_input=formal["source_path"],
                training_model=Path(str(formal["payload"].get("source_path"))).parent / f"{Path(str(formal['payload'].get('source_path'))).name}-formal-gpu-30000-model",
                pose_contract=Path(str(formal["payload"].get("source_path"))).parent / f"{Path(str(formal['payload'].get('source_path'))).name}-formal-gpu-30000-model" / "external_colmap_pose_contract.json",
                native_result=native_result_path,
                native_postcheck=postcheck_path,
                native_render_root=render_root,
                conversion_profile_id="standard30000-v1",
                status={"training_views_only": True, "held_out": False, "accepted": False, "supersplat": False},
                tool_provider=tool_provider,
            )
            manifest_path = write_manifest_once(attempt / "authority_manifest-v1.json", manifest)
            authority = load_authority_manifest(manifest_path, route_root=route, containment_root=ledger.run_dir)
        records = _artifacts([manifest_path])
        result = _stage_result(
            stage="authority-manifest",
            status="passed",
            computed_pass=True,
            reason="authority manifest built and revalidated from this run's formal/native artifacts",
            plan=False,
            artifacts=records,
            authority_manifest_path=str(manifest_path),
            authority=authority,
            camera_count=authority["camera_count"],
            camera_order=authority["camera_order"],
            camera_dimensions=authority["camera_dimensions"],
            depth_source="disabled",
        )
    except Exception as exc:
        result = _stage_result(stage="authority-manifest", status="blocked", computed_pass=False, reason=str(exc), plan=False)
        ledger.finish_attempt(stage="authority-manifest", attempt=attempt, status="blocked", result=result)
        return result, "blocked"
    ledger.finish_attempt(stage="authority-manifest", attempt=attempt, status="passed", result=result)
    return result, "passed"


def _conversion_ply_from_result(conversion: Mapping[str, Any]) -> Path:
    child = conversion.get("executor_result", {})
    structural = child.get("structural", {}) if isinstance(child, Mapping) else {}
    value = structural.get("path") or structural.get("converted_ply_path")
    if isinstance(value, str):
        return Path(value).resolve()
    executor_root = conversion.get("executor_root")
    if isinstance(executor_root, str):
        return (Path(executor_root) / "conversion_model_snapshot" / "converted_3dgs" / "point_cloud.ply").resolve()
    raise PipelineBlocked("conversion result has no technical PLY path")


def _evaluation_reused_render_available(evaluation: Mapping[str, Any], camera_count: int) -> bool:
    """Return true only when a failed evaluator left a complete PNG set."""

    executor_root = evaluation.get("executor_root")
    if not isinstance(executor_root, str):
        return False
    root = Path(executor_root).resolve()
    result_path = root / "evaluation_result.json"
    eval_root = root / "same_camera_eval"
    if not result_path.is_file() or eval_root.is_symlink() or not eval_root.is_dir():
        return False
    try:
        result = _load_json(result_path, "converted evaluator result")
    except PipelineBlocked:
        return False
    if result.get("exit_code") == 0:
        return False
    gt = eval_root / "evaluator_result_gt"
    converted = eval_root / "evaluator_result_renders"
    if not gt.is_dir() or not converted.is_dir() or gt.is_symlink() or converted.is_symlink():
        return False
    gt_files = sorted(path for path in gt.iterdir() if path.is_file() and not path.is_symlink())
    converted_files = sorted(path for path in converted.iterdir() if path.is_file() and not path.is_symlink())
    return len(gt_files) == camera_count and len(converted_files) == camera_count and [path.name for path in gt_files] == [path.name for path in converted_files]


def _converted_eval_postprocess_stage(
    *,
    ledger: RunLedger,
    route: Path,
    authority: Mapping[str, Any],
    conversion: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    plan: bool,
) -> tuple[dict[str, Any], str]:
    if plan:
        return _stage_result(stage="converted-eval-postprocess", status="planned", computed_pass=False, reason="resident CPU streaming postprocess planned", plan=True, gpu_invoked=False, render_reused=True, cuda_rerun=False), "planned"
    executor_root = evaluation.get("executor_root")
    conversion_root = conversion.get("executor_root")
    authority_path = authority.get("authority_manifest_path")
    if not all(isinstance(value, str) for value in (executor_root, conversion_root, authority_path)):
        result = _stage_result(stage="converted-eval-postprocess", status="blocked", computed_pass=False, reason="postprocess lacks authority/evaluator/conversion roots", plan=False, gpu_invoked=False, render_reused=False, cuda_rerun=False)
        _record_stage(ledger, "converted-eval-postprocess", result, status="blocked")
        return result, "blocked"
    attempt = ledger.begin_attempt(
        "converted-eval-postprocess",
        {
            "authority_manifest_path": authority_path,
            "conversion_root": conversion_root,
            "failed_evaluation_root": executor_root,
            "gpu_invoked": False,
            "render_reused": True,
            "cuda_rerun": False,
        },
    )
    try:
        from .converted_eval_postprocess import run_postprocess

        conversion_result_path = Path(conversion_root) / "conversion_result.json"
        converted_ply = _conversion_ply_from_result(conversion)
        result = run_postprocess(
            authority_manifest_path=authority_path,
            conversion_result_path=conversion_result_path,
            converted_ply_path=converted_ply,
            failed_evaluation_root=executor_root,
            output_root=attempt,
            route_root=route,
            containment_root=ledger.run_dir,
        )
        stage = _stage_result(
            stage="converted-eval-postprocess",
            status="passed" if result.get("FULL_STREAM_VALIDATION_PASS") is True else "needs_review",
            # The CPU recovery gate is technical: complete ordered finite
            # triples and no severe degradation.  It must not reintroduce a
            # manual/rough-visual acceptance requirement into the automated
            # delivery path.
            computed_pass=result.get("FULL_STREAM_VALIDATION_PASS") is True,
            reason="CPU streaming postprocess passed on reused GPU PNGs" if result.get("FULL_STREAM_VALIDATION_PASS") is True else "CPU streaming postprocess needs review",
            plan=False,
            artifacts=_artifacts([attempt / "postprocess_result.json", attempt / "metrics.json", attempt / "png_hashes.json", attempt / "fixed_gt_native_converted_contact_sheet.png"]),
            gpu_invoked=False,
            render_reused=True,
            cuda_rerun=False,
            postprocess_result_path=str(attempt / "postprocess_result.json"),
            postprocess_result=result,
        )
        status = "passed" if stage["computed_pass"] else "blocked"
        ledger.finish_attempt(stage="converted-eval-postprocess", attempt=attempt, status=status, result=stage)
        return stage, status
    except Exception as exc:
        stage = _stage_result(stage="converted-eval-postprocess", status="blocked", computed_pass=False, reason=str(exc), plan=False, gpu_invoked=False, render_reused=True, cuda_rerun=False)
        ledger.finish_attempt(stage="converted-eval-postprocess", attempt=attempt, status="blocked", result=stage)
        return stage, "blocked"


def _automated_technical_delivery(
    *,
    ledger: RunLedger,
    route: Path,
    authority: Mapping[str, Any],
    conversion: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    postprocess: Mapping[str, Any],
    early_gate: Mapping[str, Any],
    formal_gate: Mapping[str, Any],
    native_postcheck: Mapping[str, Any],
    plan: bool,
    publish_dir: str | Path | None = None,
    delivery_name: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Seal a technical PLY without fabricating SuperSplat verification."""

    stage_name = "automated-technical-delivery"
    if plan:
        return _stage_result(
            stage=stage_name,
            status="planned",
            computed_pass=False,
            reason="automated technical delivery planned",
            plan=True,
            automated_technical_gate=True,
            accepted_by_automated_policy=False,
            manual_visual_review=False,
            supersplat_format_compatible=True,
            supersplat_runtime_verified=False,
        ), "planned"
    if early_gate.get("computed_pass") is not True or formal_gate.get("computed_pass") is not True:
        result = _stage_result(
            stage=stage_name,
            status="blocked",
            computed_pass=False,
            reason="automated early/formal technical gate did not pass",
            plan=False,
            automated_technical_gate=True,
            accepted_by_automated_policy=False,
            manual_visual_review=False,
        )
        _record_stage(ledger, stage_name, result, status="blocked")
        return result, "blocked"
    if postprocess.get("computed_pass") is not True:
        result = _stage_result(
            stage=stage_name,
            status="blocked",
            computed_pass=False,
            reason="converted-eval CPU technical postprocess did not pass",
            plan=False,
            automated_technical_gate=True,
            accepted_by_automated_policy=False,
            manual_visual_review=False,
        )
        _record_stage(ledger, stage_name, result, status="blocked")
        return result, "blocked"
    attempt = ledger.begin_attempt(stage_name, {"policy_id": AUTOMATED_POLICY_ID, "authority_manifest": authority.get("authority_manifest_path")})
    try:
        from .acceptance_delivery import AcceptanceDeliveryError, create_candidate_delivery

        authority_path = Path(str(authority["authority_manifest_path"])).resolve()
        conversion_root = Path(str(conversion["executor_root"])).resolve()
        evaluation_root = Path(str(evaluation["executor_root"])).resolve()
        candidate_ply = _conversion_ply_from_result(conversion)
        evaluation_result = Path(str(postprocess.get("postprocess_result_path", ""))).resolve() if isinstance(postprocess.get("postprocess_result_path"), str) else evaluation_root / "evaluation_result.json"
        if not evaluation_result.is_file():
            evaluation_result = evaluation_root / "evaluation_result.json"
        if not evaluation_result.is_file():
            raise PipelineBlocked("automated technical delivery lacks converted evaluation result")
        post_result = _load_json(evaluation_result, "automated delivery evaluation result")
        candidate_info = post_result.get("candidate_delivery") if isinstance(post_result, Mapping) else None
        comparison_sheet = None
        if isinstance(candidate_info, Mapping) and isinstance(candidate_info.get("comparison_sheet"), str):
            comparison_sheet = Path(str(candidate_info["comparison_sheet"])).resolve()
        if comparison_sheet is None:
            candidate = evaluation_root / "same_camera_eval" / "evaluator_contact_sheet.png"
            if candidate.is_file() and not candidate.is_symlink():
                comparison_sheet = candidate
        if comparison_sheet is None or not comparison_sheet.is_file() or comparison_sheet.is_symlink():
            raise PipelineBlocked("automated technical delivery lacks same-camera comparison contact sheet")
        evidence_files: dict[str, Path] = {
            "authority_manifest.json": authority_path,
            "conversion_result.json": conversion_root / "conversion_result.json",
            "evaluation_result.json": evaluation_result,
            "automated_early_gate.json": Path(str(early_gate["automated_gate_result_path"])),
            "automated_formal_gate.json": Path(str(formal_gate["automated_gate_result_path"])),
            "formal_native_postcheck.json": Path(str(native_postcheck["postcheck_result_path"])),
        }
        if isinstance(postprocess.get("postprocess_result_path"), str):
            evidence_files["converted_eval_postprocess.json"] = Path(str(postprocess["postprocess_result_path"]))
        for name, path in list(evidence_files.items()):
            if not path.is_file() or path.is_symlink():
                raise PipelineBlocked(f"automated delivery evidence is missing: {name}: {path}")
        output_dir = attempt / "automated_technical_delivery"
        package = create_candidate_delivery(
            converted_ply=candidate_ply,
            output_dir=output_dir,
            authority_manifest=authority_path,
            evidence_files=evidence_files,
            comparison_sheet=comparison_sheet,
            route_root=route,
            containment_root=ledger.run_dir,
            automated_policy={
                "policy_id": AUTOMATED_POLICY_ID,
                "early_gate": early_gate.get("automated_gate_result_path"),
                "formal_gate": formal_gate.get("automated_gate_result_path"),
                "manual_visual_review": False,
                "held_out": False,
            },
            provenance_context={
                "automated_technical_gate": True,
                "manual_visual_review": False,
                "supersplat_runtime_verified": False,
                "screenshot_evidence": "not_required_by_current_user",
                "known_limitations": [
                    "training-view-only evaluation; held_out=false",
                    "automated policy is local evidence-driven and is not human visual acceptance",
                    "SuperSplat format compatibility was checked from the standard 3DGS PLY schema; runtime loading was not performed",
                ],
            },
        )
        published_ply = None
        published_ply_receipt: Path | None = None
        if publish_dir is not None:
            published_ply = publish_ply(
                source_ply=package["point_cloud"]["path"],
                output_dir=publish_dir,
                name=delivery_name,
            )
            verify_public_delivery(
                published_ply,
                output_dir=publish_dir,
                name=delivery_name,
            )
            published_ply_receipt = ledger.run_dir / "published_ply.json"
            write_json_once_atomic(published_ply_receipt, published_ply)
        delivery_artifacts = _artifacts(
            [
                package["candidate_manifest"],
                package["provenance"],
                package["report"],
                package["point_cloud"]["path"],
                package["sha256sums"]["path"],
                package["comparison_sheet"]["path"],
            ]
            + ([] if published_ply_receipt is None else [published_ply_receipt])
        )
        result = _stage_result(
            stage=stage_name,
            status="passed",
            computed_pass=True,
            reason="standard 3DGS technical delivery sealed by automated-technical-v1",
            plan=False,
            artifacts=delivery_artifacts,
            automated_technical_gate=True,
            automated_policy_id=AUTOMATED_POLICY_ID,
            accepted_by_automated_policy=True,
            manual_visual_review=False,
            supersplat_format_compatible=True,
            supersplat_runtime_verified=False,
            screenshot_evidence="not_required_by_current_user",
            accepted=False,
            supersplat=False,
            technical_delivery_root=package["root"],
            technical_delivery_manifest=package["candidate_manifest"],
            technical_ply=package["point_cloud"],
            vertices=package["vertices"],
            sha256sums=package["sha256sums"],
            published_ply=published_ply,
            published_ply_receipt=None if published_ply_receipt is None else str(published_ply_receipt),
        )
        ledger.finish_attempt(stage=stage_name, attempt=attempt, status="passed", result=result)
        return result, "passed"
    except (PipelineBlocked, OSError, ValueError, AcceptanceDeliveryError, PublishError) as exc:
        result = _stage_result(
            stage=stage_name,
            status="blocked",
            computed_pass=False,
            reason=str(exc),
            plan=False,
            automated_technical_gate=True,
            accepted_by_automated_policy=False,
            manual_visual_review=False,
        )
        ledger.finish_attempt(stage=stage_name, attempt=attempt, status="blocked", result=result)
        return result, "blocked"


def _run_default_formal_delivery(
    *,
    ledger: RunLedger,
    route: Path,
    source: Path,
    camera_run_dir: Path | None,
    results: dict[str, Any],
    stop_after: str,
    execute_gpu: bool,
    authority_manifest: str | Path | None,
    camera_model: str,
    matching: str,
    tool_paths: Mapping[str, str | Path] | None,
    tool_provider: Mapping[str, Any] | None,
    publish_dir: str | Path | None = None,
    delivery_name: str | None = None,
) -> dict[str, Any]:
    """Continue a passed automated early gate through technical delivery."""

    formal: Mapping[str, Any] | None = _stage_reusable(ledger, "formal-training")
    formal_input: Mapping[str, Any]
    if formal is None:
        try:
            if camera_run_dir is None:
                raise PipelineBlocked("formal profile input requires the passed dynamic camera-staging run")
            created = _formal_input(
                source=source,
                route=route,
                ledger=ledger,
                camera_run_dir=camera_run_dir,
                camera_model=camera_model,
                matching=matching,
                tool_paths=tool_paths,
            )
            formal_input = created
            formal_profile_input = _profile_input(created["payload"], _FORMAL_PROFILE)
            formal_profile_input.update({"source_path": created["source_path"], "raw_run_dir": created["raw_run_dir"]})
        except Exception as exc:
            result = _stage_result(stage="formal-training", status="blocked", computed_pass=False, reason=str(exc), plan=False, gpu_invoked=False, formal_auto_release=False)
            _record_stage(ledger, "formal-training", result, status="blocked")
            results["formal-training"] = result
            _finish_summary(ledger, results, status="blocked", reason=str(exc))
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect formal CPU input packaging")
        automated_gate = results.get("automated-early-gate")
        if not isinstance(automated_gate, Mapping):
            raise PipelineBlocked("formal training requires the passed automated early gate result")
        prepare_formal = lambda attempt, profile: _prepare_automated_formal_plan(
            attempt,
            profile,
            ledger=ledger,
            route=route,
            automated_gate=automated_gate,
            allow_existing_model=False,
        )
        preserved = _preserved_training_executor_candidate(ledger, "formal-training")
        if preserved is not None:
            formal, status = _recover_preserved_training_attempt(
                ledger=ledger,
                stage="formal-training",
                profile_input=formal_profile_input,
                route=route,
                preserved=preserved,
                prepare=lambda attempt, profile: _prepare_automated_formal_plan(
                    attempt,
                    profile,
                    ledger=ledger,
                    route=route,
                    automated_gate=automated_gate,
                    allow_existing_model=True,
                ),
                extra={"formal_input": dict(formal_input), "automated_technical_gate": True, "formal_auto_release": True},
            )
        else:
            formal, status = _execute_training_or_render(
                ledger=ledger,
                stage="formal-training",
                profile_input=formal_profile_input,
                plan=False,
                execute_gpu=execute_gpu,
                route=route,
                prepare=prepare_formal,
                extra={"formal_input": dict(formal_input), "automated_technical_gate": True, "formal_auto_release": True},
            )
        results["formal-training"] = formal
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=formal["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved formal training attempt")
    else:
        formal = {**formal, "reused": True}
        results["formal-training"] = dict(formal)
        formal_input = formal.get("formal_input") if isinstance(formal.get("formal_input"), Mapping) else {}
        formal_profile_input = _profile_input(formal_input.get("payload", formal_input), _FORMAL_PROFILE)
        if isinstance(formal.get("plan_path"), str):
            formal_profile_input["plan_path"] = formal["plan_path"]
    if isinstance(formal, Mapping) and isinstance(formal.get("plan_path"), str):
        formal_profile_input["plan_path"] = formal["plan_path"]
    if stop_after == "formal-training":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after formal-training")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at native-render")

    if not isinstance(formal, Mapping) or not isinstance(formal_profile_input, Mapping):
        _finish_summary(ledger, results, status="blocked", reason="formal training result lacks its immutable profile input")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect formal training binding")
    native = _stage_reusable(ledger, "native-render")
    if native is None:
        native, status = _execute_training_or_render(
            ledger=ledger,
            stage="native-render",
            profile_input=formal_profile_input,
            plan=False,
            execute_gpu=execute_gpu,
            route=route,
            training_evidence_root=Path(str(formal.get("executor_root", ""))),
            isolated_render_snapshot=True,
            extra={"automated_technical_gate": True, "formal_auto_release": False},
        )
        results["native-render"] = native
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=native["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved native render attempt")
    else:
        native = {**native, "reused": True}
        results["native-render"] = native
    if stop_after == "native-render":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after native-render")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at formal-native-render-postcheck")

    if native.get("quality", {}).get("quality_status") == "blocked":
        _finish_summary(ledger, results, status="blocked", reason="native render hard technical health did not pass")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect native render quality evidence")
    native_postcheck = _stage_reusable(ledger, "formal-native-render-postcheck")
    if native_postcheck is None:
        native_postcheck, status = _formal_native_render_postcheck_stage(
            ledger=ledger,
            route=route,
            native_render=native,
            formal_training=formal,
            training_input=formal_profile_input,
            frames_result=results.get("frames"),
            colmap_result=results.get("colmap"),
            plan=False,
        )
        results["formal-native-render-postcheck"] = native_postcheck
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=native_postcheck["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect native CPU postcheck")
    else:
        native_postcheck = {**native_postcheck, "reused": True}
        results["formal-native-render-postcheck"] = native_postcheck
    if stop_after == "formal-native-render-postcheck":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after formal-native-render-postcheck")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at automated-formal-gate")

    formal_gate = _stage_reusable(ledger, "automated-formal-gate")
    if formal_gate is None:
        formal_gate, status = _automated_gate_stage(
            ledger=ledger,
            stage="automated-formal-gate",
            training=formal,
            postcheck=native_postcheck,
            formal=True,
            plan=False,
        )
        results["automated-formal-gate"] = formal_gate
    else:
        formal_gate = {**formal_gate, "reused": True}
        results["automated-formal-gate"] = formal_gate
        status = "passed" if formal_gate.get("computed_pass") is True else "blocked"
    if status != "passed":
        _finish_summary(ledger, results, status="blocked", reason=formal_gate.get("reason", "automated formal gate blocked"))
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect formal automated gate; conversion is closed")
    if stop_after == "automated-formal-gate":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after automated-formal-gate")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at authority-manifest")

    authority_result = _stage_reusable(ledger, "authority-manifest")
    if authority_result is None:
        authority_result, status = _authority_stage(
            ledger=ledger,
            route=route,
            source=source,
            formal=formal_profile_input,
            native=native,
            supplied_manifest=authority_manifest,
            plan=False,
            native_postcheck=native_postcheck,
            tool_provider=tool_provider,
        )
        results["authority-manifest"] = authority_result
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=authority_result["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect authority manifest")
    else:
        authority_result = {**authority_result, "reused": True}
        results["authority-manifest"] = authority_result
    if stop_after == "authority-manifest":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after authority-manifest")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at conversion")

    conversion = _stage_reusable(ledger, "conversion")
    if conversion is None:
        conversion, status = _execute_authority_stage(ledger=ledger, stage="conversion", authority_result=authority_result, plan=False, execute_gpu=execute_gpu, route=route)
        results["conversion"] = conversion
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=conversion["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved conversion attempt")
    else:
        conversion = {**conversion, "reused": True}
        results["conversion"] = conversion
    if stop_after == "conversion":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after conversion")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at converted-eval")

    evaluation = _stage_reusable(ledger, "converted-eval")
    if evaluation is None:
        evaluation, status = _execute_authority_stage(
            ledger=ledger,
            stage="converted-eval",
            authority_result=authority_result,
            plan=False,
            execute_gpu=execute_gpu,
            route=route,
            native_evidence_root=Path(str(native["executor_root"])),
            conversion_evidence_root=Path(str(conversion["executor_root"])),
        )
        results["converted-eval"] = evaluation
        if status != "passed":
            if not _evaluation_reused_render_available(evaluation, int(authority_result.get("camera_count", 0))):
                _finish_summary(ledger, results, status="blocked", reason=evaluation["reason"])
                return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect converted-eval failure")
            evaluation = {**evaluation, "render_artifacts_available": True, "render_reused": True, "cuda_rerun": False, "evaluation_gate": "deferred_to_cpu_streaming_postprocess"}
            results["converted-eval"] = evaluation
    else:
        evaluation = {**evaluation, "reused": True}
        results["converted-eval"] = evaluation
    if stop_after == "converted-eval":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after converted-eval")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at converted-eval-postprocess")

    postprocess = _stage_reusable(ledger, "converted-eval-postprocess")
    if postprocess is None:
        if evaluation.get("render_artifacts_available") is True:
            postprocess, status = _converted_eval_postprocess_stage(
                ledger=ledger,
                route=route,
                authority=authority_result,
                conversion=conversion,
                evaluation=evaluation,
                plan=False,
            )
        else:
            postprocess = _stage_result(
                stage="converted-eval-postprocess",
                status="passed",
                computed_pass=True,
                reason="same-camera evaluator completed technical metrics; CPU rerun was not required",
                plan=False,
                gpu_invoked=False,
                render_reused=False,
                cuda_rerun=False,
                postprocess_required=False,
                full_stream_validation_pass=True,
            )
            _record_stage(ledger, "converted-eval-postprocess", postprocess, status="passed")
            status = "passed"
        results["converted-eval-postprocess"] = postprocess
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=postprocess["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect converted-eval CPU postprocess")
    else:
        postprocess = {**postprocess, "reused": True}
        results["converted-eval-postprocess"] = postprocess
    if stop_after == "converted-eval-postprocess":
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after converted-eval-postprocess")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at automated technical delivery")

    delivery = _stage_reusable(ledger, "automated-technical-delivery")
    if delivery is None:
        delivery, status = _automated_technical_delivery(
            ledger=ledger,
            route=route,
            authority=authority_result,
            conversion=conversion,
            evaluation=evaluation,
            postprocess=postprocess,
            early_gate=results["automated-early-gate"],
            formal_gate=formal_gate,
            native_postcheck=native_postcheck,
            plan=False,
            publish_dir=publish_dir,
            delivery_name=delivery_name,
        )
        results["automated-technical-delivery"] = delivery
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=delivery["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect automated technical delivery")
    else:
        delivery = {**delivery, "reused": True}
        results["automated-technical-delivery"] = delivery
    _finish_summary(ledger, results, status="stopped", reason="AUTOMATED_TECHNICAL_DELIVERY complete; SuperSplat runtime was not verified")
    return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="completed technical delivery")


def _candidate_delivery(
    *,
    ledger: RunLedger,
    route: Path,
    authority: Mapping[str, Any],
    conversion: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    postprocess: Mapping[str, Any] | None = None,
    conversion_visual_token: str | Path | None = None,
    plan: bool,
) -> tuple[dict[str, Any], str]:
    if plan:
        return _stage_result(stage="candidate-delivery", status="planned", computed_pass=False, reason="candidate packaging planned", plan=True), "planned"
    if postprocess is not None and postprocess.get("postprocess_result_path"):
        if conversion_visual_token is None:
            result = _stage_result(
                stage="candidate-delivery",
                status="blocked",
                computed_pass=False,
                reason="technical candidate requires an explicit converted-view visual decision token; this is not SuperSplat acceptance",
                plan=False,
                accepted=False,
                supersplat=False,
            )
            _record_stage(ledger, "candidate-delivery", result, status="blocked")
            return result, "blocked"
        try:
            decision = _visual_decision_valid(conversion_visual_token, label="converted-view visual decision token")
            if decision["decision"] != "pass":
                raise PipelineBlocked("converted-view visual decision did not pass")
            attempt = ledger.begin_attempt(
                "candidate-delivery",
                {
                    "authority_manifest": authority["authority_manifest_path"],
                    "postprocess_result": postprocess["postprocess_result_path"],
                    "visual_decision_token": decision,
                },
            )
            from .converted_eval_postprocess import create_candidate_from_postprocess

            candidate_root = attempt / "candidate_delivery"
            package = create_candidate_from_postprocess(
                authority_manifest_path=authority["authority_manifest_path"],
                postprocess_result_path=postprocess["postprocess_result_path"],
                candidate_output=candidate_root,
                route_root=route,
                containment_root=ledger.run_dir,
                manual_visual_decision="pass",
                manual_visual_note="Explicit converted-view technical visual decision; not SuperSplat acceptance.",
            )
            result = _stage_result(
                stage="candidate-delivery",
                status="passed",
                computed_pass=True,
                reason="technical candidate package created after complete CPU postprocess and explicit converted-view review",
                plan=False,
                artifacts=_artifacts([Path(str(package["candidate_manifest"])), Path(str(package["point_cloud"]["path"])), Path(str(package["sha256sums"]["path"]))]),
                candidate_root=package["root"],
                candidate_ply=package["point_cloud"]["path"],
                candidate_sha256=package["point_cloud"]["sha256"],
                candidate_size_bytes=package["point_cloud"]["size_bytes"],
                vertices=package["vertices"],
                accepted=False,
                supersplat=False,
                conversion_visual_decision=decision,
                manual_supersplat_required=True,
            )
            ledger.finish_attempt(stage="candidate-delivery", attempt=attempt, status="passed", result=result)
            return result, "passed"
        except Exception as exc:
            if "attempt" in locals() and isinstance(locals().get("attempt"), Path):
                result = _stage_result(stage="candidate-delivery", status="blocked", computed_pass=False, reason=str(exc), plan=False, accepted=False, supersplat=False)
                ledger.finish_attempt(stage="candidate-delivery", attempt=attempt, status="blocked", result=result)
                return result, "blocked"
            result = _stage_result(stage="candidate-delivery", status="blocked", computed_pass=False, reason=str(exc), plan=False, accepted=False, supersplat=False)
            _record_stage(ledger, "candidate-delivery", result, status="blocked")
            return result, "blocked"
    if conversion.get("executor_result", {}).get("STRUCTURAL_CONVERSION_PASS") is not True and conversion.get("executor_result", {}).get("structural_pass") is not True:
        result = _stage_result(stage="candidate-delivery", status="blocked", computed_pass=False, reason="conversion did not pass standard PLY validation", plan=False)
        _record_stage(ledger, "candidate-delivery", result, status="blocked")
        return result, "blocked"
    eval_result = evaluation.get("executor_result", {})
    if eval_result.get("SAME_CAMERA_VISUAL_PASS") == "fail":
        result = _stage_result(stage="candidate-delivery", status="blocked", computed_pass=False, reason="same-camera evaluation failed; candidate delivery is unreachable", plan=False)
        _record_stage(ledger, "candidate-delivery", result, status="blocked")
        return result, "blocked"
    attempt = ledger.begin_attempt("candidate-delivery", {"authority_manifest": authority["authority_manifest_path"], "conversion_stage": conversion["stage"], "evaluation_stage": evaluation["stage"]})
    root = attempt / "candidate_delivery"
    root.mkdir(parents=True, exist_ok=False)
    conversion_result = conversion.get("executor_result", {})
    structural = conversion_result.get("structural", {}) if isinstance(conversion_result, Mapping) else {}
    source_value = structural.get("converted_ply_path") if isinstance(structural, Mapping) else None
    if not isinstance(source_value, str):
        source_value = str(Path(str(conversion["executor_root"])) / "conversion_model_snapshot" / "converted_3dgs" / "point_cloud.ply")
    source = Path(source_value).resolve()
    if source.is_symlink() or not source.is_file():
        result = _stage_result(stage="candidate-delivery", status="blocked", computed_pass=False, reason=f"converted candidate PLY is missing: {source}", plan=False)
        ledger.finish_attempt(stage="candidate-delivery", attempt=attempt, status="blocked", result=result)
        return result, "blocked"
    destination = root / "converted_3dgs.ply"
    shutil.copyfile(source, destination)
    identity = _artifact(destination)
    if identity is None:
        raise PipelineBlocked(f"candidate copy failed: {destination}")
    candidate_manifest = {
        "schema_version": "longsplat-candidate-delivery-v1",
        "candidate": identity,
        "source_candidate": {"path": str(source), "sha256": sha256_file(source), "size_bytes": source.stat().st_size},
        "vertices": _ply_vertex_count(destination),
        "authority_manifest": authority["authority_manifest_path"],
        "conversion_profile_id": authority.get("authority", {}).get("profile_id", "standard30000-v1"),
        "same_camera_visual_pass": eval_result.get("SAME_CAMERA_VISUAL_PASS"),
        "accepted": False,
        "supersplat": False,
        "held_out": False,
        "training_views_only": True,
        "manual_supersplat_required": True,
    }
    manifest_path = root / "candidate_manifest.json"
    _write_json_once(manifest_path, candidate_manifest)
    for source_file, name in ((conversion.get("executor_root"), "conversion_evidence.json"), (evaluation.get("executor_root"), "evaluation_evidence.json"), (authority.get("authority_manifest_path"), "authority_manifest.json")):
        if not isinstance(source_file, str):
            continue
        source_path = Path(source_file)
        if source_path.is_dir():
            source_path = source_path / ("conversion_result.json" if "conversion" in name else "evaluation_result.json")
        if source_path.is_file() and not source_path.is_symlink():
            shutil.copyfile(source_path, root / name)
    result = _stage_result(
        stage="candidate-delivery",
        status="passed",
        computed_pass=True,
        reason="candidate PLY copied and cryptographically bound; manual SuperSplat review remains required",
        plan=False,
        artifacts=_artifacts([destination, manifest_path]),
        candidate_root=str(root),
        candidate_ply=str(destination),
        candidate_sha256=identity["sha256"],
        candidate_size_bytes=identity["size_bytes"],
        vertices=candidate_manifest["vertices"],
        authority_manifest_path=authority["authority_manifest_path"],
        same_camera_visual_pass=eval_result.get("SAME_CAMERA_VISUAL_PASS"),
        accepted=False,
        supersplat=False,
    )
    ledger.finish_attempt(stage="candidate-delivery", attempt=attempt, status="passed", result=result)
    return result, "passed"


def _ply_vertex_count(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            header: list[str] = []
            while True:
                line = handle.readline()
                if not line:
                    break
                text = line.decode("ascii", errors="strict").rstrip("\n")
                header.append(text)
                if text == "end_header":
                    break
    except (OSError, UnicodeDecodeError) as exc:
        raise PipelineBlocked(f"candidate PLY header cannot be read: {path}: {exc}") from exc
    for line in header:
        fields = line.split()
        if fields[:2] == ["element", "vertex"] and len(fields) == 3:
            try:
                count = int(fields[2])
            except ValueError as exc:
                raise PipelineBlocked(f"candidate PLY vertex count is invalid: {path}") from exc
            if count <= 0:
                raise PipelineBlocked(f"candidate PLY vertex count is non-positive: {path}")
            return count
    raise PipelineBlocked(f"candidate PLY vertex element is missing: {path}")


def _diagnostic_config_equivalent(existing: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Compare producer config while allowing only the diagnostic consumer slice."""

    ignored = {
        "diagnostic_profile",
        "convergence_visual_gate",
        # These are persisted identity metadata and are validated against the
        # actual resumed run directory separately; they are not producer
        # algorithm configuration.
        "output_root_input",
        "output_root_resolved",
        "run_root_resolved",
    }
    old = {key: value for key, value in existing.items() if key not in ignored}
    new = {key: value for key, value in current.items() if key not in ignored}
    old_order = old.get("stage_order")
    new_order = new.get("stage_order")
    if isinstance(old_order, list) and isinstance(new_order, list):
        convergence_stages = {
            "convergence-smoke-plan",
            "convergence-smoke-training",
            "convergence-smoke-render",
            "convergence-smoke-render-postcheck",
            "convergence-smoke-visual-gate",
        }
        normalized_new = [value for value in new_order if value not in convergence_stages]
        if normalized_new == old_order:
            new["stage_order"] = old_order
    return old == new



def _resume_identity_contract_allowed(
    *,
    existing_identity: Mapping[str, Any],
    current_identity: Mapping[str, Any],
    existing_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
) -> bool:
    """Shared resume identity/config contract.

    Exact config equality plus the 7 immutable run-identity keys and an equal
    producer-scoped code identity.  Used by both the technical consumer and the
    mid consumer resume gates so the guard logic is not duplicated.
    """
    if existing_config != current_config:
        return False
    immutable_keys = (
        "source_video_sha256",
        "source_video_path",
        "source_video_size_bytes",
        "tool_identity_sha256",
        "canonical_config_sha256",
        "output_root_resolved",
        "run_root_resolved",
    )
    for key in immutable_keys:
        old = existing_identity.get(key)
        new = current_identity.get(key)
        if key == "source_video_path" and isinstance(old, str) and isinstance(new, str):
            old, new = str(Path(old).resolve()), str(Path(new).resolve())
        if old != new:
            return False
    existing_code = existing_identity.get("code_identity")
    current_code = current_identity.get("code_identity")
    if not isinstance(existing_code, Mapping) or not isinstance(current_code, Mapping):
        return False
    return _scoped_code_identity(existing_code, "producer")["identity_sha256"] == _scoped_code_identity(current_code, "producer")["identity_sha256"]


def _producer_resume_allowed(
    *,
    existing_identity: Mapping[str, Any],
    current_identity: Mapping[str, Any],
    existing_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
) -> bool:
    """Allow unrelated documentation/test/legacy churn without reuse drift.

    The broad source-tree digest remains provenance in ``identity.json``.
    Reuse is governed by the exact producer configuration and the narrowed
    production code scope; provider, source, camera, and output bindings stay
    immutable.  Schema changes are intentionally not silently migrated here.
    """

    return _resume_identity_contract_allowed(
        existing_identity=existing_identity,
        current_identity=current_identity,
        existing_config=existing_config,
        current_config=current_config,
    )


def _stage_summary_entries(summary: Mapping[str, Any], stage: str) -> list[Any] | None:
    stages = summary.get("stages")
    if not isinstance(stages, Mapping):
        return None
    entries = stages.get(stage)
    if not isinstance(entries, list) or not entries:
        return None
    return entries


def _stage_latest_status(summary: Mapping[str, Any], stage: str) -> str | None:
    entries = _stage_summary_entries(summary, stage)
    if entries is None:
        return None
    latest = entries[-1]
    return latest.get("status") if isinstance(latest, Mapping) else None


def _stage_latest_passed_with_valid_artifacts(summary: Mapping[str, Any], run_dir: Path, stage: str) -> bool:
    """True when the latest ledger attempt for the stage is passed with valid artifacts."""
    entries = _stage_summary_entries(summary, stage)
    if entries is None:
        return False
    latest = entries[-1]
    if not isinstance(latest, Mapping) or latest.get("status") != "passed":
        return False
    result_path_value = latest.get("result_path")
    if not isinstance(result_path_value, str):
        return False
    result_path = run_dir / result_path_value
    if result_path.is_symlink() or not result_path.is_file():
        return False
    try:
        envelope = _load_json(result_path, "resume result " + stage)
    except (PipelineBlocked, OSError, json.JSONDecodeError):
        return False
    result = envelope.get("result")
    return isinstance(result, Mapping) and RunLedger._artifacts_valid(result, root=run_dir)


def _resume_orphan_allowed(
    summary: Mapping[str, Any],
    run_dir: Path,
    *,
    stage_scope: set[str],
    allow_empty_formal_orphan: bool,
    retry_stage: str | None = None,
) -> bool:
    """Permit an orphan only when it is the EMPTY (no result.json) attempt of
    the exact stage being retried (retry_stage), or, for the historical
    technical recovery, the empty formal-training attempt.  An orphan that
    carries a real result.json is genuine evidence and remains a hard stop, as
    does any other in-scope orphan - an orphan never self-promotes to a
    resumable attempt.  Orphans outside the scope are ignored.
    """
    orphan = summary.get("orphan_evidence")
    if not isinstance(orphan, Mapping) or orphan.get("stage") not in stage_scope:
        return True
    orphan_stage = orphan.get("stage")
    attempt_name = orphan.get("attempt")
    if not isinstance(attempt_name, str) or not re.fullmatch(r"attempt-\d{4}", attempt_name):
        return False
    candidate_stage: str | None = None
    if allow_empty_formal_orphan and orphan_stage == "formal-training":
        candidate_stage = "formal-training"
    elif retry_stage is not None and orphan_stage == retry_stage:
        candidate_stage = retry_stage
    if candidate_stage is None:
        return False
    orphan_root = run_dir / "stages" / candidate_stage / attempt_name / "executor"
    try:
        orphan_root.resolve(strict=False).relative_to(run_dir.resolve(strict=True))
    except (OSError, ValueError):
        return False
    if orphan_root.is_symlink():
        return False
    return not orphan_root.exists() or not (orphan_root / "result.json").exists()


def _mid_consumer_resume_allowed(
    *,
    run_dir: Path,
    existing_identity: Mapping[str, Any],
    current_identity: Mapping[str, Any],
    existing_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    requested_stop_after: str,
    execute_gpu: bool,
) -> bool:
    """Allow a bounded mid-consumer continuation.

    Reuse every passed consumer stage before the retryable blocked stage and
    rerun only that blocked stage onward.  blocked.stage must be a retryable
    non-GPU consumer stage after the conversion chain (converted-eval,
    converted-eval-postprocess, or automated-technical-delivery); the GPU
    consumers (native-render/conversion) are never retried.  All producer and
    formal-training evidence plus every consumer stage before the block must
    still be passed with valid artifacts.
    """
    if requested_stop_after != "automated-technical-delivery" or not isinstance(execute_gpu, bool):
        return False
    if not _resume_identity_contract_allowed(
        existing_identity=existing_identity,
        current_identity=current_identity,
        existing_config=existing_config,
        current_config=current_config,
    ):
        return False
    summary_path = run_dir / "run.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        return False
    try:
        summary = _load_json(summary_path, "mid consumer resume summary")
    except (PipelineBlocked, OSError, json.JSONDecodeError):
        return False
    if summary.get("status") != "blocked":
        return False
    blocked = summary.get("blocked")
    if not isinstance(blocked, Mapping):
        return False
    blocked_stage = blocked.get("stage")
    consumer_chain = (
        "native-render",
        "formal-native-render-postcheck",
        "automated-formal-gate",
        "authority-manifest",
        "conversion",
        "converted-eval",
        "converted-eval-postprocess",
        "automated-technical-delivery",
    )
    retryable = {"converted-eval", "converted-eval-postprocess", "automated-technical-delivery"}
    if blocked_stage not in retryable:
        return False
    idx = consumer_chain.index(blocked_stage)
    producer_stages = (
        "preflight",
        "probe",
        "frames",
        "colmap",
        "camera-staging",
        "longsplat-input",
        "convergence-smoke-plan",
        "convergence-smoke-training",
        "convergence-smoke-render",
        "convergence-smoke-render-postcheck",
    )
    for stage in producer_stages:
        if not _stage_latest_passed_with_valid_artifacts(summary, run_dir, stage):
            return False
    if not _stage_latest_passed_with_valid_artifacts(summary, run_dir, "formal-training"):
        return False
    for stage in consumer_chain[:idx]:
        if not _stage_latest_passed_with_valid_artifacts(summary, run_dir, stage):
            return False
    if _stage_latest_status(summary, blocked_stage) not in {None, "blocked", "failed"}:
        return False
    for stage in consumer_chain[idx + 1 :]:
        if _stage_latest_status(summary, stage) is not None:
            return False
    for stage in ("candidate-delivery", "accepted-delivery"):
        if _stage_latest_status(summary, stage) is not None:
            return False
    if summary.get("active_stage") or summary.get("active_attempt"):
        return False
    scope = set(producer_stages) | {"formal-training"} | set(consumer_chain) | {"candidate-delivery", "accepted-delivery"}
    if not _resume_orphan_allowed(summary, run_dir, stage_scope=scope, allow_empty_formal_orphan=True, retry_stage=blocked_stage):
        return False

    return True


def _technical_consumer_resume_allowed(
    *,
    run_dir: Path,
    existing_identity: Mapping[str, Any],
    current_identity: Mapping[str, Any],
    existing_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    requested_stop_after: str,
    execute_gpu: bool,
) -> bool:
    """Allow a safe consumer-only continuation of a passed default run.

    This is intentionally narrower than a general identity migration.  It is
    valid only when all raw/input and convergence producer evidence is already
    immutable and passed; only the postcheck/gate/formal/conversion consumers
    may have changed.  The historical run identity remains the ledger identity
    and every new consumer attempt is append-only.
    """

    if requested_stop_after not in DEFAULT_STAGES or not isinstance(execute_gpu, bool):
        return False
    if not _resume_identity_contract_allowed(
        existing_identity=existing_identity,
        current_identity=current_identity,
        existing_config=existing_config,
        current_config=current_config,
    ):
        return False
    summary_path = run_dir / "run.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        return False
    try:
        summary = _load_json(summary_path, "technical consumer resume summary")
    except (PipelineBlocked, OSError, json.JSONDecodeError):
        return False
    if summary.get("status") not in {"blocked", "stopped"}:
        return False
    formal_training_recovery = False
    if summary.get("status") == "blocked":
        blocked = summary.get("blocked")
        if not isinstance(blocked, Mapping) or blocked.get("stage") not in {"automated-early-gate", "formal-training"}:
            return False
        if blocked.get("stage") == "formal-training":
            formal_entries = summary.get("stages", {}).get("formal-training") if isinstance(summary.get("stages"), Mapping) else None
            if not isinstance(formal_entries, list) or not formal_entries:
                return False
            latest_formal = formal_entries[-1]
            if not isinstance(latest_formal, Mapping) or latest_formal.get("status") != "blocked":
                return False
            preserved_candidates = 0
            for formal_entry in formal_entries:
                if not isinstance(formal_entry, Mapping) or formal_entry.get("status") != "blocked":
                    return False
                result_path_value = formal_entry.get("result_path")
                if not isinstance(result_path_value, str):
                    return False
                try:
                    result_path = (run_dir / result_path_value).resolve(strict=True)
                    result_path.relative_to(run_dir.resolve(strict=True))
                    envelope = _load_json(result_path, "formal recovery stage result")
                    formal_result = envelope.get("result")
                    executor_value = formal_result.get("executor_root") if isinstance(formal_result, Mapping) else None
                    # A later CPU-only preparation/adapter attempt may have no
                    # executor root.  It is safe to append a new recovery
                    # attempt, but it must not hide an earlier completed GPU
                    # child.  Any attempt that does name an executor root is
                    # treated as a producer attempt and validated strictly.
                    if executor_value is None:
                        continue
                    if not isinstance(executor_value, str):
                        return False
                    executor_raw = Path(executor_value)
                    if not executor_raw.is_absolute() or executor_raw.is_symlink():
                        return False
                    try:
                        executor_raw.resolve(strict=False).relative_to(run_dir.resolve(strict=True))
                    except (OSError, ValueError):
                        return False
                    if not executor_raw.exists():
                        continue
                    if not executor_raw.is_dir():
                        return False
                    executor_root = executor_raw.resolve(strict=True)
                    executor_root.relative_to(run_dir.resolve(strict=True))
                    executor_result_path = executor_root / "result.json"
                    # Preparation may create an isolated executor directory
                    # before a CPU-only contract check fails.  Without a
                    # result.json there is no child-process evidence; retain
                    # that directory as an orphan and continue to the earlier
                    # completed GPU child.
                    if not executor_result_path.is_file() or executor_result_path.is_symlink():
                        continue
                    executor_request_path = executor_root / "request.json"
                    executor_argv_path = executor_root / "argv.json"
                    executor_result = _load_json(executor_result_path, "formal recovery executor result")
                    if (
                        executor_result.get("exit_code") != 0
                        or executor_result.get("gpu_invoked") is not True
                        or not isinstance(executor_result.get("model_path"), str)
                        or not executor_request_path.is_file()
                        or executor_request_path.is_symlink()
                        or not executor_argv_path.is_file()
                        or executor_argv_path.is_symlink()
                    ):
                        return False
                    model_path = Path(str(executor_result["model_path"])).resolve(strict=True)
                    model_path.relative_to(run_dir.resolve(strict=True))
                    if not model_path.is_dir() or model_path.is_symlink():
                        return False
                    preserved_candidates += 1
                except (OSError, ValueError, PipelineBlocked, json.JSONDecodeError):
                    return False
            formal_training_recovery = preserved_candidates == 1
            if not formal_training_recovery:
                return False
    stages = summary.get("stages")
    if not isinstance(stages, Mapping):
        return False
    producer_stages = (
        "preflight",
        "probe",
        "frames",
        "colmap",
        "camera-staging",
        "longsplat-input",
        "convergence-smoke-plan",
        "convergence-smoke-training",
        "convergence-smoke-render",
        "convergence-smoke-render-postcheck",
    )
    for stage in producer_stages:
        entries = stages.get(stage)
        if not isinstance(entries, list) or not entries:
            return False
        latest = entries[-1]
        if not isinstance(latest, Mapping) or latest.get("status") != "passed":
            return False
        result_path_value = latest.get("result_path")
        if not isinstance(result_path_value, str):
            return False
        result_path = run_dir / result_path_value
        if result_path.is_symlink() or not result_path.is_file():
            return False
        try:
            envelope = _load_json(result_path, f"technical consumer resume {stage} result")
        except (PipelineBlocked, OSError, json.JSONDecodeError):
            return False
        result = envelope.get("result")
        if not isinstance(result, Mapping) or not RunLedger._artifacts_valid(result, root=run_dir):
            return False
    forbidden_existing = {
        "formal-training",
        "native-render",
        "formal-native-render-postcheck",
        "automated-formal-gate",
        "authority-manifest",
        "conversion",
        "converted-eval",
        "converted-eval-postprocess",
        "automated-technical-delivery",
        "candidate-delivery",
        "accepted-delivery",
    }
    for stage in forbidden_existing:
        if stage == "formal-training" and formal_training_recovery:
            continue
        if stage in stages and stages.get(stage):
            return False
    if summary.get("active_stage") or summary.get("active_attempt"):
        return False
    _orphan_scope = forbidden_existing | {"convergence-smoke-training", "convergence-smoke-render", "convergence-smoke-render-postcheck"}
    if not _resume_orphan_allowed(summary, run_dir, stage_scope=_orphan_scope, allow_empty_formal_orphan=formal_training_recovery):
        return False
    return True


def _diagnostic_consumer_resume_allowed(
    *,
    run_dir: Path,
    existing_identity: Mapping[str, Any],
    current_identity: Mapping[str, Any],
    current_config: Mapping[str, Any],
    coverage_visual_token: str | Path | None,
    requested_stop_after: str,
    execute_gpu: bool,
) -> bool:
    """Permit only the bounded v4 consumer-only diagnostic resume.

    This is intentionally narrower than a general code-drift override: the
    run must be blocked exactly at the coverage visual gate, every producer
    stage through coverage postcheck must still be passed with valid artifacts,
    and source/tool/config identity must remain unchanged.
    """

    summary_path = run_dir / "run.json"
    config_path = run_dir / "config.json"
    if not summary_path.is_file() or not config_path.is_file():
        return False
    try:
        summary = _load_json(summary_path, "diagnostic resume run summary")
        existing_config = _load_json(config_path, "diagnostic resume config")
    except (PipelineBlocked, OSError, json.JSONDecodeError):
        return False
    if requested_stop_after not in {"convergence-smoke-plan", "convergence-smoke-visual-gate"}:
        return False
    if not isinstance(execute_gpu, bool):
        return False
    if existing_identity.get("source_video_sha256") != current_identity.get("source_video_sha256"):
        return False
    current_source_path = current_identity.get("source_video_path")
    current_source_size = current_identity.get("source_video_size_bytes")
    existing_source_path = existing_identity.get("source_video_path")
    existing_source_size = existing_identity.get("source_video_size_bytes")
    if isinstance(current_source_path, str) and isinstance(current_source_size, int):
        if isinstance(existing_source_path, str) or isinstance(existing_source_size, int):
            if existing_source_path != current_source_path or existing_source_size != current_source_size:
                return False
        else:
            legacy_video = _legacy_video_identity(run_dir)
            if legacy_video is None or {
                "path": current_source_path,
                "sha256": current_identity.get("source_video_sha256"),
                "size_bytes": current_source_size,
            } != legacy_video:
                return False
    if existing_identity.get("tool_identity_sha256") != current_identity.get("tool_identity_sha256"):
        return False
    if not isinstance(existing_identity.get("code_identity"), Mapping) or not isinstance(current_identity.get("code_identity"), Mapping):
        return False
    existing_code = existing_identity["code_identity"]
    current_code = current_identity["code_identity"]
    if isinstance(existing_code.get("files"), list) and isinstance(current_code.get("files"), list):
        if _scoped_code_identity(existing_code, "producer")["identity_sha256"] != _scoped_code_identity(current_code, "producer")["identity_sha256"]:
            return False
    if not _diagnostic_config_equivalent(existing_config, current_config):
        return False
    if coverage_visual_token is None:
        return False
    try:
        token = _visual_decision_valid(coverage_visual_token, label="coverage visual decision token")
    except PipelineBlocked:
        return False
    if token["decision"] not in {"fail", "needs_review"}:
        return False
    blocked = summary.get("blocked")
    summary_status = summary.get("status")
    if summary_status == "blocked":
        if not isinstance(blocked, Mapping) or blocked.get("stage") not in {"coverage-smoke-visual-gate", "convergence-smoke-plan"}:
            return False
    elif summary_status == "stopped":
        # A CPU-only evidence recovery may deliberately stop after the
        # corrected coverage postcheck before a decision token is supplied.
        if summary.get("last_stage") not in {"coverage-smoke-render-postcheck", "convergence-smoke-plan"}:
            return False
    else:
        return False
    stages = summary.get("stages")
    if not isinstance(stages, Mapping):
        return False
    producer_stages = (
        "preflight",
        "probe",
        "frames",
        "colmap",
        "camera-staging",
        "longsplat-input",
        "smoke100-training",
        "smoke100-render",
        "coverage-smoke-plan",
        "coverage-smoke-training",
        "coverage-smoke-render",
        "coverage-smoke-render-postcheck",
    )
    for stage in producer_stages:
        entries = stages.get(stage)
        if not isinstance(entries, list) or not entries:
            return False
        latest = entries[-1]
        if not isinstance(latest, Mapping) or latest.get("status") != "passed":
            return False
        result_path_value = latest.get("result_path")
        if not isinstance(result_path_value, str):
            return False
        result_path = (run_dir / result_path_value).resolve()
        if result_path.is_symlink() or not result_path.is_file():
            return False
        try:
            envelope = _load_json(result_path, f"diagnostic resume {stage} result")
        except (PipelineBlocked, OSError, json.JSONDecodeError):
            return False
        result = envelope.get("result")
        if not isinstance(result, Mapping) or not RunLedger._artifacts_valid(result, root=run_dir):
            return False
    forbidden_existing = {
        "convergence-smoke-training",
        "convergence-smoke-render",
        "convergence-smoke-render-postcheck",
        "convergence-smoke-visual-gate",
        "formal-training",
        "native-render",
        "authority-manifest",
        "conversion",
        "converted-eval",
        "candidate-delivery",
        "accepted-delivery",
    }
    if any(stage in stages and stages.get(stage) for stage in forbidden_existing):
        return False
    # A prior outer exception may have created an executor directory and left
    # only ``active_stage``/``orphan_evidence`` in the mutable summary.  It is
    # never reusable, but the diagnostic consumer may create a fresh
    # append-only training attempt when the orphan is the authorized
    # convergence training stage.  Orphan renders or any ledger-recorded GPU
    # stage remain a hard stop.
    orphan_stage = None
    orphan = summary.get("orphan_evidence")
    if isinstance(orphan, Mapping):
        orphan_stage = orphan.get("stage")
    elif summary.get("active_stage") in {"convergence-smoke-training"} and summary.get("active_attempt"):
        orphan_stage = summary.get("active_stage")
    if orphan_stage is not None and orphan_stage != "convergence-smoke-training":
        return False
    if summary.get("active_stage") in forbidden_existing and summary.get("active_stage") != "convergence-smoke-training":
        return False
    # The CPU plan is a permitted diagnostic consumer stage.  A legacy plan
    # from before the consumer-binding contract may be migrated by creating a
    # new append-only plan; a failed/mutated plan is never silently reused.
    plan_entries = stages.get("convergence-smoke-plan")
    if plan_entries:
        if not isinstance(plan_entries, list):
            return False
        latest_plan = plan_entries[-1]
        if not isinstance(latest_plan, Mapping) or latest_plan.get("status") not in {"passed", "blocked"}:
            return False
        if latest_plan.get("status") == "blocked":
            if summary_status != "blocked" or not isinstance(blocked, Mapping) or blocked.get("stage") != "convergence-smoke-plan":
                return False
            # A failed CPU-only plan may be replaced by a fresh append-only
            # plan after a narrowly authorized adapter fix.  GPU stages stay
            # forbidden above and cannot use this recovery path.
            return True
        plan_result_value = latest_plan.get("result_path")
        if not isinstance(plan_result_value, str):
            return False
        plan_result_path = run_dir / plan_result_value
        if plan_result_path.is_symlink() or not plan_result_path.is_file():
            return False
        try:
            plan_envelope = _load_json(plan_result_path, "diagnostic convergence plan result")
        except (PipelineBlocked, OSError, json.JSONDecodeError):
            return False
        plan_result = plan_envelope.get("result")
        if not isinstance(plan_result, Mapping) or not RunLedger._artifacts_valid(plan_result, root=run_dir):
            return False
        if plan_result.get("profile_id") != "convergence1000-v1":
            return False
        convergence_plan = plan_result.get("convergence_plan")
        if not isinstance(convergence_plan, Mapping):
            return False
        names = convergence_plan.get("active_camera_order")
        if (
            convergence_plan.get("requested_iterations") != 1000
            or not isinstance(convergence_plan.get("active_camera_count"), int)
            or convergence_plan.get("active_camera_count") <= 0
            or not isinstance(names, list)
            or len(names) != convergence_plan.get("active_camera_count")
            or len(names) != len(set(names))
        ):
            return False
    return True


def run_reconstruction(
    *,
    input_video: str | Path,
    output_root: str | Path,
    run_id: str,
    stop_after: str = "automated-technical-delivery",
    plan: bool = False,
    route_root: str | Path | None = None,
    acceptance_token: str | Path | None = None,
    authority_manifest: str | Path | None = None,
    depth_source: str = "disabled",
    execute_gpu: bool = False,
    retry_failed_stage: str | None = None,
    code_identity_override: Mapping[str, Any] | str | None = None,
    validate_only: bool = False,
    coverage_visual_token: str | Path | None = None,
    convergence_visual_token: str | Path | None = None,
    conversion_visual_token: str | Path | None = None,
    diagnostic_profile: str | None = None,
    pipeline_profile: str | None = None,
    acceptance_policy: str | None = None,
    camera_model: str = "SIMPLE_RADIAL",
    matching: str = "sequential",
    tool_paths: Mapping[str, str | Path] | None = None,
    publish_dir: str | Path | None = None,
    delivery_name: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance one generic reconstruction run through the real adapters."""

    if stop_after not in STAGE_INDEX:
        raise ValueError(f"stop-after must be one of {STAGES}")
    if retry_failed_stage not in {None, "smoke100-training"}:
        raise ValueError("retry-failed-stage currently supports only smoke100-training")
    if retry_failed_stage is not None and stop_after != retry_failed_stage:
        raise ValueError("a restricted GPU retry must stop after the authorized stage")
    if delivery_name is not None and publish_dir is None:
        raise ValueError("delivery-name requires publish-dir")
    if depth_source != "disabled":
        raise PipelineBlocked("the first one-click chain supports only --depth-source disabled")
    if camera_model not in SUPPORTED_CAMERA_MODELS:
        raise PipelineBlocked(
            f"unsupported camera model {camera_model!r}; supported models are "
            f"{sorted(SUPPORTED_CAMERA_MODELS)}"
        )
    if matching not in SUPPORTED_MATCHING_MODES:
        raise PipelineBlocked(
            f"unsupported matcher {matching!r}; supported matchers are "
            f"{sorted(SUPPORTED_MATCHING_MODES)}"
        )
    provider_tools, tool_provider = resolve_tool_provider(
        route_root or Path(__file__).resolve().parents[2], tool_paths
    )
    del provider_tools
    run_id = _safe_run_id(run_id)
    source = Path(input_video).resolve()
    if source.is_symlink() or not source.is_file():
        raise PipelineBlocked(f"input video is missing or symlinked: {source}")
    route = Path(route_root or Path(__file__).resolve().parents[2]).resolve()
    run_output_root = _route_output_root(output_root, route)
    run_root = run_output_root / run_id
    if run_root == run_output_root:
        raise PipelineBlocked("run root must be a strict child of output root")
    if run_root.is_symlink():
        raise PipelineBlocked(f"run root is symlinked: {run_root}")
    if run_root.exists() and not run_root.is_dir():
        raise PipelineBlocked(f"run root must be a directory: {run_root}")
    if validate_only:
        return validate_existing_run(
            route_root=route,
            run_root=run_output_root / run_id,
            input_video=source,
            authority_manifest=authority_manifest,
        )
    existing_config_path = run_root / "config.json"
    existing_config: Mapping[str, Any] | None = None
    if existing_config_path.is_file() and not existing_config_path.is_symlink():
        existing_config = _load_json(existing_config_path, "existing reconstruction config")
    if pipeline_profile is None:
        if isinstance(existing_config, Mapping) and isinstance(existing_config.get("pipeline_profile"), str):
            pipeline_profile = str(existing_config["pipeline_profile"])
        elif stop_after not in DEFAULT_STAGES:
            # Preserve direct callers that explicitly request a legacy stage;
            # the canonical CLI default is the new single-convergence profile.
            pipeline_profile = LEGACY_PIPELINE_PROFILE
        else:
            pipeline_profile = DEFAULT_PIPELINE_PROFILE
    if pipeline_profile not in {DEFAULT_PIPELINE_PROFILE, LEGACY_PIPELINE_PROFILE}:
        raise ValueError("pipeline-profile must be single-convergence1000-v1 or legacy-smoke-coverage-v1")
    stage_order = DEFAULT_STAGES if pipeline_profile == DEFAULT_PIPELINE_PROFILE else STAGES
    if acceptance_policy is None:
        acceptance_policy = AUTOMATED_POLICY_ID if pipeline_profile == DEFAULT_PIPELINE_PROFILE else "manual"
    if acceptance_policy not in {AUTOMATED_POLICY_ID, "manual"}:
        raise ValueError("acceptance-policy must be automated-technical-v1 or manual")
    if pipeline_profile == DEFAULT_PIPELINE_PROFILE and acceptance_policy != AUTOMATED_POLICY_ID:
        raise PipelineBlocked("the default single-convergence chain requires automated-technical-v1")
    if pipeline_profile == DEFAULT_PIPELINE_PROFILE and stop_after not in stage_order:
        raise ValueError("single-convergence1000-v1 accepts stop-after only through automated-technical-delivery")
    if diagnostic_profile not in {None, "convergence1000-v1"}:
        raise ValueError("diagnostic-profile must be convergence1000-v1 or omitted")
    config = {
        "schema_version": "reconstruct-pipeline-config-v2",
        "depth_source": depth_source,
        "pipeline_profile": pipeline_profile,
        "stage_order": list(stage_order),
        "smoke_profile": _SMOKE_PROFILE,
        "formal_profile": _FORMAL_PROFILE,
        "conversion_profile": "standard30000-v1",
        "authority_manifest_mode": "provided" if authority_manifest is not None else "generated",
        "authority_manifest": None if authority_manifest is None else str(Path(authority_manifest).resolve()),
        "acceptance_policy": acceptance_policy,
        "acceptance_import": "automated-technical-delivery" if acceptance_policy == AUTOMATED_POLICY_ID else "explicit-token-only",
        "coverage_visual_gate": "explicit-token-only",
        "convergence_visual_gate": "explicit-token-only-diagnostic-only",
        "converted_eval_postprocess": "resident-cpu-streaming-recovery",
        "diagnostic_profile": diagnostic_profile,
        "output_root_input": str(output_root),
        "output_root_resolved": str(run_output_root),
        "run_root_resolved": str(run_root),
        "camera_model": camera_model,
        "matching": matching,
        "tool_provider": tool_provider,
    }
    if publish_dir is not None:
        config["publish_dir"] = str(Path(publish_dir).resolve())
        config["delivery_name"] = delivery_name
    if source_metadata is not None:
        config["source_metadata"] = dict(source_metadata)
    code = code_identity_override if code_identity_override is not None else code_identity(route)
    identity = build_run_identity(
        source_video_sha256=sha256_file(source),
        canonical_config_sha256=stable_sha256(config),
        tool_identity_sha256=tool_provider["provider_identity_sha256"],
        code_identity_value=code,
        source_video_path=source,
        source_video_size_bytes=source.stat().st_size,
        output_root_input=output_root,
        output_root_resolved=run_output_root,
        run_root_resolved=run_root,
    )
    input_video_identity = _input_video_identity(source, identity)
    consumer_identity = _convergence_consumer_identity(identity)
    # An explicitly authorized executor-only retry may continue the immutable
    # run ledger whose CPU identity was recorded before the adapter patch.  The
    # new producer identity is separately bound into the derived GPU plan and
    # retry result; source/config/tool identity must still match exactly.
    ledger_identity = identity
    diagnostic_consumer_resume = False
    technical_consumer_resume = False
    mid_consumer_resume = False
    existing_identity_path = run_output_root / run_id / "identity.json"
    if retry_failed_stage is not None and existing_identity_path.is_file():
        existing_identity = _load_json(existing_identity_path, "existing reconstruction identity")
        for key in ("source_video_sha256", "canonical_config_sha256", "tool_identity_sha256"):
            if existing_identity.get(key) != identity.get(key):
                raise ResumeMismatchError(f"retry identity mismatch for {key}")
        ledger_identity = existing_identity
    elif pipeline_profile == DEFAULT_PIPELINE_PROFILE and existing_identity_path.is_file() and isinstance(existing_config, Mapping):
        existing_identity = _load_json(existing_identity_path, "existing reconstruction identity")
        if _technical_consumer_resume_allowed(
            run_dir=run_output_root / run_id,
            existing_identity=existing_identity,
            current_identity=identity,
            existing_config=existing_config,
            current_config=config,
            requested_stop_after=stop_after,
            execute_gpu=execute_gpu,
        ):
            ledger_identity = existing_identity
            technical_consumer_resume = True
        elif _mid_consumer_resume_allowed(
            run_dir=run_output_root / run_id,
            existing_identity=existing_identity,
            current_identity=identity,
            existing_config=existing_config,
            current_config=config,
            requested_stop_after=stop_after,
            execute_gpu=execute_gpu,
        ):
            ledger_identity = existing_identity
            mid_consumer_resume = True
    elif diagnostic_profile == "convergence1000-v1" and existing_identity_path.is_file():
        existing_identity = _load_json(existing_identity_path, "existing reconstruction identity")
        if _diagnostic_consumer_resume_allowed(
            run_dir=run_output_root / run_id,
            existing_identity=existing_identity,
            current_identity=identity,
            current_config=config,
            coverage_visual_token=coverage_visual_token,
            requested_stop_after=stop_after,
            execute_gpu=execute_gpu,
        ):
            # The v4 producer stages were already completed under the prior
            # route identity.  Keep that immutable ledger identity and bind
            # the new consumer/diagnostic identity into fresh convergence
            # plan and stage artifacts instead of rerunning producers.
            ledger_identity = existing_identity
            diagnostic_consumer_resume = True
    if (
        existing_identity_path.is_file()
        and isinstance(existing_config, Mapping)
        and ledger_identity is identity
    ):
        existing_identity = _load_json(existing_identity_path, "existing reconstruction identity")
        if _producer_resume_allowed(
            existing_identity=existing_identity,
            current_identity=identity,
            existing_config=existing_config,
            current_config=config,
        ):
            ledger_identity = existing_identity
    ledger = RunLedger.create_or_resume(output_root=run_output_root, run_id=run_id, identity=ledger_identity)
    if source_metadata is not None:
        source_receipt = ledger.run_dir / "source_receipt.json"
        write_json_once_atomic(
            source_receipt,
            {"schema_version": "longsplat-source-receipt-v1", **dict(source_metadata)},
        )
        ledger.summary["source_receipt"] = str(source_receipt)
        ledger._write_summary()
    config_path = ledger.run_dir / "config.json"
    if config_path.exists():
        existing_config = _load_json(config_path, "reconstruction config")
        if existing_config != config and not (
            diagnostic_consumer_resume
            and _diagnostic_config_equivalent(existing_config, config)
        ):
            raise ResumeMismatchError("reconstruction config differs on resume")
    else:
        _write_json_once(config_path, config)
    if not plan and publish_dir is not None:
        _verify_existing_public_delivery(
            ledger=ledger,
            publish_dir=publish_dir,
            delivery_name=delivery_name,
        )
    if diagnostic_consumer_resume:
        ledger.summary["consumer_diagnostic_resume"] = {
            "schema_version": "consumer-diagnostic-resume-v1",
            "producer_code_identity_sha256": ledger_identity.get("code_identity", {}).get("code_identity_sha256"),
            "consumer_code_identity_sha256": identity.get("code_identity", {}).get("code_identity_sha256"),
            "producer_stages_frozen_through": "coverage-smoke-render-postcheck",
            "fresh_stages": [
                "coverage-smoke-render-postcheck",
                "coverage-smoke-visual-gate",
                "convergence-smoke-plan",
                "convergence-smoke-training",
                "convergence-smoke-render",
                "convergence-smoke-render-postcheck",
                "convergence-smoke-visual-gate",
            ],
        }
        ledger._write_summary()
    elif technical_consumer_resume:
        ledger.summary["consumer_technical_resume"] = {
            "schema_version": "consumer-technical-resume-v1",
            "producer_code_identity_sha256": _scoped_code_identity(ledger_identity.get("code_identity", {}), "producer")["identity_sha256"],
            "consumer_code_identity_sha256": _scoped_code_identity(identity.get("code_identity", {}), "convergence-consumer")["identity_sha256"],
            "producer_stages_frozen_through": "convergence-smoke-render-postcheck",
            "fresh_stages": [
                "automated-early-gate",
                "formal-training",
                "native-render",
                "formal-native-render-postcheck",
                "automated-formal-gate",
                "authority-manifest",
                "conversion",
                "converted-eval",
                "converted-eval-postprocess",
                "automated-technical-delivery",
            ],
        }
        ledger._write_summary()
    elif mid_consumer_resume:
        ledger.summary["consumer_mid_resume"] = {
            "schema_version": "consumer-mid-resume-v1",
            "producer_code_identity_sha256": _scoped_code_identity(ledger_identity.get("code_identity", {}), "producer")["identity_sha256"],
            "producer_stages_frozen_through": "converted-eval",
            "reused_through": [
                "preflight",
                "probe",
                "frames",
                "colmap",
                "camera-staging",
                "longsplat-input",
                "convergence-smoke-plan",
                "convergence-smoke-training",
                "convergence-smoke-render",
                "convergence-smoke-render-postcheck",
                "formal-training",
                "native-render",
                "formal-native-render-postcheck",
                "automated-formal-gate",
                "authority-manifest",
                "conversion",
            ],
            "fresh_stages": ["converted-eval", "converted-eval-postprocess", "automated-technical-delivery"],
        }
        ledger._write_summary()

    results: dict[str, Any] = {}
    stop_reason: str | None = None

    # ``--plan`` is intentionally side-effect-light: it records the complete
    # route and does not create a raw nested run with missing pixel evidence.
    if plan:
        stage_positions = {name: index for index, name in enumerate(stage_order)}
        for stage in stage_order:
            if stage_positions[stage] > stage_positions[stop_after]:
                break
            reused = _stage_reusable(ledger, stage)
            if reused is not None:
                results[stage] = {**reused, "reused": True}
                continue
            if stage in GPU_STAGES:
                result = _stage_result(stage=stage, status="planned", computed_pass=False, reason="requires_escalated_gpu_execution", plan=True, gpu_invoked=False)
            else:
                result = _stage_result(stage=stage, status="planned", computed_pass=False, reason="one-click stage plan recorded; execution not requested", plan=True, algorithm_owner="scripts.longsplat.raw_pipeline" if stage in CPU_STAGES else "authority-driven executor")
            _record_stage(ledger, stage, result, status="planned")
            results[stage] = result
        stop_reason = f"requested plan through {stop_after}"
        _finish_summary(ledger, results, status="planned", reason=stop_reason)
        return _run_result(ledger, results, stop_after=stop_after, plan=True, next_slice="resume without --plan to execute passed CPU stages")

    # Keep the original contract tests useful when they construct an isolated
    # temporary route with a synthetic source and no project wrapper.  This
    # branch is unreachable for the authoritative route (which has dev.sh) and
    # records an explicit fixture boundary instead of pretending raw evidence
    # exists.
    if code_identity_override is not None and not (route / "dev.sh").is_file():
        stage_positions = {name: index for index, name in enumerate(stage_order)}
        for stage in stage_order:
            if stage_positions[stage] > stage_positions[stop_after]:
                break
            reused = _stage_reusable(ledger, stage)
            if reused is not None:
                results[stage] = {**reused, "reused": True}
                continue
            if stage in GPU_STAGES:
                boundary = _stage_result(stage=stage, status="requires_escalated_gpu_execution", computed_pass=False, reason="requires_escalated_gpu_execution", plan=False, gpu_invoked=False, fixture_route=True)
                _record_stage(ledger, stage, boundary, status="blocked")
                results[stage] = boundary
                break
            boundary = _stage_result(stage=stage, status="boundary_only", computed_pass=True, reason="fixture route has no executable dev.sh; raw adapter intentionally not invoked", plan=False, algorithm_owner="scripts.longsplat.raw_pipeline", fixture_route=True)
            _record_stage(ledger, stage, boundary, status="passed")
            results[stage] = boundary
        _finish_summary(ledger, results, status="blocked" if any(value.get("status") == "requires_escalated_gpu_execution" for value in results.values()) else "stopped", reason="fixture route boundary" )
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="authoritative route uses the real raw_pipeline adapter")

    # Reuse or execute the CPU raw adapter.  The adapter itself owns the
    # ffprobe/ffmpeg/COLMAP/camera-staging algorithms and its nested evidence.
    cpu_target = min(STAGE_INDEX[stop_after], STAGE_INDEX["longsplat-input"])
    camera_result = _stage_reusable(ledger, "camera-staging")
    camera_run_dir: Path | None = Path(str(camera_result["raw_run_dir"])).resolve() if camera_result and isinstance(camera_result.get("raw_run_dir"), str) else None
    if camera_run_dir is None and cpu_target >= STAGE_INDEX["preflight"]:
        from .raw_pipeline import run_raw_video_pipeline

        raw_camera_root = ledger.run_dir / "raw-camera"
        raw_summary = run_raw_video_pipeline(
            input_video=source,
            output_root=raw_camera_root,
            run_id="camera",
            route_root=route,
            depth_source=depth_source,
            stop_after=STAGES[cpu_target] if cpu_target <= STAGE_INDEX["camera-staging"] else "camera-staging",
            future_smoke_profile=_SMOKE_PROFILE,
            camera_model=camera_model,
            matching=matching,
            tool_paths=tool_paths,
        )
        camera_run_dir = Path(str(raw_summary.get("run_dir", raw_camera_root / "camera"))).resolve()
    if camera_run_dir is not None:
        for stage in CPU_STAGES[: min(cpu_target + 1, STAGE_INDEX["camera-staging"] + 1)]:
            if stage in results:
                continue
            reused = _stage_reusable(ledger, stage)
            if reused is not None:
                results[stage] = {**reused, "reused": True}
                continue
            adapted = _adapt_raw_stage(stage=stage, raw_run_dir=camera_run_dir, plan=False)
            if adapted is None:
                stop_reason = f"raw pipeline did not produce stage {stage}"
                break
            _record_stage(ledger, stage, adapted, status=str(adapted["status"]))
            results[stage] = adapted
            if adapted["status"] != "passed":
                stop_reason = adapted["reason"]
                break
            if STAGE_INDEX[stage] == cpu_target:
                break
    if stop_reason is not None:
        _finish_summary(ledger, results, status="blocked", reason=stop_reason)
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="fix or inspect the failed CPU raw_pipeline attempt")

    if cpu_target >= STAGE_INDEX["longsplat-input"]:
        input_result = _stage_reusable(ledger, "longsplat-input")
        if input_result is None:
            if camera_run_dir is None:
                raise PipelineBlocked("longsplat-input requires a passed camera-staging raw run")
            from .raw_pipeline import run_raw_video_pipeline

            raw_input_root = ledger.run_dir / "raw-smoke-input"
            raw_input_summary = run_raw_video_pipeline(
                input_video=source,
                output_root=raw_input_root,
                run_id="smoke-input",
                parent_run_root=camera_run_dir,
                containment_root=ledger.run_dir,
                route_root=route,
                depth_source=depth_source,
                stop_after="longsplat-input",
                future_smoke_profile=_SMOKE_PROFILE,
                camera_model=camera_model,
                matching=matching,
                tool_paths=tool_paths,
            )
            smoke_input_run_dir = Path(str(raw_input_summary.get("run_dir", raw_input_root / "smoke-input"))).resolve()
            adapted = _adapt_raw_stage(stage="longsplat-input", raw_run_dir=smoke_input_run_dir, plan=False)
            if adapted is None or adapted["status"] != "passed":
                stop_reason = "LongSplat smoke input packaging did not pass"
            else:
                payload = adapted["raw_result"]
                adapted.update(_profile_input(payload, _SMOKE_PROFILE))
                adapted["camera_run_dir"] = str(camera_run_dir)
                _record_stage(ledger, "longsplat-input", adapted, status="passed")
                input_result = adapted
                results["longsplat-input"] = adapted
        else:
            results["longsplat-input"] = {**input_result, "reused": True}
    if stop_reason is not None:
        _finish_summary(ledger, results, status="blocked", reason=stop_reason)
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the LongSplat input attempt")
    if STAGE_INDEX[stop_after] <= STAGE_INDEX["longsplat-input"]:
        _finish_summary(ledger, results, status="stopped", reason=f"requested stop-after {stop_after}")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at the next stage")

    if pipeline_profile == DEFAULT_PIPELINE_PROFILE:
        default_results, default_status = _run_default_single_convergence(
            ledger=ledger,
            route=route,
            run_output_root=run_output_root,
            input_result=results["longsplat-input"],
            input_video_identity=input_video_identity,
            consumer_identity=consumer_identity,
            frames_result=results.get("frames"),
            colmap_result=results.get("colmap"),
            stop_after=stop_after,
            execute_gpu=execute_gpu,
        )
        results.update(default_results)
        if default_status != "passed":
            return _run_result(
                ledger,
                results,
                stop_after=stop_after,
                plan=False,
                next_slice="inspect the preserved automated early gate; formal remains closed",
            )
        if stop_after == "automated-early-gate":
            return _run_result(
                ledger,
                results,
                stop_after=stop_after,
                plan=False,
                next_slice="resume at formal-training",
            )
        # The default profile has its own single formal/conversion/delivery
        # continuation.  Keep the legacy smoke/coverage orchestration below
        # reachable only for the explicitly selected legacy profile.
        return _run_default_formal_delivery(
            ledger=ledger,
            route=route,
            source=source,
            camera_run_dir=camera_run_dir,
            results=results,
            stop_after=stop_after,
            execute_gpu=execute_gpu,
            authority_manifest=authority_manifest,
            camera_model=camera_model,
            matching=matching,
            tool_paths=tool_paths,
            tool_provider=tool_provider,
            publish_dir=publish_dir,
            delivery_name=delivery_name,
        )

    smoke_input = _profile_input(results["longsplat-input"].get("raw_result", results["longsplat-input"]), _SMOKE_PROFILE)
    smoke_input["source_path"] = results["longsplat-input"].get("source_path", smoke_input["source_path"])
    smoke_training = _stage_reusable(ledger, "smoke100-training")
    if smoke_training is None:
        if retry_failed_stage is not None:
            previous = ledger.latest_attempt("smoke100-training")
            if previous is None or previous.get("reusable") is True:
                raise PipelineBlocked("authorized smoke100-training retry requires a preserved non-reusable prior attempt")
        smoke_training, status = _execute_training_or_render(
            ledger=ledger,
            stage="smoke100-training",
            profile_input=smoke_input,
            plan=False,
            execute_gpu=execute_gpu,
            route=route,
            retry_failed_attempt=retry_failed_stage == "smoke100-training",
        )
        results["smoke100-training"] = smoke_training
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=smoke_training["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="rerun the same run under require_escalated GPU permission" if status == "blocked" and smoke_training.get("reason") == "requires_escalated_gpu_execution" else "inspect the preserved smoke training attempt")
    else:
        results["smoke100-training"] = {**smoke_training, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["smoke100-training"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after smoke100-training")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at smoke100-render")

    smoke_render = _stage_reusable(ledger, "smoke100-render")
    if smoke_render is None:
        smoke_render, status = _execute_training_or_render(
            ledger=ledger,
            stage="smoke100-render",
            profile_input=smoke_input,
            plan=False,
            execute_gpu=execute_gpu,
            route=route,
            training_evidence_root=Path(str(smoke_training["executor_root"])),
        )
        results["smoke100-render"] = smoke_render
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=smoke_render["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved smoke render attempt")
    else:
        results["smoke100-render"] = {**smoke_render, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["smoke100-render"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after smoke100-render")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at coverage-smoke-plan")

    coverage_plan = _stage_reusable(ledger, "coverage-smoke-plan")
    if coverage_plan is None:
        coverage_plan, status = _coverage_plan_stage(
            ledger=ledger,
            input_result=results["longsplat-input"],
            frames_result=results.get("frames"),
            route=route,
            plan=False,
        )
        results["coverage-smoke-plan"] = coverage_plan
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=coverage_plan["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved coverage-smoke CPU plan")
    else:
        results["coverage-smoke-plan"] = {**coverage_plan, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["coverage-smoke-plan"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after coverage-smoke-plan")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at coverage-smoke-training")

    if smoke_render.get("computed_pass") is not True:
        smoke_reason = "smoke100 structural pass is required before coverage-smoke; rough visual remains a separate decision"
        _finish_summary(ledger, results, status="blocked", reason=smoke_reason)
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved smoke render structural evidence")

    coverage_profile_input = dict(smoke_input)
    coverage_profile_input.update(
        {
            "profile": "coverage-smoke-v1",
            "parent_plan_path": smoke_input["plan_path"],
            "coverage_plan_path": coverage_plan["coverage_plan_path"],
        }
    )
    coverage_training = _stage_reusable(ledger, "coverage-smoke-training")
    if coverage_training is None:
        coverage_training, status = _execute_training_or_render(
            ledger=ledger,
            stage="coverage-smoke-training",
            profile_input=coverage_profile_input,
            plan=False,
            execute_gpu=execute_gpu,
            route=route,
            prepare=lambda attempt, profile: _prepare_coverage_training_plan(attempt, profile, route),
        )
        results["coverage-smoke-training"] = coverage_training
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=coverage_training["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume coverage-smoke-training only under require_escalated GPU permission")
    else:
        results["coverage-smoke-training"] = {**coverage_training, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["coverage-smoke-training"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after coverage-smoke-training")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at coverage-smoke-render")

    coverage_render_input = dict(coverage_profile_input)
    if isinstance(coverage_training.get("plan_path"), str):
        coverage_render_input["plan_path"] = coverage_training["plan_path"]
    coverage_render = _stage_reusable(ledger, "coverage-smoke-render")
    if coverage_render is None:
        coverage_render, status = _execute_training_or_render(
            ledger=ledger,
            stage="coverage-smoke-render",
            profile_input=coverage_render_input,
            plan=False,
            execute_gpu=execute_gpu,
            route=route,
            training_evidence_root=Path(str(coverage_training.get("executor_root", ""))),
        )
        results["coverage-smoke-render"] = coverage_render
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=coverage_render["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved coverage-smoke render attempt")
    else:
        results["coverage-smoke-render"] = {**coverage_render, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["coverage-smoke-render"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after coverage-smoke-render")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at coverage-smoke-render-postcheck")

    coverage_postcheck = _stage_reusable(ledger, "coverage-smoke-render-postcheck")
    if coverage_postcheck is None:
        coverage_postcheck, status = _coverage_render_postcheck_stage(
            ledger=ledger,
            route=route,
            coverage_render=coverage_render,
            coverage_training=coverage_training,
            training_input=coverage_profile_input,
            frames_result=results.get("frames"),
            colmap_result=results.get("colmap"),
            plan=False,
        )
        results["coverage-smoke-render-postcheck"] = coverage_postcheck
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=coverage_postcheck["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume coverage-smoke-render-postcheck CPU-only; preserved coverage render is reusable")
    else:
        results["coverage-smoke-render-postcheck"] = {**coverage_postcheck, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["coverage-smoke-render-postcheck"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after coverage-smoke-render-postcheck")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at coverage-smoke-visual-gate")

    coverage_gate = _stage_reusable(ledger, "coverage-smoke-visual-gate")
    if coverage_gate is None:
        coverage_gate, status = _coverage_visual_gate(
            ledger=ledger,
            smoke=smoke_render,
            coverage_plan=coverage_plan,
            coverage_render=coverage_render,
            coverage_postcheck=coverage_postcheck,
            decision_token=coverage_visual_token,
            diagnostic_profile=diagnostic_profile,
        )
        results["coverage-smoke-visual-gate"] = coverage_gate
        if status != "passed" and coverage_gate.get("diagnostic_release_eligible") is not True:
            _finish_summary(ledger, results, status="blocked", reason=coverage_gate["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="supply an explicit coverage-smoke visual decision token; convergence is opt-in only")
    else:
        results["coverage-smoke-visual-gate"] = {**coverage_gate, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["coverage-smoke-visual-gate"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after coverage-smoke-visual-gate")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at formal-training")

    coverage_decision = coverage_gate.get("coverage_smoke_visual_decision")
    if coverage_gate.get("computed_pass") is not True:
        if (
            diagnostic_profile != "convergence1000-v1"
            or coverage_decision not in {"fail", "needs_review"}
            or coverage_gate.get("diagnostic_release_eligible") is not True
        ):
            _finish_summary(ledger, results, status="blocked", reason="coverage-smoke visual decision did not release formal training; convergence diagnostic requires explicit convergence1000-v1 plus fail/needs_review evidence")
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the explicit coverage visual decision or opt in to the one-shot convergence1000-v1 diagnostic")

        # Plan comparison can fail before begin_attempt.  Record this
        # invocation's intended stage first so an outer exception cannot be
        # attributed to a stale active stage from an earlier invocation.
        ledger.set_current_stage("convergence-smoke-plan")
        convergence_plan = _stage_reusable(ledger, "convergence-smoke-plan")
        convergence_plan_rebuild_reason: Mapping[str, Any] | None = None
        if convergence_plan is not None:
            comparison = _convergence_plan_matches_consumer(
                plan_result=convergence_plan,
                input_result=results["longsplat-input"],
                input_video_identity=input_video_identity,
                consumer_identity=consumer_identity,
                run_dir=ledger.run_dir,
                route=route,
                output_root=run_output_root,
                coverage_visual_token=coverage_visual_token,
                coverage_decision=coverage_decision,
            )
            if comparison["classification"] == "unsafe_drift":
                blocked_plan = _stage_result(
                    stage="convergence-smoke-plan",
                    status="blocked",
                    computed_pass=False,
                    reason=str(comparison["reason"]),
                    plan=False,
                    gpu_invoked=False,
                    diagnostic_only=True,
                    formal_auto_release=False,
                    plan_consumer_comparison=comparison,
                )
                _record_stage(ledger, "convergence-smoke-plan", blocked_plan, status="blocked")
                results["convergence-smoke-plan"] = blocked_plan
                _finish_summary(ledger, results, status="blocked", reason=blocked_plan["reason"])
                return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the immutable convergence plan drift")
            if comparison["classification"] == "stale_rebuildable":
                # Keep the old passed attempt immutable and create a fresh
                # append-only plan carrying the precise migration reason.
                convergence_plan = None
                convergence_plan_rebuild_reason = comparison
        if convergence_plan is None:
            convergence_plan, status = _convergence_plan_stage(
                ledger=ledger,
                input_result=results["longsplat-input"],
                route=route,
                output_root=run_output_root,
                plan=False,
                input_video_identity=input_video_identity,
                consumer_identity=consumer_identity,
                coverage_visual_token=coverage_visual_token,
                coverage_decision=coverage_decision,
                rebuild_reason=convergence_plan_rebuild_reason,
            )
            results["convergence-smoke-plan"] = convergence_plan
            if status != "passed":
                _finish_summary(ledger, results, status="blocked", reason=convergence_plan["reason"])
                return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved convergence1000-v1 CPU plan")
        else:
            results["convergence-smoke-plan"] = {**convergence_plan, "reused": True}
        if STAGE_INDEX[stop_after] == STAGE_INDEX["convergence-smoke-plan"]:
            _finish_summary(ledger, results, status="stopped", reason="requested stop-after convergence-smoke-plan")
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at convergence-smoke-training")

        convergence_profile_input = dict(smoke_input)
        convergence_profile_input.update(
            {
                "profile": "convergence1000-v1",
                "parent_plan_path": smoke_input["plan_path"],
                "convergence_plan_path": convergence_plan["convergence_plan_path"],
            }
        )
        convergence_training = _stage_reusable(ledger, "convergence-smoke-training")
        if convergence_training is None:
            convergence_training, status = _execute_training_or_render(
                ledger=ledger,
                stage="convergence-smoke-training",
                profile_input=convergence_profile_input,
                plan=False,
                execute_gpu=execute_gpu,
                route=route,
                prepare=lambda attempt, profile: _prepare_convergence_training_plan(attempt, profile, route),
                extra={
                    "diagnostic_only": True,
                    "coverage_visual_decision": coverage_decision,
                    "formal_auto_release": False,
                },
            )
            results["convergence-smoke-training"] = convergence_training
            if status != "passed":
                _finish_summary(ledger, results, status="blocked", reason=convergence_training["reason"])
                return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume convergence1000-v1 only under the authorized require_escalated GPU command")
        else:
            results["convergence-smoke-training"] = {**convergence_training, "reused": True}
        if STAGE_INDEX[stop_after] == STAGE_INDEX["convergence-smoke-training"]:
            _finish_summary(ledger, results, status="stopped", reason="requested stop-after convergence-smoke-training")
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at convergence-smoke-render")

        convergence_render_input = dict(convergence_profile_input)
        if isinstance(convergence_training.get("plan_path"), str):
            convergence_render_input["plan_path"] = convergence_training["plan_path"]
        convergence_render = _stage_reusable(ledger, "convergence-smoke-render")
        if convergence_render is None:
            convergence_render, status = _execute_training_or_render(
                ledger=ledger,
                stage="convergence-smoke-render",
                profile_input=convergence_render_input,
                plan=False,
                execute_gpu=execute_gpu,
                route=route,
                training_evidence_root=Path(str(convergence_training.get("executor_root", ""))),
                isolated_render_snapshot=True,
                extra={"diagnostic_only": True, "formal_auto_release": False},
            )
            results["convergence-smoke-render"] = convergence_render
            if status != "passed":
                _finish_summary(ledger, results, status="blocked", reason=convergence_render["reason"])
                return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved convergence1000-v1 render attempt")
        else:
            results["convergence-smoke-render"] = {**convergence_render, "reused": True}
        if STAGE_INDEX[stop_after] == STAGE_INDEX["convergence-smoke-render"]:
            _finish_summary(ledger, results, status="stopped", reason="requested stop-after convergence-smoke-render")
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at convergence-smoke-render-postcheck")

        smoke_attempt = Path(str(smoke_render.get("executor_root", ""))).resolve().parent
        coverage_attempt = Path(str(coverage_render.get("executor_root", ""))).resolve().parent
        convergence_postcheck = _stage_reusable(ledger, "convergence-smoke-render-postcheck")
        if convergence_postcheck is None:
            convergence_postcheck, status = _convergence_render_postcheck_stage(
                ledger=ledger,
                route=route,
                convergence_render=convergence_render,
                convergence_training=convergence_training,
                training_input=convergence_profile_input,
                frames_result=results.get("frames"),
                colmap_result=results.get("colmap"),
                baseline_render_attempt=smoke_attempt,
                comparison_render_attempt=coverage_attempt,
                plan=False,
            )
            results["convergence-smoke-render-postcheck"] = convergence_postcheck
            if status != "passed":
                _finish_summary(ledger, results, status="blocked", reason=convergence_postcheck["reason"])
                return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume convergence render postcheck CPU-only; preserve the convergence render")
        else:
            results["convergence-smoke-render-postcheck"] = {**convergence_postcheck, "reused": True}
        if STAGE_INDEX[stop_after] == STAGE_INDEX["convergence-smoke-render-postcheck"]:
            _finish_summary(ledger, results, status="stopped", reason="requested stop-after convergence-smoke-render-postcheck")
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at convergence-smoke-visual-gate")

        convergence_gate = _stage_reusable(ledger, "convergence-smoke-visual-gate")
        if convergence_gate is None:
            convergence_gate, status = _convergence_visual_gate(
                ledger=ledger,
                convergence_postcheck=convergence_postcheck,
                decision_token=convergence_visual_token,
            )
            results["convergence-smoke-visual-gate"] = convergence_gate
        else:
            results["convergence-smoke-visual-gate"] = {**convergence_gate, "reused": True}
        # A convergence decision is diagnostic evidence only.  It deliberately
        # terminates the canonical slice before formal training regardless of
        # whether the human decision is pass, needs_review, or fail.
        final_status = "stopped" if convergence_gate.get("computed_pass") is True else "blocked"
        _finish_summary(ledger, results, status=final_status, reason="convergence1000-v1 diagnostic visual gate; formal training and delivery remain closed")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="supervisor review at the convergence diagnostic/formal-release boundary")

    if diagnostic_profile == "convergence1000-v1":
        _finish_summary(ledger, results, status="blocked", reason="convergence1000-v1 is diagnostic-only and requires coverage visual fail/needs_review; it cannot follow a coverage pass")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="continue the normal formal path only after a separate coverage pass decision")

    formal = _stage_reusable(ledger, "formal-training")
    if formal is None:
        try:
            formal_input = _formal_input(
                source=source,
                route=route,
                ledger=ledger,
                camera_run_dir=camera_run_dir or Path(),
                camera_model=camera_model,
                matching=matching,
                tool_paths=tool_paths,
            )
            formal_profile_input = _profile_input(formal_input["payload"], _FORMAL_PROFILE)
            formal_profile_input.update({"source_path": formal_input["source_path"], "raw_run_dir": formal_input["raw_run_dir"]})
        except Exception as exc:
            result = _stage_result(stage="formal-training", status="blocked", computed_pass=False, reason=str(exc), plan=False, gpu_invoked=False)
            _record_stage(ledger, "formal-training", result, status="blocked")
            results["formal-training"] = result
            _finish_summary(ledger, results, status="blocked", reason=str(exc))
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the formal input packaging attempt")
        formal, status = _execute_training_or_render(ledger=ledger, stage="formal-training", profile_input=formal_profile_input, plan=False, execute_gpu=execute_gpu, route=route, extra={"formal_input": formal_input})
        results["formal-training"] = formal
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=formal["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved formal training attempt")
    else:
        results["formal-training"] = {**formal, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["formal-training"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after formal-training")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at native-render")

    formal_payload = formal.get("formal_input", {}).get("payload") if isinstance(formal.get("formal_input"), Mapping) else None
    if not isinstance(formal_payload, Mapping):
        formal_payload = _load_json(Path(str(formal.get("raw_run_dir", ""))) / "stages/longsplat-input/attempt-0001/result.json", "formal input result").get("result", {})
    formal_profile_input = _profile_input(formal_payload, _FORMAL_PROFILE)
    native = _stage_reusable(ledger, "native-render")
    if native is None:
        native, status = _execute_training_or_render(
            ledger=ledger,
            stage="native-render",
            profile_input=formal_profile_input,
            plan=False,
            execute_gpu=execute_gpu,
            route=route,
            training_evidence_root=Path(str(formal["executor_root"])),
        )
        results["native-render"] = native
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=native["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved native render attempt")
    else:
        results["native-render"] = {**native, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["native-render"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after native-render")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at authority-manifest")
    if native.get("quality", {}).get("quality_status") != "pass":
        _finish_summary(ledger, results, status="blocked", reason="native render quality gate did not pass")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="manual review is required before conversion")

    authority_result = _stage_reusable(ledger, "authority-manifest")
    if authority_result is None:
        authority_result, status = _authority_stage(
            ledger=ledger,
            route=route,
            source=source,
            formal=formal_profile_input,
            native=native,
            supplied_manifest=authority_manifest,
            plan=False,
            tool_provider=tool_provider,
        )
        results["authority-manifest"] = authority_result
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=authority_result["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the authority manifest validation")
    else:
        results["authority-manifest"] = {**authority_result, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["authority-manifest"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after authority-manifest")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at conversion")

    conversion = _stage_reusable(ledger, "conversion")
    if conversion is None:
        conversion, status = _execute_authority_stage(ledger=ledger, stage="conversion", authority_result=authority_result, plan=False, execute_gpu=execute_gpu, route=route)
        results["conversion"] = conversion
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=conversion["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved conversion attempt")
    else:
        results["conversion"] = {**conversion, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["conversion"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after conversion")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at converted-eval")

    evaluation = _stage_reusable(ledger, "converted-eval")
    if evaluation is None:
        evaluation, status = _execute_authority_stage(ledger=ledger, stage="converted-eval", authority_result=authority_result, plan=False, execute_gpu=execute_gpu, route=route, native_evidence_root=Path(str(native["executor_root"])), conversion_evidence_root=Path(str(conversion["executor_root"])))
        results["converted-eval"] = evaluation
        if status != "passed":
            if not _evaluation_reused_render_available(evaluation, int(authority_result.get("camera_count", 0))):
                _finish_summary(ledger, results, status="blocked", reason=evaluation["reason"])
                return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved converted-eval attempt")
            evaluation = {
                **evaluation,
                "render_artifacts_available": True,
                "render_reused": True,
                "cuda_rerun": False,
                "evaluation_gate": "deferred_to_cpu_streaming_postprocess",
            }
            results["converted-eval"] = evaluation
    else:
        results["converted-eval"] = {**evaluation, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["converted-eval"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after converted-eval")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at converted-eval-postprocess")

    postprocess = _stage_reusable(ledger, "converted-eval-postprocess")
    if postprocess is None:
        if evaluation.get("render_artifacts_available") is True:
            postprocess, status = _converted_eval_postprocess_stage(
                ledger=ledger,
                route=route,
                authority=authority_result,
                conversion=conversion,
                evaluation=evaluation,
                plan=False,
            )
        else:
            postprocess = _stage_result(
                stage="converted-eval-postprocess",
                status="passed",
                computed_pass=True,
                reason="GPU evaluator completed its own same-camera metrics; no recovery was needed",
                plan=False,
                gpu_invoked=False,
                render_reused=False,
                cuda_rerun=False,
                postprocess_required=False,
            )
            _record_stage(ledger, "converted-eval-postprocess", postprocess, status="passed")
            status = "passed"
        results["converted-eval-postprocess"] = postprocess
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=postprocess["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved CPU postprocess attempt")
    else:
        results["converted-eval-postprocess"] = {**postprocess, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["converted-eval-postprocess"]:
        _finish_summary(ledger, results, status="stopped", reason="requested stop-after converted-eval-postprocess")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="resume at candidate-delivery")

    candidate = _stage_reusable(ledger, "candidate-delivery")
    if candidate is None:
        candidate, status = _candidate_delivery(
            ledger=ledger,
            route=route,
            authority=authority_result,
            conversion=conversion,
            evaluation=evaluation,
            postprocess=postprocess,
            conversion_visual_token=conversion_visual_token,
            plan=False,
        )
        results["candidate-delivery"] = candidate
        if status != "passed":
            _finish_summary(ledger, results, status="blocked", reason=candidate["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="inspect the preserved candidate-delivery attempt")
    else:
        results["candidate-delivery"] = {**candidate, "reused": True}
    if STAGE_INDEX[stop_after] == STAGE_INDEX["candidate-delivery"]:
        _finish_summary(ledger, results, status="stopped", reason="candidate PLY is ready for SuperSplat manual review")
        return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="open candidate in SuperSplat and supply an explicit three-view acceptance token")

    accepted = _stage_reusable(ledger, "accepted-delivery")
    if accepted is None:
        if acceptance_token is None:
            accepted = _stage_result(stage="accepted-delivery", status="blocked", computed_pass=False, reason="explicit manual SuperSplat acceptance token is required; automatic visual acceptance is forbidden", plan=False, accepted=False, supersplat=False)
            _record_stage(ledger, "accepted-delivery", accepted, status="blocked")
            results["accepted-delivery"] = accepted
            _finish_summary(ledger, results, status="blocked", reason=accepted["reason"])
            return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="manual SuperSplat three-view acceptance")
        try:
            token = _manual_acceptance_valid(acceptance_token)
            accepted = _stage_result(
                stage="accepted-delivery",
                status="passed",
                computed_pass=True,
                reason="explicit SuperSplat acceptance token imported; screenshot archive state remains separate",
                plan=False,
                artifacts=[token],
                accepted=True,
                supersplat=True,
                acceptance_token=token,
                screenshot_file_evidence=token.get("screenshot_file_evidence"),
                screenshot_evidence_complete=token.get("screenshot_evidence_complete"),
            )
            _record_stage(ledger, "accepted-delivery", accepted, status="passed")
        except PipelineBlocked as exc:
            accepted = _stage_result(stage="accepted-delivery", status="blocked", computed_pass=False, reason=str(exc), plan=False, accepted=False, supersplat=False)
            _record_stage(ledger, "accepted-delivery", accepted, status="blocked")
    else:
        accepted = {**accepted, "reused": True}
    results["accepted-delivery"] = accepted
    _finish_summary(ledger, results, status="stopped" if accepted.get("computed_pass") else "blocked", reason=accepted.get("reason", "acceptance imported"))
    return _run_result(ledger, results, stop_after=stop_after, plan=False, next_slice="completed" if accepted.get("computed_pass") else "manual SuperSplat acceptance")


def _execute_authority_stage(
    *,
    ledger: RunLedger,
    stage: str,
    authority_result: Mapping[str, Any],
    plan: bool,
    execute_gpu: bool,
    route: Path,
    native_evidence_root: Path | None = None,
    conversion_evidence_root: Path | None = None,
) -> tuple[dict[str, Any], str]:
    if not execute_gpu:
        result = _stage_result(stage=stage, status="requires_escalated_gpu_execution", computed_pass=False, reason="requires_escalated_gpu_execution", plan=plan, gpu_invoked=False, authority_manifest_path=authority_result.get("authority_manifest_path"))
        return result, "blocked"
    attempt = ledger.begin_attempt(stage, {"authority_manifest_path": authority_result.get("authority_manifest_path"), "native_evidence_root": None if native_evidence_root is None else str(native_evidence_root), "conversion_evidence_root": None if conversion_evidence_root is None else str(conversion_evidence_root)})
    executor_root = attempt / "executor"
    # The conversion stage creates its own (fresh, append-only) executor root
    # and therefore must find it absent; every other authority stage
    # (currently converted-eval) is handed its executor root as evidence_root
    # and expects it to already exist, so create it here.
    if stage != "conversion":
        executor_root.mkdir(parents=True, exist_ok=True)
    try:
        from .smoke_executor import SmokeExecutorBlocked, execute_stage

        ledger.mark_gpu_invoked()
        child = execute_stage(
            plan_path=authority_result.get("plan_path", authority_result["authority"]["manifest"]["plan"]["path"]),
            static_contract_path=authority_result.get("static_contract_path", authority_result["authority"]["manifest"]["static_contract"]["path"]),
            evidence_root=executor_root,
            stage=stage,
            route_root=route,
            containment_root=ledger.run_dir,
            authority_manifest_path=authority_result["authority_manifest_path"],
            native_evidence_root=native_evidence_root,
            conversion_evidence_root=conversion_evidence_root,
        )
    except Exception as exc:
        from .smoke_executor import SmokeExecutorBlocked

        if not isinstance(exc, (SmokeExecutorBlocked, OSError, ValueError)):
            raise
        result = _stage_result(stage=stage, status="blocked", computed_pass=False, reason=str(exc), plan=plan, gpu_invoked=True, executor_root=str(executor_root), authority_manifest_path=authority_result["authority_manifest_path"])
        ledger.finish_attempt(stage=stage, attempt=attempt, status="blocked", result=result)
        return result, "blocked"
    passed = (child.get("stage") == "conversion" and child.get("structural_pass") is True and child.get("STRUCTURAL_CONVERSION_PASS") is True) if stage == "conversion" else child.get("SAME_CAMERA_VISUAL_PASS") in {"pass", "needs_review"}
    child_status = "passed" if passed else ("failed" if child.get("exit_code") not in (0, None) else "blocked")
    result = _stage_result(stage=stage, status=child_status, computed_pass=passed, reason="authority executor passed" if passed else str(child.get("reason", "authority executor failed")), plan=plan, artifacts=_artifacts([executor_root / "conversion_result.json", executor_root / "evaluation_result.json", executor_root / "request.json", executor_root / "argv.json", executor_root / "same_camera_eval" / "evaluator_result.json", authority_result["authority_manifest_path"]]), gpu_invoked=True, executor_root=str(executor_root), executor_result=child, authority_manifest_path=authority_result["authority_manifest_path"], plan_path=authority_result["authority"]["manifest"]["plan"]["path"], static_contract_path=authority_result["authority"]["manifest"]["static_contract"]["path"])
    ledger.finish_attempt(stage=stage, attempt=attempt, status=child_status, result=result)
    return result, child_status


def _finish_summary(ledger: RunLedger, results: Mapping[str, Any], *, status: str, reason: str) -> None:
    if status == "blocked":
        last_stage = next(reversed(results), None) if results else ledger.summary.get("active_stage")
        ledger.mark_blocked(stage=last_stage, error=reason, exit_code=2)
        return
    last = next(reversed(results.values()), {}) if results else {}
    ledger.summary["status"] = status
    # A resumed run may carry a historical blocked record from the prior
    # visual gate.  Once this invocation reaches an explicit CPU stop, that
    # stale terminal marker must not contradict the authoritative status.
    ledger.summary.pop("blocked", None)
    ledger.summary.pop("active_stage", None)
    ledger.summary.pop("active_attempt", None)
    ledger.summary.pop("current_intended_stage", None)
    ledger.summary["computed_pass"] = bool(last.get("computed_pass")) if isinstance(last, Mapping) else False
    ledger.summary["accepted"] = bool(last.get("accepted")) if isinstance(last, Mapping) else False
    delivery_stages = {"candidate-delivery", "automated-technical-delivery"}
    delivery_result = next(
        (
            value
            for value in reversed(list(results.values()))
            if isinstance(value, Mapping) and value.get("stage") in delivery_stages
        ),
        {},
    )
    ledger.summary["delivery_reachable"] = bool(delivery_result.get("computed_pass")) if isinstance(delivery_result, Mapping) else False
    ledger.summary["accepted_by_automated_policy"] = bool(delivery_result.get("accepted_by_automated_policy")) if isinstance(delivery_result, Mapping) else False
    ledger.summary["manual_visual_review"] = bool(delivery_result.get("manual_visual_review")) if isinstance(delivery_result, Mapping) else False
    ledger.summary["supersplat_runtime_verified"] = bool(delivery_result.get("supersplat_runtime_verified")) if isinstance(delivery_result, Mapping) else False
    ledger.summary["technical_delivery_root"] = delivery_result.get("technical_delivery_root") if isinstance(delivery_result, Mapping) else None
    ledger.summary["published_ply"] = delivery_result.get("published_ply") if isinstance(delivery_result, Mapping) else None
    ledger.summary["published_ply_receipt"] = delivery_result.get("published_ply_receipt") if isinstance(delivery_result, Mapping) else None
    ledger.summary["acceptance"] = {
        "accepted": ledger.summary["accepted"],
        "reason": reason,
        "final_delivery_stage": next(reversed(results)) if results else None,
    }
    ledger._write_summary()


def _mark_existing_run_blocked(
    *,
    output_root: str | Path,
    route_root: str | Path | None,
    run_id: str,
    error: str,
    exit_code: int = 2,
) -> None:
    """Make an already-created run authoritative after an outer exception."""

    try:
        route = Path(route_root or Path(__file__).resolve().parents[2]).resolve()
        run_dir = _route_output_root(output_root, route) / _safe_run_id(run_id)
        summary_path = run_dir / "run.json"
        if not summary_path.is_file() or summary_path.is_symlink():
            return
        summary = _load_json(summary_path, "reconstruction run summary")
        # Prefer the current invocation intent.  ``create_or_resume`` moves
        # an unclosed prior active stage into orphan_inventory, so it must not
        # be used as the stage for this invocation's exception.
        stage = summary.get("current_intended_stage")
        if not isinstance(stage, str) or not stage:
            stage = summary.get("active_stage")
        attempt = summary.get("active_attempt")
        if not isinstance(stage, str) or not stage:
            stages = summary.get("stages")
            stage = next(reversed(stages), None) if isinstance(stages, Mapping) and stages else None
        summary["status"] = "blocked"
        summary["computed_pass"] = False
        summary["accepted"] = False
        summary["delivery_reachable"] = False
        summary["blocked"] = {
            "stage": stage,
            "error": str(error),
            "exit_code": int(exit_code),
        }
        if isinstance(stage, str) and isinstance(attempt, str):
            summary["orphan_evidence"] = {
                "stage": stage,
                "attempt": attempt,
                "status": "not_committed_to_ledger",
                "reason": str(error),
            }
        summary["last_stage"] = stage
        summary.pop("active_stage", None)
        summary.pop("active_attempt", None)
        summary.pop("current_intended_stage", None)
        summary["acceptance"] = {
            "accepted": False,
            "reason": str(error),
            "final_delivery_stage": stage,
        }
        write_json(summary_path, summary)
    except (OSError, PipelineBlocked, ResumeMismatchError, ValueError, json.JSONDecodeError):
        # The original exception remains the authoritative CLI error when a
        # summary cannot be updated (for example, before a run is created).
        return


def _run_result(ledger: RunLedger, results: Mapping[str, Any], *, stop_after: str, plan: bool, next_slice: str) -> dict[str, Any]:
    return {
        "schema_version": "longsplat-reconstruct-run-v2",
        "run_id": ledger.summary["run_id"],
        "run_dir": str(ledger.run_dir),
        "status": ledger.summary["status"],
        "computed_pass": ledger.summary["computed_pass"],
        "delivery_reachable": ledger.summary["delivery_reachable"],
        "accepted_by_automated_policy": ledger.summary.get("accepted_by_automated_policy", False),
        "manual_visual_review": ledger.summary.get("manual_visual_review", False),
        "supersplat_runtime_verified": ledger.summary.get("supersplat_runtime_verified", False),
        "technical_delivery_root": ledger.summary.get("technical_delivery_root"),
        "published_ply": ledger.summary.get("published_ply"),
        "published_ply_receipt": ledger.summary.get("published_ply_receipt"),
        "gpu_invoked": ledger.summary["gpu_invoked"],
        "stop_after": stop_after,
        "plan": plan,
        "stage_results": dict(results),
        "next_slice": next_slice,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run/resume the generic LongSplat one-click reconstruction chain")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stop-after", choices=STAGES, default="automated-technical-delivery")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute-gpu", action="store_true", help="execute GPU stages; use only from a require_escalated WSL command")
    parser.add_argument("--retry-failed-stage", choices=("smoke100-training",), help="one explicitly authorized append-only executor retry")
    parser.add_argument("--depth-source", choices=("disabled",), default="disabled")
    parser.add_argument("--route-root")
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--matching", choices=("sequential", "exhaustive"), default="sequential")
    parser.add_argument("--backend-env", help="explicit backend environment root containing bin/python")
    parser.add_argument("--backend-python", help="explicit backend Python executable")
    parser.add_argument("--ffmpeg", help="explicit ffmpeg executable")
    parser.add_argument("--ffprobe", help="explicit ffprobe executable")
    parser.add_argument("--colmap", help="explicit COLMAP executable")
    parser.add_argument("--route-python", help="explicit route Python executable")
    parser.add_argument("--publish-dir", help="optional separate public PLY directory; legacy direct calls omit it")
    parser.add_argument("--delivery-name", help="safe public PLY stem used with --publish-dir")
    parser.add_argument("--authority-manifest")
    parser.add_argument("--acceptance-token")
    parser.add_argument("--coverage-visual-token", help="explicit human coverage-smoke visual decision JSON; never inferred from structural metrics")
    parser.add_argument("--convergence-visual-token", help="explicit human convergence1000 diagnostic visual decision JSON; never releases formal training")
    parser.add_argument("--conversion-visual-token", help="explicit technical converted-view visual decision JSON; does not accept SuperSplat")
    parser.add_argument(
        "--acceptance-policy",
        choices=(AUTOMATED_POLICY_ID, "manual"),
        default=AUTOMATED_POLICY_ID,
        help="versioned technical delivery policy; automated policy does not claim SuperSplat runtime verification",
    )
    parser.add_argument("--diagnostic-profile", choices=("convergence1000-v1",), help="explicit one-shot diagnostic opt-in; not part of the default chain")
    parser.add_argument(
        "--pipeline-profile",
        choices=(DEFAULT_PIPELINE_PROFILE, LEGACY_PIPELINE_PROFILE),
        help="versioned stage policy; new canonical runs default to single-convergence1000-v1",
    )
    parser.add_argument("--validate-only", action="store_true", help="validate existing run evidence and CPU postprocess without executing any algorithm or GPU stage")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_reconstruction(
            input_video=args.input_video,
            output_root=args.output_root,
            run_id=args.run_id,
            stop_after=args.stop_after,
            plan=args.plan,
            route_root=args.route_root,
            authority_manifest=args.authority_manifest,
            acceptance_token=args.acceptance_token,
            depth_source=args.depth_source,
            execute_gpu=args.execute_gpu,
            retry_failed_stage=args.retry_failed_stage,
            validate_only=args.validate_only,
            coverage_visual_token=args.coverage_visual_token,
            convergence_visual_token=args.convergence_visual_token,
            conversion_visual_token=args.conversion_visual_token,
            diagnostic_profile=args.diagnostic_profile,
            pipeline_profile=args.pipeline_profile,
            acceptance_policy=args.acceptance_policy,
            camera_model=args.camera_model,
            matching=args.matching,
            publish_dir=args.publish_dir,
            delivery_name=args.delivery_name,
            tool_paths={
                key: value
                for key, value in {
                    "backend_env": args.backend_env,
                    "backend_python": args.backend_python,
                    "ffmpeg": args.ffmpeg,
                    "ffprobe": args.ffprobe,
                    "colmap": args.colmap,
                    "route_python": args.route_python,
                }.items()
                if value is not None
            },
        )
    except ResumeMismatchError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:
        # ``--validate-only`` is explicitly read-only.  A validation error
        # must not rewrite a historical run's terminal stage/reason while
        # reporting the validation failure.
        if not args.validate_only:
            _mark_existing_run_blocked(
                output_root=args.output_root,
                route_root=args.route_root,
                run_id=args.run_id,
                error=str(exc),
                exit_code=2,
            )
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"planned", "stopped"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
