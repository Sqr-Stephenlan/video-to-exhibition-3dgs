from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .contracts import (
    ArtifactDescriptor,
    BackendIdentity,
    CurrentStage,
    ErrorDetail,
    InputDescriptor,
    JobSnapshot,
    ObservedProgress,
    QualityFlags,
    RouteContract,
    StageSnapshot,
)


_TRAINING_STAGES = {"convergence-smoke-training", "formal-training"}


def normalize_run_status(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "planned", "pending", "queued"}:
        return "queued"
    if normalized in {"running", "started"}:
        return "running"
    if normalized in {"complete", "completed", "passed", "accepted"}:
        return "complete"
    if normalized in {"blocked"}:
        return "blocked"
    if normalized in {"failed", "error", "errored"}:
        return "failed"
    if normalized in {"stopped", "cancelled", "canceled"}:
        return "stopped"
    return "failed"


def _normalize_stage_status(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"passed", "complete", "completed", "accepted"}:
        return "passed"
    if normalized in {"running", "started"}:
        return "running"
    if normalized in {"blocked"}:
        return "blocked"
    if normalized in {"failed", "error", "errored"}:
        return "failed"
    if normalized in {"stopped", "cancelled", "canceled"}:
        return "stopped"
    if normalized in {"skipped"}:
        return "skipped"
    return "queued"


def _timestamp_from_epoch(value: object) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat().replace("+00:00", "Z")


def _latest_stage_record(run_record: Mapping[str, Any], stage: str) -> Mapping[str, Any] | None:
    stages = run_record.get("stages")
    if not isinstance(stages, Mapping):
        return None
    attempts = stages.get(stage)
    if not isinstance(attempts, list):
        return None
    for record in reversed(attempts):
        if isinstance(record, Mapping):
            return record
    return None


def _safe_stage_summary(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        return {}
    result = record.get("result")
    if not isinstance(result, Mapping):
        return {}
    allowed = {
        "schema_version",
        "reason",
        "exit_code",
        "computed_pass",
        "accepted",
        "delivery_reachable",
        "accepted_by_automated_policy",
        "supersplat_format_compatible",
        "supersplat_runtime_verified",
        "structural_evaluation_pass",
        "visual_quality_pass",
    }
    return {
        key: value
        for key, value in result.items()
        if key in allowed and isinstance(value, (str, int, float, bool, type(None)))
    }


def _latest_training_stage(
    run_record: Mapping[str, Any],
    *,
    stage_order: Sequence[str],
) -> str | None:
    for candidate in reversed(tuple(stage_order)):
        if candidate not in _TRAINING_STAGES:
            continue
        record = _latest_stage_record(run_record, candidate)
        if not isinstance(record, Mapping):
            continue
        status = str(record.get("status") or "").strip().lower()
        if status in {"running", "passed", "complete", "completed", "accepted"}:
            return candidate
    return None


def _latest_stage_attempt(run_record: Mapping[str, Any], stage: str | None) -> str | None:
    if not isinstance(stage, str):
        return None
    record = _latest_stage_record(run_record, stage)
    attempt = record.get("attempt") if isinstance(record, Mapping) else None
    return attempt if isinstance(attempt, str) else None


def _progress_origin(
    *,
    kind: str,
    progress: Any,
    run_record: Mapping[str, Any],
    stage_order: Sequence[str],
    active_stage: str | None,
    active_attempt: str | None,
) -> tuple[str | None, str | None]:
    raw_stage = getattr(progress, "stage", None)
    raw_attempt = getattr(progress, "stage_attempt", None)
    origin_stage: str | None = None
    if kind == "conversion-iteration":
        origin_stage = "conversion"
    elif isinstance(active_stage, str) and active_stage in _TRAINING_STAGES:
        origin_stage = active_stage
    elif isinstance(raw_stage, str) and raw_stage in _TRAINING_STAGES:
        origin_stage = raw_stage
    else:
        origin_stage = _latest_training_stage(run_record, stage_order=stage_order)

    if origin_stage is None:
        return None, None
    if origin_stage == active_stage and isinstance(active_attempt, str):
        return origin_stage, active_attempt
    if isinstance(raw_stage, str) and raw_stage == origin_stage and isinstance(raw_attempt, str):
        return origin_stage, raw_attempt
    return origin_stage, _latest_stage_attempt(run_record, origin_stage)


def _observed_progress(
    progress: Any,
    *,
    run_record: Mapping[str, Any] | None = None,
    stage_order: Sequence[str] = (),
    active_stage: str | None = None,
    active_attempt: str | None = None,
) -> ObservedProgress | None:
    if progress is None:
        return None
    record = run_record if isinstance(run_record, Mapping) else {}
    candidates = (
        ("conversion-iteration", getattr(progress, "conversion", None)),
        ("training-iteration", getattr(progress, "training", None)),
    )
    for kind, value in candidates:
        iteration = getattr(value, "iteration", None)
        total = getattr(value, "total", None)
        timestamp = getattr(value, "marker_timestamp", None)
        if isinstance(iteration, int) and iteration > 0 and isinstance(total, int) and total > 0:
            if iteration <= total:
                observed_at = _timestamp_from_epoch(timestamp) or ""
                if observed_at:
                    origin_stage, origin_attempt = _progress_origin(
                        kind=kind,
                        progress=progress,
                        run_record=record,
                        stage_order=stage_order,
                        active_stage=active_stage,
                        active_attempt=active_attempt,
                    )
                    return ObservedProgress(
                        kind=kind,
                        iteration=iteration,
                        total=total,
                        observed_at=observed_at,
                        stage=origin_stage,
                        attempt=origin_attempt,
                    )
    return None


def _quality_flags(
    run_record: Mapping[str, Any],
    quality_record: Mapping[str, Any] | None,
) -> QualityFlags:
    merged: dict[str, Any] = {}
    for source in (run_record, quality_record or {}):
        if isinstance(source, Mapping):
            merged.update(source)
    acceptance = merged.get("acceptance")
    if isinstance(acceptance, Mapping):
        merged.update(acceptance)
    return QualityFlags(
        accepted_by_automated_policy=bool(merged.get("accepted_by_automated_policy", False)),
        gaussian_schema_valid=bool(merged.get("gaussian_schema_valid", False)),
        supersplat_format_compatible=bool(merged.get("supersplat_format_compatible", False)),
        supersplat_runtime_verified=bool(merged.get("supersplat_runtime_verified", False)),
        manual_visual_review=bool(merged.get("manual_visual_review", False)),
        structural_evaluation_pass=(
            merged.get("structural_evaluation_pass")
            if isinstance(merged.get("structural_evaluation_pass"), bool)
            else None
        ),
        visual_quality_pass=(
            merged.get("visual_quality_pass")
            if isinstance(merged.get("visual_quality_pass"), bool)
            else None
        ),
    )


def _next_action(status: str, current_stage: CurrentStage | None) -> str:
    if status == "queued":
        return "等待任务启动"
    if status == "running":
        return f"等待阶段 {current_stage.id} 完成" if current_stage else "等待流水线状态更新"
    if status == "complete":
        return "下载模型或在 SuperSplat 中查看"
    if status in {"blocked", "failed"}:
        return "查看失败阶段证据并重新提交任务"
    return "任务已停止"


def build_job_snapshot(
    *,
    job_record: Mapping[str, Any],
    config_record: Mapping[str, Any],
    run_record: Mapping[str, Any],
    progress: Any = None,
    artifacts: Sequence[ArtifactDescriptor | Mapping[str, Any]] = (),
    quality_record: Mapping[str, Any] | None = None,
) -> JobSnapshot:
    stage_order_value = config_record.get("stage_order")
    stage_order = [
        value for value in stage_order_value
        if isinstance(value, str) and value
    ] if isinstance(stage_order_value, list) else []

    raw_status = run_record.get("status", job_record.get("status"))
    status = normalize_run_status(raw_status)
    active_stage = run_record.get("active_stage")
    progress_stage = getattr(progress, "stage", None)
    progress_is_active = getattr(progress, "stage_active", False) is True
    progress_stage_status = str(getattr(progress, "stage_status", "")).strip().lower()
    if not isinstance(active_stage, str) and progress_is_active:
        active_stage = getattr(progress, "stage", None)
    if (
        status == "queued"
        and progress_is_active
        and isinstance(progress_stage, str)
        and progress_stage_status in {"running", "started"}
    ):
        # The worker writes the progress snapshot before the canonical run
        # ledger advances from its planned marker.  Expose the observed worker
        # state instead of making the browser report a false queued status.
        status = "running"
    if not isinstance(active_stage, str) and status in {"blocked", "failed", "stopped"}:
        marker = run_record.get(status)
        marker_stage = marker.get("stage") if isinstance(marker, Mapping) else None
        candidate = marker_stage if isinstance(marker_stage, str) else run_record.get("last_stage")
        active_stage = candidate if isinstance(candidate, str) else None
    active_attempt = run_record.get("active_attempt")
    if not isinstance(active_attempt, str) and progress_is_active:
        active_attempt = getattr(progress, "stage_attempt", None)
    if not isinstance(active_attempt, str) and isinstance(active_stage, str):
        latest = _latest_stage_record(run_record, active_stage)
        candidate = latest.get("attempt") if isinstance(latest, Mapping) else None
        active_attempt = candidate if isinstance(candidate, str) else None
    active_status = _normalize_stage_status(
        run_record.get("stage_status", getattr(progress, "stage_status", "running"))
    )
    if isinstance(active_stage, str):
        latest = _latest_stage_record(run_record, active_stage)
        if isinstance(latest, Mapping) and isinstance(latest.get("status"), str):
            active_status = _normalize_stage_status(latest.get("status"))
    if status in {"blocked", "failed", "stopped"} and active_status == "queued":
        active_status = status
    current_index = stage_order.index(active_stage) + 1 if active_stage in stage_order else None

    stage_snapshots: list[StageSnapshot] = []
    for index, stage in enumerate(stage_order, start=1):
        record = _latest_stage_record(run_record, stage)
        stage_status = _normalize_stage_status(record.get("status") if record else None)
        if stage == active_stage and status == "running":
            stage_status = "running"
        elif stage == active_stage and status in {"blocked", "failed", "stopped"}:
            stage_status = active_status
        attempt = record.get("attempt") if isinstance(record, Mapping) else None
        if stage == active_stage and isinstance(active_attempt, str):
            attempt = active_attempt
        stage_snapshots.append(
            StageSnapshot(
                id=stage,
                index=index,
                status=stage_status,
                attempt=attempt if isinstance(attempt, str) else None,
                summary=_safe_stage_summary(record),
            )
        )

    current: CurrentStage | None = None
    if (
        isinstance(active_stage, str)
        and current_index is not None
        and status in {"running", "blocked", "failed", "stopped"}
    ):
        current = CurrentStage(
            id=active_stage,
            index=current_index,
            attempt=active_attempt if isinstance(active_attempt, str) else None,
            status=active_status,
            started_at=_timestamp_from_epoch(getattr(progress, "stage_started", None)),
        )

    input_value = job_record.get("input")
    input_descriptor = InputDescriptor.model_validate(input_value) if isinstance(input_value, Mapping) else None
    artifact_models = [
        item if isinstance(item, ArtifactDescriptor) else ArtifactDescriptor.model_validate(item)
        for item in artifacts
    ]
    quality_flags = _quality_flags(run_record, quality_record)
    error_value = run_record.get("blocked")
    error = None
    if isinstance(error_value, Mapping):
        error = ErrorDetail(
            code=error_value.get("code") if isinstance(error_value.get("code"), str) else None,
            stage=error_value.get("stage") if isinstance(error_value.get("stage"), str) else None,
            message=str(error_value.get("error") or "流水线被阻塞"),
            exit_code=error_value.get("exit_code") if isinstance(error_value.get("exit_code"), int) else None,
        )
    elif isinstance(job_record.get("error"), Mapping):
        job_error = job_record["error"]
        error = ErrorDetail(
            code=job_error.get("code") if isinstance(job_error.get("code"), str) else None,
            stage=job_error.get("stage") if isinstance(job_error.get("stage"), str) else None,
            message=str(job_error.get("message") or "Web worker 未完成任务"),
            exit_code=job_error.get("exit_code") if isinstance(job_error.get("exit_code"), int) else None,
        )
    elif status == "failed":
        raw_text = str(raw_status or "")
        error = ErrorDetail(
            code="unknown_run_status" if raw_text.lower() not in {"failed", "error", "errored"} else None,
            stage=active_stage if isinstance(active_stage, str) else None,
            message=("流水线返回未知状态: " + raw_text) if raw_text else "流水线执行失败",
        )

    if status == "complete":
        artifact_ids = {artifact.id for artifact in artifact_models}
        delivery_is_complete = (
            "published-ply" in artifact_ids
            and "published-ply-receipt" in artifact_ids
            and quality_flags.accepted_by_automated_policy
            and quality_flags.gaussian_schema_valid
            and quality_flags.supersplat_format_compatible
        )
        if not delivery_is_complete:
            status = "failed"
            error = ErrorDetail(
                code="technical_delivery_incomplete",
                message="技术交付缺少有效的 published PLY、receipt、Gaussian 属性或自动技术门禁证据",
            )

    created_at = job_record.get("created_at")
    if not isinstance(created_at, str):
        created_at = str(run_record.get("created_at") or "")
    updated_at = run_record.get("updated_at")
    if not isinstance(updated_at, str):
        updated_at = str(job_record.get("updated_at") or created_at)

    route_value = job_record.get("route")
    if not isinstance(route_value, Mapping):
        route_value = {
            "pipeline_profile": config_record.get("pipeline_profile", "single-convergence1000-v1"),
            "depth_source": config_record.get("depth_source", "disabled"),
            "pose_mode": config_record.get("pose_mode", "external-fixed-pose-rgb-only"),
            "acceptance_policy": config_record.get("acceptance_policy", "automated-technical-v1"),
        }
    try:
        route = RouteContract.model_validate(route_value)
    except Exception:
        route = RouteContract()
    backend_value = job_record.get("backend")
    backend = BackendIdentity.model_validate(backend_value) if isinstance(backend_value, Mapping) else None

    return JobSnapshot(
        schema_version="web-job-snapshot-v1",
        job_id=str(job_record.get("job_id") or run_record.get("run_id") or ""),
        run_id=str(job_record.get("run_id") or run_record.get("run_id") or ""),
        status=status,
        created_at=created_at,
        updated_at=updated_at,
        route=route,
        backend=backend,
        input=input_descriptor,
        stage_order=stage_order,
        current_stage=current,
        stages=stage_snapshots,
        observed_progress=_observed_progress(
            progress,
            run_record=run_record,
            stage_order=stage_order,
            active_stage=active_stage,
            active_attempt=active_attempt,
        ),
        artifacts=artifact_models,
        quality=quality_flags,
        error=error,
        next_action=_next_action(status, current),
    )
