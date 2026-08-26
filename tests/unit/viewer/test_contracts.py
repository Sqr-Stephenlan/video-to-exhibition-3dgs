from __future__ import annotations

import pytest
from pydantic import ValidationError

from scripts.viewer.api.contracts import (
    ArtifactDescriptor,
    BackendIdentity,
    JobSnapshot,
    ObservedProgress,
    QualityFlags,
    RouteContract,
)


def _snapshot_payload() -> dict:
    return {
        "schema_version": "web-job-snapshot-v1",
        "job_id": "job-1",
        "run_id": "job-1",
        "status": "running",
        "created_at": "2026-08-24T00:00:00Z",
        "updated_at": "2026-08-24T00:01:00Z",
        "stage_order": ["probe", "conversion"],
        "current_stage": {
            "id": "conversion",
            "index": 2,
            "attempt": "attempt-0001",
            "status": "running",
            "started_at": "2026-08-24T00:00:30Z",
        },
        "stages": [],
        "observed_progress": None,
        "artifacts": [],
        "quality": QualityFlags(
            accepted_by_automated_policy=False,
            supersplat_format_compatible=False,
            supersplat_runtime_verified=False,
            manual_visual_review=False,
            structural_evaluation_pass=None,
            visual_quality_pass=None,
        ).model_dump(),
        "error": None,
        "next_action": "等待阶段完成",
    }


def test_job_snapshot_rejects_uncontracted_overall_percent():
    payload = _snapshot_payload()
    payload["percent"] = 50

    with pytest.raises(ValidationError):
        JobSnapshot.model_validate(payload)


def test_artifact_descriptor_exposes_route_url_not_a_local_path():
    artifact = ArtifactDescriptor(
        id="published-ply",
        kind="model",
        format="ply",
        download_url="/api/v1/jobs/job-1/artifacts/published-ply",
        sha256="a" * 64,
        size_bytes=12,
        vertices=1,
    )

    assert artifact.download_url.startswith("/api/")
    assert not artifact.download_url.startswith("C:")


def test_observed_progress_requires_a_real_positive_iteration():
    with pytest.raises(ValidationError):
        ObservedProgress(
            kind="conversion-iteration",
            iteration=0,
            total=30,
            observed_at="2026-08-24T00:01:00Z",
        )

    with pytest.raises(ValidationError):
        ObservedProgress(
            kind="conversion-iteration",
            iteration=31,
            total=30,
            observed_at="2026-08-24T00:01:00Z",
        )


def test_observed_progress_can_record_its_origin_stage_and_attempt():
    progress = ObservedProgress(
        kind="training-iteration",
        iteration=12,
        total=30,
        observed_at="2026-08-24T00:01:00Z",
        stage="formal-training",
        attempt="attempt-0002",
    )

    assert progress.stage == "formal-training"
    assert progress.attempt == "attempt-0002"


def test_job_snapshot_requires_fixed_route_and_backend_identity():
    payload = _snapshot_payload()
    payload["route"] = RouteContract(
        pipeline_profile="single-convergence1000-v1",
        depth_source="disabled",
        pose_mode="external-fixed-pose-rgb-only",
        acceptance_policy="automated-technical-v1",
    ).model_dump()
    payload["backend"] = BackendIdentity(
        superproject_gitlink="c" * 40,
        longsplat_commit="c" * 40,
        provider_identity="provider-sha256",
    ).model_dump()

    snapshot = JobSnapshot.model_validate(payload)

    assert snapshot.route.pipeline_profile == "single-convergence1000-v1"
    assert snapshot.route.depth_source == "disabled"
    assert snapshot.backend is not None
    assert snapshot.backend.longsplat_commit == "c" * 40
