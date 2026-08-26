from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


JobStatus = Literal["queued", "running", "complete", "blocked", "failed", "stopped"]
StageStatus = Literal["queued", "running", "passed", "blocked", "failed", "stopped", "skipped"]
ProgressKind = Literal["conversion-iteration", "training-iteration"]


class RouteContract(StrictModel):
    pipeline_profile: Literal["single-convergence1000-v1"] = "single-convergence1000-v1"
    depth_source: Literal["disabled"] = "disabled"
    pose_mode: Literal["external-fixed-pose-rgb-only"] = "external-fixed-pose-rgb-only"
    acceptance_policy: Literal["automated-technical-v1"] = "automated-technical-v1"


class BackendIdentity(StrictModel):
    superproject_gitlink: str = Field(pattern=r"^[0-9a-f]{40}$")
    longsplat_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    provider_identity: str = Field(min_length=1, max_length=200)


class InputDescriptor(StrictModel):
    name: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class ObservedProgress(StrictModel):
    kind: ProgressKind
    iteration: PositiveInt
    total: PositiveInt
    observed_at: str
    stage: str | None = Field(default=None, min_length=1, max_length=120)
    attempt: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def iteration_is_within_total(self) -> "ObservedProgress":
        if self.iteration > self.total:
            raise ValueError("observed iteration cannot exceed total")
        return self


class ArtifactDescriptor(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    kind: str = Field(min_length=1, max_length=40)
    format: str = Field(min_length=1, max_length=40)
    download_url: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    size_bytes: int = Field(ge=1)
    vertices: int | None = Field(default=None, ge=1)

    @field_validator("download_url")
    @classmethod
    def is_internal_artifact_route(cls, value: str) -> str:
        if not value.startswith("/api/v1/jobs/") or "://" in value:
            raise ValueError("artifact URLs must use the internal API route")
        if "\\" in value or ".." in value:
            raise ValueError("artifact URL contains an unsafe path")
        return value


class StageSnapshot(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    index: int = Field(ge=1)
    status: StageStatus
    attempt: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class CurrentStage(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    index: int = Field(ge=1)
    attempt: str | None = None
    status: StageStatus
    started_at: str | None = None


class StageLogSnapshot(StrictModel):
    id: str = Field(min_length=1, max_length=120)
    attempt: str | None = None
    status: StageStatus
    stdout: list[str] = Field(default_factory=list, max_length=400)
    stderr: list[str] = Field(default_factory=list, max_length=400)
    updated_at: str | None = None


class JobLogs(StrictModel):
    schema_version: Literal["web-job-logs-v1"]
    job_id: str = Field(min_length=1, max_length=120)
    stages: list[StageLogSnapshot]


class QualityFlags(StrictModel):
    accepted_by_automated_policy: bool = False
    gaussian_schema_valid: bool = False
    supersplat_format_compatible: bool = False
    supersplat_runtime_verified: bool = False
    manual_visual_review: bool = False
    structural_evaluation_pass: bool | None = None
    visual_quality_pass: bool | None = None


class ErrorDetail(StrictModel):
    code: str | None = Field(default=None, min_length=1, max_length=80)
    stage: str | None = None
    message: str = Field(min_length=1, max_length=2000)
    exit_code: int | None = None


class JobSnapshot(StrictModel):
    schema_version: Literal["web-job-snapshot-v1"]
    job_id: str = Field(min_length=1, max_length=120)
    run_id: str = Field(min_length=1, max_length=120)
    status: JobStatus
    created_at: str
    updated_at: str
    route: RouteContract = Field(default_factory=RouteContract)
    backend: BackendIdentity | None = None
    input: InputDescriptor | None = None
    stage_order: list[str]
    current_stage: CurrentStage | None = None
    stages: list[StageSnapshot]
    observed_progress: ObservedProgress | None = None
    artifacts: list[ArtifactDescriptor]
    quality: QualityFlags
    error: ErrorDetail | None = None
    next_action: str


class CreateJobResponse(StrictModel):
    schema_version: Literal["web-job-v1"]
    job_id: str
    status: JobStatus
    created_at: str
    input: InputDescriptor
    status_url: str


class ApiError(StrictModel):
    schema_version: Literal["web-api-error-v1"] = "web-api-error-v1"
    code: str
    message: str
    job_id: str | None = None
