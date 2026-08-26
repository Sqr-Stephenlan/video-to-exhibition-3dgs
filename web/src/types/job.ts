export type JobStatus = "queued" | "running" | "complete" | "blocked" | "failed" | "stopped";
export type StageStatus = "queued" | "running" | "passed" | "blocked" | "failed" | "stopped" | "skipped";

export interface RouteContract {
  pipeline_profile: "single-convergence1000-v1";
  depth_source: "disabled";
  pose_mode: "external-fixed-pose-rgb-only";
  acceptance_policy: "automated-technical-v1";
}

export interface BackendIdentity {
  superproject_gitlink: string;
  longsplat_commit: string;
  provider_identity: string;
}

export interface InputDescriptor {
  name: string;
  size_bytes: number;
  sha256: string;
}

export interface ObservedProgress {
  kind: "conversion-iteration" | "training-iteration";
  iteration: number;
  total: number;
  observed_at: string;
  stage?: string | null;
  attempt?: string | null;
}

export interface ArtifactDescriptor {
  id: string;
  kind: string;
  format: string;
  download_url: string;
  sha256: string;
  size_bytes: number;
  vertices?: number | null;
}

export interface StageSnapshot {
  id: string;
  index: number;
  status: StageStatus;
  attempt: string | null;
  started_at: string | null;
  finished_at: string | null;
  summary: Record<string, unknown>;
}

export interface CurrentStage {
  id: string;
  index: number;
  attempt: string | null;
  status: StageStatus;
  started_at: string | null;
}

export interface StageLogSnapshot {
  id: string;
  attempt: string | null;
  status: StageStatus;
  stdout: string[];
  stderr: string[];
  updated_at: string | null;
}

export interface JobLogs {
  schema_version: "web-job-logs-v1";
  job_id: string;
  stages: StageLogSnapshot[];
}

export interface QualityFlags {
  accepted_by_automated_policy: boolean;
  gaussian_schema_valid: boolean;
  supersplat_format_compatible: boolean;
  supersplat_runtime_verified: boolean;
  manual_visual_review: boolean;
  structural_evaluation_pass: boolean | null;
  visual_quality_pass: boolean | null;
}

export interface ErrorDetail {
  code?: string | null;
  stage: string | null;
  message: string;
  exit_code: number | null;
}

export interface JobSnapshot {
  schema_version: "web-job-snapshot-v1";
  job_id: string;
  run_id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  route: RouteContract;
  backend: BackendIdentity | null;
  input?: InputDescriptor | null;
  stage_order: string[];
  current_stage: CurrentStage | null;
  stages: StageSnapshot[];
  observed_progress: ObservedProgress | null;
  artifacts: ArtifactDescriptor[];
  quality: QualityFlags;
  error: ErrorDetail | null;
  next_action: string;
}

export interface CreateJobResponse {
  schema_version: "web-job-v1";
  job_id: string;
  status: JobStatus;
  created_at: string;
  input: InputDescriptor;
  status_url: string;
}

export interface ApiError {
  schema_version: "web-api-error-v1";
  code: string;
  message: string;
  job_id?: string | null;
}
