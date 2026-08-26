import { describe, expect, it } from "vitest";

import { initialJobState, jobReducer, shouldPollJob } from "../src/state/jobStore";
import type { JobSnapshot } from "../src/types/job";

const runningSnapshot: JobSnapshot = {
  schema_version: "web-job-snapshot-v1",
  job_id: "job-1",
  run_id: "job-1",
  status: "running",
  created_at: "2026-08-24T00:00:00Z",
  updated_at: "2026-08-24T00:01:00Z",
  route: {
    pipeline_profile: "single-convergence1000-v1",
    depth_source: "disabled",
    pose_mode: "external-fixed-pose-rgb-only",
    acceptance_policy: "automated-technical-v1",
  },
  backend: null,
  stage_order: ["probe", "conversion"],
  current_stage: {
    id: "conversion",
    index: 2,
    attempt: "attempt-0001",
    status: "running",
    started_at: "2026-08-24T00:00:30Z",
  },
  stages: [],
  observed_progress: {
    kind: "conversion-iteration",
    iteration: 12,
    total: 30,
    observed_at: "2026-08-24T00:01:00Z",
  },
  artifacts: [],
  quality: {
    accepted_by_automated_policy: false,
    gaussian_schema_valid: false,
    supersplat_format_compatible: false,
    supersplat_runtime_verified: false,
    manual_visual_review: false,
    structural_evaluation_pass: null,
    visual_quality_pass: null,
  },
  error: null,
  next_action: "等待阶段 conversion 完成",
};

describe("jobStore", () => {
  it("accepts backend snapshots without inventing a percent", () => {
    const state = jobReducer(initialJobState, {
      type: "snapshot",
      snapshot: runningSnapshot,
    });

    expect(state.snapshot?.current_stage?.id).toBe("conversion");
    expect(state.snapshot).not.toHaveProperty("percent");
    expect(shouldPollJob(runningSnapshot)).toBe(true);
  });

  it("stops polling only for terminal backend statuses", () => {
    expect(shouldPollJob({ ...runningSnapshot, status: "complete" })).toBe(false);
    expect(shouldPollJob({ ...runningSnapshot, status: "blocked" })).toBe(false);
    expect(shouldPollJob({ ...runningSnapshot, status: "failed" })).toBe(false);
    expect(shouldPollJob({ ...runningSnapshot, status: "queued" })).toBe(true);
  });
});
