import { describe, expect, it } from "vitest";

import { estimateStageTime } from "../src/api/estimate";
import type { JobSnapshot } from "../src/types/job";

function snapshot(overrides: Partial<JobSnapshot> = {}): JobSnapshot {
  return {
    schema_version: "web-job-snapshot-v1",
    job_id: "web-111111111111111111111111",
    run_id: "web-111111111111111111111111",
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
    stage_order: ["formal-training"],
    current_stage: {
      id: "formal-training",
      index: 1,
      attempt: "attempt-0001",
      status: "running",
      started_at: "2026-08-24T00:00:00Z",
    },
    stages: [],
    observed_progress: {
      kind: "training-iteration",
      iteration: 20,
      total: 100,
      observed_at: "2026-08-24T00:01:00Z",
      stage: "formal-training",
      attempt: "attempt-0001",
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
    next_action: "等待阶段 formal-training 完成",
    ...overrides,
  };
}

describe("estimateStageTime", () => {
  it("estimates remaining seconds from observed iterations and stage start", () => {
    const result = estimateStageTime(snapshot(), Date.parse("2026-08-24T00:01:00Z"));

    expect(result.state).toBe("estimating");
    expect(result.remainingSeconds).toBe(240);
  });

  it("does not estimate before a real active-stage progress observation", () => {
    const result = estimateStageTime(
      snapshot({ observed_progress: null, status: "queued", current_stage: null }),
      Date.parse("2026-08-24T00:01:00Z"),
    );

    expect(result.state).toBe("unavailable");
    expect(result.remainingSeconds).toBeNull();
  });

  it("does not reuse stale training progress after the stage has moved on", () => {
    const result = estimateStageTime(
      snapshot({
        current_stage: {
          id: "native-render",
          index: 1,
          attempt: "attempt-0001",
          status: "running",
          started_at: "2026-08-24T00:01:00Z",
        },
      }),
      Date.parse("2026-08-24T00:02:00Z"),
    );

    expect(result.state).toBe("unavailable");
    expect(result.remainingSeconds).toBeNull();
  });

  it("does not estimate from telemetry that has gone stale", () => {
    const result = estimateStageTime(
      snapshot({
        observed_progress: {
          kind: "training-iteration",
          iteration: 20,
          total: 100,
          observed_at: "2026-08-24T00:00:00Z",
          stage: "formal-training",
          attempt: "attempt-0001",
        },
      }),
      Date.parse("2026-08-24T00:04:00Z"),
    );

    expect(result.state).toBe("unavailable");
    expect(result.remainingSeconds).toBeNull();
    expect(result.basis).toMatch(/遥测/);
  });
});
