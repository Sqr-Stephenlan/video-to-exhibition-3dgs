import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StageTimeline } from "../src/components/StageTimeline";
import type { JobSnapshot } from "../src/types/job";

const snapshot: JobSnapshot = {
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
  stage_order: ["probe", "conversion", "automated-technical-delivery"],
  current_stage: {
    id: "conversion",
    index: 2,
    attempt: "attempt-0001",
    status: "running",
    started_at: "2026-08-24T00:00:30Z",
  },
  stages: [
    {
      id: "probe",
      index: 1,
      status: "passed",
      attempt: "attempt-0001",
      started_at: null,
      finished_at: null,
      summary: {},
    },
    {
      id: "conversion",
      index: 2,
      status: "running",
      attempt: "attempt-0001",
      started_at: null,
      finished_at: null,
      summary: {},
    },
    {
      id: "automated-technical-delivery",
      index: 3,
      status: "queued",
      attempt: null,
      started_at: null,
      finished_at: null,
      summary: {},
    },
  ],
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

describe("StageTimeline", () => {
  it("shows stage position and observed iteration without a fake percent", () => {
    render(<StageTimeline snapshot={snapshot} />);

    expect(screen.getByText("第 2/3 阶段")).toBeInTheDocument();
    expect(screen.getByText("转换迭代 12 / 30")).toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
    expect(screen.getByText("自动技术交付")).toBeInTheDocument();
  });

  it("labels a terminal complete pipeline instead of saying it is waiting", () => {
    render(<StageTimeline snapshot={{ ...snapshot, status: "complete", current_stage: null }} />);

    expect(screen.getByRole("heading", { name: "流水线已完成" })).toBeInTheDocument();
    expect(screen.queryByText("等待阶段状态")).not.toBeInTheDocument();
  });

  it("labels progress from a previous stage as historical", () => {
    render(
      <StageTimeline
        snapshot={{
          ...snapshot,
          observed_progress: {
            ...snapshot.observed_progress!,
            kind: "training-iteration",
            stage: "formal-training",
          },
        }}
      />,
    );

    expect(screen.getByText(/历史观测/)).toBeInTheDocument();
    expect(screen.getByText(/正式训练/)).toBeInTheDocument();
  });

  it("labels progress from a previous attempt as historical", () => {
    render(
      <StageTimeline
        snapshot={{
          ...snapshot,
          current_stage: {
            ...snapshot.current_stage!,
            attempt: "attempt-0002",
          },
          observed_progress: {
            ...snapshot.observed_progress!,
            stage: "conversion",
            attempt: "attempt-0001",
          },
        }}
      />,
    );

    expect(screen.getByText(/历史观测/)).toBeInTheDocument();
  });
});
