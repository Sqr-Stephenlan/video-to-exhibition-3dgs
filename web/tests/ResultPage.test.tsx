import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ResultPage } from "../src/pages/ResultPage";
import type { JobLogs, JobSnapshot } from "../src/types/job";

const snapshot: JobSnapshot = {
  schema_version: "web-job-snapshot-v1",
  job_id: "web-111111111111111111111111",
  run_id: "web-111111111111111111111111",
  status: "complete",
  created_at: "2026-08-24T00:00:00Z",
  updated_at: "2026-08-24T00:01:00Z",
  route: {
    pipeline_profile: "single-convergence1000-v1",
    depth_source: "disabled",
    pose_mode: "external-fixed-pose-rgb-only",
    acceptance_policy: "automated-technical-v1",
  },
  backend: null,
  stage_order: ["automated-technical-delivery"],
  current_stage: null,
  stages: [],
  observed_progress: null,
  artifacts: [
    {
      id: "published-ply",
      kind: "model",
      format: "ply",
      download_url: "/api/v1/jobs/web-111111111111111111111111/artifacts/published-ply",
      sha256: "a".repeat(64),
      size_bytes: 1024,
      vertices: 100,
    },
    {
      id: "published-ply-receipt",
      kind: "receipt",
      format: "json",
      download_url: "/api/v1/jobs/web-111111111111111111111111/artifacts/published-ply-receipt",
      sha256: "b".repeat(64),
      size_bytes: 300,
    },
  ],
  quality: {
    accepted_by_automated_policy: true,
    gaussian_schema_valid: true,
    supersplat_format_compatible: true,
    supersplat_runtime_verified: false,
    manual_visual_review: false,
    structural_evaluation_pass: true,
    visual_quality_pass: null,
  },
  error: null,
  next_action: "下载模型或在 SuperSplat 中查看",
};

const logs: JobLogs = {
  schema_version: "web-job-logs-v1",
  job_id: snapshot.job_id,
  stages: [
    {
      id: "automated-technical-delivery",
      attempt: "attempt-0001",
      status: "passed",
      stdout: ["technical delivery complete"],
      stderr: [],
      updated_at: "2026-08-24T00:01:00Z",
    },
  ],
};

describe("ResultPage", () => {
  it("separates technical delivery from manual review and offers download import", () => {
    render(<ResultPage snapshot={snapshot} logs={logs} />);

    expect(screen.getByText("自动技术交付：通过")).toBeInTheDocument();
    expect(screen.getByText("PLY Gaussian 属性：通过")).toBeInTheDocument();
    expect(screen.getByText("SuperSplat 运行时：未验证")).toBeInTheDocument();
    expect(screen.getByText("人工验收：未完成")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "下载 PLY" })).toHaveAttribute(
      "href",
      "/api/v1/jobs/web-111111111111111111111111/artifacts/published-ply",
    );
    expect(screen.getByText("下载后导入 SuperSplat")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "阶段日志" })).toBeInTheDocument();
    expect(screen.getByText("technical delivery complete")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "流水线已完成" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /验收通过/ })).not.toBeInTheDocument();
  });
});
