import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { RunPage } from "../src/pages/RunPage";
import type { JobLogs, JobSnapshot } from "../src/types/job";

const snapshot: JobSnapshot = {
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
  backend: {
    superproject_gitlink: "c".repeat(40),
    longsplat_commit: "c".repeat(40),
    provider_identity: "provider-sha256",
  },
  stage_order: ["probe", "conversion", "automated-technical-delivery"],
  current_stage: {
    id: "conversion",
    index: 2,
    attempt: "attempt-0001",
    status: "running",
    started_at: "2026-08-24T00:00:30Z",
  },
  stages: [
    { id: "probe", index: 1, status: "passed", attempt: "attempt-0001", started_at: null, finished_at: null, summary: {} },
    { id: "conversion", index: 2, status: "running", attempt: "attempt-0001", started_at: null, finished_at: null, summary: {} },
    { id: "automated-technical-delivery", index: 3, status: "queued", attempt: null, started_at: null, finished_at: null, summary: {} },
  ],
  observed_progress: null,
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

const logs: JobLogs = {
  schema_version: "web-job-logs-v1",
  job_id: snapshot.job_id,
  stages: [
    {
      id: "conversion",
      attempt: "attempt-0001",
      status: "running",
      stdout: ["conversion started"],
      stderr: ["waiting for model"],
      updated_at: "2026-08-24T00:01:00Z",
    },
  ],
};

describe("RunPage", () => {
  it("shows fixed route and backend identity without fake progress", () => {
    render(<RunPage snapshot={snapshot} logs={logs} />);

    expect(screen.getByText("single-convergence1000-v1")).toBeInTheDocument();
    expect(screen.getByText("external fixed-pose RGB-only")).toBeInTheDocument();
    expect(screen.getByText("provider-sha256")).toBeInTheDocument();
    expect(screen.getAllByText("当前没有可恢复的迭代观测").length).toBe(2);
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
    expect(screen.queryByText(/ETA/i)).not.toBeInTheDocument();
    expect(screen.getByText("conversion started")).toBeInTheDocument();
    expect(screen.getByText("waiting for model")).toBeInTheDocument();
  });

  it("labels a native training PLY as a non-delivery sample", () => {
    render(
      <RunPage
        snapshot={{
          ...snapshot,
          status: "blocked",
          current_stage: { ...snapshot.current_stage!, status: "blocked" },
          artifacts: [
            {
              id: "training-sample-ply",
              kind: "sample",
              format: "ply",
              download_url: "/api/v1/jobs/web-111111111111111111111111/artifacts/training-sample-ply",
              sha256: "d".repeat(64),
              size_bytes: 2048,
            },
          ],
          error: { code: "conversion_failed", stage: "conversion", message: "converter unavailable", exit_code: 193 },
          next_action: "查看失败阶段证据并重新提交任务",
        }}
        logs={logs}
      />,
    );

    expect(screen.getByRole("heading", { name: "训练样例（非交付）" })).toBeInTheDocument();
    expect(screen.getByText(/LongSplat 原生训练 PLY/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "下载训练样例 PLY" })).toBeInTheDocument();
  });

  it("shows a current-stage estimate once real iteration telemetry is available", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-24T00:01:00Z"));

    render(
      <RunPage
        snapshot={{
          ...snapshot,
          current_stage: {
            ...snapshot.current_stage!,
            id: "formal-training",
            started_at: "2026-08-24T00:00:00Z",
          },
          observed_progress: {
            kind: "training-iteration",
            iteration: 20,
            total: 100,
            observed_at: "2026-08-24T00:01:00Z",
            stage: "formal-training",
            attempt: "attempt-0001",
          },
        }}
        logs={logs}
      />,
    );

    expect(screen.getByText("预计剩余约 4 分钟")).toBeInTheDocument();
    expect(screen.getByText(/基于当前阶段已观测到的迭代速度/)).toBeInTheDocument();
    vi.useRealTimers();
  });

  it("shows an explicit task status when the job is queued", () => {
    render(
      <RunPage
        snapshot={{
          ...snapshot,
          status: "queued",
          current_stage: null,
          next_action: "等待任务启动",
        }}
        logs={null}
      />,
    );

    expect(screen.getByRole("heading", { name: "任务已排队" })).toBeInTheDocument();
    expect(screen.getByLabelText("任务状态")).toHaveTextContent("排队等待");
  });

  it("marks the last log snapshot as stale after a log read failure", () => {
    render(<RunPage snapshot={snapshot} logs={logs} logsError="日志服务暂时不可用" />);

    expect(screen.getByRole("alert")).toHaveTextContent("日志服务暂时不可用");
    expect(screen.getByText(/显示最后一份日志快照/)).toBeInTheDocument();
  });

  it("follows the current stage when a new stage log appears", () => {
    const { rerender } = render(<RunPage snapshot={snapshot} logs={logs} />);
    const formalLog = {
      id: "formal-training",
      attempt: "attempt-0001",
      status: "running" as const,
      stdout: ["formal training started"],
      stderr: [],
      updated_at: "2026-08-24T00:01:00Z",
    };

    rerender(
      <RunPage
        snapshot={{
          ...snapshot,
          stage_order: ["conversion", "formal-training"],
          current_stage: {
            id: "formal-training",
            index: 2,
            attempt: "attempt-0001",
            status: "running",
            started_at: "2026-08-24T00:01:00Z",
          },
        }}
        logs={{ ...logs, stages: [...logs.stages, formalLog] }}
      />,
    );

    expect(screen.getByRole("combobox", { name: "日志阶段" })).toHaveValue("formal-training");
    expect(screen.getByText("formal training started")).toBeInTheDocument();
  });
});
