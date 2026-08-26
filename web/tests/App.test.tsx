import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { CreateJobResponse, JobSnapshot } from "../src/types/job";

const mocks = vi.hoisted(() => ({
  createJob: vi.fn(),
  getJob: vi.fn(),
  getJobLogs: vi.fn(),
  pollJob: vi.fn(),
}));

vi.mock("../src/api/client", () => ({
  createJob: mocks.createJob,
  getJob: mocks.getJob,
  getJobLogs: mocks.getJobLogs,
}));
vi.mock("../src/api/pollJob", () => ({ pollJob: mocks.pollJob }));

import App from "../src/App";

const { createJob, getJob, pollJob } = mocks;

const response: CreateJobResponse = {
  schema_version: "web-job-v1",
  job_id: "web-111111111111111111111111",
  status: "queued",
  created_at: "2026-08-24T00:00:00Z",
  input: { name: "gallery.mp4", size_bytes: 5, sha256: "a".repeat(64) },
  status_url: "/api/v1/jobs/web-111111111111111111111111",
};

const completeSnapshot: JobSnapshot = {
  schema_version: "web-job-snapshot-v1",
  job_id: response.job_id,
  run_id: response.job_id,
  status: "complete",
  created_at: response.created_at,
  updated_at: response.created_at,
  route: {
    pipeline_profile: "single-convergence1000-v1",
    depth_source: "disabled",
    pose_mode: "external-fixed-pose-rgb-only",
    acceptance_policy: "automated-technical-v1",
  },
  backend: null,
  stage_order: [],
  current_stage: null,
  stages: [],
  observed_progress: null,
  artifacts: [
    {
      id: "published-ply",
      kind: "model",
      format: "ply",
      download_url: `/api/v1/jobs/${response.job_id}/artifacts/published-ply`,
      sha256: "b".repeat(64),
      size_bytes: 100,
      vertices: 1,
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

describe("App", () => {
  it("moves the single-page flow from upload to a real terminal snapshot", async () => {
    createJob.mockResolvedValue(response);
    pollJob.mockImplementation(async (_jobId: string, options: { onSnapshot: (snapshot: JobSnapshot) => void }) => {
      options.onSnapshot(completeSnapshot);
      return completeSnapshot;
    });

    render(<App />);
    const file = new File(["video"], "gallery.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByLabelText("视频文件"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "开始生成 3D 高斯模型" }));

    await waitFor(() => expect(screen.getByText("模型已生成")).toBeInTheDocument());
    expect(createJob).toHaveBeenCalledWith(file, "gallery");
    expect(new URLSearchParams(window.location.search).get("job")).toBe(response.job_id);
    expect(pollJob).toHaveBeenCalledWith(
      response.job_id,
      expect.objectContaining({ getSnapshot: getJob, signal: expect.any(AbortSignal) }),
    );
    expect(screen.queryByRole("button", { name: /验收通过/ })).not.toBeInTheDocument();
    window.history.replaceState({}, "", "/");
  });

  it("keeps the last snapshot visible and surfaces a polling error", async () => {
    vi.clearAllMocks();
    createJob.mockResolvedValue(response);
    pollJob.mockImplementation(async (_jobId: string, options: { onSnapshot: (snapshot: JobSnapshot) => void }) => {
      options.onSnapshot({
        ...completeSnapshot,
        status: "running",
        next_action: "等待流水线状态更新",
      });
      throw new Error("connection lost");
    });

    render(<App />);
    const file = new File(["video"], "gallery.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByLabelText("视频文件"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "开始生成 3D 高斯模型" }));

    await waitFor(() => expect(screen.getByText(/状态轮询中断：connection lost/)).toBeInTheDocument());
    expect(screen.getByText("正在生成模型")).toBeInTheDocument();
  });

  it("opens an existing job from the query string without uploading or starting training", async () => {
    vi.clearAllMocks();
    window.history.pushState({}, "", `/?job=${response.job_id}`);
    pollJob.mockImplementation(async (_jobId: string, options: { onSnapshot: (snapshot: JobSnapshot) => void }) => {
      options.onSnapshot(completeSnapshot);
      return completeSnapshot;
    });

    render(<App />);

    await waitFor(() => expect(screen.getByText("模型已生成")).toBeInTheDocument());
    expect(createJob).not.toHaveBeenCalled();
    expect(pollJob).toHaveBeenCalledWith(
      response.job_id,
      expect.objectContaining({ getSnapshot: getJob }),
    );
    window.history.replaceState({}, "", "/");
  });
});
