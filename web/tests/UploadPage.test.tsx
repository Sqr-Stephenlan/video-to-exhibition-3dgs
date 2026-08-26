import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { UploadPage } from "../src/pages/UploadPage";
import type { CreateJobResponse } from "../src/types/job";

const response: CreateJobResponse = {
  schema_version: "web-job-v1",
  job_id: "web-111111111111111111111111",
  status: "queued",
  created_at: "2026-08-24T00:00:00Z",
  input: { name: "gallery.mp4", size_bytes: 5, sha256: "a".repeat(64) },
  status_url: "/api/v1/jobs/web-111111111111111111111111",
};

describe("UploadPage", () => {
  it("submits one local video and hands the job to the flow", async () => {
    const submitJob = vi.fn().mockResolvedValue(response);
    const onCreated = vi.fn();
    render(<UploadPage submitJob={submitJob} onCreated={onCreated} />);

    const file = new File(["video"], "gallery.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByLabelText("视频文件"), { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: "开始生成 3D 高斯模型" }));

    await waitFor(() => expect(submitJob).toHaveBeenCalledWith(file, "gallery"));
    expect(onCreated).toHaveBeenCalledWith(response);
  });

  it("rejects non-video files before upload", () => {
    const submitJob = vi.fn();
    render(<UploadPage submitJob={submitJob} onCreated={vi.fn()} />);

    const file = new File(["text"], "notes.txt", { type: "text/plain" });
    fireEvent.change(screen.getByLabelText("视频文件"), { target: { files: [file] } });

    expect(screen.getByText("请选择视频文件")).toBeInTheDocument();
    expect(submitJob).not.toHaveBeenCalled();
  });

  it("rejects a video above the configured local size limit", () => {
    const submitJob = vi.fn();
    render(<UploadPage maxFileBytes={4} submitJob={submitJob} onCreated={vi.fn()} />);

    const file = new File(["12345"], "large.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByLabelText("视频文件"), { target: { files: [file] } });

    expect(screen.getByText(/超过本地上传限制/)).toBeInTheDocument();
    expect(submitJob).not.toHaveBeenCalled();
  });

  it("rejects an empty video before upload", () => {
    const submitJob = vi.fn();
    render(<UploadPage submitJob={submitJob} onCreated={vi.fn()} />);

    const file = new File([], "empty.mp4", { type: "video/mp4" });
    fireEvent.change(screen.getByLabelText("视频文件"), { target: { files: [file] } });

    expect(screen.getByText("视频文件不能为空")).toBeInTheDocument();
    expect(submitJob).not.toHaveBeenCalled();
  });
});
