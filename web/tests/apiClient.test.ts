import { describe, expect, it, vi } from "vitest";

import { createJob, downloadArtifact, getJob, getJobLogs } from "../src/api/client";
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
  stage_order: [],
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
      size_bytes: 7,
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

describe("api client", () => {
  it("uploads a local video as multipart form data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          schema_version: "web-job-v1",
          job_id: snapshot.job_id,
          status: "queued",
          created_at: snapshot.created_at,
          input: { name: "gallery.mp4", size_bytes: 5, sha256: "b".repeat(64) },
          status_url: `/api/v1/jobs/${snapshot.job_id}`,
        }),
        { status: 202, headers: { "Content-Type": "application/json" } },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["video"], "gallery.mp4", { type: "video/mp4" });

    const response = await createJob(file, "Gallery");
    const body = fetchMock.mock.calls[0][1]?.body as FormData;

    expect(response.job_id).toBe(snapshot.job_id);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/jobs");
    expect(body.get("video")).toBeInstanceOf(File);
    expect(body.get("name")).toBe("Gallery");
  });

  it("returns only the artifact URL declared by a backend snapshot", () => {
    expect(downloadArtifact(snapshot, "published-ply")).toBe(
      "/api/v1/jobs/web-111111111111111111111111/artifacts/published-ply",
    );
    expect(() => downloadArtifact(snapshot, "unknown")).toThrow("artifact");
  });

  it("reads a job snapshot and exposes structured API errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(snapshot), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(getJob(snapshot.job_id)).resolves.toEqual(snapshot);
  });

  it("reads bounded stage logs without parsing CLI output in the browser", async () => {
    const logs: JobLogs = {
      schema_version: "web-job-logs-v1",
      job_id: snapshot.job_id,
      stages: [
        {
          id: "formal-training",
          attempt: "attempt-0001",
          status: "passed",
          stdout: ["iteration 30000"],
          stderr: [],
          updated_at: "2026-08-24T00:01:00Z",
        },
      ],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(logs), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(getJobLogs(snapshot.job_id)).resolves.toEqual(logs);
    expect(fetch).toHaveBeenCalledWith(
      `/api/v1/jobs/${snapshot.job_id}/logs?tail=120`,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it("turns a disconnected API into a structured network error", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("connection refused")));

    await expect(getJob(snapshot.job_id)).rejects.toMatchObject({
      code: "network_error",
      status: 0,
    });
  });

  it("times out a hung API request", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn().mockImplementation(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("The operation was aborted", "AbortError"));
          });
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const request = getJob(snapshot.job_id);
    const result = expect(request).rejects.toMatchObject({
      code: "request_timeout",
      status: 0,
    });
    await vi.advanceTimersByTimeAsync(15_000);

    await result;
    vi.useRealTimers();
  });

  it("propagates caller cancellation to the underlying request", async () => {
    const controller = new AbortController();
    let requestSignal: AbortSignal | null | undefined;
    const fetchMock = vi.fn().mockImplementation(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          requestSignal = init?.signal;
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("The operation was aborted", "AbortError"));
          });
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const request = getJob(snapshot.job_id, { signal: controller.signal });
    controller.abort();

    await expect(request).rejects.toMatchObject({ name: "AbortError" });
    expect(requestSignal?.aborted).toBe(true);
  });
});
