import { describe, expect, it } from "vitest";

import { pollJob } from "../src/api/pollJob";
import type { JobLogs, JobSnapshot } from "../src/types/job";

function snapshot(status: JobSnapshot["status"]): JobSnapshot {
  return {
    schema_version: "web-job-snapshot-v1",
    job_id: "web-111111111111111111111111",
    run_id: "web-111111111111111111111111",
    status,
    created_at: "2026-08-24T00:00:00Z",
    updated_at: "2026-08-24T00:00:00Z",
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
    next_action: "",
  };
}

describe("pollJob", () => {
  it("emits snapshots until the backend reaches a terminal status", async () => {
    const values = [snapshot("queued"), snapshot("running"), snapshot("complete")];
    const received: string[] = [];

    const result = await pollJob("web-111111111111111111111111", {
      getSnapshot: async () => values.shift() as JobSnapshot,
      onSnapshot: (value) => received.push(value.status),
      intervalMs: 0,
      sleep: async () => undefined,
    });

    expect(received).toEqual(["queued", "running", "complete"]);
    expect(result.status).toBe("complete");
  });

  it("refreshes logs alongside snapshots without turning log failures into job failures", async () => {
    const received: string[] = [];
    const logs: JobLogs = {
      schema_version: "web-job-logs-v1",
      job_id: "web-111111111111111111111111",
      stages: [],
    };

    const result = await pollJob("web-111111111111111111111111", {
      getSnapshot: async () => snapshot("complete"),
      getLogs: async () => logs,
      onSnapshot: () => undefined,
      onLogs: (value) => received.push(value.schema_version),
      intervalMs: 0,
      sleep: async () => undefined,
    });

    expect(result.status).toBe("complete");
    expect(received).toEqual(["web-job-logs-v1"]);
  });

  it("retries transient snapshot failures and reports recovery", async () => {
    let calls = 0;
    const errors: string[] = [];
    let receivedSignal: AbortSignal | undefined;

    const result = await pollJob("web-111111111111111111111111", {
      getSnapshot: async (_jobId, request) => {
        receivedSignal = request?.signal;
        calls += 1;
        if (calls === 1) {
          throw new Error("backend offline");
        }
        return snapshot("complete");
      },
      onSnapshot: () => undefined,
      onError: (reason) => errors.push(reason instanceof Error ? reason.message : "recovered"),
      intervalMs: 0,
      sleep: async () => undefined,
    });

    expect(result.status).toBe("complete");
    expect(errors).toEqual(["backend offline", "recovered"]);
    expect(receivedSignal).toBeUndefined();
  });

  it("passes the polling signal to snapshot and log requests", async () => {
    const controller = new AbortController();
    let snapshotSignal: AbortSignal | undefined;
    let logsSignal: AbortSignal | undefined;

    await pollJob("web-111111111111111111111111", {
      getSnapshot: async (_jobId, request) => {
        snapshotSignal = request?.signal;
        return snapshot("complete");
      },
      onSnapshot: () => undefined,
      getLogs: async (_jobId, request) => {
        logsSignal = request?.signal;
        return { schema_version: "web-job-logs-v1", job_id: "web-111111111111111111111111", stages: [] };
      },
      onLogs: () => undefined,
      signal: controller.signal,
      intervalMs: 0,
      sleep: async () => undefined,
    });

    expect(snapshotSignal).toBe(controller.signal);
    expect(logsSignal).toBe(controller.signal);
  });

  it("reports a stale log snapshot without stopping status polling", async () => {
    const errors: string[] = [];

    const result = await pollJob("web-111111111111111111111111", {
      getSnapshot: async () => snapshot("complete"),
      onSnapshot: () => undefined,
      getLogs: async () => {
        throw new Error("logs offline");
      },
      onLogsError: (reason) => {
        if (reason) {
          errors.push(reason instanceof Error ? reason.message : "log failure");
        }
      },
      intervalMs: 0,
      sleep: async () => undefined,
    });

    expect(result.status).toBe("complete");
    expect(errors).toEqual(["logs offline"]);
  });

  it("stops after the configured snapshot retry limit", async () => {
    let calls = 0;

    await expect(
      pollJob("web-111111111111111111111111", {
        getSnapshot: async () => {
          calls += 1;
          throw new Error("backend offline");
        },
        onSnapshot: () => undefined,
        intervalMs: 0,
        maxSnapshotErrors: 2,
        sleep: async () => undefined,
      }),
    ).rejects.toThrow("backend offline");

    expect(calls).toBe(3);
  });

  it("clears its pending timer when the flow is unmounted", async () => {
    const controller = new AbortController();
    const queued = snapshot("queued");
    const promise = pollJob("web-111111111111111111111111", {
      getSnapshot: async () => queued,
      onSnapshot: () => controller.abort(),
      intervalMs: 60_000,
      signal: controller.signal,
    });

    await expect(promise).rejects.toMatchObject({ name: "AbortError" });
  });
});
