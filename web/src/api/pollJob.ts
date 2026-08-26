import type { JobLogs, JobSnapshot } from "../types/job";

interface PollRequestOptions {
  signal?: AbortSignal;
}

export interface PollJobOptions {
  getSnapshot: (jobId: string, options?: PollRequestOptions) => Promise<JobSnapshot>;
  onSnapshot: (snapshot: JobSnapshot) => void;
  getLogs?: (jobId: string, options?: PollRequestOptions) => Promise<JobLogs>;
  onLogs?: (logs: JobLogs) => void;
  onError?: (reason: unknown | null) => void;
  onLogsError?: (reason: unknown | null) => void;
  intervalMs?: number;
  maxSnapshotErrors?: number;
  sleep?: (durationMs: number) => Promise<void>;
  signal?: AbortSignal;
}

const terminalStatuses = new Set<JobSnapshot["status"]>([
  "complete",
  "blocked",
  "failed",
  "stopped",
]);

const defaultSleep = (durationMs: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("轮询已取消", "AbortError"));
      return;
    }
    const timer = window.setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, durationMs);
    function onAbort() {
      window.clearTimeout(timer);
      reject(new DOMException("轮询已取消", "AbortError"));
    }
    signal?.addEventListener("abort", onAbort, { once: true });
  });

export async function pollJob(jobId: string, options: PollJobOptions): Promise<JobSnapshot> {
  const intervalMs = options.intervalMs ?? 2000;
  const maxSnapshotErrors = options.maxSnapshotErrors ?? 5;
  const sleep = options.sleep ?? defaultSleep;
  let consecutiveSnapshotErrors = 0;
  let latest: JobSnapshot;
  while (true) {
    if (options.signal?.aborted) {
      throw new DOMException("轮询已取消", "AbortError");
    }
    try {
      latest = await options.getSnapshot(jobId, { signal: options.signal });
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === "AbortError") {
        throw reason;
      }
      consecutiveSnapshotErrors += 1;
      options.onError?.(reason);
      if (consecutiveSnapshotErrors > maxSnapshotErrors) {
        throw reason;
      }
      const backoff = intervalMs * Math.min(2 ** (consecutiveSnapshotErrors - 1), 4);
      if (options.sleep) {
        await sleep(backoff);
      } else {
        await defaultSleep(backoff, options.signal);
      }
      continue;
    }
    consecutiveSnapshotErrors = 0;
    options.onError?.(null);
    options.onSnapshot(latest);
    if (options.getLogs && (options.onLogs || options.onLogsError)) {
      try {
        const nextLogs = await options.getLogs(jobId, { signal: options.signal });
        options.onLogs?.(nextLogs);
        options.onLogsError?.(null);
      } catch (reason) {
        if (reason instanceof DOMException && reason.name === "AbortError") {
          throw reason;
        }
        options.onLogsError?.(reason);
      }
    }
    if (terminalStatuses.has(latest.status)) {
      return latest;
    }
    if (options.sleep) {
      await sleep(intervalMs);
    } else {
      await defaultSleep(intervalMs, options.signal);
    }
  }
}
