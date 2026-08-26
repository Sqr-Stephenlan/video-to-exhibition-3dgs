import type { ApiError, CreateJobResponse, JobLogs, JobSnapshot } from "../types/job";

interface RequestOptions {
  signal?: AbortSignal;
}

const DEFAULT_REQUEST_TIMEOUT_MS = 15_000;
const UPLOAD_REQUEST_TIMEOUT_MS = 5 * 60_000;

export class ApiClientError extends Error {
  readonly status: number;
  readonly code: string;
  readonly jobId: string | null;

  constructor(status: number, payload: Partial<ApiError> | null) {
    super(payload?.message || `请求失败（HTTP ${status}）`);
    this.name = "ApiClientError";
    this.status = status;
    this.code = payload?.code || "http_error";
    this.jobId = payload?.job_id || null;
  }
}

async function jsonResponse<T>(response: Response): Promise<T> {
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    throw new ApiClientError(
      response.status,
      payload && typeof payload === "object" ? (payload as Partial<ApiError>) : null,
    );
  }
  return payload as T;
}

async function request(
  input: RequestInfo | URL,
  init: RequestInit = {},
  timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  let timedOut = false;
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onCallerAbort = () => controller.abort();
  if (init.signal?.aborted) {
    controller.abort();
  } else {
    init.signal?.addEventListener("abort", onCallerAbort, { once: true });
  }

  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (reason) {
    if (reason instanceof DOMException && reason.name === "AbortError") {
      if (timedOut && !init.signal?.aborted) {
        throw new ApiClientError(0, {
          code: "request_timeout",
          message: "本地 Web API 请求超时，请检查服务状态",
        });
      }
      throw reason;
    }
    throw new ApiClientError(0, {
      code: "network_error",
      message: "无法连接本地 Web API，请确认 API 服务正在运行",
    });
  } finally {
    window.clearTimeout(timer);
    init.signal?.removeEventListener("abort", onCallerAbort);
  }
}

export async function createJob(file: File, name?: string): Promise<CreateJobResponse> {
  const body = new FormData();
  body.append("video", file, file.name);
  if (name?.trim()) {
    body.append("name", name.trim());
  }
  const response = await request("/api/v1/jobs", { method: "POST", body }, UPLOAD_REQUEST_TIMEOUT_MS);
  return jsonResponse<CreateJobResponse>(response);
}

export async function getJob(jobId: string, options: RequestOptions = {}): Promise<JobSnapshot> {
  const response = await request(`/api/v1/jobs/${encodeURIComponent(jobId)}`, {
    signal: options.signal,
  });
  return jsonResponse<JobSnapshot>(response);
}

export async function getJobLogs(
  jobId: string,
  options: { stage?: string; tail?: number; signal?: AbortSignal } = {},
): Promise<JobLogs> {
  const query = new URLSearchParams();
  if (options.stage) {
    query.set("stage", options.stage);
  }
  query.set("tail", String(options.tail ?? 120));
  const response = await request(`/api/v1/jobs/${encodeURIComponent(jobId)}/logs?${query.toString()}`, {
    signal: options.signal,
  });
  return jsonResponse<JobLogs>(response);
}

export function downloadArtifact(snapshot: JobSnapshot, artifactId: string): string {
  const descriptor = snapshot.artifacts.find((artifact) => artifact.id === artifactId);
  if (!descriptor) {
    throw new Error(`artifact ${artifactId} is not declared by the backend`);
  }
  if (!descriptor.download_url.startsWith("/api/v1/jobs/") || descriptor.download_url.includes("..")) {
    throw new Error(`artifact ${artifactId} has an unsafe download URL`);
  }
  return descriptor.download_url;
}
