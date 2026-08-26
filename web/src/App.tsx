import { useEffect, useState, type ReactNode } from "react";

import { getJob, getJobLogs } from "./api/client";
import { pollJob } from "./api/pollJob";
import { ResultPage } from "./pages/ResultPage";
import { RunPage } from "./pages/RunPage";
import { UploadPage } from "./pages/UploadPage";
import type { CreateJobResponse, JobLogs, JobSnapshot } from "./types/job";
import "./styles/app.css";

function AppFrame({ children, onReset }: { children: ReactNode; onReset?: () => void }) {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-mark" aria-label="Video to 3D Gaussian Splatting">
          <span className="brand-mark__dot" />
          <span>V/3DGS</span>
        </div>
        <div className="header-meta">
          <span className="status-dot" />
          <span>本地单用户工作台</span>
          {onReset && (
            <button className="ghost-button" type="button" onClick={onReset}>
              新建任务
            </button>
          )}
        </div>
      </header>
      {children}
      <footer className="app-footer">
        <span>Local reconstruction console</span>
        <span>LongSplat · fixed route · technical evidence first</span>
      </footer>
    </div>
  );
}

function PollingError({ message, onReset }: { message: string; onReset: () => void }) {
  return (
    <main className="page failure-page">
      <section className="failure-card" role="alert">
        <p className="eyebrow">STATUS CONNECTION INTERRUPTED</p>
        <h1>暂时无法读取任务状态</h1>
        <p>{message}</p>
        <p className="muted">任务可能仍在本地运行；重新导入会创建一个新的 job。</p>
        <button type="button" onClick={onReset}>
          返回上传
        </button>
      </section>
    </main>
  );
}

function PollingNotice({ message }: { message: string }) {
  return (
    <div className="polling-notice" role="alert">
      状态轮询中断：{message}。页面保留最后一份后端快照，任务是否继续请以本地 evidence 为准。
    </div>
  );
}

function jobIdFromLocation(): string | null {
  const value = new URLSearchParams(window.location.search).get("job");
  return value && /^web-[0-9a-f]{24}$/.test(value) ? value : null;
}

export default function App() {
  const [jobId, setJobId] = useState<string | null>(jobIdFromLocation);
  const [snapshot, setSnapshot] = useState<JobSnapshot | null>(null);
  const [logs, setLogs] = useState<JobLogs | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);
  const [logsError, setLogsError] = useState<string | null>(null);
  const [pollingFailed, setPollingFailed] = useState(false);
  const [uploadKey, setUploadKey] = useState(0);

  useEffect(() => {
    if (!jobId) {
      return;
    }

    const controller = new AbortController();
    setPollError(null);
    setLogsError(null);
    setPollingFailed(false);
    void pollJob(jobId, {
      getSnapshot: getJob,
      onSnapshot: setSnapshot,
      getLogs: getJobLogs,
      onLogs: setLogs,
      onError: (reason) => {
        setPollError(reason ? (reason instanceof Error ? reason.message : "读取任务状态失败") : null);
      },
      onLogsError: (reason) => {
        setLogsError(reason ? (reason instanceof Error ? reason.message : "读取阶段日志失败") : null);
      },
      signal: controller.signal,
    }).catch((reason: unknown) => {
      if (reason instanceof DOMException && reason.name === "AbortError") {
        return;
      }
      setPollingFailed(true);
      setPollError(reason instanceof Error ? reason.message : "读取任务状态失败");
    });

    return () => controller.abort();
  }, [jobId]);

  function handleCreated(response: CreateJobResponse) {
    setPollError(null);
    setLogsError(null);
    setPollingFailed(false);
    setSnapshot(null);
    setLogs(null);
    const query = new URLSearchParams(window.location.search);
    query.set("job", response.job_id);
    window.history.replaceState({}, "", `${window.location.pathname}?${query.toString()}`);
    setJobId(response.job_id);
  }

  function reset() {
    setJobId(null);
    setSnapshot(null);
    setLogs(null);
    setPollError(null);
    setLogsError(null);
    setPollingFailed(false);
    setUploadKey((value) => value + 1);
    const query = new URLSearchParams(window.location.search);
    query.delete("job");
    const suffix = query.toString() ? `?${query.toString()}` : "";
    window.history.replaceState({}, "", `${window.location.pathname}${suffix}`);
  }

  const publicBaseUrl = import.meta.env.VITE_PUBLIC_API_BASE_URL as string | undefined;

  if (!jobId) {
    return (
      <AppFrame>
        <UploadPage key={uploadKey} onCreated={handleCreated} />
      </AppFrame>
    );
  }

  if (pollingFailed && !snapshot) {
    return (
      <AppFrame onReset={reset}>
        <PollingError message={pollError ?? "读取任务状态失败"} onReset={reset} />
      </AppFrame>
    );
  }

  if (!snapshot) {
    return (
      <AppFrame onReset={reset}>
        <main className="page loading-page" aria-live="polite">
          <section className="loading-card">
            <span className="loader" aria-hidden="true" />
            <div>
              <p className="eyebrow">JOB {jobId}</p>
              <h1>正在连接本地重建任务</h1>
              <p className="muted">
                {pollError ? `状态暂时不可用，正在重试：${pollError}` : "等待后端返回第一份真实阶段快照…"}
              </p>
            </div>
          </section>
        </main>
      </AppFrame>
    );
  }

  let page = <RunPage snapshot={snapshot} logs={logs} />;
  if (snapshot.status === "complete") {
    page = (
      <ResultPage
        snapshot={snapshot}
        publicBaseUrl={publicBaseUrl}
        logs={logs}
        logsError={logsError}
      />
    );
  } else if (snapshot.status === "blocked" || snapshot.status === "failed" || snapshot.status === "stopped") {
    page = <RunPage snapshot={snapshot} logs={logs} logsError={logsError} onNewJob={reset} />;
  } else {
    page = <RunPage snapshot={snapshot} logs={logs} logsError={logsError} />;
  }

  return (
    <AppFrame onReset={reset}>
      {pollError && <PollingNotice message={pollError} />}
      {page}
    </AppFrame>
  );
}
