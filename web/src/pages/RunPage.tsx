import { FailurePanel } from "../components/FailurePanel";
import { ArtifactPanel } from "../components/ArtifactPanel";
import { ObservedProgress } from "../components/ObservedProgress";
import { StageLogs } from "../components/StageLogs";
import { StageTimeEstimate } from "../components/StageTimeEstimate";
import { StageTimeline } from "../components/StageTimeline";
import type { JobLogs, JobSnapshot, JobStatus } from "../types/job";

const taskStatusLabels: Record<JobStatus, string> = {
  queued: "排队等待",
  running: "运行中",
  complete: "已完成",
  blocked: "已阻塞",
  failed: "失败",
  stopped: "已停止",
};

export function RunPage({
  snapshot,
  logs = null,
  logsError = null,
  onNewJob,
}: {
  snapshot: JobSnapshot;
  logs?: JobLogs | null;
  logsError?: string | null;
  onNewJob?: () => void;
}) {
  const terminal = ["blocked", "failed", "stopped"].includes(snapshot.status);
  const heading =
    snapshot.status === "queued"
      ? "任务已排队"
      : snapshot.status === "blocked"
      ? "重建流程已阻塞"
      : snapshot.status === "failed"
        ? "重建流程失败"
        : snapshot.status === "stopped"
          ? "重建流程已停止"
          : "正在生成模型";

  return (
    <main className="page run-page">
      <section className="page-heading">
        <p className="eyebrow">RUN {snapshot.job_id}</p>
        <h1>{heading}</h1>
        <p className="run-status" aria-label="任务状态">
          任务状态：<strong>{taskStatusLabels[snapshot.status]}</strong>
        </p>
        <p>{snapshot.next_action}</p>
      </section>
      <section className="card route-card" aria-label="固定运行配置">
        <div>
          <span className="label">生产 profile</span>
          <strong>{snapshot.route.pipeline_profile}</strong>
        </div>
        <div>
          <span className="label">深度路线</span>
          <strong>{snapshot.route.depth_source}</strong>
        </div>
        <div>
          <span className="label">相机/姿态</span>
          <strong>
            {snapshot.route.pose_mode === "external-fixed-pose-rgb-only"
              ? "external fixed-pose RGB-only"
              : snapshot.route.pose_mode}
          </strong>
        </div>
        <div>
          <span className="label">backend</span>
          <strong>{snapshot.backend?.longsplat_commit ?? "预检尚未完成"}</strong>
        </div>
        <div>
          <span className="label">provider</span>
          <strong>{snapshot.backend?.provider_identity ?? "预检尚未完成"}</strong>
        </div>
      </section>
      <section className="card progress-card">
        <div className="progress-summary">
          <ObservedProgress
            progress={snapshot.observed_progress}
            currentStageId={snapshot.current_stage?.id}
            currentStageAttempt={snapshot.current_stage?.attempt}
          />
          <StageTimeEstimate snapshot={snapshot} />
        </div>
        {!terminal && snapshot.error && (
          <div className="failure-inline" role="alert">
            {snapshot.error.code && <code>{snapshot.error.code}</code>}
            <span>{snapshot.error.message}</span>
          </div>
        )}
      </section>
      {terminal && <FailurePanel snapshot={snapshot} onNewJob={onNewJob} />}
      {terminal && snapshot.artifacts.length > 0 && <ArtifactPanel snapshot={snapshot} />}
      <StageLogs snapshot={snapshot} logs={logs} logsError={logsError} />
      <StageTimeline snapshot={snapshot} />
    </main>
  );
}
