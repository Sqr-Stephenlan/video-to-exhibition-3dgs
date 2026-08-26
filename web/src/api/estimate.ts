import type { JobSnapshot, ObservedProgress } from "../types/job";

export type StageTimeEstimate =
  | { state: "complete"; remainingSeconds: 0; basis: string }
  | { state: "finishing"; remainingSeconds: null; basis: string }
  | { state: "estimating"; remainingSeconds: number; basis: string }
  | { state: "unavailable"; remainingSeconds: null; basis: string };

const trainingStages = new Set([
  "convergence-smoke-training",
  "formal-training",
]);
const maxProgressAgeSeconds = 120;

function progressBelongsToStage(stageId: string, progress: ObservedProgress): boolean {
  if (progress.stage !== stageId) {
    return false;
  }
  if (progress.kind === "conversion-iteration") {
    return stageId === "conversion";
  }
  return trainingStages.has(stageId);
}

export function estimateStageTime(snapshot: JobSnapshot, now = Date.now()): StageTimeEstimate {
  if (snapshot.status === "complete") {
    return { state: "complete", remainingSeconds: 0, basis: "流水线已完成" };
  }
  if (snapshot.status !== "running") {
    return {
      state: "unavailable",
      remainingSeconds: null,
      basis: "任务尚未运行或已停止，暂不提供时间预估",
    };
  }

  const current = snapshot.current_stage;
  const progress = snapshot.observed_progress;
  if (
    !current ||
    current.status !== "running" ||
    !progress ||
    !progressBelongsToStage(current.id, progress) ||
    !current.attempt ||
    progress.attempt !== current.attempt
  ) {
    return {
      state: "unavailable",
      remainingSeconds: null,
      basis: "等待当前阶段产生匹配的迭代进度日志",
    };
  }

  const startedAt = current.started_at ? Date.parse(current.started_at) : Number.NaN;
  const observedAt = Date.parse(progress.observed_at);
  const elapsedSeconds = (now - startedAt) / 1000;
  const observedElapsedSeconds = (observedAt - startedAt) / 1000;
  const progressAgeSeconds = (now - observedAt) / 1000;
  if (
    !Number.isFinite(elapsedSeconds) ||
    !Number.isFinite(observedElapsedSeconds) ||
    !Number.isFinite(progressAgeSeconds) ||
    elapsedSeconds < 1 ||
    observedElapsedSeconds < 1 ||
    progressAgeSeconds < 0 ||
    progressAgeSeconds > maxProgressAgeSeconds
  ) {
    return {
      state: "unavailable",
      remainingSeconds: null,
      basis: "最近一次迭代遥测过旧或时间不足，暂不提供预估",
    };
  }

  if (progress.iteration >= progress.total) {
    return {
      state: "finishing",
      remainingSeconds: null,
      basis: "当前阶段迭代已完成，等待阶段收尾",
    };
  }

  const estimatedTotalSeconds = (observedElapsedSeconds * progress.total) / progress.iteration;
  return {
    state: "estimating",
    remainingSeconds: Math.max(0, Math.round(estimatedTotalSeconds - elapsedSeconds)),
    basis: "基于当前阶段已观测到的迭代速度；不包含后续阶段耗时",
  };
}
