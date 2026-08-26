import type { ObservedProgress as ObservedProgressValue } from "../types/job";

const trainingStages = new Set(["convergence-smoke-training", "formal-training"]);
const stageLabels: Record<string, string> = {
  conversion: "模型转换",
  "convergence-smoke-training": "诊断训练",
  "formal-training": "正式训练",
};

function labelForStage(stage: string): string {
  return stageLabels[stage] ?? stage;
}

export function isProgressForCurrentStage(
  progress: ObservedProgressValue,
  currentStageId: string | null | undefined,
  currentStageAttempt?: string | null,
): boolean {
  if (!currentStageId) {
    return false;
  }
  if (progress.stage) {
    return (
      progress.stage === currentStageId &&
      Boolean(progress.attempt) &&
      Boolean(currentStageAttempt) &&
      progress.attempt === currentStageAttempt
    );
  }
  return progress.kind === "conversion-iteration"
    ? currentStageId === "conversion"
    : trainingStages.has(currentStageId);
}

export function ObservedProgress({
  progress,
  currentStageId,
  currentStageAttempt,
}: {
  progress: ObservedProgressValue | null;
  currentStageId?: string | null;
  currentStageAttempt?: string | null;
}) {
  if (!progress) {
    return <p className="observed-progress muted">当前没有可恢复的迭代观测</p>;
  }
  const label = progress.kind === "conversion-iteration" ? "转换迭代" : "训练迭代";
  const isCurrent = isProgressForCurrentStage(progress, currentStageId, currentStageAttempt);
  const origin = progress.stage ? labelForStage(progress.stage) : label;
  return (
    <p className="observed-progress">
      {!isCurrent && `历史观测 · ${origin} · `}
      {label} {progress.iteration} / {progress.total}
    </p>
  );
}
