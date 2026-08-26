import { estimateStageTime } from "../api/estimate";
import type { JobSnapshot } from "../types/job";

function formatRemaining(seconds: number): string {
  if (seconds < 60) {
    return "少于 1 分钟";
  }
  if (seconds < 3600) {
    return `约 ${Math.max(1, Math.ceil(seconds / 60))} 分钟`;
  }
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.ceil((seconds % 3600) / 60);
  return minutes > 0 ? `约 ${hours} 小时 ${minutes} 分钟` : `约 ${hours} 小时`;
}

export function StageTimeEstimate({ snapshot }: { snapshot: JobSnapshot }) {
  const estimate = estimateStageTime(snapshot);
  const title =
    estimate.state === "complete"
      ? "已完成"
      : estimate.state === "estimating"
        ? `预计剩余${formatRemaining(estimate.remainingSeconds)}`
        : estimate.state === "finishing"
          ? "当前阶段即将完成"
          : "暂不可估算";

  return (
    <div className="time-estimate" aria-label="时间预估">
      <span className="label">当前阶段时间预估</span>
      <strong>{title}</strong>
      <small>{estimate.basis}</small>
    </div>
  );
}
