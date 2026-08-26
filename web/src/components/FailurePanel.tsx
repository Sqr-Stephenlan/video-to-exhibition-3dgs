import type { JobSnapshot } from "../types/job";

export function FailurePanel({ snapshot, onNewJob }: { snapshot: JobSnapshot; onNewJob?: () => void }) {
  if (!snapshot.error && snapshot.status !== "blocked" && snapshot.status !== "failed" && snapshot.status !== "stopped") {
    return null;
  }
  return (
    <section className="failure-card" role="alert">
      <h2>任务未完成</h2>
      {snapshot.error && <p>{snapshot.error.message}</p>}
      <p className="muted">请保留 evidence；第一版不提供通用重试，重新提交会创建新的 job。</p>
      {onNewJob && <button onClick={onNewJob}>重新导入视频</button>}
    </section>
  );
}
