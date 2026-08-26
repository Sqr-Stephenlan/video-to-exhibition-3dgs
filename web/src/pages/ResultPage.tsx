import { buildSuperSplatEditorUrl } from "../viewer/ExternalViewerLink";
import { ArtifactPanel } from "../components/ArtifactPanel";
import { QualityStatus } from "../components/QualityStatus";
import { StageLogs } from "../components/StageLogs";
import { StageTimeline } from "../components/StageTimeline";
import type { JobLogs, JobSnapshot } from "../types/job";

export function ResultPage({
  snapshot,
  publicBaseUrl,
  logs = null,
  logsError = null,
}: {
  snapshot: JobSnapshot;
  publicBaseUrl?: string;
  logs?: JobLogs | null;
  logsError?: string | null;
}) {
  const model = snapshot.artifacts.find((artifact) => artifact.id === "published-ply");
  const viewerUrl = model
    ? buildSuperSplatEditorUrl({
        artifactUrl: model.download_url,
        publicBaseUrl,
      })
    : null;

  return (
    <main className="page result-page">
      <section className="page-heading">
        <p className="eyebrow">TECHNICAL DELIVERY COMPLETE</p>
        <h1>模型已生成</h1>
        <p>{snapshot.next_action}</p>
      </section>
      <QualityStatus quality={snapshot.quality} />
      <section className="card viewer-card" aria-label="SuperSplat 查看器交接">
        <h2>查看 3D 高斯模型</h2>
        {viewerUrl ? (
          <a className="primary-link" href={viewerUrl} target="_blank" rel="noreferrer">
            在 SuperSplat 中打开
          </a>
        ) : (
          <>
            <p>下载后导入 SuperSplat</p>
            <ol className="import-steps">
              <li>下载上方的 PLY 交付物。</li>
              <li>打开 SuperSplat，将 PLY 拖入或使用 Import。</li>
              <li>查看器加载状态与人工视觉判断不会回写本任务。</li>
            </ol>
          </>
        )}
        <p className="muted">外部查看器打开或导入成功不会自动写入人工验收状态。</p>
      </section>
      <ArtifactPanel snapshot={snapshot} />
      <StageLogs snapshot={snapshot} logs={logs} logsError={logsError} />
      <StageTimeline snapshot={snapshot} />
    </main>
  );
}
