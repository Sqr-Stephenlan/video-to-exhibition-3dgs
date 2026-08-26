import { downloadArtifact } from "../api/client";
import type { JobSnapshot } from "../types/job";

const labels: Record<string, string> = {
  "published-ply": "下载 PLY",
  "training-sample-ply": "下载训练样例 PLY",
  "published-ply-receipt": "下载发布 receipt",
  "technical-delivery-manifest": "下载技术交付 manifest",
  provenance: "下载 provenance",
  "technical-report": "下载技术报告",
  "comparison-sheet": "下载评估图",
  checksums: "下载校验和",
};

export function ArtifactPanel({ snapshot }: { snapshot: JobSnapshot }) {
  const sample = snapshot.artifacts.filter((artifact) => artifact.id === "training-sample-ply");
  const delivery = snapshot.artifacts.filter((artifact) => artifact.id !== "training-sample-ply");

  function artifactList(artifacts: JobSnapshot["artifacts"]) {
    return (
      <ul className="artifact-list">
        {artifacts.map((artifact) => {
          let href = "#";
          try {
            href = downloadArtifact(snapshot, artifact.id);
          } catch {
            href = "#";
          }
          return (
            <li key={artifact.id}>
              <a href={href} download>
                {labels[artifact.id] ?? `下载 ${artifact.format}`}
              </a>
              <small>{artifact.size_bytes.toLocaleString()} bytes · SHA {artifact.sha256.slice(0, 12)}</small>
            </li>
          );
        })}
      </ul>
    );
  }

  return (
    <>
      {delivery.length > 0 && (
        <section className="artifact-card" aria-label="交付物">
          <h2>交付物</h2>
          {artifactList(delivery)}
        </section>
      )}
      {sample.length > 0 && (
        <section className="artifact-card artifact-card--sample" aria-label="训练样例（非交付）">
          <h2>训练样例（非交付）</h2>
          <p className="muted">LongSplat 原生训练 PLY，仅用于查看训练产物，不是 viewer-ready 交付模型，也不代表交付验收通过。</p>
          {artifactList(sample)}
        </section>
      )}
      {snapshot.artifacts.length === 0 && (
        <section className="artifact-card" aria-label="交付物">
          <h2>交付物</h2>
          <p className="muted">技术交付完成后，交付物会出现在这里。</p>
        </section>
      )}
    </>
  );
}
