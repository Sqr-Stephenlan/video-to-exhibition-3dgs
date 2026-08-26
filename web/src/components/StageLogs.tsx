import { useEffect, useRef, useState } from "react";

import type { JobLogs, JobSnapshot, StageLogSnapshot } from "../types/job";

const stageLabels: Record<string, string> = {
  preflight: "环境预检",
  probe: "视频探测",
  frames: "帧提取",
  colmap: "相机解算",
  "camera-staging": "相机整理",
  "longsplat-input": "LongSplat 输入",
  "convergence-smoke-training": "诊断训练",
  "convergence-smoke-render": "诊断渲染",
  "formal-training": "正式训练",
  "native-render": "正式渲染",
  conversion: "模型转换",
  "converted-eval": "转换评估",
  "automated-technical-delivery": "自动技术交付",
};

const statusLabels: Record<StageLogSnapshot["status"], string> = {
  queued: "等待中",
  running: "进行中",
  passed: "已完成",
  blocked: "已阻塞",
  failed: "失败",
  stopped: "已停止",
  skipped: "已跳过",
};

function labelForStage(id: string): string {
  return stageLabels[id] ?? id;
}

function preferredStage(snapshot: JobSnapshot, logs: JobLogs | null): string {
  const candidates = [snapshot.current_stage?.id, snapshot.error?.stage];
  for (const candidate of candidates) {
    if (candidate && logs?.stages.some((stage) => stage.id === candidate)) {
      return candidate;
    }
  }
  return logs?.stages.at(-1)?.id ?? "";
}

function OutputBlock({ title, lines }: { title: string; lines: string[] }) {
  return (
    <div className="log-stream">
      <h3>{title}</h3>
      <pre>{lines.length > 0 ? lines.join("\n") : "（无输出）"}</pre>
    </div>
  );
}

export function StageLogs({
  snapshot,
  logs,
  logsError = null,
}: {
  snapshot: JobSnapshot;
  logs: JobLogs | null;
  logsError?: string | null;
}) {
  const [selectedStage, setSelectedStage] = useState("");
  const autoSelectedStage = useRef<string | null>(null);
  const currentStageId = snapshot.current_stage?.id ?? null;

  useEffect(() => {
    const available = logs?.stages ?? [];
    if (
      currentStageId &&
      available.some((stage) => stage.id === currentStageId) &&
      autoSelectedStage.current !== currentStageId
    ) {
      setSelectedStage(currentStageId);
      autoSelectedStage.current = currentStageId;
      return;
    }
    if (!available.some((stage) => stage.id === selectedStage)) {
      setSelectedStage(preferredStage(snapshot, logs));
    }
  }, [currentStageId, logs, selectedStage, snapshot]);

  const selected = logs?.stages.find((stage) => stage.id === selectedStage) ?? null;

  return (
    <section className="logs-card" aria-label="阶段日志">
      <div className="logs-card__header">
        <div>
          <p className="eyebrow">OBSERVABILITY</p>
          <h2>阶段日志</h2>
        </div>
        {logs?.stages.length ? (
          <label className="log-stage-picker">
            <span>查看阶段</span>
            <select
              aria-label="日志阶段"
              value={selectedStage}
              onChange={(event) => setSelectedStage(event.target.value)}
            >
              {logs.stages.map((stage) => (
                <option key={stage.id} value={stage.id}>
                  {labelForStage(stage.id)} · {statusLabels[stage.status]}
                </option>
              ))}
            </select>
          </label>
        ) : null}
      </div>
      {logsError && (
        <p className="log-stale" role="alert">
          日志暂时未更新：{logsError}。当前显示最后一份日志快照。
        </p>
      )}
      {!logs ? (
        <p className="muted">等待后端日志快照…</p>
      ) : !logs.stages.length ? (
        <p className="muted">当前阶段尚未产生日志。</p>
      ) : selected ? (
        <>
          <p className="log-meta">
            {labelForStage(selected.id)} · {selected.attempt ?? "未分配 attempt"}
            {selected.updated_at ? ` · 更新于 ${selected.updated_at}` : ""}
          </p>
          <div className="log-grid" aria-live="polite">
            <OutputBlock title="标准输出 stdout" lines={selected.stdout} />
            <OutputBlock title="错误输出 stderr" lines={selected.stderr} />
          </div>
        </>
      ) : (
        <p className="muted">选择一个有日志的阶段。</p>
      )}
    </section>
  );
}
