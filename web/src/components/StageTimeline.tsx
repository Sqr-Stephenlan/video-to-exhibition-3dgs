import type { JobSnapshot, StageSnapshot } from "../types/job";
import { ObservedProgress } from "./ObservedProgress";

const stageLabels: Record<string, string> = {
  preflight: "环境预检",
  probe: "视频探测",
  frames: "帧提取",
  colmap: "相机解算",
  "camera-staging": "相机整理",
  "longsplat-input": "LongSplat 输入",
  "convergence-smoke-plan": "convergence1000 诊断计划",
  "convergence-smoke-training": "convergence1000 诊断训练",
  "convergence-smoke-render": "convergence1000 诊断渲染",
  "convergence-smoke-render-postcheck": "convergence1000 渲染检查",
  "automated-early-gate": "早期自动门禁",
  "formal-training": "正式训练",
  conversion: "模型转换",
  "converted-eval": "转换评估",
  "converted-eval-postprocess": "评估后处理",
  "automated-technical-delivery": "自动技术交付",
};

const statusLabels: Record<StageSnapshot["status"], string> = {
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

export function StageTimeline({ snapshot }: { snapshot: JobSnapshot }) {
  const current = snapshot.current_stage;
  const position = current
    ? "第 " + current.index + "/" + snapshot.stage_order.length + " 阶段"
    : snapshot.status === "complete"
      ? "流水线已完成"
      : "等待阶段状态";

  return (
    <section className="stage-panel" aria-label="重建阶段">
      <div className="stage-panel__header">
        <div>
          <p className="eyebrow">流水线状态</p>
          <h2>{position}</h2>
        </div>
        <ObservedProgress
          progress={snapshot.observed_progress}
          currentStageId={current?.id}
          currentStageAttempt={current?.attempt}
        />
      </div>
      <ol className="stage-list">
        {snapshot.stages.map((stage) => (
          <li className={"stage stage--" + stage.status} key={stage.id}>
            <span className="stage__index">{stage.index}</span>
            <span className="stage__body">
              <strong>{labelForStage(stage.id)}</strong>
              <span>{statusLabels[stage.status]}</span>
            </span>
            {stage.attempt && <small>{stage.attempt}</small>}
          </li>
        ))}
      </ol>
    </section>
  );
}
