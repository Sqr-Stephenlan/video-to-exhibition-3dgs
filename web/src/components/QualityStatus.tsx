import type { QualityFlags } from "../types/job";

function result(value: boolean): string {
  return value ? "通过" : "未通过";
}

export function QualityStatus({ quality }: { quality: QualityFlags }) {
  return (
    <section className="quality-card" aria-label="质量状态">
      <h2>交付状态</h2>
      <ul>
        <li>自动技术交付：{result(quality.accepted_by_automated_policy)}</li>
        <li>PLY Gaussian 属性：{result(quality.gaussian_schema_valid)}</li>
        <li>SuperSplat 格式兼容：{result(quality.supersplat_format_compatible)}</li>
        <li>SuperSplat 运行时：{quality.supersplat_runtime_verified ? "已验证" : "未验证"}</li>
        <li>人工验收：{quality.manual_visual_review ? "已完成" : "未完成"}</li>
      </ul>
      <p className="muted">自动技术交付不代表查看器运行时验证或人工视觉验收。</p>
    </section>
  );
}
