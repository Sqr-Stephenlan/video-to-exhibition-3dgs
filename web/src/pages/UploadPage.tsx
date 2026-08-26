import { useState, type FormEvent } from "react";

import { createJob } from "../api/client";
import { UploadDropzone } from "../components/UploadDropzone";
import type { CreateJobResponse } from "../types/job";

interface UploadPageProps {
  submitJob?: typeof createJob;
  onCreated: (response: CreateJobResponse) => void;
  maxFileBytes?: number;
}

const DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024 * 1024;

function formatBytes(value: number): string {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(0)} MiB`;
  return `${Math.max(1, Math.ceil(value / 1024))} KiB`;
}

function defaultName(file: File): string {
  return file.name.replace(/\.[^.]+$/, "") || "video";
}

export function UploadPage({
  submitJob = createJob,
  onCreated,
  maxFileBytes = DEFAULT_MAX_FILE_BYTES,
}: UploadPageProps) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function handleFile(nextFile: File | null) {
    setError(null);
    if (!nextFile) {
      setFile(null);
      return;
    }
    if (!nextFile.type.startsWith("video/") && !/\.(avi|m4v|mkv|mov|mp4|webm)$/i.test(nextFile.name)) {
      setFile(null);
      setError("请选择视频文件");
      return;
    }
    if (nextFile.size === 0) {
      setFile(null);
      setError("视频文件不能为空");
      return;
    }
    if (nextFile.size > maxFileBytes) {
      setFile(null);
      setError(`视频超过本地上传限制（${formatBytes(maxFileBytes)}）`);
      return;
    }
    setFile(nextFile);
    setName((current) => current || defaultName(nextFile));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setError("请先选择视频文件");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const response = await submitJob(file, name.trim() || defaultName(file));
      onCreated(response);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务提交失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="page upload-page">
      <section className="hero-card">
        <p className="eyebrow">VIDEO → 3D GAUSSIAN SPLATTING</p>
        <h1>把一段视频变成可查看的 3D 高斯模型</h1>
        <p className="lede">
          页面会记录真实的 LongSplat 阶段和技术交付证据。最终模型下载后可导入外部 SuperSplat 查看。
        </p>
      </section>
      <form className="card upload-form" onSubmit={handleSubmit}>
        <UploadDropzone file={file} error={error} onFile={handleFile} />
        <label htmlFor="delivery-name">交付名称（可选）</label>
        <input
          id="delivery-name"
          value={name}
          onChange={(event) => setName(event.target.value)}
          placeholder="例如：gallery"
          maxLength={120}
        />
        <div className="route-summary">
          <strong>固定生产路线</strong>
          <span>single-convergence1000-v1 · depth disabled · RGB-only fixed pose</span>
        </div>
        <button type="submit" disabled={!file || submitting}>
          {submitting ? "正在提交…" : "开始生成 3D 高斯模型"}
        </button>
      </form>
    </main>
  );
}
