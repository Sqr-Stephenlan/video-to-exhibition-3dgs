import type { ChangeEvent, DragEvent } from "react";

interface UploadDropzoneProps {
  file: File | null;
  error: string | null;
  onFile: (file: File | null) => void;
}

export function UploadDropzone({ file, error, onFile }: UploadDropzoneProps) {
  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    onFile(event.target.files?.[0] ?? null);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    onFile(event.dataTransfer.files?.[0] ?? null);
  }

  return (
    <div
      className="upload-dropzone"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
    >
      <label htmlFor="video-file">视频文件</label>
      <input
        id="video-file"
        type="file"
        accept="video/*,.avi,.m4v,.mkv,.mov,.mp4,.webm"
        onChange={handleChange}
      />
      <p className="hint">选择或拖入一个本地视频；第一版不接受网页 URL 或多视频融合。</p>
      {file && (
        <p className="selected-file" aria-live="polite">
          已选择：{file.name}（{Math.ceil(file.size / 1024)} KB）
        </p>
      )}
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
