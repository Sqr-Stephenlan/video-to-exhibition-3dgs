# Web API v1

API 默认由 `scripts.viewer.api` 提供，绑定本机 `127.0.0.1`。

## 健康检查

`GET /api/v1/health`

返回 `web-health-v1` 和 `status=ok`。

## 创建任务

`POST /api/v1/jobs`，使用 `multipart/form-data`：

- `video`：本地视频文件；
- `name`：可选的显示/交付名称。

返回 `202` 和 `web-job-v1`。服务端会生成 job/run ID，保存受控上传文件，计算 SHA-256，执行 ffprobe 检查，并将任务放入单 worker FIFO。预检失败时响应中的状态为 `blocked`，同时保留任务记录和错误码。

## 任务快照

`GET /api/v1/jobs/{job_id}`

返回 `web-job-snapshot-v1`，包含固定 `route`、通过预检后才有的 `backend`、真实 stage/attempt、合法的 `observed_progress`、质量标记、下一步动作和错误信息。浏览器不得据此推导总体百分比或 ETA。

终态为：`complete`、`blocked`、`failed`、`stopped`。如果 RunLedger 声称完成但缺少 published PLY、receipt、Gaussian 属性或自动技术交付证据，Web snapshot 会 fail-closed 为 `failed`，不会暴露模型下载入口。

## 阶段日志

`GET /api/v1/jobs/{job_id}/logs?stage={stage_id}&tail={n}`

返回 `web-job-logs-v1`。不传 `stage` 时按任务配置返回已有日志的阶段；`tail` 默认为 120，允许范围为 1–400。每个阶段同时返回 worker 的 `stdout`、`stderr`、最新 attempt、状态和更新时间。API 会读取阶段目录及其 `executor/` 下的日志，并限制读取范围和字节数；前端不需要解析 CLI 输出即可显示训练进度与错误。

阶段快照中的 `observed_progress` 是后端从受控 telemetry 读取的观测值，不代表模型质量或总体 ETA。训练已完成但后处理阻断时，日志仍然可查询。

## Artifact 下载

`GET /api/v1/jobs/{job_id}/artifacts/{artifact_id}`

artifact ID 必须先由快照声明。当前 allowlist 包含：

- `published-ply`
- `published-ply-receipt`
- `training-sample-ply`（仅在正式训练通过且任务明确阻断于 `conversion` 时提供的 LongSplat 原生样例，不代表交付验收通过）
- `technical-delivery-manifest`
- `provenance`
- `technical-report`
- `comparison-sheet`
- `checksums`

每次解析时都会重复检查 regular file、路径 containment、symlink、SHA/size；`published-ply` 还要重复通过 Gaussian PLY header/property 校验。API 不接受 `path` 查询参数、不返回任意本地路径、不执行目录打包下载。

`training-sample-ply` 只经过同样的本地路径、regular-file、symlink、哈希和大小保护；它不会在转换仍运行、任务尚未阻断或阻断于其他阶段时暴露。LongSplat 原生 PLY 的字段布局与交付用标准 Gaussian PLY 不同，因此不会被标记为 viewer-ready 交付模型。

错误返回统一为 `web-api-error-v1`，包括 `code`、`message` 和可选 `job_id`。前端只使用错误码/消息，不解析 CLI stdout。

## 不支持的动作

当前没有 DELETE、cancel、通用 retry 或修改人工验收状态的接口。取消涉及 worker/GPU/子进程清理，重试涉及同一 run 恢复还是新 run 的审计语义，均留给后续版本。
