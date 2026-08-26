# 本地运行排障手册

## 正常启动

1. 确认项目 Python 环境已准备好；若 `.venv` 缺失，先确认后再执行 `./dev.sh bootstrap`。
2. 安装 `requirements-web.txt`，不要把 FastAPI/Uvicorn 反向写入 `requirements-route.txt`。
3. 启动 `./dev.sh python -m scripts.viewer.api`，确认 `GET /api/v1/health` 返回 `status=ok`。
4. 在 `web` 目录启动 Vite，导入本地视频。

## 开发与验证边界

当前阶段不对 `dev.sh` 或其他 POSIX Bash 脚本做 Windows 适配。Windows 上的前端/API 运行用于验证上传、任务创建、状态轮询、阶段日志和进度展示；不要把 Windows worker 执行完整重建作为前端验收条件。

前端开发使用 Vitest、TypeScript 校验、Vite 构建以及后端 fixture/mock 验证，不启动真实的完整训练。页面只有在后端返回当前阶段的真实迭代遥测后才计算“当前阶段时间预估”，没有遥测时显示“暂不可估算”，不生成伪造进度。

如果必须验证真实脚本调用，请在本机 WSL 中启动/执行 pipeline，并优先使用 `--help`、短 smoke/plan 或已有 fixture；不要为界面联通性测试启动完整的 3DGS 训练。当前 Windows 转换器若直接尝试执行 `dev.sh`，可能因 Bash 文件不是 Windows 可执行文件而以 `WinError 193` 阻断，这是运行环境边界，不是前端状态轮询故障。

如果只需要查看已经完成的本地任务，不要重新导入视频或启动训练，可直接打开：

```text
http://127.0.0.1:5173/?job=<web-job-id>
```

页面会轮询任务快照和阶段日志；对于已完成训练但转换被阻断的任务，终态页面会保留训练日志、阶段时间线，并显示 `training-sample-ply` 下载入口。该样例是 LongSplat 原生训练 PLY，不等同于已通过交付验收的标准 viewer 模型。

Windows 若工具不在 `PATH`，在启动 API 的同一 PowerShell 会话中显式绑定：

```powershell
$env:VIDEO_TO_3DGS_FFMPEG = 'C:\path\to\ffmpeg.exe'
$env:VIDEO_TO_3DGS_FFPROBE = 'C:\path\to\ffprobe.exe'
$env:VIDEO_TO_3DGS_COLMAP = 'C:\path\to\colmap.exe'
$env:LONGSPLAT_BACKEND_PYTHON = 'D:\path\to\venv\Scripts\python.exe'
```

`VIDEO_TO_3DGS_COLMAP` 是 canonical pipeline 的必需依赖；只配置 ffprobe 只能完成上传探测，不能进入抽帧和相机重建。

Windows 的 CUDA LongSplat 环境还需安装 fork requirements 中的 `evo`、`trimesh`，并将 NumPy 固定在 `2.2.x`（例如 `numpy==2.2.6`）。当前锁定 fork 的可视化代码使用 NumPy 2.3+ 已移除的二进制 `np.fromstring` 行为；该兼容约束只作用于环境，不修改 LongSplat gitlink 或其工作树。

## 常见阻断

### `backend_dirty`

LongSplat checkout 有未提交修改。Web API 不会 reset、checkout、拉取或覆盖 third-party；保留原工作区，先由负责人决定如何形成干净且锁定的 backend。

### `backend_commit_mismatch` / `backend_submodule_mismatch`

root gitlink、`runner.LONGSPLAT_COMMIT`、实际 LongSplat checkout HEAD 或 direct gitlink 不一致。不要用当前分支名或本地猜测替代锁定 commit。

### `provider_identity_missing` / `gpu_preflight_failed`

工具提供方或 GPU/CUDA/tool 预检没有得到明确身份/通过结果。任务会阻断，不会生成伪造的 running/complete 状态。

### `worker_lost_after_restart`

API 重启时发现 queued/running job 没有可恢复的 worker。job 会标为 blocked，原 RunLedger 和 evidence 保留；第一版重新导入视频创建新 job。

### `technical_delivery_incomplete` / `artifact_invalid`

检查 `.runtime/longsplat-runs/<job-id>/run.json`、`published_ply.json`、technical delivery manifest 和 published PLY 的 SHA/属性。浏览器不显示内部绝对路径，诊断应在本机 evidence 中完成。

### SuperSplat 无法打开

这是预期的本地部署边界：外部站点通常无法读取本机 `localhost` 或文件路径。下载 PLY 后在 SuperSplat 中手动 Import；该动作不会自动改变 `supersplat_runtime_verified` 或 `manual_visual_review`。

## 证据保留

不要删除 `.runtime/longsplat-runs/<job-id>/`、job record、published receipt 或技术交付目录。失败/阻断任务不提供前端删除动作，便于复核原始 run identity 和阶段结果。
