# 可视化界面分支说明

本分支：`feature/web-visualization`

## 目标

为本地单用户 3D 高斯泼溅重建提供一条可观察的操作链路：

```text
导入视频 → 创建并启动 job → 查看阶段状态 → 查看实时/最近日志与进度 → 下载模型并交接 SuperSplat
```

浏览器不负责解析 CLI 输出，也不伪造总体百分比。后端从受控的 RunLedger、阶段日志和训练遥测生成 `web-job-snapshot-v1`，前端据此展示当前阶段、attempt、迭代进度和当前阶段时间预估。

## 组成

- `web/`：React + Vite 前端；包含上传页、运行页、终态页、阶段时间线、日志、进度、ETA 和 artifact 入口。
- `scripts/viewer/api/`：FastAPI 本地接口、单 worker FIFO 调度、快照、日志和 artifact 读取。
- `scripts/viewer/`：PLY 合同、导出与 annotation 支持。
- `tests/unit/viewer/`、`tests/integration/viewer/`：后端合同、快照、任务管理和 HTTP 接口测试。
- `web/tests/`：前端组件、API 客户端、轮询、ETA 和状态展示测试。

## 本地启动

在项目规定的 Python 环境中安装 API 依赖并启动后端：

```bash
./dev.sh pip install -r requirements-web.txt
./dev.sh python -m scripts.viewer.api
```

另开终端启动前端：

```bash
cd web
npm ci
npm run dev
```

默认 API 为 `http://127.0.0.1:8000`，Vite 会把 `/api` 代理到该地址。前端页面通常为 `http://127.0.0.1:5173`；如果端口被占用，使用 Vite 输出的实际端口。

## API 链路

| 动作 | 接口 | 作用 |
| --- | --- | --- |
| 健康检查 | `GET /api/v1/health` | 确认 API 在线 |
| 导入并启动 | `POST /api/v1/jobs` | 保存视频、ffprobe 校验、创建 job 并启动单 worker |
| 任务快照 | `GET /api/v1/jobs/{job_id}` | 返回任务状态、当前阶段、attempt、进度和错误 |
| 阶段日志 | `GET /api/v1/jobs/{job_id}/logs` | 返回 stdout/stderr 及日志所属阶段 |
| 下载产物 | `GET /api/v1/jobs/{job_id}/artifacts/{artifact_id}` | 仅允许快照声明的 artifact |

当前版本没有取消、通用重试或人工验收写回接口。失败任务重新导入会创建新的 job，原 evidence 保留。

## 验证范围

前端开发阶段使用 mock/fixture、Vitest、TypeScript 检查和 Vite 构建验证，不启动真实完整训练。时间预估只有在存在当前阶段且未过期的真实迭代遥测时才显示。

若需要验证实际脚本调用，应在本机 WSL 中使用短 smoke、plan 或已有 fixture。当前不对 `dev.sh` 或其它 POSIX Bash 脚本做 Windows 适配；Windows 转换步骤遇到 `WinError 193` 属于运行环境边界，不是前端轮询故障。

## 验证命令

```bash
cd web
npm test -- --run
npm run build

cd ..
./dev.sh pytest tests/unit/viewer tests/integration/viewer tests/unit/test_terminal_progress.py -q
```
