# 测试节点安全

self-hosted CUDA runner 的权限、workflow、Secrets、Artifact 和运行目录约束以 [协作者及其 Agent 开发与测试规范](../collaborator_agent_development_guide.md#9-cuda-测试节点与-artifact-安全) 为准。

修改 `.github/workflows/**`、runner 启动脚本、环境安装逻辑、依赖锁文件或第三方 Action 前，必须先取得项目负责人审查；未经批准不得在 GPU runner 上执行。
