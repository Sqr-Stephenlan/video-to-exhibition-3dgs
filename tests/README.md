# 测试目录

- `unit/`：无 GPU 依赖的最小单元测试。
- `integration/`：可在普通 CI 运行的小型端到端测试。
- `gpu/`：需要 CUDA 测试节点的测试；不应由 `.github/workflows/cpu-check.yml` 执行。

在 `unit/` 或 `integration/` 新增 `test_*.py` 前，请同时提交当前 `dev.sh bootstrap` 支持的依赖清单（`pyproject.toml` 或 `requirements.txt`）。CPU workflow 只发现并执行这两个目录，且在发现 CPU 测试但找不到依赖清单时失败，以避免不可复现的测试环境。若要采用仅含 `uv.lock`、`poetry.lock` 或 `Pipfile` 的环境，必须先扩展并验证项目入口脚本与 CI。

GPU 测试须遵守 [协作者及其 Agent 开发与测试规范](../docs/collaborator_agent_development_guide.md)，并使用 [GPU Test Report 模板](../docs/testing/GPU_TEST_REPORT_TEMPLATE.md)。
