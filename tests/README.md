# 测试目录

- `unit/`：无 GPU 依赖的最小单元测试。
- `integration/`：可在普通 CI 运行的小型端到端测试。
- `gpu/`：需要 CUDA 测试节点的测试；不应由 `.github/workflows/cpu-check.yml` 执行。

在 `unit/` 或 `integration/` 新增 `test_*.py` 时：

- 同时提交 CPU workflow 当前支持的 `pyproject.toml` 或 `requirements.txt`。
- `dev.sh bootstrap` 在系统已安装 `uv` 时也能使用 `uv.lock`，但 CI 尚不支持仅含 `uv.lock`、`poetry.lock` 或 `Pipfile` 的环境；采用前须先扩展并验证项目脚本与 CI。
- CPU workflow 只发现并执行 `unit/` 和 `integration/`；发现测试但没有受支持依赖清单时，检查失败。

GPU 测试须遵守 [协作者及其 Agent 开发与测试规范](../docs/collaborator_agent_development_guide.md)，并使用 [GPU Test Report 模板](../docs/testing/GPU_TEST_REPORT_TEMPLATE.md)。
