# 展厅 3DGS 项目：协作者及其 Agent 开发与测试规范

> 版本：v0.2
>
> 状态：团队协作基线；由项目负责人确认后生效
> 适用对象：CUDA/GPU 协作者、协作者使用的开发或测试 Agent、项目负责人及其 Agent

## 1. 目的、适用范围与原则

本项目由项目负责人负责需求、架构、代码所有权和最终合并；CUDA 协作者的设备是受控测试节点，负责 NVIDIA/CUDA 环境验证、GPU 测试和具名人工视觉验收。协作者及其 Agent 的职责是让每次测试**可复现、可审计、可定位**，而不是以临时修改、截图或口头结论代替验证。

本规范适用于所有影响仓库执行行为、数据处理、GPU 调用链、训练、重建、导出或展示质量的 PR。纯文档或非执行性改动可按第 6 节降低测试等级，但仍需遵守分支、审查和资产边界。

1. **结果绑定不可变输入。** 测试结论必须绑定完整 commit SHA、配置指纹、数据版本、依赖环境、命令、随机种子和产物位置。
2. **Draft 用于工作中的协作，Ready 用于合并前的正式门槛。** Draft PR 可以讨论、预审和测试，但不应被当作最终批准。
3. **GPU 节点只执行明确授权的工作。** 不向任意仓库、任意 workflow 或外部 fork PR 开放协作者电脑。
4. **失败必须完整保留并分类。** 不得静默跳过、替换输入、修改阈值、反复重试直至偶然成功，或只报告成功的一次。
5. **人机责任分离。** Agent 可以执行自动化检查、整理视觉证据和标记疑点；只有具名协作者可以完成人工视觉验收。
6. **大资产不进入 Git。** 视频、帧、深度图、训练输出、模型、渲染结果和日志遵守仓库 `.gitignore`，通过受控 Artifact 或团队存储共享。

## 2. 角色、权限与边界

| 角色 | 负责事项 | 未经明确授权不得做的事 |
|---|---|---|
| 项目负责人及其 Agent | 任务拆解、架构与代码实现、测试等级、测试包、PR 审核、合并决策、日志分析 | 在没有对应 GPU 结果前宣称 GPU 训练、显存或重建质量已验证 |
| CUDA 协作者 | 维护约定环境、按测试包执行、回传报告与产物、具名视觉验收 | 直接推送他人分支、直接向 `main` 推送、私自改变测试标准或替换输入 |
| 协作者的 Agent | 环境诊断、按给定命令执行、收集日志/指标、生成自动化报告、给出诊断建议 | 读取无关个人文件或凭据、执行破坏性 Git 操作、擅自下载/替换大模型、绕过失败检查、代替人类签署视觉验收 |
| GitHub CI / self-hosted runner | 执行自动检查并保存受控产物 | 代替人工判断相机轨迹、场景方向或 Gaussian 质量 |

协作者参与修复有两种模式：

- **负责人主导（默认）**：协作者只提交测试报告、诊断和建议；项目负责人或其 Agent 负责改代码。
- **授权修复**：项目负责人在 PR/Issue 中明确授权后，协作者从待测分支创建独立 `fix/*` 分支，向原功能分支发 PR。原功能 PR 必须链接该修复 PR，并重新按本规范测试。

## 3. 仓库、第三方依赖与资产边界

本仓库是总控编排层，维护文档、脚本、配置、数据清单、实验记录和展示入口。分支职责与阶段顺序以 [分支与任务说明](branch_task_plan.md) 为准。

- 从 `main` 派生 `docs/*`、`feature/*` 或 `research/*` 分支；`main` 只保存稳定、已确认且可复现的成果。
- `third_party/` 的开源项目是外部执行后端。第三方源码修改应在其自身仓库/分支完成，主仓库只记录所需补丁、版本和运行参数。
- 每一项第三方依赖记录至少包含：仓库 URL、完整 commit SHA、取得方式（submodule/package/container）、许可证、安装命令、补丁路径与 SHA-256、本地修改说明。不得以“latest”“main”或可移动标签作为唯一版本标识。
- `data/` 保存原始视频、分段、帧、深度、掩码和 manifest；`outputs/` 保存重建、splat、压缩、渲染和 Web 产物。这些生成资产默认不受 Git 跟踪。
- 模型权重、checkpoint、`.ply`、`.splat`、视频、渲染图和日志均不直接提交。报告仅记录受控存储位置、校验值、生成 SHA 与保留期限。

### 3.1 跨平台执行入口

测试包必须提供**适合测试节点平台的完整命令**，而不是假定所有设备都能运行同一 Shell 脚本。

- 当前仓库提供 `./dev.sh`；它适用于 macOS、Linux 或已配置的 WSL/Git Bash 环境。
- 原生 Windows PowerShell 节点在 `dev.ps1` 或 Python 统一入口尚未实现前，必须使用测试包中明确给出的 PowerShell/Conda/环境命令，并把该差异写入报告。不得假装仓库已存在 `dev.ps1`。
- 任何未包含于待测 SHA 的脚本、配置或入口都不能作为通过依据。
- 后续目标是由一个跨平台逻辑入口（例如 `tools/dev.py`）承载 `doctor`、环境检查和 smoke test，再由 `dev.sh`/`dev.ps1` 转发；该目标在实现前不是现有承诺。

## 4. 分支、Draft PR、正式审核与平台门禁

### 4.1 PR 生命周期

```text
任务与测试等级确认
  ↓
从 main 创建 feature/* / research/* / docs/* 分支
  ↓
创建 Draft PR：持续开发、CI、预审、GPU 测试与设计讨论
  ↓
负责人确认完成条件后，转为 Ready for review
  ↓
正式 Review、required checks、必要审批
  ↓
负责人合并 main
```

### 4.2 Draft 与 Ready 的准确边界

Draft PR 可以查看 diff、提出行级评论、开展技术预审、运行 CI 和执行 GPU 测试；团队应把这阶段的意见视为**预审意见**，默认不构成最终合并批准。GitHub 的 Draft PR 不能合并，且不会自动请求 code owner 审查；转为 Ready 后才会触发相应的正式审查请求。[GitHub Draft PR 文档](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)

PR 转为 `Ready for review` 后，才按仓库的 ruleset/branch protection 发起正式 Review。最终有效批准应以**当前 Ready 状态下**满足所需审批、required status checks 和分支同步要求的记录为准，而非 Draft 阶段的评论或口头同意。

“只有项目负责人可以转为 Ready”属于团队流程约定，不是天然的平台权限控制：PR 作者或拥有写权限的协作者通常可以改变 PR 阶段。若必须技术强制，应在 `main` 上设置 ruleset/branch protection，例如 required reviews、dismiss stale approvals、required status checks、限制 bypass、禁止 force push 与删除分支。[GitHub ruleset 文档](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)

标签（例如 `gpu-validated` 或 `ready-owner-approved`）只用于流程可见性，**标签本身不能阻止合并**。若需要负责人门禁，应由可信 GitHub App/Workflow 生成受 ruleset 约束的 required check（例如 `owner-gate`），并限定允许写入该 check 的来源。

### 4.3 精确检出待测 commit

每个测试包必须指定完整的 40 位 commit SHA。分支名只用于定位历史，测试必须在该 SHA 的 detached HEAD 上执行，防止测试期间功能分支继续 push 后悄然测试到其他提交。

macOS、Linux、WSL 或 Git Bash 的参考流程：

```bash
git fetch --prune origin
test -z "$(git status --porcelain)" || { echo "工作区不干净"; exit 1; }
EXPECTED_SHA="$(git rev-parse --verify '<requested-40-char-sha>^{commit}')"
git switch --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA"
```

PowerShell 的等价流程：

```powershell
git fetch --prune origin
if (git status --porcelain) { throw '工作区不干净' }
$expectedSha = (git rev-parse --verify '<requested-40-char-sha>^{commit}').Trim()
git switch --detach $expectedSha
if ((git rev-parse HEAD).Trim() -ne $expectedSha) { throw 'HEAD 与请求 SHA 不一致' }
```

若目标 SHA 无法在本地解析、工作区不干净，或最终 `HEAD` 不等于测试包 SHA，必须停止并报告；不得改为测试“当前分支最新提交”。测试结束后，可按本机约定切回工作分支，但报告中的测试对象始终是 detached 的 SHA。

## 5. 测试包与执行交接

项目负责人发起 GPU 测试时，应在 Draft PR、Issue 或团队约定渠道提供一个可冻结的“测试包”。信息不完整时，协作者和 Agent 必须停止并请求补充，不得猜测。

| 必填项 | 说明 |
|---|---|
| PR、分支、完整 commit SHA | SHA 是唯一测试对象；分支仅为定位信息 |
| 测试等级 | 按第 6 节填写 `L0`–`L3`，由项目负责人确认 |
| 测试目的与范围 | 例如“验证 LongSplat 自定义视频训练可启动”，并列出不在本次范围内的项 |
| 执行平台与完整命令 | Windows PowerShell、WSL、Linux 等；含工作目录、环境变量和配置路径 |
| 输入数据与配置指纹 | 样例/受控位置、数据版本或校验值、配置文件及其 SHA-256 |
| 预期结果 | 退出码、必须产物、关键指标范围、人工验收项目、允许的已知限制 |
| 资源与网络限制 | 最大时长、GPU/显存、磁盘预算、是否允许下载及允许的来源 |
| Run ID 与可复现参数 | Run ID、随机种子、deterministic 设置、最大重试次数和重试条件 |
| 结果回传与保留 | PR comment、Artifact/共享存储位置、保留期限和访问范围 |

协作者接收测试包后按此顺序执行：

1. 用第 4.3 节的流程确认 detached `HEAD` 与指定 SHA 完全一致，并记录 `git status --porcelain` 的结果。
2. 按测试包的跨平台命令采集环境：OS、GPU、驱动、CUDA runtime、Python、PyTorch/CUDA 可用性和关键第三方版本。
3. 严格执行指定命令；不得替换数据、模型、配置、依赖版本或阈值。需要适配时先停止并取得书面确认。
4. 保存 stdout、stderr、退出码、开始/结束时间、峰值显存（如可取得）和所有要求产物。
5. 由 Agent 形成自动化执行报告；由具名协作者完成或明确暂缓人工视觉验收。

## 6. 测试等级与 Ready 门槛

PR 作者必须在 PR 描述中声明测试等级；项目负责人负责确认或调整。升级等级时，原有测试证据不足，应按更高等级补测。

| 等级 | 典型改动 | 转 Ready 前最低证据 |
|---|---|---|
| `L0` | 文档、注释、非执行性配置 | 文档/链接检查；GPU 可豁免 |
| `L1` | CPU 工具、数据清单、CLI、视频预处理 | 静态检查、相关单元测试、CPU smoke test |
| `L2` | GPU 调用链、依赖、CUDA 环境、模型导出接口 | `L1` 证据 + GPU smoke test |
| `L3` | 训练核心、相机估计、深度先验、跨段配准、3DGS 重建质量 | `L2` 证据 + 规定范围的 GPU 测试 + 具名人工视觉验收 |

对 `L2`/`L3` PR，若启用严格 required checks，最终有效测试应针对已经包含当前 `main` 的功能分支 HEAD，或由受控 merge queue/预合并检查生成的等效合并结果。`main` 变化后，不得把旧基线上的报告当作最终合并证据。[GitHub 对严格 status checks 的说明](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)

当前仓库尚未纳入统一的 `tests/`、跨平台 `doctor` 或 `run_smoke_test` 入口。在这些能力落地前，每个测试包必须列出真实存在于目标 SHA 的命令；不允许把规划中的命令写成已完成的检查。

## 7. 测试有效性、失效与重试规则

### 7.1 有效性

一份测试报告只对以下组合有效：完整 commit SHA、配置指纹、数据版本/校验值、依赖环境、完整命令、随机种子、测试等级和 Run ID。报告必须列出 base branch 的 SHA；对于 `L2`/`L3` 的最终测试，还应说明其是否已包含当时的 `main`。

### 7.2 失效规则

出现下列任一情况时，原测试报告自动成为历史证据，不再满足合并门槛，除非项目负责人书面判定该变化与测试范围无关：

- PR 新增 commit、rebase 或 force push；
- 测试配置、输入数据、随机种子、依赖锁文件、模型版本或环境关键版本变化；
- 合入或更新 base branch 后，功能分支的 merge base/预合并结果变化；
- 产物校验、测试报告或 Artifact 无法再验证；
- 原测试发现非确定性，且尚未按本节记录稳定性结论。

负责人应在 PR 中明确“完整复测”或“限定增量复测”的范围和理由。若 ruleset 启用了 stale approvals 的失效策略，新提交或 base 更新也会使批准失效；测试报告的有效性规则不依赖平台是否自动执行该操作。

### 7.3 重试规则

不得无记录地重复运行直至偶然通过。每次尝试必须保留独立 `Run ID`、`Attempt`、随机种子、命令、日志和结果，并链接上一次运行。

- 仅因 Runner 离线、网络短暂中断、Artifact 上传失败等**基础设施瞬时故障**，允许在参数完全不变时原样重试一次。
- 代码、数据、依赖、配置、模型、种子或资源限制变化时，视为新的测试运行，而不是“重试”。
- 同一组合多次结果不一致时，结论应为“非确定性/待调查”，不得只提交成功的一次。

## 8. 自动化执行与人工视觉验收

### 8.1 自动化执行

Agent 可以执行命令、记录退出码与指标、生成预览图、整理日志、标记疑似异常并提出诊断建议。其结论只能是“自动检查通过/失败/未完成”，不能替代人的视觉质量判断。

### 8.2 具名人工视觉验收

具名 CUDA 协作者应使用约定的 Viewer（如 SuperSplat）或预览工具完成检查，并在报告中签署姓名/身份、时间、Viewer 及版本。任何未由具名人员填写的视觉结论必须标为“尚未验收”。

对 `L3` 任务，人工验收至少检查：

- [ ] 相机轨迹连续，无明显漂移、跳变或镜像翻转。
- [ ] 场景朝向、地面/天花板关系和尺度符合输入视频可观察事实。
- [ ] Gaussian 没有大面积飞点、双影、碎裂、无意义背景块或严重缺失区域。
- [ ] 渲染清晰度、覆盖范围和曝光正常；训练曲线没有无法解释的发散或持续震荡。
- [ ] 多片段重叠区域自然对齐；如声明 Sim(3) 或 MASt3R 匹配，附残差、内点数或诊断证据。
- [ ] 导出模型能在指定 Viewer 加载；属性 schema、文件大小和展示效果符合测试包。
- [ ] 峰值显存、耗时和磁盘占用没有超过限制。

视觉验收失败时，应上传最小预览、相机轨迹图或短视频，标注失败区域和复现步骤；不得只写“效果不好”。

## 9. CUDA 测试节点与 Artifact 安全

self-hosted runner 是持久环境，不应假定每次任务都是干净、隔离的新机器。仅将 GPU runner 授权给本私有仓库或精确指定的 runner group，并限制其可执行的 workflow；GitHub 也建议 self-hosted runner 仅用于私有仓库。[GitHub runner group 安全说明](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access)

必须遵守：

1. 使用专用 Windows/Linux 用户、独立工作目录和最小权限账户；不以管理员身份运行，且不存放个人 SSH 密钥、浏览器登录态、密码管理器或无关项目文件。
2. 禁止外部 fork PR 自动占用 GPU runner。GPU workflow 默认通过手动 dispatch、受信标签或负责人批准触发。
3. 修改 `.github/workflows/**`、runner 启动脚本、环境安装逻辑、依赖锁文件或具有下载/执行能力的第三方 Action 的 PR，必须先经项目负责人审查；未经批准不得在 GPU runner 上执行。
4. workflow 默认采用最小 `permissions`（通常从 `contents: read` 开始），不向 GPU Job 注入不必要的 Secrets；严禁在日志、报告或 Agent 对话中输出 token、密钥和长期凭据。
5. 第三方 GitHub Action 固定到完整 commit SHA，而不是浮动 tag；下载/解压 Artifact 前检查来源、路径穿越、异常体积和文件类型。
6. 每次任务使用独立工作目录；结束后清理该运行的临时缓存与敏感日志，但不得删除其他任务的未确认产物。条件允许时优先采用一次性 runner 或任务后重建环境。
7. Agent 不得自行运行无关网络下载、系统更新、驱动升级、包升级或权限变更。出现依赖冲突时只报告诊断和建议。

## 10. 结果、Artifact 与故障分类

每次 GPU 测试至少回传：

```text
environment.json 或等价环境记录
test-report.md
stdout.log / stderr.log
metrics.json（耗时、峰值显存、关键质量指标）
preview/（必要预览图或短视频）
camera_trajectory.png（如适用）
small_result.ply 或受控存储地址与校验值（如适用）
```

推荐将一次运行保存至独立目录：

```text
outputs/test-runs/<YYYYMMDD>-<short-sha>-<test-name>-<run-id>/
```

该目录仅作为本地或 Artifact 临时产物，不提交 Git。大型模型保存在项目约定的外部存储；报告必须提供可访问位置、SHA-256（或同等校验值）、生成 SHA、Run ID 和保留期限。

失败报告必须保留原始退出码、stderr 和最小复现现场，并按下列类别归因；不确定时可以标为“待调查”，不得强行归类。

| 类别 | 示例 |
|---|---|
| 实现问题 | 逻辑错误、接口不兼容、输出 schema 错误 |
| 环境问题 | CUDA/PyTorch/驱动版本不匹配、缺失依赖 |
| 数据问题 | 损坏输入、manifest 不一致、数据校验失败 |
| 资源问题 | 显存、磁盘、内存或时长超限 |
| 基础设施问题 | Runner 离线、网络中断、挂载异常、Artifact 上传失败 |
| 测试规范问题 | 命令缺失、预期不明确、SHA/配置/数据不一致 |
| 非确定性问题 | 同一 SHA、配置和种子无法稳定复现 |

协作者或 Agent 可以给出根因假设和低风险建议，但不得直接修改待测分支。修复后的新 SHA 必须作为新的测试对象。

## 11. GPU 测试报告模板

协作者或其 Agent 请复制 [GPU Test Report 模板](testing/GPU_TEST_REPORT_TEMPLATE.md) 到 PR comment、`test-report.md` 或团队约定位置。Agent 只能填写“自动化执行”部分；“人工视觉验收”必须由具名协作者填写。下方保留同一模板，便于在本规范内直接查阅。

```markdown
# GPU Test Report

- PR / Issue:
- Test level: L0 / L1 / L2 / L3
- Requested commit SHA:
- Actual HEAD SHA:
- Base branch / SHA:
- Branch（仅定位用）:
- Run ID:
- Attempt / Previous Run:
- 执行时间与时区:
- 测试目的与范围:
- 配置文件与 SHA-256:
- 数据版本与校验值:
- Random seed / deterministic 设置:

## 环境

- OS / Shell:
- GPU / 总显存:
- NVIDIA 驱动 / CUDA runtime:
- Python / PyTorch / CUDA 可用性:
- 关键第三方版本与完整 commit SHA:

## 自动化执行（可由 Agent 填写）

- 执行者（Agent / 人）:
- detached checkout 验证:
- 完整复现命令:
- Exit code:
- 运行时长 / 峰值显存:
- 产物与校验值:
- stdout / stderr:
- Artifact / 共享存储位置:
- 自动化结论:
  - [ ] 自动检查通过
  - [ ] 自动检查失败
  - [ ] 未完成 / 基础设施中断

## 人工视觉验收（必须由具名人员填写）

- 验收人:
- 验收时间与时区:
- Viewer / 版本:
- 相机轨迹:
- 场景方向与尺度:
- 飞点 / 双影 / 缺失区域:
- 渲染质量与训练曲线:
- Viewer 兼容性:
- 人工结论:
  - [ ] 通过
  - [ ] 有限通过（列出已接受限制）
  - [ ] 失败
  - [ ] 尚未验收

## 失败详情、分类与最小复现

## 重试理由与上次运行的差异

## 建议的后续动作
```

## 12. 给协作者 Agent 的固定执行指令

将下列约束作为协作者 Agent 的固定系统提示或任务前置说明；每次任务仍须补充第 5 节的测试包。

```text
你是本项目 CUDA 测试节点的执行与诊断助手，不是代码所有者或人工验收人。

目标：在指定的不可变 commit SHA 上，以可复现、可审计的方式执行测试；完整记录环境、命令、日志、指标、尝试次数和产物，并如实报告。

必须遵守：
1. 仅在负责人指定的 SHA、数据、配置、平台和命令范围内工作。信息不完整、HEAD 不一致或工作区不干净时，停止并报告；不得猜测或替换配置。
2. 先 detached checkout 指定 SHA，再执行测试。报告中同时记录 requested SHA、actual HEAD SHA、base SHA、Run ID、Attempt 和随机种子。
3. 按测试包提供的平台入口执行命令。仅在 macOS/Linux/WSL 或明确支持的 Git Bash 环境使用 ./dev.sh；不得假装原生 Windows 已有 dev.ps1。
4. 不直接修改、提交、推送或合并代码；不得执行 git reset --hard、强制推送、删除分支或删除未知产物。
5. 不读取、输出或传输无关个人文件、凭据、token、密钥或浏览器数据；不执行未经批准的下载、系统更新、驱动升级、包升级或权限变更。
6. 不无记录地重试。每次尝试保留独立 Run ID、日志和结果；同一输入不稳定时报告“非确定性/待调查”。
7. 发生失败时保留 stderr、退出码与最小复现信息，并区分实现、环境、数据、资源、基础设施、测试规范和非确定性问题。可提出建议，但不绕过失败。
8. 你可生成视觉证据并标记疑点，但只能给出自动化结论；不得填写或声称人工视觉验收通过。最终输出使用 GPU Test Report 模板。
```

## 13. 合并前完成定义（Definition of Done）

PR 转为 `Ready for review` 前，项目负责人应确认：

- [ ] PR 描述说明目标、范围、风险、测试等级、数据/配置变更、回滚方式和授权修复（如有）。
- [ ] 所需的 `L0`–`L3` 检查均完成，或有书面批准的豁免。
- [ ] 每份测试报告都绑定当前有效 SHA、配置、数据、环境和 Run ID；没有过期测试作为合并证据。
- [ ] `L3` 的人工视觉验收由具名协作者完成；未验收不得标为通过。
- [ ] 所有重试、失败和非确定性结果均有记录，没有选择性隐藏失败运行。
- [ ] 大资产、凭据、机器特定绝对路径和临时日志未进入 Git。
- [ ] 阻断/高优先级反馈已处理，或有明确负责人、理由和后续 Issue。
- [ ] Ready 状态下 required reviews、required checks、分支同步和保护规则均满足；项目负责人执行最终合并。

## 14. 实施路线与当前能力声明

本规范定义目标行为，不把尚未实现的工具写成既有能力。当前仓库的统一入口仅有 `dev.sh`，且尚未纳入跨平台 `doctor`、统一 smoke test、测试目录或 self-hosted runner 工作流。以下为建议的实施顺序：

已落地的协作入口：

- `.github/pull_request_template.md`：PR 中的测试等级、SHA、证据与 Ready 门槛清单。
- `.github/ISSUE_TEMPLATE/gpu-test-request.yml`：单一不可变 SHA 的 GPU 测试请求表单。
- `.github/workflows/cpu-check.yml`：最小权限的 CPU/仓库静态检查；仅在未来存在依赖清单和 `test_*.py` 时运行 Python 测试。
- `.github/CODEOWNERS`：对 workflow、测试、配置和安全文档的当前 owner 审查责任。
- `tests/`、`configs/test/`、`docs/development/`、`docs/testing/`、`docs/security/`：后续模块化测试与规范资料的目录骨架。

仍待实施：

1. 为 `tests/unit/`、`tests/integration/`、`tests/gpu/` 增加最小 fixture，覆盖关键脚本与已知回归点。
2. 实现跨平台逻辑入口与 `dev.sh`/`dev.ps1` 包装，统一采集环境、执行 smoke test 和产出机器可读报告。
3. 将 lint、配置校验和完整 CPU 测试纳入依赖锁定后的普通 CI；GPU 测试先通过受控标签或手动 dispatch 触发。
4. 为私有仓库配置专用 self-hosted CUDA runner、最小权限 runner group、workflow allowlist 与 Artifact 保留策略。

---

本规范与 [仓库 AGENTS 约定](../AGENTS.md)、[分支与任务说明](branch_task_plan.md) 共同使用；若发生冲突，以项目负责人在 PR 中的明确书面决定为准。
