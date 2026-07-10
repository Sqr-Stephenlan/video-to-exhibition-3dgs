# 展厅视频驱动 3DGS 项目说明

> 文档定位：本文件是项目的架构、模块职责、分支边界和技术路线说明，主要回答 AI 协作者“这个项目要做什么、当前任务应改哪里、输出应交给谁”。
> 配套规则：`AGENTS.md` 负责环境与执行约束；本文件负责架构与任务边界；`docs/collaborator_agent_development_guide.md` 负责 PR、测试和验收流程。三者必须同时遵守。
> 当前状态：仓库已建立文档、数据、输出、测试和第三方后端的目录骨架；各算法模块仍按对应任务分支逐步实现。不得把规划中的能力描述成已经完成。

## 1. 项目目标与边界

本项目以手机或 AR 眼镜拍摄的展厅视频为输入，建立可复现的 3D Gaussian Splatting（3DGS）重建流水线，最终提供可查看、可压缩、可标注、可在 Web 端展示的展厅数字孪生场景。

### 1.1 输入

- 手机或 AR 眼镜拍摄的展厅视频。
- 可选的相机、IMU、深度或已知尺寸信息。
- 展品清单、标注元数据和受控测试配置。

### 1.2 主要输出

1. 视觉主模型：`.ply`、`.splat`、`.compressed.ply`、`.sog`。
2. 相机、训练与评估记录：相机参数、运行配置、日志、指标和渲染结果。
3. 标注数据：独立的 `annotations.json` 或等价 JSON 图层。
4. 展示产物：SuperSplat、SuperSplat Viewer 或 PlayCanvas 可加载的场景。
5. 可选辅助几何：用于导航、碰撞或拾取的 `.glb`、`.obj`、voxel 或 collision mesh。

### 1.3 明确不做的事

- 不把本仓库改造成 LongSplat、COLMAP、gsplat 等项目的源码副本；本仓库只做总控编排、接口适配和实验记录。
- 不把单目或视频深度当成可靠几何真值；它只能作为初始化、正则或低置信度区域辅助。
- 不把 3DGS PLY 当成传统 mesh，也不把 OBJ 当成主要视觉模型的自然输出。
- 不在短视频 smoke test 尚未通过时直接承诺完整展厅长视频可稳定重建。
- 不因单条研究路线失败而阻塞全部演示交付；项目始终保留保底演示线和课题创新线。

## 2. AI 协作者开始任务前必须确认

任何 Agent 在读取或修改代码前，必须按以下顺序建立上下文：

1. 阅读根目录 `AGENTS.md`，确认环境、命令和权限约束。
2. 阅读本文件，确认项目架构、当前分支职责和允许修改的模块。
3. 阅读 `docs/branch_task_plan.md`，确认分支依赖、交付物和最小完成状态。
4. 涉及开发、测试、GPU、PR 或验收时，阅读 `docs/collaborator_agent_development_guide.md`。
5. 执行 `git branch --show-current` 和 `git status --short`，确认当前分支及用户已有修改。
6. 在开始工作前写清楚：任务目标、所属模块、允许修改的路径、输入契约、输出契约和测试等级。

如果任务与当前分支职责不一致，Agent 必须停止扩展范围：报告需要哪个分支或哪个模块接手，不得自行跨模块完成。

## 3. 总体架构

本仓库是总控编排层，第三方项目是可替换的执行后端。项目数据从上游模块单向流向下游模块，模块之间通过 manifest、标准相机格式、模型文件和变换记录交接，不通过隐式路径或临时人工约定耦合。

```text
视频采集
  ↓
视频标准化、分段、抽帧、关键帧筛选
  ↓
┌──────────────────────────┬────────────────────────────┐
│ 保底演示线               │ 课题创新线                 │
│ COLMAP                    │ LongSplat / MASt3R         │
│ 官方 3DGS / gsplat       │ 无位姿、长视频联合优化     │
└──────────────────────────┴────────────────────────────┘
  ↓                         ↘
标准相机/几何/3DGS 接口 ← Video Depth Anything 深度先验
  ↓
片段 Sim(3) 对齐、pose graph、loop closure、全局融合
  ↓
裁剪、去噪、压缩、格式导出
  ↓
独立 JSON 三维标注层
  ↓
SuperSplat / PlayCanvas / Web 展示与验收
```

### 3.1 分层原则

| 层级 | 责任 | 不应承担的责任 |
|---|---|---|
| 数据层 | 保存视频、帧、深度、掩码和 manifest | 训练逻辑、展示逻辑 |
| 编排与适配层 | 调用第三方后端、转换格式、固化参数、生成记录 | 复制或大改第三方算法源码 |
| 算法后端层 | 执行 SfM、深度、3DGS 训练、配准和转换 | 决定主仓库目录和跨模块契约 |
| 产物层 | 保存重建、模型、渲染、压缩和 Web 构建结果 | 作为源码提交进 Git |
| 展示与标注层 | 加载模型、叠加标注和提供交互 | 反向改变上游训练数据定义 |
| 评估与治理层 | 测试、报告、PR 门禁、版本与资产审计 | 代替算法模块实现功能 |

## 4. 技术路线

### 4.1 阶段 A：采集与预处理

采集应采用分区、慢速、稳定、高重叠的路径，尽量锁定曝光和白平衡。反光展品需要多角度补拍，白墙、地面和天花板需要斜角及边缘覆盖。

预处理由 FFmpeg、OpenCV 和 PySceneDetect 完成：

```text
raw_video
  → 格式与帧率标准化
  → 按场景、时间、轨迹或质量分段
  → 抽帧
  → 模糊、曝光、重复度和运动质量筛选
  → frames_manifest.json
```

预处理模块只负责生成可追踪的片段、帧和质量记录，不负责相机求解或 3DGS 训练。

### 4.2 阶段 B1：保底演示线

```text
关键帧
  → COLMAP SfM
  → 相机位姿 + 稀疏点云
  → 官方 3DGS / gsplat / Nerfstudio Splatfacto
  → 标准 3DGS PLY
  → SuperSplat 清理
  → 压缩与展示
```

该路线用于尽快形成单展品或单展区的可展示基线，也是评估采集质量和对比研究路线的 baseline。纹理较丰富区域优先使用此路线。

### 4.3 阶段 B2：课题创新线

```text
关键帧或短视频
  → LongSplat / MASt3R
  → 匹配、深度/点图、相机关系和增量位姿
  → 相机位姿与 Gaussians 联合优化
  → 标准 3DGS PLY
  → 与保底线进行质量、耗时和规模对比
```

- LongSplat 是最贴近“无位姿长视频 3DGS”的研究主线，但不是未经验证的唯一 MVP。
- 首次验证使用约 10–20 秒短视频、约 10 fps 和低分辨率输入；跑通后再扩展到单展区和完整展厅。
- MASt3R / DUSt3R 与 gsplat 之间必须有显式接口层，转换相机内参、外参、点云、置信度、尺度和坐标系。不得假设二者可以直接即插即用。
- NopeRoomGS 在源码和可复现入口未确认前，只作为课题概念或论文参考，不作为唯一工程后端。

### 4.4 阶段 C：深度先验

Video Depth Anything 或 Depth Anything V2 用于弱纹理区域的深度先验、初始化候选或训练正则：

- 反光、玻璃、屏幕、遮挡边界和多帧不一致区域必须降低权重。
- 与 MASt3R 或 COLMAP 几何冲突的深度不得作为硬约束。
- 如果需要真实尺度，必须增加已知尺寸、地面标尺、AR 设备尺度、IMU 或人工标定点。
- 深度模块输出深度和置信度记录，不直接宣布“几何补全完成”。

### 4.5 阶段 D：分段融合

分段融合是项目最大工程风险，只有至少两个局部片段已经分别重建成功后才开始：

1. 相邻片段保留 30%–50% 的重叠关键帧。
2. 使用 MASt3R / DUSt3R 或其他受控方法建立跨片段匹配。
3. 估计并记录包含尺度、旋转和平移的 Sim(3) 变换。
4. 多片段建立 pose graph；闭环路径执行 loop closure。
5. 统一坐标系后合并模型，处理重复 Gaussian、漂浮噪点和颜色/曝光差异。
6. 记录残差、内点数、失败片段和回退方式；允许人工锚点或局部重采集，但必须留痕。

### 4.6 阶段 E：后处理、标注与展示

- 使用 SuperSplat 进行人工或半自动裁剪、去噪和质量检查。
- 使用 splat-transform 进行批量坐标变换、过滤、合并、压缩和格式转换。
- 标注以独立 JSON 图层保存，不写入 splat 内部。
- Web 端优先使用 PlayCanvas Engine 或 SuperSplat Viewer，并记录模型大小、Gaussian 数量、加载时间和 FPS。

输出必须区分：

| 输出类别 | 推荐格式 | 用途 |
|---|---|---|
| 视觉主模型 | `.ply`、`.splat`、`.compressed.ply`、`.sog` | 3DGS 查看与 Web 展示 |
| 辅助几何 | `.glb`、`.obj`、voxel、collision mesh | 导航、碰撞、拾取 |
| 标注数据 | `.json` | 展品、展区、路径和交互元数据 |

## 5. 模块接口与数据契约

模块之间只通过下列显式产物交接。任何字段或路径变化都视为接口变更，必须同步通知消费方并更新测试。

| 交接产物 | 生产模块 | 消费模块 | 至少应记录的内容 |
|---|---|---|---|
| `frames_manifest.json` | 预处理 | 全部重建路线、深度 | 视频/片段 ID、帧 ID、时间戳、相对路径、尺寸、选取状态与原因 |
| 深度 manifest | 深度先验 | LongSplat/gsplat 实验、评估 | 帧 ID、深度路径、模型版本、relative/metric 类型、置信度或掩码 |
| 相机与几何接口 | COLMAP 或 MASt3R 适配层 | 3DGS 训练 | 内参、外参、坐标系、尺度、点云、匹配置信度及格式版本 |
| 重建运行记录 | 重建模块 | 融合、导出、评估 | 输入 manifest、后端与 commit、配置、命令、随机种子、输出路径与指标 |
| 片段变换记录 | 融合 | 导出、展示、评估 | source/target、4×4 变换、尺度、方法、内点数、残差、坐标系版本 |
| 标准 splat | 重建或融合 | 导出与展示 | 格式、属性 schema、坐标系、Gaussian 数量、文件校验值 |
| `annotations.json` | 标注/展示 | Web 展示 | ID、名称、类型、位置、旋转、描述、媒体引用、坐标系版本 |

所有 manifest 和运行记录必须使用仓库相对路径，不得写入个人机器绝对路径、凭据或临时下载地址。

## 6. 目录职责

### 6.1 当前仓库目录

| 路径 | 职责 | 修改边界 |
|---|---|---|
| `PROJECT.md` | 架构、技术路线、模块和分支边界 | 由 `docs/pipeline-design` 维护；接口或边界变化需负责人确认 |
| `AGENTS.md` | Agent 环境和执行约束 | 受保护共享文件；功能分支不得顺手修改 |
| `.github/` | PR 模板、Issue 模板、CODEOWNERS、CI | 影响仓库执行或 GPU 安全，未经负责人审查不得修改 |
| `docs/` | 架构、协作规范、实验方案、测试与验收材料 | 文档分支维护；不得存放大模型或运行产物 |
| `configs/` | 可复现的项目配置和测试配置 | 按模块分目录；配置必须可追踪且不得含凭据/个人绝对路径 |
| `data/` | 本地视频、片段、帧、深度、掩码和 manifest | 生成资产默认不进 Git；只保留目录骨架或小型受控 fixture |
| `scripts/` | 主仓库拥有的薄编排、格式适配和检查脚本 | 不复制第三方核心算法；按模块分目录 |
| `third_party/` | 外部算法和工具的本地执行后端 | 源码默认不由主仓库跟踪；修改应在第三方自身仓库/分支完成 |
| `outputs/` | 重建、splat、压缩模型、渲染和 Web 构建产物 | 默认不进 Git；通过 Artifact 或受控存储共享 |
| `annotations/` | 小型标注 schema、模板和经审核的标注 JSON | 不存媒体大文件；必须声明坐标系版本 |
| `tests/` | CPU 单元、普通集成和 CUDA/GPU 测试 | 测试按所属模块放置；GPU 视觉质量不能由 Agent 代签 |
| `dev.sh` | 当前 macOS/Linux/WSL Python 项目入口 | 受保护共享文件；Python 命令必须经此入口运行 |

### 6.2 数据目录

| 路径 | 内容 |
|---|---|
| `data/raw_videos/` | 原始视频，只读保留，不被后续模块覆盖 |
| `data/segments/` | 分段视频 |
| `data/frames/` | 抽取和筛选后的帧 |
| `data/depth/` | 深度图或深度数组 |
| `data/masks/` | 动态物体、低置信度、反光或其他掩码 |
| `data/manifests/` | 帧、片段、深度和数据版本的结构化清单 |

### 6.3 输出目录

| 路径 | 内容 |
|---|---|
| `outputs/reconstructions/` | 后端训练目录、相机解算和中间结果 |
| `outputs/splats/` | 标准 3DGS / splat 模型 |
| `outputs/compressed/` | Web 友好压缩格式及辅助几何 |
| `outputs/renders/` | 渲染对比、预览、轨迹图和评估图 |
| `outputs/web_demo/` | 展示端构建结果 |
| `outputs/test-runs/` | 建议的本地/GPU 单次测试包产物；不提交 Git |

### 6.4 模块代码落位

新增实现时使用下列模块子目录；目录尚不存在时，只能由对应负责分支首次创建：

| 模块 | 脚本 | 配置 | 测试 |
|---|---|---|---|
| 视频预处理 | `scripts/preprocess/` | `configs/preprocess/` | `tests/unit/preprocess/`、`tests/integration/preprocess/` |
| COLMAP/3DGS 基线 | `scripts/baseline/` | `configs/baseline/` | `tests/integration/baseline/`、`tests/gpu/baseline/` |
| LongSplat 路线 | `scripts/longsplat/` | `configs/longsplat/` | `tests/integration/longsplat/`、`tests/gpu/longsplat/` |
| 深度先验 | `scripts/depth/` | `configs/depth/` | `tests/integration/depth/`、`tests/gpu/depth/` |
| 片段融合 | `scripts/fusion/` | `configs/fusion/` | `tests/unit/fusion/`、`tests/gpu/fusion/` |
| 导出、展示、标注 | `scripts/export/`、`scripts/viewer/` | `configs/export/`、`configs/viewer/` | `tests/integration/export/`、`tests/integration/viewer/` |

通用逻辑只有在至少两个模块确实复用、接口已明确且负责人同意后，才可提取到共享目录。不得预先建立宽泛的 `utils.py` 收纳跨模块代码。

## 7. 分支职责与写入范围

下表是 Agent 的默认写权限边界。未列出的路径默认为不可修改；任务中若有更窄范围，以任务范围为准。

| 分支 | 主要职责 | 默认可修改范围 | 核心交付物 | 禁止越界 |
|---|---|---|---|---|
| `main` | 稳定、已确认、可复现成果 | 只接受经审查合并 | 稳定文档、配置、脚本和入口 | 不直接开发或试验 |
| `docs/pipeline-design` | 架构、管线、任务拆分和模块边界 | `PROJECT.md`、`docs/branch_task_plan.md`、相关架构文档 | 项目说明、接口和分支规范 | 不实现算法、训练或展示功能 |
| `feature/preprocess-video` | 视频标准化、分段、抽帧和质量筛选 | `scripts/preprocess/**`、`configs/preprocess/**`、对应测试与预处理文档 | 片段、关键帧、`frames_manifest.json`、质量报告 | 不求位姿、不训练 3DGS、不改融合/展示模块 |
| `feature/baseline-colmap-3dgs` | COLMAP + 官方 3DGS/gsplat 保底线 | `scripts/baseline/**`、`configs/baseline/**`、对应测试与实验文档 | 单展区基线模型、相机结果、训练日志、可查看 PLY | 不改预处理规则、不实现 LongSplat 或融合 |
| `research/longsplat-route` | 无位姿长视频 3DGS 研究线 | `scripts/longsplat/**`、`configs/longsplat/**`、对应测试与实验文档 | 自定义短视频跑通记录、标准 PLY 和对比结果 | 不把研究接口强加给基线，不实现深度/展示模块 |
| `research/depth-prior` | 视频深度和置信度过滤 | `scripts/depth/**`、`configs/depth/**`、对应测试与实验文档 | 深度、置信度/掩码、深度 manifest | 不把深度当真值，不直接改训练核心或融合 |
| `feature/segment-fusion` | 跨片段匹配、Sim(3)、pose graph 和融合 | `scripts/fusion/**`、`configs/fusion/**`、对应测试与实验文档 | 变换矩阵、融合模型、对齐质量记录 | 不改上游抽帧/训练算法，不负责 Web 展示 |
| `feature/export-viewer-annotation` | 模型清理、压缩、转换、Web 展示和标注 | `scripts/export/**`、`scripts/viewer/**`、相关配置、测试、`annotations/**` 和展示源码 | `.sog`/`.compressed.ply`、Web demo、`annotations.json` | 不修改重建、深度和融合算法 |
| `docs/evaluation-reporting` | 指标、对比评估、失败案例和验收材料 | 评估文档、报告模板；必要的小型评估配置需单独声明 | 质量、大小、耗时、FPS、失败分析 | 不借评估之名修改被评算法 |

### 7.1 禁止跨模块修改

1. 一个功能分支只实现表中对应模块，不“顺手”重构相邻模块。
2. `AGENTS.md`、`PROJECT.md`、`dev.sh`、`.gitignore`、依赖清单和 `.github/**` 是受保护共享区；功能或研究分支需要修改时，必须先说明原因并取得负责人确认。
3. 上游只生产约定接口，下游只消费约定接口。发现对方输出不足时，应提出接口变更，不得直接进入对方目录修补实现。
4. 需要同时改动两个模块时，应拆分为独立分支/PR，或由负责人书面确认一次跨模块任务及其路径范围。
5. 不得移动、重命名或删除其他模块的输入、输出、配置和测试 fixture。
6. 不得在主仓库直接修改 `third_party/**` 核心源码。必要 patch 在第三方自身仓库维护，主仓库记录来源、commit SHA、patch 路径和校验值。
7. 不得提交 `data/**`、`outputs/**`、模型权重、日志、视频、PLY 或其他 `.gitignore` 已排除的大资产。
8. 不确定代码归属时，先按数据流确定生产者和消费者；仍不明确则暂停修改并向负责人说明边界冲突。

### 7.2 跨模块交接规则

- 上游接口变化必须版本化，并提供迁移说明或兼容期。
- 消费方不得依赖未写入 manifest 的隐式信息，例如本地目录顺序、文件修改时间或个人路径。
- 共享配置必须说明所有者；模块配置不得反向覆盖全局默认值。
- 跨模块 bug 应先定位责任模块，再由其分支修复；消费方可以增加错误提示，但不能复制生产者逻辑。

## 8. 分支依赖与实施顺序

```text
docs/pipeline-design
  ↓
feature/preprocess-video
  ├───────────────┬────────────────┐
  ↓               ↓                ↓
baseline          longsplat        depth-prior
  └───────────────┴────────┬───────┘
                           ↓
                    segment-fusion
                           ↓
                 export-viewer-annotation
                           ↓
                 evaluation-reporting
```

约束如下：

- 基线、LongSplat 和深度先验可在预处理接口稳定后并行推进。
- 深度先验是可选辅助，不应成为保底线的强制前置依赖。
- 片段融合必须等待至少两个局部重建成功，不能先写一个只对虚构输入成立的融合流程。
- 导出与展示可以用受控样例模型先做加载 smoke test，但最终验收必须使用本项目稳定输出。
- 评估记录贯穿所有阶段；`docs/evaluation-reporting` 只整理证据，不改变实验结果。

## 9. 第三方后端管理

`third_party/` 当前可包含 FFmpeg、OpenCV、PySceneDetect、COLMAP、gaussian-splatting、gsplat、LongSplat、MASt3R、Video Depth Anything、SuperSplat、splat-transform 和 PlayCanvas Engine 等本地后端。

每项依赖至少记录：

- 仓库 URL 和许可证。
- 完整 commit SHA，不以 `latest`、`main` 或浮动 tag 作为唯一版本。
- 获取和安装方式。
- 所需 CUDA、Python、PyTorch 或系统依赖。
- 本地 patch、patch SHA-256 和修改原因。
- 由哪个主仓库脚本调用、输入输出格式是什么。

主仓库的适配脚本应尽量薄：负责校验输入、调用后端、转换格式和记录运行信息，不重写第三方算法核心。

## 10. 环境、测试与完成定义

### 10.1 环境入口

Python 命令必须使用：

```bash
./dev.sh python ...
./dev.sh pip ...
./dev.sh pytest ...
./dev.sh mypy ...
./dev.sh ruff ...
```

不得使用裸 `python`、`pip`、`pytest`、`mypy` 或 `ruff`。`.venv` 缺失时，只有在用户确认后才执行 `./dev.sh bootstrap`。

### 10.2 测试等级

| 等级 | 适用改动 | 最低证据 |
|---|---|---|
| `L0` | 文档、注释、非执行性配置 | 文档和链接检查；GPU 可豁免 |
| `L1` | CPU 工具、manifest、CLI、预处理 | 静态检查、相关单元测试、CPU smoke test |
| `L2` | GPU 调用链、依赖、CUDA 环境、导出接口 | L1 + GPU smoke test |
| `L3` | 训练、位姿、深度、跨段配准、重建质量 | L2 + 规定 GPU 测试 + 具名人工视觉验收 |

Agent 可以运行自动检查、整理指标和生成视觉证据，但不得宣称 GPU 训练、显存或视觉质量已通过，除非存在对应有效报告；人工视觉验收只能由具名人员完成。

### 10.3 各分支最小完成状态

| 分支 | 最小完成状态 |
|---|---|
| `feature/preprocess-video` | 一个短视频可生成片段、关键帧和有效 manifest |
| `feature/baseline-colmap-3dgs` | 一个单展区样例可经 COLMAP + 3DGS 输出可查看 PLY |
| `research/longsplat-route` | 一个短视频可按 LongSplat 自定义输入流程训练并导出标准 PLY |
| `research/depth-prior` | 一个视频/帧序列可输出深度、置信度和帧对应记录 |
| `feature/segment-fusion` | 至少两个局部片段可估计相对变换并输出合并结果与误差记录 |
| `feature/export-viewer-annotation` | 模型可转换为展示格式并加载独立 JSON 标注层 |
| `docs/evaluation-reporting` | 可按统一模板记录质量、大小、耗时、资源和展示性能 |

## 11. 实验与验收顺序

所有路线按规模逐步扩展：

1. **单展品小场景**：验证采集、预处理、重建、导出和查看的闭环。
2. **单展区中等场景**：验证弱纹理、反光、曝光变化和局部对齐。
3. **完整展厅长视频**：验证自动分段、全局融合、压缩、标注和展示。

每次实验至少记录：

- 原始视频 ID、时长和校验值。
- 抽帧数、有效关键帧数和分段重叠率。
- 代码 commit、第三方 commit、配置指纹、完整命令和随机种子。
- GPU、显存、运行耗时和磁盘占用。
- Gaussian 数量、原始/压缩模型大小、展示端 FPS。
- 相机轨迹、对齐残差、主观视觉质量和失败区域。
- 产物位置、校验值、Run ID 和保留期限。

## 12. 主要风险与回退方案

| 风险 | 首选处理 | 回退方案 |
|---|---|---|
| LongSplat 环境复杂或自定义视频失败 | 短视频、低分辨率、约 10 fps smoke test，固定依赖版本 | MASt3R/DUSt3R 接口层 + gsplat，或回到 COLMAP 基线 |
| 白墙、地面和反光区域几何不稳 | 改善采集、过滤低置信度区域、加入软深度先验 | 局部重采集、人工掩码、接受并记录限制 |
| 多片段融合失败 | 增加重叠、Sim(3)、pose graph、loop closure 和误差诊断 | 人工锚点、减少片段、先交付单展区 |
| 模型过大、Web 卡顿 | 去噪、裁剪、压缩、分区加载 | 降低 Gaussian 数量或只展示重点区域 |
| 真实尺度不确定 | 加入已知尺寸或设备尺度约束 | 明确标记为相对尺度，不提供测量承诺 |
| OBJ/mesh 验收需求 | 单独生成 collision/辅助几何 | 保留 3DGS 为视觉主模型，说明两类格式差异 |

## 13. Agent 任务结束前检查

- [ ] 修改只发生在当前分支负责的路径内。
- [ ] 未覆盖或清理用户已有修改。
- [ ] 输入、输出、坐标系和 manifest 变化已记录。
- [ ] 未把本机绝对路径、凭据、模型、大视频、日志或生成产物提交进 Git。
- [ ] 使用真实存在的命令完成了与风险相称的检查。
- [ ] 未把未运行、GPU 未验证或人工未验收的内容写成“已通过”。
- [ ] 已说明已完成内容、未完成内容、已知风险和下游交接方式。

## 14. 参考文档

- `exhibition_3dgs_pipeline_framework.md`：原始完整技术路线与工具选型。
- `exhibition_3dgs_architecture_review.md`：对原路线的风险审核和双线 MVP 修订建议。
- `docs/branch_task_plan.md`：分支、阶段、输出与最小完成状态。
- `docs/collaborator_agent_development_guide.md`：角色权限、PR、测试等级、GPU 节点与验收规范。
- `docs/testing/README.md`：测试资料入口。
- `docs/security/README.md`：CUDA 测试节点安全边界。

若本文件与实际实现状态不一致，应先修订文档或在任务中明确偏差，不得由 Agent 默默选择一种解释。若规则之间存在冲突，遵循更严格的安全和范围约束，并由项目负责人作出书面决定。
