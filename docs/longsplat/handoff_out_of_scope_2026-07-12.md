# Out-of-Scope Code Handoff — `research/longsplat-route`

> 本文档是范围交接记录，不是实现承诺。按 preprocess、depth、fusion、shared-governance 四组记录已剥离代码的源信息、已知问题和可保留思路。
>
> **不要直接 cherry-pick 当前实现。** 这些代码在剥离时已知存在接口错误、坐标系错误和生命周期缺陷（详见 `code_review_research_longsplat_2026-07-12.md`）。目标分支负责人应理解问题后重新实现，而非复制当前代码。

- 移交日期：2026-07-13
- 源分支：`research/longsplat-route`
- 源 HEAD：`0d48826e3ebeee33e0c6bf6ef62b9454932bc271`
- 剥离依据：`rectification_agent_execution_spec_2026-07-13.md` Section 4.2
- Review 文档：`code_review_research_longsplat_2026-07-12.md`

---

## 1. Preprocess 组

### 1.1 `scripts/extract_frames.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | `--overwrite/--resume` 未接入内部函数；无 PTS 映射；无完整 provenance；CLI flag 不生效 |
| 可保留思路 | Fraction 安全帧率解析；ffprobe + FFmpeg 基本命令结构；`_extraction_meta.json` 概念 |
| 不得复用 | FFprobe `r_frame_rate` 单字段；无 time_base/VFR 记录；参数无范围验证 |
| 目标验收 | 消费 segment plan，保留源 PTS，输出 frame manifest 可被下游校验 |

### 1.2 `scripts/split_video.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | AdaptiveDetector 阈值 `30.0` 应为 `3.0`；无 overlap 段；time 模式无 metadata；编码参数写死；resume 二次校验 |
| 可保留思路 | PySceneDetect Python API 替代 CLI 解析；time/scene/exact 三模式结构 |
| 不得复用 | 阈值量纲错误；无 segment plan 抽象；无 ffprobe 回读校验 |
| 目标验收 | 先生成 segment plan（时间范围 + overlap + precision），再执行落盘并校验实际 PTS |

### 1.3 `scripts/filter_blurry_frames.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | lifecycle flags 未传递；扩展名排序非时间序；reject 目录多 segment 共享；阈值对分辨率敏感 |
| 可保留思路 | imagehash pHash 去重方向；Laplacian 模糊检测 |
| 不得复用 | 固定 blur_threshold=100 当作普适真理；无 calibration preset |
| 目标验收 | 质量指标与 selection 分离；支持不同后端不同 selection policy |

### 1.4 `scripts/build_manifest.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | 缺 schema_version、timestamp、源帧 ID、状态/原因、content SHA、producer run ID；虚构下游路径；目录策略阻止第二个 manifest；绝对路径 |
| 可保留思路 | 枚举图像尺寸并写 JSON 的基本结构 |
| 不得复用 | 凭空推导 depth_dir/segments_dir；glob 猜测而非读取上游 report |
| 目标验收 | 版本化 schema；明确 base URI；消费上游 extraction_meta 和 filter_report |

### 1.5 `scripts/normalize_exposure.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | per-channel 改色度/白平衡；median 命名失真；skip 后仍成功退出；resume 无校验 |
| 可保留思路 | simple/histogram 两种策略的 CLI 选择结构 |
| 不得复用 | 逐 BGR 通道独立调整；无 temporal smoothing；无局部 mask |
| 目标验收 | 可选 policy；默认保留原帧；用重建指标验证而非强制前置 |

### 1.6 `scripts/_common.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | `ensure_output_dir` 粒度过粗；overwrite 整目录删除；resume 不校验工具/输出；无 fsync |
| 可保留思路 | Fraction 解析；ToolError 包装；原子 JSON I/O；file_sha256/dict_fingerprint |
| 不得复用 | 目录级生命周期策略；run 身份不隔离 |
| 目标验收 | run-scoped immutable dir + 成功后发布 alias |

### 1.7 `scripts/_provenance.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | 生产代码未调用；resume 不校验工具版本/checkpoint/schema；publish 非原子 |
| 可保留思路 | inputs/params/tools/outputs 字段骨架 |
| 不得复用 | 只比较 input dict 和 params hash 的 check_resume；逐文件删除再移动的伪原子 publish |
| 目标验收 | 校验 stage/schema 版本、tool/backend commit、checkpoint hash、输出存在性及 hash |

### 1.8 `tests/conftest.py`、`tests/test_common.py`、`tests/test_provenance.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | conftest 通过修改 sys.path 掩盖包结构问题；test_common 依赖 cv2；test_provenance 未测中途失败/atomic |
| 可保留思路 | 确定种子的合成数据 fixture |
| 不得复用 | 直接 sys.path 注入；缺少真实 CLI/第三方接口测试 |

---

## 2. Depth 组

### 2.1 `scripts/_vda_adapter.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | device `cuda:0` 传入 autocast；`max_frames_per_batch` 切断 VDA 内部 temporal window；`--fp16` 无法关闭；fps 默认帧数量；preprocess 限短边而非长边；全量内存加载 |
| 可保留思路 | 使用官方 VideoDepthAnything 类；区分 relative/metric checkpoint；preset 注册表 |
| 不得复用 | 外部分批策略；设备字符串透传；preprocess 长/短边语义 |
| 目标验收 | 保留官方连续 temporal window；device/precision 正确；输出 depth manifest + confidence/mask |

### 2.2 `scripts/run_video_depth.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | fp16 store_true + default=True 无法关闭；FPS 默认帧数量；非原子保存；无 depth manifest/下游消费 |
| 可保留思路 | CLI 暴露模型、设备、输入尺寸入口 |
| 不得复用 | 无 confidence/mask；无 frame ID/输入哈希/完成标记 |
| 目标验收 | 按 manifest 时间序列推理；非原子保存改为先写 temp 再发布 |

### 2.3 `tests/test_vda_adapter.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | stub 命名过期；不运行真实官方接口；未测 device/batch boundary/output shape |
| 可保留思路 | preset 元数据和 resize 的基本测试 |
| 不得复用 | 与已实现 adapter 状态不一致的 stub |

---

## 3. Fusion 组

### 3.1 `scripts/align_segments.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | **P0**：MASt3R 返回的是 pair frame 内坐标而非 segment local 坐标；Sim(3) 复合方向与顺序错误；读取 native anchor+MLP PLY 当作标准 3DGS；normal/SH 变换错误；pose graph 实际只有线性链 |
| 可保留思路 | 前置校验框架；RANSAC 估计骨架；阶段化报告；`validate_segments()` 的存在性检查 |
| 不得复用 | 坐标系来源；Sim(3) 复合逻辑；PLY 读取路径；属性变换；pose graph 声明过度 |
| 目标验收 | 先做 2D-2D 匹配 → 提升到局部坐标 → 正确 Sim(3) 方向 → 标准 converted PLY → 正确属性变换 |

### 3.2 `scripts/_mast3r_adapter.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | **P0**：API key/shape 不匹配；`pred2['pts3d']` 字段不存在；desc 缺 squeeze；置信度语义不一致；默认 checkpoint 只是文件名 |
| 可保留思路 | 后端隔离模式；`Mast3rMatcher` 类结构；lazy-load 模型 |
| 不得复用 | 当前 `match_pair()` 的 API 调用方式；3D 点图直接返回当作 segment local 坐标 |
| 目标验收 | 先只返回经过验证的 2D 匹配和置信度；不声称返回 3D 坐标 |

### 3.3 `scripts/_ply_transform.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | **P0**：scale 线性乘（应为 log 加）；SH 未旋转；未知属性 passthrough（应拒绝）；PLY 元信息丢失 |
| 可保留思路 | 位置/四元数变换框架；属性分类器结构 |
| 不得复用 | log scale 错误；SH 透传；schema 校验只比属性名不比 dtype |
| 目标验收 | 在 converter 后的标准 PLY 上正确处理 log scale、rotation、SH 和 binary I/O |

### 3.4 `scripts/_dedup.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | 全量 vstack + 全局 tree；query_ball_point 邻居爆炸；workers=-1 占满 CPU；固定绝对半径；顺序依赖 |
| 可保留思路 | cKDTree 替代 O(N²) 显式全量距离矩阵 |
| 不得复用 | 固定 radius 0.01；全量内存加载；无 scale binding；顺序依赖合并逻辑 |
| 目标验收 | 体素/块级空间分区；流式读取；显式 memory budget；可插拔 merge policy |

### 3.5 `tests/test_mast3r_adapter.py`、`tests/test_ply_transform.py`、`tests/test_dedup.py`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | mast3r 测试依赖大 checkpoint；ply_transform 把 log scale 当线性 scale；dedup 断言过宽 |
| 可保留思路 | Sim(3) 合成测试思路；位置/旋转基础测试 |
| 不得复用 | 固化错误语义的 scale 断言；要求大权重文件的单元测试 |

---

## 4. Shared-Governance 组

### 4.1 `README.md`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | LongSplat 命令路径/参数失效；文档链接断裂；`--overwrite/--resume` 声明失实；裸 `python` 与 `./dev.sh python` 冲突 |
| 可保留思路 | 双路线概览；产物路径约定表 |
| 不得复用 | 失效命令；虚假能力声明；绝对路径引用 |
| 目标验收 | 每条命令在锁定环境中执行过；区分主仓库入口和 backend runtime 入口 |

### 4.2 `dev.ps1`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | 与 `.venv` 约定不一致；清华镜像写死；外部命令退出码未统一检查 |
| 可保留思路 | bootstrap/test/ruff 入口骨架 |
| 不得复用 | 镜像硬编码；无 `$LASTEXITCODE` 检查 |
| 目标验收 | 可配置镜像；统一退出码检查 |

### 4.3 `pyproject.toml`、`requirements.txt`

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | 全部开放下界；CPU/GPU/多后端混装；Python target 与 CI/本机不一致；无 profile/lock |
| 可保留思路 | Ruff/pytest 基础配置骨架 |
| 不得复用 | 无 profile 分离的单体依赖列表 |
| 目标验收 | 拆出 core/preprocess、test、longsplat、mast3r、vda profile |

### 4.4 `.gitignore`（分支新增规则）

| 项目 | 内容 |
|------|------|
| 源 SHA | `0d48826` |
| Review 问题 | 忽略规则与 run artifact 管理无配套说明 |
| 需保留 | `.codegraph/`（用户后续添加） |
| 需撤销 | `data/rejected_frames/**`、`box/`、demo 数据、`*.pdf`（详见 spec 4.3） |

---

## 5. 总体声明

上述代码均已在 `code_review_research_longsplat_2026-07-12.md` 中详细审查。已知存在：

- **5 个 P0 阻断问题**：MASt3R API 不匹配、坐标系错误、Sim(3) 复合方向错误、PLY 格式选错、PLY 属性变换错误
- **8 个 P1 高优先级问题**：VDA 设备/批处理、生命周期未接入、manifest 不完整、分段阈值错误……
- **6 个 P2 中优先级问题**：曝光归一化、参数验证、测试覆盖……

**目标分支负责人应**：
1. 阅读上述 Review 文档理解问题本质
2. 重新设计符合文档契约的实现
3. 不要直接 cherry-pick 或复制当前剥离的代码

本交接文档不包含 patch、复制代码或新的实现承诺。
