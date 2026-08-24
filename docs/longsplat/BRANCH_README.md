# research/longsplat-route 分支说明

> 面向协作者的分支级文档。模块 API 细节见同目录 [README.md](README.md)(英文)。
> 分支: `research/longsplat-route` | 更新: 2026-07-21
> 状态: **research prototype, manually validated**

## 1. 分支定位

本分支交付 **LongSplat 路线**的训练编排层: 以预处理分支产出的帧 manifest 为输入, 调用锁定版本的 [NVlabs/LongSplat](https://github.com/NVlabs/LongSplat) 完成免位姿 3DGS 重建, 输出标准 PLY 及完整 run record.

本仓库只包含**编排层代码**; 真正执行训练的 `train.py` 位于 `third_party/LongSplat` (锁定 commit, 不入库).

## 2. 训练入口

唯一推荐 CLI:

```bash
./dev.sh python -m scripts.longsplat.run_experiment \
  --manifest <manifest.json> \
  --segment-id <id> \
  --config <config.json> \
  --repo-root third_party/LongSplat \
  --output-dir outputs/<video> \
  --project-root . \
  [--depth-manifest <depth_manifest.json>]
```

`run_experiment.py` 是薄 CLI, 解析参数后委托 `run_pipeline()`. 底层 `run_pipeline()` / `run_training()` / `run_conversion()` 仍可按库函数调用.

## 3. 交付清单

```
scripts/longsplat/
├── run_experiment.py     # 薄 CLI 入口 (python -m)
├── orchestrator.py       # 完整管线编排
├── runner.py             # 训练/转换命令构建, 后端校验, 配置加载
├── depth_bridge.py       # VDA 深度物化与集合相等校验
├── validate_input.py     # 输入 manifest 契约校验
├── manifest_adapter.py   # 预处理 manifest → 本模块消费格式
├── prepare_input.py      # 帧文件复制到 run 目录
├── convert.py            # PLY 转换, NaN/Inf 清洗, PLY 校验
└── run_record.py         # run record 状态机

configs/longsplat/
├── smoke.json            # baseline 冒烟 (~100 iter)
├── smoke_vda.json        # VDA 冒烟 (~100 iter, depth_source=vda)
├── full.json             # baseline 完整训练 (30000 iter)
├── full_vda.json         # VDA 完整训练 (30000 iter)
└── environment.example.json

tests/integration/longsplat/  # 115 CPU 集成测试
docs/longsplat/
├── README.md
├── BRANCH_README.md           # 本文档
├── REPRODUCTION_RUNBOOK.md    # 复现步骤
├── experiment_log.md          # 实验记录模板
└── patches/                   # LongSplat backend patches
    ├── vda_depth_injection.md          # patch 说明
    ├── longsplat_stability_fixes.patch  # 稳定性修复
    ├── longsplat_training_improvements.patch  # 训练改进
    └── vda_depth_injection.patch       # VDA 深度注入
```

## 4. 快速开始

```bash
# 0. 确认 backend Python 可用 (LongSplat CUDA 环境)
<LongSplat-CUDA-Python> -c "import torch; print(torch.cuda.is_available())"
# 应输出 True

# 1. 冒烟测试
./dev.sh python -m scripts.longsplat.run_experiment \
  --manifest data/manifests/<video>/preprocess_manifest.json \
  --segment-id <segment_id> \
  --config configs/longsplat/smoke.json \
  --repo-root third_party/LongSplat \
  --output-dir outputs/<video> \
  --project-root . \
  --backend-python <LongSplat-CUDA-Python>

# 2. VDA 冒烟 (需要 depth manifest)
./dev.sh python -m scripts.longsplat.run_experiment \
  --manifest data/manifests/<video>/preprocess_manifest.json \
  --segment-id <segment_id> \
  --config configs/longsplat/smoke_vda.json \
  --repo-root third_party/LongSplat \
  --output-dir outputs/<video> \
  --project-root . \
  --depth-manifest data/manifests/<video>/depth_manifest.json \
  --backend-python <LongSplat-CUDA-Python>

# 3. 冒烟通过后换 full.json / full_vda.json 正式训练
```

产出位于 `outputs/<video>/<run_id>/`:
- `input/` — 输入帧不可变快照
- `input/depths/` — 物化深度 .npy (VDA 模式)
- `longsplat_model/converted_3dgs/point_cloud.ply` — 最终 PLY
- `reconstruction_run.json` — run record (配置快照, SHA-256, 退出码, 耗时, VDA usage, submodule_diffs)

## 5. 后端锁定与环境

- LongSplat 锁定 commit: `19750775a9d19f30aa05a8333c4c6c231b2d5f4a`
- `_check_repo()` 在 `locked_clean` 模式下**拒绝**不匹配的 commit 或 uncommitted changes
- `research_local` 模式允许 dirty 工作区 (记录 `diff_sha256` 用于溯源)
- LongSplat 需独立 Python 环境 (CUDA, submodules 编译); 编排层通过 `--backend-python` 指定
- 编排层仅依赖标准库 + `plyfile` + `numpy`
- 应用顺序: stability_fixes → training_improvements → vda_depth_injection (见 `docs/longsplat/patches/vda_depth_injection.md`)
- 三 patch 可在干净 locked base 上顺序 forward-apply 重建当前 backend 修改
- 最终工作树上仅 VDA 和 stability 两份 patch 可单独 reverse-check; training_improvements 因与 VDA patch 重叠同一文件, 需通过隔离 checkout forward-apply 验证

## 6. 输入/输出契约

**输入**: 预处理分支产出的 `preprocess_manifest.json` (schema `"1.0"`), 经 `manifest_adapter` 适配. 必需字段: `segment_id`, 逐帧 `frame_id/path/width/height/sha256`.

**深度 (VDA 模式)**: `depth_manifest.json` 声明 depth 来源, 逐帧 SHA-256. `depth_bridge.py` 执行集合相等校验 (manifest sources ≡ mapping sources), 拒绝 surplus/missing/duplicate.

**输出**: 标准 3DGS PLY, 顶点属性 `x y z`, `f_dc_0..2`, `opacity`, `scale_0..2`, `rot_0..3`. 转换时执行 NaN/Inf 顶点清洗.

## 7. 已验证结果与已知问题

**baseline 4 帧** (2026-07-21): 完整跑通, training 185s, conversion 14s, PLY 50,983 vertices, status `complete`.

**VDA 4 帧** (2026-07-21): 深度物化 4/4, aligned=4, missing=0, rejected=0, training 182s, conversion 13s, PLY 48,955 vertices, status `complete`. 本轮复用已有 NPZ, 未重新运行 depth-prior producer.

**已知问题**:
- 共面物体融合: 贴在同一面墙上的展示牌被融合到墙面上. 引入 depth-prior 分支的 VDA 深度先验, 展示牌在深度图中相对墙面有 3~20σ 的可分辨凸出信号.
- Nested submodule 指纹: 当前 `$displaypath` 已正确发现嵌套 submodule 并记录 HEAD + diff SHA-256; 但四级及以上嵌套未实测. 见 FV-02.

## 8. 测试

```bash
# CPU 集成测试 (无需 GPU)
./dev.sh pytest tests/integration/longsplat -q

# Patch parity (在 third_party/LongSplat)
git -C third_party/LongSplat apply --check --reverse ../../docs/longsplat/patches/vda_depth_injection.patch
git -C third_party/LongSplat apply --check --reverse ../../docs/longsplat/patches/longsplat_stability_fixes.patch
```

## 9. 2026-07-22 有限值门禁与训练遥测

- 后端补丁顺序现为：`longsplat_stability_fixes.patch` →
  `longsplat_training_improvements.patch` → `vda_depth_injection.patch` →
  `longsplat_conversion_telemetry.patch`。
- `scripts/longsplat/telemetry.py` 解析 `POSE_TELEMETRY`、
  `VDA_TELEMETRY`、`CONVERSION_TELEMETRY` 三类严格 JSON marker。
- `reconstruction_run.json` 在 `telemetry.pose`、`telemetry.vda`、
  `telemetry.conversion` 中保存汇总和逐帧记录；旧的 `depth.usage` 继续保留。
- 新转换在 `log(sqrt(distance))` 前把距离限制到至少 `1e-7`，并在导出前
  断言所有 scale 为有限值。
- 正常流水线不再自动删除 NaN/Inf 顶点。原始 PLY 只要含非有限值，运行就会
  失败并保留文件供诊断；清洗器仅用于显式恢复历史产物。

最终补丁可在当前后端直接反向校验：

```bash
git -C third_party/LongSplat apply --check --reverse ../../docs/longsplat/patches/longsplat_conversion_telemetry.patch
```

## 10. 2026-07-22 MASt3R/CUDA 主机内存修复

- `mast3r_low_memory_load.patch` 应用于嵌套仓库
  `third_party/LongSplat/submodules/mast3r`。
- 本地 MASt3R checkpoint 使用 `mmap=True` 加载，并在模型搬运到 CUDA 前释放
  约 2.9 GB 的 checkpoint 对象，降低 Windows 主机内存瞬时峰值。
- 该补丁不改变模型权重、精度、推理流程或 LongSplat 优化参数。

```bash
git -C third_party/LongSplat/submodules/mast3r apply --check --reverse ../../../../docs/longsplat/patches/mast3r_low_memory_load.patch
```
