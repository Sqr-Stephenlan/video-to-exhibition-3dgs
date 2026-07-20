# video-to-exhibition-3dgs — depth prior 分支

基于视频输入，通过 Video Depth Anything 生成逐帧相对深度先验，注入 LongSplat 训练以解决弱纹理/共面区域的几何歧义。

## 快速开始

### 1. 环境

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r configs/depth/requirements.txt
pip install -r third_party/Video-Depth-Anything/requirements.txt
```

### 2. 深度先验生成

```bash
# 环境校验
python scripts/depth/run_depth_prior.py doctor

# 生成逐帧深度（需先有 frames_manifest.json）
python scripts/depth/run_depth_prior.py run
```

输入：`data/manifests/frames_manifest.json`（帧列表）
输出：
- `data/depth/<frame_id>.npz` — 逐帧 VDA relative 深度
- `data/manifests/depth_manifest.json` — 深度清单
- `outputs/reconstructions/depth_prior/run_record.json` — 运行记录

详见 `docs/depth_prior_io.md`。

### 3. 注入 LongSplat 训练

```bash
# 一键启动（含深度物化 + 训练）
python scripts/longsplat/runner.py train \
    --source-path data/frames/<scene_name> \
    --images <images_subdir> \
    --depth-source vda --prepare

# 或仅预演（不执行训练）
python scripts/longsplat/runner.py train \
    --source-path data/frames/<scene_name> \
    --images <images_subdir> \
    --depth-source vda --prepare --dry-run
```

首次使用需先给 `third_party/LongSplat/` 打补丁（`docs/longsplat/patches/`）。

**手动分步执行：**

```bash
# 步骤 1: 物化深度（NPZ → npy）
python scripts/longsplat/materialize_depth.py \
    data/manifests/depth_manifest.json \
    data/frames/<scene_name>

# 步骤 2: 训练
python third_party/LongSplat/train.py \
    --eval --source_path data/frames/<scene_name> \
    --model_path outputs/<scene_name>/longsplat_vda \
    --images <images_subdir> --mode custom \
    --depth_source vda
```

## 目录

```
├── scripts/
│   ├── depth/                 # 深度先验模块（doctor + run）
│   │   ├── run_depth_prior.py # CLI 入口
│   │   ├── backend_vda.py     # VDA 子进程运行器
│   │   ├── manifest.py        # manifest 读写
│   │   └── config.py          # 配置加载/校验
│   └── longSplat/               # LongSplat 编排层
│       ├── runner.py             # 通用训练启动器
│       ├── run_record.py         # 训练元数据追踪
│       └── materialize_depth.py # NPZ→npy 物化
├── configs/depth/             # 深度先验配置
├── docs/
│   ├── depth_prior_io.md                  # 深度先验 IO 契约
│   ├── depth_prior_review_2026-07-17.md   # 代码审查
│   ├── longSplat_depth_injection_design_2026-07-17.md # 注入设计文档
│   └── longSplat/
│       └── patches/            # LongSplat 第三方补丁
│           ├── vda_depth_injection.patch       # P1-P4 深度注入
│           ├── longSplat_stability_fixes.patch # 稳定性修复
│           └── vda_depth_injection.md          # 补丁说明
├── data/
│   ├── depth/<scene>/         # 逐帧深度 NPZ
│   └── manifests/             # manifest JSON
└── third_party/
    ├── Video-Depth-Anything/  # VDA（pinned commit，不入库）
    └── LongSplat/               # LongSplat（pinned commit，不入库）
```

## 主要依赖

| 环节 | 工具 | pinned commit |
|------|------|---------------|
| 深度估计 | Video Depth Anything (vitb) | `4f5ae23` |
| 3DGS 训练 | LongSplat | `1975077` |
| 匹配 | MASt3R (LongSplat 子模块) | pinned |
