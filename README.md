# 展厅 3DGS 重建项目

基于手机/AR眼镜视频输入，实现展厅环境的 3D Gaussian Splatting 重建、模型精简、三维标注与 Web 展示。

## 快速开始

### 1. 环境安装

```powershell
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装 PyTorch（RTX 5060 需要 CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --trusted-host download.pytorch.org

# 安装其余依赖（清华镜像）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 2. 预处理流水线

```bash
# 步骤1: 视频抽帧
python scripts/extract_frames.py --input data/raw_videos/scene.mp4 --fps 5

# 步骤2: 视频分段（可选，长视频推荐）
python scripts/split_video.py --input data/raw_videos/scene.mp4 --method time --segment_duration 30

# 步骤3: 模糊帧过滤
python scripts/filter_blurry_frames.py --input data/frames/ --blur_threshold 100

# 步骤4: 生成 manifest
python scripts/build_manifest.py --input data/frames_filtered/ --segment scene_01
```

### 3. 路线选择

#### 路线 A — 保底演示线（传统 COLMAP + 3DGS）

```bash
# COLMAP SfM 估计位姿
colmap automatic_reconstructor --workspace_path data/colmap_ws --image_path data/frames_filtered/

# 官方 3DGS 训练
cd third_party/gaussian-splatting
python train.py -s ../../data/colmap_ws
```

#### 路线 B — 课题创新线（LongSplat 无位姿）

```bash
# 复现 LongSplat
cd third_party/LongSplat
python train.py --input ../../data/frames_filtered/
```

### 4. 模型后处理

```bash
# SuperSplat 手动清理（浏览器打开）
# 或 splat-transform 批量处理
npx splat-transform compress input.ply output.compressed.ply
```

## 项目目录结构

```
exhibition-3dgs/
├── data/               # 原始视频、帧、深度图等
├── scripts/            # 预处理和工具脚本
├── third_party/        # 开源依赖仓库
├── outputs/            # 重建结果、压缩文件、渲染图
├── annotations/        # 展品标注 JSON
└── docs/               # 项目文档和实验记录
```

## 主要依赖

| 环节 | 主选 | 备选 |
|------|------|------|
| 视频预处理 | FFmpeg + OpenCV | - |
| 深度估计 | Video Depth Anything | Depth Anything V2 |
| 无位姿几何 | MASt3R | DUSt3R |
| 长视频 3DGS | LongSplat | MASt3R + gsplat |
| 基线 SfM | COLMAP | GLOMAP |
| 3DGS 训练 | gsplat | 官方 3DGS |
| 模型编辑 | SuperSplat | - |
| 格式转换 | splat-transform | - |
| Web 展示 | PlayCanvas Engine | SuperSplat Viewer |

## 参考文档

- [项目实现框架](exhibition_3dgs_pipeline_framework.md)
- [架构审核意见](exhibition_3dgs_architecture_review.md)
- [NopeRoomGS 论文](2437_NopeRoomGS_Indoor_3D_Gaus.pdf)
