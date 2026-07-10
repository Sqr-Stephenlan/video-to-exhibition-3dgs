# 展厅 3DGS 重建项目

基于手机/AR眼镜视频输入，实现展厅环境的 3D Gaussian Splatting 重建、模型精简、三维标注与 Web 展示。

## 快速开始

### 1. 环境安装

```powershell
python -m venv venv
venv\Scripts\activate

# 安装 PyTorch（RTX 5060 需要 CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 安装其余依赖
pip install -r requirements.txt
```

或使用开发辅助脚本（推荐）：

```powershell
.\dev.ps1 bootstrap    # Windows
./dev.sh bootstrap     # Linux / macOS
```

之后所有 Python 命令通过 `./dev.sh python` / `.\dev.ps1 python` 调用，确保使用项目虚拟环境。

### 2. 预处理流水线

以下命令从项目根目录执行，默认输出到 `data/` 子目录：

```bash
# 步骤1: 视频抽帧
python scripts/extract_frames.py --input data/raw_videos/scene.mp4 --fps 5

# 步骤2: 视频分段（可选，长视频推荐）
python scripts/split_video.py --input data/raw_videos/scene.mp4 --method time --segment_duration 30
# 按场景切换检测分段：
python scripts/split_video.py --input data/raw_videos/scene.mp4 --method scene

# 步骤3: 模糊/曝光/重复帧过滤
python scripts/filter_blurry_frames.py --input data/frames/ --blur_threshold 100

# 步骤4: 生成结构化帧清单
python scripts/build_manifest.py --input data/frames_filtered/ --segment scene_01
```

产物路径约定：

| 阶段 | 默认输出 |
|------|----------|
| 抽帧 | `data/frames/` |
| 分段 | `data/segments/` |
| 过滤 | `data/frames_filtered/` |
| 深度图 | `data/depth/` |
| 曝光归一化 | `data/frames_normalized/` |
| 帧清单 | `data/manifests/` |

所有脚本支持 `--output` 自定义路径，以及 `--overwrite` / `--resume` 控制产物生命周期。

### 3. 路线选择

#### 路线 A — 保底演示线（传统 COLMAP + 3DGS）

```bash
colmap automatic_reconstructor --workspace_path data/colmap_ws --image_path data/frames_filtered/

cd third_party/gaussian-splatting
python train.py -s ../../data/colmap_ws
```

#### 路线 B — 课题创新线（LongSplat 无位姿）

```bash
# 使用 LongSplat 官方 train_custom.sh 工作流
cd third_party/LongSplat
bash train_custom.sh ../../data/frames_filtered/
```

### 4. 模型后处理

```bash
# SuperSplat 浏览器清理 https://playcanvas.com/supersplat
# 或 splat-transform 批量压缩
npx splat-transform compress input.ply output.compressed.ply
```

## 项目目录结构

```
exhibition-3dgs/
├── data/               # 原始视频、帧、深度图等
├── scripts/            # 预处理和工具脚本
├── third_party/        # 开源依赖仓库 (LongSplat, MASt3R, VDA...)
├── outputs/            # 重建结果、压缩文件、渲染图
├── annotations/        # 展品标注 JSON
├── docs/               # 项目文档和实验记录
└── tests/              # 自动化测试
```

## 产物生命周期

所有预处理脚本遵循一致的输出目录协议：

- **默认行为**：拒绝写入非空目录，防止旧产物污染新结果
- `--overwrite`：清除旧产物后重新处理
- `--resume`：校验输入指纹和参数后复用已有产物（指纹不一致时拒绝）

每阶段写入机器可读的 provenance 元数据（`_provenance.json`），供下游阶段验证输入完整性。

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
- [NopeRoomGS 论文](docs/2437_NopeRoomGS_Indoor_3D_Gaus.pdf)
