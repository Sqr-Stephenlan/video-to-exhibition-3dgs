# 基于视频驱动的展厅 3D 高斯泼溅重建项目实现框架

> 适用课题：基于手机或 AR 眼镜视频输入，实现展厅环境的 3D Gaussian Splatting（3DGS）重建、模型精简、三维标注与 Web/查看器展示。  
> 当前判断：NopeRoomGS 可作为“算法概念参考”，但不宜作为唯一复现仓库；工程实现应采用公开可运行项目组合完成。

---

## 0. 项目总体目标

输入：

```text
手机 / AR 眼镜拍摄的展厅视频
```

输出：

```text
1. 可查看的 3DGS 场景文件，例如 .ply / .splat / .compressed.ply / .sog
2. 可在 SuperSplat / DasViewer / PlayCanvas 中展示的展厅数字孪生场景
3. 支持展品或空间点位标注的三维坐标系统
4. 可复现实验流程：采集 → 抽帧 → 位姿/深度估计 → 3DGS训练 → 后处理 → 展示
```

推荐主线：

```text
视频采集
  ↓
视频预处理：FFmpeg + OpenCV + PySceneDetect
  ↓
关键帧筛选：OpenCV + 自定义质量评分
  ↓
视频分段：PySceneDetect / 轨迹分段 / 手动分段
  ↓
无位姿初始化：LongSplat / MASt3R / DUSt3R / Depth Anything
  ↓
3DGS 训练：LongSplat 或 gsplat / 官方 3DGS
  ↓
片段融合与全局优化：LongSplat 思路 + MASt3R/DUSt3R 对齐 + 自定义坐标融合
  ↓
模型精简：SuperSplat + splat-transform
  ↓
三维标注：SuperSplat / PlayCanvas / 自定义 JSON 标注层
  ↓
模型展示：SuperSplat Viewer / PlayCanvas Engine / DasViewer
```

---

## 1. 视频采集模块

### 1.1 目标

保证后续 3DGS 重建有足够的视角覆盖、稳定相机运动和可匹配视觉信息。

### 1.2 采集规范

建议采集原则：

```text
1. 分区采集：入口区、主展区、展品近景区、墙面/天花/地面补拍区。
2. 环绕路径：围绕展品和空间边界进行多角度覆盖。
3. 速度控制：慢速平稳移动，避免快速转身。
4. 光照控制：尽量固定曝光、固定白平衡，减少自动曝光造成的颜色漂移。
5. 近远结合：全局空间视频 + 重点展品近景视频。
6. 重叠率：相邻关键帧之间保持较高视觉重叠。
7. 反光展品：多角度补拍，避免单一角度只拍到高光。
```

### 1.3 对应开源项目

这一环节主要是采集规范，不依赖特定开源项目。  
如需后续自动读取视频元数据，可使用：

- FFmpeg：读取视频帧率、分辨率、编码格式、时间戳。
- OpenCV：读取视频帧、计算模糊度、抽取关键帧。

---

## 2. 视频预处理模块

### 2.1 目标

把长视频转化为适合重建的图像序列或短片段，剔除明显无效帧。

输入：

```text
raw_video.mp4
```

输出：

```text
frames/
segments/
metadata.json
```

### 2.2 子任务

#### A. 视频格式标准化

使用 FFmpeg 完成：

```text
1. 转码为统一格式，例如 H.264/H.265 mp4。
2. 提取固定帧率图像，例如 2 fps、5 fps、10 fps。
3. 降分辨率到训练可承受范围，例如 1080p 或 1600px 长边。
4. 保留原始视频备份。
```

依赖项目：

- FFmpeg

#### B. 自动分段

展厅视频通常较长，直接整体训练容易出现：

```text
1. 位姿漂移
2. 局部失败
3. 显存压力过大
4. 弱纹理区域整体拖垮重建
```

建议先分段：

```text
raw_video.mp4
  ↓
segment_001.mp4
segment_002.mp4
segment_003.mp4
...
```

分段依据可以是：

```text
1. 场景切换 / 镜头突变：PySceneDetect
2. 时间窗口：每 20–60 秒一段
3. 轨迹语义：入口、展台 A、展台 B、走廊、墙面
4. 质量指标：连续模糊段、快速旋转段、曝光突变段
```

依赖项目：

- PySceneDetect
- FFmpeg
- OpenCV

#### C. 关键帧筛选

每段视频中提取关键帧，避免图片过密导致训练成本上升。

建议指标：

```text
1. 模糊度：Laplacian variance
2. 相邻帧相似度：SSIM / 感知哈希 / 特征匹配数量
3. 曝光异常：过曝 / 欠曝像素比例
4. 运动异常：光流过大或陀螺仪突变
5. 重叠控制：避免连续帧几乎完全重复
```

依赖项目：

- OpenCV
- scikit-image（可选，用于 SSIM）
- imagehash（可选，用于感知哈希）

---

## 3. 位姿、深度与初始几何模块

这是整个课题的关键。经典 3DGS 依赖 COLMAP 提供相机位姿和稀疏点云，但展厅场景有大量弱纹理、反光和重复结构，COLMAP 容易失败。因此本项目应采用“无位姿 / 弱位姿 / 深度先验”的路线。

---

### 3.1 主选路线：LongSplat 路线

推荐作为主复现路线。

```text
关键帧序列
  ↓
MASt3R / learned 3D prior 估计深度和对应关系
  ↓
增量式位姿估计
  ↓
Octree anchor 初始化 Gaussians
  ↓
相机位姿 + 3D Gaussians 联合优化
  ↓
长视频场景重建
```

适合本课题原因：

```text
1. 面向 casually captured long videos。
2. 不要求预先给定准确相机位姿。
3. 适合“展厅绕行一圈”的长视频输入。
4. 包含增量重建、位姿估计、Gaussian 优化和内存控制思想。
5. 与课题中的“自动分段处理”和“场景完整融合”高度匹配。
```

依赖项目：

- LongSplat
- MASt3R
- DUSt3R
- gsplat / 3DGS backend

建议定位：

```text
LongSplat = 本项目最接近“视频驱动 + 无位姿 + 长序列 + 3DGS”的主参考实现。
```

---

### 3.2 备选路线：MASt3R / DUSt3R + gsplat

如果 LongSplat 难以稳定跑通，可以拆成更可控的组合路线：

```text
关键帧序列
  ↓
DUSt3R / MASt3R
  ↓
获得相机位姿、稠密点图、对应关系
  ↓
转换为 3DGS 初始化输入
  ↓
gsplat / 官方 3DGS 训练
```

适合用途：

```text
1. 作为可控实验路线。
2. 替代 COLMAP 的初始化。
3. 对弱纹理区域比传统 SfM 更友好。
4. 便于把每个模块拆开调试。
```

依赖项目：

- DUSt3R
- MASt3R
- gsplat
- 官方 3DGS

---

### 3.3 深度先验路线：Depth Anything / Video Depth Anything

对于墙面、地面、天花板、展厅空白区域，可以额外引入单目或视频深度估计：

```text
关键帧
  ↓
Depth Anything V2 / Video Depth Anything
  ↓
深度图
  ↓
辅助 Gaussian 初始化 / 深度正则 / 弱纹理补全
```

适合解决：

```text
1. 白墙、地面、天花板等弱纹理区域。
2. 近似平面区域的几何稳定性。
3. COLMAP 或匹配模型失败区域的补偿。
4. 长视频深度时序一致性。
```

依赖项目：

- Depth Anything V2
- Video Depth Anything

注意：

```text
单目深度通常存在尺度不确定性。若要做展厅真实尺寸测量，需要额外尺度约束，例如已知展品尺寸、地面标尺、AR 设备深度/IMU 或手动标定点。
```

---

### 3.4 基线路线：COLMAP + 官方 3DGS

虽然课题偏向视频驱动和 No-pose 室内重建，但仍建议保留 COLMAP 作为基线。

```text
关键帧
  ↓
COLMAP SfM
  ↓
相机位姿 + 稀疏点云
  ↓
官方 3DGS / gsplat
```

用途：

```text
1. 与传统 3DGS 结果对比。
2. 检查数据采集质量。
3. 作为论文/报告中的 baseline。
4. 如果某些展厅区域纹理足够丰富，COLMAP 反而可能更稳定。
```

依赖项目：

- COLMAP
- 官方 3DGS
- gsplat
- Nerfstudio / Splatfacto（可选）

---

## 4. 3D Gaussian Splatting 训练模块

### 4.1 目标

把位姿、图像和初始点云/深度转换为最终的 3DGS 表示。

输出：

```text
point_cloud.ply
cameras.json / transforms.json
training_logs/
rendered_views/
metrics.json
```

---

### 4.2 主训练框架选择

#### A. LongSplat

当输入是长视频、位姿未知、需要分段/增量融合时，优先考虑 LongSplat。

适合：

```text
1. 从视频直接构建完整展厅。
2. 长序列。
3. 无准确位姿。
4. 需要联合优化相机位姿与 Gaussians。
```

#### B. gsplat

当你需要二次开发时，优先使用 gsplat。

适合：

```text
1. 自己写训练脚本。
2. 加深度正则。
3. 加曝光/颜色补偿。
4. 加展品标注或分割信息。
5. 用 PyTorch 更灵活地调试。
```

#### C. 官方 3DGS

作为理论学习和最小基线使用。

适合：

```text
1. 理解原始 3DGS 训练流程。
2. 建立 baseline。
3. 验证数据格式是否正确。
```

#### D. Nerfstudio / Splatfacto

作为工程化训练与可视化备选。

适合：

```text
1. 快速跑通 NeRF/3DGS pipeline。
2. 使用 Nerfstudio 的数据处理工具。
3. 需要 Web 可视化训练过程。
```

---

## 5. 光照变化、反光与弱纹理处理模块

### 5.1 问题

展厅常见干扰：

```text
1. 灯光强，局部高光明显。
2. 展柜玻璃、金属、屏幕有反光。
3. 白墙、地板、天花板弱纹理。
4. 手机自动曝光/自动白平衡导致同一物体颜色漂移。
5. 行人或工作人员造成动态遮挡。
```

### 5.2 工程处理思路

#### A. 采集侧处理

```text
1. 锁定曝光和白平衡。
2. 避免边走边快速旋转。
3. 对反光展品多角度补拍。
4. 对弱纹理区域加入斜角拍摄和边缘覆盖。
5. 分区采集，避免一次长视频承担全部场景。
```

#### B. 预处理侧处理

```text
1. 删除严重模糊帧。
2. 删除过曝帧。
3. 对曝光突变片段单独分段。
4. 对动态遮挡帧降权或剔除。
```

#### C. 训练侧处理

```text
1. 使用深度先验增强弱纹理区域。
2. 使用分段训练降低局部失败扩散。
3. 对不同片段做颜色归一化。
4. 必要时引入 Splatfacto-W / appearance embedding 思路处理光照差异。
```

依赖项目：

- OpenCV
- Depth Anything / Video Depth Anything
- gsplat
- Splatfacto-W / Nerfstudio（参考）

---

## 6. 片段融合与完整展厅场景合并

### 6.1 目标

课题明确要求“自动分段处理”和“场景完整融合”。这一步是工程难点。

### 6.2 推荐策略

#### A. 优先用 LongSplat 的增量长视频思路

如果 LongSplat 能直接处理完整长视频，优先避免手动融合。

```text
完整长视频
  ↓
LongSplat 增量优化
  ↓
统一坐标系下的完整 3DGS 场景
```

#### B. 分段训练 + 位姿图融合

如果必须分段：

```text
segment_001 → local_splat_001
segment_002 → local_splat_002
segment_003 → local_splat_003
```

需要建立片段间对齐关系：

```text
1. 相邻片段保留重叠帧。
2. 使用 MASt3R / DUSt3R 获取跨片段匹配。
3. 估计 Sim(3) 变换：尺度 + 旋转 + 平移。
4. 把各段 splat 统一到全局坐标系。
5. 再进行全局 fine-tune 或后处理裁剪。
```

依赖项目：

- MASt3R / DUSt3R
- LongSplat
- Open3D（可选，用于点云配准和可视化）
- gsplat
- splat-transform

---

## 7. 模型精简与导出模块

### 7.1 目标

课题要求在保证视觉质量的前提下控制模型数据量，并输出通用格式。

### 7.2 处理内容

```text
1. 删除漂浮噪点。
2. 裁剪展厅外部无效区域。
3. 降低冗余 Gaussian 数量。
4. 压缩颜色、协方差、透明度等属性。
5. 导出 .ply / .splat / .compressed.ply / .sog 等格式。
6. 必要时额外导出 OBJ/mesh 作为辅助碰撞或导航几何。
```

### 7.3 主工具

#### A. SuperSplat

适合人工或半自动处理：

```text
1. 查看模型质量。
2. 手动删除噪点。
3. 裁剪边界。
4. 编辑相机视角。
5. 输出可分享场景。
```

#### B. splat-transform

适合自动化批处理：

```text
1. 格式转换。
2. 坐标变换。
3. 数据过滤。
4. 统计分析。
5. 压缩和导出。
```

依赖项目：

- SuperSplat
- splat-transform
- PlayCanvas 工具链

---

## 8. 三维场景标注模块

### 8.1 目标

支持在重建的展厅数字孪生中标注展品、展区、路径或特殊位置。

输入：

```text
3DGS 场景
展品清单 / 手动点击点 / 图像识别结果
```

输出：

```text
annotations.json
```

建议标注格式：

```json
[
  {
    "id": "item_001",
    "name": "展品A",
    "type": "object",
    "position": [1.23, 0.85, -2.40],
    "rotation": [0, 0, 0],
    "description": "展品说明文字",
    "media": "images/item_001.jpg"
  }
]
```

### 8.2 实现方式

#### A. 手动标注

```text
SuperSplat / PlayCanvas 中点击位置
  ↓
记录世界坐标
  ↓
生成 annotations.json
```

#### B. 半自动标注

```text
关键帧中检测展品
  ↓
通过位姿和深度反投影到 3D
  ↓
在 3DGS 场景中生成标注点
```

可选依赖项目：

- OpenCV
- Segment Anything / GroundingDINO（如需图像分割和目标定位）
- PlayCanvas Engine
- SuperSplat Viewer

### 8.3 注意

3DGS 本身主要是视觉表示，不天然等同于可编辑 CAD/mesh 模型。标注系统建议作为单独的 JSON 图层叠加到 3DGS 场景上，而不是直接写入 splat 文件内部。

---

## 9. 模型展示模块

### 9.1 本地查看

适合调试阶段：

```text
1. SuperSplat Editor
2. DasViewer
3. 官方 SIBR viewer
4. Nerfstudio viewer
```

### 9.2 Web 展示

适合课题验收和数字孪生演示：

```text
3DGS file
  ↓
SuperSplat / splat-transform 转换
  ↓
PlayCanvas Engine
  ↓
Web 页面展示
  ↓
可选 WebXR / AR 浏览
```

依赖项目：

- SuperSplat
- SuperSplat Viewer
- PlayCanvas Engine
- splat-transform

### 9.3 展示端功能建议

```text
1. 固定漫游路线。
2. 自由视角查看。
3. 展品点击标注。
4. 小地图或楼层平面图。
5. 展区导航。
6. 切换“原始模型 / 精简模型”对比。
7. 显示模型统计信息：Gaussian 数量、文件大小、FPS。
```

---

## 10. 推荐复现顺序

### 阶段 1：建立基础认识

```text
1. 跑通官方 3DGS 示例。
2. 理解输入数据格式：images、cameras、sparse point cloud。
3. 用 SuperSplat 打开官方输出的 PLY。
```

依赖：

- 官方 3DGS
- COLMAP
- SuperSplat

### 阶段 2：搭建视频预处理流水线

```text
1. FFmpeg 抽帧。
2. PySceneDetect 自动分段。
3. OpenCV 模糊度检测。
4. 生成 frames_manifest.json。
```

依赖：

- FFmpeg
- PySceneDetect
- OpenCV

### 阶段 3：复现长视频 / 无位姿路线

```text
1. 复现 LongSplat 示例。
2. 用一段短展厅/室内视频测试。
3. 观察位姿估计、Gaussian 初始化和最终渲染质量。
```

依赖：

- LongSplat
- MASt3R
- DUSt3R

### 阶段 4：建立备选重建路线

```text
1. DUSt3R / MASt3R 单独估计几何。
2. 转换到 gsplat 或官方 3DGS 可读格式。
3. 与 LongSplat 结果对比。
```

依赖：

- DUSt3R
- MASt3R
- gsplat

### 阶段 5：后处理与展示

```text
1. SuperSplat 清理模型。
2. splat-transform 自动压缩和转换。
3. PlayCanvas 加载模型。
4. 添加 annotations.json 标注层。
```

依赖：

- SuperSplat
- splat-transform
- PlayCanvas Engine

---

## 11. 项目目录建议

```text
exhibition-3dgs/
│
├── README.md
├── docs/
│   ├── capture_guide.md
│   ├── pipeline_design.md
│   ├── reconstruction_notes.md
│   └── evaluation_protocol.md
│
├── data/
│   ├── raw_videos/
│   ├── segments/
│   ├── frames/
│   ├── masks/
│   ├── depth/
│   └── manifests/
│
├── scripts/
│   ├── extract_frames.py
│   ├── split_video.py
│   ├── filter_blurry_frames.py
│   ├── build_manifest.py
│   ├── run_depth.py
│   ├── run_reconstruction.py
│   ├── align_segments.py
│   └── export_splats.py
│
├── third_party/
│   ├── gaussian-splatting/
│   ├── gsplat/
│   ├── LongSplat/
│   ├── dust3r/
│   ├── mast3r/
│   ├── Depth-Anything-V2/
│   ├── Video-Depth-Anything/
│   ├── supersplat/
│   └── splat-transform/
│
├── outputs/
│   ├── reconstructions/
│   ├── splats/
│   ├── compressed/
│   ├── renders/
│   └── web_demo/
│
└── annotations/
    ├── annotations.json
    └── item_metadata.csv
```

---

## 12. 各环节与开源项目对应表

| 环节 | 主选项目 | 备选项目 | 作用 |
|---|---|---|---|
| 视频转码/抽帧 | FFmpeg | OpenCV | 视频读取、抽帧、转码 |
| 视频分段 | PySceneDetect | FFmpeg 时间切片 | 自动切分长视频 |
| 模糊检测/帧筛选 | OpenCV | scikit-image / imagehash | 剔除低质量帧 |
| 深度估计 | Video Depth Anything | Depth Anything V2 / Metric3D | 弱纹理区域深度先验 |
| 无位姿几何估计 | MASt3R | DUSt3R | 相机位姿、匹配、点图 |
| 长视频无位姿 3DGS | LongSplat | 分段 MASt3R + gsplat | 主重建路线 |
| 基线 SfM | COLMAP | GLOMAP | 传统相机位姿和稀疏点云 |
| 3DGS 训练 | gsplat | 官方 3DGS / Nerfstudio Splatfacto | Gaussian 优化训练 |
| 模型编辑 | SuperSplat | DasViewer | 清理、裁剪、检查 |
| 模型转换/压缩 | splat-transform | SuperSplat | 批量转换、压缩、过滤 |
| Web 展示 | PlayCanvas Engine | SuperSplat Viewer | 数字孪生展示 |
| 三维标注 | PlayCanvas + JSON | SuperSplat + 自定义脚本 | 展品标注、位置定位 |

---

## 13. 项目地址清单

1. FFmpeg  
   https://github.com/FFmpeg/FFmpeg

2. OpenCV  
   https://github.com/opencv/opencv

3. PySceneDetect  
   https://github.com/Breakthrough/PySceneDetect

4. COLMAP  
   https://github.com/colmap/colmap

5. 官方 3D Gaussian Splatting  
   https://github.com/graphdeco-inria/gaussian-splatting

6. gsplat  
   https://github.com/nerfstudio-project/gsplat

7. Nerfstudio  
   https://github.com/nerfstudio-project/nerfstudio

8. DUSt3R  
   https://github.com/naver/dust3r

9. MASt3R  
   https://github.com/naver/mast3r

10. Depth Anything V2  
    https://github.com/DepthAnything/Depth-Anything-V2

11. Video Depth Anything  
    https://github.com/DepthAnything/Video-Depth-Anything

12. LongSplat  
    https://github.com/NVlabs/LongSplat

13. NoPoSplat  
    https://github.com/cvg/NoPoSplat

14. PF3plat  
    https://github.com/cvlab-kaist/PF3plat

15. SelfSplat  
    https://github.com/Gynjn/selfsplat

16. SuperSplat  
    https://github.com/playcanvas/supersplat

17. SuperSplat Viewer  
    https://github.com/playcanvas/supersplat-viewer

18. splat-transform  
    https://github.com/playcanvas/splat-transform

19. PlayCanvas Engine  
    https://github.com/playcanvas/engine

---

## 14. 当前最推荐的最小可行路线

如果只做一条优先级最高的 MVP，不建议一开始堆太多模型。推荐：

```text
手机视频
  ↓
FFmpeg 抽帧
  ↓
OpenCV 模糊帧过滤
  ↓
PySceneDetect 分段
  ↓
LongSplat 重建
  ↓
SuperSplat 清理
  ↓
splat-transform 压缩/转换
  ↓
PlayCanvas 展示
  ↓
JSON 标注层
```

这条路线最贴合课题中的：

```text
1. 视频作为唯一输入
2. 长视频自动分段
3. 弱纹理/无位姿室内重建
4. 模型精简输出
5. 数字孪生展示
6. 三维场景标注
```

---

## 15. 风险与备选方案

### 风险 1：LongSplat 环境复杂或难以跑通

备选：

```text
MASt3R / DUSt3R 提取几何
  ↓
gsplat / 官方 3DGS 训练
```

### 风险 2：弱纹理区域仍然破碎

备选：

```text
Video Depth Anything / Depth Anything V2 提供深度先验
```

### 风险 3：分段模型融合困难

备选：

```text
1. 缩短采集路径，减少分段数量。
2. 保证片段之间有 30% 以上重叠。
3. 使用 MASt3R / DUSt3R 做跨片段对齐。
4. 先完成单区域高质量重建，再扩展到全展厅。
```

### 风险 4：PLY 文件太大，展示端卡顿

备选：

```text
1. SuperSplat 手动清理。
2. splat-transform 批处理压缩。
3. PlayCanvas 使用压缩格式。
4. 分区加载，而不是一次加载全场景。
```

### 风险 5：OBJ 输出与 3DGS 不完全等价

说明：

```text
3DGS 的主要输出是 splat/PLY 类型的 Gaussian 表示，不是传统 mesh。
OBJ 更适合作为辅助几何、碰撞体或导航网格，不应作为主要视觉模型格式。
```

---

## 16. 最终验收建议

建议准备三组实验：

### A. 单展品小场景

目标：

```text
验证采集、抽帧、重建和查看全流程。
```

### B. 单展区中等场景

目标：

```text
验证弱纹理、反光和局部融合问题。
```

### C. 完整展厅长视频

目标：

```text
验证自动分段、全局融合、模型压缩和数字孪生展示。
```

每组记录：

```text
1. 原始视频时长
2. 抽帧数量
3. 有效关键帧数量
4. 重建耗时
5. GPU 型号和显存占用
6. Gaussian 数量
7. 原始模型大小
8. 压缩后模型大小
9. 展示端 FPS
10. 主观视觉质量
11. 失败区域截图和原因分析
```
