# 展厅 3DGS 项目分支与任务说明

本文档用于说明当前仓库的任务分支设计，帮助组员理解每条分支的职责、先后关系和交付物。项目主仓库定位为总控编排层，负责文档、脚本、配置、数据清单、实验记录和展示入口；`third_party/` 下的开源项目作为可替换执行后端，不直接纳入主仓库源码管理。

## 一、总体协作原则

1. `main` 只保存稳定成果、已确认文档和可复现实验入口。
2. 所有开发任务从 `main` 派生到 `docs/*`、`feature/*` 或 `research/*` 分支。
3. 第三方仓库源码保留在 `third_party/`，如需改动第三方项目，应在对应子仓库内部处理，并在主仓库文档中记录必要 patch 或运行参数。
4. 大体量视频、帧、深度图、模型、日志和训练输出不提交到 Git，按 `.gitignore` 放入 `data/`、`outputs/` 或本地外部存储。
5. 每个实验都应记录输入视频、抽帧数量、有效关键帧、训练命令、输出模型、失败区域和质量指标。

## 二、推荐管线

```text
手机 / AR 眼镜视频采集
  ↓
视频标准化、抽帧、分段、关键帧筛选
  ↓
相机位姿、匹配、深度先验与初始几何估计
  ↓
3D Gaussian Splatting 重建训练
  ↓
片段坐标对齐、场景融合与全局一致化
  ↓
模型裁剪、去噪、压缩与格式导出
  ↓
三维标注 JSON 图层
  ↓
SuperSplat / PlayCanvas / Web 展示验证
```

## 三、分支说明

| 分支 | 主要职责 | 关键子仓库或工具 | 预期交付物 |
|---|---|---|---|
| `main` | 稳定基线、已确认方案、可复现入口 | 无固定后端 | 稳定文档、目录骨架、合并后的脚本 |
| `docs/pipeline-design` | 管线设计、任务拆分、仓库协作规范 | 架构文档、技术方案 | 管线说明、分支说明、模块边界 |
| `feature/preprocess-video` | 视频标准化、抽帧、分段、关键帧筛选 | FFmpeg、OpenCV、PySceneDetect | `segments/`、`frames/`、`frames_manifest.json`、质量筛选报告 |
| `feature/baseline-colmap-3dgs` | 保底演示线：传统 SfM + 3DGS | COLMAP、gaussian-splatting、gsplat | 单展区基线模型、训练日志、SuperSplat 可查看结果 |
| `research/longsplat-route` | 课题创新线：无位姿长视频 3DGS | LongSplat、MASt3R | 自定义视频 LongSplat 跑通记录、标准 3DGS PLY 导出 |
| `research/depth-prior` | 弱纹理深度先验与置信度过滤 | Video Depth Anything | 深度图、深度 manifest、弱纹理辅助策略 |
| `feature/segment-fusion` | 多片段对齐、Sim(3) 估计、完整展厅融合 | MASt3R、splat-transform、Open3D 可选 | 片段变换矩阵、融合模型、对齐质量记录 |
| `feature/export-viewer-annotation` | 模型压缩、展示端和标注层 | SuperSplat、splat-transform、PlayCanvas engine | `.sog` / `.compressed.ply`、Web demo、`annotations.json` |
| `docs/evaluation-reporting` | 实验指标、对比评估和验收材料 | 渲染脚本、表格模板 | 实验记录、模型大小/FPS/质量指标、失败案例分析 |

## 四、两条主线

### 1. 保底演示线

```text
视频
  ↓
FFmpeg / OpenCV / PySceneDetect
  ↓
COLMAP
  ↓
官方 3DGS 或 gsplat
  ↓
SuperSplat 清理
  ↓
splat-transform 压缩
  ↓
PlayCanvas / SuperSplat 展示
```

这条线优先保证可展示结果，适合单展品、单展区和纹理较丰富的测试视频。

### 2. 课题创新线

```text
视频
  ↓
关键帧筛选
  ↓
LongSplat / MASt3R
  ↓
无位姿 3DGS 重建
  ↓
标准 PLY 导出
  ↓
压缩、展示、对比评估
```

这条线对应课题的研究重点：手机或 AR 眼镜视频输入、未知相机位姿、长视频、室内弱纹理和完整展厅重建。

## 五、建议开发顺序

1. `docs/pipeline-design`：先统一文档、目录约定和任务边界。
2. `feature/preprocess-video`：先得到可复用的分段、抽帧和关键帧清单。
3. `feature/baseline-colmap-3dgs`：尽快跑通保底演示线。
4. `research/longsplat-route`：并行验证无位姿长视频路线。
5. `research/depth-prior`：针对弱纹理和白墙、地面、天花板补充深度先验。
6. `feature/segment-fusion`：在至少两个片段都能重建后再做融合。
7. `feature/export-viewer-annotation`：在有稳定模型后完成压缩、Web 展示和标注。
8. `docs/evaluation-reporting`：贯穿记录实验，但最终集中整理验收材料。

## 六、每条任务分支的最小完成标准

| 分支 | 最小完成标准 |
|---|---|
| `feature/preprocess-video` | 输入一个短视频后，可以生成片段、关键帧和 manifest |
| `feature/baseline-colmap-3dgs` | 一个单展区样例能通过 COLMAP + 3DGS 输出可查看 PLY |
| `research/longsplat-route` | 一个短视频样例能按 LongSplat 自定义输入流程训练并导出 PLY |
| `research/depth-prior` | 一个视频或关键帧序列能输出深度图，并记录相机/帧对应关系 |
| `feature/segment-fusion` | 至少两个局部片段能估计相对变换并输出合并结果 |
| `feature/export-viewer-annotation` | 能把模型转换为展示格式，并加载独立 JSON 标注层 |
| `docs/evaluation-reporting` | 能按统一模板记录模型质量、大小、耗时和展示性能 |

## 七、数据与输出约定

```text
data/raw_videos/       原始视频
data/segments/         视频分段
data/frames/           抽帧结果
data/depth/            深度先验
data/masks/            掩码或低置信度区域
data/manifests/        结构化清单

outputs/reconstructions/  训练目录和重建中间结果
outputs/splats/           标准 3DGS / splat 输出
outputs/compressed/       压缩模型和 Web 友好格式
outputs/renders/          渲染对比图和视频
outputs/web_demo/         展示端构建结果
```

## 八、组员认领建议

1. 熟悉视频处理的组员优先认领 `feature/preprocess-video`。
2. 熟悉传统 SfM 或 3DGS 的组员优先认领 `feature/baseline-colmap-3dgs`。
3. 熟悉深度学习环境和 CUDA 的组员优先认领 `research/longsplat-route` 与 `research/depth-prior`。
4. 熟悉几何、配准或点云处理的组员优先认领 `feature/segment-fusion`。
5. 熟悉前端或交互展示的组员优先认领 `feature/export-viewer-annotation`。
6. 负责汇报材料的组员持续维护 `docs/evaluation-reporting`。

## 九、风险提示

1. LongSplat、MASt3R、Video Depth Anything 都依赖较复杂的 GPU / CUDA / PyTorch 环境，应先用短视频 smoke test。
2. 单目深度不能直接等同于真实几何，只能作为深度先验或训练正则。
3. 分段融合是最大工程风险，需要保留重叠帧、记录 Sim(3) 变换，并允许人工锚点或局部重采集作为兜底。
4. 3DGS 的 `.ply` 保存的是 Gaussian 参数，不等同于传统 OBJ/mesh；如果需要碰撞或导航几何，需要额外生成辅助 mesh 或 voxel。
5. 展示端优先使用 `.sog`、`.compressed.ply` 或其他 Web 友好格式，不建议直接加载超大原始 PLY。
