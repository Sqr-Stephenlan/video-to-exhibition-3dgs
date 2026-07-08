# 展厅 3DGS 架构文档关键审核意见

审核对象：[exhibition_3dgs_pipeline_framework.md](/Users/stephenlan/Documents/3DGS/exhibition_3dgs_pipeline_framework.md)  
用途：供小组逐条核对原架构文档、外部项目说明与后续修订方向。  
日期：2026-07-07

---

## 总体结论

原架构文档的总体方向合理：视频采集与预处理、无位姿/弱位姿几何、3DGS 训练、后处理、展示和标注都覆盖到了课题要求。

但文档目前偏“技术路线综述”，对若干关键项目的工程风险写得不够明确。需要重点修订以下 7 处：

1. NopeRoomGS 来源必须确认。
2. LongSplat 适合作研究主线，但不能未经验证就作为唯一 MVP。
3. MASt3R / DUSt3R + gsplat 不是即插即用，需要中间格式。
4. Video Depth Anything 只能作为深度先验，不能直接保证几何补全。
5. 分段融合是最大工程风险。
6. OBJ/mesh 不是 3DGS 的自然输出。
7. MVP 应改为“保底演示线 + 课题创新线”。

---

## 1. NopeRoomGS 来源需要更明确

**原文指向**

- [第 4 行：NopeRoomGS 可作为算法概念参考](/Users/stephenlan/Documents/3DGS/exhibition_3dgs_pipeline_framework.md:4)

**审核判断**

原文判断“不可作为唯一复现仓库”基本正确，但还应写得更明确。课题要求中写了“3DGS 建模算法：NopeRoomGS”，如果小组无法拿到论文、源码或私有实现，就不能把它当作可复现主算法。

**外部核验链接**

- GitHub 检索 `NopeRoomGS`：https://github.com/search?q=NopeRoomGS&type=repositories
- GitHub 检索 `"NoPeRoomGS"`：https://github.com/search?q=%22NoPeRoomGS%22&type=repositories
- GitHub 检索 `"Nope Room GS"`：https://github.com/search?q=%22Nope+Room+GS%22&type=repositories
- 可替代参考 NoPoSplat：[NoPoSplat README 第 29 行说明其从 unposed sparse images 预测 3D Gaussians](https://github.com/cvg/NoPoSplat/blob/main/README.md?plain=1#L29)
- 可替代参考 NoPe-NeRF：[NoPe-NeRF 论文页面](https://arxiv.org/abs/2212.07388)

**建议修订**

```text
NopeRoomGS 当前需补充可核验来源。若无公开源码或课题方提供实现，本文档将其作为算法概念参考，不作为唯一工程主线。工程实现暂采用 LongSplat、MASt3R、Video Depth Anything、gsplat、SuperSplat 等公开可运行项目组合。
```

---

## 2. LongSplat 主路线合理，但风险偏低估

**原文指向**

- [第 183-222 行：主选路线 LongSplat](/Users/stephenlan/Documents/3DGS/exhibition_3dgs_pipeline_framework.md:183)

**审核判断**

把 LongSplat 作为研究主线是合理的。它确实面向无位姿长视频 3DGS，并使用 MASt3R、增量联合优化和 octree anchoring。

但原文对工程难度写得偏乐观。LongSplat 需要 CUDA 和多个子模块编译；官方对自定义视频建议约 10 fps，长视频可缩到 512px 宽以加速。原文中 “1080p 或 1600px 长边” 的建议应区分场景：官方 3DGS baseline 可用较高分辨率，LongSplat 初跑应先低分辨率验证。

**外部核验链接**

- LongSplat 定位：[README 第 29 行，unposed long-video 3DGS、MASt3R、incremental joint optimization、octree anchoring](https://github.com/NVlabs/LongSplat/blob/main/README.md?plain=1#L29)
- 安装复杂度：[README 第 57-63 行，conda 与 simple-knn / diff-gaussian-rasterization / fused-ssim 子模块](https://github.com/NVlabs/LongSplat/blob/main/README.md?plain=1#L57-L63)
- 自定义视频建议：[README 第 117-124 行，约 10 fps、长视频 512px 宽、保证 overlap](https://github.com/NVlabs/LongSplat/blob/main/README.md?plain=1#L117-L124)
- 3DGS 转换：[README 第 139 行，`convert_3dgs.py`](https://github.com/NVlabs/LongSplat/blob/main/README.md?plain=1#L139)

**建议修订**

```text
LongSplat 是本课题最贴近“无位姿长视频 3DGS”的研究主线，但应先作为实验路线验证。初跑建议使用 10-20 秒短视频、约 10 fps、低分辨率输入，跑通后再扩展到单展区和完整展厅。
```

---

## 3. MASt3R / DUSt3R + gsplat 不是即插即用

**原文指向**

- [第 226-240 行：MASt3R / DUSt3R + gsplat 备选路线](/Users/stephenlan/Documents/3DGS/exhibition_3dgs_pipeline_framework.md:226)

**审核判断**

这条备选路线成立，但原文需要补充“接口层”。MASt3R 是 3D matching / reconstruction 工具，不是完整的一键式展厅 3DGS 重建系统。它输出的匹配、点图、相机关系需要转换成 3DGS 可用的数据格式。

MASt3R 仓库也明确说明，部分使用 MASt3R matches 做 COLMAP / GLOMAP 标准 SfM 的脚本是 “toys”，未充分测试，可能在边缘场景失败。

**外部核验链接**

- MASt3R 定位：[README 第 3-4 行，Grounding Image Matching in 3D with MASt3R](https://github.com/naver/mast3r/blob/main/README.md?plain=1#L3-L4)
- MASt3R-SfM 说明：[README 第 177-184 行，部分 COLMAP/GLOMAP 脚本是 toys、未充分测试、可能失败](https://github.com/naver/mast3r/blob/main/README.md?plain=1#L177-L184)
- Demo 能力边界：[README 第 187-193 行，demo 面向 small/larger scenes 的 sparse global alignment](https://github.com/naver/mast3r/blob/main/README.md?plain=1#L187-L193)

**建议修订**

文档应增加明确中间格式与脚本边界：

```text
MASt3R / DUSt3R 输出
  ↓
相机内参、外参、点云、匹配置信度、尺度、坐标系
  ↓
COLMAP-compatible cameras/images/points3D 或 gsplat transforms
  ↓
gsplat / 官方 3DGS 训练
```

建议新增脚本责任：

```text
mast3r_to_colmap.py
mast3r_to_gsplat.py
```

---

## 4. Video Depth Anything 只能作为深度先验

**原文指向**

- [第 260-292 行：Depth Anything / Video Depth Anything 深度先验路线](/Users/stephenlan/Documents/3DGS/exhibition_3dgs_pipeline_framework.md:260)

**审核判断**

使用 Video Depth Anything / Depth Anything V2 做弱纹理深度先验是合理的，但不能写成“补全几何”的确定能力。它能提供长视频一致深度和 relative / metric depth，但单目深度仍会存在尺度、边界、反光表面、多视角一致性问题。

应把它定义为：

```text
深度正则 / 初始化候选 / 低置信度区域辅助
```

而不是直接作为可靠几何真值。

**外部核验链接**

- Video Depth Anything 定位：[README 第 19 行，arbitrarily long videos、consistency、generalization](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/README.md?plain=1#L19)
- 模型类型：[README 第 80 行，robust and consistent video depth estimation](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/README.md?plain=1#L80)
- relative / metric depth：[README 第 109-115 行](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/README.md?plain=1#L109-L115)
- metric 模型训练域：[README 第 126 行，metric depth models trained on Virtual KITTI and IRS datasets](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/README.md?plain=1#L126)
- streaming 模式风险：[README 第 133 行，training/testing gap 导致性能下降](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/README.md?plain=1#L133)

**建议修订**

增加置信度过滤：

```text
1. 反光、玻璃、屏幕区域降低深度权重。
2. 图像边缘和遮挡边界降低深度权重。
3. 多帧深度不一致区域不直接用于几何初始化。
4. 与 MASt3R / COLMAP 点云冲突的深度不作为硬约束。
5. 若需真实尺度，加入展品尺寸、地面标尺或 AR 设备尺度辅助。
```

---

## 5. 分段融合是最大工程风险

**原文指向**

- [第 454-500 行：片段融合与完整展厅场景合并](/Users/stephenlan/Documents/3DGS/exhibition_3dgs_pipeline_framework.md:454)

**审核判断**

原文提到 Sim(3) 对齐是正确的，但还不够。展厅长视频分段后，真正困难的是全局一致性：pose graph、loop closure、重叠帧策略、重复 Gaussian 合并、颜色/曝光一致化、失败片段回退。

室内手机 3DGS 案例 LighthouseGS 也说明，手机室内拍摄会遇到窄基线、纹理弱、位姿漂移、自动曝光等问题，需要几何先验、平面结构、姿态修正和光度修正。

**外部核验链接**

- LighthouseGS 项目页：https://vision3d-lab.github.io/lighthousegs/
- LighthouseGS 对室内手机风险的说明：页面摘要中提到 single mobile device、textureless indoor scenes、rough geometric priors、indoor planar structures、motion drift、auto-exposure correction
- splat-transform 可合并/转换 splats：[README 第 12-18 行，convert/edit/merge splats](https://github.com/playcanvas/splat-transform/blob/main/README.md?plain=1#L12-L18)
- splat-transform 合并示例：[README 第 332-336 行](https://github.com/playcanvas/splat-transform/blob/main/README.md?plain=1#L332-L336)

**建议修订**

将“分段融合”提升为核心研究模块，并补充：

```text
1. 相邻片段保留 30%-50% 重叠关键帧。
2. 使用 MASt3R / DUSt3R 建立跨片段匹配。
3. 估计相邻片段 Sim(3)。
4. 多片段构建 pose graph。
5. 闭环路径做 loop closure。
6. 合并后删除重复 Gaussian 和漂浮噪点。
7. 做颜色/曝光归一化。
8. 匹配失败时允许人工锚点或局部重采集。
```

---

## 6. 输出格式需要区分 3DGS 与 mesh / OBJ

**原文指向**

- [第 512-519 行：模型精简与导出格式](/Users/stephenlan/Documents/3DGS/exhibition_3dgs_pipeline_framework.md:512)

**审核判断**

`.ply / .splat / .compressed.ply / .sog` 对 3DGS 展示和压缩是合理的。但 OBJ 不是 3DGS 的自然输出。3DGS 的 PLY 通常保存 Gaussian 参数，不等同于传统 mesh；OBJ 需要单独的 mesh 提取或碰撞几何生成路线。

**外部核验链接**

- SuperSplat Viewer 支持格式：[README 第 27 行，`.ply`、`.sog`、`.compressed.ply`、`.meta.json`、`.lod-meta.json`](https://github.com/playcanvas/supersplat-viewer/blob/main/README.md?plain=1#L27)
- splat-transform 格式表：[README 第 55-88 行，PLY / SOG / compressed.ply / glb / voxel 等](https://github.com/playcanvas/splat-transform/blob/main/README.md?plain=1#L55-L88)
- collision mesh 支持：[splat-transform README 第 218 行，生成 `.collision.glb`](https://github.com/playcanvas/splat-transform/blob/main/README.md?plain=1#L218)
- voxel/collision 说明：[splat-transform README 第 366-368 行](https://github.com/playcanvas/splat-transform/blob/main/README.md?plain=1#L366-L368)
- AGS-Mesh 项目页：https://xuqianren.github.io/ags_mesh_website/
- AGS-Mesh 对深度/法线先验风险的说明：页面摘要中提到 mobile depth 质量有限、monocular estimators 存在 poor multi-view consistency，需要 adaptive filtering

**建议修订**

将输出分成三类：

| 类型 | 格式 | 用途 |
|---|---|---|
| 视觉主模型 | `.ply`、`.splat`、`.compressed.ply`、`.sog` | SuperSplat / PlayCanvas 展示 |
| 辅助几何 | `.glb`、`.obj`、voxel / collision mesh | 导航、碰撞、拾取、空间标注 |
| 标注数据 | `.json` | 展品点位和交互信息 |

若验收必须包含 OBJ/mesh，应新增：

```text
3DGS → mesh / voxel / collision geometry
```

可调研 TSDF、SuGaR、2DGS/AGS-Mesh 或 splat-transform collision mesh。

---

## 7. MVP 不应只押 LongSplat，应改为双线

**原文指向**

- [第 886-908 行：当前最推荐的最小可行路线](/Users/stephenlan/Documents/3DGS/exhibition_3dgs_pipeline_framework.md:886)

**审核判断**

“最小可行路线直接 LongSplat”偏研究导向，验收风险较高。更稳妥的做法是拆成两条线：

**保底演示线**

```text
COLMAP / Nerfstudio Splatfacto / 官方 3DGS
  ↓
SuperSplat 清理
  ↓
PlayCanvas / SuperSplat Viewer 展示
```

**课题创新线**

```text
LongSplat
  ↓
MASt3R 辅助位姿/匹配
  ↓
Video Depth Anything 深度先验
  ↓
分段融合实验
```

**外部核验链接**

- 官方 3DGS 仓库：https://github.com/graphdeco-inria/gaussian-splatting
- Nerfstudio 仓库：https://github.com/nerfstudio-project/nerfstudio
- gsplat 仓库：https://github.com/nerfstudio-project/gsplat
- SuperSplat 仓库：https://github.com/playcanvas/supersplat
- SuperSplat Viewer 仓库：https://github.com/playcanvas/supersplat-viewer
- PlayCanvas Engine 仓库：https://github.com/playcanvas/engine
- LongSplat 仓库：https://github.com/NVlabs/LongSplat
- MASt3R 仓库：https://github.com/naver/mast3r
- Video Depth Anything 仓库：https://github.com/DepthAnything/Video-Depth-Anything

**建议修订**

```text
MVP-A：保底演示线
短视频/单展区 → COLMAP 或 Nerfstudio → 3DGS/Splatfacto → SuperSplat → PlayCanvas → JSON 标注

MVP-B：课题创新线
短视频/单展区 → LongSplat → MASt3R/Video Depth Anything 辅助 → 输出 PLY/SOG → 与 MVP-A 对比
```

这样既能保证小组有可展示结果，也能覆盖“无位姿、弱纹理、长视频、分段融合”的研究难点。

---

## 建议回填到原架构文档的最小修改清单

1. 在第 4 行附近新增“`NopeRoomGS` 来源待确认，不作为唯一主线”。
2. 在 LongSplat 小节补充官方建议：约 10 fps、长视频 512px、CUDA/子模块、先短视频 smoke test。
3. 在 MASt3R / DUSt3R + gsplat 小节补充中间格式和转换脚本责任。
4. 在 Video Depth Anything 小节把“弱纹理补全”改为“深度先验/正则/辅助”，并加入置信度过滤。
5. 在分段融合小节新增 pose graph、loop closure、重叠帧、重复 Gaussian、曝光一致化和失败回退。
6. 在模型导出小节明确 3DGS PLY 与 OBJ/mesh 的区别。
7. 将 MVP 路线改为“保底演示线 + 课题创新线”。

