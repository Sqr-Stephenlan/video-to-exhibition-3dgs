# 框架文档 vs 复现计划 —— 对比与风险分析

> 对照 exhibition_3dgs_pipeline_framework.md（下称"框架"）与 exhibition_3dgs_architecture_review.md（下称"审核"），
> 结合实际代码探索结果，补充和修正风险评估。
>
> 日期：2026-07-08

---

## 一、框架总体评价

框架文档的 16 节结构完整，覆盖了采集到展示的全链条。审核文档的 7 条修正意见全部合理且必要。

**但**两文档均写于代码实际探索之前。经过对 LongSplat / MASt3R / VDA / gsplat 源码的深入探索，以下 7 个风险需要更精确的评估。

---

## 二、逐项风险对比

### 风险 1：LongSplat 不直接等于 NopeRoomGS 替代品

| 维度 | 框架/审核的认知 | 实际代码发现 |
|------|----------------|-------------|
| 深度损失 | 审核未提及深度损失不足的问题 | LongSplat 仅有 **Pearson 相关系数**深度损失（`loss_utils.py:63-83`），缺乏尺度不变（SSI）和梯度一致性损失。在弱纹理区域这严重不足 |
| 弱纹理应对 | 框架 §3.3 将 VDA 列为"辅助 Gaussian 初始化/弱纹理补全" | LongSplat **无任何弱纹理专项参数或逻辑**，全局搜索 `textureless`/`low_texture` 返回零结果 |
| 对比基线 | 框架将 NopeRoomGS 列为"算法概念参考" | NopeRoomGS 的四个核心机制（LNGR/PPA/AOS/DAL）中，LongSplat 仅天然具备 AOS（交替优化），其余三个**完全缺失** |

**结论**：框架对 LongSplat 的描述偏乐观，它只是一个无位姿长视频基线，不是针对室内弱纹理的解决方案。
**修正**：必须在 loss、深度源、PnP 参数三方面做实质性改造。

---

### 风险 2：MASt3R / DUSt3R + gsplat 的中间格式问题比审核更严重

审核文档 §3 已指出需要 `mast3r_to_colmap.py` 脚本。实际探索后发现还有额外问题：

| 问题 | 详情 |
|------|------|
| DUSt3R 权重**已被移除** | DUSt3R 仓库 README 明确提示 checkpoint 暂时无法下载，意味着 DUSt3R 目前**不可用** |
| MASt3R 许可限制 | 使用 MASt3R 权重需同意 MapFree 等数据集的**极严格许可**，商用项目需慎重 |
| MASt3R 全局对齐输出格式不标准 | `global_align()` 返回 `pts3d`(稀疏) 和 `world2cam`(4×4 RT)，非 COLMAP 原生格式；需要额外转换坐标系（OpenGL ↔ COLMAP） |
| "toy" SfM 脚本确如审核所言不可靠 | `demo_glomap.py` 和 `kapture_mast3r_mapping.py` 的 README 明确标注"未充分测试，可能在边缘场景失败" |

**结论**：审核的判断"不是即插即用"是正确的。实际情况更严峻：DUSt3R 当前不可用，MASt3R 全局对齐的鲁棒性在室内弱纹理场景未经充分验证。

**修正**：应**优先采用 LongSplat 内置的 MASt3R 集成**（它在 `train.py` 中已经做了 pairwise matching + PnP 的组合，比单独的 MASt3R 全局对齐更鲁棒）。

---

### 风险 3：Video Depth Anything 的运行成本被低估

| 维度 | 框架的认知 | 实际 |
|------|-----------|------|
| 模型大小 | 框架未区分 | Small 28M / Base 113M / Large 382M（Apache/NC/NC 不同许可） |
| Large 模型显存 | 框架未提及 | **23.6 GB**（FP16），远超 RTX 5060 的 8GB |
| Small 模型 | 未提及 | 6.8 GB + Apache 2.0 许可，**是唯一可行的选项** |
| 流式模式风险 | 审核 §4 已提及 | 流式模式 d1 从 0.926 降到 0.836，精度下降显著 |
| metric depth 域偏移 | 框架未提及 | metric 模型在 Virtual KITTI + IRS 训练，与展厅场景有域差距 |
| 推理时间 | 框架未提及 | Large 14ms/clip，处理 3 分钟 5fps 视频（900帧, ~28 clips）约 0.4 秒；但 I/O + 后处理远超此 |

**结论**：框架将 VDA 列为"弱纹理补全"过于乐观。RTX 5060 只能跑 Small 模型，且仅能作为**深度先验/正则项**，不能作为几何真值。

**修正**：
1. 明确指定使用 Video-Depth-Anything-**Small**
2. 深度仅用于 loss 正则，不作为初始化点云
3. 需置信度过滤（反光/边界/多帧不一致区域）

---

### 风险 4：分段融合 —— 审核已标为"最大风险"，但工程链路更长

审核 §5 给出的 8 条建议（pose graph / loop closure / Sim(3) 等）是正确的。实际实现时还有：

| 新增问题 | 详情 |
|----------|------|
| 坐标尺度不一致 | LongSplat 每段独立训练，MASt3R 估计的尺度不同。分段间的 Sim(3) 不仅包含旋转平移，还有**尺度因子** |
| PLY 合并并非简单拼接 | 两个 PLY 的坐标系对齐后，重叠区域的 Gaussians 需要**去重和融合**，而非简单 `cat` |
| 全局 fine-tune 的位姿问题 | 合并后的场景，位姿是每段独立优化的。全局 fine-tune 时要么固定所有位姿（可能锁死），要么重新优化（可能再次漂移） |
| LongSplat 锚点模型局限性 | LongSplat 输出的是 anchor+MLP 表示，需 `convert_3dgs.py` 转标准 3DGS 格式后才能被 gsplat/SuperSplat 消费 |

**修正**：分段融合必须设计为独立的阶段，包含：
1. 重叠帧 MASt3R pairwise matching
2. Sim(3) 估计（尺度 + 旋转 + 平移）
3. 全局 pose graph 优化
4. PLY 坐标变换 + 重叠区去重（基于空间距离阈值）
5. `convert_3dgs.py` 转标准格式
6. gsplat 全局 fine-tune

---

### 风险 5：框架推荐的 MVP 路线中有隐含瓶颈

框架 §14 的推荐路线：

```
手机视频 → FFmpeg 抽帧 → 模糊过滤 → PySceneDetect 分段 → LongSplat 重建 → SuperSplat 清理 → splat-transform → PlayCanvas → JSON 标注
```

审核 §7 已建议双线。进一步发现的问题：

| 环节 | 隐含瓶颈 |
|------|---------|
| PySceneDetect 分段 | 展厅无场景切换（连续行走），PySceneDetect 很可能**切不出段**，回退到固定时间窗口 |
| LongSplat 重建（最长链路） | 需要编译 3 个 CUDA 子模块 + 下载 MASt3R 权重 + 调试参数。如果没有烟雾测试保底，**可能数天跑不出第一个 PLY** |
| splat-transform | 是 npm 工具，需要 Node.js 环境。Windows 上 npm 的安装复杂度被忽略 |
| PlayCanvas | 是 WebGL 引擎，需要前端开发能力部署。与 Python 技术栈完全不同 |

**结论**：审核的双线建议非常正确。保底线（COLMAP + gsplat）应在第一天就跑通，确保整个下游链路（PLY → SuperSplat → PlayCanvas）可用。

---

### 风险 6：NopeRoomGS 中真正能借鉴的部分有限

复现计划从 NopeRoomGS 借鉴了 5 条，但实际可行性比预想低：

| 借鉴项 | 可行性 | 原因 |
|--------|--------|------|
| SSI + Gradient loss | **高** | 纯 Python 实现，20 行代码 |
| Video-Depth-Anything 深度先验 | **中** | 需要下载模型 + 批量推理脚本，显存受限 |
| 弱纹理自适应权重 | **高** | 基于图像梯度，10 行代码 |
| 分段平面性假设 (PPA) | **低** | 需要 per-pixel 法向量估计 + 跨帧平面参数变换 + 边缘掩码。实现复杂，且 LongSplat 为 anchor+MLP 架构不直接支持 per-pixel 操作 |
| 交替优化 (AOS) | **无需** | LongSplat 已经是联合优化（每次 iter 同时更新 pose 和 Gaussian），不需要额外实现 |
| CoTracker 多帧跟踪 | **低** | 需要集成 CoTracker 模型 + 修改训练流程。收益不确定，短期不建议 |

**结论**：真正可落地的 NopeRoomGS 借鉴只有 3 条（SSI loss, VDA depth, texture mask），其余要么不需要（AOS），要么成本过高（PPA, CoTracker）。

---

## 三、框架文档应更新的内容

基于以上分析，建议框架文档做以下更新：

### 更新 1：LongSplat 小节补充已知限制

在 §3.1 末尾新增：

```text
已知限制：
1. 深度损失仅为 Pearson 相关，弱纹理区域无效 → 需改造增加 SSI + Gradient loss
2. 无弱纹理专项处理逻辑 → 需增加纹理掩码自适应权重
3. 新帧注册仅匹配前一帧 → 弱纹理需放宽 PnP 参数
4. 锚点模型输出非标准 PLY → 需 convert_3dgs.py 转换
```

### 更新 2：Video Depth Anything 明确模型选择和定位

在 §3.3 中明确：

```text
GPU 8GB 仅能用 Small (28M, Apache 2.0)
定位：深度正则/辅助，不作为几何真值
需配合：置信度过滤 + SSI loss（解决尺度不确定性）
```

### 更新 3：分段融合升级为核心攻关模块

在 §6 中增加完整工程链路（pose graph → Sim(3) → 去重 → 全局 fine-tune）。

### 更新 4：MVP 改为阶段化双线

将 §14 的单条路线替换为：

```text
阶段 0：烟雾测试 (LongSplat 标准参数, 短视频)
阶段 1：保底线 (COLMAP + gsplat, 单展区)
阶段 2：创新线改造 (LongSplat 改造版, 单展区)
阶段 3：分段融合 (多段对齐合并, 2-3 展区)
阶段 4：完整展厅
```

---

## 四、风险矩阵总结

| 风险 | 框架原评估 | 实际评估 | 优先级 |
|------|-----------|---------|--------|
| LongSplat 弱纹理处理不足 | 未提及 | **高** — 必须改造 loss + 深度源 | P0 |
| LongSplat 环境编译复杂 | 审核 §2 提及 | **高** — 3 CUDA 模块 + MASt3R 权重 | P0 |
| DUSt3R 权重不可用 | 未提及 | **高** — 当前无法使用，需依赖 MASt3R | P0 |
| VDA 只能用 Small 模型 | 未提及 | **中** — 8GB 显存约束，但 Small 已足够 | P1 |
| 分段融合工程链路长 | 审核 §5 提及 | **高** — 6 步链路，每步都可能失败 | P0 |
| MASt3R 许可限制 | 未提及 | **中** — 学术使用 OK，商用需谨慎 | P2 |
| LongSplat 输出格式非标 | 未提及 | **中** — 需 convert_3dgs.py 转换 | P1 |
| splat-transform 需 Node.js | 未提及 | **低** — 仅影响最后展示环节 | P2 |
| CoTracker 多帧跟踪 | 未提及 | **高** — 可暂不做，等基线跑通再评估 | P3 |
| PPA 分段平面性假设 | 未提及 | **高** — 实现复杂，优先度最低 | P3 |
