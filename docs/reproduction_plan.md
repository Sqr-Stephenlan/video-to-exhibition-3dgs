# 展厅 3DGS 重建复现计划

> 基于 LongSplat + NopeRoomGS 思路，从手机视频实现无位姿室内 3DGS 重建
>
> 日期：2026-07-08

---

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 5060 Laptop (Blackwell sm_120, 8GB) |
| PyTorch | 2.11.0+cu128 |
| Python | 3.13.5 (venv) |
| 核心框架 | LongSplat (ICCV 2025) + NopeRoomGS (NeurIPS 2025, 算法参考) |

---

## 总体策略：双线并行

```
保底线（阶段1）：COLMAP + gsplat → 单展区高质量 baseline → 保证可展示
创新线（阶段2-4）：改造 LongSplat → 无位姿长视频 + 深度先验 + 分段融合
```

### 五条核心改造方向

| # | 改造 | 借鉴来源 | 说明 |
|---|------|----------|------|
| 1 | 深度损失增强（SSI + Gradient） | NopeRoomGS 的 Marigold 深度先验 | 弱纹理区域光度梯度为零，需深度正则 |
| 2 | Video-Depth-Anything 替代 MASt3R 深度 | NopeRoomGS 的 foundation model depth prior | VDA 时序一致性优于单帧 MASt3R 深度 |
| 3 | 弱纹理自适应权重 | 自主设计 | 纹理梯度 → 低纹理区深度 loss ×3 |
| 4 | 启用 appearance embedding | NeRF-W 思路 | 补偿手机自动曝光造成的颜色漂移 |
| 5 | 分段重建 + 跨段融合 | NopeRoomGS 的 local-to-global | 展厅长视频无法一次训练，必须分段 |

---

## 阶段化实施

### 阶段 0：烟雾测试（预计 1-2 天）

**目标**：验证 LongSplat 在自定义视频上全流程跑通。

1. 拍 15-20 秒室内短视频
2. 编译 LongSplat CUDA 子模块（simple-knn / diff-gaussian-rasterization / fused-ssim）
3. 下载 MASt3R 权重
4. 抽帧 512px，跑标准 LongSplat 命令
5. 检查 PLY 和渲染结果

### 阶段 1：保底路线（预计 2-3 天）

COLMAP SfM → gsplat 训练 → 记录 PSNR/SSIM baseline

### 阶段 2：LongSplat 核心改造（预计 1-2 周）

**P0 必须修改**：

| 文件 | 修改 |
|------|------|
| `LongSplat/utils/loss_utils.py` | 新增 `ssi_loss()`, `gradient_loss()` |
| `LongSplat/arguments/__init__.py` | 新增 `ssi_loss_weight`, `grad_loss_weight`, `depth_source` 参数 |
| `LongSplat/train.py` | 四个训练阶段均添加 SSI/Gradient loss 分支 |
| `LongSplat/scene/__init__.py` | 支持从外部 `.npy` 文件加载 VDA 深度 |

**P1 推荐修改**：

| 文件 | 修改 |
|------|------|
| `LongSplat/utils/graphics_utils.py` | 新增 `compute_texture_mask()` |
| `LongSplat/train.py` L234 | solvePnPRansac: reprojError 5→8, iterationsCount 200→300 |
| `LongSplat/train.py` L197-198 | max_pnp_retries 10→15 |

**关键参数调优**：

| 参数 | 标准值 | 展厅值 | 原因 |
|------|--------|--------|------|
| window_size | 5 | 8 | 慢速移动需更大窗口 |
| pose_iteration | 200 | 300 | 弱纹理 PnP 多迭代 |
| local_iter | 400 | 600 | 更多局部优化 |
| global_iter | 900 | 1500 | 长视频全局一致性 |
| post_iter | 20000 | 30000 | 改善细节 |
| voxel_size | 0.1 | 0.15 | 避免弱纹理区过度细分 |
| appearance_dim | 0 | 32 | 启用曝光补偿 |
| n_offsets | 10 | 5 | 节省显存 |

### 阶段 3：分段重建（预计 1-2 周）

```
长视频 → PySceneDetect 分段 (30-40s/段, 15-20帧重叠)
  → 每段独立 LongSplat（阶段2改造版）
  → MASt3R 跨段对齐（利用重叠帧, Sim(3) 变换）
  → PLY 合并 + 去重 → gsplat 全局 fine-tune
```

新增：`scripts/align_segments.py`, `scripts/run_video_depth.py`

### 阶段 4：完整展厅（预计 1-2 周）

3-5 分钟长视频 → 分段 → 每段跑改造版 LongSplat → 对齐融合 → 后处理

---

## 显存控制策略

| 措施 | 节省量 |
|------|--------|
| 预抽帧到 512px | 避免训练时重复缩放 |
| `--n_offsets 5` | 锚点子 Gaussian 减半 |
| `--voxel_size 0.2` | 减少锚点总数 |
| MASt3R bfloat16 | ~1GB |
| 分段训练 | 每段仅需当前窗口显存 |

---

## 预期风险

| 风险 | 概率 | 应对 |
|------|------|------|
| MASt3R 弱纹理匹配失败 | 中 | 选有纹理帧作为起始；启用 VDA 深度补充 |
| 位姿漂移累积 | 高 | 增大 window_size 和 global_iter；定期重做 MASt3R 全局对齐 |
| 反光展柜深度异常 | 中 | 高光掩码降权；多角度互补 |
| 分段接缝 | 高 | 重叠帧保留原始文件名；接缝处 alpha 衰减；全局 fine-tune |
| 8GB 显存 OOM | 中 | n_offsets=5, feat_dim=16, voxel_size=0.2, ratio=2 |

---

## 验证标准

| 阶段 | 标准 |
|------|------|
| 烟雾测试 | PLY 生成、渲染可辨认 |
| 保底线 | PSNR > 25, SSIM > 0.8 |
| 创新线 | 位姿轨迹连续、无大面积空洞、PSNR > 23 |
| 分段融合 | 接缝无明显双影、颜色平滑 |
| 全展厅 | 所有展区完整、展示端 FPS > 20 |
