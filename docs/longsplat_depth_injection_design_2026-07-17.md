# LongSplat 深度先验注入:调研与设计文档

> 日期:2026-07-17
> 相关分支:`research/longsplat-route`(消费端)、`research/depth-prior`(生产端)
> LongSplat 锁定 commit:`1975077`(gitee 镜像 NVlabs/LongSplat)
> 状态:**设计评审中,未实现**

---

## 1. 背景与调研原因

### 1.1 问题现象

墙面视频(197 帧,segment_0002)经 LongSplat 完整管线训练后,整体重建质量良好(322,958 顶点、零 NaN/Inf),但**贴在同一面墙上的两块展示牌位置估计错误,融合在一起**。

### 1.2 根因分析

三个因素叠加:

1. **共面歧义**:两块展示牌与墙面近似共面,光度信号(L1 + SSIM)对近共面平面的深度差不敏感;
2. **重复纹理跨帧误匹配**:展示牌外观相似,MASt3R 两视图匹配把 A 牌的特征匹配到 B 牌上;
3. **增量误差累积**:LongSplat 逐帧注册,早期误匹配随位姿链传播,全局优化(post 20000 iter)只能平滑无法纠正拓扑错误。

### 1.3 已验证的前提

在决定走深度先验路线之前,已用 `research/depth-prior` 模块(Video Depth Anything, vitb, relative)对相同素材生成逐帧深度图,并做了信号检验实验(平面拟合 + 残差分析 + 平台/斜坡判别):

- **展示牌在 VDA 深度图中相对墙面有 3~20σ 的可分辨凸出信号**(近景斜视达 20σ,远景正视 3~5σ);
- 信号跨帧连续,符合真实物体运动轨迹;
- 检测区域经人工确认确为展示牌。

结论:VDA 深度先验包含解决融合问题所需的信号,值得设计消费端注入。

## 2. 调研目的

1. 查明 LongSplat(锁定 commit)是否已有深度监督接口,避免重复造轮子;
2. 查明现有深度监督为什么没有防住展示牌融合;
3. 设计 VDA 深度注入方案,明确第三方补丁范围与编排层改动范围。

## 3. 调研结果

### 3.1 LongSplat 已有完整深度监督接口,且默认启用

| 组件 | 位置 | 说明 |
|---|---|---|
| 渲染深度 | `gaussian_renderer` → `render_pkg["depth"]` | 每次迭代随颜色一起渲染 |
| `depth_loss` | `utils/loss_utils.py:65` | **1 − Pearson 相关**,尺度+平移不变 |
| `ssi_loss` | `utils/loss_utils.py:88` | log 空间尺度不变方差 |
| `gradient_loss` | `utils/loss_utils.py:98` | 深度梯度 L1(**非尺度不变**) |
| 默认权重 | `arguments/__init__.py:144-146` | `depth_loss_weight=0.1`、`ssi_loss_weight=0.05`、`grad_loss_weight=0.02` |
| 生效范围 | `train.py:131-141, 370-380, 489-498, 688-697` | **四个阶段全部生效**:init、pose、local、post/global |

监督信号来源统一为 `viewpoint_cam.depth_map`:

- **scene 初始化**(init 窗口,默认 3 帧):来自 MASt3R 初始化的深度图(`scene/__init__.py:145`);
- **增量注册**(其余全部帧):`matcher._forward` 返回该帧 MASt3R 深度(`train.py:226`),再由 `compute_scale`(`utils/graphics_utils.py:215`,中位数/MAD 仿射)对齐到当前场景尺度(`train.py:275-276`)。

### 3.2 为什么现有深度监督没有防住融合

**MASt3R 深度同样来自两视图匹配**,与融合根因(共面歧义 + 重复纹理误匹配)同源。当匹配把 A 牌特征匹配到 B 牌时,产生的深度监督信号本身就是错的——深度 loss 开着,但在强化错误几何。

VDA 是单帧语义深度,不依赖跨帧匹配,是正交的信号源。这同时说明**注入点就是替换 `depth_map` 的来源**,现有 loss 管道可全部复用。

### 3.3 已有外部深度钩子,但存在两处断裂(死代码)

`scene/__init__.py:153-165`(注释明确写着 *"e.g., Video-Depth-Anything"*):

```python
if hasattr(args, 'depth_source') and args.depth_source != "mast3r":
    depth_dir = os.path.join(args.source_path, "depths")
    for cam in self.getTrainCameras():
        depth_file = os.path.join(depth_dir, os.path.splitext(cam.image_name)[0] + "_depth.npy")
        if os.path.exists(depth_file):
            ext_depth = torch.from_numpy(np.load(depth_file)).float().cuda()
            cam.depth_map = F.interpolate(...)
```

**断裂 1 — 参数接线错误**:`depth_source` 定义在 `OptimizationParams`(`arguments/__init__.py:147`),但 `Scene` 只接收 `ModelParams`(`train.py:73` `scene = Scene(dataset, gaussians)`)。`ModelParams` 没有 `depth_source` 属性,`hasattr` 永远为 False → 钩子**从不执行**。

**断裂 2 — 增量注册覆盖**:即使钩子生效,每个增量注册的帧在 `train.py:226` 被 `matcher._forward` 的输出覆盖 `depth_map`。外部深度只在 init 窗口(3 帧)存活,其余 194 帧仍然吃 MASt3R 深度。

### 3.4 关键陷阱:深度方向相反

| | 方向 | 语义 |
|---|---|---|
| VDA relative 输出 | **值大 = 近** | 视差型(disparity-like) |
| LongSplat `depth_map` / 渲染深度 | **值大 = 远** | 相机坐标 z 深度 |

`depth_loss` 是 1 − Pearson:负相关信号会产生接近 2 的 loss 和**反向梯度**。直接把 VDA 原始 NPZ 喂进钩子会**主动破坏训练**,不是无效而是有害。必须先做逆深度仿射对齐(见 §4.3)。

### 3.5 `depth_map` 的其他消费方(替换时必须兼容)

| 消费方 | 位置 | 要求 |
|---|---|---|
| `densify_occlusion` | `train.py:328-329` | 遮挡区域致密化,需要**场景尺度**深度 |
| 下一帧的 `compute_scale` 参照链 | `train.py:275` | 用的是 `pre_depth_map`(matcher 原始输出),与替换 `depth_map` 无冲突 |
| 分辨率适配插值 | `scene/cameras.py:142-143` | 双线性插值,数值域无要求 |

结论:只要把 VDA 深度**对齐到场景尺度的 z 深度域**再替换,所有消费方兼容。

## 4. 注入方案设计

### 4.1 总体思路

**全替换方案**:将 `viewpoint_cam.depth_map` 的来源从 MASt3R 换成"对齐到场景尺度的 VDA 深度",复用全部现成 loss 管道,不新增 loss 项、不调权重(默认 0.1/0.05/0.02 保持)。

需要两个第三方小补丁 + 编排层一处改动。

### 4.2 第三方补丁(需按后端 pin 规范记录)

**P1 — 参数接线修复**(一行级):
把 `depth_source` 从 `OptimizationParams` 移到 `ModelParams`(`arguments/__init__.py`),使 `Scene` 能读到它。CLI 参数名不变(`--depth_source`),由 ParamGroup 自动生成。

**P2 — 增量注册持久化**(约 15 行):
`train.py:276`(`compute_scale` 对齐之后)插入:若 `depth_source != "mast3r"` 且该帧外部深度文件存在,则:

1. 加载外部深度 `v`(VDA 视差,已由编排层物化为 `.npy`);
2. 以刚对齐好的 MASt3R `depth_map`(场景尺度 z 深度)为参照,在**逆深度空间**最小二乘拟合 `1/D_mast3r ≈ a·v + b`;
3. `D_vda = 1 / clamp(a·v + b, eps)`,替换 `viewpoint_cam.depth_map`。

同样的对齐逻辑封装为一个工具函数(建议放 `utils/graphics_utils.py`,与 `compute_scale` 相邻),scene 初始化钩子(§3.3)也改为调用它——现有钩子直接加载原始 npy 不做方向转换,同样有 §3.4 的符号问题。

**为什么对齐必须在第三方运行时做**:场景尺度由增量位姿链动态决定,编排层在训练前无法预知;而参照信号(对齐后的 MASt3R 深度 / 渲染深度)只在训练循环内可得。

### 4.3 对齐算法细节

```
输入:v (VDA 视差, H×W), D_ref (场景尺度 z 深度, H×W)
1. mask = isfinite(v) & (D_ref > eps)
2. 最小二乘拟合 a, b:  argmin Σ_mask ( a·v + b − 1/D_ref )²
   (可选:一轮 2.5σ 剔除后重拟合,抗展示牌区域本身的偏差)
3. D_vda = 1 / clamp(a·v + b, eps)
4. 若 a 拟合为负或相关系数过低(阈值如 |ρ|<0.3),回退到 MASt3R 深度并记录警告
```

第 4 步是安全阀:个别帧 VDA 失效时不至于注入垃圾信号。

### 4.4 编排层改动(`research/longsplat-route`,本仓库)

| 改动 | 位置 | 内容 |
|---|---|---|
| 深度物化 | `scripts/longsplat/prepare_input.py` | 新增可选步骤:读 `depth_manifest.json`,把每帧 NPZ(键 `depth`)转存为 `<source_path>/depths/<image_stem>_depth.npy`,帧名映射沿用 manifest_adapter 的 frame_id↔文件名规则 |
| 配置传参 | `configs/longsplat/*.json` | `extra_train_args` 加 `{"depth_source": "vda"}`——现有 passthrough 已支持,`runner.py` 零改动 |
| run record | 自动 | 深度目录进入输入快照,manifest SHA-256 已有机制覆盖 |

注意:物化时**不做**方向转换和对齐(保持 VDA 原始视差值),转换全部留给第三方运行时——单一职责,避免两边各转一半。

### 4.5 方案对比

| 方案 | 改动量 | 风险 | 判定 |
|---|---|---|---|
| **A. 全替换 `depth_map`**(本设计)| 第三方 ~20 行 + 编排层 1 处 | VDA 深度边缘偏软,可能轻微钝化整体几何 | **推荐先跑** |
| B. 双深度项(保留 MASt3R,另加 VDA loss)| 需改 4 处 loss 代码 + 新增权重参数 | 两个矛盾信号互相拉扯,调参成本高 | 备选,A 失败再试 |
| C. 仅替换 init 窗口(只修 P1)| 最小 | init 窗口仅 3 帧,展示牌帧大概率不在其中,预期无效 | 否决 |

### 4.6 验证计划

1. **单元验证**:对齐函数在合成数据上的往返测试(已知 a、b 可恢复);
2. **对照实验**:相同 manifest、相同 seed,跑 A 方案 vs 已有 baseline(`outputs/wall_test/longsplat_segment_0002_20260717T004850Z`,322,958 顶点),约 3.5 h;
3. **评价指标**:
   - 展示牌区域:两块牌在 PLY 中是否空间分离(对应帧相机视角下的深度渲染 vs VDA 深度的残差);
   - 整体质量回归:顶点数、NaN/Inf 计数、训练 PSNR 曲线不显著劣化;
4. **失败回退**:若整体质量下降,降 `grad_loss_weight`(非尺度不变项,对对齐误差最敏感)或转 B 方案。

## 5. 风险与开放问题

| # | 风险/问题 | 缓解 |
|---|---|---|
| 1 | 第三方补丁违反"锁定 commit 无补丁"原则 | 补丁以 diff 文件形式入库(`docs/longsplat/patches/`),pin 文档记录补丁 SHA-256;或推动协作者 fork 并重新锁定 |
| 2 | VDA 深度与训练分辨率不一致(384×512 vs 训练时 resize)| 现有钩子已做双线性插值;梯度 loss 在插值后深度上仍有效 |
| 3 | 对齐参照(MASt3R 深度)本身在展示牌区域是错的 | 拟合是全图最小二乘,展示牌只占 1~8% 像素,影响被墙面主导;§4.3 的 σ 剔除进一步抑制 |
| 4 | `ssi_loss` 的 `mask = depth > 0`:对齐后深度必须为正 | §4.3 第 3 步 clamp 保证 |
| 5 | VDA 时序扭曲(抽帧不均匀被均匀编码)导致个别帧深度质量差 | §4.3 第 4 步相关性安全阀 + 回退 |

## 6. 附录:关键代码位置速查

| 内容 | 文件:行 |
|---|---|
| 深度 loss 定义 | `utils/loss_utils.py:65,88,98` |
| 深度 loss 调用(4 阶段) | `train.py:131,370,489,688` |
| loss 权重与 `depth_source` 参数 | `arguments/__init__.py:144-147` |
| 外部深度钩子(死代码) | `scene/__init__.py:153-165` |
| 增量注册深度覆盖 | `train.py:226` |
| 场景尺度对齐 | `train.py:275-276` + `utils/graphics_utils.py:215` |
| 遮挡致密化(深度消费方) | `train.py:328-329` |
| Scene 构造(接线断点) | `train.py:73` |
| VDA 深度产出(生产端) | `data/depth/wall_test/*.npz`、`data/manifests/wall_test/depth_manifest.json` |
