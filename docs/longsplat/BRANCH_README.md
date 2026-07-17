# research/longsplat-route 分支说明

> 面向协作者的分支级文档。模块 API 细节见同目录 [README.md](README.md)(英文)。
> 分支最新提交:`4568099` | 更新日期:2026-07-17

## 1. 分支定位

本分支交付 **LongSplat 路线**的完整训练编排层:以预处理分支(`feature/preprocess-video`)产出的帧 manifest 为输入,调用锁定版本的 [NVlabs/LongSplat](https://github.com/NVlabs/LongSplat) 完成免位姿(pose-free)3DGS 重建,输出标准 3DGS PLY 及完整的 run record。

本仓库只包含**编排层代码**;真正执行训练的 `train.py` 位于 third_party 的 LongSplat 仓库(锁定 commit,不入库,见 §6)。

## 2. 训练脚本入口(常见问题)

**本分支没有多个独立训练脚本**,只有一套分层模块、两个可发起训练的入口:

| 入口 | 位置 | 说明 |
|---|---|---|
| `run_pipeline()` | `scripts/longsplat/orchestrator.py` | **完整入口(推荐)**。串起全流程:manifest 校验 → 输入准备 → 训练 → 3DGS 转换 → PLY 清洗与校验 → run record 落盘 |
| `run_training()` / `run_conversion()` | `scripts/longsplat/runner.py` | 底层构件,`run_pipeline` 内部调用它们。单独使用时**不含**输入校验、PLY 清洗校验和 run record,仅用于调试 |

两者均为库函数(无 `if __name__ == "__main__"`),通过 `python -c` 或自写启动文件调用,示例见 §5。

## 3. 交付清单

```
scripts/longsplat/
├── orchestrator.py      # 完整管线编排(唯一推荐训练入口)
├── runner.py            # 训练/转换命令构建与子进程执行、后端 commit 校验、配置加载
├── validate_input.py    # 输入 manifest 契约校验
├── manifest_adapter.py  # 预处理 manifest(生产者格式)→ 本模块消费格式适配
├── prepare_input.py     # 帧文件复制到 run 目录(不可变输入快照)
├── convert.py           # LongSplat 检查点 → 标准 3DGS PLY 转换、NaN/Inf 清洗、PLY 校验
└── run_record.py        # run record 状态机(planned → running → complete/failed)

configs/longsplat/
├── smoke.json               # 冒烟配置(~100 iter,验证管线通路)
├── full.json                 # 完整训练配置(30000 iter)
└── environment.example.json  # 环境配置模板(LongSplat 仓库路径、python 解释器)

tests/integration/longsplat/   # 6 个 CPU 集成测试(命令构建、契约、失败路径、适配器等)
tests/gpu/longsplat/           # GPU 端到端测试说明(手动执行)

docs/longsplat/
├── README.md                  # 模块 API 文档(英文)
├── BRANCH_README.md           # 本文档
└── handoff_out_of_scope_2026-07-12.md  # 范围外文件移交记录
```

## 4. 管线阶段与实测耗时

LongSplat 训练分五个阶段(197 帧、384×512、RTX 3060 Laptop 实测,总计约 3.5 小时):

| 阶段 | 内容 | 实测耗时 |
|---|---|---|
| MASt3R 初始化 | 帧间匹配与初始几何 | 60~90 min |
| Init | 初始窗口优化(full.json 默认 3000 iter)| 短 |
| Incremental | 逐帧扩展:每新帧 pose 200 + local 400 + global 900 iter | 主要耗时 |
| Post | 全局精调 20000 iter | ~40 min |
| Convert + 校验 | 检查点 → 标准 3DGS PLY、NaN/Inf 清洗、属性校验 | ~27 min |

注意:**训练中途不落盘检查点**,只有完整跑完才有产出;中断即从头再来。规划机时请按上表预估。

## 5. 快速开始

```bash
# 1. 复制环境配置并填写本机路径
cp configs/longsplat/environment.example.json configs/longsplat/environment.json

# 2. 先冒烟验证管线通路(约几分钟)
python -c "
from scripts.longsplat.orchestrator import run_pipeline
from scripts.longsplat.runner import load_config

config = load_config('configs/longsplat/smoke.json')
raise SystemExit(run_pipeline(
    manifest_path='data/manifests/<video>/preprocess_manifest.json',
    segment_id='<segment_id>',
    config=config,
    repo_root='<LongSplat 仓库路径>',
    output_dir='outputs/<video>',
    python_exe='<LongSplat 环境的 python>',
))
"

# 3. 冒烟通过后换 full.json 正式训练
```

产出位于 `outputs/<video>/<run_id>/`:
- `input/` — 输入帧不可变快照
- `longsplat_model/converted_3dgs/point_cloud.ply` — 最终 3DGS 点云
- `reconstruction_run.json` — run record(配置快照、输入 SHA-256、各阶段退出码与耗时、产物哈希)

## 6. 后端锁定与环境

- LongSplat 锁定 commit:`19750775a9d19f30aa05a8333c4c6c231b2d5f4a`,`runner._check_repo()` 执行前校验(不匹配时警告)
- LongSplat 需独立 Python 环境(CUDA、submodules 编译),与本仓库编排层环境分离;通过 `python_exe` 参数指定
- 编排层自身仅依赖标准库 + `plyfile` + `numpy`

## 7. 输入 / 输出契约

**输入**:预处理分支产出的 `preprocess_manifest.json`(schema `"1.0"`),经 `manifest_adapter` 适配。必需字段:`segment_id`、逐帧 `frame_id/path/width/height/sha256`。

**输出**:标准 3DGS PLY,顶点属性 `x y z`、`f_dc_0..2`、`opacity`、`scale_0..2`、`rot_0..3`(SH degree 3 时另有 `f_rest_*`)。转换时执行 NaN/Inf 顶点清洗(曾实测清除 26% 无效顶点,原因是 LongSplat 转换器对未收敛高斯点输出非法 scale)。

## 8. 已验证结果与已知问题

**已验证**(2026-07-17,墙面视频 197 帧):管线端到端跑通,产出 322,958 顶点、80.1MB、零 NaN/Inf 的 PLY,整体重建质量良好。

**已知问题:共面物体融合。** 贴在同一面墙上的两块展示牌被融合到墙面/彼此之间。根因:共面歧义 + 重复纹理跨帧误匹配 + 增量误差累积,全局优化无法仅靠光度信号分开两个近共面平面。

**解决路线(进行中)**:引入 `research/depth-prior` 分支的单帧深度先验(Video Depth Anything)。已在相同素材上验证:展示牌在深度图中相对墙面有 3~20σ 的可分辨凸出信号。下一步是在训练循环中注入尺度-平移不变的深度 loss(软正则,权重 ≤0.1)——这需要与 depth-prior 分支协作设计消费接口。

## 9. 测试

```bash
# CPU 集成测试(无需 GPU,CI 可跑)
python -m pytest tests/integration/longsplat/ -v

# GPU 端到端(手动,需 CUDA + LongSplat 环境)
python -m pytest tests/gpu/longsplat/ -v
```
