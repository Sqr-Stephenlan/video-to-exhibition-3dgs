# 一键把视频转换为标准 3DGS PLY

本文是当前 production checkout 的用户使用说明。它只描述正式的一键链：
一个本地视频文件或一个直接指向视频文件的 HTTP(S) URL，最终得到一个标准
3DGS PLY。它不是历史研究路线、VDA/深度实验或手工点云拼接的操作手册。

## 先看结论

当前入口一次只接受一个视频输入：

```bash
cd <production-or-checkout>
./video-to-3dgs /abs/path/video.mp4
./video-to-3dgs 'https://example.invalid/video.mp4'
```

第二个例子必须是“直接返回视频文件”的 URL，不是视频网站页面。重定向后的
最终地址也必须仍是 `http` 或 `https`。网页 URL 不会被偷偷解析；需要网页适配
时应由另行受控的可选适配器处理，`yt-dlp` 不属于本核心依赖闭包。

多视频融合、跨视频配准和展厅级拼接目前尚未落地。不要把多个视频先拼成一个
文件来冒充多视频融合，也不要把多个独立点云直接拼在一起当作本链的输出。

## 获取仓库与锁定后端

正式根仓库和 production 实现必须来自包含 `video-to-3dgs` 的 checkout。仓库地址
和 submodule 来源以当前 checkout 的 Git 配置为准；新 checkout 可按下面的顺序准备：

```bash
git clone https://github.com/Sqr-Stephenlan/video-to-exhibition-3dgs.git <checkout>
cd <checkout>
git checkout <ref-that-contains-video-to-3dgs>
git submodule sync -- third_party/LongSplat
git submodule update --init -- third_party/LongSplat
```

不要把某次 checkout 的 branch 或 SHA 当作永久手工 checkout 目标。根仓库的
`third_party/LongSplat` gitlink、runner 中的锁和 `.gitmodules` 中的 fork 来源才是
当前可复现状态；可以用下面的命令读取它们：

```bash
git config -f .gitmodules --get submodule.third_party/LongSplat.url
git config -f .gitmodules --get submodule.third_party/LongSplat.branch
git rev-parse HEAD:third_party/LongSplat
./dev.sh python -c "from scripts.longsplat import runner; print(runner.LONGSPLAT_COMMIT)"
```

推荐用仓库自带的分发检查确认远端、根 gitlink、runner lock 和四个直接 gitlink
一致：

```bash
./dev.sh python -m scripts.longsplat.verify_remote_distribution --root . --json
```

该检查是显式的 direct-submodule 检查，不会为默认一键链递归拉取可选的
MASt3R/DUSt3R 后代。新 checkout 不要手工再初始化这些可选后代；nested fork
保留 `submodules/mast3r` 这个锁定 gitlink，不等于默认训练/渲染/转换会加载
MASt3R 或 DUSt3R 几何路线。

## 环境分层

一键链故意分成四层。它们不是把所有依赖塞进同一个 Python 环境：

| 层 | 用途 | 需要的内容 |
| --- | --- | --- |
| checkout `.venv` | root route、视频契约、COLMAP/camera staging、PLY 校验、进度与发布 | `requirements-route.txt` 的 `numpy`、`opencv-python`、`plyfile`；CPU 测试另加 `pytest` |
| 独立 CUDA backend | LongSplat 的 train/render/convert 与 GPU converted-eval | Python/CUDA/PyTorch/torchvision、锁定 fork 自身 requirements、四个已编译直接 gitlink；不要安装到 checkout `.venv` |
| media tools | 视频探测、规范化和抽帧 | `ffmpeg`、`ffprobe`；优先使用 workspace 下的 `backend-envs/media-tools/bin/` |
| 系统与驱动 | COLMAP 和 GPU 启动条件 | `colmap` 在 `PATH` 或 provider 配置中；WSL2 的 NVIDIA 驱动/CUDA 能力供 GPU backend 使用 |

标准 workspace 布局会从 checkout 的祖先目录自动发现：

```text
<workspace>/backend-envs/longsplat-cu128/bin/python
<workspace>/backend-envs/media-tools/bin/ffmpeg
<workspace>/backend-envs/media-tools/bin/ffprobe
colmap                         # PATH 中的外部程序
```

独立 clone 不在这样的 workspace 目录下时，复制被 `.gitignore` 忽略的本地配置
模板并替换占位符：

```bash
cp configs/provider.local.example.json configs/provider.local.json
# 编辑 configs/provider.local.json，填写本机 backend_env 或 backend_python、
# ffmpeg、ffprobe 和可选的 colmap 路径
```

不要把本机用户名、绝对路径、GPU UUID 或某次 run 的路径写回仓库文档或依赖声明。

## 安装与检查

仓库提供了现有的 route wrapper；没有额外虚构的 bootstrap 命令。若 checkout
没有 `.venv`，确认允许安装后运行：

```bash
./dev.sh bootstrap
```

它根据 `requirements.txt` 准备 root CPU route + 测试环境；只想查看最小运行闭包
时，使用 `requirements-route.txt`。GPU backend 不由这个命令安装或升级，应使用
已经准备好的独立 CUDA 环境和锁定 nested fork。

安装或复用现有环境后，先做不运行真实媒体的检查：

```bash
./dev.sh doctor
./dev.sh python -c "import numpy, cv2, plyfile; print('route runtime: ok')"
./video-to-3dgs --help
./dev.sh python -m scripts.longsplat.verify_remote_distribution --root . --json
```

GPU backend 的检查应使用该 backend 自己的解释器，而不是把 `torch` 加进 root
`.venv`：

```bash
<backend-python> -c "import torch; print(torch.cuda.is_available())"
```

命令输出 `True` 只是说明解释器能看到 CUDA；正式一键链仍会在 GPU 阶段前执行
`nvidia-smi` 和 backend CUDA preflight。

## 默认一键链做什么

默认 profile 是 `single-convergence1000-v1`。一次运行大致经历：

```text
探测视频 → 抽帧/CPU 契约 → COLMAP 稀疏模型 → centered-PINHOLE camera staging
→ external fixed-pose RGB-only LongSplat 输入
→ convergence1000 诊断 → formal30000 训练/渲染
→ 标准 3DGS 转换 → converted-eval/postprocess → automated technical delivery → 发布
```

默认值明确是 `depth_source=disabled`、external fixed-pose、RGB-only。默认链不启用
旧的 `smoke100/coverage` GPU 路线、pose-search、VDA、MASt3R 或 DUSt3R 可选几何
路径；这些词出现在历史资料或代码兼容层中，不代表默认安装项或默认执行项。

默认一键链只接收一个视频；它不会自动把多个视频合并。相机模型、匹配器和每个
视频的帧/相机身份由当前 run 的契约记录，不能用一个全局固定相机参数代替。

## 进度、耗时与输出

默认 `--progress auto` 只在 stderr 输出观察性进度；stdout 保留机器可读的最终
契约。参数形式为 `--progress auto|plain|off`，也可以显式选择：

```bash
./video-to-3dgs --progress auto /abs/path/video.mp4
./video-to-3dgs --progress plain /abs/path/video.mp4
./video-to-3dgs --progress off /abs/path/video.mp4
```

TTY 下 `auto` 可能刷新一行；非 TTY 或 `plain` 只报告阶段变化和约一分钟一次的
心跳。它不会编造训练百分比、迭代总数或 ETA。`--plan` 只建立 CPU 计划，不启动
GPU、训练、渲染、转换或评估：

```bash
./video-to-3dgs --plan /abs/path/video.mp4
```

真实运行时间没有固定承诺：短视频可能是分钟级，较长视频、高分辨率、COLMAP
匹配规模、GPU 显存和磁盘速度都可能把运行拉长到更久甚至小时级。请把它当作
影响因素区间，而不是 SLA。

默认公共输出目录是：

```text
outputs/<video-stem>__<sha12>.ply
```

其中 `<sha12>` 是发布 PLY 的 SHA-256 前缀。成功时 stdout 按顺序包含：

```text
SUCCESS
PLY: <absolute-path-to-public-ply>
EVIDENCE: <absolute-path-to-run-evidence>
```

`PLY:` 指向独立的普通文件，供用户导入标准 3DGS viewer。内部 evidence 位于
workspace 的：

```text
<workspace>/.runtime/longsplat-runs/<run-id>/
```

它保存阶段记录、输入/配置身份、日志、checkpoint/转换证据、`run.json` 和发布
receipt，主要用于诊断、受控恢复和审计；它不是用户日常交付目录。不要删除
`.runtime` 来“清理”一次运行，也不要把其中的中间 PLY 当作最终公共输出。

## 怎样判断成功

至少同时满足以下条件：

1. 进程返回码为 0，并打印 `SUCCESS`、`PLY:`、`EVIDENCE:`。
2. `PLY:` 文件存在，是独立普通文件，路径和 SHA 前缀与发布 receipt 一致。
3. PLY 通过标准 3DGS 结构校验，包含位置、DC SH、opacity、scale 和 quaternion
   rotation 等必要字段。
4. `run.json` 的最终阶段和 automated technical delivery 记录为通过。

这表示自动化技术链和标准 PLY 契约通过，不表示已经完成 SuperSplat 或人工视觉
验收。当前产品把自动技术交付和人工视觉判断明确分开。

## 常见错误

### COLMAP blocked

COLMAP 需要能读到规范化帧，并且场景有足够纹理、重叠和可匹配视角。相机模型/匹配
器必须是当前 contract 支持的值；稀疏模型组件也必须满足当前的单一明确选择规则。
失败时先查看对应 run 的 COLMAP 命令、stderr 和 component inventory。不要手工把
多个独立模型强行合并，也不要为了绕过阻塞去改变 camera 数学或删除证据。

### WSL GPU 假阴性

如果 `nvidia-smi` 失败，或 `<backend-python>` 的 `torch.cuda.is_available()` 为
`False`，一键链会在 GPU 阶段前 fail-fast。检查 WSL2 GPU 透传、Windows/NVIDIA
驱动、CUDA/PyTorch 版本匹配和 backend 环境本身。root `.venv` 不需要也不应因为
这个错误安装 `torch`；它是 CPU route 环境。

### converted-eval 失败或需要 CPU recovery

converted-eval 失败后，先保留同一个 run 和已有阶段证据。当前实现支持在保留的
转换/渲染产物上做受控的 CPU postprocess/recovery；它不会把新的随机 run 当作原
run 的继续。依据 `run.json` 的 `next_slice` 和受控恢复接口继续，不能删除
evidence、重跑已经通过的上游阶段或手工拼一个新的 evaluation JSON。

### 磁盘、主机内存或显存不足

长视频会同时产生下载临时文件、规范化帧、COLMAP 中间文件、训练 checkpoint、渲染
和转换 evidence。预留足够磁盘和主机内存；GPU 显存还会受帧数、分辨率、模型和
backend residency 策略影响。可先使用 `--plan` 验证契约，但它不能替代真实 GPU
运行。不要通过删除 `.runtime` 或擅自修改训练/转换参数来恢复身份一致性。

### URL 不是视频文件

网页响应、非 HTTP(S) 的最终重定向、超过下载上限、`Content-Length` 不一致，或
`ffprobe` 找不到视频流都会阻塞。请提供直接视频文件 URL；URL 中的凭据、query 和
fragment 不会作为常规输入身份写入日志。

## 原地恢复已有 run

`./video-to-3dgs <input>` 是“新 run”入口，每次会分配新的 run ID；不要用它恢复
已有 run。受支持的恢复必须复用原 run 的全部身份：同一个 runtime/output root、
同一个 `run-id`、同一个源视频身份（URL 输入要使用 evidence 中记录的 canonical
下载文件）、同一个 camera/matcher/profile 和同一组 provider。

受控 operator 可以先对原 run 做 CPU-only 校验：

```bash
./dev.sh python -m scripts.longsplat.reconstruct_pipeline \
  --input-video <same-canonical-local-video> \
  --output-root <same-runtime>/longsplat-runs \
  --run-id <existing-run-id> \
  --route-root . \
  --validate-only
```

只有校验确认 identity、provider 和已有阶段证据一致后，才由同一受控接口继续
`run_reconstruction` 的下一阶段；继续 GPU 阶段时还必须显式使用原 backend/provider
配置和受控 GPU 权限。不要换新的 `run-id`、`output-root`、源视频路径、camera/
matching/profile，也不要直接手写一段 Python 去“补”缺失的 run JSON。若当前
operator/API 不在手边，应停在这里并交给维护者；不要用普通一键命令制造 identity
mismatch。

`reconvert_existing` 是隔离 checkpoint 转换工具，不是普通 run 的原地恢复入口；
没有明确的 authority manifest 和维护者指令时不要使用它。

## 当前边界与不推荐操作

当前正式默认链的边界是：

- depth disabled；
- external fixed-pose RGB-only；
- 自动化技术交付，不等于人工视觉验收；
- 单视频/单 URL 输入，不等于多视频融合；
- nested fork 和 gitlink SHA 必须保持锁定，不修改 nested LongSplat 代码或 gitlink。

不推荐：

- 拼接多个视频来伪装多视频融合；
- 独立点云硬拼；
- 为默认链擅自安装 VDA、MASt3R、DUSt3R、额外 depth 或 pose-search 包；
- 删除 `.runtime`、run evidence 或 outputs 来“解决”失败；
- 重跑已经通过的上游 stage，或绕过受控 recovery 直接改 run JSON；
- 把历史 `BRANCH_README.md`、旧 `REPRODUCTION_RUNBOOK.md` 中的 VDA/旧 smoke 命令
  当作当前 production 一键入口。
