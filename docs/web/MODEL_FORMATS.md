# 模型交付与 SuperSplat 交接

## v1 模型合同

第一版交付标准 Gaussian PLY pass-through，不进行坐标变换或格式猜测。PLY header 的 vertex element 至少需要：

- `x`、`y`、`z`
- 一个或多个 `f_dc*`
- `opacity`
- 一个或多个 `scale*`
- 一个或多个 `rot*`

服务端只读取 header/property 做 schema 检查，再计算完整文件的 SHA-256、尺寸和 vertex count。普通点云 PLY、缺少属性的 PLY 或 symlink 文件都不能作为模型 artifact。

公开 PLY 必须和 `published_ply.json` receipt 绑定；receipt 的 published identity、文件 SHA、尺寸、vertex count 和当前文件必须一致。`technical_delivery_manifest` 的自动技术门禁也必须通过，才会产生 `complete` 快照。

## 质量状态的含义

- `accepted_by_automated_policy`：canonical pipeline 的自动技术交付通过。
- `gaussian_schema_valid`：published PLY 满足标准 Gaussian 属性合同。
- `supersplat_format_compatible`：交付 evidence 声明格式兼容；不是运行时加载证明。
- `supersplat_runtime_verified`：第一版默认 `false`，不会由打开外部链接自动改写。
- `manual_visual_review`：第一版默认 `false`，不会由 Web 页面自动改写。

## SuperSplat

默认操作：下载 `published-ply`，打开 SuperSplat 后使用 Import 或拖入 PLY。若 `VITE_PUBLIC_API_BASE_URL` 指向浏览器可访问的 HTTP(S) API，结果页才会生成 `https://superspl.at/editor?load=...` 形式的链接。localhost、绝对本地路径和未配置的 URL 不会被拼入外部链接。

`.compressed.ply`、`.splat`、`.sog` 和其它 viewer converter 必须等仓库提供真实 converter、独立 manifest 和校验合同后，才能加入 artifact allowlist。
