# Third-Party Dependencies

This directory contains external repositories used by the exhibition 3DGS research pipeline.

The root Git repository intentionally does not track third-party source trees. Recreate them with shallow clones when needed.

## Depth prior (`research/depth-prior`)

Default backend: **Video Depth Anything**, relative **Base (`vitb`)**, pin documented in `configs/depth/backend_pin.md`.

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything.git third_party/Video-Depth-Anything
git -C third_party/Video-Depth-Anything checkout 4f5ae23172ba60fd7bc11ef671cca678842c7072
```

Download only the Base relative checkpoint into `third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitb.pth`.

## Other backends (optional)

```bash
git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git third_party/gsplat
git clone --depth 1 https://github.com/playcanvas/supersplat.git third_party/supersplat
git clone --depth 1 https://github.com/naver/mast3r.git third_party/mast3r
```

Keep local dependency changes inside the dependency repository itself, or document required patches before committing them to the root project.

Depth-prior currently requires one documented local patch on Video Depth Anything
(`utils/dc_utils.py` matplotlib 3.9+ colormap compat). See `configs/depth/backend_pin.md`
and `configs/depth/patches/vda_matplotlib_colormap.md`. The orchestrator applies it
automatically; do not commit the patched third_party tree.

