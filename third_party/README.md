# Third-Party Dependencies

This directory contains external repositories used by the exhibition 3DGS research pipeline.

The root Git repository intentionally does not track third-party source trees. Recreate them with shallow clones when needed:

```bash
git clone --depth 1 https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git third_party/gsplat
git clone --depth 1 https://github.com/playcanvas/supersplat.git third_party/supersplat
git clone --depth 1 https://github.com/DepthAnything/Video-Depth-Anything.git third_party/Video-Depth-Anything
git clone --depth 1 https://github.com/naver/mast3r.git third_party/mast3r
```

Keep local dependency changes inside the dependency repository itself, or document required patches before committing them to the root project.

