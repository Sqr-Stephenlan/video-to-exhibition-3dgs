# LongSplat VDA depth injection patches

> Branch: `research/longsplat-route`
> LongSplat base: NVlabs/LongSplat pinned commit `19750775a9d19f30aa05a8333c4c6c231b2d5f4a`
> Date: 2026-07-22
> Status: **applied locally, not upstreamed**

## Applying

Apply in order:

```bash
cd third_party/LongSplat
git apply ../../docs/longsplat/patches/longsplat_stability_fixes.patch
git apply ../../docs/longsplat/patches/longsplat_training_improvements.patch
git apply ../../docs/longsplat/patches/vda_depth_injection.patch
git apply ../../docs/longsplat/patches/longsplat_conversion_telemetry.patch
git -C submodules/mast3r apply ../../../../docs/longsplat/patches/mast3r_low_memory_load.patch
```

## Patch 0 — Stability fixes (pre-requisite)

**File:** `docs/longsplat/patches/longsplat_stability_fixes.patch`

Prior baseline-training fixes applied before VDA injection:

| File | Fix |
|---|---|
| `scene/dataset_readers.py` | Lazy-import pytorch3d (avoid module-level ImportError) |
| `scene/gaussian_model.py` | `torch.cuda.empty_cache()` before optimizer re-assignment (3 sites) + `grad is None` guard in densification |
| `utils/loss_utils.py` | Empty-tensor guard in `l1_loss`; add `ssi_loss` and `gradient_loss` (missing in upstream) |

## Patch 1 — Training improvements (pre-requisite)

**File:** `docs/longsplat/patches/longsplat_training_improvements.patch`

Additional stability and quality improvements across 4 files:

| File | Changes |
|---|---|
| `arguments/__init__.py` | `ssi_loss_weight=0.05`, `grad_loss_weight=0.02` in `OptimizationParams`; fix `init_iteraion` → `init_iteration` typo |
| `scene/__init__.py` | Whitespace cleanup |
| `train.py` | Wire `ssi_loss`/`gradient_loss` into training loop (4 sites); guard `loss.backward()` against NaN/Inf (3 sites); `torch.cuda.empty_cache()` after incremental registration; `max_pnp_retries` 10→15; `cv2.solvePnPRansac` iterations 200→300, reprojError 5→8; occlusion mask empty-guard in pose estimation; whitespace |
| `utils/graphics_utils.py` | `compute_texture_mask` function |

## Patch 2 — VDA depth injection

**File:** `docs/longsplat/patches/vda_depth_injection.patch`

VDA-specific changes across 4 files:

### `arguments/__init__.py` — tuning knobs

| Parameter | Default | Purpose |
|---|---|---|
| `depth_source` | `"mast3r"` | Select depth source (`mast3r` = original, any other = VDA) |
| `vda_corr_threshold` | `0.3` | Minimum Pearson \|ρ\| for alignment acceptance |
| `vda_outlier_sigma` | `2.5` | Residual threshold (in σ) for re-fit rejection |

### `utils/graphics_utils.py` — `align_vda_depth()`

Aligns VDA relative depth (disparity-like, higher=near) to scene z-depth (higher=far). Fits `a·v + b ≈ 1/D_ref` in inverse-depth space via least squares with one round of outlier rejection. Safety valves: slope check (a ≤ 0 → reject), correlation check (ρ < corr_threshold → reject). Returns `None` on failure; caller responsible for fallback.

### `scene/__init__.py` — scene init injection point

After MASt3R depth initialization, iterates over train cameras and loads materialized VDA `.npy` files from `<source_path>/depths/<image_stem>_depth.npy`. Calls `align_vda_depth()` with per-frame `corr_threshold`/`outlier_sigma` from `args`. Emits `VDA_USAGE` marker for every frame (aligned / rejected / missing).

### `train.py` — incremental injection point

After `compute_scale` in incremental registration, loads VDA `.npy` and attempts alignment with the scale-corrected MASt3R depth as reference. Emits `VDA_USAGE` marker for every frame.

## Patch 3 — finite conversion and structured telemetry

**File:** `docs/longsplat/patches/longsplat_conversion_telemetry.patch`

This ordered patch must be applied after `vda_depth_injection.patch`. It:

- clamps squared nearest-neighbour distances to `[1e-7, voxel_size]` before
  `log(sqrt(distance))`, preventing duplicate points from producing
  `scale_0..2 = -inf`;
- aborts conversion if any initialized scale is still non-finite;
- emits per-attempt PnP match/inlier/reprojection metrics;
- returns VDA fit statistics through `align_vda_depth_with_stats()` while
  keeping `align_vda_depth()` backward-compatible;
- emits conversion, pose, and VDA diagnostics as strict JSON stdout markers.

## Patch 4 — MASt3R low-host-memory checkpoint loading

**File:** `docs/longsplat/patches/mast3r_low_memory_load.patch`

This patch targets the nested repository at
`third_party/LongSplat/submodules/mast3r`. It memory-maps the 2.9 GB local
checkpoint and releases the checkpoint object before moving the initialized
network to CUDA. This reduces the transient Windows host-memory peak without
changing model weights, precision, inference, or LongSplat optimization.

## Telemetry markers

Structured marker lines use one JSON object after the prefix:

```text
POSE_TELEMETRY {"frame":"...","success":true,"match_count":100,"inlier_count":80,"inlier_ratio":0.8,"reprojection_rmse_px":1.25,...}
VDA_TELEMETRY {"frame":"...","result":"aligned","correlation":0.91,"inlier_ratio":0.88,"slope":0.5,"offset":0.1,...}
CONVERSION_TELEMETRY {"point_count":40000,"nonpositive_distance_count":1200,"finite_scale_count":40000,...}
```

The orchestrator records these under `telemetry.pose`, `telemetry.vda`, and
`telemetry.conversion` in `reconstruction_run.json`. Malformed JSON markers
are ignored, and numeric aggregates exclude booleans, strings, NaN, and
infinity.

### Backward-compatible VDA_USAGE markers

Every VDA depth injection emits a stdout marker:

```
VDA_USAGE result=aligned stage=scene_init frame=<image_name>
VDA_USAGE result=rejected stage=scene_init frame=<image_name>
VDA_USAGE result=missing stage=scene_init frame=<image_name>
VDA_USAGE result=aligned stage=incremental frame=<image_name>
VDA_USAGE result=rejected stage=incremental frame=<image_name>
VDA_USAGE result=missing stage=incremental frame=<image_name>
```

The orchestrator parses these markers from training stdout and records counts (`aligned`, `missing`, `rejected`). **If `aligned == 0` the pipeline fails before conversion** — there is no silent fallback.

## Consumption contract

Materialized `.npy` files must contain **raw VDA relative disparity** (higher=near), not pre-converted z-depth. Conversion and alignment happen at runtime where scene-scale reference depth is available.

Depth files are read from: `<source_path>/depths/<image_stem>_depth.npy`

## Verification

```powershell
git -C third_party/LongSplat rev-parse HEAD
git -C third_party/LongSplat diff --check
git -C third_party/LongSplat apply --check --reverse ../../docs/longsplat/patches/longsplat_conversion_telemetry.patch
git -C third_party/LongSplat/submodules/mast3r apply --check --reverse ../../../../docs/longsplat/patches/mast3r_low_memory_load.patch
git -C third_party/LongSplat ls-files --others --exclude-standard -- "*.py" "*.cu" "*.cpp"
```

All should return 0; the last command should produce no output. Earlier
patches overlap files changed by the final patch, so verify their parity by
forward-applying all four LongSplat patches and the independent nested MASt3R
patch to their locked bases as documented in `docs/longsplat/REPRODUCTION_RUNBOOK.md`.
