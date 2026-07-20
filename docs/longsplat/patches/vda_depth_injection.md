# LongSplat VDA depth injection patches

> Branch: `research/depth-prior`
> LongSplat base: NVlabs/LongSplat pinned commit `1975077`
> Date: 2026-07-20
> Status: **applied locally, not upstreamed**

## Applying

```bash
cd third_party/LongSplat
git apply ../../docs/longsplat/patches/longsplat_stability_fixes.patch
git apply ../../docs/longsplat/patches/vda_depth_injection.patch
```

## Patch 0 — Stability fixes (pre-requisite)

**File:** `docs/longsplat/patches/longsplat_stability_fixes.patch`

Prior baseline-training fixes applied before VDA injection:

| File | Fix |
|---|---|
| `scene/dataset_readers.py` | Lazy-import pytorch3d (avoid module-level ImportError) |
| `scene/gaussian_model.py` | `torch.cuda.empty_cache()` before optimizer re-assignment (3 sites) + `grad is None` guard in densification |
| `utils/loss_utils.py` | Empty-tensor guard in `l1_loss`; add `ssi_loss` and `gradient_loss` (missing in upstream) |

## Patch 1 — Fix `depth_source` parameter wiring

| Field | Value |
|---|---|
| File | `third_party/LongSplat/arguments/__init__.py` |
| Type | Bug fix — dead code resurrection |
| Lines | Move `self.depth_source = "mast3r"` from `OptimizationParams` (was line 147) to `ModelParams` (inserted after `self.init_frame_num = 3`) |

**Why:** `Scene.__init__` receives `ModelParams`, not `OptimizationParams`, so the external depth hook at `scene/__init__.py:154` never saw `depth_source`. This patch makes the hook reachable.

## Patch 2 — `align_vda_depth()` alignment utility

| Field | Value |
|---|---|
| File | `third_party/LongSplat/utils/graphics_utils.py` |
| Type | New function (~55 lines after `compute_scale`) |
| Signature | `align_vda_depth(v, D_ref, corr_threshold=0.3, outlier_sigma=2.5)` |

**Why:** VDA relative depth is disparity-like (higher=near); LongSplat's `depth_map` is z-depth (higher=far). Direct substitution produces reverse gradients. This function fits `a·v + b ≈ 1/D_ref` in inverse-depth space with one round of outlier rejection, then returns `1 / clamp(a·v + b, eps)`. A Pearson correlation safety valve returns `None` if the fit is weak.

## Patch 3 — VDA loading in incremental registration

| Field | Value |
|---|---|
| File | `third_party/LongSplat/train.py` |
| Type | Feature — depth source injection point |
| Lines | After `compute_scale` (was line 276), ~16 lines inserted |

**Why:** Even with Patch 1, `train.py:227` overwrites `depth_map` with MASt3R output during incremental registration. This patch loads the materialized VDA `.npy`, interpolates to training resolution, calls `align_vda_depth()` with the MASt3R-aligned depth as reference, and replaces `depth_map` if alignment succeeds. Falls back silently to MASt3R on missing files or failed alignment.

## Patch 4 — Scene init hook direction fix

| Field | Value |
|---|---|
| File | `third_party/LongSplat/scene/__init__.py` |
| Type | Bug fix — incorrect depth domain |
| Lines | Rewrite of existing hook at lines 153-165 |

**Why:** The existing hook loaded raw VDA disparity and directly assigned it to `cam.depth_map` (which expects z-depth). Fixed to call `align_vda_depth(ext_depth, cam.depth_map)` using MASt3R's `depth_maps[i]` as reference, replacing only when alignment succeeds.

## Consumption contract

These patches read VDA depth from `<source_path>/depths/<image_stem>_depth.npy`.

The materialized `.npy` files must contain **raw VDA relative disparity** (higher=near), not pre-converted z-depth. The conversion and alignment happen at runtime inside the training loop where scene-scale reference depth is available.
