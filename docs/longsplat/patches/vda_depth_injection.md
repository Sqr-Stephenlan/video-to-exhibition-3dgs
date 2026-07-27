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
git apply ../../docs/longsplat/patches/longsplat_pose_quality_gates.patch
git apply ../../docs/longsplat/patches/longsplat_depth_quality.patch
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

## Patch 4 — Pose quality gates with reference lookback and SO(3)

**File:** `docs/longsplat/patches/longsplat_pose_quality_gates.patch`

This ordered patch must be applied after `longsplat_conversion_telemetry.patch`. It:

- Adds 14 gate parameters to `ModelParams`:
  `min_match_count`, `min_inlier_count`, `min_inlier_ratio`,
  `max_reprojection_rmse_px`, `min_grid_coverage`, `min_positive_depth_ratio`,
  `max_rotation_step_deg`, `max_translation_step_ratio`, `reference_lookback`,
  `min_correlation`, `min_vda_inlier_ratio`, `max_normalized_rmse`,
  `min_aligned_fraction`
- Adds `project_to_so3()`, `keypoint_grid_coverage()`, and `rotation_step_deg()`
  helpers to `utils/pose_utils.py`; `project_to_so3()` projects a 3x3 matrix
  to the closest SO(3) element via SVD with determinant correction
- Hardens `Camera.update_RT()` to validate shape `(3,3)`/`(3,)`, finiteness,
  orthogonality error <= 1e-3, and determinant in SO(3)
- Calls `project_to_so3()` at three sites: MASt3R global align init cameras,
  PnP/least-squares candidates before `update_RT`, and SE(3) delta synthesis
- Replaces the old `end_view_id -= 1` PnP retry loop with a quality-gated
  reference lookback: tries up to `reference_lookback` most recent accepted
  cameras, evaluates each candidate against all hard acceptance criteria,
  emits `POSE_TELEMETRY` with `allow_nan=False` for every attempt, and raises
  `RuntimeError("POSE_GATE_REJECTED")` if all references fail
- Emits `POSE_TELEMETRY` for global align init cameras (stage=scene_init,
  accepted=true) with `allow_nan=False`

Hard acceptance criteria for incremental registration:
- OpenCV `solvePnPRansac` success
- match_count >= 128
- inlier_count >= 64
- inlier_ratio >= 0.80
- refined reprojection RMSE <= 2.0 px
- 4x4 grid coverage >= 6/16 cells
- positive depth ratio >= 0.95 in PnP native camera coordinates
- rotation step relative to last accepted camera <= 25 deg
- translation step <= 6x the median of up to 8 recent steps (after >= 3 steps)

## Patch 5 — MASt3R low-host-memory checkpoint loading

**File:** `docs/longsplat/patches/mast3r_low_memory_load.patch`

This patch targets the nested repository at
`third_party/LongSplat/submodules/mast3r`. It memory-maps the 2.9 GB local
checkpoint and releases the checkpoint object before moving the initialized
network to CUDA. This reduces the transient Windows host-memory peak without
changing model weights, precision, inference, or LongSplat optimization.

## Patch 6 — VDA depth quality gate

**File:** `docs/longsplat/patches/longsplat_depth_quality.patch`

This ordered patch must be applied after `longsplat_pose_quality_gates.patch`. It adds quality gates to VDA depth alignment, rejecting weak fits that previously passed silently. Changes across 3 files:

### `utils/graphics_utils.py` — quality gates in `align_vda_depth_with_stats()`

Adds two new parameters and rejection criteria:

| Parameter | Default | Purpose |
|---|---|---|
| `min_inlier_ratio` | `0.0` | Minimum fraction of inliers after outlier rejection |
| `max_normalized_rmse` | `None` | Maximum normalized RMSE `residual_rmse / median(|1/D_ref|)` |

Rejection reasons emitted via `stats["reason"]`:
- `low_inlier_ratio` — inlier fraction below `min_inlier_ratio`
- `high_normalized_rmse` — normalized RMSE above `max_normalized_rmse`

Also adds `normalized_rmse` field to stats dictionary for telemetry.

### `scene/__init__.py` — scene init injection updated

Reads quality thresholds from args and passes them to `align_vda_depth_with_stats()`:

```python
min_correlation = getattr(args, 'min_correlation', 0.90)
min_vda_inlier_ratio = getattr(args, 'min_vda_inlier_ratio', 0.95)
max_normalized_rmse_vda = getattr(args, 'max_normalized_rmse', 0.10)
```

Old `corr_threshold` usage replaced with `min_correlation` (0.90 instead of 0.3).

### `train.py` — incremental injection updated

Same threshold reads and parameter passing as `scene/__init__.py`.

### Post-training gate (Python side)

The orchestrator (`scripts/longsplat/orchestrator.py`) enforces after training but before conversion:
- `record_count == train_camera_count` (every train frame got a VDA_TELEMETRY record)
- `missing == 0` (no missing depth files)
- `materialized_count == train + test` (all frames materialized)
- `aligned / train_camera_count >= 0.98` (at least 98% aligned)

Failure emits `vda_quality_gate` stage with detailed rejection reasons and returns exit code 1.

## Telemetry markers

Structured marker lines use one JSON object after the prefix:

```text
POSE_TELEMETRY {"stage":"scene_init","frame":"...","accepted":true,"method":"mast3r_global_align"}
POSE_TELEMETRY {"stage":"incremental","frame":"...","reference":"...","method":"pnp_ransac_refined","accepted":true,"match_count":412,"inlier_count":190,...}
POSE_TELEMETRY {"stage":"incremental","frame":"...","reference":"...","method":"pnp_ransac_refined","accepted":false,"rejection_reasons":["inlier_ratio","grid_coverage"],...}
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
git -C third_party/LongSplat apply --check --reverse ../../docs/longsplat/patches/longsplat_depth_quality.patch
git -C third_party/LongSplat apply --check --reverse ../../docs/longsplat/patches/longsplat_pose_quality_gates.patch
git -C third_party/LongSplat apply --check --reverse ../../docs/longsplat/patches/longsplat_conversion_telemetry.patch
git -C third_party/LongSplat/submodules/mast3r apply --check --reverse ../../../../docs/longsplat/patches/mast3r_low_memory_load.patch
git -C third_party/LongSplat ls-files --others --exclude-standard -- "*.py" "*.cu" "*.cpp"
```

All should return 0; the last command should produce no output. Earlier
patches overlap files changed by the final patch, so verify their parity by
forward-applying all five LongSplat patches and the independent nested MASt3R
patch to their locked bases as documented in `docs/longsplat/REPRODUCTION_RUNBOOK.md`.
