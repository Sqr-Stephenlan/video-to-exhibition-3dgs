# LongSplat Reproduction Runbook

> Branch: `research/longsplat-route`
> Status: research prototype, manually validated

## Prerequisites

- Python 3.10+ in project `.venv` with `plyfile`, `numpy`
- LongSplat backend at `third_party/LongSplat` at locked commit `19750775a9d19f30aa05a8333c4c6c231b2d5f4a`
- Patches applied (see [vda_depth_injection.md](patches/vda_depth_injection.md)):
  ```bash
  cd third_party/LongSplat
  git apply ../../docs/longsplat/patches/longsplat_stability_fixes.patch
  git apply ../../docs/longsplat/patches/longsplat_training_improvements.patch
  git apply ../../docs/longsplat/patches/vda_depth_injection.patch
  git apply ../../docs/longsplat/patches/longsplat_conversion_telemetry.patch
  git -C submodules/mast3r apply ../../../../docs/longsplat/patches/mast3r_low_memory_load.patch
  ```
- Input frames manifest from preprocess pipeline
- CUDA-capable GPU with compiled LongSplat submodules
- **Separate LongSplat CUDA Python environment** (not the project `.venv`)
  ```bash
  # Verify backend Python has CUDA:
  <LongSplat-CUDA-Python> -c "import torch; print(torch.cuda.is_available())"
  # Must print True
  ```

## Verify patch parity

### Working tree reverse-check (当前 dirty backend)

The final finite/telemetry patch can reverse-check directly on the final
working tree. Earlier overlapping patches are verified by clean-base replay:

```powershell
git -C third_party/LongSplat rev-parse HEAD
git -C third_party/LongSplat diff --check
git -C third_party/LongSplat apply --check --reverse ../../docs/longsplat/patches/longsplat_conversion_telemetry.patch
git -C third_party/LongSplat/submodules/mast3r apply --check --reverse ../../../../docs/longsplat/patches/mast3r_low_memory_load.patch
git -C third_party/LongSplat ls-files --others --exclude-standard -- "*.py" "*.cu" "*.cpp"
```

All should exit 0; last command should produce no output.

### Clean-base forward-apply (完整重建)

To verify all four LongSplat patches plus the nested MASt3R patch can rebuild
from locked bases, use a temporary clean checkout:

```powershell
$base = $env:TEMP + "\longsplat-clean-verify"
git clone https://github.com/NVlabs/LongSplat $base
git -C $base checkout 19750775a9d19f30aa05a8333c4c6c231b2d5f4a
git -C $base apply (Resolve-Path docs/longsplat/patches/longsplat_stability_fixes.patch)
git -C $base apply (Resolve-Path docs/longsplat/patches/longsplat_training_improvements.patch)
git -C $base apply (Resolve-Path docs/longsplat/patches/vda_depth_injection.patch)
git -C $base apply (Resolve-Path docs/longsplat/patches/longsplat_conversion_telemetry.patch)
git -C $base submodule update --init submodules/mast3r
git -C "$base/submodules/mast3r" apply (Resolve-Path docs/longsplat/patches/mast3r_low_memory_load.patch)
git -C $base diff --check
git -C "$base/submodules/mast3r" diff --check
git -C $base ls-files --others --exclude-standard -- "*.py" "*.cu" "*.cpp"
```

All should exit 0.

Then verify the reconstructed files match the current backend exactly
(only the 7 files touched by the four LongSplat patches):

```powershell
$files = @(
  "arguments/__init__.py",
  "scene/__init__.py",
  "scene/dataset_readers.py",
  "scene/gaussian_model.py",
  "train.py",
  "utils/graphics_utils.py",
  "utils/loss_utils.py"
)

foreach ($file in $files) {
  git diff --no-index --ignore-cr-at-eol --exit-code -- "third_party/LongSplat/$file" "$base/$file"
  if ($LASTEXITCODE -ne 0) {
    throw "Patch reconstruction differs: $file"
  }
}
```

All 7 files must be identical. The four-patch order must remain
*stability → training-improvements → vda-depth-injection →
finite-conversion-telemetry*. Only
report patch parity as PASS when every file compares equal.

## Verify CLI

```bash
./dev.sh python -m scripts.longsplat.run_experiment --help
```

Must exit 0 and print usage. Then:

```bash
./dev.sh python -m scripts.longsplat.run_experiment --definitely-invalid
```

Must exit 2 and print argparse error.

## Run tests

```bash
./dev.sh pytest tests/integration/longsplat -q
```

Expect >= 115 passed.

## Baseline smoke (4 frames, ~2 min)

Uses MASt3R depth only (no external depth). Config: `configs/longsplat/smoke.json`.

```bash
./dev.sh python -m scripts.longsplat.run_experiment \
  --manifest data/manifests/<video>/preprocess_manifest.json \
  --segment-id <segment_id> \
  --config configs/longsplat/smoke.json \
  --repo-root third_party/LongSplat \
  --output-dir outputs/<video> \
  --project-root . \
  --backend-python <LongSplat-CUDA-Python>
```

## VDA smoke (4 frames, ~3 min)

Uses Video-Depth-Anything depth priors. Config: `configs/longsplat/smoke_vda.json`.
Requires depth manifest from `depth-prior` pipeline.

```bash
./dev.sh python -m scripts.longsplat.run_experiment \
  --manifest data/manifests/<video>/preprocess_manifest.json \
  --segment-id <segment_id> \
  --config configs/longsplat/smoke_vda.json \
  --repo-root third_party/LongSplat \
  --output-dir outputs/<video> \
  --project-root . \
  --depth-manifest data/manifests/<video>/depth_manifest.json \
  --backend-python <LongSplat-CUDA-Python>
```

Success criteria:
- `aligned > 0` in run record depth.usage
- `telemetry.pose` contains every incremental PnP attempt
- `telemetry.vda` contains fit quality and rejection reasons
- `telemetry.conversion.all_scales_finite == true`
- training exit code 0
- conversion exit code 0
- PLY passes vertex/attribute/finite validation
- run status: `complete`

New runs validate the raw converted PLY without automatic cleanup. Any
NaN/Inf value fails the run and leaves the original PLY intact for diagnosis.
`clean_converted_ply()` remains available only as an explicit recovery tool
for historical exports.

## Full training (197 frames, ~3.5 hours)

Replace `smoke.json` / `smoke_vda.json` with `full.json` / `full_vda.json`.
Adjust `iterations` in config as needed (default 30000).

## Isolated checkpoint reconversion

Re-convert an existing LongSplat checkpoint without touching the original
model directory.  Useful for diagnosing conversion-only issues.

```powershell
# Dry-run first
./dev.sh python -m scripts.longsplat.reconvert_existing `
  --source-model outputs/wall_test/<existing_run> `
  --destination-model outputs/wall_test/<new_run> `
  --source-path data/frames/wall_test `
  --repo-root third_party/LongSplat `
  --backend-python D:/video-to-exhibition-3dgs/venv/Scripts/python.exe `
  --backend-mode research_local `
  --checkpoint-iteration 50000 `
  --conversion-iterations 100 `
  --prune-ratio 0.6 `
  --output-record outputs/wall_test/<new_run>/reconversion_record.json `
  --dry-run

# Real execution (omit --dry-run)
```

The snapshot copies only cfg_args, camera JSON files, and the specified
checkpoint iteration (PLY + 3 MLP files).  ``cameras_all_train.json`` is
duplicated as ``cameras_all.json`` for the Eval loader.  The original model
is never modified (SHA verified before and after).

## Output structure

```
outputs/<video>/<run_id>/
├── input/                        # Immutable frame snapshots
│   ├── depths/                   # Materialised VDA .npy files (VDA mode only)
│   └── frame_mapping.json
├── longsplat_model/              # LongSplat training output
│   └── converted_3dgs/
│       └── point_cloud.ply       # Final 3DGS point cloud
├── logs/
│   ├── train_stdout.log
│   └── train_stderr.log
└── reconstruction_run.json       # Full provenance record
```
