# LongSplat module

Thin runner for pose-free 3DGS reconstruction of long videos via
[NVlabs/LongSplat](https://github.com/NVlabs/LongSplat).

## Current production closure

The current distribution pins the nested fork to
`c6496dc43c4ced6d072e3896fd1b172f6658b259` (tree
`d240c4113a95f632b58b56f3d197160e4dab24cc`). External fixed-pose with depth
disabled resolves `image_residency=auto` to `cpu-stream-v1`; optional MASt3R,
VDA, depth, and pose-search routes retain their historical policy unless an
explicit supported strategy is selected. Runtime evidence is versioned as
`image-residency-telemetry-v1` and is advisory except for identity, transfer,
device, and residency contract violations.

The exact nested replay delta is
[`0004-longsplat-cpu-stream-image-residency.patch`](patches/0004-longsplat-cpu-stream-image-residency.patch).

For the supported product entry point, start with
[ONE_CLICK_RECONSTRUCTION.md](ONE_CLICK_RECONSTRUCTION.md):

```bash
./video-to-3dgs "/path/to/video.mp4"
```

## Architecture

```
frame manifest ──► validate_input ──► prepare_input ──► runner ──► convert ──► standard 3DGS PLY
                       │                    │               │            │
                       ▼                    ▼               ▼            ▼
                   input contract       frame copies    run_record    converted PLY
                                                     (immutable,     validation
                                                      provenance)
```

Each run is scoped to an immutable run directory. The run record tracks every
input, configuration, and output artifact with SHA-256 hashes.

## Quick start

### 1. Prepare a frame manifest

The module consumes frame manifests produced by the preprocessing pipeline
(`feature/preprocess-video` branch). A minimal manifest:

```json
{
  "schema_version": 1,
  "segment_id": "demo_segment",
  "base": "/data/frames",
  "frames": [
    {
      "frame_id": 0,
      "path": "frame_000000.jpg",
      "width": 1920,
      "height": 1080,
      "sha256": "abcdef0123456789..."
    }
  ]
}
```

### 2. Validate and prepare input

```bash
./dev.sh python -c "
from scripts.longsplat.validate_input import validate_manifest
from scripts.longsplat.prepare_input import prepare_input

manifest = validate_manifest('manifest.json')
prepare_input(manifest, 'runs/run_001')
"
```

### 3. Run the full pipeline (recommended)

The single-entry orchestrator handles validation, preparation, training,
conversion, PLY validation, and run recording:

```bash
./dev.sh python -c "
from scripts.longsplat.orchestrator import run_pipeline
from scripts.longsplat.runner import load_config

config = load_config('configs/longsplat/smoke.json')
exit_code = run_pipeline(
    manifest_path='path/to/preprocess_manifest.json',
    segment_id='your_segment_id',
    config=config,
    repo_root='/path/to/LongSplat',
    output_dir='runs',
)
raise SystemExit(exit_code)
"
```

Or run training and conversion separately (advanced):

```bash
./dev.sh python -c "
from scripts.longsplat.runner import (
    LongSplatConfig, load_config, run_training, run_conversion
)

config = load_config('configs/longsplat/smoke.json')
config.source_path = 'runs/run_001/input'
config.model_path = 'runs/run_001/longsplat_model'

run_training('/path/to/LongSplat', config)
run_conversion('/path/to/LongSplat', config)
"
```

### 4. Validate the output

```bash
./dev.sh python -c "
from scripts.longsplat.convert import validate_converted_ply

meta = validate_converted_ply('runs/run_001/longsplat_model/converted_3dgs/point_cloud.ply')
print(f'{meta[\"vertex_count\"]} gaussians, SHA-256: {meta[\"sha256\"][:12]}...')
"
```

## Configuration

Drop a JSON file into `configs/longsplat/` matching the `LongSplatConfig` schema:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `source_path` | str | — | Path to input images |
| `model_path` | str | — | Path for model output |
| `images` | str | `images` | Subdirectory name for images |
| `resolution` | int | `-1` | Resolution override (-1 = native) |
| `sh_degree` | int | `3` | Spherical harmonics degree |
| `iterations` | int | `30000` | Training iterations |
| `seed` | int | `0` | Random seed (locked backend uses 0) |
| `mode` | str | `custom` | Dataset mode (must be `custom` for video input) |
| `extra_train_args` | dict | `{}` | Passthrough args to `train.py` |
| `image_residency` | str | `None` | Optional `auto`, `cpu-stream-v1`, or `gpu-all-v0` policy |
| `convert_iteration` | int | `30000` | Iteration to convert at |
| `convert_prune_ratio` | float | `0.6` | Prune ratio for conversion |

## Environment

Copy `configs/longsplat/environment.example.json` and set:

- `longsplat_repo_path` — path to the LongSplat repo at the locked commit
- `python_exe` — Python interpreter for the backend (must have LongSplat deps)

## Run record

Every run writes `reconstruction_run.json` to its run directory. It follows a
state machine:

```
planned ──► running ──► complete
                │
                └──────► failed
```

The record includes:
- Unique run ID (UUID v4)
- Full configuration snapshot
- Input manifest SHA-256
- Backend repo identity (URL + commit)
- Per-stage exit codes, errors, and durations
- Output artifact paths with SHA-256 hashes

## Locked backend

The runner is pinned to LongSplat commit
`c6496dc43c4ced6d072e3896fd1b172f6658b259` from
`https://github.com/Sqr-Stephenlan/LongSplat`.
`_check_repo()` verifies the commit and all required submodules before execution.

## Testing

```bash
# CPU-only integration tests (no GPU required)
./dev.sh pytest tests/integration/longsplat/ -v

# GPU end-to-end tests (manual, requires CUDA)
./dev.sh pytest tests/gpu/longsplat/ -v
```

## Output contract

The conversion step produces a standard 3DGS PLY at
`<model_path>/converted_3dgs/point_cloud.ply` with these vertex attributes:

- `x`, `y`, `z` — position
- `f_dc_0`, `f_dc_1`, `f_dc_2` — DC spherical harmonics
- `opacity` — opacity
- `scale_0`, `scale_1`, `scale_2` — covariance scale
- `rot_0`, `rot_1`, `rot_2`, `rot_3` — covariance rotation (quaternion)

Additional SH degrees (`f_rest_*`) are allowed but not required.
