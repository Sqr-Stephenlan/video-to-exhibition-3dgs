# Depth prior module (`research/depth-prior`)

Thin orchestration over **Video Depth Anything** for weak-texture exhibition depth priors.

Full prerequisites, input/output schemas, and deferred fields: [`docs/depth_prior_io.md`](../../docs/depth_prior_io.md).

## Defaults for this branch

| Item | Value |
|---|---|
| Model | Relative **Base** (`vitb`) |
| Config | `configs/depth/default_vitb.yaml` |
| Backend pin | `configs/depth/backend_pin.md` |
| Sample frames | Not required for `doctor`; required for `run` |

Depth is a soft prior only. Do not treat outputs as metric ground truth.

## Commands (Windows PowerShell)

Project `.venv` uses `Scripts\python.exe` on Windows (`dev.sh` expects Unix paths).

```powershell
# Environment checks (no frames needed)
.\.venv\Scripts\python.exe scripts\depth\run_depth_prior.py doctor

# After frames_manifest + frames exist
.\.venv\Scripts\python.exe scripts\depth\run_depth_prior.py run --video-id <video_id> --run-id <run_id>
.\.venv\Scripts\python.exe scripts\depth\run_depth_prior.py run --video-id <video_id> --run-id <run_id> --dry-run
```

Git Bash / WSL (when `.venv/bin/python` exists):

```bash
./dev.sh python scripts/depth/run_depth_prior.py doctor
```

## Setup checklist

1. Create `.venv` and install `configs/depth/requirements.txt`
2. Clone and pin Video Depth Anything (see `configs/depth/backend_pin.md`)
3. Install backend `requirements.txt` into the same venv
4. Download `video_depth_anything_vitb.pth` into `third_party/Video-Depth-Anything/checkpoints/`
5. Run `doctor` until status is ok
6. Adapt preprocess output (or write `data/manifests/<video_id>/frames_manifest.json`) and run with `--video-id`

## Outputs

- `data/depth/<video_id>/<run_id>/<frame_id>.npz` — one file per selected frame; each stores a single `depth` array (`float32`, finite)
- `data/manifests/<video_id>/<run_id>/depth_manifest.json` — includes per-frame `sha256`, `producer_gates` (file integrity vs VDA quality), `source_video_id`, `source_frames_manifest`, and `frame_depth_mapping: strict_positional`
- `outputs/reconstructions/depth_prior/<video_id>/<run_id>/run_record.json` — repository-relative paths, backend commit, I/O, and a sanitized command using `<project>` / `<temp>` placeholders

Bounded / low-VRAM chunking (overlap count, not start index):

```powershell
.\.venv\Scripts\python.exe scripts\depth\split_selected_manifest.py ... --chunk a=0:49 --chunk b=39:91 ...
.\.venv\Scripts\python.exe scripts\depth\assemble_depth_manifests.py ... --drop-b-prefix 10 ...
```

See `docs/depth_prior_io.md` for details. File integrity pass ≠ VDA geometric quality pass.

VDA `run.py` emits a single `*_depths.npz` with key `depths` shaped `(N,H,W)` per
invocation. The orchestrator may run **one invocation per `segment_id`**
(`runtime.infer_per_segment`), concatenates depth arrays in selected-frame order,
validates total `N`, and splits into per-frame NPZ files. Temp videos use fixed
`target_fps`; pinned VDA does not consume container PTS, so gap-aware ffmpeg timing
is **not** treated as temporal reset.

If `backend.checkpoint` is set, it must be project-relative **and**
`backend.allow_custom_checkpoint: true`. The orchestrator stages that file to the
checkpoint filename hardcoded by pinned VDA before inference, restores the original
target afterward, and records both paths without machine-specific absolute paths.

Temp-video assembly passes `-frames:v N` so the ffmpeg concat demuxer’s trailing duplicate file entry does not produce an extra frame (which would fail the strict N-depth check).

`doctor` / `run` also apply the documented VDA matplotlib 3.9+ colormap local patch when needed (see `configs/depth/backend_pin.md`).

## Bridge from preprocess

```powershell
.\.venv\Scripts\python.exe scripts\depth\adapt_preprocess_manifest.py `
  data\manifests\<video_id>\frames_manifest.json `
  --run-id baseline

.\.venv\Scripts\python.exe scripts\depth\run_depth_prior.py `
  --config configs/depth/smoke_joint.yaml run --video-id <video_id> --run-id baseline
```

See `docs/depth_prior_io.md` for the producer contract. LongSplat mapping /
materialize / `depth_source=vda` / fail-closed consumption belong on
`research/longsplat-route` (PR #3), not this module.

## Current deferred items

The following config fields are intentionally documented but not implemented in this PR:

- `runtime.device`
- `runtime.skip_existing`
- `io.mask_dir` / `confidence_path`

They remain deferred until sample frames and the next execution path changes are ready. Do not assume they currently affect runtime behavior.
