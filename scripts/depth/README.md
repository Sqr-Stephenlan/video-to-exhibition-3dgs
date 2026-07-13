# Depth prior module (`research/depth-prior`)

Thin orchestration over **Video Depth Anything** for weak-texture exhibition depth priors.

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
.\.venv\Scripts\python.exe scripts\depth\run_depth_prior.py run
.\.venv\Scripts\python.exe scripts\depth\run_depth_prior.py run --dry-run
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
6. When sample frames arrive, write `data/manifests/frames_manifest.json` and run `run`

## GPU note

`doctor` currently passes with CPU `torch`. For real inference, install a CUDA build of PyTorch that matches the local NVIDIA driver, still targeting `torch==2.1.1` / `torchvision==0.16.1` when possible to stay close to VDA's pin. Re-run `doctor` and confirm `cuda_available: True` before long runs.

## Outputs

- `data/depth/<frame_id>.npz`
- `data/manifests/depth_manifest.json`
- `outputs/reconstructions/depth_prior/run_record.json`
