# Video Depth Anything backend pin

| Field | Value |
|---|---|
| Repository | https://github.com/DepthAnything/Video-Depth-Anything |
| Acquisition | local clone under `third_party/Video-Depth-Anything` (not tracked by this repo) |
| Pinned commit | `4f5ae23172ba60fd7bc11ef671cca678842c7072` (`4f5ae23`) |
| Default model | Relative depth, encoder `vitb` (Video-Depth-Anything-Base) |
| Default checkpoint | `checkpoints/video_depth_anything_vitb.pth` |
| Checkpoint SHA-256 | `775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b` |
| Local patches | none |
| License note | Base/Large weights use CC-BY-NC-4.0 |
| Invoked by | `scripts/depth/run_depth_prior.py` → backend `run.py` |

Default config fingerprint:

| File | SHA-256 |
|---|---|
| `configs/depth/default_vitb.yaml` | `254be26a1330708b27713690c5c670df11d4648b84b1b3714f7d0ba1553deb52` |

## Clone (do not commit the clone)

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything.git third_party/Video-Depth-Anything
git -C third_party/Video-Depth-Anything checkout 4f5ae23172ba60fd7bc11ef671cca678842c7072
```

## Weights (Base / relative only)

```bash
mkdir -p third_party/Video-Depth-Anything/checkpoints
# Hugging Face:
# https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth
```

## Local environment notes (Windows)

- Project venv: `.venv\Scripts\python.exe`
- Validated local stack for doctor: `torch 2.1.1+cu118`, CUDA available on RTX 3060 Laptop
- `ffmpeg` must be on `PATH` before assembling frames into a temp video
- Prefer PowerShell commands on native Windows; do not assume `dev.ps1` exists

Do not use floating `main` / `latest` as the sole version identifier in run records.
