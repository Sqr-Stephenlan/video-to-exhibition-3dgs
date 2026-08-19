# Video Depth Anything backend pin

| Field | Value |
|---|---|
| Repository | https://github.com/DepthAnything/Video-Depth-Anything |
| Acquisition | local clone under `third_party/Video-Depth-Anything` (not tracked by this repo) |
| Pinned commit | `4f5ae23172ba60fd7bc11ef671cca678842c7072` (`4f5ae23`) |
| Default model | Relative depth, encoder `vitb` (Video-Depth-Anything-Base) |
| Default checkpoint | `checkpoints/video_depth_anything_vitb.pth` |
| Checkpoint SHA-256 | `775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b` |
| Local patches | **yes** — `utils/dc_utils.py` matplotlib 3.9+ colormap compat (auto-applied by orchestrator; see below) |
| License note | Base/Large weights use CC-BY-NC-4.0 |
| Invoked by | `scripts/depth/run_depth_prior.py` → backend `run.py` |

Default config fingerprint (canonical Git blob / LF bytes; use `git show HEAD:configs/depth/default_vitb.yaml | sha256sum`):

| File | SHA-256 |
|---|---|
| `configs/depth/default_vitb.yaml` | `288cd01a97bc4b13fdcf686bc846fc4a57dd817eeca4b3be38867b1b545165df` |

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

## Optional Large weights (12GB quality config)

`configs/depth/default_vitl.yaml` uses relative **Large** (`vitl`). Place:

`third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitl.pth`

```text
https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth
```

Doctor currently pins SHA-256 only for the default Base relative checkpoint. Large is existence-checked by filename; keep `backend.checkpoint` empty so VDA `run.py` loads the expected `vitl` name. Keep `depth_type: relative` (do not switch this quality line to metric).

`backend.checkpoint` may point to another project-relative weight file **only when**
`backend.allow_custom_checkpoint: true` is set (explicit reviewer opt-in). Before
inference, the orchestrator temporarily stages it to the filename that pinned VDA
`run.py` hardcodes under `third_party/Video-Depth-Anything/checkpoints/`, then
restores any pre-existing target file after the run. Both source and staged paths
are recorded as repository-relative paths.

`doctor` treats the pinned git commit and the default checkpoint SHA-256 above as
required invariants when using the default checkpoint. A custom `backend.checkpoint`
without `allow_custom_checkpoint: true` is a hard failure so `run` cannot silently
skip staging. With the opt-in flag, doctor records a note and skips the default
SHA-256 pin (the custom weight is outside the published pin).

## Local patch: matplotlib colormap (required for matplotlib >= 3.9)

Pinned VDA `utils/dc_utils.py` calls `matplotlib.cm.get_cmap`, which was removed in
matplotlib 3.9+. With newer venvs, VDA can finish inference and then crash while
saving the visualization video—**before** writing `*_depths.npz`.

This repository records that deviation explicitly (Local patches ≠ none). The
orchestrator applies an **idempotent** compatibility fix via
`scripts.depth.backend_vda.ensure_vda_matplotlib_compat()` during `doctor` and
before `run`:

```python
try:
    colormap = np.array(cm.get_cmap("inferno").colors)
except AttributeError:  # matplotlib >= 3.9
    from matplotlib import colormaps
    colormap = np.array(colormaps["inferno"].colors)
```

Reference copy of the intended change: `configs/depth/patches/vda_matplotlib_colormap.md`.
Re-cloning VDA at the pinned commit without running `doctor`/`run` leaves the stock
file; the next `doctor` or `run` re-applies the patch.

Do not commit the `third_party/Video-Depth-Anything` tree into this repo.

## Local environment notes (Windows)

- Project venv: `.venv\Scripts\python.exe`
- Validated local stack for doctor: `torch 2.1.1+cu118`, CUDA available on RTX 3060 Laptop
- `ffmpeg` must be on `PATH` before assembling frames into a temp video
- Prefer PowerShell commands on native Windows; do not assume `dev.ps1` exists

Do not use floating `main` / `latest` as the sole version identifier in run records.
