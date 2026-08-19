# Depth prior configs

Default experiment line for this branch:

- Backend: Video Depth Anything
- Depth type: **relative**
- Encoder: **vitb** (Base)
- Config: `configs/depth/default_vitb.yaml`

Quality line for 12GB laptop GPUs (RTX 4080 Laptop):

- Encoder: **vitl** (Large), still **relative**
- Config: `configs/depth/default_vitl.yaml` (`max_res: 1024`, `runtime.run_id: vitl`)
- Do not use `smoke_joint.yaml` (512) for quality runs; if OOM, lower `max_res` or chunk, do not switch to metric

See `backend_pin.md` for clone/commit/weight pins.
See [`docs/depth_prior_io.md`](../../docs/depth_prior_io.md) for the full I/O contract (prerequisites, manifests, run record).

Until sample frames exist, use:

```text
python scripts/depth/run_depth_prior.py doctor --config configs/depth/default_vitb.yaml
python scripts/depth/run_depth_prior.py doctor --config configs/depth/default_vitl.yaml
```

`run` requires a real `frames_manifest.json` (see `frames_manifest.example.json`).
