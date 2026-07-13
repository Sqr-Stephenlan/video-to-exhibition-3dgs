# Depth prior configs

Default experiment line for this branch:

- Backend: Video Depth Anything
- Depth type: **relative**
- Encoder: **vitb** (Base)
- Config: `configs/depth/default_vitb.yaml`

See `backend_pin.md` for clone/commit/weight pins.

Until sample frames exist, use:

```text
python scripts/depth/run_depth_prior.py doctor --config configs/depth/default_vitb.yaml
```

`run` requires a real `frames_manifest.json` (see `frames_manifest.example.json`).
