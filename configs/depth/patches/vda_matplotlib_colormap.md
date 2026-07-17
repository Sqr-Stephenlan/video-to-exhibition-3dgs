# Patch: VDA matplotlib colormap compat

| Field | Value |
|---|---|
| Backend | Video Depth Anything @ `4f5ae23172ba60fd7bc11ef671cca678842c7072` |
| File | `third_party/Video-Depth-Anything/utils/dc_utils.py` |
| Reason | `matplotlib.cm.get_cmap` removed in matplotlib 3.9+; stock VDA crashes while saving depth visualization video and may never write `*_depths.npz` |
| Applied by | `scripts.depth.backend_vda.ensure_vda_matplotlib_compat()` (idempotent; called from `doctor` and `run`) |
| Tracked in root git? | No (third_party clone stays untracked); this note + orchestrator apply logic are tracked |

## Before

```python
colormap = np.array(cm.get_cmap("inferno").colors)
```

## After

```python
try:
    colormap = np.array(cm.get_cmap("inferno").colors)
except AttributeError:  # matplotlib >= 3.9
    from matplotlib import colormaps
    colormap = np.array(colormaps["inferno"].colors)
```
