# Depth prior I/O contract

Single-page contract for the `research/depth-prior` module (Video Depth Anything Relative Base / `vitb`).

Related files:

| Role | Path |
|---|---|
| CLI entry | `scripts/depth/run_depth_prior.py` |
| Preprocess adapter | `scripts/depth/adapt_preprocess_manifest.py` |
| Default config | `configs/depth/default_vitb.yaml` |
| Low-VRAM smoke config | `configs/depth/smoke_joint.yaml` (resolution only; deferred fields still unused) |
| Backend pin | `configs/depth/backend_pin.md` |
| Setup / commands | `scripts/depth/README.md` |
| Frames manifest example | `configs/depth/frames_manifest.example.json` |

Depth outputs are a **soft prior** only. Do not treat them as metric ground truth.

---

## Commands

```powershell
# Environment checks (no frames required)
.\.venv\Scripts\python.exe scripts\depth\run_depth_prior.py doctor

# Full run (needs frames_manifest + frame files; --video-id / --run-id expand IO templates)
.\.venv\Scripts\python.exe scripts\depth\run_depth_prior.py run --video-id demo_short --run-id baseline

# Validate config + manifest only
.\.venv\Scripts\python.exe scripts\depth\run_depth_prior.py run --video-id demo_short --run-id baseline --dry-run
```

Optional: `--config <repo-relative-yaml>` (default `configs/depth/default_vitb.yaml`).
When `io.*` paths contain `{video_id}` / `{run_id}`, pass matching CLI flags (or set `runtime.run_id`). `--video-id` must match `frames_manifest.video_id` when both are set.

---

## Prerequisites

### Always (for `doctor` and `run`)

| Requirement | Notes |
|---|---|
| Project `.venv` | Install `configs/depth/requirements.txt` |
| VDA clone | `third_party/Video-Depth-Anything` at pinned commit (see `backend_pin.md`) |
| VDA Python deps | Install backend `requirements.txt` into the same venv |
| Default weights | `third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitb.pth` with pinned SHA-256 |
| `ffmpeg` on `PATH` | Used to assemble image sequences into a temp video |
| PyTorch | Importable; CUDA build recommended for real inference |

`doctor` additionally enforces:

- Backend clone has a readable git `HEAD` matching `backend.commit`
- Default checkpoint SHA-256 matches the pin in `backend_pin.md`
- A non-empty `backend.checkpoint` requires `backend.allow_custom_checkpoint: true`; otherwise doctor/`run` fail closed. With the opt-in, default SHA pin is skipped and a note is recorded.

### Extra for `run`

| Requirement | Notes |
|---|---|
| `data/manifests/<video_id>/<run_id>/frames_manifest.json` | Default `io.frames_manifest` template; expand with `--video-id` / `--run-id` |
| Frame image files | Every selected frame `path` must exist and stay inside the repo |
| `doctor` clean | Non–dry-run `run` aborts if doctor reports issues |
| Non-conflicting outputs | Existing depth NPZ / depth_manifest / run_record refuse overwrite unless `runtime.overwrite: true` |

`doctor` does **not** need sample frames. `run` does.

---

## Inputs

### Config (`configs/depth/default_vitb.yaml`)

All paths are **repository-relative**. Absolute paths and `../` escapes are rejected.

| Key | Purpose |
|---|---|
| `backend.repo_dir` | Local VDA clone |
| `backend.commit` | Expected git pin |
| `backend.encoder` / `depth_type` | Default: `vitb` / `relative` |
| `backend.checkpoint` | Empty = default VDA filename; else project-relative custom weights (staged temporarily) |
| `backend.allow_custom_checkpoint` | Must be `true` to use a non-empty `checkpoint`; default `false` |
| `io.frames_manifest` | Input frame list (may use `{video_id}` / `{run_id}`) |
| `io.depth_dir` | Per-frame depth NPZ output directory (namespaced by `{video_id}/{run_id}`) |
| `io.depth_manifest` | Depth manifest output |
| `io.run_record` | Run record output |
| `runtime.run_id` | Default run namespace when CLI `--run-id` is omitted (default `default`) |
| `runtime.target_fps` | Fixed fps used when assembling each VDA temp video (VDA does not consume container PTS) |
| `runtime.infer_per_segment` | When all selected frames have `segment_id`, run one VDA invocation per segment (default `true`) |
| `runtime.dedupe_identical_timestamps` | Collapse same-`timestamp_sec` frames with identical bytes **within each VDA batch** (default `true`) |
| `runtime.overwrite` | Allow clobbering existing depth outputs (default `false`) |

### Frames manifest

Default path template: `data/manifests/{video_id}/{run_id}/frames_manifest.json`
Example schema: `configs/depth/frames_manifest.example.json`

```json
{
  "schema_version": "1.0",
  "video_id": "demo_short",
  "source_video": "data/raw_videos/demo_short.mp4",
  "frames": [
    {
      "frame_id": "demo_short_000001",
      "path": "data/frames/demo_short/000001.png",
      "timestamp_sec": 0.0,
      "width": 1280,
      "height": 720,
      "selected": true,
      "reason": "example"
    }
  ]
}
```

| Field | Required | Rules |
|---|---|---|
| `schema_version` | yes | Currently only `"1.0"` |
| `frames` | yes | Non-empty list |
| `frames[].frame_id` | yes | 1–128 chars of `[A-Za-z0-9._-]`, must start with alphanumeric; no path segments |
| `frames[].path` | yes | Repository-relative image path |
| `frames[].selected` | no | Default `true`; `false` skips the frame |
| `frames[].timestamp_sec` | no | If any selected frame has it, **all** selected frames must; used to order frames **within** a VDA batch |
| `frames[].width` / `height` | required after adapt | Adapter fills from image bytes when preprocess omits them |
| `frames[].segment_id` / `reason` / quality fields | no | Preserved from preprocess when present; copied into depth_manifest. When every selected frame has `segment_id` and `runtime.infer_per_segment` is true, each segment is a separate VDA invocation |
| `video_id` / `source_video` | strongly recommended | Combined with `run_id`, namespaces IO so concurrent preprocess routes do not clobber each other |

At least one frame must remain after selection.

Frame→depth pairing is **strict positional**: after `selected_frames()` ordering (per-segment sort/dedupe when enabled), concatenated VDA depth slices map `depths[i]` to selected frame `i`. A count mismatch fails fast.

### Temporal strategy (important)

Pinned Video Depth Anything consumes a decoded frame array / index windows. It does **not** use ffmpeg container PTS to reset temporal state. Therefore:

- Writing real inter-frame gaps into a VFR container does **not** restore model timing across large gaps or scene cuts.
- This module resets temporal state by **separate VDA invocations per `segment_id`** (`runtime.infer_per_segment: true`).
- Within one invocation, frames are assembled at fixed `runtime.target_fps`.
- Identical-timestamp duplicates are collapsed only **within** a batch; conflicting content at the same timestamp inside one batch fails closed. Cross-segment same timestamps are allowed because they never share a VDA window.

### Runtime data flow (brief)

1. Expand `{video_id}` / `{run_id}` IO templates (`--video-id`, `--run-id` / `runtime.run_id`, and/or manifest `video_id`)
2. Load selected frames; partition into VDA batches (usually one per `segment_id`); refuse overwrite unless enabled
3. For each batch: assemble a temporary MP4 at fixed `target_fps`, run VDA, collect `*_depths.npz`
4. Concatenate depth arrays in selected-frame order; write per-frame NPZ under `data/depth/<video_id>/<run_id>/` + namespaced depth manifest + run record

Temp-video assembly uses ffmpeg concat with a trailing duplicate file entry (so the last
frame’s `duration` applies). The orchestrator passes `-frames:v N` so the encoded video
contains exactly `N` frames and matches strict positional depth mapping.

Before calling VDA, the orchestrator idempotently applies the documented matplotlib 3.9+
colormap patch under `third_party/Video-Depth-Anything/utils/dc_utils.py`
(see `configs/depth/backend_pin.md`).

---

## Outputs

Default paths from `io.*` in `default_vitb.yaml`.

### Per-frame depth maps

Path pattern: `data/depth/<video_id>/<run_id>/<frame_id>.npz`

| Key | Content |
|---|---|
| `depth` | One array per frame (slice of VDA `depths[i]`) |

### Depth manifest

Default path: `data/manifests/<video_id>/<run_id>/depth_manifest.json`

```json
{
  "schema_version": "1.0",
  "source_video_id": "demo_short",
  "source_frames_manifest": "data/manifests/frames_manifest.json",
  "frame_depth_mapping": "strict_positional",
  "depth_scale": {
    "mode": "relative",
    "unit": null
  },
  "backend": {
    "name": "video-depth-anything",
    "commit": "<actual or pinned commit>",
    "encoder": "vitb",
    "depth_type": "relative",
    "checkpoint": null
  },
  "frames": [
    {
      "frame_id": "demo_short_000001",
      "rgb_path": "data/frames/demo_short/000001.png",
      "depth_path": "data/depth/demo_short_000001.npz",
      "depth_type": "relative",
      "confidence_path": null,
      "depth_index": 0
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `source_video_id` | From input `video_id` (may be null) |
| `source_frames_manifest` | Repository-relative path of the input frames manifest |
| `frame_depth_mapping` | Always `strict_positional` |
| `depth_scale.mode` | `relative` or `metric` |
| `depth_scale.unit` | `null` for relative soft prior; `"meters"` when metric |

`confidence_path` is always `null` in this PR (mask / confidence export deferred).

### Run record

Default path: `outputs/reconstructions/depth_prior/<video_id>/<run_id>/run_record.json`

| Field | Meaning |
|---|---|
| `module` | `"depth-prior"` |
| `started_at` / `finished_at` | UTC ISO timestamps |
| `config` / `frames_manifest` | Repository-relative paths |
| `backend_*` / `encoder` / `depth_type` | Backend identity |
| `checkpoint` / `checkpoint_vda_path` / `checkpoint_sha256` | Weight source and staged path |
| `checkpoint_staged` | Whether a custom (or non-identity) file was copied into VDA’s expected name |
| `checkpoint_target_restored` / `checkpoint_target_originally_present` | Staging restore metadata |
| `command` | Sanitized argv (`<project>` / `<temp>` / `<external>` placeholders; no machine absolute paths) |
| `command_returncode` / `stdout_tail` / `stderr_tail` | Subprocess outcome |
| `vda_depths_npz` | Basename of the VDA NPZ artifact (temp dir is not retained) |
| `outputs.depth_dir` / `depth_manifest` / `run_record` / `frame_count` | Written artifacts |

---

## Deferred (config present, not implemented)

Documented in config comments and `scripts/depth/README.md`. Do **not** assume these change behavior yet:

- `runtime.device`
- `runtime.skip_existing`
- `io.mask_dir`
- per-frame `confidence_path`
- LongSplat mapping / materialize / fail-closed consumer (owned by PR #3)

---

## Bridge from video preprocessing

Preprocess PR (`feature/preprocess-video`, not part of this PR) writes:

`data/manifests/<video_id>/preprocess_manifest.json`

with `frames[].id` (not `frame_id`). Depth-prior expects a frames_manifest with
`frames[].frame_id`, defaulting to `data/manifests/<video_id>/<run_id>/frames_manifest.json`.

Convert with:

```powershell
.\.venv\Scripts\python.exe scripts\depth\adapt_preprocess_manifest.py `
  data\manifests\<video_id>\preprocess_manifest.json `
  --run-id baseline
# default --output data/manifests/{video_id}/{run_id}/frames_manifest.json
```

Adapter rules (fail closed):

- selected frames must have a non-null repository-relative `path`
- `width`/`height` are required (filled from image bytes when preprocess omits them)
- preserve `segment_id`, selection `reason`, and quality provenance fields when present
- `timestamp_sec` must be all-present or all-absent (null counts as absent)
- paths may not escape the repository root
- duplicate `frame_id` values are rejected

Then run depth-prior with matching `--video-id` and `--run-id`. Notes for limited GPUs:

- Low VRAM / OOM: use `configs/depth/smoke_joint.yaml` (`input_size: 308`, `max_res: 512`)
- Broken `xformers` CUDA build: uninstall or reinstall a wheel matching local torch/CUDA; lowering resolution alone does not fix xformers operator errors

This branch records **frame↔depth** correspondence only. Camera pose / intrinsics are out of scope for depth-prior (handled by later SfM / LongSplat stages).

## Producer contract vs LongSplat consumer (PR #3)

Depth-prior is a **producer** of versioned `depth_manifest.json` + per-frame NPZ.
It does **not** implement LongSplat `prepare_input` materialization, `frame_mapping`
binding, `depth_source=vda` wiring, or fail-closed consumer checks. Those belong on
`research/longsplat-route` (PR #3).

### What this module guarantees (producer)

| Field | Meaning for consumers |
|---|---|
| `frames[].frame_id` | Stable id (often from preprocess `id` after adapt) |
| `frames[].rgb_path` | Repository-relative RGB used for depth inference |
| `frames[].depth_path` | Repository-relative NPZ with key `depth` |
| `frames[].segment_id` | Present when upstream provided it; used for VDA batching |
| `frame_depth_mapping` | Always `strict_positional` vs the selected frames order at run time |
| `vda_batching` | Records per-segment VDA invocation sizes when batching is used |

### What LongSplat / PR #3 must own

1. Interpret its own `frame_mapping` schema (list or dict) correctly.
2. Bind prepared stems (`frame_{id:06d}`) to depth_manifest `frame_id` / `rgb_path`.
3. Materialize NPZ → `frame_{id:06d}_depth.npy` next to prepared images.
4. Enable training with an explicit depth source flag (e.g. `depth_source=vda`).
5. **Fail closed** when a prepared frame has no matching depth — never silently fall back to MASt3R.

Do not treat any helper formerly living under `scripts/depth/` as proof that real
LongSplat consumption is verified on this branch.

### Responsibility boundary

| Concern | Owner |
|---|---|
| Preprocess source schema, sizing policy, selection strategy | PR #2 (`feature/preprocess-video`) |
| VDA producer, output isolation, real temporal segmentation via per-segment VDA runs | this PR (`research/depth-prior`) |
| Mapping, prepared-stem materialize, `depth_source=vda`, fail-closed consumer | PR #3 (`research/longsplat-route`) |

### Joint testing note

A temporary merge branch may be used to verify the three-way wire, but
`requirements.txt` add/add conflicts across the three feature branches must be
resolved by hand (union of PyYAML/Pillow, OpenCV/scenedetect, plyfile, numpy bounds).
Do not permanently fold LongSplat/preprocess sources into this PR.

## Acceptance notes for this branch

| Goal | Status in current PR |
|---|---|
| `doctor` validates clone commit + default checkpoint hash | yes |
| CPU unit/integration tests for NPZ split, path bounds, staging restore, adapter | yes |
| GPU end-to-end `run` on real exhibition frames | deferred until sample frames + GPU report / waiver |
| Per-segment VDA invocations (`infer_per_segment`) | yes (when `segment_id` present) |
| Mask / confidence filtering | deferred |
| Camera pose fields in depth_manifest | out of scope (frame correspondence only) |
| LongSplat materialize / fail-closed consumer | out of scope (PR #3) |
