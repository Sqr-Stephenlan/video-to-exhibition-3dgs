# Preprocess Video

This feature converts one raw exhibition video into normalized video, time or
scene segments, selected reconstruction frames, and a deterministic manifest for
later pipeline stages.

It owns only this part of the exhibition 3DGS pipeline:

```text
raw video
  -> normalized video
  -> video segments
  -> selected reconstruction frames
  -> preprocess_manifest.json
```

It does not run reconstruction, depth estimation, COLMAP, LongSplat, MASt3R,
DUSt3R, 3DGS training, model export, web display, or annotation UI. Downstream
stages should consume the manifest and generated frame paths from this feature
instead of re-scanning directories or guessing preprocessing settings.

## Requirements

- Use the project Python entrypoint for every Python command:

```bash
./dev.sh python ...
./dev.sh pytest
./dev.sh pip ...
```

- Install FFmpeg so both `ffmpeg` and `ffprobe` are on `PATH`.
- Python dependencies are listed in `requirements.txt`.
- Put source videos under `data/raw_videos/` when practical.

If FFmpeg is missing, the CLI fails before writing outputs with a setup message.
FFmpeg-dependent tests are skipped when `ffmpeg` or `ffprobe` is unavailable.

## Quick Start

Run the baseline preset on one local video:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 --video-id sample
```

For repeated real-video tuning, use the checked-in tuning profile:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 \
  --video-id sample_tuning \
  --config configs/preprocess/baseline_real_video.json \
  --force
```

For higher-fidelity reconstruction frames, keep compatible MP4 segment outputs
but sample PNG frames directly from the source video:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 \
  --video-id sample_png \
  --frame-source source \
  --frame-format png \
  --force
```

Use `--force` only when replacing generated outputs for the same `video_id` is
intended. Without `--force`, existing output directories are protected.

## CLI Options

The single entrypoint is:

```bash
./dev.sh python scripts/preprocess_video.py <source_video> --video-id <id> [options]
```

| Option | Default | Purpose |
|---|---:|---|
| `<source_video>` | required | Raw input video path. |
| `--output-root` | `data` | Root for generated `segments/`, `frames/`, and `manifests/`. |
| `--video-id` | required | Stable output identifier. Use only letters, numbers, `_`, `-`, and `.`. |
| `--config` | none | JSON tuning file. See `configs/preprocess/baseline_real_video.json`. |
| `--preset` | `baseline` | Preset name: `baseline` or `longsplat`. |
| `--target-fps` | preset value | Frame sampling rate per segment. |
| `--max-long-edge` | preset value | Maximum long edge for normalized video and source-sampled frames. |
| `--segment-method` | `scene,time` | Segmentation strategy: `scene`, `time`, or `scene,time`. |
| `--segment-length-sec` | preset value | Maximum segment window length. |
| `--segment-overlap-sec` | preset value | Overlap when long windows are split. |
| `--min-segment-sec` | `2.0` | Merge or avoid tiny segments below this length. |
| `--blur-threshold` | `40.0` | Reject frames with Laplacian variance below this value. |
| `--overexposed-ratio` | `0.6` | Reject frames above this near-white pixel ratio. |
| `--underexposed-ratio` | `0.6` | Reject frames above this near-black pixel ratio. |
| `--duplicate-hash-threshold` | `4` | Reject near duplicates when average-hash distance is less than or equal to this value. |
| `--save-rejected` / `--no-save-rejected` | `false` | Control whether rejected frame images are written. Manifest rows are always written. |
| `--frame-format` | `jpg` | Frame image format: `jpg` or `png`. |
| `--frame-source` | `segment` | Sample frames from generated segment MP4s or directly from the source video. |
| `--force` | `false` | Delete and replace generated outputs for the same `video_id`. |

Presets:

| Preset | Target fps | Max long edge | Segment length | Segment overlap | Intended use |
|---|---:|---:|---:|---:|---|
| `baseline` | `5` | `1600` | `30s` | `10s` | General preprocessing and COLMAP/3DGS-compatible frame sets. |
| `longsplat` | `10` | `512` | `30s` | `10s` | Low-resolution LongSplat smoke tests before scaling up. |

Configuration precedence is:

```text
built-in preset defaults
  -> JSON config values
  -> explicit CLI options
```

This means temporary overrides can be tested without editing the shared tuning
file:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 \
  --video-id sample_blur_55 \
  --config configs/preprocess/baseline_real_video.json \
  --blur-threshold 55 \
  --no-save-rejected \
  --force
```

The JSON file accepts only preprocessing settings: `preset`, `target_fps`,
`max_long_edge`, segmentation settings, quality thresholds, `frame_format`,
`frame_source`, and `save_rejected`. Input paths, `video_id`, output paths, and
`force` remain explicit CLI options. Unknown fields, invalid value types, and
invalid choices fail with a clear error so tuning typos are not ignored.

## Outputs

For `--video-id sample`, outputs are:

```text
data/segments/sample/
  normalized.mp4
  segment_0001.mp4
  segment_0002.mp4

data/frames/sample/
  selected/segment_0001/*.jpg or *.png
  rejected/segment_0001/*.jpg or *.png

data/manifests/sample/
  preprocess_manifest.json
```

Rejected images are written only with `--save-rejected`. Rejected frame manifest
entries still exist when rejected images are not saved, with `path` set to
`null`.

By default frames are sampled from generated segment MP4s and written as JPGs.
Use `--frame-source source --frame-format png` to avoid sampling from the
second-generation H.264 segment files and avoid JPEG compression for final
reconstruction frame assets. The normalized and segment videos remain H.264 MP4s
for tool compatibility.

Generated videos, frames, manifests, logs, model files, and temporary outputs
should stay out of Git. Commit source code, tests, docs, and reusable tuning
configs only.

## Manifest Contract

The manifest schema version is `1.0` and includes:

- `video_id`
- `source`
- `normalized`
- `settings`
- `segments`
- `frames`
- `summary`

`source` and `normalized` include stable relative paths, file/video metadata,
dimensions, fps, duration, codec, and source size where available.

Each segment includes:

```text
id, path, index, start_sec, end_sec, duration_sec, reason
```

Each frame includes:

```text
id, segment_id, path, timestamp_sec, frame_index, selected,
blur_score, overexposed_ratio, underexposed_ratio, duplicate_score,
reject_reasons
```

`summary` includes total segment/frame counts, selected and rejected frame
counts, per-segment frame counts, and reject counts by reason. Downstream stages
should use:

- `frames` filtered by `selected == true` for reconstruction images.
- `segment_id`, `timestamp_sec`, and `frame_index` to preserve temporal context.
- `segments` for segment windows and generated segment MP4 paths.
- `settings` to record exactly how the frame set was produced.

Paths are stable relative paths when outputs are under the repository root.

## Consumer Contract

Preprocess video is a **producer** of normalized MP4 segments, selected frame
assets, and `preprocess_manifest.json`. It does **not** rename frames for a
specific reconstruction backend, generate COLMAP files, run LongSplat, or create
depth / pose / camera-intrinsic records. Each downstream branch owns the adapter
from this manifest into its own input tree.

### What this module guarantees

| Field or artifact | Meaning for consumers |
|---|---|
| `schema_version` | Manifest schema version. Current value is `1.0`. |
| `video_id` | Stable run identifier used in `data/segments/<video_id>`, `data/frames/<video_id>`, and `data/manifests/<video_id>`. |
| `source.path` | Repository-relative raw video path used for this run. |
| `normalized.path` | Repository-relative H.264 MP4 normalized video path. |
| `settings` | Resolved settings after preset defaults, config, and CLI overrides. |
| `segments[]` | Deterministically ordered segment windows and generated segment MP4 paths. |
| `frames[]` | Deterministically ordered sampled frame records. |
| `frames[].id` | Stable preprocess frame id for this manifest, not a backend training index. |
| `frames[].segment_id` | Segment id that ties the frame back to `segments[].id`. |
| `frames[].path` | Repository-relative selected/rejected image path, or `null` when a rejected image was not saved. |
| `frames[].timestamp_sec` | Timestamp in the segment/source timeline used by preprocessing. |
| `frames[].frame_index` | Source or segment frame index, depending on `settings.frame_source`. |
| `frames[].selected` | `true` only for frames intended as reconstruction candidates. |
| `frames[].reject_reasons` | Reasons a frame was rejected: `blur`, `overexposed`, `underexposed`, and/or `duplicate`. |
| `summary` | Counts for quick validation and handoff reporting. |

### What downstream consumers must do

LongSplat, COLMAP, gsplat, official 3DGS, depth-prior, and evaluation branches
must read `preprocess_manifest.json` and materialize their own backend-specific
input layout. A reliable bridge must:

1. Treat the manifest `frames[]` array order as the authoritative preprocess
   frame order for this run.
2. Filter reconstruction inputs with `selected == true`; rejected rows are for
   audit and tuning, not training input.
3. Resolve frame images through `frames[].path`, not by scanning
   `data/frames/<video_id>/selected/**` and guessing sort order.
4. Preserve a mapping from backend training image name/index back to
   `frames[].id`, `segment_id`, `timestamp_sec`, and `frame_index`.
5. Generate backend-specific names in the consumer branch, such as
   `frame_{index:06d}.jpg`, COLMAP `images/`, or LongSplat prepared image stems.
6. Fail closed, or at least emit an ERROR, when a selected manifest frame cannot
   be resolved on disk.
7. Record the manifest path and `settings` used for the backend run so later
   quality or pose failures can be traced back to preprocessing decisions.

Do **not** derive backend names from preprocess image basenames such as
`frame_000001_t0003.200.png`. Those filenames encode preprocess sample order and
timestamp only; they are not guaranteed to match LongSplat, COLMAP, gsplat, or
depth-prior training stems.

### LongSplat bridge note

The `longsplat` preset only prepares low-resolution smoke-test-friendly
preprocess outputs. It does not mean the preprocess manifest is already in
LongSplat's input format.

A LongSplat adapter should read the selected `frames[]`, copy or convert them
into LongSplat's prepared image tree, write a `frame_mapping.json`, and keep that
mapping as the authority for later depth injection or render comparison. Depth
or pose files added by other branches must be named from the LongSplat prepared
training stems, not from preprocess frame basenames.

### Joint testing note

A temporary merge branch may be used to verify preprocess -> depth-prior ->
LongSplat or preprocess -> COLMAP / 3DGS wiring. Keep the responsibility split:
this feature owns the manifest and selected frames; consumer branches own their
input adapters, backend naming, pose/depth records, training commands, and model
outputs. Do not permanently fold consumer code into the preprocess PR.

## Tuning Guidance

Quality filters are intentionally conservative. The goal is to reject obvious
bad frames while keeping enough coverage for downstream reconstruction.

- Increase `blur_threshold` to reject more soft or motion-blurred frames.
- Decrease `blur_threshold` when low-texture but usable footage would otherwise
  lose too much coverage.
- Keep `overexposed_ratio` and `underexposed_ratio` near `0.6` unless visual
  review shows severe lighting failures are passing through.
- `duplicate_hash_threshold` is easy to misread: rejection happens when
  `duplicate_score <= duplicate_hash_threshold`. Lower values are less strict;
  higher values reject more near duplicates.
- Reducing `target_fps` lowers disk use and review volume, but may remove useful
  reconstruction coverage.
- Reducing `segment_overlap_sec` lowers duplicated output, but can make segment
  handoff and later fusion weaker.
- Use `--save-rejected` during tuning so rejected frames can be inspected. Turn
  it off for production-style runs when disk pressure matters.
- Prefer `--frame-source source --frame-format png` for final reconstruction
  frame sets when storage allows it.

Suggested real-video loop:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/<video>.mp4 \
  --video-id <video>_candidate \
  --config configs/preprocess/baseline_real_video.json \
  --force
```

Then inspect:

- `data/manifests/<video_id>/preprocess_manifest.json`
- `summary.selected_frames` and `summary.rejected_frames`
- `summary.reject_reasons`
- selected frame quality under `data/frames/<video_id>/selected/`
- rejected frames when `save_rejected` was enabled

If a sample rejects nearly everything, lower `blur_threshold` first. If too many
near-identical frames remain, raise `duplicate_hash_threshold` carefully. If the
selected frames look sharp but too sparse, raise `target_fps` or lower the blur
threshold depending on the failure mode.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `ffmpeg` or `ffprobe` setup error | FFmpeg is not on `PATH`. | Install FFmpeg and reopen the shell so PATH updates are visible. |
| `Output already exists` | The same `video_id` was already generated. | Use a new `--video-id` or rerun with `--force`. |
| `--video-id` validation error | The id contains spaces or unsafe characters. | Use names like `ios_test_blur200_fps5`. |
| Config fails before writing outputs | Unknown field, wrong type, or invalid choice. | Fix the JSON key/value; config validation is intentionally strict. |
| Manifest has rejected rows with `path: null` | Rejected images were not saved. | Rerun with `--save-rejected` for visual review. |
| Selected frame count is too low | Blur/duplicate thresholds are too strict for that video. | Lower `blur_threshold` or lower `duplicate_hash_threshold`. |
| Selected frame count is too high | Filters are too loose or fps is too high. | Raise `blur_threshold`, raise `duplicate_hash_threshold`, or lower `target_fps`. |
| Source-frame timing looks suspicious | Variable-frame-rate phone footage may seek imperfectly through OpenCV. | Compare `frame-source segment` and `frame-source source` on a representative clip. |

## Verification

Run focused tests:

```bash
./dev.sh pytest
```

When FFmpeg is available, smoke test both presets on a local sample:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 --video-id sample_baseline --preset baseline --force
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 --video-id sample_longsplat --preset longsplat --force
```

For handoff, report the exact command, `video_id`, config file path, manifest
path, selected/rejected frame counts, and any known visual-quality concerns.
