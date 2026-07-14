# Preprocess Video

This feature converts one raw exhibition video into normalized video, time or
scene segments, selected reconstruction frames, and a deterministic manifest for
later pipeline stages.

It does not run reconstruction, depth estimation, 3DGS training, model export,
web display, or annotation UI.

## Requirements

- Use the project Python entrypoint for every Python command:

```bash
./dev.sh python ...
./dev.sh pytest
./dev.sh pip ...
```

- Install FFmpeg so both `ffmpeg` and `ffprobe` are on `PATH`.
- Python dependencies are listed in `requirements.txt`.

If FFmpeg is missing, the CLI fails before writing outputs with a setup message.
FFmpeg-dependent tests are skipped when `ffmpeg` or `ffprobe` is unavailable.

## Usage

Place source videos under `data/raw_videos/` and run:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 --video-id sample
```

Useful options:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 \
  --video-id sample \
  --preset baseline \
  --segment-method scene,time \
  --save-rejected \
  --force
```

For repeated real-video tuning, edit
`configs/preprocess/baseline_real_video.json` and pass it with `--config`:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 \
  --video-id sample_tuning \
  --config configs/preprocess/baseline_real_video.json \
  --force
```

Configuration precedence is built-in preset defaults, then values from the
JSON file, then explicitly supplied CLI options. This makes temporary overrides
possible without editing the shared tuning file:

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
The checked-in profile uses source-frame PNGs and keeps rejected images for
visual comparison; set `save_rejected` to `false` after tuning to reduce disk
usage.

For higher-fidelity reconstruction frames, keep the compatible MP4 segment
outputs but sample PNG frames directly from the source video:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 \
  --video-id sample_png \
  --frame-source source \
  --frame-format png \
  --force
```

Presets:

- `baseline`: 5 fps frame sampling, max long edge 1600, 30 second segments, 10 second overlap.
- `longsplat`: 10 fps frame sampling, max long edge 512, 30 second segments, 10 second overlap.

Quality filters are intentionally conservative. Frames are rejected for obvious
blur, severe overexposure or underexposure, and near duplicates. Borderline
frames are kept for downstream systems.

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
second-generation H.264 segment files and avoid JPEG compression for the final
frame assets. The normalized and segment videos remain H.264 MP4s for tool
compatibility.

Existing output directories are protected. Use `--force` to replace generated
outputs for the same `video_id`.

## Manifest

The manifest schema version is `1.0` and includes:

- `video_id`
- `source`
- `normalized`
- `settings`
- `segments`
- `frames`
- `summary`

Each segment includes `id`, `path`, `index`, `start_sec`, `end_sec`,
`duration_sec`, and `reason`.

Each frame includes `id`, `segment_id`, `path`, `timestamp_sec`, `frame_index`,
`selected`, `blur_score`, `overexposed_ratio`, `underexposed_ratio`,
`duplicate_score`, and `reject_reasons`.

Paths are stable relative paths when outputs are under the repository root.

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
