# Codex Prompt: Preprocess Video Feature

## Role

You are working in the `video-to-exhibition-3dgs` project. The user is
responsible only for the `preprocess video` feature of a larger exhibition 3DGS
pipeline.

When asked about this feature, ground your answer in this prompt and in the
current repository. Prefer implementing concrete, small, testable pieces over
expanding into reconstruction, training, model compression, or web display.

Before doing work, also read the current feature status:

```text
.codex/status/preprocess_video_feature.md
```

Update that status file whenever the phase changes, when implementation starts
or completes, when tests are added or run, or when a new blocker/decision
appears. The status file is intentionally tracked so future Codex conversations
can resume without guessing.

## Source Context

The larger project target is:

```text
phone / AR-glasses exhibition video
  -> preprocessing
  -> pose/depth/geometric initialization
  -> 3DGS training
  -> post-processing
  -> display and annotation
```

This feature owns only:

```text
raw video
  -> normalized video
  -> video segments
  -> selected reconstruction frames
  -> preprocess_manifest.json
```

Do not implement LongSplat, MASt3R, DUSt3R, Depth Anything, COLMAP, gsplat,
SuperSplat, PlayCanvas, annotations, or model export as part of this feature.
The preprocess output should be compatible with those later stages.

## Repository Conventions

- Run Python only through:

```bash
./dev.sh python ...
./dev.sh pip ...
./dev.sh pytest ...
```

- Do not use bare `python`, `pip`, `pytest`, `mypy`, or `ruff`.
- If `.venv` exists, use it.
- If `.venv` is missing, ask before running `./dev.sh bootstrap`.
- Store implementation code in `scripts/`.
- Store feature docs in `docs/`.
- Store tests in `tests/`.
- Store generated preprocess data under `data/`.
- Keep generated data, videos, images, manifests, model files, weights, caches,
  and third-party source trees out of git. The repository `.gitignore` already
  ignores the main generated paths.

## Intended File Layout

```text
scripts/
  preprocess_video.py

docs/
  preprocess_video.md

tests/
  test_preprocess_video.py

data/raw_videos/
  <video_id>.mp4

data/segments/<video_id>/
  normalized.mp4
  segment_0001.mp4
  segment_0002.mp4

data/frames/<video_id>/
  selected/
    segment_0001/
      frame_000001_t0003.200.jpg
  rejected/
    segment_0001/
      frame_000001_t0003.200.jpg

data/manifests/<video_id>/
  preprocess_manifest.json
```

Generated files under `data/` should not be committed. The `.gitkeep` files that
preserve empty directories may stay committed.

## Dependencies

Use the minimal preprocess dependency set first:

```text
FFmpeg / ffprobe    system command for metadata, transcode, frame extraction, cutting
OpenCV              Python library for frame reading and quality metrics
PySceneDetect       Python library for automatic scene segmentation
NumPy               Python library for numeric operations
pytest              tests
```

Recommended `requirements.txt` entries:

```text
numpy
opencv-python
scenedetect[opencv]
pytest
```

Optional dependencies only when justified:

```text
scikit-image   SSIM
imagehash      perceptual hash
```

Large third-party repositories belong in `third_party/` only when needed by other
pipeline stages. Do not clone them for this preprocess feature.

## CLI Contract

Implement a single entrypoint:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/sample.mp4 --video-id sample
```

Recommended options:

```text
--output-root data
--video-id <name>
--preset baseline|longsplat
--target-fps <float>
--max-long-edge <int>
--segment-method scene|time|scene,time
--segment-length-sec <float>
--segment-overlap-sec <float>
--min-segment-sec <float>
--blur-threshold <float>
--overexposed-ratio <float>
--underexposed-ratio <float>
--duplicate-hash-threshold <int>
--save-rejected
--force
```

Preset defaults:

```text
baseline:
  target-fps: 5
  max-long-edge: 1600
  segment-length-sec: 30
  segment-overlap-sec: 10

longsplat:
  target-fps: 10
  max-long-edge: 512
  segment-length-sec: 30
  segment-overlap-sec: 10
```

The `longsplat` preset is for smoke tests and follows the architecture review:
low resolution first, around 10 fps, then scale up only after the route works.

## Behavior

1. Validate inputs.
   - Source video must exist.
   - `ffmpeg` and `ffprobe` must be available, or the CLI should fail with a
     clear setup message.
   - Existing output directories should not be overwritten unless `--force` is
     provided.

2. Probe metadata.
   - Use `ffprobe` to collect duration, width, height, fps, codec, stream info,
     and file size.

3. Normalize video.
   - Write `data/segments/<video_id>/normalized.mp4`.
   - Apply scale based on `--max-long-edge`.
   - Use a stable MP4/H.264 output suitable for later tools.
   - Preserve the source video in `data/raw_videos/`; do not modify it.

4. Segment video.
   - Prefer PySceneDetect content detection when `scene` is enabled.
   - Merge tiny segments shorter than `--min-segment-sec`.
   - Split segments longer than `--segment-length-sec` into overlapping time
     windows using `--segment-overlap-sec`.
   - If scene detection finds no useful cuts, fall back to time windows.
   - Write `segment_0001.mp4`, `segment_0002.mp4`, etc.

5. Extract and filter frames per segment.
   - Sample at `--target-fps`.
   - Compute blur score using Laplacian variance.
   - Compute exposure ratios using near-black and near-white pixel thresholds.
   - Detect near duplicates with a lightweight image hash or OpenCV-based
     similarity method.
   - Save selected frames to `data/frames/<video_id>/selected/...`.
   - Save rejected frames only when `--save-rejected` is set.
   - Prefer conservative filtering: hard reject obvious blur, severe exposure
     errors, and near duplicates; keep borderline frames for downstream systems.

6. Write manifest.
   - Write `data/manifests/<video_id>/preprocess_manifest.json`.
   - Use deterministic ordering and stable relative paths.

## Manifest Shape

The manifest should include:

```json
{
  "schema_version": "1.0",
  "video_id": "sample",
  "source": {
    "path": "data/raw_videos/sample.mp4",
    "size_bytes": 123456
  },
  "normalized": {
    "path": "data/segments/sample/normalized.mp4",
    "duration_sec": 30.0,
    "fps": 30.0,
    "width": 1280,
    "height": 720,
    "codec": "h264"
  },
  "settings": {},
  "segments": [],
  "frames": [],
  "summary": {}
}
```

Each segment entry should include:

```text
id, path, index, start_sec, end_sec, duration_sec, reason
```

Each frame entry should include:

```text
id, segment_id, path, timestamp_sec, frame_index, selected,
blur_score, overexposed_ratio, underexposed_ratio, duplicate_score,
reject_reasons
```

The summary should include counts by segment and counts by reject reason.

## Tests and Verification

Use tests that can run without committing large media:

- Generate a tiny synthetic video during the test with FFmpeg if available.
- Test segmentation window math without FFmpeg.
- Test blur/exposure/duplicate scoring using synthetic NumPy images.
- Test manifest writing with temporary directories.

Recommended commands:

```bash
./dev.sh pytest
./dev.sh python scripts/preprocess_video.py data/raw_videos/<video>.mp4 --video-id <id> --preset baseline
./dev.sh python scripts/preprocess_video.py data/raw_videos/<video>.mp4 --video-id <id> --preset longsplat
```

If `.venv` is missing, ask before bootstrapping.

## Acceptance Criteria

- A new user can run one command and get normalized video, segments, selected
  frames, and a manifest.
- Output paths follow the repository layout.
- Generated heavy files are ignored by git.
- The manifest is enough for later reconstruction stages to consume frames by
  segment and understand why frames were kept or rejected.
- The implementation is documented in `docs/preprocess_video.md`.
- The feature has focused tests for core logic.

## Out of Scope

- Training or running 3DGS.
- Running LongSplat, MASt3R, DUSt3R, Depth Anything, COLMAP, or gsplat.
- Model cleanup, compression, or conversion.
- Web viewer or PlayCanvas work.
- 3D annotation UI or annotation schema beyond preserving compatible frame and
  segment metadata.
