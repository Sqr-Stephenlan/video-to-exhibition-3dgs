# Preprocess Video Feature Status

Last updated: 2026-07-13

## Current Phase

Stage 2 - Minimal preprocess MVP implemented and locally unit-tested.

The feature now has a single CLI entrypoint, focused tests, and user-facing
documentation. Full end-to-end video smoke testing still requires FFmpeg and
ffprobe to be installed on PATH.

## Completed

- Read and followed `.codex/prompts/preprocess_video_feature.md`.
- Added `requirements.txt` with the minimal preprocess dependency set:
  `numpy`, `opencv-python`, `scenedetect[opencv]`, and `pytest`.
- Updated `dev.sh` so the project Python entrypoint can use either
  `.venv/bin/python` or Windows `.venv/Scripts/python.exe`.
- Added `scripts/preprocess_video.py` with:
  - CLI options for output root, video id, presets, segmentation settings,
    frame quality thresholds, rejected-frame saving, and forced overwrite.
  - `ffmpeg` and `ffprobe` availability checks with clear setup errors.
  - ffprobe metadata probing.
  - H.264 MP4 normalization.
  - time-window segmentation.
  - optional PySceneDetect scene segmentation with time-window fallback.
  - OpenCV frame sampling, blur/exposure scoring, average-hash duplicate
    filtering, selected/rejected frame handling, and manifest generation.
- Added `tests/test_preprocess_video.py` covering:
  - time-window segmentation and tiny-tail merge behavior.
  - offset window math.
  - blur, exposure, and duplicate hash scoring.
  - manifest relative paths and summary counts.
  - FFmpeg smoke flow, skipped when `ffmpeg` or `ffprobe` is unavailable.
- Added `docs/preprocess_video.md` with usage, requirements, outputs, manifest
  shape, and verification instructions.

## Verification

- Ran through the project Python entrypoint with Git Bash:
  `./dev.sh pytest`
- Result: `10 passed, 1 skipped`.
- The skipped test is the FFmpeg smoke test because this machine currently does
  not have `ffmpeg` and `ffprobe` on PATH.
- Ran CLI help successfully:
  `./dev.sh python scripts/preprocess_video.py --help`
- Ran the missing-FFmpeg failure path successfully; it exits with a clear setup
  message before writing outputs.

## Next Step

Install FFmpeg so both `ffmpeg` and `ffprobe` are on PATH, then run baseline and
longsplat smoke tests on a real or tiny synthetic video:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/<video>.mp4 --video-id <id> --preset baseline --force
./dev.sh python scripts/preprocess_video.py data/raw_videos/<video>.mp4 --video-id <id> --preset longsplat --force
```

## Open Decisions

- No new virtual environment is needed; `.venv` already exists.
- No dependency installation was run in this session because the required Python
  packages were already present in `.venv`.
- Scene segmentation is included in the MVP, but real scene-cut behavior still
  needs validation after FFmpeg is available.

## Current Non-Goals

- Do not clone LongSplat, MASt3R, DUSt3R, Depth Anything, gsplat, SuperSplat, or
  other large third-party repositories for this feature.
- Do not implement reconstruction, depth estimation, model compression, web
  display, annotation UI, or model export.

## Status Update Rules

When work continues, update this file before finishing the conversation if any
of the following changed:

- Current phase.
- Completed implementation or tests.
- Next recommended step.
- New blocker, risk, or decision.
- Dependency or file-layout convention.
