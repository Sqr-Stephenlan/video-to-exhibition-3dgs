# Preprocess Video Feature Status

Last updated: 2026-07-14

## Current Phase

Stage 2 - Minimal preprocess MVP implemented and aligned with the main-branch
test and API conventions. Fidelity frame extraction is implemented on branch
`codex/preprocess-fidelity-frames`.

The feature now has a single CLI entrypoint, focused tests, and user-facing
documentation. FFmpeg and ffprobe are now installed on PATH, and the FFmpeg
smoke test passes.

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
- Added `tests/unit/test_preprocess_video.py` covering:
  - time-window segmentation and tiny-tail merge behavior.
  - offset window math.
  - blur, exposure, and duplicate hash scoring.
  - manifest relative paths and summary counts.
  - FFmpeg smoke flow, skipped when `ffmpeg` or `ffprobe` is unavailable.
- Added `docs/preprocess_video.md` with usage, requirements, outputs, manifest
  shape, and verification instructions.
- Fixed Windows Unicode frame output: OpenCV image encoding is now written via
  Python file I/O, and write failures raise a preprocessing error instead of
  producing manifest paths for missing images.
- Added a regression test covering image writes under Unicode paths.
- Moved preprocess tests to `tests/unit/` so the main-branch CPU workflow
  discovers and executes them; retained the required
  `tests/integration/` and `tests/gpu/` directory placeholders.
- Migrated PySceneDetect scene-window conversion from deprecated
  `FrameTimecode.get_seconds()` calls to the `seconds` property.
- Installed FFmpeg 8.1.2 through winget as a user-scoped package. Both
  `ffmpeg` and `ffprobe` resolve from the installed `bin` directory, with the
  required `libx264` H.264 encoder and MP4 muxer available.
- Added an optional fidelity frame path:
  - `--frame-format jpg|png` controls selected/rejected frame image encoding.
  - `--frame-source segment|source` controls whether frames are sampled from
    generated segment MP4s or directly from the original source video using the
    segment time windows.
  - The default remains `segment` + `jpg` for backward compatibility.
  - `source` + `png` avoids second-generation segment-video sampling and JPEG
    compression for final reconstruction frames while keeping H.264 MP4
    normalized/segment outputs for downstream tool compatibility.

## Verification

- Ran through the project Python entrypoint with Git Bash:
  `./dev.sh pytest`
- Result with the installed FFmpeg bin directory on PATH: `12 passed`.
- Ran the main-branch CI test scope:
  `./dev.sh pytest tests/unit tests/integration`; result: `12 passed`.
- Ran `bash -n dev.sh` and `./dev.sh python -m compileall -q scripts tests/unit`.
- Ran a synthetic FFmpeg smoke flow through the CLI. It produced 2 segments,
  4 sampled frames, 2 selected frames, and a complete manifest; temporary
  outputs were removed after validation.
- Re-ran the diagnostic preprocess flow on `data/raw_videos/bad_video_1.mp4`
  with filtering disabled and rejected-frame saving enabled. The result had 1
  segment, 101 sampled frames, 87 selected JPGs, and 14 rejected JPGs; all
  selected and rejected manifest paths exist.
- Ran CLI help successfully:
  `./dev.sh python scripts/preprocess_video.py --help`
- Ran the fidelity frame extraction tests through the project Python entrypoint:
  `./dev.sh pytest tests/unit/test_preprocess_video.py`; result: `14 passed`.
- Ran the full test suite through the project Python entrypoint:
  `./dev.sh pytest`; result: `14 passed`.
- Ran compile check:
  `./dev.sh python -m compileall -q scripts tests/unit`.
- Ran the missing-FFmpeg failure path successfully; it exits with a clear setup
  message before writing outputs.
- Ran a real-video baseline pressure pass on
  `data/raw_videos/pressure_test.mp4` with `scene,time` segmentation and
  rejected-frame saving enabled. The 79.4 second HEVC source completed in 25.8
  seconds and produced 5 segments, 497 sampled frames, 281 selected frames, and
  216 rejected frames.
- Validated every pressure-pass artifact: all 6 generated MP4 files decode with
  FFmpeg, all 497 JPEG files decode with OpenCV, and all 503 paths referenced by
  the manifest exist.
- The pressure pass used about 150.3 MiB of generated output from an 11.1 MiB
  source when rejected JPEGs were retained.

## Next Step

Review the `pressure_test` frame-selection quality and scene boundaries, then
run the longsplat preset on a real sample video:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/<video>.mp4 --video-id <id> --preset longsplat --force
```

## Open Decisions

- No new virtual environment is needed; `.venv` already exists.
- FFmpeg is installed user-scoped through winget; a newly opened shell will
  inherit the updated user PATH.
- Existing VSCode/Git Bash processes opened before the installation retain the
  old PATH. A fresh VSCode process resolves both `ffmpeg` and `ffprobe` to the
  installed FFmpeg 8.1.2 `bin` directory.
- No dependency installation was run in this session because the required Python
  packages were already present in `.venv`.
- Scene segmentation is included in the MVP; real scene-cut behavior still
  needs validation on a representative exhibition video.
- A real-video diagnostic exposed and resolved a Windows Unicode-path issue in
  frame writing. The original run recorded 87 selected frames but wrote zero
  JPGs; the fixed run is consistent.
- The baseline pressure pass technically succeeds, but real-video quality needs
  manual acceptance. Scene detection isolated a 3 second fast-camera-motion
  window as `segment_0004`, where only 3 of 15 frames were selected, and visual
  sampling found some motion-blurred frames that still scored just above the
  current blur threshold of 40.
- PySceneDetect emits deprecation warnings for `FrameTimecode.get_seconds()` in
  the current scene-window conversion code. Resolved by using the `seconds`
  property; a representative real-video scene run is still recommended.
- The fidelity frame path improves the final frame assets but does not make
  resizing mathematically lossless; downscaling still resamples pixels. It is a
  project-compatible way to avoid extra H.264 and JPEG generation loss for
  downstream reconstruction inputs.
- Source-based frame sampling currently uses OpenCV FPS/frame-index seeking.
  The synthetic CFR smoke test passes, but representative phone/AR footage,
  especially variable-frame-rate input, still needs timing-alignment validation.

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
