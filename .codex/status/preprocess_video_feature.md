# Preprocess Video Feature Status

Last updated: 2026-07-15

## Current Phase

Stage 2 - Minimal preprocess MVP implemented, review feedback addressed, and
aligned with the main-branch test and API conventions. Fidelity frame
extraction is implemented on branch `codex/preprocess-fidelity-frames`.

The feature now has a single CLI entrypoint, focused tests, and user-facing
documentation. FFmpeg and ffprobe are now installed on PATH, the FFmpeg smoke
test passes, and the remaining Ready for review checklist items are now in
progress through the PR body update.

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
- Addressed the outstanding review feedback by:
  - ensuring `cv2.VideoCapture` is released in a `finally` block even when
    frame sampling raises an error;
  - adding `-hide_banner -loglevel error` to per-segment FFmpeg calls so long
    runs do not buffer banner noise in memory.
- Added regression tests covering:
  - quiet segment FFmpeg command construction;
  - `VideoCapture` release on frame-write failure.
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
- Added config-driven parameter tuning:
  - `--config <path>` loads preprocessing settings from a dependency-free JSON
    file.
  - Settings resolve in the order built-in preset defaults, JSON config, then
    explicit CLI options.
  - Unknown fields, invalid types, and invalid choices fail with clear errors
    instead of silently falling back to defaults.
  - `--save-rejected` now also supports `--no-save-rejected` so CLI overrides
    work in both directions.
  - Added `configs/preprocess/baseline_real_video.json` as the editable
    real-video tuning profile and documented the repeatable tuning workflow.
- Ran a real-video parameter sweep against `data/raw_videos/pressure_test.mp4`
  using the checked-in baseline config plus targeted CLI overrides for:
  - blur threshold: `40`, `50`, `55`, `60`
  - duplicate hash threshold: `4` and `3`
  - target FPS: `5` and `4`
  - segment overlap: `10` and `6`
  - minimum segment length: `2` and `3`
  - exposure ratios: `0.6` and `0.5`
- Confirmed an important tuning semantic from the implementation: lower
  `duplicate_hash_threshold` values are less strict, because duplicate rejection
  happens when `duplicate_score <= duplicate_hash_threshold`.
- Identified the first tuning pass recommendation for this sample:
  - keep `duplicate_hash_threshold` at `4`
  - increase `blur_threshold` to around `55`
  - optionally reduce `target_fps` from `5` to `4` when disk and frame-count
    pressure matter more than maximizing coverage
  - leave exposure thresholds unchanged for now because no frames on
    `pressure_test.mp4` tripped either exposure filter at `0.6` or `0.5`

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
- Re-ran the full test suite after adding JSON tuning support through the
  project Python entrypoint: `./dev.sh pytest`; result: `20 passed`.
- Re-ran `./dev.sh pytest tests/unit/test_preprocess_video.py` after the review
  feedback fix; result: `20 passed, 2 skipped`.
- Re-ran `./dev.sh python -m compileall -q scripts tests/unit`.
- Re-ran the full test suite after the review feedback fix:
  `./dev.sh pytest`; result: `20 passed, 2 skipped`.
- Ran CLI help and compile checks after the config change:
  `./dev.sh python scripts/preprocess_video.py --help` and
  `./dev.sh python -m compileall -q scripts tests/unit`.
- Verified the checked-in tuning profile resolves to `baseline`, source-frame
  PNG output, and rejected-frame saving, while an explicit
  `--blur-threshold 55` overrides the JSON value.
- Verified a missing `--config` path exits with a clear one-line error before
  output directories are written.
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
- Ran a focused real-video tuning sweep with source-frame PNG output and saved
  rejected frames. Key `pressure_test.mp4` results:
  - baseline config (`blur=40`, `dup=4`, `fps=5`, `overlap=10`):
    497 sampled, 285 selected, 212 rejected
  - `blur=55`, `dup=4`, `fps=5`, `overlap=10`:
    497 sampled, 264 selected, 233 rejected
  - `blur=55`, `dup=4`, `fps=5`, `overlap=6`:
    457 sampled, 254 selected, 203 rejected
  - `blur=55`, `dup=4`, `fps=4`, `overlap=10`:
    428 sampled, 250 selected, 178 rejected
  - `blur=55`, `dup=3`, `fps=5`, `overlap=10`:
    497 sampled, 304 selected, 193 rejected
  - `blur=55`, `dup=3`, `fps=5`, `overlap=6`:
    457 sampled, 291 selected, 166 rejected
- The sweep showed:
  - raising blur threshold from `40` to `55` increased blur rejections from
    `82` to `104` on the same sample
  - lowering duplicate threshold from `4` to `3` reduced duplicate rejections
    from `129` to `89`, which confirms it is a looser setting rather than a
    stricter one
  - `min_segment_sec=3` had no effect on this sample versus `2`
  - `overexposed_ratio=0.5` and `underexposed_ratio=0.5` still rejected zero
    frames on this sample
- The pressure sweep temporarily exhausted free disk space while retaining
  rejected PNGs for every run. Experimental outputs were partially cleaned to
  recover workspace capacity after capturing the metrics above.
- Re-ran a narrower keep-output comparison for manual review on
  `pressure_test.mp4` and retained these result sets:
  - `pressure_keep_baseline_blur40_dup4_fps5`
  - `pressure_keep_blur55_dup4_fps5`
  - `pressure_keep_blur60_dup4_fps5`
  - `pressure_keep_blur55_dup4_fps4`
- Computed selected-frame blur statistics from the retained manifests. For this
  sample, the highest average selected blur score came from
  `pressure_keep_blur60_dup4_fps5`, followed closely by
  `pressure_keep_blur55_dup4_fps5`.
- Generated a visual review video from the selected frames of
  `pressure_keep_blur60_dup4_fps5` at:
  `data/frames/pressure_keep_blur60_dup4_fps5/selected/selected_review.mp4`
- Ran a second retained real-video matrix on the supplied 39.1-second iOS HEVC
  sample. Output identifiers use `ios_test_*` because CLI video IDs cannot
  contain spaces. All runs used source-frame PNG extraction, time fallback
  segmentation, duplicate threshold `4`, and did not save rejected images to
  limit disk usage:
  - `ios_test_blur55_fps5_dup4`, `blur=55`, `fps=5`: 118/246 selected
    (47.97%), selected-frame mean blur score 421.16.
  - `ios_test_blur60_fps5_dup4` and `ios_test_blur65_fps5_dup4`: identical
    result to `blur=55`; the threshold did not reject any additional frame.
  - `ios_test_blur100_fps5_dup4`: 116/246 selected (47.15%), mean 426.89.
  - `ios_test_blur150_fps5_dup4`: 113/246 selected (45.93%), mean 434.97.
  - `ios_test_blur200_fps5_dup4`: 109/246 selected (44.31%), mean 445.40;
    this is the sharpest retained practical candidate for this sample.
  - `ios_test_blur60_fps6_dup4`: 122/295 selected (41.36%), mean 403.94;
    raising sampling density added only four selected frames and reduced mean
    selected-frame sharpness.
- Generated the iOS quality-first visual review video at:
  `data/frames/ios_test_blur200_fps5_dup4/selected/selected_review.mp4`.
- Ran a retained tuning matrix on `data/raw_videos/restricted_test.mp4`, a
  62.167-second 960x720 controlled sample. All runs used source-frame PNG
  extraction, `max_long_edge=1600`, `duplicate_hash_threshold=4` unless
  explicitly noted, and `save_rejected=false` to limit output volume:
  - Locked baseline `blur=55`, `fps=5`, `overlap=10`: 411 sampled, 115
    selected (27.98%), selected mean blur 91.14, selected PNGs 47.0 MiB.
  - `blur=60`, `fps=5`, `overlap=10`: 105 selected (25.55%), mean 94.38;
    this is the current balanced quality candidate.
  - `blur=65`: 89 selected (21.65%), mean 100.03.
  - `blur=70`: 81 selected (19.71%), mean 103.29.
  - `blur=80`: 69 selected (16.79%), mean 108.99; quality-first extreme.
  - `blur=60`, `duplicate_hash_threshold=3`: 107 selected, mean 93.86;
    loosening duplicate rejection added only two frames and slightly reduced
    mean quality, so it is not recommended.
  - `blur=60`, `fps=4`: 81/310 selected (26.13%), mean 92.62; lower sampling
    did not improve quality.
  - `blur=60`, `fps=6`: 110/493 selected (22.31%), mean 95.75; it added only
    five selected frames over fps 5 while increasing candidate volume.
  - `blur=60`, `fps=5`, `overlap=6`: 93/371 selected (25.07%), mean 92.04;
    it reduces segment output from 42.3 MiB to 39.8 MiB but slightly lowers
    selected-frame quality.
- The restricted-video matrix supports keeping `duplicate_hash_threshold=4`,
  `target_fps=5`, and `segment_overlap_sec=10`. The current recommendation is
  `blur_threshold=60` for balanced reconstruction coverage, with `70` or `80`
  reserved for a quality-first run after visual review.
- Generated review videos for the two decision points:
  - `data/frames/restricted_test_blur60_d4_fps5_r1600/selected/selected_review.mp4`
  - `data/frames/restricted_test_blur80_d4_fps5_r1600/selected/selected_review.mp4`

## Next Step

Update the PR body to reflect the current HEAD `373ea4e`, complete the Ready
for review checklist, and request reviewer approval before switching the PR
out of Draft.

For `restricted_test`, review the `blur=60` and `blur=80` outputs before
deciding whether the baseline should move from `55` to `60`, or whether the
quality-first setting should remain an explicit per-video override.

- candidate baseline update for the next pass:
  - `blur_threshold: 55.0`
  - keep `duplicate_hash_threshold: 4`
  - optionally set `target_fps: 4.0` if storage and review load need to come
    down
  - keep exposure thresholds at `0.6`
- iOS quality-first candidate, kept separate pending manual visual acceptance:
  - `blur_threshold: 200.0`
  - `target_fps: 5.0`
  - `duplicate_hash_threshold: 4`
- confirmation command:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/<video>.mp4 --video-id <id> --config configs/preprocess/baseline_real_video.json --force
```

- follow-up comparison:

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
- The first tuning sweep supports increasing `blur_threshold` into the mid-50s
  for this footage, but it also confirms that tightening blur alone reduces
  retained coverage. The next configuration update should avoid simultaneously
  loosening duplicate rejection unless that tradeoff is explicitly desired.
- Reducing overlap or target FPS lowers generated output size and review volume,
  but it also reduces total candidate frames. That tradeoff should be accepted
  only if the remaining selected frames are still sufficient for downstream
  reconstruction.
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
