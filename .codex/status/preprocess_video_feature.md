# Preprocess Video Feature Status

Last updated: 2026-07-10

## Current Phase

Stage 0 - Project prompt and tracking setup.

The preprocess video feature has not been implemented yet. The project now has a
Codex-readable prompt that defines scope, output layout, CLI contract,
dependencies, tests, and out-of-scope boundaries.

## Completed

- Read the overall exhibition 3DGS framework and architecture review documents.
- Scoped the user's responsibility to `preprocess video` only.
- Confirmed this repository is currently a skeleton with directories for
  `data/`, `scripts/`, `docs/`, `outputs/`, `third_party/`, and `annotations/`.
- Added the feature prompt:
  `.codex/prompts/preprocess_video_feature.md`
- Added `AGENTS.md` pointers so new Codex conversations know to read the prompt
  and this status file.
- Updated `.gitignore` so tracked Codex prompts/status can coexist with ignored
  local Codex scratch directories.

## Next Step

Stage 1 - Implement the minimal preprocess MVP.

Recommended first implementation slice:

1. Add `requirements.txt` with only the preprocess dependencies.
2. Add `scripts/preprocess_video.py`.
3. Implement `ffmpeg`/`ffprobe` availability checks.
4. Implement video metadata probing.
5. Implement time-window segmentation first.
6. Implement OpenCV frame sampling and basic quality metrics.
7. Write `preprocess_manifest.json`.
8. Add focused tests for segment window math, image quality metrics, and manifest
   generation.
9. Add `docs/preprocess_video.md`.

PySceneDetect scene segmentation can be added after the time-window path works.

## Open Decisions

- Whether to bootstrap `.venv`; current instructions require asking first if it
  is missing.
- Whether to install dependencies immediately or wait until implementation
  starts.
- Whether the MVP should include PySceneDetect in the first pass or land it in a
  second pass after time-window segmentation.

## Current Non-Goals

- Do not clone LongSplat, MASt3R, DUSt3R, Depth Anything, gsplat, SuperSplat, or
  other large third-party repositories for the preprocess MVP.
- Do not implement reconstruction, depth estimation, model compression, web
  display, or annotation UI in this feature.

## Status Update Rules

When work continues, update this file before finishing the conversation if any
of the following changed:

- Current phase.
- Completed implementation or tests.
- Next recommended step.
- New blocker, risk, or decision.
- Dependency or file-layout convention.
