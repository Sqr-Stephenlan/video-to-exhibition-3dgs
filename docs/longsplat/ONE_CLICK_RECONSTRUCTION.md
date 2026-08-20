# One-click LongSplat reconstruction

From the long-lived production checkout, the supported user command is:

```bash
cd <production-checkout>
./video-to-3dgs "/path/to/video.mp4"
```

On success the command prints a public PLY path and an evidence path. The
public file is written below `outputs/` by default, for example:

```text
outputs/video__<ply-sha12>.ply
```

The full append-only run remains below
`<workspace>/.runtime/longsplat-runs/<run-id>/` (the workspace containing the
production checkout and backend environments). It contains the stage evidence,
checkpoints, manifests, logs, `published_ply.json`, and the authoritative
`run.json`. The public PLY is an ordinary independent file, never a symlink or
hardlink into the evidence tree.

## Setup

The route environment is intentionally separate from the CUDA backend:

- `./dev.sh bootstrap` creates the checkout-local `.venv` and installs only
  CPU orchestration, media-contract, PLY, and test dependencies.
- `workspace/backend-envs/longsplat-cu128` is the independent backend Python
  environment used by LongSplat GPU stages.
- `workspace/backend-envs/media-tools` supplies the preferred `ffmpeg` and
  `ffprobe`; `colmap` is discovered from `PATH` unless overridden.

This is an intentional two-environment design, not dependency duplication.
The product script always uses this checkout's `.venv`; it does not require a
`VENV_DIR` pointing at another worktree.

After setup, the non-GPU checks are:

```bash
./dev.sh doctor
./video-to-3dgs --help
```

If the checkout is not below a workspace carrying the backend directories,
copy `configs/provider.local.example.json` to the ignored
`configs/provider.local.json` and replace the placeholders with local paths.
In that standalone layout the runtime falls back to the checkout's
`.runtime/`; the same provider file can be selected explicitly with
`--provider-config`.

## Inputs and options

The positional input is either a local regular video file or a direct
`http://`/`https://` video-file URL. Redirects are followed. URL downloads use
`.runtime/longsplat-incoming/*.part`, are fsynced and hashed, then are checked
with `ffprobe` before the canonical pipeline starts. Only a sanitized URL
origin is recorded; credentials, query strings, and fragments are never put
in logs or manifests.

A webpage URL is not silently parsed. It fails with a message that a webpage
needs an optional `yt-dlp` adapter; that adapter is not part of the core
dependency closure.

Useful options:

```bash
./video-to-3dgs --help
./video-to-3dgs --name east-gallery "/path/to/video.mp4"
./video-to-3dgs --output-dir outputs/review "/path/to/video.mp4"
./video-to-3dgs --plan "/path/to/video.mp4"
```

`--plan` is a CPU-only contract check. It does not run `nvidia-smi`, CUDA,
training, rendering, conversion, or evaluation.

## Canonical chain and safety

The product entry point calls the existing canonical orchestrator once:

`single-convergence1000-v1 → automated gates → formal30000 → authority →
standard30000 conversion → converted-eval/postprocess → automated technical
delivery → publish`.

The default route keeps depth disabled and does not enable the legacy
smoke100/coverage route, manual visual gates, VDA, pose search, MASt3R, or
DUSt3R optional paths. The final GPU preflight is fail-fast and does not alter
the driver, CUDA installation, or either environment.

Published names are sanitized and include the final PLY SHA-256 prefix. A
same-name, same-SHA file is safely reused; any different-content collision
stops without overwriting the existing delivery. The receipt records source
technical PLY path, SHA/size/vertex count, published path, SHA/size/vertex
count, and the atomic-copy policy.

The CPU product contract and focused tests are verified. The final real GPU
execution remains a manual user step; no GPU E2E is implied by the CPU checks.
