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

The download guard defaults to 8 GiB and checks both `Content-Length` and the
actual streamed byte count. It can be lowered or raised (up to the built-in
64 GiB ceiling) with `--max-download-bytes N` or
`LONGSPLAT_MAX_DOWNLOAD_BYTES=N`. A redirect whose final scheme is not
`http`/`https`, an over-limit response, or a byte-count mismatch is rejected
and its `.part` file is removed.

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

### Observational terminal progress

The one-click command accepts `--progress auto|plain|off` (default `auto`).
The reporter writes to stderr only, so the canonical stdout contract remains
unchanged: successful runs still emit `SUCCESS`, `PLY:`, and `EVIDENCE:` in
that order, while plan runs still emit `PLAN` and `EVIDENCE:`. `auto` uses a
single refreshed line on a capable TTY; `plain` and CI/non-TTY runs append only
stage changes plus an approximately 60-second heartbeat. `TERM=dumb`,
`NO_COLOR`, or `off` never adds ANSI control sequences. Direct URL downloads
report byte counts and `Content-Length` when available; without a length they
report MiB only and never print the URL, query, fragment, or token.

The UI is a read-only observer, not a second state machine. It reads the
current run's `config.json` stage order, `run.json`, the two raw-stage
`run.json` files, and the current conversion sidecar. The authoritative state
remains `run.json`/RunLedger and the existing return-code, PLY-validator, and
stage contracts. A missing conversion marker is shown as `conversion
(unobserved)`; no percentage, iteration zero, total, ETA, or pass/fail state is
invented. After 15 minutes without a new observed marker the UI may print
`WARNING: no observed iteration heartbeat`; it never terminates or blocks the
child automatically.

For each conversion attempt, the executor creates three fresh files below that
attempt's evidence directory:

```text
conversion-stdout-live.log       complete raw child stdout
conversion-stderr-live.log       complete raw child stderr (separate stream)
conversion-progress-v1.jsonl     append-only structured observations
```

The sidecar uses `schema_version: "conversion-progress-v1"`, `total`,
`timestamp_utc`, and `elapsed_seconds`. It records `child_started`, valid
strictly increasing integer `iteration` observations, and `child_exited` (or a
parent-error record during cleanup). Iterations come only from the child's
explicit `CONVERSION_SHAPE_TELEMETRY` marker; duplicate, regressing,
malformed, or out-of-range markers remain in the raw log but are not written as
normal progress. The reader consumes only a bounded tail of the append-only
JSONL and ignores a final partial line. The files are exclusive, non-symlink
observation artifacts contained within the current attempt; legacy final
`stdout.log`/`stderr.log`, conversion records, telemetry summaries, validators,
and resume semantics remain authoritative and unchanged.

For a fresh remote checkout, the CPU workflow materializes the root LongSplat
fork and exactly the four direct locked inner gitlinks with:

```bash
./dev.sh python -m scripts.longsplat.verify_remote_distribution --root . --json
```

This command uses the GitHub remotes recorded in `.gitmodules`, verifies the
locked SHAs and origin URLs, and does not recursively initialize optional
MASt3R/DUSt3R descendants.

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
count, nlink, and the atomic-copy policy. Publication uses a same-directory
temporary file plus Linux `renameat2(RENAME_NOREPLACE)`, fsyncs the file and
directory, and blocks if the strict no-replace primitive is unavailable. A
resume rechecks the canonical public path, receipt, SHA/size/vertices, and
`st_nlink == 1` before reusing a completed delivery.

The CPU product contract and focused tests are verified. The final real GPU
execution remains a manual user step; no GPU E2E is implied by the CPU checks.
