# CORE_REQUIRED remediation memo

This memo defines the safety boundary for the default raw-video → external
fixed-pose → standard 3DGS PLY chain. It is governance/provenance text only;
it is not a production input and must never be included in a production stage
identity.

## Camera and matcher policy — CORE_REQUIRED

Status this round: CPU contract and local clean-head CPU replay verified; real
COLMAP replay remains pending. `SIMPLE_RADIAL` + `sequential` remains the versioned default. The
existing COLMAP contract also admits `PINHOLE` and `SIMPLE_PINHOLE`, plus
`exhaustive`; the raw and reconstruct entry points now record and forward the
selected model and matcher. Unsupported values stop with a structured
`PipelineBlocked` error and never trigger matcher search.

Completion requires all of the following:

- every accepted model has a complete COLMAP command, camera-prior, parser,
  undistortion, centered-PINHOLE staging, and LongSplat input contract;
- camera order, per-video K/distortion, crop, and frame names remain dynamic
  evidence rather than global defaults;
- CPU tests cover at least two models, two matchers, and an unsupported stop;
- a clean-head replay proves the selected configuration is present in config,
  manifest, and run identity.

## Tool/backend provider layout — CORE_REQUIRED

Status this round: explicit provider paths, CPU identity fixtures, and local
clean-head CPU replay verified; real provider preflight remains pending. The known layout remains
the default, while `backend_env`/`backend_python`, COLMAP, ffmpeg, ffprobe, and
route-python may be supplied explicitly. Provider records preserve requested
path, resolved target, symlink state, executable state, file SHA/size, and the
versioned preflight identity. Commands remain argv lists with `shell=False`.

Completion requires ordinary-file/executable/symlink behavior to be
fail-closed and documented, two independent external tool roots to work, and
missing or changed provider identity to produce a structured blocked resume.
No provider is required to live below the output root.

## Stage-scoped resume identity — CORE_REQUIRED

Status this round: producer-scoped reuse and immutable drift checks verified;
old-schema migration is explicit/blocking rather than silently promoted, so
the item remains partial. Broad source-tree identity is retained as
provenance, but stage reuse binds production code/provider/schema/profile and
does not churn for documentation, tests, legacy, or optional-module changes.

Completion requires exact reuse for unchanged records, an explicit
`stale_rebuildable` decision for append-only schema/consumer metadata, and an
`unsafe_drift` hard stop for source video SHA/path/size, camera order/K/pose,
selected segment, checkpoint/MLP, converter/evaluator/backend, or provider
identity changes. No old record may be silently reused merely because its
aggregate hash is convenient.

## Non-relaxable safety conditions

- `depth_source=disabled` and external fixed poses remain explicit; no hidden
  VDA, MASt3R, DUSt3R, depth, quality, or geometry fallback is permitted.
- CPU import must not invoke GPU discovery, shell commands, network listeners,
  training, rendering, conversion, or evaluation.
- Partial snapshots are removed on copy failure; immutable records are never
  overwritten; output-root containment and symlink checks remain fail-closed.
- This memo and other docs/tests are never part of the production stage
  identity, even though broad audit provenance may mention them.
