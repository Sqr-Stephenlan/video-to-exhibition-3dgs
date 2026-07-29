# Desk/Table Parameter Matrix - 2026-07-29

## Scope

This record captures the preprocessor-only parameter matrix for
`data/raw_videos/desk.mp4`. The file SHA-256 is
`e57b8388ae82fc716f45d77f7d631b0782764537b2f82461d2984e1322ddd79f`, which
matches the expected source hash for the table-parallax experiment.

All runs used independent `video_id` values. Generated video, frames,
manifests, and reports remain under ignored `data/` paths and are not part of
this document's Git change.

## Controls

The first matrix fixes source-frame PNG extraction at 5 FPS, 960-pixel long
edge, 30-second windows with 10-second overlap, duplicate threshold 4, and no
saved rejected frames. It varies only the policy and absolute blur threshold.

| Policy | Blur | Selected / sampled | Audit | Maximum selected gap |
|---|---:|---:|---|---:|
| legacy | 40 | 401 / 507 | passed | 1.2 s |
| coverage_v1 | 40 | 90 / 507 | failed | 29.8 s |
| legacy | 55 | 401 / 507 | passed | 1.2 s |
| coverage_v1 | 55 | 90 / 507 | failed | 29.8 s |
| legacy | 70 | 394 / 507 | passed | 1.2 s |
| coverage_v1 | 70 | 90 / 507 | failed | 29.8 s |

The coverage results are identical across the blur sweep. The bottleneck is
not the absolute blur floor: selected coverage frames already pass all three
blur settings.

## LongSplat Cadence Matrix

The second matrix fixes the `coverage_v1` geometry policy and a 512-pixel long
edge, varying only sampling cadence. Its 10 FPS row is the formal candidate
configuration from the integration plan.

| Video ID | FPS | Selected / sampled | Audit | Maximum selected gap |
|---|---:|---:|---|---:|
| desk_matrix_longsplat_fps5 | 5.0 | 91 / 507 | failed | 30.0 s |
| desk_matrix_longsplat_fps7p5 | 7.5 | 93 / 760 | failed | 30.0 s |
| table_parallax_candidate_20260729 | 10.0 | 88 / 1013 | failed | 30.0 s |

For the formal 10 FPS candidate, `scene,time` produced four `time_fallback`
windows: 0.0-30.0 s, 20.0-50.0 s, 40.0-70.0 s, and 60.0-71.3 s. Its selected
frame counts were 48, 1, 33, and 6. The corresponding maximum gaps were 20.6,
30.0, 24.3, and 10.2 seconds.

The candidate rejected 684 frames for insufficient bidirectional tracks and
151 for insufficient tracked points. It also rejected 152 blur, 77 duplicate,
34 cadence, 3 grid-coverage, and 3 motion-inlier-ratio cases. No bridge frames
were selected in this candidate.

## Decision

The formal candidate is rejected. It fails the `coverage_v1` quality status and
the required 0.3-second temporal-coverage gate in every segment. It must not be
adapted for VDA, materialized for LongSplat, or used in a depth-source A/B.

Do not lower optical-flow, inlier, grid, or coverage thresholds, and do not
shorten windows merely to make boundary gaps pass. The next producer action is
to inspect the capture for tracking-hostile motion or cuts and reshoot or
replace the source when necessary. Any proposed selector change requires a
separate safety review with the existing negative controls.

## Reproduction

The formal rejected candidate was produced with:

```bash
./dev.sh python scripts/preprocess_video.py data/raw_videos/desk.mp4 \
  --video-id table_parallax_candidate_20260729 \
  --config configs/preprocess/coverage_v1.json \
  --preset longsplat \
  --target-fps 10 \
  --max-long-edge 512
```

Its audit command returns exit code 2, as expected for this rejected result:

```bash
./dev.sh python scripts/preprocess_video.py \
  --audit-manifest data/manifests/table_parallax_candidate_20260729/frames_manifest.json
```
