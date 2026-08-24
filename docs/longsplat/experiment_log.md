# LongSplat Experiment Log

> Template for recording individual training runs.
> Fill one section per run; commit after each meaningful result.

---

## Run: `<run_id>`

- **Date:** YYYY-MM-DD
- **Branch:** `research/longsplat-route`
- **Config:** `configs/longsplat/<config>.json`
- **Segment:** `<video>/<segment_id>`
- **Frames:** N

### Input

| Field | Value |
|---|---|
| Manifest | `data/manifests/<video>/preprocess_manifest.json` |
| Depth manifest | `data/manifests/<video>/depth_manifest.json` (VDA only) |
| Depth source | `mast3r` / `vda` |

### Backend identity

| Field | Value |
|---|---|
| LongSplat commit | `19750775a9d19f30aa05a8333c4c6c231b2d5f4a` |
| Backend mode | `research_local` |
| diff_sha256 | `<sha256>` |
| submodule_diffs | `<sub_path>: <sha256 or null>` |
| Patches applied | stability_fixes, training_improvements, vda_depth_injection |

### Results

| Stage | Exit code | Duration (s) | Notes |
|---|---|---|---|
| Input preparation | | | |
| Training | | | |
| VDA usage (aligned/missing/rejected) | — | — | |
| Conversion | | | |
| PLY validation | | | |

### PLY

| Metric | Value |
|---|---|
| Vertices | |
| Size (bytes) | |
| SHA-256 | |
| NaN/Inf count | |

### Observations

<!-- Free-form notes: visual quality, artifacts, convergence behavior, unusual warnings -->

---

## Run: `<run_id>`

- **Date:** YYYY-MM-DD
- **Branch:** `research/longsplat-route`
- **Config:** `configs/longsplat/<config>.json`
- **Segment:** `<video>/<segment_id>`
- **Frames:** N

### Input

| Field | Value |
|---|---|
| Manifest | |
| Depth manifest | |
| Depth source | |

### Backend identity

| Field | Value |
|---|---|
| LongSplat commit | |
| Backend mode | |
| diff_sha256 | |
| submodule_diffs | |
| Patches applied | |

### Results

| Stage | Exit code | Duration (s) | Notes |
|---|---|---|---|
| Input preparation | | | |
| Training | | | |
| VDA usage | | | |
| Conversion | | | |
| PLY validation | | | |

### PLY

| Metric | Value |
|---|---|
| Vertices | |
| Size (bytes) | |
| SHA-256 | |

### Observations

---

## 2026-07-22: finite conversion + pose/VDA telemetry short A/B

- **Branch:** `research/longsplat-route`
- **Manifest:** `outputs/task123_pipeline_inputs/preprocess_4f.json`
- **Manifest SHA-256:** `17767ffd15d7dce56551b78ca9e8d631b244f86528e0720e88974a72ffd36397`
- **Frames:** 4 (`20.0`, `20.2`, `20.3`, `20.5` seconds)
- **Depth manifest SHA-256:** `499b475fb9095e53c9bd4c8564c92035b46a8f00d95979afb54bc8e6d25fc41c`
- **LongSplat commit:** `19750775a9d19f30aa05a8333c4c6c231b2d5f4a`
- **Backend diff SHA-256:** `4a2b81c9bd15cc06a66ec08013f59cf17b11b9178bb1fc1bb5d1640e94b89287`
- **Patches:** stability, training improvements, VDA injection, finite conversion/telemetry

### New-training outcome

| Run | Resolution | Status | Training (s) | Exit | Last completed work |
|---|---:|---|---:|---:|---|
| `baseline_4f` | 384×512 | failed | 130.2 | `3221225477` | 3-frame MASt3R global alignment + initial optimization |
| `baseline_4f_retry1` | 384×512 | failed | 128.2 | `3221225477` | same location on independent retry |
| `baseline_4f_lowmem` | 192×256 | failed | 111.0 | `3221225477` | same location with `resolution=2` |

All three terminal records are under
`outputs/wall_test/finite_pose_ab_20260722/`. Windows exit code
`3221225477` is `0xC0000005` (access violation). The process stopped inside
the fourth-frame incremental MASt3R forward pass, before PnP returned, so no
new `POSE_TELEMETRY` record existed to capture. Reducing camera resolution did
not change the failure. The VDA training arm was not launched after the third
deterministic failure, avoiding another equivalent GPU run.

The two low-memory configs are `configs/longsplat/smoke_lowmem.json` and
`configs/longsplat/smoke_vda_lowmem.json`; they differ from the original smoke
configs only in `resolution=2`.

### Isolated conversion replay

To test the conversion root fix without overwriting historical artifacts, the
current converter was replayed on isolated copies of the most recent completed
same-manifest checkpoints:

- baseline source: `outputs/task123_latest_agent_selftest/latestfv_baseline_4f_20260722`
  (anchor PLY SHA-256 `af061e408d9ee3cfcd937784cc1d8230cb03b6289d8024accdcc52a3cca28ccb`)
- VDA source: `outputs/task123_latest_agent_selftest/latestfv_vda_4f_20260722`
  (anchor PLY SHA-256 `748ac609ba581befb2db317eac07d195dff0517720e0fe3bfc48a91628ff809d`;
  historical VDA usage `aligned=4, missing=0, rejected=0`)

| Metric | Baseline replay | VDA replay |
|---|---:|---:|
| Converted vertices | 52,868 | 52,696 |
| Nonpositive NN distance | 4,285 (8.105%) | 3,814 (7.238%) |
| Minimum distance before clamp | 0 | 0 |
| Minimum distance after clamp | `1.0e-7` | `1.0e-7` |
| Finite scale rows | 52,868 / 52,868 | 52,696 / 52,696 |
| Strict raw PLY validation | passed | passed |
| PLY SHA-256 | `3e6a6f8192c1d7ed88d9d9fd2e0a09516580717de645e7deb41005890c768deb` | `ffbef91d14d4f46408374ef760f8122123b597eee44bbe9cb46fe263ec48e266` |

Replay outputs:

- `outputs/wall_test/finite_pose_ab_20260722/conversion_replay_baseline/longsplat_model/converted_3dgs/point_cloud.ply`
- `outputs/wall_test/finite_pose_ab_20260722/conversion_replay_vda/longsplat_model/converted_3dgs/point_cloud.ply`

No vertex cleanup was run. The result directly validates the duplicate-point
distance floor and finite-scale export, but it is **not** a fresh training A/B
and cannot establish a pose/VDA quality difference. Repeat the controlled
training arms after the local MASt3R/CUDA access violation is resolved; the
new run-record parser now preserves partial telemetry even on non-zero exits.

---

## 2026-07-22: MASt3R/CUDA native memory remediation and verified A/B

- **Branch:** `research/longsplat-route`
- **Manifest:** `outputs/task123_pipeline_inputs/preprocess_4f.json`
- **Manifest SHA-256:** `17767ffd15d7dce56551b78ca9e8d631b244f86528e0720e88974a72ffd36397`
- **Depth manifest:** `outputs/task123_pipeline_inputs/depth_4f.json`
- **Depth manifest SHA-256:** `499b475fb9095e53c9bd4c8564c92035b46a8f00d95979afb54bc8e6d25fc41c`
- **Frames:** 4
- **LongSplat commit:** `19750775a9d19f30aa05a8333c4c6c231b2d5f4a`
- **Backend diff SHA-256:** `4a2b81c9bd15cc06a66ec08013f59cf17b11b9178bb1fc1bb5d1640e94b89287`

### Native-process diagnosis

| Check | Result |
|---|---|
| Runtime | Python 3.13.5, PyTorch `2.11.0+cu128`, CUDA build/driver `12.8` / `573.22` |
| GPU | RTX 5060 Laptop GPU, compute capability 12.0; PyTorch arch list includes `sm_120` |
| CUDA extensions | `diff_gaussian_rasterization`, `simple_knn`, and a real CUDA `torch_scatter.scatter_max` smoke test passed |
| Isolated MASt3R | Three sequential forwards passed; `global_align -> forward` also passed with launch blocking |
| Synchronized full failure | `fatal: Memory allocation failure`, followed by `torch.AcceleratorError: CUDA error: unknown error` in `mast3r/cloud_opt/sparse_ga.py` |
| Host state at failure | About 0.6-0.7 GB physical memory available and about 42/52 GB committed |
| Leaking process | `SenaryAudioApp.Svc`: about 9.9-10.09 GB Private Bytes and 3.79-3.85 million handles, still increasing |
| Later healthy service state | New PID 5704, 24,784,896 Private Bytes, 353 handles; the attempted non-admin restart was denied and the UAC launch was cancelled, so the exact external restart trigger is not attributed here |
| Host state after final A/B | 6,529,576,960 bytes available; 19,929,309,184 / 42,108,530,688 bytes committed |

The repeated Windows `0xC0000005` exits were therefore a secondary symptom of
host memory/resource exhaustion. Bad input frames, unsupported GPU
architecture, a PyTorch/CUDA ABI mismatch, and a basic MASt3R forward defect
were excluded by the isolated probes. Lowering the image resolution alone had
already failed at the same fourth-frame location.

### Low-peak loader mitigation

`third_party/LongSplat/submodules/mast3r/mast3r/model.py` now loads the local
2.9 GB checkpoint with `mmap=True`, releases `ckpt`, and runs `gc.collect()`
before `net.to(device)`. The reproducible nested-repository patch is
`docs/longsplat/patches/mast3r_low_memory_load.patch`. It does not change model
weights, precision, inference, or LongSplat optimizer settings.

The structured marker parser was also corrected to accept only LongSplat's
known `safe_state` timestamp suffix (`[DD/MM HH:MM:SS]`) while continuing to
reject arbitrary trailing text. The final runs below therefore contain the
authoritative pose and VDA summaries directly in their run records.

### Fresh native-CUDA short A/B

Both arms used `resolution=2`, 100 training iterations, seed 0, unique run
directories, and no `CUDA_LAUNCH_BLOCKING`.

| Metric | Baseline | VDA |
|---|---:|---:|
| Status | `complete` | `complete` |
| Training exit / duration | 0 / 91.0 s | 0 / 87.8 s |
| Conversion exit / duration | 0 / 10.4 s | 0 / 11.5 s |
| PnP successes / attempts | 1 / 1 | 1 / 1 |
| Minimum PnP inlier ratio | 0.9669967 | 0.9867987 |
| Maximum reprojection RMSE | 1.2013 px | 1.4247 px |
| VDA aligned / missing / rejected | - | 4 / 0 / 0 |
| Minimum VDA correlation | - | 0.9239189 |
| Mean VDA inlier ratio | - | 0.9890340 |
| Converted vertices | 52,408 | 52,244 |
| Nonpositive NN distances floored | 18,143 | 21,363 |
| All scales finite | yes | yes |
| PLY SHA-256 | `3a89308f75d8fc3982f84cffccacb207a1c6ecd24d0929ead10c2e222641974c` | `73bfead1b27095991c05ae124d2cae847bff59f0b987fb5ad5e0c893056dc859` |

Authoritative outputs:

- baseline run record: `outputs/wall_test/mast3r_native_fix_20260722/baseline_4f_lowpeak_tsfixed/reconstruction_run.json`
- baseline PLY: `outputs/wall_test/mast3r_native_fix_20260722/baseline_4f_lowpeak_tsfixed/longsplat_model/converted_3dgs/point_cloud.ply`
- VDA run record: `outputs/wall_test/mast3r_native_fix_20260722/vda_4f_lowpeak_tsfixed/reconstruction_run.json`
- VDA PLY: `outputs/wall_test/mast3r_native_fix_20260722/vda_4f_lowpeak_tsfixed/longsplat_model/converted_3dgs/point_cloud.ply`

This verifies that the local native MASt3R/CUDA path works once host resources
are healthy, and that the lower-peak checkpoint loader remains compatible with
fresh baseline and VDA training. The loader mitigation reduces transient
pressure but cannot cure an unrelated process that continuously leaks memory
and handles; service health remains an operational preflight requirement.
