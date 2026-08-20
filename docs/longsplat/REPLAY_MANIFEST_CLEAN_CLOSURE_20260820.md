# Clean-head closure replay manifest

This manifest records the reviewable local closure for the default
depth-disabled external fixed-pose route. It is rooted at PR #3 clean head
0577d497588e0664cbf686457465cd0bcc04af53; the historical c17415a line and
its VDA/depth/quality history are not part of this closure. No /tmp path is
required to replay the committed result.

## Authority and commits

| item | value |
|---|---|
| root branch | codex/pr3-clean-closure-final-20260820 |
| root base | 0577d497588e0664cbf686457465cd0bcc04af53 |
| root base parent | 97a5ba8a0a53e94b6e18d05267ad420705542969 |
| nested base | faa6e3a76f479f7c9b5f764538c0c507c1f03b39 |
| nested base tree | 821073e6300237fa40334e9daf73f1a82f1ce383 |
| nested final commit | bf766eb903c3d9144d64088b8c80b2da67d39411 |
| nested final parent | faa6e3a76f479f7c9b5f764538c0c507c1f03b39 |
| nested final tree | 79f232c42e9128092df0520c5b0209d7ac2b08fd |
| root gitlink | third_party/LongSplat -> bf766eb903c3d9144d64088b8c80b2da67d39411 |

The final root commit SHA is reported with the handoff because a commit cannot
contain its own SHA. Its first closure parent is the root base above.

## Versioned replay inputs

| path | SHA-256 |
|---|---|
| docs/longsplat/REPRODUCTION_RUNBOOK.md | 9e0077cec9fb66931462ea5e093b41ce8040e597f5245abe5a040763b6c3b493 |
| docs/longsplat/patches/0001-longsplat-external-colmap-route.patch | 0f49e2b588167939005b2e9f33ef0087a9e3bef2bbc58ec4362c2012bf75bfdc |
| docs/longsplat/patches/0002-longsplat-active-pose-contract.patch | 602aff3ed5fce867756381cc20ca580ef4a9a18b8b880c53aeed84e662535897 |
| docs/longsplat/patches/0003-longsplat-clean-faa6-external-fixed-pose-minimal.patch | 01e897337a6abd93523344c52d8abfd64d4d51e4a1e430064ce01a56d22978a2 |
| docs/longsplat/CORE_GOAL_MEMO.md | 88f47d6d4436342bb3146a5a4c31a72f88eb598f1da77c9634fae3ff2a1783ad |
| docs/longsplat/HANDOFF_CURRENT_STATUS_20260820.md | 26059ed8c75d63cd8d514cd0c5a590e872bd0216f9ace482fcd8528f8bfc3d8b |
| .gitmodules | bf216b7946ccfdecef47e21a143d0aa07d35c0701ac9ac606b0c9b0167853123 |

The 0003 patch was regenerated from nested faa6e3a to final commit bf766eb.
It includes the final import-boundary files and no ephemeral artifact path;
because it intentionally has zero context, replay uses git apply --unidiff-zero.
The superseded nested patch SHA
52f1b93d45d5a923afda476b29f48567051000547dab431b0eafa369e30e057e is not
used. The reviewed updated remediation SHA is
f4678269ab6ad01151b1461abe4251bb5e7a961a6825a89db7d8f23697b7913e.

## Nested distribution identity

The root gitlink is distributed from the user-owned fork of the upstream
repository:

- upstream parent: NVlabs/LongSplat
- fork: https://github.com/Sqr-Stephenlan/LongSplat
- fork branch: research/external-fixed-pose-clean-closure
- published nested commit: bf766eb903c3d9144d64088b8c80b2da67d39411
- published nested tree: 79f232c42e9128092df0520c5b0209d7ac2b08fd
- .gitmodules URL: https://github.com/Sqr-Stephenlan/LongSplat.git

The nested fork branch must be published before the root branch. The root
runner lock and gitlink must remain identical to the published nested commit.

## Nested closure

The final nested commit contains exactly these eleven files:

- convert_3dgs.py
- gaussian_renderer/network_gui.py
- render.py
- scene/__init__.py
- scene/dataset_readers.py
- scene/gaussian_model.py
- tests/test_external_colmap_pose.py
- train.py
- utils/camera_sampling_telemetry.py
- utils/colmap_utils.py
- utils/external_colmap_pose.py

Final file SHA-256 values:

- convert_3dgs.py: b7513bcd2c08c0f82c3edb68ae2581edc2b18b8666047c975f9d70bf45325156
- gaussian_renderer/network_gui.py: b74495a971e7a0be8fb822e24d1525b461e6351e7ad708c4682f640867f2ac2e
- render.py: 4ac3800e40bee17378dcada56abf2158f404c9f2845f3a7384033d76906a13c0
- scene/__init__.py: f7500b174f42ed3074744db507a79e3f301825026a4cab2dabbc7096ec63b58d
- scene/dataset_readers.py: d39fb41a2320c061565bf485f818c689b77536dbc5d1f0bf1f87dc9846a3b66a
- scene/gaussian_model.py: 37a852ed6c2059845bb8589bcbde66e0199ea1f19e89f6e8bc9b2e6ad48adf72
- tests/test_external_colmap_pose.py: be0af0662bbc511e83541a19ad9464bdecc26576dc72899407554571f4555062
- train.py: 404f5f73e50a65555dc7eb176e340f8723361916db473059360faf4971ad7641
- utils/camera_sampling_telemetry.py: 6e96127e15df4a86c9150c427ffe77d43f180689853b4510fb21b2ec3e76136e
- utils/colmap_utils.py: cc204264bd3a62cfbb34a2d4bf339bcf556492b52c1c23d4a2446d6dbc9c17fd
- utils/external_colmap_pose.py: f4e1d327a0c55a1955c3e6b6cbe5ca50bcca438273ad839e6ab8ecbf854ca6d1

Four retained submodule gitlinks:

- submodules/diff-gaussian-rasterization: 401a405b2360677f3a71ab1930af94f8f2bfc1c3
- submodules/fused-ssim: 085e0f36d9009ebd241e019ea762442dd1aaeca9
- submodules/mast3r: f5209afc300cec36239a7ac992263f36847bbba0
- submodules/simple-knn: 86710c2d4b46680c02301765dd79e465819c8f19

Explicitly excluded are the render matcher cap, pose selection, MASt3R/DUSt3R
geometry, VDA/depth/quality/coverage changes, and dirty MASt3R submodule
content. The telemetry EOF normalization is whitespace-only.

## Root closure and patch provenance

The minimal clean-head overlay SHA is
7d4ec40e298b418511914cf2bab2e0f8d56a3b1034f14c954148e564ccef58b2.
The canonical nested overlay used to construct final 0003 is
75a166412775e3666e8a7c3f812ebd345764b3998ba454e2bbede407b1da9433.

Reviewed remediation records:

- root-dependency-remediation.patch:
  3331da16ca3e4e1711ec248f2c2cc800adec71f052df27cf90505bba5ec3eaab
- root-generalization.patch:
  55d332476d503a40af7450322ed102b133e770908e8090a74c5cbf99c81641cf
- docs-core-goal.patch:
  d61e4d1f5be71f372f9e493a343b90dc8cfab4cc0996e6e2e98f63ac0f673687
- updated nested-import-boundary.patch:
  f4678269ab6ad01151b1461abe4251bb5e7a961a6825a89db7d8f23697b7913e
- fresh four-patch apply-check evidence:
  07fc69fed8805160cdf43ea71db4fd7ce846c6a7a029433b1213ae3fee39854a
- nested import CPU evidence:
  c53a6a6ec2866ed7c4a6d936246eaf784e977d92c412ec01524f6c68edaeac3b

Final root production target SHA-256 values:

- dev.sh: 782ff11cec794abcebd528da7573999fb74af8111a49e71a8501d44815276a23
- scripts/longsplat/runner.py: 6e82650845aa8c006d83792c5bd619446d8a534f6cac85d69e93f4678bfecd1
- scripts/longsplat/reconvert_existing.py: a95c6449e242e430dc50bd35b5beb20c83d2055e7b745dc1ee9ebd3ca566cc43
- scripts/longsplat/backend_identity.py: 38fbb13af54bbfba8392aa9d3cb4620fca87b7649d0cf392374e923508e16ae1
- scripts/longsplat/tool_provider.py: d75adbc8e2fa6ad71e39f2ba94e1b66221f046de6bc64d286a9da05b2d6442fe
- scripts/longsplat/authority_manifest.py: d95383bc30ef1c825213a1c6311128b9acb9f4ef674988d86fae689d875f39dd
- scripts/longsplat/colmap_contract.py: 940187a448cb2d7e4eca4bacea773d7c75a7f1871c66d5af57a465b74e6f2b70
- scripts/longsplat/pipeline_contract.py: 6493b112dd6cc6a52c482bddd6a886a4c9e586f004f83138c5c63f78a820d5ed
- scripts/longsplat/raw_pipeline.py: beb5703a44afd399cb40e65f7949217869917630969bf99aeb45f5e5acc7c2ca
- scripts/longsplat/reconstruct_pipeline.py: 44690b41db18878a836ddc98e2f8a31f58f13e5f037a262d785114c20f3ec8bf
- scripts/longsplat/orchestrator.py: bd1fa9fa6b02755ed7c0c09d284e1dbf1ffd9b42b9fd863ce453feac5a9fdabf
- scripts/longsplat/convergence_smoke.py: 9ddc5e533f2b02331272a1faff217e2215cc84f34bbbc8f07a9b2586dcb54b4c
- scripts/longsplat/smoke_executor.py: 4ccbdf810661cc492cbf76c6f4bb1ae3a69001571abbbaa9213e3ff962454578
- scripts/longsplat/conversion_executor.py: 0bc4228639ae9fca5092edf2bea3a560ae4e0768f2f39125fe5dff4971b4f63b
- tests/unit/test_remediation_contracts.py: f1ba2feb19054b5485cb045a44a9f6300bee31e84f8db09a1263b0227a0d819f
- tests/unit/test_clean_distribution_contract.py: 489073c4386c68a4dc819b54100a9caed532e601dc3e6a74297635adc7d6d0b8

## Verification boundary and status

The clean CPU boundary uses ./dev.sh and an existing environment only. It
does not run GPU/CUDA, nvidia-smi, network stubs, COLMAP, ffmpeg, training,
render, conversion, or evaluation. The updated nested import evidence proves
plain imports of train, render, convert_3dgs, and gaussian_renderer.network_gui
with listener is None; default forbidden modules remain unloaded.

The truthful remediation classifications remain:

- DEPENDENCY_BOUNDARY_VERIFIED
- CAMERA_MATCHER_GENERALIZATION_PARTIAL — no real COLMAP replay in this CPU scope
- TOOL_PROVIDER_GENERALIZATION_PARTIAL — provider fixtures pass, real provider preflight is deferred
- STAGE_IDENTITY_PARTIAL — old-schema migration remains explicitly blocking
- COMBINED_CPU_VERIFIED

The three CORE_REQUIRED items are therefore not declared globally complete.
The nested fork content is now published; the root branch publication remains
pending. The next authorized stage after root publication is one GPU
preflight/run. No GPU execution is part of this local closure.
