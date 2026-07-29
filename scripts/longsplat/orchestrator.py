"""
Single LongSplat orchestration entry point.

Executes the full pipeline in order:

  record → preflight → commands → prepare → depth → training →
  telemetry → conversion → PLY validation → provenance → complete

Every failure path (including preflight) leaves a terminal ``failed``
record.  The run record is created before any fallible check so no
failure is silent.
"""

from __future__ import annotations

import hashlib
import json
import os as _os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


from .convert import validate_converted_ply
from .depth_bridge import (
    DepthContractError,
    materialize_all,
)
from .manifest_adapter import adapt_manifest
from .prepare_input import prepare_input
from .provenance import sha256_file
from .quality_gates import audit_pose_quality
from .quality_metrics import analyze_ply_quality
from .run_record import (
    RunStatus,
    add_artifact,
    create_run_record,
    transition_status,
    write_run_record,
)
from .telemetry import (
    summarize_conversion_telemetry,
    summarize_loss_telemetry,
    summarize_pose_telemetry,
    summarize_vda_telemetry,
)
from .runner import (
    BackendValidationError,
    LongSplatConfig,
    _check_python,
    _check_repo,
    backend_subprocess_env,
    build_effective_commands,
    config_to_dict,
)
from .validate_input import validate_manifest

_RUN_ID_RE = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    manifest_path: str | Path,
    segment_id: str,
    config: LongSplatConfig,
    repo_root: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    depth_manifest_path: str | Path | None = None,
    run_id: str | None = None,
    python_exe: str = "python",
) -> int:
    """Run the full LongSplat pipeline for one segment.

    Parameters
    ----------
    manifest_path : str or Path
        Path to a preprocess manifest (schema_version ``"2.0"``).  A
        ``coverage_v1`` manifest requires ``keyframe_quality_report.json`` in
        the same directory.
    segment_id : str
        Which segment from the manifest to process.
    config : LongSplatConfig
        Training and conversion configuration.
    repo_root : str or Path
        Root of the locked LongSplat repository.
    output_dir : str or Path
        Parent directory for run-scoped output.
    project_root : str or Path
        Repository root for resolving manifest paths.  Required.
    depth_manifest_path : str or Path or None
        Optional depth manifest from depth-prior module.
    run_id : str or None
        Custom run identifier.  Auto-generated when omitted.
    python_exe : str
        Path to the Python interpreter for the LongSplat backend.

    Returns
    -------
    int
        0 on success, non-zero on failure.
    """
    run_dir: Path | None = None
    record: dict[str, Any] | None = None

    try:
        pr = Path(project_root).resolve(strict=True)

        # --- 1. Create run directory and record BEFORE any preflight ---
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_seg = _sanitise_run_id(segment_id)
        if run_id is not None:
            r_id = _sanitise_run_id(run_id)
        else:
            r_id = f"longsplat_{safe_seg}_{ts}_{uuid.uuid4().hex[:12]}"
        run_dir = out / r_id
        run_dir.mkdir(parents=False, exist_ok=False)
        _echo(f"Run directory: {run_dir}")

        record = create_run_record(
            run_dir=run_dir,
            run_id=r_id,
            manifest_path=manifest_path,
            project_root=pr,
            repo_root=repo_root,
            depth_manifest_path=depth_manifest_path,
        )

        # --- 2. Pre-flight validation (record already exists) ---
        _echo("Validating backend repository ...")
        try:
            validated_repo = _check_repo(repo_root, backend_mode=config.backend_mode)
        except BackendValidationError as exc:
            transition_status(
                record,
                run_dir,
                RunStatus.FAILED,
                stage="preflight",
                details={
                    "status": "failed",
                    "check": "backend_repo",
                    "reason": type(exc).__name__,
                    "message": str(exc)[:2000],
                },
            )
            _echo(f"Backend validation failed: {exc}", file=sys.stderr)
            return 1

        try:
            _check_python(python_exe)
        except BackendValidationError as exc:
            transition_status(
                record,
                run_dir,
                RunStatus.FAILED,
                stage="preflight",
                details={
                    "status": "failed",
                    "check": "python_exe",
                    "reason": type(exc).__name__,
                    "message": str(exc)[:2000],
                },
            )
            _echo(f"Python validation failed: {exc}", file=sys.stderr)
            return 1

        # Validate main repo identity (only when project_root is a git repo)
        if (pr / ".git").exists():
            repo_sha_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(pr),
            )
            if repo_sha_result.returncode != 0 or not _SHA_RE.match(
                repo_sha_result.stdout.strip(),
            ):
                transition_status(
                    record,
                    run_dir,
                    RunStatus.FAILED,
                    stage="preflight",
                    details={
                        "status": "failed",
                        "check": "repo_sha",
                        "reason": "BackendValidationError",
                        "message": f"Failed to get valid repo SHA: "
                        f"{repo_sha_result.stdout.strip()[:200]}",
                    },
                )
                return 1
            repo_sha = repo_sha_result.stdout.strip()
        else:
            repo_sha = "unknown"
        record["repo_sha"] = repo_sha

        # Resolve actual backend identity
        backend_identity = _resolve_backend_identity(
            validated_repo,
            backend_mode=config.backend_mode,
        )
        record["backend"]["commit_actual"] = backend_identity["commit"]
        record["backend"]["dirty"] = backend_identity["dirty"]
        record["backend"]["submodules"] = backend_identity["submodules"]
        record["backend"]["mode"] = backend_identity["mode"]
        record["backend"]["diff_sha256"] = backend_identity["diff_sha256"]
        record["backend"]["submodule_diffs"] = backend_identity["submodule_diffs"]
        record["effective_seed"] = 0
        # Snapshot config before orchestrator mutates source_path / model_path
        record["config_requested"] = config_to_dict(config)
        write_run_record(record, run_dir)

        # --- 3. Validate and adapt manifest ---
        _echo("Validating producer manifest ...")
        with open(manifest_path, "rb") as fh:
            producer_raw = fh.read()
        producer_sha = hashlib.sha256(producer_raw).hexdigest()
        record["_pv"] = {
            "project_root": str(pr),
            "producer_manifest_sha256": producer_sha,
            "producer_quality_report_sha256": None,
            "producer_quality_report_binding": None,
            "backend_commit_actual": backend_identity["commit"],
            "backend_dirty": backend_identity["dirty"],
            "backend_submodules": backend_identity["submodules"],
        }
        write_run_record(record, run_dir)
        producer = json.loads(producer_raw)

        quality_report: dict[str, Any] | None = None
        producer_run = producer.get("run")
        if isinstance(producer_run, dict):
            producer_run_provenance: dict[str, Any] = {}
            if "id" in producer_run:
                producer_run_provenance["id"] = producer_run["id"]
            producer_code = producer_run.get("code")
            if isinstance(producer_code, dict):
                producer_code_provenance = {
                    key: producer_code[key]
                    for key in ("commit", "working_tree_dirty")
                    if key in producer_code
                }
                if producer_code_provenance:
                    producer_run_provenance["code"] = producer_code_provenance
            if producer_run_provenance:
                record["_pv"]["producer_run"] = producer_run_provenance
        write_run_record(record, run_dir)

        producer_settings = producer.get("settings")
        if (
            isinstance(producer_settings, dict)
            and producer_settings.get("keyframe_policy") == "coverage_v1"
        ):
            quality_report_path = (
                Path(manifest_path).resolve().parent / "keyframe_quality_report.json"
            )
            record["_pv"]["producer_quality_report_path"] = str(quality_report_path)
            record["_pv"]["producer_quality_report_binding"] = (
                "semantic_only_no_producer_manifest_digest"
            )
            write_run_record(record, run_dir)
            with quality_report_path.open("rb") as fh:
                quality_report_raw = fh.read()
            record["_pv"]["producer_quality_report_sha256"] = hashlib.sha256(
                quality_report_raw
            ).hexdigest()
            write_run_record(record, run_dir)
            quality_report = json.loads(quality_report_raw)

        consumer = adapt_manifest(
            producer,
            segment_id,
            pr,
            quality_report=quality_report,
        )

        consumer_manifest_path = run_dir / "consumer_manifest.json"
        consumer_manifest_path.write_text(
            json.dumps(consumer, indent=2) + "\n",
            encoding="utf-8",
        )
        validate_manifest(consumer_manifest_path)

        # --- 4. Set run-time config paths, THEN build commands ONCE ---
        config.source_path = str(run_dir / "input")
        config.model_path = str(run_dir / "longsplat_model")

        _echo("Building effective commands ...")
        commands = build_effective_commands(validated_repo, config, python_exe)
        record["commands"] = {
            "train": {"argv": list(commands.train), "cwd": str(validated_repo)},
            "convert": {"argv": list(commands.convert), "cwd": str(validated_repo)},
        }
        write_run_record(record, run_dir)

        # --- 5. Prepare input ---
        _echo("Preparing input ...")
        prepare_input(consumer, run_dir)

        # --- 6. VDA depth materialisation ---
        depth_source = config.extra_train_args.get("depth_source", "mast3r")
        if depth_source != "mast3r":
            _echo(f"VDA depth source={depth_source}, materialising ...")
            if not depth_manifest_path:
                raise PipelineError(
                    "depth_source is VDA but no depth_manifest_path was provided"
                )

            mapping_path = run_dir / "input" / "frame_mapping.json"
            if not mapping_path.is_file():
                raise PipelineError("frame_mapping.json missing in run input directory")

            with mapping_path.open(encoding="utf-8") as fh:
                frame_mapping = json.load(fh)

            depth_dir = run_dir / "input" / "depths"

            try:
                mat_result = materialize_all(
                    depth_manifest_path=Path(depth_manifest_path),
                    frame_mapping=frame_mapping,
                    project_root=pr,
                    output_depth_dir=depth_dir,
                )
            except DepthContractError as exc:
                transition_status(
                    record,
                    run_dir,
                    RunStatus.FAILED,
                    stage="depth_materialization",
                    details={
                        "status": "failed",
                        "reason": type(exc).__name__,
                        "message": str(exc)[:2000],
                    },
                )
                raise PipelineError(str(exc)) from exc

            record.setdefault("depth", {})["materialization"] = {
                "expected_count": mat_result.expected_count,
                "materialized_count": mat_result.materialized_count,
                "depth_manifest_sha256": mat_result.depth_manifest_sha256,
                "frames": [
                    {
                        "rgb_path": f.rgb_path,
                        "prepared_name": f.prepared_name,
                        "source_sha256": f.source_sha256,
                        "output_sha256": f.output_sha256,
                        "shape": list(f.shape),
                        "dtype": f.dtype,
                    }
                    for f in mat_result.frames
                ],
            }
            _echo(
                f"Depth materialised: {mat_result.materialized_count}/"
                f"{mat_result.expected_count} frames"
            )
            write_run_record(record, run_dir)

        # --- 7. Record provenance ---
        effective_argv_data = {
            "train": list(commands.train),
            "convert": list(commands.convert),
            "python_exe": python_exe,
            "backend_cwd": str(validated_repo),
            "seed": config.seed,
        }
        argv_path = run_dir / "effective_argv.json"
        argv_tmp = argv_path.with_suffix(argv_path.suffix + ".tmp")
        with open(argv_tmp, "w", encoding="utf-8") as fh:
            json.dump(effective_argv_data, fh, indent=2, ensure_ascii=False)
        _os.replace(argv_tmp, argv_path)

        add_artifact(
            record,
            run_dir,
            artifact_path=str(argv_path),
            artifact_type="effective_argv",
            stage="provenance",
            metadata={"sha256": sha256_file(argv_path)},
        )
        write_run_record(record, run_dir)

        # --- 8. Run training ---
        _echo("Running training ...")
        transition_status(
            record,
            run_dir,
            RunStatus.RUNNING,
            stage="training",
            details={"status": "started"},
        )
        train_start = time.monotonic()
        train_result = subprocess.run(
            list(commands.train),
            capture_output=True,
            text=True,
            cwd=str(validated_repo),
            env=backend_subprocess_env(validated_repo),
        )
        train_duration = time.monotonic() - train_start

        log_paths = _save_subprocess_logs(
            run_dir,
            "train",
            train_result.stdout,
            train_result.stderr,
        )

        # Persist any diagnostics emitted before a crash/non-zero exit.
        telemetry = record.setdefault("telemetry", {})
        telemetry["pose"] = summarize_pose_telemetry(train_result.stdout)
        usage = None
        if depth_source != "mast3r":
            usage = _summarize_vda_usage(train_result.stdout)
            record.setdefault("depth", {})["usage"] = usage
            telemetry["vda"] = summarize_vda_telemetry(train_result.stdout)
        write_run_record(record, run_dir)

        if train_result.returncode != 0:
            transition_status(
                record,
                run_dir,
                RunStatus.FAILED,
                stage="training",
                details={
                    "status": "failed",
                    "exit_code": train_result.returncode,
                    "duration_s": round(train_duration, 1),
                    "stderr": train_result.stderr[-2000:],
                    **log_paths,
                },
            )
            raise TrainingFailed(
                f"Training exited with code {train_result.returncode}",
                returncode=train_result.returncode,
                stdout=train_result.stdout,
                stderr=train_result.stderr,
            )

        _echo(f"Training completed in {train_duration:.1f}s")
        transition_status(
            record,
            run_dir,
            RunStatus.RUNNING,
            stage="training",
            details={
                "status": "completed",
                "exit_code": 0,
                "duration_s": round(train_duration, 1),
                **log_paths,
            },
        )
        add_artifact(
            record,
            run_dir,
            artifact_path=str(run_dir / "longsplat_model"),
            artifact_type="longsplat_model",
            stage="training",
            metadata={"iteration": config.iterations},
        )

        # --- 8b. Enforce backward-compatible VDA usage gate ---
        if depth_source != "mast3r":
            if usage is None or usage["aligned"] == 0:
                raise PipelineError(
                    "VDA was requested but no frame reported aligned depth"
                )

        # --- 8c. Post-training pose audit ---
        cameras_json = Path(config.model_path) / "cameras_all_train.json"
        if cameras_json.is_file():
            _echo("Auditing camera poses ...")
            pose_telemetry = telemetry.get("pose", {})
            audit_kwargs: dict[str, Any] = {}
            if config.expected_native_checkpoint_iteration is not None:
                audit_kwargs["model_path"] = Path(config.model_path)
                audit_kwargs["expected_native_checkpoint_iteration"] = (
                    config.expected_native_checkpoint_iteration
                )
            # Use config gate thresholds when available
            if config.quality_gates is not None:
                pg = config.quality_gates.pose
                audit_kwargs.update(
                    {
                        "max_orthogonality_error": pg.max_orthogonality_error,
                        "max_determinant_error": pg.max_determinant_error,
                        "max_rotation_step_deg": pg.max_rotation_step_deg,
                        "max_translation_step_ratio": pg.max_translation_step_ratio,
                    }
                )
            audit_result = audit_pose_quality(
                cameras_json, pose_telemetry, **audit_kwargs
            )
            record.setdefault("quality_gates", {})["pose"] = {
                "thresholds": {
                    "max_orthogonality_error": audit_kwargs.get(
                        "max_orthogonality_error", 1e-4
                    ),
                    "max_determinant_error": audit_kwargs.get(
                        "max_determinant_error", 1e-4
                    ),
                    "max_rotation_step_deg": audit_kwargs.get(
                        "max_rotation_step_deg", 25.0
                    ),
                    "max_translation_step_ratio": audit_kwargs.get(
                        "max_translation_step_ratio", 6.0
                    ),
                },
                "passed": audit_result["passed"],
                "reasons": audit_result["reasons"],
                "trajectory": audit_result["trajectory"],
                "telemetry": audit_result["telemetry"],
            }
            write_run_record(record, run_dir)

            if not audit_result["passed"]:
                transition_status(
                    record,
                    run_dir,
                    RunStatus.FAILED,
                    stage="pose_quality_gate",
                    details={
                        "status": "failed",
                        "reasons": audit_result["reasons"],
                        "trajectory": audit_result["trajectory"],
                        "telemetry": audit_result["telemetry"],
                    },
                )
                _echo(
                    f"Pose audit FAILED: {'; '.join(audit_result['reasons'])}",
                    file=sys.stderr,
                )
                return 1
            _echo("Pose audit passed.")
        else:
            _echo("No cameras_all_train.json found, skipping pose audit.")

        # --- 8d. Post-training VDA quality gate ---
        if depth_source != "mast3r":
            _echo("Auditing VDA depth quality ...")
            vda_telemetry = telemetry.get("vda", {})
            cameras_json = Path(config.model_path) / "cameras_all_train.json"
            train_camera_count = 0
            if cameras_json.is_file():
                with open(cameras_json, encoding="utf-8") as fh:
                    train_camera_count = len(json.load(fh))
            test_cameras_json = Path(config.model_path) / "cameras_all_test.json"
            test_camera_count = 0
            if test_cameras_json.is_file():
                with open(test_cameras_json, encoding="utf-8") as fh:
                    test_camera_count = len(json.load(fh))

            materialized_count = (
                record.get("depth", {})
                .get("materialization", {})
                .get("materialized_count", 0)
            )

            vda_reasons: list[str] = []
            record_count = len(vda_telemetry.get("records", []))
            missing = vda_telemetry.get("missing", 0)
            aligned = vda_telemetry.get("aligned", 0)
            rejected = vda_telemetry.get("rejected", 0)

            if train_camera_count > 0 and record_count != train_camera_count:
                vda_reasons.append(
                    f"VDA record count {record_count} != train cameras {train_camera_count}"
                )

            if missing > 0:
                vda_reasons.append(f"VDA missing {missing} frames")

            total_expected = train_camera_count + test_camera_count
            if total_expected > 0 and materialized_count != total_expected:
                vda_reasons.append(
                    f"materialized depth count {materialized_count} "
                    f"!= expected {total_expected} "
                    f"(train {train_camera_count} + test {test_camera_count})"
                )

            vda_min_aligned = 0.98
            if config.quality_gates is not None:
                vda_min_aligned = config.quality_gates.vda.min_aligned_fraction

            aligned_ratio = (
                aligned / train_camera_count if train_camera_count > 0 else 0.0
            )
            if aligned_ratio < vda_min_aligned:
                vda_reasons.append(
                    f"VDA aligned ratio {aligned_ratio:.4f} < {vda_min_aligned} "
                    f"(aligned={aligned}, rejected={rejected}, missing={missing})"
                )

            record.setdefault("quality_gates", {})["vda"] = {
                "thresholds": {
                    "min_aligned_ratio": vda_min_aligned,
                    "max_missing": 0,
                },
                "passed": len(vda_reasons) == 0,
                "reasons": vda_reasons,
                "telemetry": {
                    "record_count": record_count,
                    "aligned": aligned,
                    "rejected": rejected,
                    "missing": missing,
                    "train_camera_count": train_camera_count,
                    "test_camera_count": test_camera_count,
                    "materialized_count": materialized_count,
                },
            }
            write_run_record(record, run_dir)

            if vda_reasons:
                transition_status(
                    record,
                    run_dir,
                    RunStatus.FAILED,
                    stage="vda_quality_gate",
                    details={
                        "status": "failed",
                        "reasons": vda_reasons,
                        "telemetry": record["quality_gates"]["vda"]["telemetry"],
                    },
                )
                _echo(
                    f"VDA audit FAILED: {'; '.join(vda_reasons)}",
                    file=sys.stderr,
                )
                return 1
            _echo("VDA audit passed.")

        # --- 8e. Post-training non-finite loss gate ---
        loss_telemetry = summarize_loss_telemetry(train_result.stdout)
        record.setdefault("quality_gates", {})["loss"] = {
            "passed": not loss_telemetry["has_nonfinite"],
            "has_nonfinite": loss_telemetry["has_nonfinite"],
            "record_count": loss_telemetry["record_count"],
        }
        write_run_record(record, run_dir)

        if loss_telemetry["has_nonfinite"]:
            transition_status(
                record,
                run_dir,
                RunStatus.FAILED,
                stage="loss_finite_gate",
                details={
                    "status": "failed",
                    "reason": "non-finite loss detected in training telemetry",
                    "telemetry": loss_telemetry,
                },
            )
            _echo(
                "Loss audit FAILED: non-finite loss detected in LOSS_TELEMETRY",
                file=sys.stderr,
            )
            return 1
        _echo("Loss audit passed.")

        # --- 9. Run conversion ---
        _echo("Running conversion ...")
        cameras_src = Path(config.model_path) / "cameras_all_train.json"
        cameras_dst = Path(config.model_path) / "cameras_all.json"
        if cameras_src.exists() and not cameras_dst.exists():
            shutil.copyfile(cameras_src, cameras_dst)

        transition_status(
            record,
            run_dir,
            RunStatus.RUNNING,
            stage="conversion",
            details={"status": "started"},
        )
        conv_start = time.monotonic()
        conv_result = subprocess.run(
            list(commands.convert),
            capture_output=True,
            text=True,
            cwd=str(validated_repo),
            env=backend_subprocess_env(validated_repo),
        )
        conv_duration = time.monotonic() - conv_start

        log_paths = _save_subprocess_logs(
            run_dir,
            "convert",
            conv_result.stdout,
            conv_result.stderr,
        )
        record.setdefault("telemetry", {})["conversion"] = (
            summarize_conversion_telemetry(conv_result.stdout)
        )
        write_run_record(record, run_dir)

        if conv_result.returncode != 0:
            transition_status(
                record,
                run_dir,
                RunStatus.FAILED,
                stage="conversion",
                details={
                    "status": "failed",
                    "exit_code": conv_result.returncode,
                    "duration_s": round(conv_duration, 1),
                    "stderr": conv_result.stderr[-2000:],
                    **log_paths,
                },
            )
            raise ConversionFailed(
                f"Conversion exited with code {conv_result.returncode}",
                returncode=conv_result.returncode,
                stdout=conv_result.stdout,
                stderr=conv_result.stderr,
            )

        _echo(f"Conversion completed in {conv_duration:.1f}s")
        transition_status(
            record,
            run_dir,
            RunStatus.RUNNING,
            stage="conversion",
            details={
                "status": "completed",
                "exit_code": 0,
                "duration_s": round(conv_duration, 1),
                **log_paths,
            },
        )
        # --- 10. Strictly validate the raw converted PLY ---
        converted_ply = Path(config.model_path) / "converted_3dgs" / "point_cloud.ply"
        _echo("Validating PLY ...")
        try:
            meta = validate_converted_ply(converted_ply)
            _echo(
                f"PLY valid: {meta['vertex_count']} vertices, "
                f"sha256={meta['sha256'][:12]}..."
            )
        except Exception as exc:
            transition_status(
                record,
                run_dir,
                RunStatus.FAILED,
                stage="ply_validation",
                details={"status": "failed", "error": str(exc)},
            )
            raise PLYValidationFailed(str(exc)) from exc

        add_artifact(
            record,
            run_dir,
            artifact_path=str(converted_ply),
            artifact_type="ply",
            stage="ply_validation",
            metadata={
                "iteration": config.convert_iteration,
                "prune_ratio": config.convert_prune_ratio,
                "vertex_count": meta["vertex_count"],
                "sha256": meta["sha256"],
                "finite_validation": "passed",
            },
        )

        # --- 11. PLY publication gate ---
        if (
            config.quality_gates is not None
            and config.quality_gates.ply.mode == "enforce"
        ):
            _echo("Running PLY publication gate ...")
            ply_quality = analyze_ply_quality(converted_ply)

            ply_gate = config.quality_gates.ply
            ply_reasons: list[str] = []

            effective_frac = ply_quality.get("effective_fraction", 0.0)
            if effective_frac < ply_gate.min_effective_fraction:
                ply_reasons.append(
                    f"effective_fraction {effective_frac:.4f} < {ply_gate.min_effective_fraction}"
                )

            aniso_q99 = ply_quality.get("anisotropy_q99", 0.0)
            if aniso_q99 > ply_gate.max_anisotropy_q99:
                ply_reasons.append(
                    f"anisotropy_q99 {aniso_q99:.2f} > {ply_gate.max_anisotropy_q99}"
                )

            quat_frac = ply_quality.get("quaternion_within_1pct_fraction", 0.0)
            if quat_frac < ply_gate.min_unit_quaternion_fraction:
                ply_reasons.append(
                    f"unit_quaternion_fraction {quat_frac:.6f} < {ply_gate.min_unit_quaternion_fraction}"
                )

            record.setdefault("quality_gates", {})["ply"] = {
                "thresholds": {
                    "min_effective_fraction": ply_gate.min_effective_fraction,
                    "max_anisotropy_q99": ply_gate.max_anisotropy_q99,
                    "min_unit_quaternion_fraction": ply_gate.min_unit_quaternion_fraction,
                },
                "passed": len(ply_reasons) == 0,
                "reasons": ply_reasons,
                "metrics": {
                    "effective_fraction": effective_frac,
                    "anisotropy_q99": aniso_q99,
                    "quaternion_within_1pct_fraction": quat_frac,
                    "finite_core_fraction": ply_quality.get(
                        "finite_core_fraction", 0.0
                    ),
                },
            }
            write_run_record(record, run_dir)

            if ply_reasons:
                transition_status(
                    record,
                    run_dir,
                    RunStatus.FAILED,
                    stage="ply_publication_gate",
                    details={
                        "status": "failed",
                        "reasons": ply_reasons,
                        "metrics": record["quality_gates"]["ply"]["metrics"],
                    },
                )
                _echo(
                    f"PLY publication gate FAILED: {'; '.join(ply_reasons)}",
                    file=sys.stderr,
                )
                return 1
            _echo("PLY publication gate passed.")

        # --- 12. Complete ---
        transition_status(record, run_dir, RunStatus.COMPLETE)
        _echo("Pipeline complete.")
        return 0

    except KeyboardInterrupt:
        _write_failed_safe(record, run_dir, "interrupted", "KeyboardInterrupt")
        _echo("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        _write_failed_safe(
            record,
            run_dir,
            "exception",
            f"{type(exc).__name__}: {exc}",
        )
        _echo(f"FAILED: {exc}", file=sys.stderr)
        if run_dir is not None and run_dir.exists():
            _echo(f"Run directory preserved for debugging: {run_dir}")
        return 1


# ---------------------------------------------------------------------------
# Pipeline-specific errors
# ---------------------------------------------------------------------------


class PipelineError(Exception):
    """Base for pipeline-stage errors."""


class TrainingFailed(PipelineError):
    def __init__(self, msg: str, *, returncode: int, stdout: str, stderr: str):
        super().__init__(msg)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class ConversionFailed(PipelineError):
    def __init__(self, msg: str, *, returncode: int, stdout: str, stderr: str):
        super().__init__(msg)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class PLYValidationFailed(PipelineError):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _summarize_vda_usage(stdout: str) -> dict[str, int]:
    """Parse VDA_USAGE markers from backend stdout and return counts."""
    counts: dict[str, int] = {"aligned": 0, "missing": 0, "rejected": 0}
    for result in counts:
        counts[result] = stdout.count(f"VDA_USAGE result={result} ")
    return counts


def _echo(msg: str, file=sys.stdout) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=file, flush=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitise_run_id(run_id: str) -> str:
    """Validate and return *run_id* if it matches the allowed pattern."""
    if not _RUN_ID_RE.match(run_id):
        raise PipelineError(
            f"Invalid run_id {run_id!r}: must match {_RUN_ID_RE.pattern}"
        )
    return run_id


def _resolve_backend_identity(repo_root: Path, *, backend_mode: str) -> dict[str, Any]:
    """Query the actual backend identity: commit, dirty flag, submodule SHAs,
    and in research_local mode a SHA-256 of ``git diff --binary HEAD``.

    CR-T123-07: submodule diffs are hashed recursively so the run record
    can distinguish different local patches at the same HEAD.
    CR-T123-09: all git query failures are fail-closed — a non-zero exit
    code or missing output is a ``BackendValidationError``, never a
    silent default to clean / empty.
    """
    commit_result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        raise BackendValidationError(
            f"Failed to get backend HEAD commit: exit {commit_result.returncode}"
        )
    commit = commit_result.stdout.strip()
    if not _SHA_RE.match(commit):
        raise BackendValidationError(f"Backend commit is not a valid SHA: {commit!r}")

    # ---- dirty check: fail closed (CR-T123-09) ----
    dirty_result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if dirty_result.returncode != 0:
        raise BackendValidationError(
            f"git status failed with exit {dirty_result.returncode}: "
            f"{dirty_result.stderr.strip()[:500]}"
        )
    dirty = bool(dirty_result.stdout.strip())

    # ---- discover submodules recursively ----
    # $displaypath is relative to the top-level superproject, unlike $sm_path
    # which is relative to the immediate parent (FV-02).
    sub_list_result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "submodule",
            "foreach",
            "--recursive",
            "--quiet",
            "echo $displaypath",
        ],
        capture_output=True,
        text=True,
    )
    if sub_list_result.returncode != 0:
        raise BackendValidationError(
            f"git submodule foreach failed with exit "
            f"{sub_list_result.returncode}: "
            f"{sub_list_result.stderr.strip()[:500]}"
        )
    sub_paths = [
        p.strip() for p in sub_list_result.stdout.strip().splitlines() if p.strip()
    ]

    canonical_root = repo_root.resolve()

    submodules: dict[str, str] = {}
    resolved_sub_paths: dict[str, Path] = {}  # saved for diff loop (RC-FV-F05)
    for sub_path in sub_paths:
        sp = (canonical_root / sub_path).resolve()
        # Containment: resolved path must be inside the canonical repo root
        try:
            sp.relative_to(canonical_root)
        except ValueError:
            raise BackendValidationError(
                f"Submodule path {sub_path} resolves outside repo root "
                f"{canonical_root}: {sp}"
            )
        if not sp.is_dir():
            # Fail closed: if git foreach reported a path it must exist (FV-02)
            raise BackendValidationError(
                f"Submodule path reported by git foreach does not exist: {sub_path}"
            )
        sha_result = subprocess.run(
            ["git", "-C", str(sp), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if sha_result.returncode != 0:
            # CR-T123-09: fail closed instead of silent omit
            raise BackendValidationError(
                f"Failed to get HEAD for submodule {sub_path}: "
                f"exit {sha_result.returncode} — "
                f"{sha_result.stderr.strip()[:500]}"
            )
        sha = sha_result.stdout.strip()
        if not _SHA_RE.match(sha):
            raise BackendValidationError(
                f"Submodule {sub_path} HEAD is not a valid SHA: {sha!r}"
            )
        submodules[sub_path] = sha
        resolved_sub_paths[sub_path] = sp

    # Fingerprint local patches in research_local mode
    diff_sha256: str | None = None
    submodule_diffs: dict[str, str | None] = {}
    if backend_mode == "research_local":
        diff_result = subprocess.run(
            ["git", "-C", str(canonical_root), "diff", "--binary", "HEAD", "--"],
            capture_output=True,
        )
        if diff_result.returncode != 0:
            raise BackendValidationError("Failed to fingerprint backend diff")
        diff_sha256 = (
            hashlib.sha256(diff_result.stdout).hexdigest()
            if diff_result.stdout
            else None
        )
        if dirty and diff_sha256 is None:
            raise BackendValidationError(
                "research_local backend is dirty but git diff produced no output"
            )

        # ---- recursive submodule diffs (CR-T123-07) ----
        for sub_path in sub_paths:
            # Re-resolve and re-validate containment before each diff query
            # (LATEST-FV-03): the path could have been replaced with a symlink
            # or junction pointing outside the repo after the HEAD loop.
            expected = resolved_sub_paths[sub_path]
            current = (canonical_root / sub_path).resolve()
            try:
                current.relative_to(canonical_root)
            except ValueError as exc:
                raise BackendValidationError(
                    f"Submodule path {sub_path} resolves outside repo root "
                    f"during diff query: {current}"
                ) from exc
            if current != expected:
                raise BackendValidationError(
                    f"Submodule path {sub_path} changed between HEAD and diff "
                    f"queries: was {expected}, now {current}"
                )
            if not current.is_dir():
                raise BackendValidationError(
                    f"Submodule path vanished between HEAD and diff queries: {sub_path}"
                )
            sd_result = subprocess.run(
                ["git", "-C", str(current), "diff", "--binary", "HEAD", "--"],
                capture_output=True,
            )
            if sd_result.returncode != 0:
                raise BackendValidationError(
                    f"Failed to fingerprint submodule diff for {sub_path}"
                )
            submodule_diffs[sub_path] = (
                hashlib.sha256(sd_result.stdout).hexdigest()
                if sd_result.stdout
                else None
            )

    return {
        "commit": commit,
        "dirty": dirty,
        "submodules": submodules,
        "mode": backend_mode,
        "diff_sha256": diff_sha256,
        "submodule_diffs": submodule_diffs,
    }


def _write_failed_safe(
    record: dict[str, Any] | None,
    run_dir: Path | None,
    reason: str,
    message: str,
) -> None:
    """Best-effort write of a terminal ``failed`` status."""
    if record is None or run_dir is None:
        return
    try:
        transition_status(
            record,
            run_dir,
            RunStatus.FAILED,
            stage="exception",
            details={"reason": reason, "message": str(message)[:2000]},
        )
    except Exception:
        record["status"] = RunStatus.FAILED.value
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        record.setdefault("stages", {})["exception"] = {
            "reason": reason,
            "message": str(message)[:2000],
        }
        try:
            write_run_record(record, run_dir)
        except Exception:
            pass


def _save_subprocess_logs(
    run_dir: Path,
    stage: str,
    stdout: str,
    stderr: str,
) -> dict[str, str]:
    """Save subprocess stdout/stderr to log files and return their paths."""
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{stage}_stdout.log"
    stderr_path = logs_dir / f"{stage}_stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
    return {
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
