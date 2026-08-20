"""One CPU-only orchestrator for raw video through camera staging.

The orchestrator is intentionally unable to enter training, conversion, or
delivery.  Those stages belong to a later reviewed slice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .camera_staging import CameraStagingBlocked, parse_colmap_camera, stage_centered_pinhole
from .colmap_contract import (
    CameraPrior,
    ColmapCommand,
    build_colmap_commands,
    collect_intrinsics_evidence,
    evaluate_sparse_geometry,
    enumerate_model_components,
    parse_colmap_image_pose_records,
    load_camera_prior,
    require_single_model_component,
    run_colmap_command,
    SUPPORTED_CAMERA_MODELS,
    SUPPORTED_MATCHING_MODES,
    validate_camera_prior_for_canonical,
)
from .frame_contract import FrameMetric, FrameSelectionConfig, select_frames
from .longsplat_input import (
    build_longsplat_input_stage,
    future_workload_profile,
    load_parent_camera_staging_evidence,
)
from .pipeline_contract import (
    PipelineBlocked,
    RunLedger,
    build_run_identity,
    code_identity,
    preflight_dependencies,
    sha256_file,
    stable_sha256,
    write_json_once,
)
from .tool_provider import default_tool_paths, resolve_tool_provider
from .video_contract import (
    build_ffprobe_command,
    build_frame_extract_command,
    candidate_sampling_plan,
    canonical_media_record,
    parse_ffprobe_stdout,
    require_ready_canonical_media,
)


STOP_AFTER = ("preflight", "probe", "frames", "colmap", "camera-staging", "longsplat-input")
STAGE_ORDER = {name: index for index, name in enumerate(STOP_AFTER)}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
def _safe_run_id(value: str) -> str:
    if not value or not _RUN_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError("run-id must contain only letters, numbers, '.', '_' and '-'")
    return value


def _write_once_or_verify(path: Path, value: Any) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise PipelineBlocked(f"immutable record differs on resume: {path}")
        return
    write_json_once(path, value)


def _run_process(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    try:
        result = runner(
            list(argv),
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
            check=False,
            shell=False,
        )
        return {
            "argv": list(argv),
            "cwd": None if cwd is None else str(cwd),
            "exit_code": int(getattr(result, "returncode", 1)),
            "stdout": str(getattr(result, "stdout", "") or ""),
            "stderr": str(getattr(result, "stderr", "") or ""),
        }
    except OSError as exc:
        return {
            "argv": list(argv),
            "cwd": None if cwd is None else str(cwd),
            "exit_code": 127,
            "stdout": "",
            "stderr": str(exc),
        }


def _probe_once(
    *,
    command: Sequence[str],
    source: Path,
    source_sha256: str,
    run_dir: Path,
    plan: bool,
    ffprobe_payload: Mapping[str, Any] | None,
    runner: Callable[..., Any],
) -> tuple[dict[str, Any] | None, dict[str, Any], str, str]:
    if ffprobe_payload is not None:
        probe = parse_ffprobe_stdout(
            json.dumps(ffprobe_payload),
            source_path=source,
            source_sha256=source_sha256,
            source_size_bytes=source.stat().st_size,
        )
        return (
            probe,
            {"argv": list(command), "cwd": str(run_dir), "exit_code": 0, "status": "mocked"},
            json.dumps(ffprobe_payload),
            "",
        )
    if plan:
        return (
            None,
            {"argv": list(command), "cwd": str(run_dir), "status": "planned", "exit_code": None},
            "",
            "",
        )
    execution = _run_process(command, cwd=run_dir, runner=runner)
    if execution["exit_code"] != 0:
        return None, execution, execution["stdout"], execution["stderr"]
    probe = parse_ffprobe_stdout(
        execution["stdout"],
        source_path=source,
        source_sha256=source_sha256,
        source_size_bytes=source.stat().st_size,
    )
    return probe, execution, execution["stdout"], execution["stderr"]


def _frame_stage_once(
    *,
    command: Sequence[str],
    run_dir: Path,
    frames_dir: Path,
    probe: Mapping[str, Any],
    plan: bool,
    preprocess_path: Path | None,
    candidate_fps: float,
    runner: Callable[..., Any],
) -> tuple[list[FrameMetric], list[Path], dict[str, Any], str, str]:
    if preprocess_path is not None:
        raise PipelineBlocked(
            "--preprocess-manifest is disabled until a strict source/canonical/pixel adapter exists"
        )
    if plan:
        return (
            [],
            [],
            {"argv": list(command), "cwd": str(run_dir), "status": "planned", "exit_code": None},
            "",
            "",
        )
    execution = _run_process(command, cwd=run_dir, runner=runner)
    stdout, stderr = execution["stdout"], execution["stderr"]
    if execution["exit_code"] != 0:
        return [], [], execution, stdout, stderr
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise PipelineBlocked("ffmpeg produced no canonical frames")
    metrics = _derive_image_metrics(
        frame_paths,
        fps=candidate_fps,
        timestamp_start=0.0,
    )
    expected_width = int(probe["canonical"]["width"])
    expected_height = int(probe["canonical"]["height"])
    if any(metric.width != expected_width or metric.height != expected_height for metric in metrics):
        raise PipelineBlocked(
            "decoded canonical frame dimensions do not match the probe canonical media contract"
        )
    return metrics, frame_paths, execution, stdout, stderr


def _default_tools(route_root: Path) -> dict[str, str]:
    return default_tool_paths(route_root)


def _derive_image_metrics(
    paths: Sequence[Path], *, fps: float, timestamp_start: float = 0.0
) -> list[FrameMetric]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise PipelineBlocked("OpenCV/numpy are required for CPU frame metrics") from exc
    metrics: list[FrameMetric] = []
    previous_gray: Any = None
    previous_hash: Any = None
    for index, path in enumerate(sorted(paths)):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.ndim < 2:
            raise PipelineBlocked(f"cannot decode extracted frame: {path}")
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        overexposed = float(np.mean(gray >= 250))
        underexposed = float(np.mean(gray <= 5))
        motion = None
        if previous_gray is not None:
            motion = float(np.mean(cv2.absdiff(gray, previous_gray)) / 255.0)
        tiny = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        bits = tiny >= float(tiny.mean())
        duplicate = None
        if previous_hash is not None:
            duplicate = int(np.count_nonzero(bits != previous_hash))
        metrics.append(
            FrameMetric(
                frame_id=path.stem,
                timestamp_sec=timestamp_start + float(index) / max(fps, 1e-9),
                frame_index=index,
                path=str(path.resolve()),
                width=int(width),
                height=int(height),
                sha256=sha256_file(path),
                sharpness=sharpness,
                overexposed_ratio=overexposed,
                underexposed_ratio=underexposed,
                motion_score=motion,
                coverage_score=1.0,
                duplicate_score=duplicate,
                provenance="route-cpu-metrics-v1",
            )
        )
        previous_gray = gray
        previous_hash = bits
    return metrics


def _frame_pixel_sha(paths: Sequence[Path]) -> str:
    return stable_sha256(
        [{"name": path.name, "sha256": sha256_file(path)} for path in sorted(paths)]
    )


def _selected_frame_records(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    selected = [frame for frame in selection["frames"] if frame.get("selected")]
    for index, frame in enumerate(selected):
        result.append(
            {
                "frame_id": frame["frame_id"],
                "staged_name": f"frame_{index:06d}.png",
                "path": frame["path"],
                "sha256": frame.get("sha256"),
                "width": frame["width"],
                "height": frame["height"],
                "timestamp_sec": frame["timestamp_sec"],
                "provenance": "selected_canonical_frame",
            }
        )
    return result


def _component_image_names(component: Path) -> list[str]:
    """Read registered image names from one COLMAP component without guessing."""

    path = component / "images.txt"
    if not path.is_file():
        binary = component / "images.bin"
        if not binary.is_file():
            return []
        try:
            from .longsplat_input import parse_colmap_model_binary

            model = parse_colmap_model_binary(component)
            return [str(item["name"]) for item in model.images.values()]
        except Exception:
            return []
    return [str(record["name"]) for record in parse_colmap_image_pose_records(path)]


def _select_mapper_component(
    components: Sequence[Path],
    selected_records: Sequence[Mapping[str, Any]],
) -> tuple[Path, list[str], dict[str, Any]]:
    """Choose one unambiguous temporally contiguous COLMAP component.

    The mapper inventory remains authoritative.  A component is selected only
    by its longest registered run in the current ordered frame contract; ties,
    cross-component duplicates, or an empty/non-contiguous ambiguous choice are
    hard stops.  No component is merged and no count is rewritten.
    """

    if len(components) == 1:
        only = components[0].resolve()
        names = _component_image_names(only)
        if not names:
            # A valid binary model may be inspected by the downstream
            # model-converter TXT contract only after bundle adjustment.  The
            # one-component fact is still truthful here; names are explicitly
            # marked deferred rather than guessed or synthesized.
            return only, [], {
                "schema_version": "colmap-component-inventory-v1",
                "mapper_component_count": 1,
                "active_model_component_count": 1,
                "components": [
                    {
                        "component_name": only.name,
                        "component_path": str(only),
                        "registered_image_names": None,
                        "registered_image_names_source": "deferred_to_model_converter_txt",
                    }
                ],
                "selected_component_name": only.name,
                "selected_component_path": str(only),
                "selection_rule": "exactly one mapper component; registered names deferred to strict TXT evidence",
            }

    inventory: list[dict[str, Any]] = []
    owners: dict[str, str] = {}
    for component in components:
        names = _component_image_names(component)
        if not names or len(names) != len(set(names)):
            raise PipelineBlocked(f"COLMAP component has missing or duplicate registered names: {component}")
        duplicate = sorted(set(names) & set(owners))
        if duplicate:
            raise PipelineBlocked(f"COLMAP components overlap registered names: {duplicate}")
        for name in names:
            owners[name] = component.name
        positions = [
            index
            for index, record in enumerate(selected_records)
            if str(record.get("staged_name")) in set(names)
        ]
        runs: list[list[int]] = []
        current: list[int] = []
        for position in positions:
            if current and position != current[-1] + 1:
                runs.append(current)
                current = []
            current.append(position)
        if current:
            runs.append(current)
        longest = max((len(run) for run in runs), default=0)
        best_runs = [run for run in runs if len(run) == longest and longest > 0]
        inventory.append(
            {
                "component_name": component.name,
                "component_path": str(component.resolve()),
                "registered_image_names": names,
                "registered_image_count": len(names),
                "selected_frame_positions": positions,
                "candidate_contiguous_runs": [
                    {"start_index": run[0], "end_index": run[-1], "count": len(run)}
                    for run in runs
                ],
                "longest_contiguous_run_count": longest,
            }
        )
        if len(best_runs) > 1:
            raise PipelineBlocked(f"COLMAP component has ambiguous non-contiguous registered segments: {component}")
    ranked = sorted(inventory, key=lambda item: (-int(item["longest_contiguous_run_count"]), str(item["component_name"])))
    if not ranked or int(ranked[0]["longest_contiguous_run_count"]) <= 0:
        raise PipelineBlocked("COLMAP mapper components have no registered member in the selected frame contract")
    if len(ranked) > 1 and ranked[0]["longest_contiguous_run_count"] == ranked[1]["longest_contiguous_run_count"]:
        raise PipelineBlocked("COLMAP mapper component selection is ambiguous: equal contiguous segments")
    selected = ranked[0]
    selected_path = Path(str(selected["component_path"])).resolve()
    return selected_path, list(selected["registered_image_names"]), {
        "schema_version": "colmap-component-inventory-v1",
        "mapper_component_count": len(components),
        "active_model_component_count": 1,
        "components": inventory,
        "selected_component_name": selected["component_name"],
        "selected_component_path": str(selected_path),
        "selection_rule": "longest_temporally_contiguous registered segment; ties/overlap blocked; no merge",
    }


def _segment_provenance(
    selected_records: Sequence[Mapping[str, Any]],
    registered_names: Sequence[str],
    *,
    component_count: int,
    component_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe component/gap candidates and the deterministic selected segment.

    The selected COLMAP component is not silently merged with another model.
    Registration gaps remain explicit excluded intervals.  The current raw
    model retains every registered image in the chosen component because its
    TXT/BIN evidence is already a single COLMAP model; the longest contiguous
    run is recorded for downstream quality review and future filtered-model
    support.
    """

    registered = set(str(name) for name in registered_names)
    ordered = [record for record in selected_records if str(record.get("staged_name")) in registered]
    candidates: list[dict[str, Any]] = []
    current: list[Mapping[str, Any]] = []
    for record in selected_records:
        if str(record.get("staged_name")) in registered:
            current.append(record)
            continue
        if current:
            candidates.append({"records": current})
            current = []
    if current:
        candidates.append({"records": current})
    candidate_rows = []
    for item in candidates:
        records = list(item["records"])
        candidate_rows.append(
            {
                "frame_names": [str(record.get("staged_name")) for record in records],
                "frame_count": len(records),
                "start_time_sec": records[0].get("timestamp_sec"),
                "end_time_sec": records[-1].get("timestamp_sec"),
            }
        )
    excluded = [record for record in selected_records if str(record.get("staged_name")) not in registered]
    excluded_intervals: list[dict[str, Any]] = []
    current_excluded: list[Mapping[str, Any]] = []
    previous_index: int | None = None
    for index, record in enumerate(selected_records):
        if str(record.get("staged_name")) in registered:
            if current_excluded:
                excluded_intervals.append(
                    {
                        "frame_names": [str(item.get("staged_name")) for item in current_excluded],
                        "start_time_sec": current_excluded[0].get("timestamp_sec"),
                        "end_time_sec": current_excluded[-1].get("timestamp_sec"),
                        "reason": "not_registered_in_selected_component",
                    }
                )
                current_excluded = []
            previous_index = index
        else:
            if previous_index is None or index == previous_index + 1:
                current_excluded.append(record)
            else:
                current_excluded.append(record)
            previous_index = index
    if current_excluded:
        excluded_intervals.append(
            {
                "frame_names": [str(item.get("staged_name")) for item in current_excluded],
                "start_time_sec": current_excluded[0].get("timestamp_sec"),
                "end_time_sec": current_excluded[-1].get("timestamp_sec"),
                "reason": "not_registered_in_selected_component",
            }
        )
    chosen = max(candidate_rows, key=lambda row: (int(row["frame_count"]), -float(row["start_time_sec"] or 0.0))) if candidate_rows else None
    return {
        "schema_version": "longsplat-segment-provenance-v1",
        "selection_policy": "longest_temporally_contiguous registered candidate; no component merge",
        "component_count_observed": int(component_count),
        "mapper_component_count": int(component_count),
        "active_model_component_count": 1,
        "component_inventory": None if component_inventory is None else dict(component_inventory),
        "registered_component_frame_count": len(ordered),
        "selected_input_frame_count": len(selected_records),
        "coverage_ratio": (len(ordered) / len(selected_records)) if selected_records else 0.0,
        "candidate_segments": candidate_rows,
        "selected_segment": chosen,
        "excluded_time_intervals": excluded_intervals,
        "excluded_frame_count": len(excluded),
        "model_scope": "all registered images in the chosen single COLMAP component; excluded intervals are not staged downstream",
    }


def _verify_pixel_record(path: Path, *, name: str, expected_width: int, expected_height: int, provenance: str) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineBlocked(f"{provenance} pixel is missing: {path}")
    try:
        import cv2
    except ImportError as exc:
        raise PipelineBlocked("OpenCV is required to verify pipeline pixels") from exc
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise PipelineBlocked(f"cannot decode {provenance} pixel: {path}")
    height, width = image.shape[:2]
    if (width, height) != (expected_width, expected_height):
        raise PipelineBlocked(
            f"{provenance} pixel dimensions disagree for {name}: "
            f"actual={width}x{height}, expected={expected_width}x{expected_height}"
        )
    return {
        "frame_id": name,
        "staged_name": name,
        "name": name,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "decoded_pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "width": int(width),
        "height": int(height),
        "provenance": provenance,
    }


def _colmap_input_records(
    selected_records: Sequence[Mapping[str, Any]],
    image_dir: Path,
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in selected_records:
        source_path = Path(str(record["path"])).resolve()
        destination = image_dir / str(record["staged_name"])
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source_path:
                raise PipelineBlocked(f"COLMAP input frame collision: {destination}")
        else:
            os.symlink(source_path, destination)
        verified = _verify_pixel_record(
            destination,
            name=str(record["staged_name"]),
            expected_width=width,
            expected_height=height,
            provenance="colmap_input_frame",
        )
        declared_sha = record.get("sha256")
        if declared_sha is not None and declared_sha != verified["sha256"]:
            raise PipelineBlocked(f"selected canonical frame SHA mismatch: {source_path}")
        verified["selected_source_path"] = str(source_path)
        verified["selected_source_sha256"] = record.get("sha256")
        records.append(verified)
    return records


def _undistorted_output_records(
    image_dir: Path,
    *,
    registered_names: Sequence[str],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    names = sorted(str(name) for name in registered_names)
    if not names:
        raise CameraStagingBlocked("undistorted COLMAP model has no registered image names")
    output: list[dict[str, Any]] = []
    for name in names:
        output.append(
            _verify_pixel_record(
                image_dir / name,
                name=name,
                expected_width=width,
                expected_height=height,
                provenance="undistorted_output_frame",
            )
        )
    actual_names = sorted(path.name for path in image_dir.iterdir() if path.is_file()) if image_dir.is_dir() else []
    if actual_names != names:
        raise CameraStagingBlocked(
            "undistorted image set does not exactly match registered COLMAP names"
        )
    return output


def _load_prior(value: Mapping[str, Any] | str | Path | None, source_sha: str) -> CameraPrior | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        payload = json.loads(Path(value).read_text(encoding="utf-8"))
    else:
        payload = dict(value)
    return load_camera_prior(payload, source_video_sha256=source_sha)


def _stop(ledger: RunLedger, *, status: str, computed_pass: bool, reason: str) -> dict[str, Any]:
    ledger.stop_cpu(status=status, computed_pass=computed_pass, reason=reason)
    return ledger.summary


def _run_longsplat_input_from_parent(
    *,
    input_video: str | Path,
    output_root: str | Path,
    run_id: str | None,
    parent_run_root: str | Path,
    containment_root: str | Path | None,
    depth_source: str,
    camera_model: str,
    matching: str,
    route_root: str | Path | None,
    tool_paths: Mapping[str, str | Path] | None,
    runner: Callable[..., Any],
    code_identity_override: Mapping[str, Any] | str | None,
    future_smoke_profile: str,
) -> dict[str, Any]:
    """Create one derived input run from an immutable camera-staging parent."""

    if depth_source != "disabled":
        raise PipelineBlocked("LongSplat input packaging requires depth_source=disabled")
    if camera_model not in SUPPORTED_CAMERA_MODELS:
        raise PipelineBlocked(
            f"unsupported camera model {camera_model!r}; supported models are "
            f"{sorted(SUPPORTED_CAMERA_MODELS)}"
        )
    if matching not in SUPPORTED_MATCHING_MODES:
        raise PipelineBlocked(
            f"unsupported matcher {matching!r}; supported matchers are "
            f"{sorted(SUPPORTED_MATCHING_MODES)}"
        )
    future_workload_profile(future_smoke_profile)
    source = Path(input_video).resolve()
    if not source.is_file():
        raise PipelineBlocked(f"input video does not exist: {source}")
    source_sha = sha256_file(source)
    route = Path(route_root or Path(__file__).resolve().parents[2]).resolve()
    tools, tool_provider = resolve_tool_provider(route, tool_paths)
    code = code_identity_override if code_identity_override is not None else code_identity(route)
    preflight = preflight_dependencies(
        route_python=tools.get("route_python"),
        backend_python=tools.get("backend_python"),
        ffmpeg=tools.get("ffmpeg"),
        ffprobe=tools.get("ffprobe"),
        colmap=tools.get("colmap"),
        runner=runner,
    )
    parent_binding: dict[str, Any] | None = None
    parent_binding_error: str | None = None
    if preflight["status"] == "passed":
        try:
            parent_evidence = load_parent_camera_staging_evidence(
                parent_run_root,
                expected_source_video_sha256=source_sha,
                current_colmap_identity=preflight["dependencies"]["colmap"],
            )
            parent_binding = dict(parent_evidence["parent_binding"])
        except (PipelineBlocked, OSError, ValueError, json.JSONDecodeError) as exc:
            parent_binding_error = str(exc)
    config = {
        "schema_version": "pipeline-config-v1",
        "stage": "longsplat-input",
        "parent_run_root": str(Path(parent_run_root).resolve()),
        "parent_binding": parent_binding,
        "parent_binding_sha256": None if parent_binding is None else parent_binding.get("binding_sha256"),
        "depth_source": depth_source,
        "camera_model": camera_model,
        "matching": matching,
        "camera_prior": None,
        "observation_rewrite": True,
        "conversion": "CPU COLMAP model_converter TXT->BIN->TXT",
        "future_smoke_profile": future_smoke_profile,
        "tool_provider": tool_provider,
    }
    if parent_binding_error is not None:
        config["parent_binding_error"] = parent_binding_error
    config_sha = stable_sha256(config)
    identity = build_run_identity(
        source_video_sha256=source_sha,
        canonical_config_sha256=config_sha,
        tool_identity_sha256=preflight["tool_identity_sha256"],
        code_identity_value=code,
        source_video_path=source,
        source_video_size_bytes=source.stat().st_size,
    )
    default_run_id = f"raw_video_{source_sha[:16]}_longsplat_input_{identity['run_identity_sha256'][:8]}"
    actual_run_id = _safe_run_id(run_id or default_run_id)
    ledger = RunLedger.create_or_resume(
        output_root=output_root,
        run_id=actual_run_id,
        identity=identity,
    )
    effective_containment_root = Path(containment_root or output_root).resolve()
    _write_once_or_verify(ledger.run_dir / "config.json", config)
    _write_once_or_verify(ledger.run_dir / "preflight.json", preflight)
    existing_preflight = ledger.latest_result("preflight")
    if existing_preflight is None:
        attempt = ledger.begin_attempt("preflight", {"config_sha256": config_sha, "source_sha256": source_sha})
        ledger.finish_attempt(stage="preflight", attempt=attempt, status=preflight["status"], result=preflight)
        existing_preflight = preflight if preflight["status"] == "passed" else None
    if preflight["status"] != "passed":
        return _stop(ledger, status="blocked", computed_pass=False, reason="dependency preflight blocked")
    existing = ledger.latest_result("longsplat-input")
    if existing is not None:
        return _stop(
            ledger,
            status="stopped",
            computed_pass=bool(existing.get("computed_pass")),
            reason="requested stop-after longsplat-input; immutable stage reused",
        )
    attempt = ledger.begin_attempt(
        "longsplat-input",
        {
            "parent_run_root": str(Path(parent_run_root).resolve()),
            "parent_binding": parent_binding,
            "parent_binding_sha256": None if parent_binding is None else parent_binding.get("binding_sha256"),
            "source_video_sha256": source_sha,
            "camera_prior": None,
            "depth_source": depth_source,
            "future_smoke_profile": future_smoke_profile,
        },
    )
    try:
        if parent_binding_error is not None or parent_binding is None:
            raise PipelineBlocked(parent_binding_error or "parent binding was not established")
        colmap_entry = preflight["dependencies"]["colmap"]
        backend_entry = preflight["dependencies"].get("backend_python")
        result = build_longsplat_input_stage(
            parent_run_root=parent_run_root,
            output_dir=attempt / "longsplat_input",
            source_video_sha256=source_sha,
            colmap=colmap_entry["resolved_path"],
            colmap_identity=colmap_entry,
            backend_python=backend_entry["resolved_path"] if backend_entry else tools["backend_python"],
            backend_identity=backend_entry,
            route_root=route,
            containment_root=effective_containment_root,
            code_identity=code if isinstance(code, Mapping) else {"code_identity_sha256": str(code)},
            derived_identity=identity,
            parent_binding=parent_binding,
            runner=runner,
            workload_profile=future_smoke_profile,
        )
        from .validate_external_colmap_contract import validate_training_input

        training_root = Path(result["source_path"])
        static_contract = validate_training_input(
            training_root,
            containment_root=effective_containment_root,
        )
        static_path = training_root / "contract" / "static_contract.json"
        _write_once_or_verify(static_path, static_contract)
        result["computed_pass"] = bool(static_contract.get("computed_pass"))
        result["static_contract_path"] = str(static_path)
        result["static_contract_sha256"] = sha256_file(static_path)
        result["artifacts"] = list(result.get("artifacts", [])) + [
            {"path": str(static_path), "sha256": sha256_file(static_path)}
        ]
    except (PipelineBlocked, OSError, ValueError, json.JSONDecodeError) as exc:
        ledger.finish_attempt(
            stage="longsplat-input",
            attempt=attempt,
            status="blocked",
            result={"reason": str(exc), "parent_run_root": str(Path(parent_run_root).resolve())},
        )
        return _stop(ledger, status="blocked", computed_pass=False, reason=str(exc))
    ledger.finish_attempt(stage="longsplat-input", attempt=attempt, status="passed", result=result)
    return _stop(ledger, status="stopped", computed_pass=True, reason="requested stop-after longsplat-input")


def run_raw_video_pipeline(
    *,
    input_video: str | Path,
    output_root: str | Path,
    run_id: str | None = None,
    camera_prior: Mapping[str, Any] | str | Path | None = None,
    depth_source: str = "disabled",
    stop_after: str = "preflight",
    plan: bool = False,
    camera_model: str = "SIMPLE_RADIAL",
    matching: str = "sequential",
    selection_config: FrameSelectionConfig | None = None,
    preprocess_manifest: str | Path | None = None,
    parent_run_root: str | Path | None = None,
    containment_root: str | Path | None = None,
    route_root: str | Path | None = None,
    tool_paths: Mapping[str, str | Path] | None = None,
    runner: Callable[..., Any] = subprocess.run,
    ffprobe_payload: Mapping[str, Any] | None = None,
    code_identity_override: Mapping[str, Any] | str | None = None,
    future_smoke_profile: str = "smoke100",
) -> dict[str, Any]:
    """Run the requested CPU stage and stop before any training/delivery."""

    if stop_after not in STOP_AFTER:
        raise ValueError(f"stop-after must be one of {STOP_AFTER}")
    if stop_after == "longsplat-input":
        if parent_run_root is None:
            raise PipelineBlocked("--parent-run is required for stop-after longsplat-input")
        return _run_longsplat_input_from_parent(
            input_video=input_video,
            output_root=output_root,
            run_id=run_id,
            parent_run_root=parent_run_root,
            containment_root=containment_root,
            depth_source=depth_source,
            camera_model=camera_model,
            matching=matching,
            route_root=route_root,
            tool_paths=tool_paths,
            runner=runner,
            code_identity_override=code_identity_override,
            future_smoke_profile=future_smoke_profile,
        )
    if parent_run_root is not None:
        raise PipelineBlocked("--parent-run is only valid with stop-after longsplat-input")
    if depth_source != "disabled":
        raise PipelineBlocked("the first raw-video slice supports only --depth-source disabled")
    if camera_model not in SUPPORTED_CAMERA_MODELS:
        raise PipelineBlocked(
            f"unsupported camera model {camera_model!r}; supported models are "
            f"{sorted(SUPPORTED_CAMERA_MODELS)}"
        )
    if matching not in SUPPORTED_MATCHING_MODES:
        raise PipelineBlocked(
            f"unsupported matcher {matching!r}; supported matchers are "
            f"{sorted(SUPPORTED_MATCHING_MODES)}"
        )
    if preprocess_manifest is not None:
        raise PipelineBlocked(
            "--preprocess-manifest is disabled until a strict source/canonical/pixel adapter exists"
        )
    source = Path(input_video).resolve()
    if not source.is_file():
        raise PipelineBlocked(f"input video does not exist: {source}")
    source_sha = sha256_file(source)
    route = Path(route_root or Path(__file__).resolve().parents[2]).resolve()
    tools, tool_provider = resolve_tool_provider(route, tool_paths)
    prior = _load_prior(camera_prior, source_sha)
    selection = selection_config or FrameSelectionConfig()
    config = {
        "schema_version": "pipeline-config-v1",
        "depth_source": depth_source,
        "camera_model": camera_model,
        "matching": matching,
        "selection": selection.to_dict(),
        "camera_prior": None if prior is None else prior.to_dict(),
        "preprocess_manifest": None,
        "preprocess_manifest_status": "disabled_strict_adapter_pending",
        "tool_provider": tool_provider,
    }
    config_sha = stable_sha256(config)
    code = code_identity_override if code_identity_override is not None else code_identity(route)
    preflight = preflight_dependencies(
        route_python=tools.get("route_python"),
        backend_python=tools.get("backend_python"),
        ffmpeg=tools.get("ffmpeg"),
        ffprobe=tools.get("ffprobe"),
        colmap=tools.get("colmap"),
        runner=runner,
    )
    identity = build_run_identity(
        source_video_sha256=source_sha,
        canonical_config_sha256=config_sha,
        tool_identity_sha256=preflight["tool_identity_sha256"],
        code_identity_value=code,
        source_video_path=source,
        source_video_size_bytes=source.stat().st_size,
    )
    default_run_id = f"raw_{source_sha[:16]}_{config_sha[:8]}_{identity['run_identity_sha256'][:8]}"
    actual_run_id = _safe_run_id(run_id or default_run_id)
    ledger = RunLedger.create_or_resume(
        output_root=output_root,
        run_id=actual_run_id,
        identity=identity,
    )
    _write_once_or_verify(ledger.run_dir / "config.json", config)
    _write_once_or_verify(ledger.run_dir / "preflight.json", preflight)

    existing = ledger.latest_result("preflight")
    if existing is None:
        attempt = ledger.begin_attempt("preflight", {"config_sha256": config_sha})
        preflight_status = preflight["status"]
        ledger.finish_attempt(
            stage="preflight",
            attempt=attempt,
            status=preflight_status,
            result=preflight,
        )
        existing = preflight
    if preflight["status"] != "passed":
        return _stop(ledger, status="blocked", computed_pass=False, reason="dependency preflight blocked")
    if STAGE_ORDER[stop_after] <= STAGE_ORDER["preflight"]:
        return _stop(ledger, status="stopped", computed_pass=True, reason="requested stop-after preflight")

    probe: dict[str, Any] | None = None
    existing = ledger.latest_result("probe")
    if existing is not None and not plan and existing.get("execution", {}).get("status") == "planned":
        existing = None
    if existing is not None:
        probe = existing.get("probe")
    else:
        ffprobe = preflight["dependencies"]["ffprobe"]["resolved_path"]
        command = build_ffprobe_command(ffprobe, source)
        attempt = ledger.begin_attempt("probe", {"argv": command, "source_sha256": source_sha})
        try:
            probe, execution, stdout, stderr = _probe_once(
                command=command,
                source=source,
                source_sha256=source_sha,
                run_dir=ledger.run_dir,
                plan=plan,
                ffprobe_payload=ffprobe_payload,
                runner=runner,
            )
        except (PipelineBlocked, ValueError) as exc:
            ledger.finish_attempt(
                stage="probe",
                attempt=attempt,
                status="blocked",
                result={"reason": str(exc), "argv": command},
            )
            return _stop(ledger, status="blocked", computed_pass=False, reason=str(exc))
        if execution["exit_code"] not in (0, None):
            ledger.finish_attempt(stage="probe", attempt=attempt, status="failed", result=execution, stdout=stdout, stderr=stderr)
            return _stop(ledger, status="failed", computed_pass=False, reason="ffprobe failed")
        result = {"execution": execution, "probe": probe, "probe_path": None, "artifacts": []}
        if probe is not None:
            probe_path = ledger.run_dir / "video_probe-v1.json"
            _write_once_or_verify(probe_path, probe)
            result["probe_path"] = str(probe_path)
            result["artifacts"] = [{"path": str(probe_path), "sha256": sha256_file(probe_path)}]
        ledger.finish_attempt(stage="probe", attempt=attempt, status="planned" if plan and probe is None else "passed", result=result, stdout=stdout, stderr=stderr)
    if probe is None:
        return _stop(ledger, status="stopped", computed_pass=False, reason="probe planned; no ffprobe execution requested")
    if bool(probe.get("timing", {}).get("vfr")):
        return _stop(
            ledger,
            status="blocked",
            computed_pass=False,
            reason="VFR input requires a real per-frame timestamp adapter; first slice is CFR-only",
        )
    if STAGE_ORDER[stop_after] <= STAGE_ORDER["probe"]:
        return _stop(ledger, status="stopped", computed_pass=True, reason="requested stop-after probe")
    if prior is not None:
        validate_camera_prior_for_canonical(
            prior,
            canonical_width=int(probe["canonical"]["width"]),
            canonical_height=int(probe["canonical"]["height"]),
        )

    existing = ledger.latest_result("frames")
    selection_result: dict[str, Any] | None = None
    media: dict[str, Any] | None = None
    if existing is not None and not plan and existing.get("execution", {}).get("status") == "planned":
        existing = None
    if existing is not None:
        selection_result = existing.get("selection")
        media = existing.get("canonical_media")
    else:
        sampling = candidate_sampling_plan(
            probe,
            target_fps=selection.target_fps,
            min_frames=selection.min_frames,
            max_frames=selection.max_frames,
        )
        attempt = ledger.begin_attempt(
            "frames",
            {
                "source_sha256": source_sha,
                "sampling_plan": sampling,
            },
        )
        frames_dir = attempt / "canonical_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        pattern = frames_dir / "frame_%06d.png"
        ffmpeg = preflight["dependencies"]["ffmpeg"]["resolved_path"]
        command = build_frame_extract_command(
            ffmpeg,
            source,
            pattern,
            probe,
            candidate_fps=float(sampling["candidate_fps"]),
            candidate_cap=int(sampling["candidate_cap"]),
        )
        try:
            metrics, frame_paths, execution, stdout, stderr = _frame_stage_once(
                command=command,
                run_dir=ledger.run_dir,
                frames_dir=frames_dir,
                probe=probe,
                plan=plan,
                preprocess_path=None,
                candidate_fps=float(sampling["candidate_fps"]),
                runner=runner,
            )
        except (PipelineBlocked, ValueError, OSError, json.JSONDecodeError) as exc:
            ledger.finish_attempt(
                stage="frames",
                attempt=attempt,
                status="blocked",
                result={"reason": str(exc), "argv": command},
            )
            return _stop(ledger, status="blocked", computed_pass=False, reason=str(exc))
        if execution["exit_code"] not in (0, None):
            ledger.finish_attempt(stage="frames", attempt=attempt, status="failed", result=execution, stdout=stdout, stderr=stderr)
            return _stop(ledger, status="failed", computed_pass=False, reason="canonical frame extraction failed")
        if not metrics:
            result = {"execution": execution, "selection": None, "canonical_media": None, "sampling_plan": sampling, "artifacts": []}
            ledger.finish_attempt(stage="frames", attempt=attempt, status="planned", result=result, stdout=stdout, stderr=stderr)
            return _stop(ledger, status="stopped", computed_pass=False, reason="frame stage planned; no pixels materialized")
        pixel_sha = _frame_pixel_sha(frame_paths)
        media = canonical_media_record(
            probe,
            source_path=source,
            source_sha256=source_sha,
            canonical_pixels_sha256=pixel_sha,
        )
        canonical_sha = media["binding"]["canonical_media_sha256"]
        selection_result = select_frames(
            metrics,
            duration_sec=float(probe["timing"]["duration_sec"]),
            source_video_sha256=source_sha,
            canonical_media_sha256=canonical_sha,
            canonical_width=int(media["canonical"]["width"]),
            canonical_height=int(media["canonical"]["height"]),
            config=selection,
        )
        media_path = ledger.run_dir / "canonical_media-v1.json"
        selection_path = ledger.run_dir / "frame_selection-v1.json"
        _write_once_or_verify(media_path, media)
        _write_once_or_verify(selection_path, selection_result)
        result = {
            "execution": execution,
            "sampling_plan": sampling,
            "canonical_media": media,
            "canonical_media_path": str(media_path),
            "selection": selection_result,
            "selection_path": str(selection_path),
            "artifacts": [
                {"path": str(media_path), "sha256": sha256_file(media_path)},
                {"path": str(selection_path), "sha256": sha256_file(selection_path)},
                *[
                    {"path": str(path), "sha256": sha256_file(path)}
                    for path in frame_paths
                ],
            ],
        }
        ledger.finish_attempt(stage="frames", attempt=attempt, status="passed", result=result, stdout=stdout, stderr=stderr)
    if selection_result is None or media is None:
        return _stop(ledger, status="stopped", computed_pass=False, reason="frame stage has no canonical pixel evidence")
    if STAGE_ORDER[stop_after] <= STAGE_ORDER["frames"]:
        return _stop(ledger, status="stopped", computed_pass=True, reason="requested stop-after frames")

    colmap_result: dict[str, Any] | None = ledger.latest_result("colmap")
    if colmap_result is None:
        canonical_sha = require_ready_canonical_media(media)
        selected_records = _selected_frame_records(selection_result)
        attempt = ledger.begin_attempt(
            "colmap",
            {
                "source_sha256": source_sha,
                "canonical_media_sha256": canonical_sha,
                "selected_frame_names": [record["staged_name"] for record in selected_records],
            },
        )
        workspace = attempt / "workspace"
        image_dir = workspace / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        input_records = _colmap_input_records(
            selected_records,
            image_dir,
            width=int(media["canonical"]["width"]),
            height=int(media["canonical"]["height"]),
        )
        pre_plan = build_colmap_commands(
            colmap=preflight["dependencies"]["colmap"]["resolved_path"],
            workspace=workspace,
            image_dir=image_dir,
            source_video_sha256=source_sha,
            canonical_media_sha256=canonical_sha,
            canonical_width=int(media["canonical"]["width"]),
            canonical_height=int(media["canonical"]["height"]),
            camera_model=camera_model,
            camera_prior=prior,
            matching=matching,
        )
        # COLMAP normally creates some of these paths itself, but the contract
        # owns the layout and creates every output parent explicitly.  Each
        # attempt gets a fresh workspace, so a retry cannot consume leftovers.
        paths = pre_plan["paths"]
        for directory in (
            Path(paths["mapper_output_root"]),
            Path(paths["bundle_adjusted_binary"]),
            Path(paths["bundle_adjusted_txt"]),
            Path(paths["undistorted_root"]),
            Path(paths["undistorted_sparse_txt"]),
        ):
            directory.mkdir(parents=True, exist_ok=True)
        command_results: list[dict[str, Any]] = []
        if plan:
            status = "planned"
            plan_data = pre_plan
            geometry = {"computed_pass": False, "accepted": False, "hard_failures": ["commands_planned"], "warnings": []}
            evidence = None
        else:
            status = "passed"
            mapper_stage_count = 3
            for command_data in pre_plan["commands"][:mapper_stage_count]:
                result = run_colmap_command(
                    ColmapCommand(str(command_data["stage"]), tuple(command_data["argv"]), str(command_data["cwd"])),
                    tool_identity=preflight["dependencies"]["colmap"],
                    log_dir=attempt / "logs",
                    runner=runner,
                )
                command_results.append(result)
                if result["exit_code"] != 0:
                    status = "failed"
                    break
            if status == "failed":
                plan_data = pre_plan
                geometry = {"computed_pass": False, "accepted": False, "hard_failures": ["command_failed"], "warnings": []}
                evidence = None
            else:
                try:
                    components = enumerate_model_components(pre_plan["paths"]["mapper_output_root"])
                    if not components:
                        raise PipelineBlocked("COLMAP mapper produced no model component")
                    mapper_component, mapper_names, component_inventory = _select_mapper_component(
                        components,
                        selected_records,
                    )
                except PipelineBlocked as exc:
                    plan_data = pre_plan
                    geometry = {"computed_pass": False, "accepted": False, "hard_failures": [str(exc)], "warnings": []}
                    evidence = None
                    status = "blocked"
                else:
                    plan_data = build_colmap_commands(
                        colmap=preflight["dependencies"]["colmap"]["resolved_path"],
                        workspace=workspace,
                        image_dir=image_dir,
                        source_video_sha256=source_sha,
                        canonical_media_sha256=canonical_sha,
                        canonical_width=int(media["canonical"]["width"]),
                        canonical_height=int(media["canonical"]["height"]),
                        camera_model=camera_model,
                        camera_prior=prior,
                        matching=matching,
                        mapper_component=mapper_component,
                    )
                    for command_data in plan_data["commands"][mapper_stage_count:]:
                        result = run_colmap_command(
                            ColmapCommand(str(command_data["stage"]), tuple(command_data["argv"]), str(command_data["cwd"])),
                            tool_identity=preflight["dependencies"]["colmap"],
                            log_dir=attempt / "logs",
                            runner=runner,
                        )
                        command_results.append(result)
                        if result["exit_code"] != 0:
                            status = "failed"
                            break
                    evidence = None
                    geometry = {"computed_pass": False, "accepted": False, "hard_failures": ["post_mapper_command_failed"], "warnings": []}
                    if status == "passed":
                        ba_txt = Path(plan_data["paths"]["bundle_adjusted_txt"])
                        undist_txt = Path(plan_data["paths"]["undistorted_sparse_txt"])
                        required_models = (ba_txt, undist_txt)
                        if not all((model / "cameras.txt").is_file() and (model / "images.txt").is_file() and (model / "points3D.txt").is_file() for model in required_models):
                            status = "blocked"
                            geometry = {"computed_pass": False, "accepted": False, "hard_failures": ["TXT model_converter sidecars missing"], "warnings": []}
                        else:
                            registered_names = _component_image_names(Path(mapper_component))
                            if not registered_names:
                                for raw_image in (ba_txt / "images.txt").read_text(encoding="utf-8").splitlines():
                                    fields = raw_image.split()
                                    if len(fields) >= 10 and fields[0].lstrip("+-").isdigit():
                                        registered_names.append(fields[9])
                            segment_provenance = _segment_provenance(
                                selected_records,
                                registered_names,
                                component_count=len(components),
                                component_inventory=component_inventory,
                            )
                            evidence = collect_intrinsics_evidence(
                                ba_txt,
                                source_video_sha256=source_sha,
                                canonical_media_sha256=canonical_sha,
                                component_count=1,
                                mapper_component_count=len(components),
                                component_inventory=component_inventory,
                                expected_image_names=registered_names,
                                refine_argv=next(
                                    command["argv"]
                                    for command in plan_data["commands"]
                                    if command["stage"] == "bundle_adjuster"
                                ),
                            )
                            geometry = evaluate_sparse_geometry(
                                evidence,
                                expected_image_names=registered_names,
                            )
                            if not geometry["computed_pass"]:
                                status = "blocked"
        colmap_result = {
            "plan": plan_data,
            "command_results": command_results,
            "execution_pass": bool(command_results) and all(item.get("exit_code") == 0 for item in command_results),
            "computed_pass": bool(geometry.get("computed_pass")),
            "geometry": geometry,
            "evidence": evidence,
            "sparse_required": True,
            "dense_mvs": {"enabled": False, "blocking": False, "next_stage": "future"},
            "selected_frame_records": selected_records,
            "colmap_input_frame_records": input_records,
            "segment_provenance": locals().get("segment_provenance"),
            "component_inventory": locals().get("component_inventory"),
            "mapper_component_count": len(locals().get("components", [])),
            "active_model_component_count": 1 if evidence is not None else 0,
            "artifacts": [] if evidence is None else [
                {"path": str(Path(plan_data["paths"]["bundle_adjusted_txt"]) / name), "sha256": sha256_file(Path(plan_data["paths"]["bundle_adjusted_txt"]) / name)}
                for name in ("cameras.txt", "images.txt", "points3D.txt")
            ],
        }
        ledger.finish_attempt(stage="colmap", attempt=attempt, status=status, result=colmap_result)
        if status == "failed":
            return _stop(ledger, status="failed", computed_pass=False, reason="COLMAP command execution failed")
        if status == "blocked":
            return _stop(ledger, status="blocked", computed_pass=False, reason="COLMAP geometry/artifact contract failed")
        if status == "planned":
            return _stop(ledger, status="stopped", computed_pass=False, reason="COLMAP plan recorded without execution")
    if STAGE_ORDER[stop_after] <= STAGE_ORDER["colmap"]:
        return _stop(
            ledger,
            status="stopped" if colmap_result.get("computed_pass") else "blocked",
            computed_pass=bool(colmap_result.get("computed_pass")),
            reason="requested stop-after colmap",
        )

    existing = ledger.latest_result("camera-staging")
    if existing is not None:
        return _stop(ledger, status="stopped", computed_pass=bool(existing.get("contract")), reason="requested stop-after camera-staging")
    attempt = ledger.begin_attempt(
        "camera-staging",
        {"colmap_contract_sha256": colmap_result["plan"]["contract_sha256"], "source_sha256": source_sha},
    )
    try:
        paths = colmap_result["plan"]["paths"]
        raw_model = Path(paths["bundle_adjusted_txt"]) / "cameras.txt"
        undistorted_root = Path(paths["undistorted_sparse_txt"])
        undistorted_model = undistorted_root / "cameras.txt"
        if not raw_model.is_file() or not undistorted_model.is_file():
            raise CameraStagingBlocked("COLMAP TXT sidecar camera models are missing")
        raw_camera = parse_colmap_camera(raw_model)
        undistorted_camera = parse_colmap_camera(undistorted_model)
        registered_names = colmap_result.get("evidence", {}).get("registered_image_names", [])
        undistorted_records = _undistorted_output_records(
            Path(paths["undistorted_images"]),
            registered_names=registered_names,
            width=undistorted_camera.width,
            height=undistorted_camera.height,
        )
        undistorter_argv = next(
            command["argv"] for command in colmap_result["plan"]["commands"]
            if command["stage"] == "image_undistorter"
        )
        contract = stage_centered_pinhole(
            raw_camera=raw_camera,
            undistorted_camera=undistorted_camera,
            source_video_sha256=source_sha,
            canonical_media_sha256=str(media["binding"]["canonical_media_sha256"]),
            frame_records=undistorted_records,
            output_dir=attempt / "camera_staging",
            undistorter_argv=undistorter_argv,
            canonical_media_record=media,
            canonical_media_record_path=ledger.run_dir / "canonical_media-v1.json",
            pixel_provenance={
                "selected_canonical_frames": colmap_result["selected_frame_records"],
                "colmap_input_frames": colmap_result["colmap_input_frame_records"],
            },
        )
        contract_path = attempt / "camera_staging" / "camera_contract-v1.json"
        _write_once_or_verify(contract_path, contract)
        final_artifacts = [
            {"path": str(contract_path), "sha256": sha256_file(contract_path)},
            *[
                {"path": str(frame["path"]), "sha256": sha256_file(frame["path"])}
                for frame in contract["frames"]
            ],
        ]
    except (CameraStagingBlocked, PipelineBlocked, OSError, ValueError) as exc:
        ledger.finish_attempt(stage="camera-staging", attempt=attempt, status="blocked", result={"reason": str(exc), "artifacts": []})
        return _stop(ledger, status="blocked", computed_pass=False, reason=str(exc))
    ledger.finish_attempt(
        stage="camera-staging",
        attempt=attempt,
        status="passed",
        result={"contract": contract, "contract_path": str(contract_path), "artifacts": final_artifacts},
    )
    return _stop(ledger, status="stopped", computed_pass=True, reason="requested stop-after camera-staging")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only raw video to LongSplat camera staging")
    parser.add_argument("--input-video", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--parent-run",
        dest="parent_run_root",
        help="immutable camera-staging parent run root for the derived longsplat-input stage",
    )
    parser.add_argument("--camera-prior")
    parser.add_argument("--depth-source", default="disabled")
    parser.add_argument("--stop-after", choices=STOP_AFTER, default="preflight")
    parser.add_argument("--plan", action="store_true", help="record commands without executing ffprobe/ffmpeg/COLMAP")
    parser.add_argument("--camera-model", default="SIMPLE_RADIAL")
    parser.add_argument("--matching", choices=("sequential", "exhaustive"), default="sequential")
    parser.add_argument("--backend-env", help="explicit backend environment root containing bin/python")
    parser.add_argument("--backend-python", help="explicit backend Python executable")
    parser.add_argument("--ffmpeg", help="explicit ffmpeg executable")
    parser.add_argument("--ffprobe", help="explicit ffprobe executable")
    parser.add_argument("--colmap", help="explicit COLMAP executable")
    parser.add_argument("--route-python", help="explicit route Python executable")
    parser.add_argument(
        "--future-smoke-profile",
        choices=("smoke100", "smoke100-v1", "formal30000", "formal30000-v1"),
        default="smoke100",
        help="fixed future workload contract for the derived LongSplat input stage",
    )
    parser.add_argument(
        "--preprocess-manifest",
        help="currently disabled; strict source/canonical/pixel adapter is pending",
    )
    parser.add_argument("--target-fps", type=float, default=2.0)
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=240)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        result = run_raw_video_pipeline(
            input_video=args.input_video,
            output_root=args.output_root,
            run_id=args.run_id,
            parent_run_root=args.parent_run_root,
            camera_prior=args.camera_prior,
            depth_source=args.depth_source,
            stop_after=args.stop_after,
            plan=args.plan,
            camera_model=args.camera_model,
            matching=args.matching,
            preprocess_manifest=args.preprocess_manifest,
            selection_config=FrameSelectionConfig(
                target_fps=args.target_fps,
                min_frames=args.min_frames,
                max_frames=args.max_frames,
            ),
            future_smoke_profile=args.future_smoke_profile,
            tool_paths={
                key: value
                for key, value in {
                    "backend_env": args.backend_env,
                    "backend_python": args.backend_python,
                    "ffmpeg": args.ffmpeg,
                    "ffprobe": args.ffprobe,
                    "colmap": args.colmap,
                    "route_python": args.route_python,
                }.items()
                if value is not None
            },
        )
    except (PipelineBlocked, ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"stopped", "planned"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
