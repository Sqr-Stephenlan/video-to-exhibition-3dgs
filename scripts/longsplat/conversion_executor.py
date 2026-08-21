"""Generic conversion and same-camera evaluation executor.

This module consumes an authority manifest produced by the current run.  It
does not know any historical run path, camera count, image name, or evidence
digest.  ``validate-only`` builds the exact plan without launching a child.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .authority_manifest import (
    AuthorityManifestError,
    load_authority_manifest,
)
from .pipeline_contract import PipelineBlocked, resolve_contained_path, resolve_containment_root


class ConversionExecutorBlocked(RuntimeError):
    """A generic conversion/evaluation contract failed."""


def _fail(message: str) -> None:
    raise ConversionExecutorBlocked(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not valid JSON: {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is missing or symlinked: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _training_image_residency_strategy(model: Path) -> str | None:
    """Read the already-produced training policy for conversion forwarding."""

    path = model / "image_residency_training-v1.json"
    if not path.exists():
        return None
    payload = _load_json(path, "training image residency telemetry")
    strategy = payload.get("strategy")
    if strategy not in {"cpu-stream-v1", "gpu-all-v0"}:
        _fail("training image residency telemetry strategy is unsupported")
    return str(strategy)


def _validate_image_residency_telemetry(
    snapshot: Path,
    *,
    authority: Mapping[str, Any],
    required: bool,
) -> dict[str, Any]:
    """Validate conversion-phase residency evidence without a VRAM estimate gate."""

    path = snapshot / "image_residency_conversion-v1.json"
    if not path.exists():
        if required:
            _fail(f"conversion image residency telemetry is missing: {path}")
        return {
            "schema_version": "image-residency-telemetry-v1",
            "status": "legacy_absent_allowed",
            "path": str(path),
        }
    payload = _load_json(path, "conversion image residency telemetry")
    if payload.get("schema_version") != "image-residency-telemetry-v1":
        _fail("conversion image residency telemetry schema differs")
    if payload.get("phase") != "conversion":
        _fail("conversion image residency telemetry phase differs")
    if payload.get("strategy") not in {"cpu-stream-v1", "gpu-all-v0"}:
        _fail("conversion image residency telemetry strategy is unsupported")
    if payload.get("camera_count") != int(authority["camera_count"]):
        _fail("conversion image residency telemetry camera count differs")
    contract = _load_json(snapshot / "external_colmap_pose_contract.json", "external pose contract")
    names = contract.get("image_names")
    if not isinstance(names, list) or not all(isinstance(value, str) for value in names):
        _fail("conversion external pose contract image_names are missing")
    if payload.get("camera_order") != names:
        _fail("conversion image residency telemetry camera order differs")
    if payload.get("camera_order_sha256") is not None:
        import hashlib

        expected_order_sha = hashlib.sha256(
            json.dumps(names, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if payload.get("camera_order_sha256") != expected_order_sha:
            _fail("conversion image residency telemetry camera order SHA differs")
    dimensions = payload.get("dimensions")
    expected_dimensions = authority.get("camera_dimensions", {})
    if not isinstance(dimensions, Mapping) or not isinstance(expected_dimensions, Mapping):
        _fail("conversion image residency telemetry dimensions are missing")
    if dimensions.get("width") != expected_dimensions.get("width") or dimensions.get("height") != expected_dimensions.get("height"):
        _fail("conversion image residency telemetry dimensions differ")
    if not isinstance(payload.get("image_dtype"), str) or not payload["image_dtype"]:
        _fail("conversion image residency telemetry image_dtype is missing")
    for field in (
        "cpu_resident_image_bytes",
        "gpu_resident_gt_frame_count",
        "gpu_resident_gt_frame_bytes",
        "gpu_resident_gt_frame_count_peak",
        "gpu_resident_gt_frame_bytes_peak",
        "transfer_count",
        "transfer_bytes",
        "device_errors",
    ):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(f"conversion image residency field is invalid: {field}")
    if payload.get("device_errors") != 0:
        _fail("conversion image residency telemetry reports a device error")
    if payload.get("theory_is_advisory") is not True:
        _fail("conversion image residency telemetry must mark theory as advisory")
    if payload.get("transfer_count") == 0 and payload.get("transfer_bytes") != 0:
        _fail("conversion image residency telemetry reports transfer bytes without a transfer")
    if payload.get("transfer_count", 0) > 0 and payload.get("transfer_bytes", 0) <= 0:
        _fail("conversion image residency telemetry has transfers without bytes")
    if payload.get("strategy") == "cpu-stream-v1" and (
        payload.get("gpu_resident_gt_frame_count") != 0
        or payload.get("gpu_resident_gt_frame_bytes") != 0
        or int(payload.get("gpu_resident_gt_frame_count_peak", 0)) > 1
    ):
        _fail("conversion cpu-stream-v1 telemetry reports camera-held CUDA GT residency")
    return {**payload, "path": str(path), "sha256": _sha256(path)}


def _route_outputs(
    route_root: str | Path,
    *,
    containment_root: str | Path | None = None,
) -> tuple[Path, Path]:
    route = Path(route_root).resolve()
    if containment_root is not None:
        try:
            return route, resolve_containment_root(containment_root)
        except PipelineBlocked as exc:
            _fail(str(exc))
    outputs = route / "outputs"
    if outputs.is_symlink() or not outputs.is_dir():
        _fail(f"route outputs directory is missing or symlinked: {outputs}")
    return route, outputs.resolve()


def output_path(
    value: str | Path,
    route_root: str | Path,
    label: str,
    *,
    must_exist: bool = False,
    containment_root: str | Path | None = None,
) -> Path:
    """Resolve a path below the verified dynamic data root."""

    route, outputs = _route_outputs(route_root, containment_root=containment_root)
    if containment_root is not None:
        try:
            return resolve_contained_path(value, root=outputs, label=label, must_exist=must_exist)
        except PipelineBlocked as exc:
            _fail(str(exc))
    raw = Path(value)
    if not raw.is_absolute():
        _fail(f"{label} must be absolute: {raw}")
    project_name = route.parents[1].name if len(route.parents) > 1 else route.name
    if raw.parts.count(project_name) > 1:
        _fail(f"{label} contains a repeated worktree/project prefix: {raw}")
    try:
        relative = raw.relative_to(outputs)
    except ValueError:
        _fail(f"{label} must be below route outputs: {raw}")
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        _fail(f"{label} must be a strict descendant of route outputs: {raw}")
    probe = outputs
    if probe.is_symlink():
        _fail(f"{label} traverses a symlinked outputs root: {outputs}")
    for part in relative.parts:
        probe = probe / part
        if probe.is_symlink():
            _fail(f"{label} traverses a symlink: {probe}")
    resolved = raw.resolve(strict=False)
    try:
        resolved.relative_to(outputs)
    except ValueError:
        _fail(f"{label} resolves outside route outputs: {resolved}")
    if resolved == outputs:
        _fail(f"{label} cannot equal route outputs")
    if must_exist and not resolved.exists():
        _fail(f"{label} is missing: {resolved}")
    return resolved


def source_relatives(*, iteration: int) -> tuple[str, ...]:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= 0:
        _fail("checkpoint iteration must be a positive integer")
    return (
        "cfg_args",
        "cameras_all_train.json",
        "cameras_all_test.json",
        "external_colmap_pose_contract.json",
        f"point_cloud/iteration_{iteration}/point_cloud.ply",
        f"point_cloud/iteration_{iteration}/color_mlp.pt",
        f"point_cloud/iteration_{iteration}/cov_mlp.pt",
        f"point_cloud/iteration_{iteration}/opacity_mlp.pt",
    )


def _manifest_paths(authority: Mapping[str, Any], route: Path) -> dict[str, Path]:
    manifest = authority["manifest"]
    profile = authority["profile"]
    model = Path(str(manifest["training_model"]["path"])).resolve()
    training_input = Path(str(manifest["training_input"]["path"])).resolve()
    nested_raw = route / "third_party/LongSplat"
    if nested_raw.is_symlink():
        _fail(f"nested LongSplat is symlinked: {nested_raw}")
    nested = nested_raw.resolve()
    provider = manifest.get("tool_provider")
    backend_value = None
    if isinstance(provider, Mapping):
        effective = provider.get("effective")
        if isinstance(effective, Mapping) and isinstance(effective.get("backend_python"), str):
            backend_value = effective["backend_python"]
    backend = backend_value or os.environ.get("LONGSPLAT_BACKEND_PYTHON", "")
    if backend:
        backend_python = Path(backend).resolve()
    else:
        from .tool_provider import default_tool_paths

        backend_python = Path(default_tool_paths(route)["backend_python"]).resolve()
    if provider is not None and (not backend_python.is_file() or not os.access(backend_python, os.X_OK)):
        _fail(f"declared backend provider is missing or not executable: {backend_python}")
    for label, path in (("training model", model), ("training input", training_input), ("nested LongSplat", nested)):
        if not path.is_dir() or path.is_symlink():
            _fail(f"{label} is missing or symlinked: {path}")
    return {
        "route": route,
        "source": training_input,
        "model": model,
        "nested": nested,
        "backend_python": backend_python,
        "iteration": Path(str(profile["checkpoint_iteration"])),
    }


def _conversion_environment(paths: Mapping[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    backend_python = paths["backend_python"]
    env["VENV_DIR"] = str(backend_python.parent.parent)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    route = str(paths["route"])
    nested = str(paths["nested"])
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(item for item in (nested, route, previous) if item)
    return env


def _build_conversion_argv(
    *,
    authority: Mapping[str, Any],
    evidence_root: Path,
    snapshot: Path,
    manifest_path: Path,
    route: Path,
) -> tuple[list[str], list[str]]:
    try:
        from .runner import LongSplatConfig, build_convert_command
    except ImportError:  # direct script compatibility
        from scripts.longsplat.runner import LongSplatConfig, build_convert_command  # type: ignore
    paths = _manifest_paths(authority, route)
    route = paths["route"]
    profile = authority["profile"]
    config = LongSplatConfig(
        source_path=str(paths["source"]),
        model_path=str(snapshot),
        iterations=profile["conversion_iterations"],
        seed=profile["seed"],
        backend_mode=profile["backend_mode"],
        convert_iteration=profile["checkpoint_iteration"],
        convert_prune_ratio=profile["prune_ratio"],
        convert_anisotropy_reg_weight=profile["anisotropy_reg_weight"],
        convert_anisotropy_soft_limit=profile["anisotropy_soft_limit"],
        image_residency=_training_image_residency_strategy(paths["model"]),
    )
    nested_argv = build_convert_command(paths["nested"], config, str(paths["backend_python"]))
    record_path = evidence_root / "conversion_record.json"
    argv = [
        str(route / "dev.sh"),
        "python",
        "-m",
        "scripts.longsplat.reconvert_existing",
        "--authority-manifest",
        str(manifest_path),
        "--destination-model",
        str(snapshot),
        "--repo-root",
        str(paths["nested"]),
        "--backend-python",
        str(paths["backend_python"]),
        "--output-record",
        str(record_path),
        "--precreated-snapshot",
        "--conversion-observability-root",
        str(evidence_root),
        "--conversion-live-stdout",
        str(evidence_root / "conversion-stdout-live.log"),
        "--conversion-live-stderr",
        str(evidence_root / "conversion-stderr-live.log"),
        "--conversion-progress",
        str(evidence_root / "conversion-progress-v1.jsonl"),
    ]
    return argv, nested_argv


def build_conversion_argv_for_paths(
    *,
    route: Path,
    source: Path,
    model: Path,
    nested: Path,
    backend_python: Path,
    evidence_root: Path,
    snapshot: Path,
    profile_id: str = "standard30000-v1",
) -> tuple[list[str], list[str]]:
    """Compatibility helper used by focused command-construction tests."""

    try:
        from .authority_manifest import conversion_profile
        from .runner import LongSplatConfig, build_convert_command
    except ImportError:  # direct script compatibility
        from scripts.longsplat.authority_manifest import conversion_profile  # type: ignore
        from scripts.longsplat.runner import LongSplatConfig, build_convert_command  # type: ignore
    profile = conversion_profile(profile_id)
    config = LongSplatConfig(
        source_path=str(source),
        model_path=str(snapshot),
        iterations=profile["conversion_iterations"],
        seed=profile["seed"],
        backend_mode=profile["backend_mode"],
        convert_iteration=profile["checkpoint_iteration"],
        convert_prune_ratio=profile["prune_ratio"],
        convert_anisotropy_reg_weight=profile["anisotropy_reg_weight"],
        convert_anisotropy_soft_limit=profile["anisotropy_soft_limit"],
    )
    nested_argv = build_convert_command(nested, config, str(backend_python))
    argv = [
        str(route / "dev.sh"),
        "python",
        "-m",
        "scripts.longsplat.reconvert_existing",
        "--source-model",
        str(model),
        "--destination-model",
        str(snapshot),
        "--source-path",
        str(source),
        "--repo-root",
        str(nested),
        "--backend-python",
        str(backend_python),
        "--checkpoint-iteration",
        str(profile["checkpoint_iteration"]),
        "--conversion-iterations",
        str(profile["conversion_iterations"]),
        "--prune-ratio",
        str(profile["prune_ratio"]),
        "--anisotropy-reg-weight",
        str(profile["anisotropy_reg_weight"]),
        "--anisotropy-soft-limit",
        str(profile["anisotropy_soft_limit"]),
        "--backend-mode",
        str(profile["backend_mode"]),
        "--output-record",
        str(evidence_root / "conversion_record.json"),
        "--precreated-snapshot",
        "--conversion-observability-root",
        str(evidence_root),
        "--conversion-live-stdout",
        str(evidence_root / "conversion-stdout-live.log"),
        "--conversion-live-stderr",
        str(evidence_root / "conversion-stderr-live.log"),
        "--conversion-progress",
        str(evidence_root / "conversion-progress-v1.jsonl"),
    ]
    return argv, nested_argv


def _snapshot(
    *,
    authority: Mapping[str, Any],
    evidence_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    model = Path(str(authority["manifest"]["training_model"]["path"])).resolve()
    iteration = int(authority["profile"]["checkpoint_iteration"])
    snapshot = evidence_root / "conversion_model_snapshot"
    if snapshot.exists() or snapshot.is_symlink():
        _fail(f"conversion snapshot must be fresh: {snapshot}")
    snapshot.mkdir(parents=True, exist_ok=False)
    records = []
    before: dict[str, dict[str, Any]] = {}
    relatives = list(source_relatives(iteration=iteration))
    optional_residency = "image_residency_training-v1.json"
    if (model / optional_residency).is_file():
        relatives.append(optional_residency)
    for relative in relatives:
        source = model / relative
        source_identity = _identity(source, f"conversion source {relative}")
        before[relative] = source_identity
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        destination_identity = _identity(destination, f"conversion snapshot {relative}")
        if (source_identity["sha256"], source_identity["size_bytes"]) != (
            destination_identity["sha256"], destination_identity["size_bytes"]
        ):
            _fail(f"conversion snapshot identity differs: {relative}")
        records.append({"relative_path": relative, "source": source_identity, "destination": destination_identity})
    alias_source = model / "cameras_all_train.json"
    alias_destination = snapshot / "cameras_all.json"
    shutil.copy2(alias_source, alias_destination)
    alias = _identity(alias_destination, "conversion cameras_all alias")
    if alias["sha256"] != before["cameras_all_train.json"]["sha256"]:
        _fail("cameras_all alias is not byte-identical to cameras_all_train")
    after = {relative: _identity(model / relative, f"conversion source recheck {relative}") for relative in relatives}
    if {key: value["sha256"] for key, value in before.items()} != {key: value["sha256"] for key, value in after.items()}:
        _fail("conversion source mutated while creating snapshot")
    manifest = {
        "schema_version": "longsplat-conversion-snapshot-v2",
        "source_model": str(model),
        "destination_model": str(snapshot),
        "allowlist": [*relatives, "cameras_all.json"],
        "files": records,
        "alias": {
            "source": before["cameras_all_train.json"],
            "destination": alias,
            "byte_equal": True,
            "reason": "conversion eval-false entry",
            "training_view_count": authority.get("camera_count"),
        },
        "conversion_profile_id": authority["profile_id"],
        "camera_count": authority["camera_count"],
        "camera_order": authority["camera_order"],
        "source_video_sha256": authority["source_video_sha256"],
    }
    manifest_path = snapshot / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return snapshot, manifest_path, {"before": before, "after": after, "alias": alias}


def verify_snapshot(snapshot: Path, manifest_path: Path, *, allow_converted_output: bool) -> dict[str, Any]:
    manifest = _load_json(manifest_path, "conversion snapshot manifest")
    if manifest.get("schema_version") != "longsplat-conversion-snapshot-v2":
        _fail("conversion snapshot manifest schema differs")
    expected = {manifest_path.resolve()}
    for record in manifest.get("files", []):
        if not isinstance(record, Mapping):
            _fail("conversion snapshot record is malformed")
        relative = record.get("relative_path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            _fail("conversion snapshot relative path is unsafe")
        current = _identity(snapshot / relative, f"conversion snapshot {relative}")
        declared = record.get("destination")
        if current != declared:
            _fail(f"conversion snapshot identity drifted: {relative}")
        source = record.get("source")
        if not isinstance(source, Mapping):
            _fail(f"conversion snapshot source identity missing: {relative}")
        current_source = _identity(Path(str(source["path"])), f"conversion source {relative}")
        if current_source != source:
            _fail(f"conversion source changed after snapshot: {relative}")
        expected.add((snapshot / relative).resolve())
    alias_destination = _identity(snapshot / "cameras_all.json", "conversion cameras_all alias")
    if alias_destination != manifest.get("alias", {}).get("destination"):
        _fail("conversion cameras_all alias identity drifted")
    if alias_destination["sha256"] != manifest.get("alias", {}).get("source", {}).get("sha256"):
        _fail("conversion cameras_all alias is not byte-identical")
    expected.add((snapshot / "cameras_all.json").resolve())
    actual = {path.resolve() for path in snapshot.rglob("*") if path.is_file() and not path.is_symlink()}
    phase_evidence = {
        (snapshot / "image_residency_conversion-v1.json").resolve()
    } if (snapshot / "image_residency_conversion-v1.json").is_file() else set()
    unexpected = actual - expected - phase_evidence - ({path.resolve() for path in (snapshot / "converted_3dgs").rglob("*") if path.is_file()} if allow_converted_output and (snapshot / "converted_3dgs").is_dir() else set())
    if unexpected:
        _fail("conversion snapshot contains files outside the allowlist")
    return {"manifest_path": str(manifest_path), "manifest_sha256": _sha256(manifest_path), "allowlist_file_count": len(expected)}


def _build_request(
    *,
    authority: Mapping[str, Any],
    manifest_path: Path,
    paths: Mapping[str, Path],
    evidence_root: Path,
    snapshot: Path,
    command: Sequence[str],
    nested_command: Sequence[str],
) -> dict[str, Any]:
    profile = authority["profile"]
    return {
        "schema_version": "longsplat-generic-conversion-request-v1",
        "stage": "conversion",
        "authority_manifest": str(manifest_path),
        "authority_manifest_sha256": _sha256(manifest_path),
        "conversion_profile_id": authority["profile_id"],
        "conversion_profile": profile,
        "source_path": str(paths["source"]),
        "source_model_path": str(paths["model"]),
        "snapshot_model_path": str(snapshot),
        "camera_count": authority["camera_count"],
        "camera_order": authority["camera_order"],
        "camera_dimensions": authority["camera_dimensions"],
        "held_out": authority["status"]["held_out"],
        "argv": list(command),
        "nested_conversion_argv": list(nested_command),
        "shell": False,
        "cwd": str(paths["route"]),
        "environment": {
            "VENV_DIR": str(paths["backend_python"].parent.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "freshness": "evidence root, snapshot, conversion output, and record must be absent",
    }


def execute_conversion(
    *,
    authority_manifest_path: str | Path,
    evidence_root: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    dry_run: bool,
    validate_only: bool,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    route, _ = _route_outputs(route_root, containment_root=containment_root)
    try:
        authority = load_authority_manifest(
            authority_manifest_path,
            route_root=route,
            containment_root=containment_root,
        )
    except AuthorityManifestError as exc:
        _fail(str(exc))
    root = output_path(evidence_root, route, "conversion evidence root", containment_root=containment_root)
    if root.exists() or root.is_symlink():
        _fail(f"conversion evidence root must be fresh and append-only: {root}")
    paths = _manifest_paths(authority, route)
    manifest_path = Path(authority["manifest_path"] or authority_manifest_path).resolve()
    snapshot = root / "conversion_model_snapshot"
    command, nested_command = _build_conversion_argv(
        authority=authority,
        evidence_root=root,
        snapshot=snapshot,
        manifest_path=manifest_path,
        route=route,
    )
    request = _build_request(
        authority=authority,
        manifest_path=manifest_path,
        paths=paths,
        evidence_root=root,
        snapshot=snapshot,
        command=command,
        nested_command=nested_command,
    )
    if dry_run or validate_only:
        return {"validated": True, "stage": "conversion", "request": request, "command": command}
    root.mkdir(parents=True, exist_ok=False)
    snapshot, snapshot_manifest, details = _snapshot(authority=authority, evidence_root=root)
    request["snapshot_manifest_path"] = str(snapshot_manifest)
    request["snapshot_details"] = details
    (root / "request.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "argv.json").write_text(json.dumps({"argv": command, "nested_conversion_argv": nested_command, "shell": False, "shell_escaped": shlex.join(command)}, indent=2) + "\n", encoding="utf-8")
    env = _conversion_environment(paths)
    completed = runner(command, cwd=str(route), env=env, shell=False, capture_output=True, text=True, check=False)
    stdout = str(getattr(completed, "stdout", "") or "")
    stderr = str(getattr(completed, "stderr", "") or "")
    (root / "stdout.log").write_text(stdout, encoding="utf-8")
    (root / "stderr.log").write_text(stderr, encoding="utf-8")
    code = int(getattr(completed, "returncode", 1))
    result: dict[str, Any] = {
        "schema_version": "longsplat-generic-conversion-result-v1",
        "stage": "conversion",
        "exit_code": code,
        "status": "failed" if code else "technical_pass",
        "authority_manifest": str(manifest_path),
        "conversion_profile_id": authority["profile_id"],
        "camera_count": authority["camera_count"],
        "camera_order": authority["camera_order"],
        "held_out": authority["status"]["held_out"],
        "snapshot_model_path": str(snapshot),
        "snapshot_manifest_path": str(snapshot_manifest),
        "accepted": False,
        "supersplat": False,
    }
    if code == 0:
        try:
            from .convert import validate_converted_ply

            converted = snapshot / "converted_3dgs/point_cloud.ply"
            standard = validate_converted_ply(converted)
            image_residency = _validate_image_residency_telemetry(
                snapshot,
                authority=authority,
                required=(snapshot / "image_residency_training-v1.json").is_file(),
            )
            result.update({
                "structural_pass": True,
                "STRUCTURAL_CONVERSION_PASS": True,
                "structural": standard,
                "image_residency": image_residency,
            })
        except Exception as exc:
            result.update({"structural_pass": False, "STRUCTURAL_CONVERSION_PASS": False, "reason": str(exc)})
    else:
        result.update({"structural_pass": False, "STRUCTURAL_CONVERSION_PASS": False, "reason": "conversion child returned nonzero; no retry permitted"})
    (root / "conversion_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result


def _read_png(path: Path, dimensions: Mapping[str, Any]) -> Any:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        _fail(f"OpenCV/numpy are required for CPU evaluation postprocess: {exc}")
    if path.is_symlink() or not path.is_file():
        _fail(f"evaluation PNG is missing or symlinked: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        _fail(f"evaluation PNG cannot be decoded: {path}")
    expected = (int(dimensions["height"]), int(dimensions["width"]))
    if tuple(image.shape[:2]) != expected:
        _fail(f"evaluation PNG dimensions differ at {path}: {image.shape[:2]} != {expected}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if not np.all(np.isfinite(rgb)):
        _fail(f"evaluation PNG contains non-finite pixels: {path}")
    return rgb


def _metrics(left: Any, right: Any) -> dict[str, float | None]:
    import numpy as np

    difference = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    mse = float(np.mean(difference * difference))
    psnr = None if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(((a - ma) ** 2).mean()), float(((b - mb) ** 2).mean())
    covariance = float(((a - ma) * (b - mb)).mean())
    c1, c2 = 0.01**2, 0.03**2
    denominator = (ma * ma + mb * mb + c1) * (va + vb + c2)
    ssim = float(((2 * ma * mb + c1) * (2 * covariance + c2)) / denominator) if denominator else 0.0
    return {"mse": mse, "psnr_db": psnr, "ssim": ssim, "mean_pixel_diff": float(np.mean(np.abs(difference)))}


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    import numpy as np

    output: dict[str, float | None] = {}
    for field in ("mse", "psnr_db", "ssim", "mean_pixel_diff"):
        values = [float(item[field]) for item in records if item.get(field) is not None]
        output[field + "_mean"] = float(np.mean(values)) if values else None
        output[field + "_min"] = float(np.min(values)) if values else None
        output[field + "_max"] = float(np.max(values)) if values else None
    return output


def postprocess_evaluation(
    *,
    authority: Mapping[str, Any],
    eval_root: Path,
    snapshot: Path,
    converted_ply: Path,
) -> dict[str, Any]:
    evaluator_path = eval_root / "evaluator_result.json"
    evaluator = _load_json(evaluator_path, "converted evaluator result")
    count = authority["camera_count"]
    order = authority["camera_order"]
    if evaluator.get("camera_count") != count or evaluator.get("evaluated_count") != count:
        _fail("converted evaluator did not evaluate all training cameras")
    if evaluator.get("camera_indices") != list(range(count)):
        _fail("converted evaluator camera indices are not the exact ordered full set")
    per_view = evaluator.get("per_view")
    if not isinstance(per_view, list) or len(per_view) != count:
        _fail("converted evaluator per-view records do not match cameras_all_train")
    dimensions = authority["camera_dimensions"]
    rows = []
    converted_vs_gt = []
    native_vs_converted = []
    native_root_value = authority["native_render_evidence"].get("render_root")
    native_root = Path(str(native_root_value)).resolve()
    native_gt_root = native_root.parent / "gt"
    native_files = authority["native_render_evidence"].get("render_files")
    if not isinstance(native_files, list) or len(native_files) != count:
        # Older accepted B evidence used the canonical numeric render names;
        # new manifests record the ordered filenames explicitly.
        native_files = [f"{ordinal:05d}.png" for ordinal in range(count)]
    for record in per_view:
        if not isinstance(record, Mapping):
            _fail("converted evaluator per-view record is malformed")
        ordinal = record.get("ordinal")
        if not isinstance(ordinal, int) or not 0 <= ordinal < count:
            _fail("converted evaluator ordinal is invalid")
        if record.get("camera_index") != ordinal or Path(str(record.get("image_name", ""))).stem != Path(order[ordinal]).stem:
            _fail(f"converted evaluator camera identity differs at ordinal {ordinal}")
        render = Path(str(record.get("render_path", ""))).resolve()
        gt = Path(str(record.get("gt_path", ""))).resolve()
        try:
            render.relative_to(eval_root.resolve())
            gt.relative_to(eval_root.resolve())
        except ValueError:
            _fail("converted evaluator output escapes evaluation root")
        converted = _read_png(render, dimensions)
        target = _read_png(gt, dimensions)
        converted_metric = _metrics(converted, target)
        converted_vs_gt.append(converted_metric)
        row: dict[str, Any] = {
            "ordinal": ordinal,
            "image_name": order[ordinal],
            "converted_render_path": str(render),
            "converted_gt_path": str(gt),
            "converted_vs_gt": converted_metric,
        }
        native_render = native_root / str(native_files[ordinal])
        if native_render.is_file() and not native_render.is_symlink():
            native = _read_png(native_render, dimensions)
            native_metric = _metrics(native, converted)
            native_vs_converted.append(native_metric)
            row["native_render_path"] = str(native_render)
            row["native_vs_converted"] = native_metric
            native_gt = native_gt_root / str(native_files[ordinal])
            if native_gt.is_file() and not native_gt.is_symlink():
                row["native_vs_gt"] = _metrics(native, _read_png(native_gt, dimensions))
        rows.append(row)
    severe = any(float(item["mse"]) > 0.25 for item in converted_vs_gt)
    quality_advisories = []
    if severe:
        quality_advisories.append("converted-vs-GT MSE exceeded the visual advisory threshold; structural evaluation remains valid")
    return {
        "schema_version": "longsplat-generic-conversion-ab-v1",
        "STRUCTURAL_CONVERSION_PASS": True,
        "STRUCTURAL_EVALUATION_PASS": True,
        "SAME_CAMERA_VISUAL_PASS": "fail" if severe else "needs_review",
        "accepted": False,
        "supersplat": False,
        "held_out": authority["status"]["held_out"],
        "training_views_only": authority["status"]["training_views_only"],
        "camera_count": count,
        "camera_order": order,
        "camera_dimensions": dimensions,
        "converted_ply": _identity(converted_ply, "converted PLY"),
        "converted_vs_gt": _aggregate(converted_vs_gt),
        "native_vs_converted": _aggregate(native_vs_converted),
        "per_view": rows,
        "severe_degradation": severe,
        "quality_advisories": quality_advisories,
        "evaluator_result_path": str(evaluator_path),
        "evaluator_result_sha256": _sha256(evaluator_path),
    }


def execute_evaluation(
    *,
    authority_manifest_path: str | Path,
    evidence_root: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    conversion_evidence_root: str | Path | None = None,
    dry_run: bool,
    validate_only: bool,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    route, _ = _route_outputs(route_root, containment_root=containment_root)
    try:
        authority = load_authority_manifest(
            authority_manifest_path,
            route_root=route,
            containment_root=containment_root,
        )
    except AuthorityManifestError as exc:
        _fail(str(exc))
    root = output_path(evidence_root, route, "conversion evaluation evidence root", must_exist=True, containment_root=containment_root)
    # The conversion stage writes conversion_result.json and the model
    # snapshot under its OWN executor root.  The converted-eval evidence
    # root only holds the evaluation's own outputs (same_camera_eval,
    # evaluation_result.json), so the conversion evidence must be located
    # explicitly and independently of the eval root.
    if conversion_evidence_root is None:
        _fail("converted-eval requires the conversion evidence root (conversion stage executor root)")
    conversion_root = output_path(conversion_evidence_root, route, "conversion evidence root", must_exist=True, containment_root=containment_root)
    paths = _manifest_paths(authority, route)
    conversion_result = _load_json(conversion_root / "conversion_result.json", "conversion result")
    if conversion_result.get("STRUCTURAL_CONVERSION_PASS") is not True or conversion_result.get("structural_pass") is not True:
        _fail("converted evaluation requires a technical conversion pass")
    snapshot = (conversion_root / "conversion_model_snapshot").resolve()
    # Authoritative write-side key is structural.path (written by
    # convert.py validate_converted_ply); converted_ply_path is kept as a
    # compatibility fallback for older conversion result schemas.
    structural = conversion_result.get("structural", {})
    if not isinstance(structural, dict):
        structural = {}
    converted_value = structural.get("path") or structural.get("converted_ply_path")
    converted_ply = Path(str(converted_value)).resolve() if converted_value else snapshot / "converted_3dgs/point_cloud.ply"
    if converted_ply != snapshot / "converted_3dgs/point_cloud.ply":
        _fail("converted evaluator PLY is not the technical-pass output")
    eval_root = root / "same_camera_eval"
    if eval_root.is_symlink() or (eval_root.exists() and not (dry_run or validate_only)):
        _fail(f"same-camera evaluation root must be fresh: {eval_root}")
    manifest_path = Path(authority["manifest_path"] or authority_manifest_path).resolve()
    command = [
        str(route / "dev.sh"),
        "python",
        str(route / "scripts/longsplat/evaluate_converted_ply.py"),
        "--authority-manifest",
        str(manifest_path),
        *(["--containment-root", str(containment_root)] if containment_root is not None else []),
        "--ply",
        str(converted_ply),
        "--model-path",
        str(snapshot),
        "--source-path",
        str(authority["manifest"]["training_input"]["path"]),
        "--output",
        str(eval_root / "evaluator_result.json"),
        "--contact-sheet",
        str(eval_root / "evaluator_contact_sheet.png"),
        "--iteration",
        str(authority["profile"]["checkpoint_iteration"]),
    ]
    request = {
        "schema_version": "longsplat-generic-converted-evaluation-request-v1",
        "stage": "converted-eval",
        "authority_manifest": str(manifest_path),
        "authority_manifest_sha256": _sha256(manifest_path),
        "conversion_profile_id": authority["profile_id"],
        "source_path": str(paths["source"]),
        "snapshot_model_path": str(snapshot),
        "converted_ply_path": str(converted_ply),
        "camera_count": authority["camera_count"],
        "camera_order": authority["camera_order"],
        "camera_dimensions": authority["camera_dimensions"],
        "native_evidence_root": authority["native_render_evidence"]["render_root"],
        "held_out": authority["status"]["held_out"],
        "argv": command,
        "shell": False,
        "cwd": str(route),
    }
    if dry_run or validate_only:
        return {"validated": True, "stage": "converted-eval", "request": request, "command": command}
    eval_root.mkdir(parents=True, exist_ok=False)
    (eval_root / "request.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (eval_root / "argv.json").write_text(json.dumps({"argv": command, "shell": False, "shell_escaped": shlex.join(command)}, indent=2) + "\n", encoding="utf-8")
    completed = runner(
        command,
        cwd=str(route),
        env=_conversion_environment(paths),
        shell=False,
        capture_output=True,
        text=True,
        check=False,
    )
    (eval_root / "stdout.log").write_text(str(getattr(completed, "stdout", "") or ""), encoding="utf-8")
    (eval_root / "stderr.log").write_text(str(getattr(completed, "stderr", "") or ""), encoding="utf-8")
    code = int(getattr(completed, "returncode", 1))
    result: dict[str, Any] = {"schema_version": "longsplat-generic-converted-evaluation-v1", "stage": "converted-eval", "exit_code": code, "accepted": False, "supersplat": False, "held_out": authority["status"]["held_out"]}
    if code != 0:
        result.update({"SAME_CAMERA_VISUAL_PASS": "fail", "reason": "converted evaluator returned nonzero; no retry permitted"})
    else:
        try:
            ab = postprocess_evaluation(authority=authority, eval_root=eval_root, snapshot=snapshot, converted_ply=converted_ply)
            (eval_root / "ab_metrics.json").write_text(json.dumps(ab, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            result.update(ab)
        except ConversionExecutorBlocked as exc:
            result.update({"SAME_CAMERA_VISUAL_PASS": "fail", "reason": str(exc)})
    (root / "evaluation_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return result
