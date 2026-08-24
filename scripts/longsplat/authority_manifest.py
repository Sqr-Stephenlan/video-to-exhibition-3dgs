"""Immutable, per-run authority for conversion and evaluation.

The raw-video chain deliberately produces run-specific evidence.  Conversion
and evaluation consume that evidence through this manifest instead of carrying
paths, camera counts, image dimensions, or hashes from one historical run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .pipeline_contract import PipelineBlocked


SCHEMA_VERSION = "longsplat-authority-manifest-v1"

# This is a versioned *policy*, not evidence from a particular run.  Adding a
# profile requires an explicit code/test change; arbitrary parameter search is
# intentionally impossible through this module.
CONVERSION_PROFILES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "standard30000-v1": MappingProxyType(
            {
                "checkpoint_iteration": 30_000,
                "conversion_iterations": 30_000,
                "seed": 0,
                "prune_ratio": 0.6,
                "anisotropy_reg_weight": 0.01,
                "anisotropy_soft_limit": 30.0,
                "backend_mode": "research_local",
            }
        )
    }
)


class AuthorityManifestError(ValueError):
    """A manifest, profile, or bound artifact violates the run contract."""


def _fail(message: str) -> None:
    raise AuthorityManifestError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail(f"cannot hash authority artifact {path}: {exc}")
    return digest.hexdigest()


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{label}.path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        _fail(f"{label}.path must be absolute: {path}")
    return path


def _reject_symlink_components(path: Path, label: str) -> None:
    """Reject a symlink anywhere in an authority artifact's path."""

    probe = Path(path.anchor)
    for component in path.parts[1:]:
        probe /= component
        if probe.is_symlink():
            _fail(f"{label} traverses a symlink: {probe}")


def file_identity(path: str | Path, label: str, *, directory: bool = False) -> dict[str, Any]:
    """Return a fresh identity and reject symlinked or missing artifacts."""

    value = Path(path)
    if not value.is_absolute():
        _fail(f"{label} must be absolute: {value}")
    if value.is_symlink():
        _fail(f"{label} must not be a symlink: {value}")
    _reject_symlink_components(value, label)
    value = value.resolve(strict=False)
    if directory:
        if not value.is_dir():
            _fail(f"{label} directory is missing: {value}")
        return {"path": str(value)}
    if not value.is_file():
        _fail(f"{label} file is missing: {value}")
    return {"path": str(value), "sha256": _sha256(value), "size_bytes": value.stat().st_size}


def _declared_identity(value: Any, label: str, *, directory: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be an object")
    path = _absolute_path(value.get("path"), label)
    current = file_identity(path, label, directory=directory)
    if directory:
        return current
    for key in ("sha256", "size_bytes"):
        declared = value.get(key)
        if not isinstance(declared, (str, int)):
            _fail(f"{label}.{key} is required")
        if current[key] != declared:
            _fail(f"{label} identity drifted at {key}: expected {declared!r}, got {current[key]!r}")
    return current


def _assert_under_root(path: str | Path, root: Path, label: str) -> None:
    """Apply run-local containment without treating the route code root as data."""

    value = Path(path)
    root = root.resolve()
    if value.is_symlink():
        _fail(f"{label} must not be a symlink: {value}")
    _reject_symlink_components(value, label)
    resolved = value.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        _fail(f"{label} is outside dynamic run containment root: {resolved}")
    if not relative.parts:
        _fail(f"{label} cannot equal dynamic run containment root: {resolved}")


def conversion_profile(profile_id: str) -> dict[str, Any]:
    if not isinstance(profile_id, str) or profile_id not in CONVERSION_PROFILES:
        _fail(f"unknown conversion profile: {profile_id!r}")
    return dict(CONVERSION_PROFILES[profile_id])


def validate_profile(profile_id: str, parameters: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate a whitelisted profile and return its immutable parameters."""

    expected = conversion_profile(profile_id)
    if parameters is not None:
        for key, value in expected.items():
            if parameters.get(key) != value:
                _fail(f"conversion profile {profile_id} parameter {key} is not authorized")
        unknown = set(parameters) - set(expected)
        if unknown:
            _fail(f"conversion profile has unsupported parameters: {sorted(unknown)}")
    return expected


def _load_json(path: Path, label: str) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not valid JSON: {path}: {exc}")
    return value


def _camera_name(record: Mapping[str, Any], index: int, label: str) -> str:
    for key in ("image_name", "image", "name", "frame_id", "id"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
    _fail(f"{label}[{index}] has no camera identity")


def load_camera_file(path: str | Path, label: str) -> dict[str, Any]:
    """Read ordered camera identities and dimensions from one JSON array."""

    file_path = Path(path)
    value = _load_json(file_path, label)
    if not isinstance(value, list) or not value:
        if label.endswith("test") and value == []:
            return {"names": [], "count": 0, "dimensions": None}
        _fail(f"{label} must be a non-empty JSON array")
    names: list[str] = []
    dimensions: tuple[int, int] | None = None
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            _fail(f"{label}[{index}] is not an object")
        name = _camera_name(raw, index, label)
        if name in names:
            _fail(f"{label} contains duplicate camera identity: {name}")
        names.append(name)
        width, height = raw.get("width"), raw.get("height")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            _fail(f"{label}[{index}].width must be a positive integer")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            _fail(f"{label}[{index}].height must be a positive integer")
        current = (width, height)
        if dimensions is None:
            dimensions = current
        elif dimensions != current:
            _fail(f"{label} contains mixed camera dimensions")
    return {"names": names, "count": len(names), "dimensions": dimensions}


def _validate_native_images(native: Mapping[str, Any], expected_count: int, expected_dimensions: tuple[int, int]) -> None:
    render_root_value = native.get("render_root")
    if not isinstance(render_root_value, str):
        _fail("native_render_evidence.render_root is required")
    render_root = Path(render_root_value)
    if not render_root.is_absolute():
        _fail(f"native render root must be absolute: {render_root}")
    _reject_symlink_components(render_root, "native render root")
    render_root = render_root.resolve(strict=False)
    if not render_root.is_dir():
        _fail(f"native render root is missing or symlinked: {render_root}")
    renders = sorted(path for path in render_root.glob("*.png") if path.is_file() and not path.is_symlink())
    if len(renders) != expected_count:
        _fail(f"native render count differs from cameras_all_train: {len(renders)} != {expected_count}")
    declared_files = native.get("render_files")
    if declared_files is not None:
        actual_files = [path.name for path in renders]
        if declared_files != actual_files:
            _fail("native render file order differs from the declared evidence")
    declared_dimensions = native.get("dimensions")
    if not isinstance(declared_dimensions, Mapping):
        _fail("native_render_evidence.dimensions is required")
    actual_dimensions = (declared_dimensions.get("width"), declared_dimensions.get("height"))
    if actual_dimensions != expected_dimensions:
        _fail("native render dimensions differ from cameras_all_train")
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        for path in renders:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None or tuple(image.shape[:2]) != (expected_dimensions[1], expected_dimensions[0]):
                _fail(f"native render dimensions differ at {path}")


def _validate_status(status: Any) -> dict[str, Any]:
    if not isinstance(status, Mapping):
        _fail("status is required")
    result = dict(status)
    for key in ("training_views_only", "held_out", "accepted", "supersplat"):
        if not isinstance(result.get(key), bool):
            _fail(f"status.{key} must be boolean")
    if result["training_views_only"] != (not result["held_out"]):
        _fail("status.training_views_only and status.held_out disagree")
    return result


def validate_authority_manifest(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    route_root: str | Path | None = None,
    containment_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and re-hash every bound artifact in a run manifest."""

    if isinstance(manifest_or_path, Mapping):
        manifest = dict(manifest_or_path)
        manifest_path: Path | None = None
    else:
        manifest_path = Path(manifest_or_path).resolve()
        if manifest_path.is_symlink() or not manifest_path.is_file():
            _fail(f"authority manifest is missing or symlinked: {manifest_path}")
        raw = _load_json(manifest_path, "authority manifest")
        if not isinstance(raw, Mapping):
            _fail("authority manifest must be an object")
        manifest = dict(raw)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _fail(f"authority manifest schema must be {SCHEMA_VERSION}")
    profile_id = manifest.get("conversion_profile_id")
    profile = validate_profile(profile_id, manifest.get("conversion_profile"))
    provider = manifest.get("tool_provider")
    if provider is not None:
        try:
            from .tool_provider import verify_tool_provider

            verify_tool_provider(provider)
        except (AuthorityManifestError, PipelineBlocked, ValueError, TypeError) as exc:
            _fail(f"tool provider identity is invalid or drifted: {exc}")

    run_root: Path | None = None
    if containment_root is not None:
        run_root = Path(containment_root).resolve()
        if not run_root.is_dir() or run_root == Path(run_root.anchor):
            _fail(f"dynamic authority containment root is invalid: {run_root}")
        _reject_symlink_components(run_root, "dynamic authority containment root")
        if manifest_path is not None:
            _assert_under_root(manifest_path, run_root, "authority manifest")

    for field in ("source_video", "plan", "static_contract", "training_input", "training_model"):
        if field not in manifest:
            _fail(f"authority manifest missing {field}")
    _declared_identity(manifest["source_video"], "source_video")
    for field in ("plan", "static_contract"):
        identity = _declared_identity(manifest[field], field)
        if run_root is not None:
            _assert_under_root(identity["path"], run_root, field)
    training_input_identity = _declared_identity(manifest["training_input"], "training_input", directory=True)
    training_model_identity = _declared_identity(manifest["training_model"], "training_model", directory=True)
    if run_root is not None:
        _assert_under_root(training_input_identity["path"], run_root, "training_input")
        _assert_under_root(training_model_identity["path"], run_root, "training_model")

    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        _fail("checkpoint is required")
    iteration = checkpoint.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= 0:
        _fail("checkpoint.iteration must be a positive integer")
    if iteration != profile["checkpoint_iteration"]:
        _fail("checkpoint iteration does not match conversion profile")
    files = checkpoint.get("files")
    if not isinstance(files, Mapping):
        _fail("checkpoint.files is required")
    for key in ("point_cloud", "color_mlp", "cov_mlp", "opacity_mlp"):
        checkpoint_identity = _declared_identity(files.get(key), f"checkpoint.files.{key}")
        if run_root is not None:
            _assert_under_root(checkpoint_identity["path"], run_root, f"checkpoint.files.{key}")

    cameras = manifest.get("cameras")
    if not isinstance(cameras, Mapping):
        _fail("cameras is required")
    train = cameras.get("train")
    test = cameras.get("test")
    train_identity = _declared_identity(train, "cameras.train")
    test_identity = _declared_identity(test, "cameras.test")
    if run_root is not None:
        _assert_under_root(train_identity["path"], run_root, "cameras.train")
        _assert_under_root(test_identity["path"], run_root, "cameras.test")
    train_data = load_camera_file(train_identity["path"], "cameras_all_train")
    test_data = load_camera_file(test_identity["path"], "cameras_all_test")
    if train_data["count"] <= 0:
        _fail("cameras_all_train must not be empty")
    expected_names = train_data["names"]
    if cameras.get("count") != len(expected_names):
        _fail("cameras.count differs from cameras_all_train")
    if cameras.get("order") != expected_names:
        _fail("cameras.order differs from cameras_all_train")
    dimensions_value = cameras.get("dimensions")
    if not isinstance(dimensions_value, Mapping):
        _fail("cameras.dimensions is required")
    dimensions = (dimensions_value.get("width"), dimensions_value.get("height"))
    if dimensions != train_data["dimensions"]:
        _fail("cameras.dimensions differs from cameras_all_train")
    if cameras.get("test_order") != test_data["names"]:
        _fail("cameras.test_order differs from cameras_all_test")

    pose = manifest.get("pose_contract")
    pose_identity = _declared_identity(pose, "pose_contract")
    if run_root is not None:
        _assert_under_root(pose_identity["path"], run_root, "pose_contract")
    native = manifest.get("native_render_evidence")
    if not isinstance(native, Mapping):
        _fail("native_render_evidence is required")
    for field in ("result", "postcheck"):
        native_identity = _declared_identity(native.get(field), f"native_render_evidence.{field}")
        if run_root is not None:
            _assert_under_root(native_identity["path"], run_root, f"native_render_evidence.{field}")
    native_order = native.get("camera_order")
    if native_order != expected_names:
        _fail("native render camera order differs from cameras_all_train")
    if native.get("camera_count") != len(expected_names):
        _fail("native render camera count differs from cameras_all_train")
    if run_root is not None:
        render_root_value = native.get("render_root")
        if not isinstance(render_root_value, str):
            _fail("native_render_evidence.render_root is required")
        _assert_under_root(render_root_value, run_root, "native_render_evidence.render_root")
        render_files = native.get("render_files")
        if isinstance(render_files, list):
            for index, value in enumerate(render_files):
                if isinstance(value, str):
                    declared_file = Path(value)
                    if declared_file.is_absolute():
                        _assert_under_root(value, run_root, f"native_render_evidence.render_files[{index}]")
                    elif declared_file.name != value or value in {".", ".."}:
                        _fail(f"native_render_evidence.render_files[{index}] must be a basename")
    _validate_native_images(native, len(expected_names), dimensions)

    status = _validate_status(manifest.get("status"))
    if route_root is not None:
        root = Path(route_root).resolve()
        if not root.is_dir():
            _fail(f"route root is missing: {root}")
        for field in ("plan", "static_contract", "training_input", "training_model", "checkpoint", "cameras", "pose_contract"):
            _ = field
        # Output/evidence containment is checked by the executor with its
        # existing strict symlink-aware helper.  Here we only ensure the
        # manifest itself, when supplied as a path, belongs to this route.
        if manifest_path is not None and containment_root is None:
            try:
                manifest_path.relative_to(root)
            except ValueError:
                _fail("authority manifest is outside route root")

    return {
        "manifest": manifest,
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "profile": profile,
        "profile_id": profile_id,
        "camera_count": len(expected_names),
        "camera_order": list(expected_names),
        "camera_dimensions": {"width": dimensions[0], "height": dimensions[1]},
        "test_camera_order": list(test_data["names"]),
        "source_video_sha256": manifest["source_video"]["sha256"],
        "train_cameras": train_identity,
        "test_cameras": test_identity,
        "native_render_evidence": dict(native),
        "status": status,
    }


def load_authority_manifest(
    path: str | Path,
    *,
    route_root: str | Path | None = None,
    containment_root: str | Path | None = None,
) -> dict[str, Any]:
    return validate_authority_manifest(path, route_root=route_root, containment_root=containment_root)


def _identity_for_build(path: str | Path, label: str, *, directory: bool = False) -> dict[str, Any]:
    return file_identity(Path(path), label, directory=directory)


def build_authority_manifest(
    *,
    source_video: str | Path,
    plan: str | Path,
    static_contract: str | Path,
    training_input: str | Path,
    training_model: str | Path,
    pose_contract: str | Path,
    native_result: str | Path,
    native_postcheck: str | Path,
    native_render_root: str | Path,
    conversion_profile_id: str = "standard30000-v1",
    test_cameras: str | Path | None = None,
    status: Mapping[str, Any] | None = None,
    tool_provider: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest from current run artifacts and validate it immediately."""

    profile = conversion_profile(conversion_profile_id)
    model = Path(training_model)
    _reject_symlink_components(model, "training_model")
    model = model.resolve()
    iteration = profile["checkpoint_iteration"]
    train_path = model / "cameras_all_train.json"
    test_path = model / "cameras_all_test.json" if test_cameras is None else Path(test_cameras)
    train_data = load_camera_file(train_path, "cameras_all_train")
    test_data = load_camera_file(test_path, "cameras_all_test")
    dimensions = train_data["dimensions"]
    if dimensions is None:
        _fail("cameras_all_train has no dimensions")
    render_root = Path(native_render_root)
    if not render_root.is_absolute():
        _fail(f"native_render_root must be absolute: {render_root}")
    _reject_symlink_components(render_root, "native_render_root")
    render_files = sorted(path.name for path in render_root.glob("*.png") if path.is_file() and not path.is_symlink())
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "conversion_profile_id": conversion_profile_id,
        "conversion_profile": profile,
        "source_video": _identity_for_build(source_video, "source_video"),
        "plan": _identity_for_build(plan, "plan"),
        "static_contract": _identity_for_build(static_contract, "static_contract"),
        "training_input": _identity_for_build(training_input, "training_input", directory=True),
        "training_model": _identity_for_build(training_model, "training_model", directory=True),
        "checkpoint": {
            "iteration": iteration,
            "files": {
                "point_cloud": _identity_for_build(model / f"point_cloud/iteration_{iteration}/point_cloud.ply", "checkpoint point_cloud"),
                "color_mlp": _identity_for_build(model / f"point_cloud/iteration_{iteration}/color_mlp.pt", "checkpoint color_mlp"),
                "cov_mlp": _identity_for_build(model / f"point_cloud/iteration_{iteration}/cov_mlp.pt", "checkpoint cov_mlp"),
                "opacity_mlp": _identity_for_build(model / f"point_cloud/iteration_{iteration}/opacity_mlp.pt", "checkpoint opacity_mlp"),
            },
        },
        "cameras": {
            "train": _identity_for_build(train_path, "cameras.train"),
            "test": _identity_for_build(test_path, "cameras.test"),
            "count": train_data["count"],
            "order": train_data["names"],
            "test_order": test_data["names"],
            "dimensions": {"width": dimensions[0], "height": dimensions[1]},
        },
        "pose_contract": _identity_for_build(pose_contract, "pose_contract"),
        "tool_provider": None if tool_provider is None else dict(tool_provider),
        "native_render_evidence": {
            "result": _identity_for_build(native_result, "native result"),
            "postcheck": _identity_for_build(native_postcheck, "native postcheck"),
            "render_root": str(render_root.resolve()),
            "render_files": render_files,
            "camera_count": train_data["count"],
            "camera_order": train_data["names"],
            "dimensions": {"width": dimensions[0], "height": dimensions[1]},
        },
        "status": dict(status or {
            "training_views_only": True,
            "held_out": False,
            "accepted": False,
            "supersplat": False,
        }),
    }
    validate_authority_manifest(manifest)
    return manifest


def write_manifest_once(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    if destination.exists() or destination.is_symlink():
        _fail(f"authority manifest destination must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
