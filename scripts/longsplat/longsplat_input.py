"""CPU-only LongSplat training-input packaging contracts.

This module is deliberately independent of the nested LongSplat runtime.  It
rewrites the COLMAP *reference* model consumed by a future training smoke,
copies the already accepted camera-staging pixels, and proves that the text,
binary, and round-trip text models are internally identical.  It never
imports torch/CUDA and it never creates a PLY or a training artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .camera_staging import CameraParameters
from .colmap_contract import ColmapCommand, parse_colmap_image_pose_records, run_colmap_command
from .pipeline_contract import (
    PipelineBlocked,
    RUN_IDENTITY_SCHEMA,
    resolve_contained_path,
    resolve_containment_root,
    sha256_file,
    stable_sha256,
    write_json_once,
)


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_POSITIVE_INTEGER = re.compile(r"^[1-9][0-9]*$")
_CAMERA_PARAM_COUNTS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
}
_CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
}
_FUTURE_SMOKE_PLAN_SCHEMA = "longsplat-future-smoke-plan-v2"
_PARENT_BINDING_SCHEMA = "longsplat-parent-binding-v1"
_SAFE_STATE_SOURCE = "third_party/LongSplat/utils/general_utils.py::safe_state"
_SAFE_STATE_FILE = "third_party/LongSplat/utils/general_utils.py"
_TRAIN_ENTRYPOINT = "third_party/LongSplat/train.py"
_CAMERA_SAMPLING_TELEMETRY_SCHEMA = "camera-sampling-telemetry-v1"
_FUTURE_WORKLOAD_PROFILES: dict[str, dict[str, Any]] = {
    "convergence1000-v1": {
        "iterations": 1000,
        "render_iteration": 1000,
        "model_suffix": "-convergence-gpu-1000it-model",
    },
    "coverage-smoke-v1": {
        "iterations": None,
        "render_iteration": None,
        "iteration_source": "immutable coverage-smoke-plan-v1 requested_iterations",
        "model_suffix": "-coverage-gpu-model",
    },
    "smoke100-v1": {
        "iterations": 100,
        "render_iteration": 100,
        "model_suffix": "-future-gpu-100it-model",
    },
    "formal30000-v1": {
        "iterations": 30000,
        "render_iteration": 30000,
        "model_suffix": "-formal-gpu-30000-model",
    },
    # Compatibility aliases for prior raw-pipeline fixtures and CLI callers.
    "smoke100": {
        "iterations": 100,
        "render_iteration": 100,
        "model_suffix": "-future-gpu-100it-model",
    },
    "formal30000": {
        "iterations": 30000,
        "render_iteration": 30000,
        "model_suffix": "-formal-gpu-30000-model",
    },
}


class LongSplatInputBlocked(PipelineBlocked):
    """The training-specific static input contract cannot be proven."""


@dataclass
class ColmapModel:
    cameras: dict[int, dict[str, Any]]
    images: dict[int, dict[str, Any]]
    points3D: dict[int, dict[str, Any]]


def _fail(message: str) -> None:
    raise LongSplatInputBlocked(message)


def future_workload_profile(name: str) -> dict[str, Any]:
    """Return one explicitly allowlisted, non-searchable future workload."""

    try:
        profile = _FUTURE_WORKLOAD_PROFILES[name]
    except KeyError:
        _fail(f"unsupported future workload profile: {name!r}")
    return dict(profile)


def resolve_future_workload_profile(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve fixed profiles or the plan-bound dynamic coverage profile."""

    name = plan.get("workload_profile")
    if not isinstance(name, str):
        _fail("future smoke workload_profile is required")
    profile = future_workload_profile(name)
    if profile.get("iterations") is not None:
        return profile
    if name != "coverage-smoke-v1":
        _fail(f"future smoke profile has no iteration policy: {name!r}")
    binding = plan.get("coverage_plan")
    if not isinstance(binding, Mapping):
        _fail("coverage-smoke-v1 requires an immutable coverage plan binding")
    iterations = binding.get("iterations")
    camera_count = binding.get("active_camera_count")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        _fail("coverage-smoke-v1 binding iterations must be a positive integer")
    if isinstance(camera_count, bool) or not isinstance(camera_count, int) or camera_count <= 0:
        _fail("coverage-smoke-v1 binding camera count must be a positive integer")
    resolved = dict(profile)
    resolved.update(
        {
            "iterations": iterations,
            "render_iteration": iterations,
            "active_camera_count": camera_count,
        }
    )
    return resolved


def _positive_integer(value: str, *, field: str, path: Path, line_number: int) -> int:
    if not _POSITIVE_INTEGER.fullmatch(value):
        _fail(f"{path}:{line_number}: {field} must be a positive integer")
    return int(value)


def _integer(value: str, *, field: str, path: Path, line_number: int) -> int:
    if not _INTEGER.fullmatch(value):
        _fail(f"{path}:{line_number}: {field} must be an integer")
    return int(value)


def _finite(value: str, *, field: str, path: Path, line_number: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _fail(f"{path}:{line_number}: {field} must be numeric")
    if not math.isfinite(number):
        _fail(f"{path}:{line_number}: {field} must be finite")
    return number


def _camera_line(record: Mapping[str, Any], *, path: Path, line_number: int) -> dict[str, Any]:
    fields = list(record.get("fields", []))
    if len(fields) < 4:
        _fail(f"{path}:{line_number}: malformed camera record")
    camera_id = _positive_integer(fields[0], field="camera_id", path=path, line_number=line_number)
    model = fields[1]
    if model not in _CAMERA_PARAM_COUNTS:
        _fail(f"{path}:{line_number}: unsupported camera model {model}")
    parameter_count = _CAMERA_PARAM_COUNTS[model]
    if len(fields) != 4 + parameter_count:
        _fail(f"{path}:{line_number}: camera token count does not match {model}")
    width = _positive_integer(fields[2], field="camera width", path=path, line_number=line_number)
    height = _positive_integer(fields[3], field="camera height", path=path, line_number=line_number)
    params = [
        _finite(value, field=f"camera parameter {index}", path=path, line_number=line_number)
        for index, value in enumerate(fields[4:])
    ]
    return {
        "camera_id": camera_id,
        "model": model,
        "width": width,
        "height": height,
        "params": params,
    }


def parse_colmap_cameras_text(path: str | Path) -> dict[int, dict[str, Any]]:
    """Strictly parse the camera records used by the packaging contract."""

    source = Path(path).resolve()
    if not source.is_file():
        _fail(f"COLMAP cameras.txt is missing: {source}")
    cameras: dict[int, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        text = raw_line.strip()
        if not text or text.startswith("#"):
            continue
        record = _camera_line({"fields": text.split()}, path=source, line_number=line_number)
        camera_id = int(record["camera_id"])
        if camera_id in cameras:
            _fail(f"{source}:{line_number}: duplicate camera id {camera_id}")
        cameras[camera_id] = record
    if not cameras:
        _fail(f"COLMAP cameras.txt has no camera records: {source}")
    return cameras


def _parse_image_header(fields: Sequence[str], *, path: Path, line_number: int) -> dict[str, Any]:
    # A header is exactly ten tokens.  In particular, a long POINTS2D line is
    # never eligible for this parser, even when split(maxsplit=9) would make it
    # appear header-like.
    if len(fields) != 10:
        _fail(
            f"{path}:{line_number}: image header must have exactly 10 tokens; "
            "POINTS2D rows are not image headers"
        )
    image_id = _positive_integer(fields[0], field="image_id", path=path, line_number=line_number)
    qvec = tuple(
        _finite(value, field=f"qvec[{index}]", path=path, line_number=line_number)
        for index, value in enumerate(fields[1:5])
    )
    tvec = tuple(
        _finite(value, field=f"tvec[{index}]", path=path, line_number=line_number)
        for index, value in enumerate(fields[5:8])
    )
    camera_id = _positive_integer(fields[8], field="camera_id", path=path, line_number=line_number)
    name = fields[9]
    if not name or Path(name).name != name or any(character.isspace() for character in name):
        _fail(f"{path}:{line_number}: image NAME must be a basename without whitespace")
    if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
        _fail(f"{path}:{line_number}: image NAME must have an image extension")
    return {
        "image_id": image_id,
        "qvec": qvec,
        "tvec": tvec,
        "camera_id": camera_id,
        "name": name,
        "points2D": [],
    }


def _parse_points2d_line(raw_line: str, *, path: Path, line_number: int) -> list[dict[str, Any]]:
    fields = raw_line.strip().split()
    if len(fields) % 3 != 0:
        _fail(f"{path}:{line_number}: POINTS2D row must contain triples")
    points: list[dict[str, Any]] = []
    for offset in range(0, len(fields), 3):
        x = _finite(fields[offset], field=f"POINTS2D[{offset // 3}].x", path=path, line_number=line_number)
        y = _finite(fields[offset + 1], field=f"POINTS2D[{offset // 3}].y", path=path, line_number=line_number)
        point_id = _integer(
            fields[offset + 2],
            field=f"POINTS2D[{offset // 3}].point3D_id",
            path=path,
            line_number=line_number,
        )
        if point_id < -1:
            _fail(f"{path}:{line_number}: POINTS2D point3D_id must be -1 or positive")
        points.append({"x": x, "y": y, "point3D_id": point_id})
    return points


def parse_colmap_images_text(path: str | Path) -> dict[int, dict[str, Any]]:
    """Parse COLMAP's two-line image representation with strict state."""

    source = Path(path).resolve()
    if not source.is_file():
        _fail(f"COLMAP images.txt is missing: {source}")
    # Share the header/POINTS2D state contract with the raw COLMAP evidence
    # parser before decoding the full point triples below.  The second parser
    # remains responsible for LongSplat's point observations, but neither
    # path can silently reinterpret a missing points row as the next header.
    parse_colmap_image_pose_records(source)
    lines = source.read_text(encoding="utf-8").splitlines()
    images: dict[int, dict[str, Any]] = {}
    names: set[str] = set()
    index = 0
    while index < len(lines):
        raw_header = lines[index]
        text = raw_header.strip()
        if not text or text.startswith("#"):
            index += 1
            continue
        header_line = index + 1
        image = _parse_image_header(text.split(), path=source, line_number=header_line)
        image_id = int(image["image_id"])
        name = str(image["name"])
        if image_id in images:
            _fail(f"{source}:{header_line}: duplicate image id {image_id}")
        if name in names:
            _fail(f"{source}:{header_line}: duplicate image NAME {name}")
        images[image_id] = image
        names.add(name)
        index += 1
        if index >= len(lines):
            _fail(f"{source}:{header_line}: missing POINTS2D row")
        # The empty row is meaningful: it is the valid zero-observation form.
        if lines[index].lstrip().startswith("#"):
            _fail(f"{source}:{index + 1}: comment cannot replace a POINTS2D row")
        image["points2D"] = _parse_points2d_line(lines[index], path=source, line_number=index + 1)
        index += 1
    if not images:
        _fail(f"COLMAP images.txt has no image headers: {source}")
    return images


def parse_colmap_points3d_text(path: str | Path) -> dict[int, dict[str, Any]]:
    """Strictly parse points and their image/POINT2D_IDX tracks."""

    source = Path(path).resolve()
    if not source.is_file():
        _fail(f"COLMAP points3D.txt is missing: {source}")
    points: dict[int, dict[str, Any]] = {}
    track_pairs: set[tuple[int, int]] = set()
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        text = raw_line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) < 8 or (len(fields) - 8) % 2:
            _fail(f"{source}:{line_number}: malformed points3D record/track")
        point_id = _positive_integer(fields[0], field="point3D_id", path=source, line_number=line_number)
        if point_id in points:
            _fail(f"{source}:{line_number}: duplicate point3D id {point_id}")
        xyz = tuple(
            _finite(value, field=f"XYZ[{index}]", path=source, line_number=line_number)
            for index, value in enumerate(fields[1:4])
        )
        rgb: list[int] = []
        for index, value in enumerate(fields[4:7]):
            channel = _integer(value, field=f"RGB[{index}]", path=source, line_number=line_number)
            if channel < 0 or channel > 255:
                _fail(f"{source}:{line_number}: RGB channel is out of range")
            rgb.append(channel)
        error = _finite(fields[7], field="reprojection error", path=source, line_number=line_number)
        track: list[tuple[int, int]] = []
        for offset in range(8, len(fields), 2):
            image_id = _positive_integer(fields[offset], field="track image_id", path=source, line_number=line_number)
            point2d_idx = _integer(fields[offset + 1], field="track POINT2D_IDX", path=source, line_number=line_number)
            if point2d_idx < 0:
                _fail(f"{source}:{line_number}: track POINT2D_IDX must be non-negative")
            pair = (image_id, point2d_idx)
            if pair in track_pairs:
                _fail(f"{source}:{line_number}: duplicate track pair {pair}")
            track_pairs.add(pair)
            track.append(pair)
        points[point_id] = {
            "point3D_id": point_id,
            "xyz": xyz,
            "rgb": tuple(rgb),
            "error": error,
            "track": track,
        }
    return points


def parse_colmap_model_text(model_dir: str | Path) -> ColmapModel:
    root = Path(model_dir).resolve()
    model = ColmapModel(
        cameras=parse_colmap_cameras_text(root / "cameras.txt"),
        images=parse_colmap_images_text(root / "images.txt"),
        points3D=parse_colmap_points3d_text(root / "points3D.txt"),
    )
    validate_colmap_model(model)
    return model


def _read_exact(handle: Any, size: int, *, path: Path, field: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        _fail(f"{path}: truncated binary {field}")
    return data


def _read_struct(handle: Any, fmt: str, *, path: Path, field: str) -> tuple[Any, ...]:
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, _read_exact(handle, size, path=path, field=field))


def _read_c_string(handle: Any, *, path: Path) -> str:
    payload = bytearray()
    while True:
        value = _read_exact(handle, 1, path=path, field="image name")[0]
        if value == 0:
            break
        payload.append(value)
        if len(payload) > 4096:
            _fail(f"{path}: binary image name is unreasonably long")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"{path}: binary image name is not UTF-8")
        raise AssertionError from exc


def parse_colmap_cameras_binary(path: str | Path) -> dict[int, dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        _fail(f"COLMAP cameras.bin is missing: {source}")
    cameras: dict[int, dict[str, Any]] = {}
    with source.open("rb") as handle:
        count = _read_struct(handle, "<Q", path=source, field="camera count")[0]
        for _ in range(count):
            camera_id, model_id = _read_struct(handle, "<ii", path=source, field="camera header")
            if model_id not in _CAMERA_MODEL_IDS:
                _fail(f"{source}: unsupported binary camera model id {model_id}")
            model, parameter_count = _CAMERA_MODEL_IDS[model_id]
            width, height = _read_struct(handle, "<QQ", path=source, field="camera dimensions")
            params = _read_struct(handle, "<" + "d" * parameter_count, path=source, field="camera params")
            if camera_id <= 0 or camera_id in cameras or width <= 0 or height <= 0:
                _fail(f"{source}: malformed or duplicate binary camera identity")
            if not all(math.isfinite(value) for value in params):
                _fail(f"{source}: binary camera has non-finite params")
            cameras[int(camera_id)] = {
                "camera_id": int(camera_id),
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": [float(value) for value in params],
            }
        if handle.read(1):
            _fail(f"{source}: trailing bytes after cameras.bin records")
    if not cameras:
        _fail(f"COLMAP cameras.bin has no records: {source}")
    return cameras


def parse_colmap_images_binary(path: str | Path) -> dict[int, dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        _fail(f"COLMAP images.bin is missing: {source}")
    images: dict[int, dict[str, Any]] = {}
    names: set[str] = set()
    with source.open("rb") as handle:
        count = _read_struct(handle, "<Q", path=source, field="image count")[0]
        for _ in range(count):
            values = _read_struct(handle, "<idddddddi", path=source, field="image header")
            image_id = int(values[0])
            qvec = tuple(float(value) for value in values[1:5])
            tvec = tuple(float(value) for value in values[5:8])
            camera_id = int(values[8])
            if image_id <= 0 or image_id in images or camera_id <= 0:
                _fail(f"{source}: malformed or duplicate binary image identity")
            if not all(math.isfinite(value) for value in (*qvec, *tvec)):
                _fail(f"{source}: binary image has non-finite pose")
            name = _read_c_string(handle, path=source)
            if not name or Path(name).name != name or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                _fail(f"{source}: binary image name is not a staged basename")
            if name in names:
                _fail(f"{source}: duplicate binary image name {name}")
            point_count = _read_struct(handle, "<Q", path=source, field="POINTS2D count")[0]
            points = []
            for _ in range(point_count):
                x, y, point_id = _read_struct(handle, "<ddq", path=source, field="POINTS2D record")
                if not all(math.isfinite(value) for value in (x, y)) or point_id < -1:
                    _fail(f"{source}: malformed binary POINTS2D record")
                points.append({"x": float(x), "y": float(y), "point3D_id": int(point_id)})
            images[image_id] = {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "points2D": points,
            }
            names.add(name)
        if handle.read(1):
            _fail(f"{source}: trailing bytes after images.bin records")
    return images


def parse_colmap_points3d_binary(path: str | Path) -> dict[int, dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        _fail(f"COLMAP points3D.bin is missing: {source}")
    points: dict[int, dict[str, Any]] = {}
    track_pairs: set[tuple[int, int]] = set()
    with source.open("rb") as handle:
        count = _read_struct(handle, "<Q", path=source, field="point count")[0]
        for _ in range(count):
            point_id, x, y, z, red, green, blue, error = _read_struct(
                handle, "<QdddBBBd", path=source, field="point record"
            )
            if point_id <= 0 or point_id in points or not all(math.isfinite(value) for value in (x, y, z, error)):
                _fail(f"{source}: malformed or duplicate binary point")
            track_count = _read_struct(handle, "<Q", path=source, field="track count")[0]
            track = []
            for _ in range(track_count):
                image_id, point2d_idx = _read_struct(handle, "<ii", path=source, field="track record")
                if image_id <= 0 or point2d_idx < 0 or (image_id, point2d_idx) in track_pairs:
                    _fail(f"{source}: malformed or duplicate binary track")
                track_pairs.add((image_id, point2d_idx))
                track.append((int(image_id), int(point2d_idx)))
            points[int(point_id)] = {
                "point3D_id": int(point_id),
                "xyz": (float(x), float(y), float(z)),
                "rgb": (int(red), int(green), int(blue)),
                "error": float(error),
                "track": track,
            }
        if handle.read(1):
            _fail(f"{source}: trailing bytes after points3D.bin records")
    return points


def parse_colmap_model_binary(model_dir: str | Path) -> ColmapModel:
    root = Path(model_dir).resolve()
    model = ColmapModel(
        cameras=parse_colmap_cameras_binary(root / "cameras.bin"),
        images=parse_colmap_images_binary(root / "images.bin"),
        points3D=parse_colmap_points3d_binary(root / "points3D.bin"),
    )
    validate_colmap_model(model)
    return model


def validate_colmap_model(
    model: ColmapModel,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
    require_pinhole: bool = False,
) -> None:
    """Validate IDs, poses, observations, and track referential integrity."""

    if not model.cameras:
        _fail("COLMAP model has no cameras")
    if not model.images:
        _fail("COLMAP model has no images")
    camera_ids = set(model.cameras)
    names: set[str] = set()
    for camera_id, camera in model.cameras.items():
        if int(camera_id) <= 0 or int(camera.get("camera_id", -1)) != int(camera_id):
            _fail("COLMAP camera identity is invalid")
        if int(camera["width"]) <= 0 or int(camera["height"]) <= 0:
            _fail("COLMAP camera dimensions are invalid")
        if require_pinhole and camera["model"] != "PINHOLE":
            _fail(f"training input camera must be PINHOLE, got {camera['model']}")
        if not all(math.isfinite(float(value)) for value in camera["params"]):
            _fail("COLMAP camera params must be finite")
    for image_id, image in model.images.items():
        if int(image_id) <= 0 or int(image.get("image_id", -1)) != int(image_id):
            _fail("COLMAP image identity is invalid")
        if int(image["camera_id"]) not in camera_ids:
            _fail(f"COLMAP image {image_id} references missing camera {image['camera_id']}")
        name = str(image["name"])
        if not name or Path(name).name != name or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            _fail(f"COLMAP image {image_id} has an invalid staged name")
        if name in names:
            _fail(f"COLMAP image names are duplicated: {name}")
        names.add(name)
        if len(image["qvec"]) != 4 or len(image["tvec"]) != 3 or not all(
            math.isfinite(float(value)) for value in (*image["qvec"], *image["tvec"])
        ):
            _fail(f"COLMAP image {image_id} pose is not finite")
        for point in image["points2D"]:
            if not all(math.isfinite(float(point[key])) for key in ("x", "y")):
                _fail(f"COLMAP image {image_id} has a non-finite POINTS2D observation")
            if image_width is not None and image_height is not None:
                if not (0.0 <= float(point["x"]) < image_width and 0.0 <= float(point["y"]) < image_height):
                    _fail(f"COLMAP image {image_id} has an out-of-bounds POINTS2D observation")
    track_pairs: set[tuple[int, int]] = set()
    for point_id, point in model.points3D.items():
        if int(point_id) <= 0 or int(point.get("point3D_id", -1)) != int(point_id):
            _fail("COLMAP point identity is invalid")
        if not all(math.isfinite(float(value)) for value in (*point["xyz"], point["error"])):
            _fail(f"COLMAP point {point_id} is non-finite")
        for image_id, point2d_idx in point["track"]:
            pair = (int(image_id), int(point2d_idx))
            if pair in track_pairs:
                _fail(f"COLMAP duplicate track pair {pair}")
            track_pairs.add(pair)
            if image_id not in model.images:
                _fail(f"COLMAP point {point_id} has dangling image track {image_id}")
            observations = model.images[image_id]["points2D"]
            if point2d_idx < 0 or point2d_idx >= len(observations):
                _fail(f"COLMAP point {point_id} has dangling POINT2D_IDX {pair}")
            if int(observations[point2d_idx]["point3D_id"]) != int(point_id):
                _fail(f"COLMAP point {point_id} track disagrees with image observation {pair}")
    for image_id, image in model.images.items():
        for point2d_idx, observation in enumerate(image["points2D"]):
            point_id = int(observation["point3D_id"])
            if point_id == -1:
                continue
            if point_id not in model.points3D:
                _fail(f"COLMAP image observation references missing point {point_id}")
            if (image_id, point2d_idx) not in track_pairs:
                _fail(f"COLMAP image observation has no reciprocal track {(image_id, point2d_idx)}")


def _format_float(value: float) -> str:
    return format(float(value), ".17g")


def _write_text_once_or_verify(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            _fail(f"immutable text artifact differs on resume: {path}")
        return
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def write_colmap_model_text(model: ColmapModel, model_dir: str | Path) -> None:
    root = Path(model_dir).resolve()
    validate_colmap_model(model)
    cameras = ["# Camera list with one line of data per camera:", "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    cameras.extend(
        "{} {} {} {} {}".format(
            camera_id,
            camera["model"],
            camera["width"],
            camera["height"],
            " ".join(_format_float(value) for value in camera["params"]),
        )
        for camera_id, camera in sorted(model.cameras.items())
    )
    images = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for image_id, image in sorted(model.images.items()):
        images.append(
            "{} {} {} {} {} {} {} {} {} {}".format(
                image_id,
                *(_format_float(value) for value in image["qvec"]),
                *(_format_float(value) for value in image["tvec"]),
                image["camera_id"],
                image["name"],
            )
        )
        images.append(
            " ".join(
                f"{_format_float(point['x'])} {_format_float(point['y'])} {int(point['point3D_id'])}"
                for point in image["points2D"]
            )
        )
    points = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for point_id, point in sorted(model.points3D.items()):
        track = " ".join(f"{image_id} {point2d_idx}" for image_id, point2d_idx in point["track"])
        values = [
            str(point_id),
            *(_format_float(value) for value in point["xyz"]),
            *(str(int(value)) for value in point["rgb"]),
            _format_float(point["error"]),
            track,
        ]
        points.append(" ".join(value for value in values if value != ""))
    _write_text_once_or_verify(root / "cameras.txt", "\n".join(cameras) + "\n")
    _write_text_once_or_verify(root / "images.txt", "\n".join(images) + "\n")
    _write_text_once_or_verify(root / "points3D.txt", "\n".join(points) + "\n")


def _validate_crop(crop: Mapping[str, Any], *, input_width: int, input_height: int) -> dict[str, int]:
    values = {}
    for key in ("x0", "y0", "width", "height"):
        value = crop.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(f"crop {key} must be an integer")
        values[key] = int(value)
    if values["x0"] < 0 or values["y0"] < 0 or values["width"] <= 0 or values["height"] <= 0:
        _fail("crop ROI must have non-negative origin and positive size")
    if values["x0"] + values["width"] > input_width or values["y0"] + values["height"] > input_height:
        _fail("crop ROI lies outside the input camera dimensions")
    return values


def _camera_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    model = str(record.get("model"))
    if model != "PINHOLE":
        _fail(f"training input camera must be PINHOLE, got {model}")
    params = record.get("params")
    if not isinstance(params, (list, tuple)) or len(params) != 4:
        _fail("PINHOLE camera requires four params")
    values = [float(item) for item in params]
    if not all(math.isfinite(value) for value in values) or values[0] <= 0 or values[1] <= 0:
        _fail("PINHOLE camera params must be finite and positive in focal length")
    return {
        "model": "PINHOLE",
        "width": int(record["width"]),
        "height": int(record["height"]),
        "params": values,
    }


def _final_camera_from_contract(
    source_camera: Mapping[str, Any],
    crop: Mapping[str, int],
    final_camera: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = _camera_from_record(source_camera)
    expected = {
        "model": "PINHOLE",
        "width": crop["width"],
        "height": crop["height"],
        "params": [
            source["params"][0],
            source["params"][1],
            source["params"][2] - crop["x0"],
            source["params"][3] - crop["y0"],
        ],
    }
    if final_camera is None:
        return expected
    model = str(final_camera.get("model"))
    values = [
        float(final_camera.get(key))
        for key in ("fx", "fy", "cx", "cy")
    ]
    actual = {
        "model": model,
        "width": int(final_camera.get("width")),
        "height": int(final_camera.get("height")),
        "params": values,
    }
    if actual["model"] != expected["model"] or (actual["width"], actual["height"]) != (expected["width"], expected["height"]):
        _fail("camera contract final dimensions/model disagree with crop")
    if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8) for a, b in zip(actual["params"], expected["params"])):
        _fail("camera contract final K disagrees with crop affine")
    return actual


def rewrite_colmap_model_for_crop(
    model: ColmapModel,
    *,
    crop: Mapping[str, Any],
    final_camera: Mapping[str, Any] | None = None,
) -> tuple[ColmapModel, dict[str, Any]]:
    """Rewrite a sparse model for an integer crop, including all tracks."""

    validate_colmap_model(model)
    if len(model.cameras) != 1:
        _fail(f"training-specific model requires exactly one source camera, found {len(model.cameras)}")
    source_camera_id, source_camera = next(iter(sorted(model.cameras.items())))
    if source_camera["model"] != "PINHOLE":
        _fail("undistorted source model must contain a PINHOLE camera")
    crop_values = _validate_crop(
        crop,
        input_width=int(source_camera["width"]),
        input_height=int(source_camera["height"]),
    )
    output_camera = _final_camera_from_contract(source_camera, crop_values, final_camera)
    output_camera["camera_id"] = source_camera_id

    output_images: dict[int, dict[str, Any]] = {}
    observation_index: dict[tuple[int, int], int] = {}
    source_observations = 0
    source_tracked_observations = 0
    output_observations = 0
    output_tracked_observations = 0
    dropped_observations = 0
    dropped_tracked_observations = 0
    for image_id, image in sorted(model.images.items()):
        points: list[dict[str, Any]] = []
        for old_index, point in enumerate(image["points2D"]):
            source_observations += 1
            point_id = int(point["point3D_id"])
            if point_id != -1:
                source_tracked_observations += 1
            inside = (
                crop_values["x0"] <= float(point["x"]) < crop_values["x0"] + crop_values["width"]
                and crop_values["y0"] <= float(point["y"]) < crop_values["y0"] + crop_values["height"]
            )
            if not inside:
                dropped_observations += 1
                if point_id != -1:
                    dropped_tracked_observations += 1
                continue
            new_index = len(points)
            transformed = {
                "x": float(point["x"]) - crop_values["x0"],
                "y": float(point["y"]) - crop_values["y0"],
                "point3D_id": point_id,
            }
            if not (0.0 <= transformed["x"] < crop_values["width"] and 0.0 <= transformed["y"] < crop_values["height"]):
                _fail(f"crop transform produced an out-of-bounds observation {(image_id, old_index)}")
            points.append(transformed)
            observation_index[(image_id, old_index)] = new_index
        output_observations += len(points)
        output_tracked_observations += sum(int(point["point3D_id"]) != -1 for point in points)
        output_images[image_id] = {
            "image_id": int(image_id),
            "qvec": tuple(image["qvec"]),
            "tvec": tuple(image["tvec"]),
            "camera_id": source_camera_id,
            "name": image["name"],
            "points2D": points,
        }

    output_points: dict[int, dict[str, Any]] = {}
    points_dropped_no_valid_track = 0
    tracks_before = 0
    tracks_after = 0
    for point_id, point in sorted(model.points3D.items()):
        tracks_before += len(point["track"])
        filtered_track: list[tuple[int, int]] = []
        for image_id, old_index in point["track"]:
            key = (int(image_id), int(old_index))
            if key not in observation_index:
                continue
            new_index = observation_index[key]
            if int(output_images[image_id]["points2D"][new_index]["point3D_id"]) != int(point_id):
                _fail(f"rewritten track disagrees with POINTS2D for point {point_id}")
            filtered_track.append((int(image_id), int(new_index)))
        if not filtered_track:
            points_dropped_no_valid_track += 1
            continue
        tracks_after += len(filtered_track)
        output_points[int(point_id)] = {
            "point3D_id": int(point_id),
            "xyz": tuple(point["xyz"]),
            "rgb": tuple(point["rgb"]),
            "error": float(point["error"]),
            "track": filtered_track,
        }

    # A point can only disappear through the explicit no-valid-track rule; a
    # retained POINTS2D observation must have its point and reciprocal track.
    for image in output_images.values():
        for point in image["points2D"]:
            if int(point["point3D_id"]) != -1 and int(point["point3D_id"]) not in output_points:
                _fail("rewritten image retains an observation for a dropped point")
    output_model = ColmapModel(
        cameras={source_camera_id: output_camera},
        images=output_images,
        points3D=output_points,
    )
    validate_colmap_model(
        output_model,
        image_width=crop_values["width"],
        image_height=crop_values["height"],
        require_pinhole=True,
    )
    stats = {
        "source_camera_count": len(model.cameras),
        "output_camera_count": len(output_model.cameras),
        "source_image_count": len(model.images),
        "output_image_count": len(output_model.images),
        "source_points3D_count": len(model.points3D),
        "output_points3D_count": len(output_model.points3D),
        "source_observation_count": source_observations,
        "source_tracked_observation_count": source_tracked_observations,
        "output_observation_count": output_observations,
        "output_tracked_observation_count": output_tracked_observations,
        "tracks_before": tracks_before,
        "tracks_after": tracks_after,
        "dropped_observation_count": dropped_observations,
        "dropped_tracked_observation_count": dropped_tracked_observations,
        "points_dropped_no_valid_track": points_dropped_no_valid_track,
        "drop_rule": "drop POINTS2D outside integer ROI; drop points3D with no remaining reciprocal track",
        "observation_rewrite": True,
        "crop": crop_values,
    }
    return output_model, stats


def model_counts(model: ColmapModel) -> dict[str, int]:
    observations = sum(len(image["points2D"]) for image in model.images.values())
    tracked = sum(int(point["point3D_id"]) != -1 for image in model.images.values() for point in image["points2D"])
    tracks = sum(len(point["track"]) for point in model.points3D.values())
    return {
        "cameras": len(model.cameras),
        "images": len(model.images),
        "observations": observations,
        "tracked_observations": tracked,
        "points3D": len(model.points3D),
        "tracks": tracks,
    }


def _max_abs(values: Sequence[float]) -> float:
    return max((abs(float(value)) for value in values), default=0.0)


def compare_colmap_models(expected: ColmapModel, actual: ColmapModel, *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Compare model identities with strict pose/geometry/track tolerances."""

    if set(expected.cameras) != set(actual.cameras):
        _fail("COLMAP camera IDs differ across TXT/BIN/round-trip")
    if set(expected.images) != set(actual.images):
        _fail("COLMAP image IDs differ across TXT/BIN/round-trip")
    if set(expected.points3D) != set(actual.points3D):
        _fail("COLMAP point IDs differ across TXT/BIN/round-trip")
    camera_max = 0.0
    for camera_id in expected.cameras:
        left, right = expected.cameras[camera_id], actual.cameras[camera_id]
        if left["model"] != right["model"] or (left["width"], left["height"]) != (right["width"], right["height"]):
            _fail(f"COLMAP camera {camera_id} identity differs")
        camera_max = max(camera_max, _max_abs([a - b for a, b in zip(left["params"], right["params"])]))
    pose_max = 0.0
    observation_max = 0.0
    for image_id in expected.images:
        left, right = expected.images[image_id], actual.images[image_id]
        if left["name"] != right["name"] or left["camera_id"] != right["camera_id"]:
            _fail(f"COLMAP image {image_id} name/camera identity differs")
        pose_max = max(
            pose_max,
            _max_abs([a - b for a, b in zip(left["qvec"], right["qvec"])]),
            _max_abs([a - b for a, b in zip(left["tvec"], right["tvec"])]),
        )
        if len(left["points2D"]) != len(right["points2D"]):
            _fail(f"COLMAP image {image_id} POINTS2D count differs")
        for left_point, right_point in zip(left["points2D"], right["points2D"]):
            if int(left_point["point3D_id"]) != int(right_point["point3D_id"]):
                _fail(f"COLMAP image {image_id} POINTS2D point identity differs")
            observation_max = max(
                observation_max,
                abs(float(left_point["x"]) - float(right_point["x"])),
                abs(float(left_point["y"]) - float(right_point["y"])),
            )
    point_max = 0.0
    track_pairs_equal = True
    for point_id in expected.points3D:
        left, right = expected.points3D[point_id], actual.points3D[point_id]
        if tuple(left["rgb"]) != tuple(right["rgb"]) or list(left["track"]) != list(right["track"]):
            track_pairs_equal = False
            _fail(f"COLMAP point {point_id} RGB/track identity differs")
        point_max = max(
            point_max,
            _max_abs([a - b for a, b in zip(left["xyz"], right["xyz"])]),
            abs(float(left["error"]) - float(right["error"])),
        )
    if max(camera_max, pose_max, observation_max, point_max) > tolerance:
        _fail(
            "COLMAP TXT/BIN/round-trip numeric difference exceeds tolerance: "
            f"camera={camera_max}, pose={pose_max}, observations={observation_max}, points={point_max}"
        )
    return {
        "camera_ids_equal": True,
        "image_ids_equal": True,
        "point_ids_equal": True,
        "names_equal": True,
        "camera_max_abs_diff": camera_max,
        "pose_qvec_tvec_max_abs_diff": pose_max,
        "observation_xy_max_abs_diff": observation_max,
        "point_xyz_error_max_abs_diff": point_max,
        "track_pairs_equal": track_pairs_equal,
        "tolerance": tolerance,
    }


def decoded_pixel_sha256(path: str | Path) -> str:
    try:
        import cv2
    except ImportError as exc:
        _fail("OpenCV is required for LongSplat input pixel verification")
        raise AssertionError from exc
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        _fail(f"cannot decode training input image: {path}")
    return hashlib.sha256(image.tobytes()).hexdigest()


def materialize_training_images(
    records: Sequence[Mapping[str, Any]],
    destination: str | Path,
    *,
    expected_width: int,
    expected_height: int,
) -> tuple[list[dict[str, Any]], str]:
    """Copy final camera-staging PNGs and bind file/pixel identity."""

    try:
        import cv2
    except ImportError as exc:
        _fail("OpenCV is required for LongSplat input image materialization")
        raise AssertionError from exc
    output_dir = Path(destination).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for record in sorted(records, key=lambda item: str(item.get("staged_name", item.get("name", "")))):
        name = str(record.get("staged_name", record.get("name", "")))
        source_value = record.get("path") or record.get("output_path")
        if not name or Path(name).name != name or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            _fail(f"invalid training image staged name: {name}")
        if name in names:
            _fail(f"duplicate training image staged name: {name}")
        if not isinstance(source_value, str):
            _fail(f"missing parent image path for {name}")
        source = Path(source_value).resolve()
        if not source.is_file():
            _fail(f"parent final canonical image is missing: {source}")
        destination_path = output_dir / name
        if destination_path.exists() or destination_path.is_symlink():
            if destination_path.is_symlink() or not destination_path.is_file():
                _fail(f"training image destination is not an independent file: {destination_path}")
            if sha256_file(destination_path) != sha256_file(source):
                _fail(f"existing training image differs from parent final pixel: {destination_path}")
        else:
            shutil.copy2(source, destination_path)
        image = cv2.imread(str(destination_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim < 2 or image.shape[1] != expected_width or image.shape[0] != expected_height:
            _fail(f"training image dimensions failed for {name}")
        source_sha = sha256_file(source)
        file_sha = sha256_file(destination_path)
        if file_sha != source_sha:
            _fail(f"training image file SHA differs after copy for {name}")
        pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
        record_output = {
            "name": name,
            "staged_name": name,
            "source_path": str(source),
            "source_file_sha256": source_sha,
            "path": str(destination_path),
            "file_sha256": file_sha,
            "sha256": file_sha,
            "decoded_pixel_sha256": pixel_sha,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        "provenance": "camera-staging-final-canonical-independent-copy",
        }
        result.append(record_output)
        names.add(name)
    if not result:
        _fail("training input requires at least one image")
    aggregate = stable_sha256(
        [{"name": item["name"], "decoded_pixel_sha256": item["decoded_pixel_sha256"]} for item in result]
    )
    return result, aggregate


def _copy_once_or_verify(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            _fail(f"immutable copied artifact differs: {destination}")
        return
    shutil.copy2(source, destination)


def prepare_converter_output_dirs(*directories: str | Path) -> None:
    """Create only the private destinations owned by this conversion attempt."""

    for directory in directories:
        Path(directory).resolve().mkdir(parents=True, exist_ok=True)


def _nested_backend_code_identity(route: Path, train_script: Path) -> dict[str, str]:
    safe_state_file = route / _SAFE_STATE_FILE
    if not safe_state_file.is_file():
        _fail(f"nested safe_state source is missing: {safe_state_file}")
    if not train_script.is_file():
        _fail(f"nested training entrypoint is missing: {train_script}")
    unsigned = {
        "safe_state_source": _SAFE_STATE_SOURCE,
        "safe_state_file_sha256": sha256_file(safe_state_file),
        "train_entrypoint": _TRAIN_ENTRYPOINT,
        "train_entrypoint_sha256": sha256_file(train_script),
    }
    return {
        **unsigned,
        "identity_sha256": stable_sha256(unsigned),
    }


def _seed_contract(nested_identity: Mapping[str, str]) -> dict[str, Any]:
    return {
        "value": 0,
        "source": _SAFE_STATE_SOURCE,
        "application_order": "before_GaussianModel_and_Scene_and_training",
        "python_random": True,
        "numpy": True,
        "torch_manual_seed": True,
        "cli_flag": None,
        "scope": "seeded stochastic initialization/camera sampling",
        "bitwise_cuda_determinism": False,
        "limitation": "CUDA/custom rasterizer kernels may remain nondeterministic",
        "nested_backend_code_identity_sha256": nested_identity["identity_sha256"],
    }


def validate_future_smoke_plan(
    plan: Mapping[str, Any],
    *,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    training_root: str | Path | None = None,
    allow_producer_code_drift: bool = False,
) -> None:
    """Validate the workload, seed, and dynamic run containment contracts.

    ``route_root`` is a code identity root, never an output containment root.
    Current and legacy v2 plans therefore require the caller to provide the
    immutable reconstruction run root explicitly.  This keeps plans produced
    under arbitrary output roots valid without falling back to
    ``route/outputs`` or trusting an unbound common ancestor.
    """

    if plan.get("schema_version") != _FUTURE_SMOKE_PLAN_SCHEMA:
        _fail(f"future smoke plan schema must be {_FUTURE_SMOKE_PLAN_SCHEMA}")
    argv = plan.get("argv")
    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
        _fail("future smoke plan argv must be a string list")
    if "--seed" in argv:
        _fail("future smoke plan must not invent a --seed CLI flag")
    frozen = plan.get("frozen_contract")
    if not isinstance(frozen, Mapping):
        _fail("future smoke plan frozen_contract is required")
    workload_profile = plan.get("workload_profile")
    if not isinstance(workload_profile, str):
        _fail("future smoke workload_profile is required")
    profile = resolve_future_workload_profile(plan)
    if frozen.get("workload_profile") != workload_profile:
        _fail("future smoke frozen workload_profile differs from plan")
    if frozen.get("iterations") != profile["iterations"]:
        _fail(f"future smoke {workload_profile} iterations are fixed at {profile['iterations']}")
    seed = frozen.get("seed")
    if not isinstance(seed, Mapping):
        _fail("future smoke plan frozen_contract.seed is required")
    expected_seed = {
        "value": 0,
        "source": _SAFE_STATE_SOURCE,
        "application_order": "before_GaussianModel_and_Scene_and_training",
        "python_random": True,
        "numpy": True,
        "torch_manual_seed": True,
        "cli_flag": None,
        "scope": "seeded stochastic initialization/camera sampling",
        "bitwise_cuda_determinism": False,
        "limitation": "CUDA/custom rasterizer kernels may remain nondeterministic",
    }
    for key, value in expected_seed.items():
        if seed.get(key) != value:
            _fail(f"future smoke seed contract mismatch at {key}")
    if seed.get("value") != 0 or seed.get("cli_flag") is not None:
        _fail("future smoke seed must remain the fixed backend seed 0 without a CLI flag")
    telemetry = frozen.get("camera_sampling_telemetry")
    if telemetry is not None:
        if not isinstance(telemetry, Mapping):
            _fail("future smoke camera_sampling_telemetry must be an object")
        if telemetry.get("schema_version") != _CAMERA_SAMPLING_TELEMETRY_SCHEMA:
            _fail("future smoke camera_sampling_telemetry schema mismatch")
        if telemetry.get("required_for_new_execution") is not True:
            _fail("future smoke camera_sampling_telemetry must be required for new execution")
        if telemetry.get("scope") != "external_colmap_pose_only":
            _fail("future smoke camera_sampling_telemetry scope mismatch")
        if telemetry.get("legacy_evidence_compatibility") != "absent_allowed":
            _fail("future smoke camera_sampling_telemetry legacy compatibility mismatch")
    if containment_root is None:
        _fail("future smoke plan validation requires an explicit containment_root")
    try:
        containment = resolve_containment_root(
            containment_root,
            label="future smoke plan containment root",
        )
    except PipelineBlocked as exc:
        _fail(str(exc))

    source_value = plan.get("source_path")
    model_value = plan.get("model_path")
    if not isinstance(source_value, str) or not Path(source_value).is_absolute():
        _fail("future smoke source_path must be an absolute path")
    if not isinstance(model_value, str) or not Path(model_value).is_absolute():
        _fail("future smoke model_path must be an absolute path")
    try:
        source_path = resolve_contained_path(
            source_value,
            root=containment,
            label="future smoke source_path",
            must_exist=True,
            directory=True,
        )
        model_path = resolve_contained_path(
            model_value,
            root=containment,
            label="future smoke model_path",
            must_exist=False,
        )
    except PipelineBlocked as exc:
        _fail(str(exc))
    if training_root is not None:
        try:
            expected_training_root = resolve_contained_path(
                training_root,
                root=containment,
                label="future smoke training_root",
                must_exist=True,
                directory=True,
            )
        except PipelineBlocked as exc:
            _fail(str(exc))
        if source_path != expected_training_root:
            _fail("future smoke source_path differs from training root")
    route = Path(route_root).resolve()
    train_script = route / _TRAIN_ENTRYPOINT
    actual_nested = _nested_backend_code_identity(route, train_script)
    declared_nested = plan.get("nested_backend_code_identity")
    if declared_nested != actual_nested and not allow_producer_code_drift:
        _fail("future smoke nested backend code identity mismatch")
    if seed.get("nested_backend_code_identity_sha256") != actual_nested["identity_sha256"] and not allow_producer_code_drift:
        _fail("future smoke seed contract is not bound to nested backend identity")
    if plan.get("train_script_sha256") != actual_nested["train_entrypoint_sha256"] and not allow_producer_code_drift:
        _fail("future smoke train_script SHA differs from nested backend identity")
    safe_state_text = (route / _SAFE_STATE_FILE).read_text(encoding="utf-8")
    for marker in (
        "random.seed(0)",
        "np.random.seed(0)",
        "torch.manual_seed(0)",
        "torch.cuda.set_device(torch.device(\"cuda:0\"))",
    ):
        if marker not in safe_state_text:
            _fail(f"nested safe_state seed contract marker is missing: {marker}")
    train_text = train_script.read_text(encoding="utf-8")
    safe_state_index = train_text.find("safe_state(args.quiet)")
    training_index = train_text.find("training(lp.extract")
    if safe_state_index < 0 or training_index < 0 or safe_state_index >= training_index:
        _fail("safe_state must be applied before the training entrypoint")
    if plan.get("model_path_exists_at_plan_time") is not False:
        _fail("future smoke model_path must be absent at plan time")
    declared_parent_binding = plan.get("parent_binding")
    if declared_parent_binding is not None:
        if not isinstance(declared_parent_binding, Mapping):
            _fail("future smoke parent_binding must be an object")
        _validate_parent_binding_digest(declared_parent_binding)
        if plan.get("parent_binding_sha256") != declared_parent_binding.get("binding_sha256"):
            _fail("future smoke parent_binding SHA binding failed")
    required_values = {
        "--images": "images",
        "--mode": "custom",
        "--resolution": "1",
        "--iterations": str(profile["iterations"]),
        "--depth_source": "disabled",
        "--loss_2d_correspondence_weight": "0",
        "--depth_loss_weight": "0",
        "--rotation_lr_init": "0",
        "--translation_lr_init": "0",
    }
    for flag, expected in required_values.items():
        try:
            actual = argv[argv.index(flag) + 1]
        except (ValueError, IndexError):
            _fail(f"future smoke argv is missing {flag}")
        if actual != expected:
            _fail(f"future smoke argv {flag} must be {expected}")
    for flag, expected in (
        ("--source_path", str(source_path)),
        ("--model_path", str(model_path)),
    ):
        try:
            actual = argv[argv.index(flag) + 1]
        except (ValueError, IndexError):
            _fail(f"future smoke argv is missing {flag}")
        if actual != expected:
            _fail(f"future smoke argv {flag} differs from the contained plan path")
    for flag in ("--external_colmap_pose", "--disable_resize", "--model_path"):
        if flag not in argv:
            _fail(f"future smoke argv is missing {flag}")


def _resolve_inside(root: Path, value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{label} escapes training input root: {resolved}")
    return resolved


def _parent_path(root: Path, value: str | Path, *, label: str) -> Path:
    """Resolve a parent artifact and prove that it stays inside that run."""

    if not isinstance(value, (str, Path)) or not str(value):
        _fail(f"parent {label} path is missing")
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"parent {label} path escapes the parent run: {resolved}")
    return resolved


def _parent_artifacts_valid(root: Path, result: Mapping[str, Any]) -> bool:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            return False
        try:
            path = _parent_path(root, artifact.get("path", ""), label="artifact")
        except LongSplatInputBlocked:
            return False
        expected_sha = artifact.get("sha256")
        if not path.is_file() or not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
            return False
    return True


def _parent_stage_attempts(root: Path, run: Mapping[str, Any], stage: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entries = run.get("stages", {}).get(stage)
    if not isinstance(entries, list) or not entries:
        _fail(f"parent has no {stage} attempts")
    inventory: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            _fail(f"parent {stage} attempt entry is malformed")
        result_path = _parent_path(root, entry.get("result_path", ""), label=f"{stage} result")
        if not result_path.is_file():
            _fail(f"parent {stage} result is missing: {result_path}")
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record.get("stage") != stage or record.get("identity") != run.get("identity"):
            _fail(f"parent {stage} result identity/stage drifted: {result_path}")
        result = record.get("result")
        if not isinstance(result, Mapping):
            _fail(f"parent {stage} result payload is malformed: {result_path}")
        status = record.get("status")
        if entry.get("status") != status:
            _fail(f"parent {stage} ledger status disagrees with result: {result_path}")
        attempt_name = str(entry.get("attempt", result_path.parent.name))
        match = re.fullmatch(r"attempt-([0-9]+)", attempt_name)
        attempt_number = int(match.group(1)) if match else ordinal + 1
        inventory.append(
            {
                "attempt": attempt_name,
                "attempt_number": attempt_number,
                "status": status,
                "result_path": str(result_path),
                "result_file_sha256": sha256_file(result_path),
                "artifacts_valid": _parent_artifacts_valid(root, result),
                "reusable": status == "passed" and _parent_artifacts_valid(root, result),
                "result": dict(result),
            }
        )
    reusable = [item for item in inventory if item["reusable"]]
    if not reusable:
        _fail(f"parent has no reusable successful {stage} attempt")
    selected = max(reusable, key=lambda item: (item["attempt_number"], inventory.index(item)))
    return selected, inventory


def _parent_model_aggregate(*model_dirs: Path) -> str:
    records: list[dict[str, str]] = []
    for kind, model_dir in zip(("txt", "bin"), model_dirs):
        for name in ("cameras", "images", "points3D"):
            suffix = ".txt" if kind == "txt" else ".bin"
            path = model_dir / f"{name}{suffix}"
            if not path.is_file():
                _fail(f"parent undistorted {kind} model file is missing: {path}")
            records.append({"kind": kind, "name": path.name, "sha256": sha256_file(path)})
    return stable_sha256(records)


def _validate_parent_binding_digest(binding: Mapping[str, Any]) -> str:
    declared = binding.get("binding_sha256")
    if not isinstance(declared, str):
        _fail("parent_binding.binding_sha256 is required")
    unsigned = dict(binding)
    unsigned.pop("binding_sha256", None)
    actual = stable_sha256(unsigned)
    if actual != declared:
        _fail("parent_binding stable SHA failed")
    if binding.get("schema_version") != _PARENT_BINDING_SCHEMA:
        _fail(f"parent_binding schema must be {_PARENT_BINDING_SCHEMA}")
    return declared


def load_parent_camera_staging_evidence(
    parent_run_root: str | Path,
    *,
    expected_source_video_sha256: str,
    current_colmap_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and verify a generic, immutable camera-staging parent run."""

    root = Path(parent_run_root).resolve()
    if not root.is_dir():
        _fail(f"parent run root is missing: {root}")
    identity_path = root / "identity.json"
    run_path = root / "run.json"
    config_path = root / "config.json"
    if not identity_path.is_file() or not run_path.is_file() or not config_path.is_file():
        _fail("parent identity, run ledger, or config is missing")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(identity, Mapping):
        _fail("parent identity payload is malformed")
    if run.get("identity") != identity:
        _fail("parent run ledger identity differs from identity.json")
    required_identity_fields = (
        "source_video_sha256",
        "canonical_config_sha256",
        "tool_identity_sha256",
        "code_identity",
    )
    missing_identity_fields = [field for field in required_identity_fields if field not in identity]
    if missing_identity_fields:
        _fail("parent run identity is missing " + ", ".join(missing_identity_fields))
    for field in ("source_video_sha256", "canonical_config_sha256", "tool_identity_sha256"):
        if not isinstance(identity.get(field), str):
            _fail(f"parent run identity field must be a string: {field}")
    if not isinstance(identity.get("code_identity"), (str, Mapping)):
        _fail("parent run identity field must be a string or object: code_identity")
    if identity.get("source_video_sha256") != expected_source_video_sha256:
        _fail("parent source video SHA does not match current input")
    if identity.get("schema_version") != RUN_IDENTITY_SCHEMA:
        _fail(f"parent run identity schema must be {RUN_IDENTITY_SCHEMA}")
    declared_run_identity = identity.get("run_identity_sha256")
    if not isinstance(declared_run_identity, str):
        _fail("parent run identity SHA is required")
    unsigned_identity = dict(identity)
    unsigned_identity.pop("schema_version", None)
    unsigned_identity.pop("run_identity_sha256", None)
    if stable_sha256(unsigned_identity) != declared_run_identity:
        _fail("parent run identity stable SHA failed")
    if run.get("computed_pass") is not True:
        _fail("parent run is not a computed-pass CPU reference")
    if config.get("depth_source") != "disabled" or config.get("camera_model") != "SIMPLE_RADIAL" or config.get("matching") != "sequential":
        _fail("parent config does not match the supported external COLMAP profile")
    if config.get("camera_prior") is not None:
        _fail("parent camera prior must be null for the supported profile")

    preflight_path = root / "preflight.json"
    if not preflight_path.is_file():
        _fail(f"parent preflight identity is missing: {preflight_path}")
    parent_preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if parent_preflight.get("status") != "passed":
        _fail("parent dependency preflight is not passed")
    parent_colmap_identity = parent_preflight.get("dependencies", {}).get("colmap")
    if not isinstance(parent_colmap_identity, Mapping) or not isinstance(parent_colmap_identity.get("sha256"), str):
        _fail("parent COLMAP tool identity is missing")
    if current_colmap_identity is not None and current_colmap_identity.get("sha256") != parent_colmap_identity.get("sha256"):
        _fail("current COLMAP executable bytes differ from parent evidence")

    camera_attempt, camera_inventory = _parent_stage_attempts(root, run, "camera-staging")
    camera_result = camera_attempt["result"]
    contract_path = _parent_path(root, camera_result.get("contract_path", ""), label="camera contract")
    if not contract_path.is_file() or contract_path.parent != _parent_path(root, camera_attempt["result_path"], label="camera result").parent / "camera_staging":
        _fail("camera contract is not inside the selected camera-staging attempt")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if camera_result.get("contract") != contract:
        _fail("camera-staging result contract differs from the immutable contract file")
    contract_sha = contract.get("contract_sha256")
    unsigned_contract = dict(contract)
    unsigned_contract.pop("contract_sha256", None)
    if not isinstance(contract_sha, str) or stable_sha256(unsigned_contract) != contract_sha:
        _fail("parent camera contract stable SHA failed")
    contract_file_sha = sha256_file(contract_path)
    matching_contract_artifacts = [
        item for item in camera_result.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("path") == str(contract_path)
    ]
    if len(matching_contract_artifacts) != 1 or matching_contract_artifacts[0].get("sha256") != contract_file_sha:
        _fail("camera-staging result does not bind the camera contract file SHA")
    if contract.get("source_video_sha256") != expected_source_video_sha256:
        _fail("camera contract source SHA differs from parent identity")
    if contract.get("accepted") is not False or contract.get("delivery_reachable") is not False:
        _fail("camera contract is not a non-delivery CPU reference")

    final_camera_record = contract.get("camera")
    if not isinstance(final_camera_record, Mapping) or final_camera_record.get("model") != "PINHOLE":
        _fail("parent camera contract has no final PINHOLE camera")
    final_width = int(final_camera_record.get("width", 0))
    final_height = int(final_camera_record.get("height", 0))
    if final_width <= 0 or final_height <= 0:
        _fail("parent final camera dimensions are invalid")
    frames = contract.get("frames")
    frame_names = contract.get("frame_names")
    if not isinstance(frames, list) or not frames or not isinstance(frame_names, list) or len(frames) != len(frame_names):
        _fail("parent camera contract frame list is missing or inconsistent")
    if contract.get("frame_count") != len(frames):
        _fail("parent camera contract frame_count disagrees with frame records")
    if len(set(frame_names)) != len(frame_names):
        _fail("parent camera contract frame names are not unique")
    final_dir = contract_path.parent / "canonical_images"
    records: list[dict[str, Any]] = []
    try:
        import cv2
    except ImportError as exc:
        _fail("OpenCV is required to verify parent camera-staging pixels")
        raise AssertionError from exc
    for index, record in enumerate(frames):
        if not isinstance(record, Mapping):
            _fail(f"parent camera frame record {index} is malformed")
        name = str(record.get("staged_name", ""))
        if not name or Path(name).name != name or any(character.isspace() for character in name) or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            _fail(f"parent camera frame name is not a safe staged basename: {name}")
        if name != frame_names[index]:
            _fail("parent camera frame order differs from frame_names")
        expected_path = (final_dir / name).resolve()
        declared_path = _parent_path(root, record.get("path", ""), label=f"camera frame {name}")
        if declared_path != expected_path or not expected_path.is_file() or expected_path.is_symlink():
            _fail(f"parent camera frame is not an immutable canonical image: {name}")
        image = cv2.imread(str(expected_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim < 2 or image.shape[1] != final_width or image.shape[0] != final_height:
            _fail(f"parent camera frame dimensions failed: {name}")
        file_sha = sha256_file(expected_path)
        pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
        if record.get("sha256") != file_sha or record.get("decoded_pixel_sha256") != pixel_sha:
            _fail(f"parent camera frame identity failed: {name}")
        records.append(dict(record, path=str(expected_path)))
    aggregate = stable_sha256(
        [{"name": item["staged_name"], "decoded_pixel_sha256": item["decoded_pixel_sha256"]} for item in records]
    )
    if aggregate != contract.get("staging", {}).get("final_staged_pixel_aggregate_sha256"):
        _fail("parent camera frame aggregate does not match the camera contract")

    media_record_path = _parent_path(root, contract.get("canonical_media_record_path", ""), label="canonical media record")
    if not media_record_path.is_file() or contract.get("canonical_media_record_sha256") != sha256_file(media_record_path):
        _fail("parent canonical media record file SHA failed")
    media = json.loads(media_record_path.read_text(encoding="utf-8"))
    binding = media.get("binding", {})
    canonical = media.get("canonical", {})
    if binding.get("source_video_sha256") != expected_source_video_sha256 or media.get("source", {}).get("sha256") != expected_source_video_sha256:
        _fail("parent canonical media record source SHA failed")
    if binding.get("canonical_media_sha256") != contract.get("canonical_media_binding_sha256"):
        _fail("parent canonical media binding SHA failed")
    if canonical.get("pixels_sha256") != contract.get("canonical_media_pixels_sha256"):
        _fail("parent canonical decoded pixel SHA failed")

    colmap_attempt, colmap_inventory = _parent_stage_attempts(root, run, "colmap")
    colmap_result = colmap_attempt["result"]
    if colmap_result.get("computed_pass") is not True or colmap_result.get("execution_pass") is not True:
        _fail("selected parent COLMAP result is not a computed/execution pass")
    for command_result in colmap_result.get("command_results", []):
        if not isinstance(command_result, Mapping) or command_result.get("exit_code") != 0:
            _fail("selected parent COLMAP command did not exit zero")
        command_tool = command_result.get("tool_identity", {})
        if command_tool.get("sha256") != parent_colmap_identity.get("sha256"):
            _fail("selected parent COLMAP command tool identity drifted")
    plan = colmap_result.get("plan", {})
    if plan.get("source_video_sha256") != expected_source_video_sha256 or plan.get("canonical_media_sha256") != contract.get("canonical_media_pixels_sha256"):
        _fail("parent COLMAP source/canonical pixel binding failed")
    if plan.get("single_camera") is not True or plan.get("camera_model_requested") != config.get("camera_model") or plan.get("camera_prior") is not None:
        _fail("parent COLMAP plan does not match the configured camera profile")
    if not any(isinstance(item, Mapping) and item.get("stage") == "matching" and "sequential_matcher" in item.get("argv", []) for item in colmap_result.get("command_results", [])):
        _fail("parent COLMAP result is not from the sequential matching contract")
    evidence_geometry = colmap_result.get("evidence", {})
    if evidence_geometry.get("component_count") != 1 or evidence_geometry.get("registered_image_count") != len(records):
        _fail("parent COLMAP registered/component contract failed")
    if sorted(evidence_geometry.get("registered_image_names", [])) != sorted(frame_names):
        _fail("parent COLMAP registered names differ from camera-staging names")
    paths = plan.get("paths", {})
    und_txt = _parent_path(root, paths.get("undistorted_sparse_txt", ""), label="undistorted TXT model")
    und_bin = _parent_path(root, paths.get("undistorted_sparse_binary", ""), label="undistorted binary model")
    if not und_txt.is_dir() or not und_bin.is_dir():
        _fail("parent undistorted TXT/binary sparse paths are missing")
    txt_model = parse_colmap_model_text(und_txt)
    bin_model = parse_colmap_model_binary(und_bin)
    comparison = compare_colmap_models(txt_model, bin_model, tolerance=1e-9)
    if sorted(image["name"] for image in txt_model.images.values()) != sorted(frame_names):
        _fail("parent undistorted model names differ from camera-staging names")
    if len(txt_model.cameras) != 1 or next(iter(txt_model.cameras.values()))["model"] != "PINHOLE":
        _fail("parent undistorted source must contain exactly one PINHOLE camera")
    und_camera = next(iter(txt_model.cameras.values()))
    declared_und_camera = contract.get("staging", {}).get("undistorted_camera", {})
    expected_und = {
        "model": declared_und_camera.get("model"),
        "width": declared_und_camera.get("width"),
        "height": declared_und_camera.get("height"),
        "params": [declared_und_camera.get("fx"), declared_und_camera.get("fy"), declared_und_camera.get("cx"), declared_und_camera.get("cy")],
    }
    if und_camera["model"] != expected_und["model"] or (und_camera["width"], und_camera["height"]) != (expected_und["width"], expected_und["height"]):
        _fail("parent undistorted camera dimensions/model disagree with camera contract")
    if any(not math.isclose(a, float(b), rel_tol=0.0, abs_tol=1e-8) for a, b in zip(und_camera["params"], expected_und["params"])):
        _fail("parent undistorted camera K disagrees with camera contract")
    model_aggregate = _parent_model_aggregate(und_txt, und_bin)
    parent_binding = {
        "schema_version": _PARENT_BINDING_SCHEMA,
        "parent_run_root": str(root),
        "parent_run_identity_sha256": identity.get("run_identity_sha256"),
        "parent_code_identity_sha256": identity.get("code_identity", {}).get("code_identity_sha256"),
        "source_video_sha256": expected_source_video_sha256,
        "canonical_media_record_path": str(media_record_path),
        "canonical_media_record_file_sha256": sha256_file(media_record_path),
        "canonical_media_binding_sha256": contract.get("canonical_media_binding_sha256"),
        "canonical_media_pixels_sha256": contract.get("canonical_media_pixels_sha256"),
        "camera_staging_attempt": camera_attempt["attempt"],
        "camera_staging_result_path": camera_attempt["result_path"],
        "camera_staging_result_file_sha256": camera_attempt["result_file_sha256"],
        "camera_contract_path": str(contract_path),
        "camera_contract_file_sha256": contract_file_sha,
        "camera_contract_internal_sha256": contract_sha,
        "final_staged_pixel_aggregate_sha256": aggregate,
        "frame_count": len(records),
        "frame_names": list(frame_names),
        "frame_names_sha256": stable_sha256(list(frame_names)),
        "final_camera": dict(final_camera_record),
        "undistorted_camera": dict(declared_und_camera),
        "undistorted_txt_path": str(und_txt),
        "undistorted_binary_path": str(und_bin),
        "undistorted_model_aggregate_sha256": model_aggregate,
        "colmap_attempt": colmap_attempt["attempt"],
        "colmap_result_path": colmap_attempt["result_path"],
        "colmap_result_file_sha256": colmap_attempt["result_file_sha256"],
        "colmap_tool_identity_sha256": parent_colmap_identity.get("sha256"),
        "colmap_resolved_path": parent_colmap_identity.get("resolved_path"),
        "camera_staging_attempt_inventory": [
            {key: value for key, value in item.items() if key != "result"} for item in camera_inventory
        ],
        "colmap_attempt_inventory": [
            {key: value for key, value in item.items() if key != "result"} for item in colmap_inventory
        ],
    }
    parent_binding["binding_sha256"] = stable_sha256(parent_binding)
    return {
        "root": str(root),
        "identity": identity,
        "config": config,
        "run": run,
        "contract_path": str(contract_path),
        "contract": contract,
        "contract_sha256": contract_sha,
        "contract_file_sha256": contract_file_sha,
        "records": records,
        "final_pixel_aggregate_sha256": aggregate,
        "undistorted_txt": str(und_txt),
        "undistorted_binary": str(und_bin),
        "undistorted_txt_model": txt_model,
        "undistorted_binary_comparison": comparison,
        "source_camera": und_camera,
        "colmap_plan": plan,
        "preflight": parent_preflight,
        "parent_colmap_identity": dict(parent_colmap_identity),
        "parent_binding": parent_binding,
        "parent_binding_sha256": parent_binding["binding_sha256"],
    }


def build_future_smoke_plan(
    *,
    training_root: str | Path,
    backend_python: str | Path,
    route_root: str | Path,
    containment_root: str | Path,
    code_identity: Mapping[str, Any],
    backend_identity: Mapping[str, Any] | None,
    parent_binding: Mapping[str, Any] | None = None,
    workload_profile: str = "smoke100",
) -> dict[str, Any]:
    """Freeze, but do not execute, one supported future GPU workload."""

    root = Path(training_root).resolve()
    route = Path(route_root).resolve()
    profile = future_workload_profile(workload_profile)
    model_path = root.parent / f"{root.name}{profile['model_suffix']}"
    if model_path.exists():
        _fail(f"future smoke model_path must be absent before this CPU stage: {model_path}")
    train_script = route / "third_party" / "LongSplat" / "train.py"
    if not train_script.is_file():
        _fail(f"future smoke train.py is missing: {train_script}")
    nested_identity = _nested_backend_code_identity(route, train_script)
    seed = _seed_contract(nested_identity)
    argv = [
        str(Path(backend_python).resolve()),
        str(train_script),
        "--source_path",
        str(root),
        "--images",
        "images",
        "--mode",
        "custom",
        "--resolution",
        "1",
        "--iterations",
        str(profile["iterations"]),
        "--external_colmap_pose",
        "--disable_resize",
        "--depth_source",
        "disabled",
        "--loss_2d_correspondence_weight",
        "0",
        "--depth_loss_weight",
        "0",
        "--rotation_lr_init",
        "0",
        "--translation_lr_init",
        "0",
        "--model_path",
        str(model_path),
    ]
    plan = {
        "schema_version": _FUTURE_SMOKE_PLAN_SCHEMA,
        "workload_profile": workload_profile,
        "execution_status": "plan_only_not_executed",
        "accepted": False,
        "delivery_reachable": False,
        "gpu_invoked": False,
        "training_invoked": False,
        "conversion_invoked": False,
        "source_path": str(root),
        "images": "images",
        "model_path": str(model_path),
        "model_path_exists_at_plan_time": False,
        "argv": argv,
        "frozen_contract": {
            "mode": "custom",
            "resolution": 1,
            "workload_profile": workload_profile,
            "iterations": profile["iterations"],
            "render_iteration": profile["render_iteration"],
            "external_colmap_pose": True,
            "disable_resize": True,
            "depth_source": "disabled",
            "depth_loss_weight": 0.0,
            "correspondence_loss_weight": 0.0,
            "rotation_lr_init": 0.0,
            "translation_lr_init": 0.0,
            "vda_enabled": False,
            "mast3r_enabled": False,
            "matcher_ab_enabled": False,
            "camera_prior": None,
            "conversion": False,
            "profile_semantics": {
                "smoke100-v1": "local structural/numeric smoke only; not a rough-visual formal-release gate",
                "formal30000-v1": "local versioned policy; not the official final training standard",
            }.get(workload_profile, "compatibility alias for a versioned local profile"),
            "camera_sampling_telemetry": {
                "schema_version": _CAMERA_SAMPLING_TELEMETRY_SCHEMA,
                "required_for_new_execution": True,
                "scope": "external_colmap_pose_only",
                "legacy_evidence_compatibility": "absent_allowed",
                "resume_semantics": "artifact inspection only; checkpoint is not bit-exact training resume",
            },
            "seed": seed,
        },
        "nested_backend_code_identity": nested_identity,
        "route_code_identity_sha256": code_identity.get("code_identity_sha256"),
        "route_code_identity": dict(code_identity),
        "train_script_sha256": sha256_file(train_script),
        "backend_identity": None if backend_identity is None else dict(backend_identity),
    }
    if parent_binding is not None:
        parent_binding_copy = dict(parent_binding)
        _validate_parent_binding_digest(parent_binding_copy)
        plan["parent_binding"] = parent_binding_copy
        plan["parent_binding_sha256"] = parent_binding_copy["binding_sha256"]
    validate_future_smoke_plan(
        plan,
        route_root=route,
        containment_root=containment_root,
        training_root=root,
    )
    return plan


def _artifact_records(root: Path, paths: Sequence[Path]) -> list[dict[str, str]]:
    records = []
    for path in sorted({Path(value).resolve() for value in paths}):
        if not path.is_file():
            _fail(f"required training input artifact is missing: {path}")
        records.append({"path": str(path), "sha256": sha256_file(path)})
    return records


def build_longsplat_input_stage(
    *,
    parent_run_root: str | Path,
    output_dir: str | Path,
    source_video_sha256: str,
    colmap: str | Path,
    colmap_identity: Mapping[str, Any],
    backend_python: str | Path,
    backend_identity: Mapping[str, Any] | None,
    route_root: str | Path,
    containment_root: str | Path,
    code_identity: Mapping[str, Any],
    derived_identity: Mapping[str, Any],
    parent_binding: Mapping[str, Any],
    runner: Callable[..., Any],
    workload_profile: str = "smoke100",
) -> dict[str, Any]:
    """Create one append-only LongSplat training-specific source path."""

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    parent = load_parent_camera_staging_evidence(
        parent_run_root,
        expected_source_video_sha256=source_video_sha256,
        current_colmap_identity=colmap_identity,
    )
    if dict(parent["parent_binding"]) != dict(parent_binding):
        _fail("parent binding changed between derived-run preflight and stage execution")
    parent_colmap_identity = parent["parent_colmap_identity"]
    if parent_colmap_identity.get("sha256") != colmap_identity.get("sha256"):
        _fail("current COLMAP executable bytes differ from immutable parent evidence")
    contract = parent["contract"]
    final_camera_record = contract.get("camera")
    if not isinstance(final_camera_record, Mapping):
        _fail("parent camera contract has no final camera")
    final_camera = {
        "model": final_camera_record["model"],
        "width": int(final_camera_record["width"]),
        "height": int(final_camera_record["height"]),
        "fx": float(final_camera_record["fx"]),
        "fy": float(final_camera_record["fy"]),
        "cx": float(final_camera_record["cx"]),
        "cy": float(final_camera_record["cy"]),
    }
    crop = contract.get("staging", {}).get("crop_roi")
    if not isinstance(crop, Mapping):
        _fail("parent camera contract has no crop ROI")
    source_model = parent["undistorted_txt_model"]
    rewritten_model, rewrite_stats = rewrite_colmap_model_for_crop(
        source_model,
        crop=crop,
        final_camera=final_camera,
    )
    txt_model_dir = root / "_model_txt"
    bin_model_dir = root / "_model_bin"
    roundtrip_model_dir = root / "_model_roundtrip_txt"
    final_model_dir = root / "sparse" / "0"
    write_colmap_model_text(rewritten_model, txt_model_dir)
    # COLMAP 3.9.1's model_converter expects the destination directory to
    # exist before it opens cameras.bin/images.bin/points3D.bin.  Creating the
    # private destination explicitly is part of the exact staging contract;
    # it does not touch the immutable parent model.
    prepare_converter_output_dirs(bin_model_dir, roundtrip_model_dir, final_model_dir)
    image_records, image_aggregate = materialize_training_images(
        parent["records"],
        root / "images",
        expected_width=final_camera["width"],
        expected_height=final_camera["height"],
    )
    if image_aggregate != parent["final_pixel_aggregate_sha256"]:
        _fail("independent training image aggregate differs from parent camera staging")

    commands = [
        ColmapCommand(
            "training_input_model_converter_txt_to_bin",
            (
                str(colmap),
                "model_converter",
                "--input_path",
                str(txt_model_dir),
                "--output_path",
                str(bin_model_dir),
                "--output_type",
                "BIN",
            ),
            str(root),
        ),
        ColmapCommand(
            "training_input_model_converter_bin_to_txt",
            (
                str(colmap),
                "model_converter",
                "--input_path",
                str(bin_model_dir),
                "--output_path",
                str(roundtrip_model_dir),
                "--output_type",
                "TXT",
            ),
            str(root),
        ),
    ]
    command_results: list[dict[str, Any]] = []
    for command in commands:
        result = run_colmap_command(
            command,
            tool_identity=colmap_identity,
            log_dir=root / "logs",
            runner=runner,
        )
        command_results.append(result)
        if result["exit_code"] != 0:
            _fail(f"COLMAP model_converter failed at {command.stage}")
    binary_model = parse_colmap_model_binary(bin_model_dir)
    roundtrip_model = parse_colmap_model_text(roundtrip_model_dir)
    txt_bin_comparison = compare_colmap_models(rewritten_model, binary_model, tolerance=1e-9)
    roundtrip_comparison = compare_colmap_models(rewritten_model, roundtrip_model, tolerance=1e-9)
    # Keep the canonical final model self-contained: both TXT and BIN are
    # present under sparse/0; the private conversion directories remain as
    # immutable evidence and are never referenced as the future source_path.
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        _copy_once_or_verify(txt_model_dir / name, final_model_dir / name)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        _copy_once_or_verify(bin_model_dir / name, final_model_dir / name)
    final_txt_model = parse_colmap_model_text(final_model_dir)
    final_bin_model = parse_colmap_model_binary(final_model_dir)
    final_comparison = compare_colmap_models(final_txt_model, final_bin_model, tolerance=1e-9)
    if final_comparison != txt_bin_comparison:
        _fail("final sparse/0 comparison differs from converter evidence")
    camera = next(iter(final_txt_model.cameras.values()))
    expected_params = [final_camera["fx"], final_camera["fy"], final_camera["cx"], final_camera["cy"]]
    if camera["model"] != "PINHOLE" or (camera["width"], camera["height"]) != (final_camera["width"], final_camera["height"]):
        _fail("final sparse/0 camera does not match camera contract")
    if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8) for a, b in zip(camera["params"], expected_params)):
        _fail("final sparse/0 K does not match camera contract")
    camera_contract_copy = root / "camera_contract-v1.json"
    _copy_once_or_verify(Path(parent["contract_path"]), camera_contract_copy)
    smoke_plan = build_future_smoke_plan(
        training_root=root,
        backend_python=backend_python,
        route_root=route_root,
        containment_root=containment_root,
        code_identity=code_identity,
        backend_identity=backend_identity,
        parent_binding=parent["parent_binding"],
        workload_profile=workload_profile,
    )
    smoke_plan_path = root / "future_smoke_plan-v2.json"
    if not smoke_plan_path.exists():
        write_json_once(smoke_plan_path, smoke_plan)
    elif json.loads(smoke_plan_path.read_text(encoding="utf-8")) != smoke_plan:
        _fail("future smoke plan differs on resume")
    manifest = {
        "schema_version": "longsplat-input-staging-v1",
        "status": "computed_pass",
        "accepted": False,
        "delivery_reachable": False,
        "gpu_invoked": False,
        "scope": "LongSplat training-specific model",
        "validation_scope": "staging/COLMAP reference only; no active Camera objects",
        "source_path": str(root),
        "parent_run_root": str(Path(parent_run_root).resolve()),
        "parent_binding": parent["parent_binding"],
        "parent_binding_sha256": parent["parent_binding_sha256"],
        "parent_camera_contract_path": parent["contract_path"],
        "parent_camera_contract_sha256": parent["contract_sha256"],
        "camera_contract_path": str(camera_contract_copy.relative_to(root)),
        "camera_contract_sha256": parent["contract_sha256"],
        "camera_contract_file_sha256": sha256_file(camera_contract_copy),
        "source_video_sha256": source_video_sha256,
        "canonical_media_binding_sha256": contract.get("canonical_media_binding_sha256"),
        "canonical_media_binding_sha256_semantics": "canonical-media-v1 binding record, not decoded pixels",
        "canonical_media_pixels_sha256": contract.get("canonical_media_pixels_sha256"),
        "canonical_media_pixels_sha256_semantics": "decoded canonical input pixels",
        "undistorted_input_pixel_aggregate_sha256": contract.get("staging", {}).get("undistorted_input_pixel_aggregate_sha256"),
        "final_staged_pixel_aggregate_sha256": image_aggregate,
        "observation_rewrite": True,
        "crop_provenance": {
            "strategy": contract.get("staging", {}).get("strategy"),
            "roi": dict(crop),
            "crop_affine": contract.get("staging", {}).get("crop_affine"),
            "K_raw": contract.get("staging", {}).get("K_raw"),
            "K_undistorted": contract.get("staging", {}).get("K_undistorted"),
            "K_canonical": contract.get("staging", {}).get("K_canonical"),
            "no_interpolation": True,
            "no_padding": True,
            "observation_mapping": "x'=x-x0,y'=y-y0; retain half-open ROI; rebuild POINT2D_IDX and tracks",
        },
        "camera": final_camera,
        "image_records": image_records,
        "images_relative": "images",
        "sparse_relative": "sparse/0",
        "sparse_txt_evidence_path": str(txt_model_dir),
        "sparse_binary_evidence_path": str(bin_model_dir),
        "roundtrip_txt_evidence_path": str(roundtrip_model_dir),
        "rewrite_statistics": rewrite_stats,
        "model_counts": {
            "source_undistorted": model_counts(source_model),
            "rewritten_txt": model_counts(rewritten_model),
            "final_txt": model_counts(final_txt_model),
            "final_bin": model_counts(final_bin_model),
        },
        "converter_commands": command_results,
        "txt_bin_comparison": txt_bin_comparison,
        "roundtrip_comparison": roundtrip_comparison,
        "final_txt_bin_comparison": final_comparison,
        "camera_contract_binding": {
            "source_video_sha256": source_video_sha256,
            "canonical_media_binding_sha256": contract.get("canonical_media_binding_sha256"),
            "decoded_canonical_input_pixel_sha256": contract.get("canonical_media_pixels_sha256"),
            "undistorted_input_pixel_aggregate_sha256": contract.get("staging", {}).get("undistorted_input_pixel_aggregate_sha256"),
            "final_staged_pixel_aggregate_sha256": image_aggregate,
        },
        "tool_identity": dict(colmap_identity),
        "parent_tool_identity": parent_colmap_identity,
        "code_identity": dict(code_identity),
        "derived_run_identity": dict(derived_identity),
        "parent_code_identity": parent["identity"].get("code_identity"),
        "future_smoke_plan_path": str(smoke_plan_path),
        "future_smoke_profile": smoke_plan["workload_profile"],
        "future_smoke_iterations": smoke_plan["frozen_contract"]["iterations"],
        "static_validator": "scripts.longsplat.validate_external_colmap_contract.validate_training_input",
    }
    manifest_path = root / "staging_manifest.json"
    if not manifest_path.exists():
        write_json_once(manifest_path, manifest)
    elif json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        _fail("staging_manifest differs on resume")
    return {
        "status": "computed_pass",
        "accepted": False,
        "delivery_reachable": False,
        "gpu_invoked": False,
        "source_path": str(root),
        "staging_manifest_path": str(manifest_path),
        "camera_contract_path": str(camera_contract_copy),
        "camera_contract_sha256": parent["contract_sha256"],
        "parent_binding": parent["parent_binding"],
        "parent_binding_sha256": parent["parent_binding_sha256"],
        "future_smoke_plan_path": str(smoke_plan_path),
        "future_smoke_profile": smoke_plan["workload_profile"],
        "future_smoke_iterations": smoke_plan["frozen_contract"]["iterations"],
        "rewrite_statistics": rewrite_stats,
        "model_counts": manifest["model_counts"],
        "image_count": len(image_records),
        "final_staged_pixel_aggregate_sha256": image_aggregate,
        "converter_commands": command_results,
        "txt_bin_comparison": txt_bin_comparison,
        "roundtrip_comparison": roundtrip_comparison,
        "final_txt_bin_comparison": final_comparison,
        "artifacts": _artifact_records(
            root,
            [
                manifest_path,
                camera_contract_copy,
                smoke_plan_path,
                *[Path(item["path"]) for item in image_records],
                *[final_model_dir / name for name in ("cameras.txt", "images.txt", "points3D.txt", "cameras.bin", "images.bin", "points3D.bin")],
                *[Path(item[key]) for item in command_results for key in ("stdout_log", "stderr_log")],
            ],
        ),
    }
