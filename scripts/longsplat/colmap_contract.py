"""CPU COLMAP command/evidence contracts with per-video camera binding."""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .pipeline_contract import PipelineBlocked, sha256_file, stable_sha256, write_text_once


COLMAP_CONTRACT_SCHEMA = "colmap-contract-v1"
INTRINSICS_EVIDENCE_SCHEMA = "intrinsics-evidence-v1"
SUPPORTED_CAMERA_MODELS = {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "PINHOLE"}
_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_POSITIVE_INTEGER = re.compile(r"^[1-9][0-9]*$")


class CameraPriorError(PipelineBlocked):
    """A camera prior is absent, malformed, or bound to another source video."""


@dataclass(frozen=True)
class CameraPrior:
    source_video_sha256: str
    model: str
    params: tuple[float, ...]
    calibration_width: int
    calibration_height: int
    coordinate_space: str
    source: str = "trusted_camera_prior"

    def __post_init__(self) -> None:
        if not self.source_video_sha256 or len(self.source_video_sha256) != 64:
            raise CameraPriorError("camera prior must contain a current source_video_sha256")
        if self.model not in SUPPORTED_CAMERA_MODELS:
            raise CameraPriorError(f"unsupported camera prior model: {self.model}")
        expected = _camera_param_count(self.model)
        if len(self.params) != expected or not all(math.isfinite(value) for value in self.params):
            raise CameraPriorError(f"{self.model} camera prior requires {expected} finite params")
        if self.calibration_width <= 0 or self.calibration_height <= 0:
            raise CameraPriorError("camera prior calibration dimensions must be positive")
        if self.coordinate_space != "canonical_pixels_v1":
            raise CameraPriorError("camera prior coordinate_space must be canonical_pixels_v1")
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params
        else:
            fx = fy = self.params[0]
            cx, cy = self.params[1:3]
        if fx <= 0 or fy <= 0:
            raise CameraPriorError("camera prior focal lengths must be positive")
        if not (-0.5 * self.calibration_width <= cx <= 1.5 * self.calibration_width):
            raise CameraPriorError("camera prior principal point cx is unreasonable")
        if not (-0.5 * self.calibration_height <= cy <= 1.5 * self.calibration_height):
            raise CameraPriorError("camera prior principal point cy is unreasonable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_video_sha256": self.source_video_sha256,
            "model": self.model,
            "params": list(self.params),
            "calibration_width": self.calibration_width,
            "calibration_height": self.calibration_height,
            "coordinate_space": self.coordinate_space,
            "source": self.source,
        }


def load_camera_prior(
    value: Mapping[str, Any],
    *,
    source_video_sha256: str,
    canonical_width: int | None = None,
    canonical_height: int | None = None,
) -> CameraPrior:
    bound = value.get("source_video_sha256", value.get("source_sha256"))
    if bound != source_video_sha256:
        raise CameraPriorError(
            "camera prior source_video_sha256 does not match the current video"
        )
    model = value.get("model", "SIMPLE_RADIAL")
    params = value.get("params")
    calibration_width = value.get("calibration_width", value.get("width"))
    calibration_height = value.get("calibration_height", value.get("height"))
    coordinate_space = value.get("coordinate_space")
    if not isinstance(model, str) or not isinstance(params, (list, tuple)):
        raise CameraPriorError("camera prior requires model and params")
    if not isinstance(calibration_width, int) or not isinstance(calibration_height, int):
        raise CameraPriorError("camera prior requires calibration_width/calibration_height")
    try:
        normalized_params = tuple(float(item) for item in params)
    except (TypeError, ValueError) as exc:
        raise CameraPriorError("camera prior params must be numeric") from exc
    prior = CameraPrior(
        source_video_sha256=source_video_sha256,
        model=model,
        params=normalized_params,
        calibration_width=calibration_width,
        calibration_height=calibration_height,
        coordinate_space=str(coordinate_space or ""),
        source=str(value.get("source", "trusted_camera_prior")),
    )
    validate_camera_prior_for_canonical(
        prior,
        canonical_width=canonical_width,
        canonical_height=canonical_height,
    )
    return prior


def validate_camera_prior_for_canonical(
    prior: CameraPrior,
    *,
    canonical_width: int | None,
    canonical_height: int | None,
) -> None:
    if canonical_width is None or canonical_height is None:
        return
    if (prior.calibration_width, prior.calibration_height) != (canonical_width, canonical_height):
        raise CameraPriorError(
            "camera prior calibration dimensions do not match canonical pixels; "
            "automatic rotation/scaling of K is not supported"
        )


@dataclass(frozen=True)
class ColmapCommand:
    stage: str
    argv: tuple[str, ...]
    cwd: str

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "argv": list(self.argv), "cwd": self.cwd}


def build_colmap_commands(
    *,
    colmap: str | Path,
    workspace: str | Path,
    image_dir: str | Path,
    source_video_sha256: str,
    canonical_media_sha256: str,
    canonical_width: int,
    canonical_height: int,
    camera_model: str = "SIMPLE_RADIAL",
    camera_prior: CameraPrior | None = None,
    matching: str = "sequential",
    sequence_overlap: int = 10,
    mapper_component: str | Path | None = None,
) -> dict[str, Any]:
    """Return exact argv for sparse SfM and explicit undistortion stages."""

    if camera_model not in SUPPORTED_CAMERA_MODELS:
        raise PipelineBlocked(f"unsupported COLMAP camera model: {camera_model}")
    if len(canonical_media_sha256) != 64 or canonical_width <= 0 or canonical_height <= 0:
        raise PipelineBlocked("COLMAP contract requires canonical media SHA and dimensions")
    if camera_prior is not None and camera_prior.source_video_sha256 != source_video_sha256:
        raise CameraPriorError("camera prior is bound to a different source SHA")
    if camera_prior is not None and camera_prior.model != camera_model:
        raise CameraPriorError(
            f"camera prior model {camera_prior.model} does not match requested {camera_model}"
        )
    if camera_prior is not None:
        validate_camera_prior_for_canonical(
            camera_prior,
            canonical_width=canonical_width,
            canonical_height=canonical_height,
        )
    root = Path(workspace).resolve()
    images = Path(image_dir).resolve()
    database = root / "database.db"
    mapper = root / "mapper"
    ba = root / "bundle_adjusted"
    evidence = root / "evidence"
    ba_txt = evidence / "bundle_adjusted_txt"
    undistorted = root / "undistorted"
    undistorted_txt = evidence / "undistorted_txt"
    cwd = str(root)
    feature = [
        str(colmap),
        "feature_extractor",
        "--database_path",
        str(database),
        "--image_path",
        str(images),
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_model",
        camera_model,
        "--SiftExtraction.use_gpu",
        "0",
    ]
    camera_params_source = "colmap_estimate"
    if camera_prior is not None:
        feature.extend(["--ImageReader.camera_params", ",".join(str(item) for item in camera_prior.params)])
        camera_params_source = camera_prior.source
    if matching == "sequential":
        match = [
            str(colmap),
            "sequential_matcher",
            "--database_path",
            str(database),
            "--SequentialMatching.overlap",
            str(sequence_overlap),
            "--SiftMatching.use_gpu",
            "0",
        ]
    elif matching == "exhaustive":
        match = [
            str(colmap),
            "exhaustive_matcher",
            "--database_path",
            str(database),
            "--SiftMatching.use_gpu",
            "0",
        ]
    else:
        raise PipelineBlocked(f"unsupported COLMAP matching mode: {matching}")
    component_path = (
        Path(mapper_component).resolve()
        if mapper_component is not None
        else mapper / "<component-id-required>"
    )
    commands = [
        ColmapCommand("feature_extractor", tuple(feature), cwd),
        ColmapCommand("matching", tuple(match), cwd),
        ColmapCommand(
            "mapper",
            (
                str(colmap),
                "mapper",
                "--database_path",
                str(database),
                "--image_path",
                str(images),
                "--output_path",
                str(mapper),
            ),
            cwd,
        ),
    ]
    commands.extend(
        [
            ColmapCommand(
                "bundle_adjuster",
                (
                    str(colmap),
                    "bundle_adjuster",
                    "--input_path",
                    str(component_path),
                    "--output_path",
                    str(ba),
                    "--BundleAdjustment.refine_focal_length",
                    "1",
                    "--BundleAdjustment.refine_principal_point",
                    "1",
                    "--BundleAdjustment.refine_extra_params",
                    "1",
                ),
                cwd,
            ),
            ColmapCommand(
                "model_converter_ba",
                (
                    str(colmap),
                    "model_converter",
                    "--input_path",
                    str(ba),
                    "--output_path",
                    str(ba_txt),
                    "--output_type",
                    "TXT",
                ),
                cwd,
            ),
            ColmapCommand(
                "image_undistorter",
                (
                    str(colmap),
                    "image_undistorter",
                    "--image_path",
                    str(images),
                    "--input_path",
                    str(ba),
                    "--output_path",
                    str(undistorted),
                    "--output_type",
                    "COLMAP",
                ),
                cwd,
            ),
            ColmapCommand(
                "model_converter_undistorted",
                (
                    str(colmap),
                    "model_converter",
                    "--input_path",
                    str(undistorted / "sparse"),
                    "--output_path",
                    str(undistorted_txt),
                    "--output_type",
                    "TXT",
                ),
                cwd,
            ),
        ]
    )
    plan = {
        "schema_version": COLMAP_CONTRACT_SCHEMA,
        "source_video_sha256": source_video_sha256,
        "canonical_media_sha256": canonical_media_sha256,
        "canonical_width": canonical_width,
        "canonical_height": canonical_height,
        "single_camera": True,
        "camera_model_requested": camera_model,
        "camera_params_source": camera_params_source,
        "camera_prior": None if camera_prior is None else camera_prior.to_dict(),
        "compute_mode": "cpu",
        "compute_contract": {
            "sift_extraction_use_gpu": 0,
            "sift_matching_use_gpu": 0,
            "gpu_hardware_capability": "not_probed",
        },
        "paths": {
            "database": str(database),
            "mapper_output_root": str(mapper),
            "mapper_component": None if mapper_component is None else str(component_path),
            "mapper_component_argv": str(component_path),
            "bundle_adjusted_binary": str(ba),
            "bundle_adjusted_txt": str(ba_txt),
            "undistorted_root": str(undistorted),
            "undistorted_images": str(undistorted / "images"),
            "undistorted_sparse_binary": str(undistorted / "sparse"),
            "undistorted_sparse_txt": str(undistorted_txt),
        },
        "component_discovery": {
            "required": mapper_component is None,
            "policy": "exactly_one_mapper_component; no automatic merge",
            "placeholder_argv": mapper_component is None,
        },
        "ba_refine_flags": {
            "focal_length": True,
            "principal_point": True,
            "extra_params": True,
        },
        "sparse_sfm_required": True,
        "dense_mvs": {"enabled": False, "blocking": False, "next_stage": "future"},
        "commands": [command.to_dict() for command in commands],
    }
    plan["contract_sha256"] = stable_sha256(plan)
    return plan


def enumerate_model_components(root: str | Path) -> list[Path]:
    """Enumerate actual COLMAP model component directories, never inventing ``0``."""

    directory = Path(root).resolve()
    if not directory.is_dir():
        return []
    components: list[Path] = []
    for candidate in sorted(directory.iterdir()):
        if candidate.is_symlink():
            raise PipelineBlocked(f"COLMAP model component is symlinked: {candidate}")
        if not candidate.is_dir():
            continue
        has_camera = (candidate / "cameras.bin").is_file() or (candidate / "cameras.txt").is_file()
        has_images = (candidate / "images.bin").is_file() or (candidate / "images.txt").is_file()
        has_points = (candidate / "points3D.bin").is_file() or (candidate / "points3D.txt").is_file()
        if has_camera and has_images and has_points:
            components.append(candidate)
    return components


def require_single_model_component(root: str | Path) -> Path:
    components = enumerate_model_components(root)
    if len(components) != 1:
        raise PipelineBlocked(
            f"COLMAP mapper produced {len(components)} model components; "
            "首版只接受 exactly one and does not merge components"
        )
    return components[0]


def run_colmap_command(
    command: ColmapCommand,
    *,
    tool_identity: Mapping[str, Any],
    log_dir: str | Path,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run exactly one command and return evidence suitable for an attempt record."""

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        result = runner(
            list(command.argv),
            cwd=command.cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        exit_code = int(getattr(result, "returncode", 1))
        stdout = str(getattr(result, "stdout", "") or "")
        stderr = str(getattr(result, "stderr", "") or "")
    except OSError as exc:
        exit_code, stdout, stderr = 127, "", str(exc)
    stdout_path = directory / f"{command.stage}.stdout.log"
    stderr_path = directory / f"{command.stage}.stderr.log"
    write_text_once(stdout_path, stdout)
    write_text_once(stderr_path, stderr)
    return {
        "stage": command.stage,
        "argv": list(command.argv),
        "cwd": command.cwd,
        "tool_identity": dict(tool_identity),
        "exit_code": exit_code,
        "status": "passed" if exit_code == 0 else "failed",
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stdout_sha256": stable_sha256({"stdout": stdout}),
        "stderr_sha256": stable_sha256({"stderr": stderr}),
    }


def _camera_param_count(model: str) -> int:
    return {"SIMPLE_PINHOLE": 3, "SIMPLE_RADIAL": 4, "PINHOLE": 4}[model]


def parse_cameras_text(path: str | Path) -> list[dict[str, Any]]:
    cameras: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) < 4:
            continue
        model = fields[1]
        if model not in SUPPORTED_CAMERA_MODELS:
            continue
        expected = _camera_param_count(model)
        if len(fields) < 4 + expected:
            raise PipelineBlocked(f"camera record has too few params: {line}")
        cameras.append(
            {
                "camera_id": int(fields[0]),
                "model": model,
                "width": int(fields[2]),
                "height": int(fields[3]),
                "params": [float(item) for item in fields[4 : 4 + expected]],
            }
        )
    if not cameras:
        raise PipelineBlocked(f"no supported COLMAP cameras found in {path}")
    return cameras


def _positive_integer(value: str, *, field: str, line_number: int, path: Path) -> int:
    if not _POSITIVE_INTEGER.fullmatch(value):
        raise PipelineBlocked(
            f"COLMAP images.txt {field} must be a positive integer at {path}:{line_number}"
        )
    return int(value)


def _finite_float(value: str, *, field: str, line_number: int, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineBlocked(
            f"COLMAP images.txt {field} must be numeric at {path}:{line_number}"
        ) from exc
    if not math.isfinite(result):
        raise PipelineBlocked(
            f"COLMAP images.txt {field} must be finite at {path}:{line_number}"
        )
    return result


def parse_colmap_image_pose_records(path: str | Path) -> list[dict[str, Any]]:
    """Parse only strict COLMAP image-pose headers from a two-line TXT model.

    COLMAP writes one pose header followed by one arbitrary-length POINTS2D
    line.  The parser consumes that second line as a second-line record before
    looking for the next header.  This state is important: a POINTS2D line is
    not reinterpreted as a pose merely because some of its numeric tokens look
    like a quaternion or an ID.
    """

    path = Path(path).resolve()
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        line_number = index + 1
        text = lines[index].strip()
        if text.startswith("#") or not text:
            index += 1
            continue
        if not text:
            index += 1
            continue
        fields = text.split()
        if len(fields) != 10:
            raise PipelineBlocked(f"malformed COLMAP pose header at {path}:{line_number}")
        try:
            image_id = _positive_integer(fields[0], field="IMAGE_ID", line_number=line_number, path=path)
            qvec = tuple(
                _finite_float(value, field=f"Q{i}", line_number=line_number, path=path)
                for i, value in enumerate(fields[1:5])
            )
            tvec = tuple(
                _finite_float(value, field=f"T{i}", line_number=line_number, path=path)
                for i, value in enumerate(fields[5:8])
            )
            camera_id = _positive_integer(fields[8], field="CAMERA_ID", line_number=line_number, path=path)
            name = fields[9]
            name_path = Path(name)
            if (
                not name
                or name_path.name != name
                or "/" in name
                or "\\" in name
                or name_path.suffix.lower() not in _IMAGE_EXTENSIONS
            ):
                raise PipelineBlocked(
                    f"COLMAP image NAME must be a unique staged image basename at {path}:{line_number}"
                )
        except PipelineBlocked:
            raise
        except (TypeError, ValueError) as exc:
            raise PipelineBlocked(f"malformed COLMAP pose header at {path}:{line_number}") from exc
        if image_id in seen_ids:
            raise PipelineBlocked(f"duplicate COLMAP IMAGE_ID {image_id} at {path}:{line_number}")
        if name in seen_names:
            raise PipelineBlocked(f"duplicate COLMAP image NAME {name} at {path}:{line_number}")
        seen_ids.add(image_id)
        seen_names.add(name)
        records.append(
            {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
        )
        points_index = index + 1
        if points_index >= len(lines):
            raise PipelineBlocked(f"COLMAP images.txt missing POINTS2D row after header at {path}:{line_number}")
        points_text = lines[points_index].strip()
        if points_text.startswith("#"):
            raise PipelineBlocked(f"COLMAP images.txt comment cannot replace POINTS2D row at {path}:{points_index + 1}")
        # A valid pose header in this position means the required second line
        # was omitted.  A POINTS2D payload is otherwise opaque here (the full
        # parser validates its triples); an empty payload is valid.
        point_fields = points_text.split()
        if len(point_fields) == 10 and _POSITIVE_INTEGER.fullmatch(point_fields[0]) and _POSITIVE_INTEGER.fullmatch(point_fields[8]) and Path(point_fields[9]).suffix.lower() in _IMAGE_EXTENSIONS:
            raise PipelineBlocked(f"COLMAP images.txt header replaced required POINTS2D row at {path}:{points_index + 1}")
        index += 2
    return records


def _parse_colmap_pose_headers(path: Path) -> list[dict[str, Any]]:
    """Compatibility wrapper for the shared strict two-line parser."""

    return parse_colmap_image_pose_records(path)


def _count_colmap_images(path: Path) -> tuple[int, list[str]]:
    records = _parse_colmap_pose_headers(path)
    names = [str(record["name"]) for record in records]
    return len(records), names


def _summary(values: Sequence[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    middle = len(finite) // 2
    median = finite[middle] if len(finite) % 2 else (finite[middle - 1] + finite[middle]) / 2.0
    return {
        "count": len(finite),
        "min": finite[0],
        "max": finite[-1],
        "mean": sum(finite) / len(finite),
        "median": median,
    }


def _parse_points(path: Path) -> tuple[list[float], list[int], int]:
    errors: list[float] = []
    track_lengths: list[int] = []
    if not path.is_file():
        return errors, track_lengths, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        fields = text.split()
        if len(fields) < 8:
            continue
        try:
            error = float(fields[7])
        except (TypeError, ValueError):
            continue
        if math.isfinite(error):
            errors.append(error)
        track_lengths.append(max(0, (len(fields) - 8) // 2))
    return errors, track_lengths, sum(track_lengths)


def _quaternion_rotation_angle(first: Sequence[float], second: Sequence[float]) -> float:
    first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
    second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
    if first_norm <= 0 or second_norm <= 0:
        return float("nan")
    dot = abs(sum(float(a) * float(b) for a, b in zip(first, second)) / (first_norm * second_norm))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _camera_center(qvec: Sequence[float], tvec: Sequence[float]) -> tuple[float, float, float]:
    qw, qx, qy, qz = (float(value) for value in qvec)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0 or not math.isfinite(norm):
        return (float("nan"), float("nan"), float("nan"))
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    r00, r01, r02 = 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)
    r10, r11, r12 = 2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)
    r20, r21, r22 = 2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)
    tx, ty, tz = (float(value) for value in tvec)
    return (
        -(r00 * tx + r10 * ty + r20 * tz),
        -(r01 * tx + r11 * ty + r21 * tz),
        -(r02 * tx + r12 * ty + r22 * tz),
    )


def _parse_colmap_poses(path: Path) -> tuple[list[str], list[tuple[float, float, float]], list[tuple[float, ...]]]:
    names: list[str] = []
    centers: list[tuple[float, float, float]] = []
    quaternions: list[tuple[float, ...]] = []
    for record in _parse_colmap_pose_headers(path):
        qvec = tuple(float(value) for value in record["qvec"])
        tvec = tuple(float(value) for value in record["tvec"])
        names.append(str(record["name"]))
        centers.append(_camera_center(qvec, tvec))
        quaternions.append(qvec)
    return names, centers, quaternions


def _refine_flags_from_argv(argv: Sequence[str] | None) -> dict[str, Any]:
    if argv is None:
        return {"focal_length": None, "principal_point": None, "distortion": None, "source": "not_recorded"}
    flags: dict[str, Any] = {"focal_length": None, "principal_point": None, "distortion": None, "source": "bundle_adjuster_argv"}
    mapping = {
        "--BundleAdjustment.refine_focal_length": "focal_length",
        "--BundleAdjustment.refine_principal_point": "principal_point",
        "--BundleAdjustment.refine_extra_params": "distortion",
    }
    for index, item in enumerate(argv):
        if item in mapping and index + 1 < len(argv):
            flags[mapping[item]] = str(argv[index + 1]) in {"1", "true", "True"}
    return flags


def collect_intrinsics_evidence(
    model_dir: str | Path,
    *,
    source_video_sha256: str,
    canonical_media_sha256: str,
    source: str = "colmap_sparse_model",
    reprojection_error: float | None = None,
    trajectory_count: int | None = None,
    component_count: int | None = None,
    mapper_component_count: int | None = None,
    component_inventory: Mapping[str, Any] | None = None,
    expected_image_names: Sequence[str] | None = None,
    refine_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse real TXT model evidence; no matcher A/B is run here."""

    root = Path(model_dir)
    cameras_path = root / "cameras.txt"
    cameras = parse_cameras_text(cameras_path)
    images_count, image_names = _count_colmap_images(root / "images.txt")
    point_errors, track_lengths, track_count = _parse_points(root / "points3D.txt")
    points_count = len(point_errors)
    pose_names, centers, quaternions = _parse_colmap_poses(root / "images.txt")
    center_steps = [
        math.sqrt(sum((right[index] - left[index]) ** 2 for index in range(3)))
        for left, right in zip(centers, centers[1:])
        if all(math.isfinite(value) for value in (*left, *right))
    ]
    rotation_steps = [
        _quaternion_rotation_angle(left, right)
        for left, right in zip(quaternions, quaternions[1:])
    ]
    rotation_steps = [value for value in rotation_steps if math.isfinite(value)]
    if component_count is None:
        sibling_components = enumerate_model_components(root.parent)
        component_count = len(sibling_components)
        component_count_source = "model_directory_siblings"
    else:
        component_count_source = "orchestrator_component_discovery"
    if mapper_component_count is None:
        mapper_component_count = int(component_count)
    if isinstance(mapper_component_count, bool) or int(mapper_component_count) < 1:
        raise PipelineBlocked("mapper_component_count must be a positive integer")
    if int(component_count) != 1:
        raise PipelineBlocked("active_model_component_count must equal one")
    models = []
    for camera in cameras:
        params = camera["params"]
        if camera["model"] == "PINHOLE":
            fx, fy, cx, cy = params
            distortion: list[float] = []
        elif camera["model"] == "SIMPLE_RADIAL":
            fx = fy = params[0]
            cx, cy, radial = params[1:]
            distortion = [radial]
        else:
            fx = fy = params[0]
            cx, cy = params[1:]
            distortion = []
        models.append(
            {
                "camera_id": camera["camera_id"],
                "model": camera["model"],
                "width": camera["width"],
                "height": camera["height"],
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "distortion": distortion,
            }
        )
    evidence = {
        "schema_version": INTRINSICS_EVIDENCE_SCHEMA,
        "binding": {
            "source_video_sha256": source_video_sha256,
            "canonical_media_sha256": canonical_media_sha256,
        },
        "source": source,
        "registered_image_count": images_count,
        "registered_image_ratio": (
            None if expected_image_names is None or not expected_image_names
            else images_count / len(expected_image_names)
        ),
        "registered_image_names": image_names,
        "component_count": int(component_count),
        "component_count_source": component_count_source,
        "mapper_component_count": int(mapper_component_count),
        "active_model_component_count": int(component_count),
        "component_inventory": None if component_inventory is None else dict(component_inventory),
        "reprojection_error_px": reprojection_error if reprojection_error is not None else (
            _summary(point_errors)["mean"] if point_errors else None
        ),
        "point_error_px": _summary(point_errors),
        "reprojection_error_distribution": _summary(point_errors),
        "track_count": track_count,
        "track_length": _summary(track_lengths),
        "track_length_distribution": _summary(track_lengths),
        "triangulated_point_count": points_count,
        "triangulation_angle_deg": {
            "available": False,
            "reason": "COLMAP TXT points3D does not retain the two-ray geometry needed for a reliable angle",
            "summary": _summary([]),
        },
        "trajectory_count": trajectory_count if trajectory_count is not None else len(pose_names),
        "trajectory": {
            "pose_count": len(pose_names),
            "pose_names": pose_names,
            "camera_centers": [
                [value if math.isfinite(value) else None for value in center]
                for center in centers
            ],
            "centers_finite": bool(centers) and all(
                math.isfinite(value) for center in centers for value in center
            ),
            "center_step": _summary(center_steps),
            "rotation_step_rad": _summary(rotation_steps),
            "continuous": bool(centers) and len(centers) == images_count,
        },
        "models": models,
        "camera_count": len(cameras),
        "refine_flags": _refine_flags_from_argv(refine_argv),
        "sparse_required": True,
        "dense_mvs_optional": True,
        "expected_image_names": None if expected_image_names is None else list(expected_image_names),
    }
    evidence["evidence_sha256"] = stable_sha256(evidence)
    return evidence


def evaluate_sparse_geometry(
    evidence: Mapping[str, Any],
    *,
    expected_image_names: Sequence[str] | None = None,
    min_registered_images: int = 2,
) -> dict[str, Any]:
    """Apply the non-negotiable sparse geometry gates without claiming acceptance."""

    failures: list[str] = []
    warnings: list[str] = []
    registered = int(evidence.get("registered_image_count", 0) or 0)
    if int(evidence.get("component_count", 0) or 0) != 1:
        failures.append("component_count_must_equal_one")
    if registered < min_registered_images:
        failures.append("too_few_registered_images")
    models = evidence.get("models")
    if not isinstance(models, list) or len(models) != 1:
        failures.append("exactly_one_camera_required")
    else:
        camera = models[0]
        for key in ("width", "height", "fx", "fy", "cx", "cy"):
            try:
                value = float(camera[key])
            except (KeyError, TypeError, ValueError):
                failures.append(f"invalid_camera_{key}")
                continue
            if not math.isfinite(value):
                failures.append(f"non_finite_camera_{key}")
        if float(camera.get("fx", 0.0)) <= 0 or float(camera.get("fy", 0.0)) <= 0:
            failures.append("non_positive_focal_length")
    if int(evidence.get("triangulated_point_count", 0) or 0) <= 0:
        failures.append("sparse_points_required")
    trajectory = evidence.get("trajectory", {})
    if not trajectory.get("centers_finite", False):
        failures.append("camera_trajectory_not_finite")
    if not trajectory.get("continuous", False):
        failures.append("camera_trajectory_not_continuous")
    if expected_image_names is not None:
        actual = sorted(str(name) for name in evidence.get("registered_image_names", []))
        expected = sorted(str(name) for name in expected_image_names)
        if actual != expected:
            failures.append("registered_names_do_not_match_staged_set")
    if evidence.get("reprojection_error_px") is None:
        warnings.append("reprojection_error_not_available")
    if not evidence.get("track_length", {}).get("count"):
        warnings.append("track_length_not_available")
    if not evidence.get("triangulation_angle_deg", {}).get("available", False):
        warnings.append("triangulation_angle_not_available")
    if not evidence.get("refine_flags", {}).get("source") == "bundle_adjuster_argv":
        warnings.append("bundle_adjustment_refine_flags_not_bound_to_argv")
    return {
        "computed_pass": not failures,
        "accepted": False,
        "hard_failures": failures,
        "warnings": warnings,
        "evidence_sha256": evidence.get("evidence_sha256"),
    }


def score_model_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pure CPU ranking hook; it records candidates but never runs A/B matching."""

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        registered = float(candidate.get("registered_image_count", 0) or 0)
        points = float(candidate.get("triangulated_point_count", 0) or 0)
        reprojection = candidate.get("reprojection_error_px")
        reprojection_score = 0.0 if reprojection is None else 1.0 / (1.0 + max(0.0, float(reprojection)))
        score = registered + math.log1p(max(0.0, points)) * 0.25 + reprojection_score
        ranked.append({"model": dict(candidate), "score": score})
    ranked.sort(
        key=lambda item: (-item["score"], str(item["model"].get("model_name", "")))
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked
