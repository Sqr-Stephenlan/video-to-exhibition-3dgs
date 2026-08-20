"""Strict centered-PINHOLE staging and intrinsic-matrix mathematics."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pipeline_contract import PipelineBlocked, sha256_file, stable_sha256, write_json_once


CAMERA_CONTRACT_SCHEMA = "camera-contract-v1"
PIXEL_CENTER_CONVENTION = "half_extent_center_xy_equals_width_over_2_height_over_2"


class CameraStagingBlocked(PipelineBlocked):
    """Staging cannot prove a safe LongSplat-compatible camera contract."""


@dataclass(frozen=True)
class CameraParameters:
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()
    camera_id: int | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise CameraStagingBlocked("camera dimensions must be positive")
        if not all(math.isfinite(float(value)) for value in (self.fx, self.fy, self.cx, self.cy)):
            raise CameraStagingBlocked("camera intrinsics must be finite")

    @classmethod
    def from_colmap_record(cls, record: Mapping[str, Any]) -> "CameraParameters":
        params = [float(value) for value in record["params"]]
        model = str(record["model"])
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
            distortion: tuple[float, ...] = ()
        elif model == "SIMPLE_RADIAL":
            fx = fy = params[0]
            cx, cy = params[1:3]
            distortion = (params[3],)
        elif model == "SIMPLE_PINHOLE":
            fx = fy = params[0]
            cx, cy = params[1:3]
            distortion = ()
        else:
            raise CameraStagingBlocked(f"unsupported camera model: {model}")
        return cls(
            model=model,
            width=int(record["width"]),
            height=int(record["height"]),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion=distortion,
            camera_id=record.get("camera_id"),
        )

    def K(self) -> np.ndarray:
        return np.asarray([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion": list(self.distortion),
            "camera_id": self.camera_id,
        }


def parse_colmap_camera(path: str | Path, camera_id: int | None = None) -> CameraParameters:
    """Parse one camera from a text model without depending on COLMAP Python."""

    from .colmap_contract import parse_cameras_text

    records = parse_cameras_text(path)
    if camera_id is not None:
        records = [record for record in records if record["camera_id"] == camera_id]
    if len(records) != 1:
        raise CameraStagingBlocked(f"expected one camera record, found {len(records)}")
    return CameraParameters.from_colmap_record(records[0])


def _orientation_affine(orientation: str, width: int, height: int) -> tuple[np.ndarray, int, int]:
    if orientation == "identity":
        return np.eye(3), width, height
    if orientation == "rotate90_cw":
        return np.asarray([[0.0, -1.0, float(height)], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), height, width
    if orientation == "rotate180":
        return np.asarray([[-1.0, 0.0, float(width)], [0.0, -1.0, float(height)], [0.0, 0.0, 1.0]]), width, height
    if orientation == "rotate270_ccw":
        return np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, float(width)], [0.0, 0.0, 1.0]]), height, width
    raise CameraStagingBlocked(f"unsupported pixel orientation: {orientation}")


def transform_intrinsics(
    camera: CameraParameters,
    *,
    orientation: str = "identity",
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    """Apply an explicit pixel affine transform and update K accordingly."""

    if scale_x <= 0 or scale_y <= 0 or not all(math.isfinite(value) for value in (scale_x, scale_y, offset_x, offset_y)):
        raise CameraStagingBlocked("pixel transform scale/offset must be finite and positive")
    orientation_matrix, oriented_width, oriented_height = _orientation_affine(
        orientation, camera.width, camera.height
    )
    scale_matrix = np.asarray(
        [[scale_x, 0.0, offset_x], [0.0, scale_y, offset_y], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    matrix = scale_matrix @ orientation_matrix
    # A right-angle image rotation also rotates the camera's image basis.  A
    # raw pixel homography (A @ K) would leave a 90-degree skew in K, which is
    # not representable by LongSplat's PINHOLE reader.  Express that basis
    # rotation in the extrinsic convention and keep the canonical K diagonal.
    if orientation == "identity":
        oriented_K = camera.K()
    elif orientation == "rotate90_cw":
        oriented_K = np.asarray(
            [[camera.fy, 0.0, camera.height - camera.cy],
             [0.0, camera.fx, camera.cx],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )
    elif orientation == "rotate180":
        oriented_K = np.asarray(
            [[camera.fx, 0.0, camera.width - camera.cx],
             [0.0, camera.fy, camera.height - camera.cy],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )
    else:  # rotate270_ccw
        oriented_K = np.asarray(
            [[camera.fy, 0.0, camera.cy],
             [0.0, camera.fx, camera.width - camera.cx],
             [0.0, 0.0, 1.0]],
            dtype=float,
        )
    transformed = scale_matrix @ oriented_K
    transformed /= transformed[2, 2]
    output_width = int(round(oriented_width * scale_x))
    output_height = int(round(oriented_height * scale_y))
    if output_width <= 0 or output_height <= 0:
        raise CameraStagingBlocked("pixel transform produced invalid output dimensions")
    details = {
        "orientation": orientation,
        "affine": matrix.tolist(),
        "camera_basis_rotation_absorbed": orientation != "identity",
        "camera_basis_convention": "standard_diagonal_PINHOLE",
        "scale_x": scale_x,
        "scale_y": scale_y,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "pixel_center_convention": PIXEL_CENTER_CONVENTION,
    }
    return transformed, output_width, output_height, details


def camera_from_K(
    K: np.ndarray,
    *,
    width: int,
    height: int,
    model: str = "PINHOLE",
) -> CameraParameters:
    return CameraParameters(
        model=model,
        width=width,
        height=height,
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
    )


def validate_centered_pinhole(
    camera: CameraParameters,
    *,
    source_video_sha256: str,
    canonical_media_sha256: str,
    frame_names: Sequence[str],
    expected_width: int,
    expected_height: int,
    tolerance_px: float = 0.5,
) -> dict[str, Any]:
    """Hard-stop unless the exact camera consumed by LongSplat is safe."""

    if camera.model != "PINHOLE":
        raise CameraStagingBlocked(f"LongSplat camera must be PINHOLE, got {camera.model}")
    if camera.distortion:
        raise CameraStagingBlocked("LongSplat camera must have no distortion")
    if camera.width != expected_width or camera.height != expected_height:
        raise CameraStagingBlocked(
            f"camera/image dimensions disagree: camera={camera.width}x{camera.height}, "
            f"canonical={expected_width}x{expected_height}"
        )
    if not source_video_sha256 or not canonical_media_sha256:
        raise CameraStagingBlocked("camera contract is missing source/canonical SHA binding")
    if not frame_names or len(set(frame_names)) != len(frame_names):
        raise CameraStagingBlocked("camera contract requires unique frame names")
    expected_cx = expected_width / 2.0
    expected_cy = expected_height / 2.0
    if abs(camera.cx - expected_cx) > tolerance_px or abs(camera.cy - expected_cy) > tolerance_px:
        raise CameraStagingBlocked(
            "principal point is not centered within the declared pixel-center tolerance"
        )
    if camera.fx <= 0 or camera.fy <= 0:
        raise CameraStagingBlocked("focal lengths must be positive")
    return {
        "schema_version": CAMERA_CONTRACT_SCHEMA,
        "status": "computed_pass",
        "accepted": False,
        "source_video_sha256": source_video_sha256,
        "canonical_media_sha256": canonical_media_sha256,
        "camera": camera.to_dict(),
        "pixel_center_convention": PIXEL_CENTER_CONVENTION,
        "principal_point_tolerance_px": tolerance_px,
        "expected_center": {"cx": expected_cx, "cy": expected_cy},
        "frame_names": list(frame_names),
        "frame_count": len(frame_names),
        "longsplat_compatible": True,
        "delivery_reachable": False,
    }


def _axis_centered_crop(
    input_length: int,
    principal_point: float,
    *,
    tolerance_px: float,
    minimum_retained_fraction: float,
    axis: str,
) -> dict[str, Any]:
    if input_length <= 0 or not math.isfinite(principal_point):
        raise CameraStagingBlocked(f"{axis} crop input is invalid")
    minimum_length = max(2, int(math.ceil(input_length * minimum_retained_fraction)))
    epsilon = 1e-9
    for output_length in range(input_length, minimum_length - 1, -1):
        low = principal_point - output_length / 2.0 - tolerance_px
        high = principal_point - output_length / 2.0 + tolerance_px
        first_start = max(0, int(math.ceil(low - epsilon)))
        last_start = min(input_length - output_length, int(math.floor(high + epsilon)))
        if first_start > last_start:
            continue
        # Maximum length is primary.  The minimum legal origin is the fixed
        # tie-break, preserving the earliest pixels when more than one ROI is
        # equally centered.
        start = first_start
        center_error = abs((principal_point - start) - output_length / 2.0)
        if center_error <= tolerance_px + epsilon:
            return {
                "input_length": input_length,
                "principal_point": principal_point,
                "output_length": output_length,
                "origin": start,
                "center_error_px": center_error,
                "retained_fraction": output_length / input_length,
                "axis": axis,
                "tie_break": "maximum_length_then_minimum_origin",
            }
    raise CameraStagingBlocked(
        f"no legal centered integer ROI on {axis} axis at retained fraction "
        f">= {minimum_retained_fraction}"
    )


def plan_centered_integer_crop(
    camera: CameraParameters,
    *,
    tolerance_px: float = 0.5,
    minimum_retained_fraction: float = 0.75,
) -> dict[str, Any]:
    """Plan a deterministic maximum-area integer crop without touching pixels."""

    if camera.model != "PINHOLE" or camera.distortion:
        raise CameraStagingBlocked("integer crop requires a distortion-free PINHOLE camera")
    if not 0.0 < minimum_retained_fraction <= 1.0:
        raise CameraStagingBlocked("minimum retained fraction must be in (0, 1]")
    if tolerance_px < 0 or not math.isfinite(tolerance_px):
        raise CameraStagingBlocked("crop tolerance must be finite and non-negative")
    x = _axis_centered_crop(
        camera.width,
        camera.cx,
        tolerance_px=tolerance_px,
        minimum_retained_fraction=minimum_retained_fraction,
        axis="x",
    )
    y = _axis_centered_crop(
        camera.height,
        camera.cy,
        tolerance_px=tolerance_px,
        minimum_retained_fraction=minimum_retained_fraction,
        axis="y",
    )
    crop = {
        "strategy": "identity" if x["output_length"] == camera.width and y["output_length"] == camera.height else "maximum-area-integer-crop-centered-v1",
        "input_width": camera.width,
        "input_height": camera.height,
        "x0": x["origin"],
        "y0": y["origin"],
        "width": x["output_length"],
        "height": y["output_length"],
        "retained_width_fraction": x["retained_fraction"],
        "retained_height_fraction": y["retained_fraction"],
        "retained_area_fraction": (x["output_length"] * y["output_length"]) / (camera.width * camera.height),
        "center_error_px": {"cx": x["center_error_px"], "cy": y["center_error_px"]},
        "axes": {"x": x, "y": y},
        "tie_break": "maximum_length_per_axis_then_minimum_origin",
        "tolerance_px": tolerance_px,
        "minimum_retained_fraction": minimum_retained_fraction,
    }
    warnings: list[str] = []
    if x["retained_fraction"] < 0.9 or y["retained_fraction"] < 0.9 or crop["retained_area_fraction"] < 0.9:
        warnings.append("crop_retained_fraction_below_0.9")
    crop["warnings"] = warnings
    return crop


def _decoded_pixel_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _canonical_media_identity(
    canonical_media_sha256: str,
    canonical_media_record: Mapping[str, Any] | None,
    canonical_media_record_path: str | Path | None,
) -> dict[str, Any]:
    binding = str(canonical_media_sha256)
    pixels: str | None = None
    record_binding: str | None = None
    if canonical_media_record is not None:
        declared_binding = canonical_media_record.get("binding", {}).get("canonical_media_sha256")
        if declared_binding is not None:
            record_binding = str(declared_binding)
            if record_binding != binding:
                raise CameraStagingBlocked("canonical media binding does not match camera staging input")
        pixels_value = canonical_media_record.get("canonical", {}).get("pixels_sha256")
        if pixels_value is not None:
            pixels = str(pixels_value)
    record_path: str | None = None
    record_sha: str | None = None
    if canonical_media_record_path is not None:
        path = Path(canonical_media_record_path).resolve()
        if not path.is_file():
            raise CameraStagingBlocked(f"canonical media record is missing: {path}")
        record_path = str(path)
        record_sha = sha256_file(path)
    return {
        "canonical_media_binding_sha256": binding,
        "canonical_media_record_binding_sha256": record_binding or binding,
        "canonical_media_pixels_sha256": pixels,
        "canonical_media_record_path": record_path,
        "canonical_media_record_file_sha256": record_sha,
        "canonical_media_sha256_semantics": "canonical_media-v1.binding.canonical_media_sha256; not decoded pixels",
    }


def _stage_output_path(input_path: Path, destination: Path | None, name: str) -> Path:
    if destination is None:
        return input_path.resolve()
    output_dir = destination / "canonical_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / name
    if output_path.exists() and not output_path.is_file():
        raise CameraStagingBlocked(f"camera staging output is not a file: {output_path}")
    return output_path


def _write_contract_once_or_verify(path: Path, contract: Mapping[str, Any]) -> None:
    """Make camera-contract resume idempotent without overwriting evidence."""

    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CameraStagingBlocked(f"existing camera contract is unreadable: {path}") from exc
        if existing != contract:
            raise CameraStagingBlocked(f"existing camera contract differs: {path}")
        return
    write_json_once(path, contract)


def stage_centered_pinhole(
    *,
    raw_camera: CameraParameters,
    undistorted_camera: CameraParameters,
    source_video_sha256: str,
    canonical_media_sha256: str,
    frame_records: Sequence[Mapping[str, Any]],
    output_dir: str | Path | None = None,
    pixel_transform: Mapping[str, Any] | None = None,
    warp_artifact_sha256: str | None = None,
    undistorter_argv: Sequence[str] | None = None,
    tolerance_px: float = 0.5,
    canonical_media_record: Mapping[str, Any] | None = None,
    canonical_media_record_path: str | Path | None = None,
    pixel_provenance: Mapping[str, Any] | None = None,
    minimum_retained_fraction: float = 0.75,
) -> dict[str, Any]:
    """Materialize a verified LongSplat image contract from real pixels.

    The only non-identity transform is the deterministic maximum-area integer
    crop.  It never resizes, interpolates, rotates, pads, or accepts a caller's
    claimed warp SHA as evidence.
    """

    if undistorted_camera.model != "PINHOLE" or undistorted_camera.distortion:
        raise CameraStagingBlocked("image_undistorter output must be distortion-free PINHOLE")
    if pixel_transform is not None or warp_artifact_sha256 is not None:
        raise CameraStagingBlocked(
            "external pixel_transform/warp_artifact_sha256 is not accepted; staging plans and verifies real pixels"
        )
    destination = None if output_dir is None else Path(output_dir).resolve()
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "canonical_images").mkdir(parents=True, exist_ok=True)
    crop = plan_centered_integer_crop(
        undistorted_camera,
        tolerance_px=tolerance_px,
        minimum_retained_fraction=minimum_retained_fraction,
    )
    crop_x0, crop_y0 = int(crop["x0"]), int(crop["y0"])
    output_width, output_height = int(crop["width"]), int(crop["height"])
    crop_affine = np.asarray(
        [[1.0, 0.0, -float(crop_x0)], [0.0, 1.0, -float(crop_y0)], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    canonical_K = crop_affine @ undistorted_camera.K()
    canonical_camera = camera_from_K(
        canonical_K,
        width=output_width,
        height=output_height,
        model="PINHOLE",
    )
    frames: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    try:
        import cv2
    except ImportError as exc:
        raise CameraStagingBlocked("OpenCV is required to materialize camera staging pixels") from exc
    for record in frame_records:
        name = record.get("staged_name", record.get("frame_id", record.get("name")))
        path = record.get("path", record.get("source_path"))
        if not isinstance(name, str) or not name or not isinstance(path, str) or Path(name).name != name:
            raise CameraStagingBlocked("frame record requires name and path")
        if name in seen_names:
            raise CameraStagingBlocked(f"duplicate camera staging frame name: {name}")
        seen_names.add(name)
        input_path = Path(path).resolve()
        if not input_path.is_file():
            raise CameraStagingBlocked(f"undistorted staged frame is missing: {input_path}")
        image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim < 2:
            raise CameraStagingBlocked(f"cannot decode undistorted staged frame: {input_path}")
        input_height, input_width = image.shape[:2]
        if (input_width, input_height) != (undistorted_camera.width, undistorted_camera.height):
            raise CameraStagingBlocked(
                f"undistorted frame dimensions do not match camera for {name}: "
                f"actual={input_width}x{input_height}, "
                f"camera={undistorted_camera.width}x{undistorted_camera.height}"
            )
        declared_sha = record.get("sha256")
        input_file_sha = sha256_file(input_path)
        if declared_sha is not None and str(declared_sha) != input_file_sha:
            raise CameraStagingBlocked(f"undistorted frame SHA mismatch: {input_path}")
        input_pixel_sha = _decoded_pixel_sha256(image)
        output_path = _stage_output_path(input_path, destination, name)
        if crop["strategy"] == "identity":
            output_image = image
            output_path = input_path
            output_file_sha = input_file_sha
            output_pixel_sha = input_pixel_sha
            materialized = False
        else:
            if destination is None:
                raise CameraStagingBlocked("output_dir is required to materialize an integer crop")
            roi = image[crop_y0 : crop_y0 + output_height, crop_x0 : crop_x0 + output_width]
            if roi.shape[:2] != (output_height, output_width):
                raise CameraStagingBlocked(f"crop ROI is outside the undistorted image: {name}")
            if output_path.exists():
                output_image = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
                if output_image is None or not np.array_equal(output_image, roi):
                    raise CameraStagingBlocked(f"existing camera staging artifact differs: {output_path}")
            else:
                if not cv2.imwrite(str(output_path), roi):
                    raise CameraStagingBlocked(f"failed to materialize camera staging pixel: {output_path}")
                output_image = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
            if output_image is None or not np.array_equal(output_image, roi):
                raise CameraStagingBlocked(f"camera staging ROI pixel verification failed: {output_path}")
            output_file_sha = sha256_file(output_path)
            output_pixel_sha = _decoded_pixel_sha256(output_image)
            if output_pixel_sha != _decoded_pixel_sha256(roi):
                raise CameraStagingBlocked(f"camera staging decoded pixel SHA failed: {output_path}")
            materialized = True
        frames.append(
            {
                "name": name,
                "staged_name": name,
                "path": str(output_path),
                "output_path": str(output_path),
                "input_path": str(input_path),
                "sha256": output_file_sha,
                "output_file_sha256": output_file_sha,
                "input_file_sha256": input_file_sha,
                "decoded_pixel_sha256": output_pixel_sha,
                "input_decoded_pixel_sha256": input_pixel_sha,
                "width": output_width,
                "height": output_height,
                "input_width": input_width,
                "input_height": input_height,
                "roi": {"x0": crop_x0, "y0": crop_y0, "width": output_width, "height": output_height},
                "provenance": str(record.get("provenance", "undistorted_output_frame")),
                "staging_provenance": "camera_staging_integer_crop_v1" if materialized else "camera_staging_identity_v1",
                "materialized": materialized,
            }
        )
    if not frames:
        raise CameraStagingBlocked("camera staging requires at least one undistorted frame")
    input_pixel_aggregate = stable_sha256(
        [{"name": frame["name"], "decoded_pixel_sha256": frame["input_decoded_pixel_sha256"]} for frame in frames]
    )
    final_pixel_aggregate = stable_sha256(
        [{"name": frame["name"], "decoded_pixel_sha256": frame["decoded_pixel_sha256"]} for frame in frames]
    )
    contract = validate_centered_pinhole(
        canonical_camera,
        source_video_sha256=source_video_sha256,
        canonical_media_sha256=canonical_media_sha256,
        frame_names=[frame["name"] for frame in frames],
        expected_width=output_width,
        expected_height=output_height,
        tolerance_px=tolerance_px,
    )
    media_identity = _canonical_media_identity(
        canonical_media_sha256,
        canonical_media_record,
        canonical_media_record_path,
    )
    pixel_context = dict(pixel_provenance or {})
    contract.update(
        {
            "canonical_media_binding_sha256": media_identity["canonical_media_binding_sha256"],
            "canonical_media_pixels_sha256": media_identity["canonical_media_pixels_sha256"],
            "canonical_media_record_sha256": media_identity["canonical_media_record_file_sha256"],
            "canonical_media_record_path": media_identity["canonical_media_record_path"],
            "canonical_media_sha256_semantics": media_identity["canonical_media_sha256_semantics"],
            "staging": {
                "strategy": crop["strategy"],
                "raw_camera": raw_camera.to_dict(),
                "undistorted_camera": undistorted_camera.to_dict(),
                "canonical_camera": canonical_camera.to_dict(),
                "K_raw": raw_camera.K().tolist(),
                "K_undistorted": undistorted_camera.K().tolist(),
                "crop_affine": crop_affine.tolist(),
                "K_canonical": canonical_camera.K().tolist(),
                "crop_roi": crop,
                "pixel_transform": {
                    "orientation": "identity",
                    "affine": crop_affine.tolist(),
                    "interpolation": "none",
                    "resize": False,
                    "rotation": False,
                    "padding": False,
                    "pixel_center_convention": PIXEL_CENTER_CONVENTION,
                },
                "warp_status": "identity_no_reencode" if crop["strategy"] == "identity" else "materialized_integer_crop",
                "image_undistorter_argv": None if undistorter_argv is None else list(undistorter_argv),
                "raw_to_undistorted": "COLMAP image_undistorter or equivalent recorded step",
                "undistorted_input_pixel_aggregate_sha256": input_pixel_aggregate,
                "final_staged_pixel_aggregate_sha256": final_pixel_aggregate,
                "no_interpolation": True,
                "no_padding": True,
                "coordinate_observation_rewrite": False,
                "contract_scope": "LongSplat camera/image staging only; not a rewritten generic COLMAP model",
            },
            "frames": frames,
            "pixel_provenance": {
                "source_video_sha256": source_video_sha256,
                "canonical_media_binding_sha256": media_identity["canonical_media_binding_sha256"],
                "canonical_media_pixels_sha256": media_identity["canonical_media_pixels_sha256"],
                "selected_canonical_frames": pixel_context.get("selected_canonical_frames", []),
                "colmap_input_frames": pixel_context.get("colmap_input_frames", []),
                "undistorted_output_frames": frames,
                "undistorted_input_pixel_aggregate_sha256": input_pixel_aggregate,
                "final_staged_pixel_aggregate_sha256": final_pixel_aggregate,
            },
        }
    )
    contract["contract_sha256"] = stable_sha256(contract)
    if output_dir is not None:
        _write_contract_once_or_verify(destination / "camera_contract-v1.json", contract)
    return contract
