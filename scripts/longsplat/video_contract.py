"""Versioned ffprobe and canonical-media contracts for one source video.

``display.rotation_degrees`` is the orthogonal transform that the contract
will apply to ``-noautorotate`` decoded pixels.  Raw signed metadata is kept
separately in ``raw_rotation_degrees``/``raw_orientation``; those fields are
not themselves the pixel transform.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .pipeline_contract import PipelineBlocked, sha256_file, stable_sha256


VIDEO_PROBE_SCHEMA = "video_probe-v1"
CANONICAL_MEDIA_SCHEMA = "canonical_media-v1"


class UnsupportedDisplayMetadata(PipelineBlocked):
    """The video has orientation/SAR metadata this contract cannot normalize."""


def _finite_float(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedDisplayMetadata(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise UnsupportedDisplayMetadata(f"{label} is not finite: {value!r}")
    return result


def parse_rational(value: Any, *, label: str, default: float | None = None) -> float | None:
    if value is None or value in ("", "N/A", "0/0"):
        return default
    if isinstance(value, str):
        text = value.strip()
        if ":" in text and "/" not in text:
            numerator, denominator = text.split(":", 1)
        elif "/" in text:
            numerator, denominator = text.split("/", 1)
        else:
            return _finite_float(text, label=label)
        try:
            denominator_value = float(denominator)
            numerator_value = float(numerator)
        except ValueError as exc:
            raise UnsupportedDisplayMetadata(f"invalid {label}: {value!r}") from exc
        if denominator_value == 0:
            raise UnsupportedDisplayMetadata(f"invalid {label}: {value!r}")
        result = numerator_value / denominator_value
    else:
        result = _finite_float(value, label=label)
    if result <= 0:
        raise UnsupportedDisplayMetadata(f"{label} must be positive: {value!r}")
    return result


def build_ffprobe_command(ffprobe: str | Path, source: str | Path) -> list[str]:
    """Build the complete metadata probe command recorded in the run."""

    return [
        str(ffprobe),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-show_chapters",
        str(Path(source).resolve()),
    ]


def _matrix_values(value: Any) -> list[float] | None:
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 9:
            return None
        values: list[float] = []
        for item in value:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                return None
        return values
    lines = [line.strip() for line in str(value).splitlines() if line.strip()]
    if not lines:
        return None
    has_colon = any(":" in line for line in lines)
    if has_colon:
        if len(lines) != 3:
            return None
        values = []
        for line in lines:
            match = re.fullmatch(r"[0-9A-Fa-f]+\s*:\s*(.*)", line)
            if match is None:
                return None
            row = match.group(1).split()
            if len(row) != 3 or any(re.fullmatch(number_pattern, item) is None for item in row):
                return None
            values.extend(float(item) for item in row)
        return values if len(values) == 9 else None
    tokens = " ".join(lines).split()
    if len(tokens) != 9 or any(re.fullmatch(number_pattern, item) is None for item in tokens):
        return None
    return [float(item) for item in tokens]


def _rotation_from_matrix(value: Any) -> int:
    """Return the applied canonical rotation for a raw FFmpeg matrix.

    The matrix encodes the metadata orientation.  The returned value is the
    inverse pixel transform needed after ``-noautorotate`` so that it matches
    FFmpeg's default autorotate output.
    """

    values = _matrix_values(value)
    if values is None or len(values) != 9:
        raise UnsupportedDisplayMetadata("display matrix is not a 3x3 numeric matrix")
    matrix = [values[0:3], values[3:6], values[6:9]]
    # FFmpeg stores the 2-D part as signed 16.16 values and the bottom-right
    # value in a different fixed-point scale.  Normalize only the 2-D part.
    scale = max(abs(number) for row in matrix[:2] for number in row[:2]) or 1.0
    a, b, c, d = (matrix[0][0] / scale, matrix[0][1] / scale,
                  matrix[1][0] / scale, matrix[1][1] / scale)
    eps = 0.02
    if abs(a - 1.0) < eps and abs(d - 1.0) < eps and abs(b) < eps and abs(c) < eps:
        return 0
    if abs(a) < eps and abs(d) < eps and b > 0.0 and c < 0.0:
        return 90
    if abs(a) < eps and abs(d) < eps and b < 0.0 and c > 0.0:
        return 270
    if abs(a + 1.0) < eps and abs(d + 1.0) < eps and abs(b) < eps and abs(c) < eps:
        return 180
    raise UnsupportedDisplayMetadata(f"unsupported display matrix: {value!r}")


def _applied_rotation_from_metadata(value: Any) -> tuple[int, float]:
    """Convert a raw signed rotation tag to the applied pixel transform."""

    raw = _finite_float(value, label="rotation")
    normalized = int(round(raw)) % 360
    if normalized not in (0, 90, 180, 270):
        raise UnsupportedDisplayMetadata(f"unsupported rotation: {value!r}")
    return (-normalized) % 360, raw


def _display_rotation(stream: Mapping[str, Any]) -> tuple[int, str, Any, float | None]:
    matrix_value: Any = None
    rotation_value: Any = None
    for side_data in stream.get("side_data_list", []) or []:
        if not isinstance(side_data, Mapping):
            continue
        if "displaymatrix" in side_data:
            matrix_value = side_data["displaymatrix"]
        elif "display_matrix" in side_data:
            matrix_value = side_data["display_matrix"]
        if "rotation" in side_data:
            rotation_value = side_data["rotation"]
    if matrix_value is not None:
        rotation = _rotation_from_matrix(matrix_value)
        raw_rotation = None
        if rotation_value is not None:
            tagged, raw_rotation = _applied_rotation_from_metadata(rotation_value)
            if tagged != rotation:
                raise UnsupportedDisplayMetadata("display matrix and rotation tag disagree")
        return rotation, "display_matrix", matrix_value, raw_rotation
    orientation_source = "none"
    if rotation_value is None:
        tags = stream.get("tags") or {}
        if isinstance(tags, Mapping) and "rotate" in tags:
            rotation_value = tags["rotate"]
            orientation_source = "rotation_tag"
        else:
            rotation_value = 0
    else:
        orientation_source = "rotation_tag"
    rotation, raw_rotation = _applied_rotation_from_metadata(rotation_value)
    return rotation, orientation_source, rotation_value, raw_rotation


def _frame_count(stream: Mapping[str, Any], duration: float, fps: float | None) -> tuple[int | None, str]:
    raw = stream.get("nb_frames")
    if raw not in (None, "", "N/A"):
        try:
            count = int(raw)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            return count, "stream.nb_frames"
    if duration > 0 and fps and fps > 0:
        return max(1, int(round(duration * fps))), "duration_times_fps_estimate"
    return None, "unknown"


def parse_ffprobe_payload(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path,
    source_sha256: str | None = None,
    source_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Parse ffprobe JSON and reject ambiguous display metadata."""

    streams = [
        item
        for item in payload.get("streams", [])
        if isinstance(item, Mapping) and item.get("codec_type") == "video"
    ]
    if not streams:
        raise PipelineBlocked("ffprobe returned no video stream")
    stream = streams[0]
    raw_format = payload.get("format", {}) or {}
    format_info = raw_format if isinstance(raw_format, Mapping) else {}
    source = Path(source_path).resolve()
    if source_sha256 is None and source.is_file():
        source_sha256 = sha256_file(source)
    if source_size_bytes is None and source.is_file():
        source_size_bytes = source.stat().st_size
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    coded_width = int(stream.get("coded_width") or width)
    coded_height = int(stream.get("coded_height") or height)
    if width <= 0 or height <= 0:
        raise PipelineBlocked("video stream has invalid encoded dimensions")
    duration = parse_rational(
        stream.get("duration") or format_info.get("duration"),
        label="duration",
        default=0.0,
    ) or 0.0
    average_fps = parse_rational(stream.get("avg_frame_rate"), label="avg_frame_rate")
    nominal_fps = parse_rational(stream.get("r_frame_rate"), label="r_frame_rate")
    fps = average_fps or nominal_fps
    frame_count, frame_count_source = _frame_count(stream, duration, fps)
    if duration <= 0 or fps is None or fps <= 0:
        raise PipelineBlocked("video duration/fps metadata is incomplete")
    sar = parse_rational(stream.get("sample_aspect_ratio") or "1:1", label="sample_aspect_ratio")
    dar = parse_rational(
        stream.get("display_aspect_ratio") or format_info.get("display_aspect_ratio"),
        label="display_aspect_ratio",
        default=(width * (sar or 1.0)) / height,
    )
    rotation, orientation_source, raw_orientation, raw_rotation = _display_rotation(stream)
    visible_width, visible_height = (height, width) if rotation in (90, 270) else (width, height)
    sar_value = sar or 1.0
    sar_width = max(1, int(round(width * sar_value)))
    canonical_width, canonical_height = (
        (height, sar_width) if rotation in (90, 270) else (sar_width, height)
    )
    vfr = bool(
        average_fps is not None
        and nominal_fps is not None
        and abs(average_fps - nominal_fps) > max(0.1, nominal_fps * 0.01)
    )
    probe = {
        "schema_version": VIDEO_PROBE_SCHEMA,
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            "size_bytes": source_size_bytes,
        },
        "stream": {
            "index": stream.get("index"),
            "codec_name": stream.get("codec_name"),
            "codec_long_name": stream.get("codec_long_name"),
            "pixel_format": stream.get("pix_fmt"),
            "encoded_width": width,
            "encoded_height": height,
            "coded_width": coded_width,
            "coded_height": coded_height,
            "visible_width": visible_width,
            "visible_height": visible_height,
        },
        "display": {
            "rotation_degrees": rotation,
            "rotation_semantics": "applied_to_noautorotate_pixels",
            "orientation_source": orientation_source,
            "raw_orientation": raw_orientation,
            "raw_rotation_degrees": raw_rotation,
            "sample_aspect_ratio": {"value": sar, "canonical": "1:1"},
            "display_aspect_ratio": dar,
        },
        "canonical": {
            "width": canonical_width,
            "height": canonical_height,
            "square_pixel_scale_x": sar_value,
            "square_pixel_scale_y": 1.0,
        },
        "timing": {
            "fps": fps,
            "average_fps": average_fps,
            "nominal_fps": nominal_fps,
            "duration_sec": duration,
            "frame_count": frame_count,
            "frame_count_source": frame_count_source,
            "vfr": vfr,
        },
        "color": {
            key: stream.get(key)
            for key in (
                "color_range",
                "color_space",
                "color_transfer",
                "color_primaries",
                "chroma_location",
            )
        },
        "format": {
            "format_name": format_info.get("format_name"),
            "format_long_name": format_info.get("format_long_name"),
            "bit_rate": format_info.get("bit_rate"),
        },
    }
    probe["probe_sha256"] = stable_sha256(probe)
    return probe


def parse_ffprobe_stdout(
    stdout: str,
    *,
    source_path: str | Path,
    source_sha256: str | None = None,
    source_size_bytes: int | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise PipelineBlocked(f"ffprobe did not return JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PipelineBlocked("ffprobe JSON root is not an object")
    return parse_ffprobe_payload(
        payload,
        source_path=source_path,
        source_sha256=source_sha256,
        source_size_bytes=source_size_bytes,
    )


def canonical_filter_graph(probe: Mapping[str, Any]) -> str:
    rotation = int(probe["display"]["rotation_degrees"])
    sar = float(probe["display"]["sample_aspect_ratio"].get("value") or 1.0)
    filters = {
        0: "setsar=1",
        90: "transpose=1,setsar=1",
        180: "hflip,vflip,setsar=1",
        270: "transpose=2,setsar=1",
    }
    try:
        orientation_filter = filters[rotation]
    except KeyError as exc:
        raise UnsupportedDisplayMetadata(f"rotation {rotation} has no canonical filter") from exc
    if not math.isclose(sar, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return f"scale=round(iw*{sar:.12g}):ih,{orientation_filter}"
    return orientation_filter


def candidate_sampling_plan(
    probe: Mapping[str, Any],
    *,
    target_fps: float,
    min_frames: int,
    max_frames: int,
) -> dict[str, Any]:
    """Derive a bounded CFR extraction plan before any pixels are decoded."""

    timing = probe.get("timing", {})
    if bool(timing.get("vfr")):
        raise UnsupportedDisplayMetadata(
            "VFR input is unsupported in the first slice: real per-frame timestamps are required"
        )
    duration = float(timing.get("duration_sec", 0.0) or 0.0)
    if duration <= 0 or target_fps <= 0 or min_frames <= 0 or max_frames < min_frames:
        raise PipelineBlocked("invalid duration/fps/frame budget for candidate extraction")
    budget = max(min_frames, min(max_frames, int(math.ceil(duration * target_fps))))
    # Sample exactly the duration-derived budget (subject to the explicit
    # upper cap) so a short video does not silently produce fewer than the
    # configured minimum candidates.
    candidate_fps = max(float(budget) / duration, 1.0 / duration)
    return {
        "mode": "cfr_sampled",
        "target_fps": float(target_fps),
        "candidate_fps": candidate_fps,
        "candidate_budget": budget,
        "candidate_cap": int(max_frames),
        "duration_sec": duration,
        "timestamp_plan": "index / candidate_fps; CFR input only",
        "filter_suffix": f"fps={candidate_fps:.12g}",
    }


def build_canonical_ffmpeg_command(
    ffmpeg: str | Path,
    source: str | Path,
    output: str | Path,
    probe: Mapping[str, Any],
) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-noautorotate",
        "-i",
        str(Path(source).resolve()),
        "-map",
        "0:v:0",
        "-vf",
        canonical_filter_graph(probe),
        "-fps_mode",
        "passthrough",
        "-c:v",
        "ffv1",
        "-an",
        str(Path(output).resolve()),
    ]


def build_frame_extract_command(
    ffmpeg: str | Path,
    source: str | Path,
    output_pattern: str | Path,
    probe: Mapping[str, Any],
    candidate_fps: float | None = None,
    candidate_cap: int | None = None,
) -> list[str]:
    if bool(probe.get("timing", {}).get("vfr")):
        raise UnsupportedDisplayMetadata(
            "VFR input cannot be extracted without real per-frame timestamps"
        )
    if candidate_fps is None or candidate_cap is None:
        raise PipelineBlocked("bounded candidate_fps and candidate_cap are required")
    if candidate_fps <= 0 or not math.isfinite(candidate_fps):
        raise PipelineBlocked("candidate_fps must be finite and positive")
    if int(candidate_cap) <= 0:
        raise PipelineBlocked("candidate_cap must be positive")
    filter_graph = f"{canonical_filter_graph(probe)},fps={candidate_fps:.12g}"
    return [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-noautorotate",
        "-i",
        str(Path(source).resolve()),
        "-map",
        "0:v:0",
        "-vf",
        filter_graph,
        "-vsync",
        "0",
        "-frames:v",
        str(int(candidate_cap)),
        "-start_number",
        "0",
        str(Path(output_pattern).resolve()),
    ]


def canonical_media_record(
    probe: Mapping[str, Any],
    *,
    source_path: str | Path,
    source_sha256: str,
    canonical_video_path: str | Path | None = None,
    canonical_video_sha256: str | None = None,
    canonical_pixels_sha256: str | None = None,
) -> dict[str, Any]:
    """Create the media contract consumed by every downstream stage."""

    canonical_dimensions = probe.get("canonical", {})
    width = int(canonical_dimensions.get("width", probe["stream"]["visible_width"]))
    height = int(canonical_dimensions.get("height", probe["stream"]["visible_height"]))
    record = {
        "schema_version": CANONICAL_MEDIA_SCHEMA,
        "status": "ready" if canonical_pixels_sha256 or canonical_video_sha256 else "planned",
        "source": {
            "path": str(Path(source_path).resolve()),
            "sha256": source_sha256,
            "size_bytes": probe["source"].get("size_bytes"),
            "video_probe_sha256": probe.get("probe_sha256"),
        },
        "canonical": {
            "width": width,
            "height": height,
            "sar": "1:1",
            "rotation_degrees": int(probe["display"]["rotation_degrees"]),
            "rotation_semantics": probe["display"].get(
                "rotation_semantics", "applied_to_noautorotate_pixels"
            ),
            "raw_rotation_degrees": probe["display"].get("raw_rotation_degrees"),
            "filter_graph": canonical_filter_graph(probe),
            "video_path": None if canonical_video_path is None else str(Path(canonical_video_path).resolve()),
            "video_sha256": canonical_video_sha256,
            "pixels_sha256": canonical_pixels_sha256,
            "pixel_contract": "actual decoded canonical pixels; no default dimensions",
        },
        "binding": {
            "source_video_sha256": source_sha256,
            "canonical_media_sha256": None,
        },
        "downstream_guard": {
            "requires_canonical_sha256": True,
            "requires_actual_width_height": True,
            "rejects_implicit_fixed_dimensions": True,
        },
    }
    record["binding"]["canonical_media_sha256"] = stable_sha256(record["canonical"])
    return record


def require_ready_canonical_media(record: Mapping[str, Any]) -> str:
    canonical = record.get("canonical", {})
    binding = record.get("binding", {})
    source_sha = binding.get("source_video_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise PipelineBlocked("canonical media source video SHA binding is required")
    digest = canonical.get("pixels_sha256") or canonical.get("video_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise PipelineBlocked("canonical media pixels/video SHA is required before downstream stages")
    width = int(canonical.get("width", 0))
    height = int(canonical.get("height", 0))
    if width <= 0 or height <= 0:
        raise PipelineBlocked("canonical media dimensions are missing or invalid")
    canonical_sha = binding.get("canonical_media_sha256")
    if not isinstance(canonical_sha, str) or len(canonical_sha) != 64:
        raise PipelineBlocked("canonical media SHA binding is required")
    return digest
