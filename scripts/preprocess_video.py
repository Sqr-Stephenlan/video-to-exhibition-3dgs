#!/usr/bin/env python3
"""Preprocess raw exhibition video into segments, selected frames, and a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

import preprocess_keyframes as pk


SCHEMA_VERSION = "2.0"
MANIFEST_FILENAME = "frames_manifest.json"
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

PRESETS = {
    "baseline": {
        "target_fps": 5.0,
        "max_long_edge": 1600,
        "segment_length_sec": 30.0,
        "segment_overlap_sec": 10.0,
    },
    "longsplat": {
        "target_fps": 10.0,
        "max_long_edge": 512,
        "segment_length_sec": 30.0,
        "segment_overlap_sec": 10.0,
    },
}

DEFAULT_PRESET = "baseline"
DEFAULT_SETTINGS = {
    "segment_method": "scene,time",
    "min_segment_sec": 2.0,
    "blur_threshold": 40.0,
    "overexposed_ratio": 0.6,
    "underexposed_ratio": 0.6,
    "duplicate_hash_threshold": 4,
    "duplicate_time_window_sec": 2.0,
    "max_selected_gap_sec": 2.0,
    "save_rejected": False,
    "frame_format": "jpg",
    "frame_source": "segment",
    **pk.KEYFRAME_DEFAULT_SETTINGS,
}
CONFIG_SETTING_KEYS = frozenset(
    {
        "preset",
        "target_fps",
        "max_long_edge",
        "segment_method",
        "segment_length_sec",
        "segment_overlap_sec",
        "min_segment_sec",
        "blur_threshold",
        "overexposed_ratio",
        "underexposed_ratio",
        "duplicate_hash_threshold",
        "duplicate_time_window_sec",
        "max_selected_gap_sec",
        "save_rejected",
        "frame_format",
        "frame_source",
        *pk.KEYFRAME_DEFAULT_SETTINGS,
    }
)
FLOAT_CONFIG_SETTINGS = frozenset(
    {
        "target_fps",
        "segment_length_sec",
        "segment_overlap_sec",
        "min_segment_sec",
        "blur_threshold",
        "overexposed_ratio",
        "underexposed_ratio",
        "duplicate_time_window_sec",
        "max_selected_gap_sec",
        "adaptive_blur_percentile",
        "selection_min_gap_sec",
        "selection_target_gap_sec",
        "selection_max_gap_sec",
        "duplicate_window_sec",
        "min_motion_inlier_ratio",
        "min_motion_grid_coverage",
        "max_motion_residual_diag_ratio",
        "max_median_displacement_diag_ratio",
        "max_affine_rotation_deg",
        "min_affine_scale",
        "max_affine_scale",
        "flow_forward_backward_max_error_px",
        "fundamental_ransac_threshold_px",
        "fundamental_ransac_confidence",
        "bridge_min_blur_ratio",
        "max_bridge_window_sec",
        "max_bridge_fraction",
        "adaptive_blur_percentile",
        "selection_min_gap_sec",
        "selection_target_gap_sec",
        "selection_max_gap_sec",
        "duplicate_window_sec",
        "min_motion_inlier_ratio",
        "min_motion_grid_coverage",
        "max_motion_residual_diag_ratio",
        "max_median_displacement_diag_ratio",
        "max_affine_rotation_deg",
        "min_affine_scale",
        "max_affine_scale",
    }
)
INT_CONFIG_SETTINGS = frozenset(
    {
        "max_long_edge",
        "duplicate_hash_threshold",
        "quality_analysis_long_edge",
        "flow_analysis_long_edge",
        "min_tracked_points",
    }
)
CHOICE_CONFIG_SETTINGS = {
    "preset": frozenset(PRESETS),
    "segment_method": frozenset({"scene", "time", "scene,time"}),
    "frame_format": frozenset({"jpg", "png"}),
    "frame_source": frozenset({"segment", "source"}),
    "keyframe_policy": frozenset({"legacy", "coverage_v1"}),
    "motion_model": frozenset({"affine", "auto"}),
}


class PreprocessError(RuntimeError):
    """User-facing preprocessing error."""


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    duration_sec: float
    fps: float
    width: int
    height: int
    codec: str
    size_bytes: int
    streams: list[dict[str, Any]]
    sha256: str | None = None
    rotation_degrees: int = 0
    is_vfr: bool = False


@dataclass(frozen=True)
class SegmentWindow:
    id: str
    index: int
    start_sec: float
    end_sec: float
    reason: str
    path: Path | None = None
    sha256: str | None = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def with_path(self, path: Path) -> "SegmentWindow":
        return SegmentWindow(
            id=self.id,
            index=self.index,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            reason=self.reason,
            path=path,
            sha256=sha256_file(path) if path.exists() else None,
        )


@dataclass(frozen=True)
class FrameRecord:
    id: str
    segment_id: str
    path: str | None
    timestamp_sec: float
    frame_index: int
    selected: bool
    blur_score: float
    overexposed_ratio: float
    underexposed_ratio: float
    duplicate_score: int | None
    reject_reasons: list[str]
    width: int = 0
    height: int = 0
    sha256: str | None = None
    motion_score: float | None = None
    matched_frame_id: str | None = None
    calibrated_blur_score: float | None = None
    keyframe: dict[str, Any] | None = None


def parse_fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def round_sec(value: float) -> float:
    return round(float(value), 3)


def relative_path(path: Path, base: Path | None = None) -> str:
    base = (base or Path.cwd()).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        raise PreprocessError(f"Path is outside the repository and cannot be written to a manifest: {resolved}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PreprocessError(f"Could not hash artifact {path}: {exc}") from exc
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tool_version(name: str) -> str:
    try:
        result = subprocess.run([name, "-version"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0].strip() if first_line else "unknown"


def repository_state(repo_root: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def manifest_command_args(arguments: list[str], repo_root: Path, source_path: Path) -> list[str]:
    normalized = list(arguments)
    path_options = {"--config", "--output-root"}
    expect_path = False
    for index, value in enumerate(normalized):
        if expect_path:
            normalized[index] = relative_path(Path(value), repo_root)
            expect_path = False
            continue
        if value in path_options:
            expect_path = True
            continue
        for option in path_options:
            prefix = f"{option}="
            if value.startswith(prefix):
                normalized[index] = prefix + relative_path(Path(value[len(prefix) :]), repo_root)
                break
        else:
            try:
                if not value.startswith("-") and Path(value).resolve() == source_path.resolve():
                    normalized[index] = relative_path(source_path, repo_root)
            except OSError:
                pass
    return ["./dev.sh", "python", "scripts/preprocess_video.py", *normalized]


def require_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    raise PreprocessError(
        f"Required tool '{name}' was not found on PATH. Install FFmpeg and ensure "
        f"both ffmpeg and ffprobe are available before running preprocessing."
    )


def run_command(command: list[str], description: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PreprocessError(f"{description} failed because '{command[0]}' was not found.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or f"exit code {exc.returncode}"
        raise PreprocessError(f"{description} failed: {detail}") from exc


def probe_video(source: Path) -> VideoMetadata:
    require_tool("ffprobe")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(source),
    ]
    result = run_command(command, "Probing video metadata")
    payload = json.loads(result.stdout)
    video_streams = [stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"]
    if not video_streams:
        raise PreprocessError(f"No video stream found in {source}.")

    stream = video_streams[0]
    format_info = payload.get("format", {})
    duration_raw = stream.get("duration") or format_info.get("duration") or 0.0
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError):
        duration = 0.0
    average_fps = parse_fraction(stream.get("avg_frame_rate"))
    nominal_fps = parse_fraction(stream.get("r_frame_rate"))
    fps = average_fps or nominal_fps
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if duration <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise PreprocessError(f"Incomplete video metadata for {source}.")

    rotation_degrees = 0
    rotation_value = stream.get("tags", {}).get("rotate")
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation_value = side_data["rotation"]
            break
    try:
        rotation_degrees = int(round(float(rotation_value or 0))) % 360
    except (TypeError, ValueError):
        rotation_degrees = 0

    return VideoMetadata(
        path=source,
        duration_sec=duration,
        fps=fps,
        width=width,
        height=height,
        codec=str(stream.get("codec_name") or ""),
        size_bytes=source.stat().st_size,
        streams=video_streams,
        sha256=sha256_file(source),
        rotation_degrees=rotation_degrees,
        is_vfr=bool(
            average_fps
            and nominal_fps
            and abs(average_fps - nominal_fps) > max(0.1, nominal_fps * 0.01)
        ),
    )


def even_dimension(value: int) -> int:
    value = max(2, int(value))
    return value if value % 2 == 0 else value - 1


def scaled_dimensions(width: int, height: int, max_long_edge: int) -> tuple[int, int]:
    if max_long_edge <= 0:
        raise ValueError("max_long_edge must be greater than zero")
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return even_dimension(width), even_dimension(height)
    scale = max_long_edge / long_edge
    return even_dimension(round(width * scale)), even_dimension(round(height * scale))


def normalize_video(source: Path, destination: Path, metadata: VideoMetadata, max_long_edge: int) -> VideoMetadata:
    require_tool("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    display_width, display_height = metadata.width, metadata.height
    if metadata.rotation_degrees in {90, 270}:
        display_width, display_height = display_height, display_width
    width, height = scaled_dimensions(display_width, display_height, max_long_edge)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"scale={width}:{height}:flags=lanczos",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    run_command(command, "Normalizing video")
    return probe_video(destination)


def validate_window_settings(segment_length_sec: float, segment_overlap_sec: float, min_segment_sec: float) -> None:
    if segment_length_sec <= 0:
        raise ValueError("segment_length_sec must be greater than zero")
    if segment_overlap_sec < 0:
        raise ValueError("segment_overlap_sec cannot be negative")
    if segment_overlap_sec >= segment_length_sec:
        raise ValueError("segment_overlap_sec must be less than segment_length_sec")
    if min_segment_sec < 0:
        raise ValueError("min_segment_sec cannot be negative")


def build_time_windows(
    duration_sec: float,
    segment_length_sec: float,
    segment_overlap_sec: float,
    min_segment_sec: float,
    offset_sec: float = 0.0,
) -> list[tuple[float, float]]:
    validate_window_settings(segment_length_sec, segment_overlap_sec, min_segment_sec)
    if duration_sec <= 0:
        return []

    if duration_sec <= segment_length_sec:
        return [(round_sec(offset_sec), round_sec(offset_sec + duration_sec))]

    windows: list[tuple[float, float]] = []
    step = segment_length_sec - segment_overlap_sec
    start = 0.0
    while start + segment_length_sec < duration_sec:
        windows.append((round_sec(offset_sec + start), round_sec(offset_sec + start + segment_length_sec)))
        start += step

    tail_start = start
    tail_duration = duration_sec - tail_start
    if tail_duration < min_segment_sec and windows:
        tail_start = max(0.0, duration_sec - segment_length_sec)
        shifted_tail = (round_sec(offset_sec + tail_start), round_sec(offset_sec + duration_sec))
        if len(windows) >= 2 and shifted_tail[0] <= windows[-2][1]:
            windows[-1] = shifted_tail
            return windows
        if shifted_tail[0] == windows[-1][0]:
            windows[-1] = shifted_tail
            return windows
    windows.append((round_sec(offset_sec + tail_start), round_sec(offset_sec + duration_sec)))
    return windows


def merge_short_segments(windows: list[tuple[float, float]], min_segment_sec: float) -> list[tuple[float, float]]:
    if not windows:
        return []

    ordered = sorted(windows)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if end <= start:
            continue
        if end - start < min_segment_sec and merged:
            prev_start, _ = merged[-1]
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_segment_sec:
        first_start, _ = merged[0]
        _, second_end = merged[1]
        merged = [(first_start, second_end), *merged[2:]]
    return [(round_sec(start), round_sec(end)) for start, end in merged]


def detect_scene_windows(video_path: Path, min_segment_sec: float) -> list[tuple[float, float]]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ImportError as exc:
        raise PreprocessError(
            "PySceneDetect is required for scene segmentation. Install requirements.txt "
            "or use --segment-method time."
        ) from exc

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video)
    scenes = scene_manager.get_scene_list()
    windows = [
        (round_sec(start.seconds), round_sec(end.seconds))
        for start, end in scenes
        if end.seconds > start.seconds
    ]

    if len(windows) <= 1:
        return []
    return merge_short_segments(windows, min_segment_sec)


def build_segment_windows(
    duration_sec: float,
    segment_method: str,
    segment_length_sec: float,
    segment_overlap_sec: float,
    min_segment_sec: float,
    normalized_path: Path | None = None,
) -> list[SegmentWindow]:
    methods = {part.strip() for part in segment_method.split(",") if part.strip()}
    if not methods or not methods <= {"scene", "time"}:
        raise ValueError("segment_method must be scene, time, or scene,time")

    raw_windows: list[tuple[float, float]] = []
    source_reason = "time"
    if "scene" in methods and normalized_path is not None:
        raw_windows = detect_scene_windows(normalized_path, min_segment_sec)
        source_reason = "scene"

    if not raw_windows:
        raw_windows = build_time_windows(duration_sec, segment_length_sec, segment_overlap_sec, min_segment_sec)
        source_reason = "time_fallback" if "scene" in methods else "time"

    split_windows: list[tuple[float, float, str]] = []
    for start, end in raw_windows:
        window_duration = end - start
        if window_duration > segment_length_sec:
            for split_start, split_end in build_time_windows(
                window_duration,
                segment_length_sec,
                segment_overlap_sec,
                min_segment_sec,
                offset_sec=start,
            ):
                split_windows.append((split_start, split_end, f"{source_reason}_split"))
        else:
            split_windows.append((round_sec(start), round_sec(end), source_reason))

    segments = []
    for index, (start, end, reason) in enumerate(split_windows, start=1):
        segments.append(
            SegmentWindow(
                id=f"segment_{index:04d}",
                index=index,
                start_sec=round_sec(start),
                end_sec=round_sec(end),
                reason=reason,
            )
        )
    return segments


def write_segment(normalized_path: Path, segment: SegmentWindow, destination: Path) -> SegmentWindow:
    require_tool("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{segment.start_sec:.3f}",
        "-i",
        str(normalized_path),
        "-t",
        f"{segment.duration_sec:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    run_command(command, f"Writing {segment.id}")
    return segment.with_path(destination)


def blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_ratios(frame: np.ndarray, low_threshold: int = 5, high_threshold: int = 250) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total = float(gray.size)
    under = float(np.count_nonzero(gray <= low_threshold)) / total
    over = float(np.count_nonzero(gray >= high_threshold)) / total
    return over, under


def average_hash(frame: np.ndarray, hash_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return resized > resized.mean()


def hash_distance(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))


def frame_filename(sample_index: int, timestamp_sec: float, frame_format: str = "jpg") -> str:
    extension = "png" if frame_format == "png" else "jpg"
    return f"frame_{sample_index:06d}_t{timestamp_sec:010.3f}.{extension}"


def write_image(path: Path, frame: np.ndarray, frame_format: str = "jpg") -> None:
    """Write an image through Python file I/O so Unicode Windows paths work."""
    if frame_format == "png":
        extension = ".png"
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        extension = ".jpg"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    encoded_ok, encoded = cv2.imencode(extension, frame, params)
    if not encoded_ok:
        raise PreprocessError(f"OpenCV could not encode frame for {path}.")
    try:
        path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise PreprocessError(f"Could not write frame image {path}: {exc}") from exc


def resize_frame(frame: np.ndarray, dimensions: tuple[int, int] | None) -> np.ndarray:
    if dimensions is None:
        return frame
    width, height = dimensions
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)


def rotate_frame(frame: np.ndarray, rotation_degrees: int) -> np.ndarray:
    rotation = rotation_degrees % 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def motion_difference(previous: np.ndarray | None, current: np.ndarray) -> float | None:
    if previous is None:
        return None
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    if previous_gray.shape != current_gray.shape:
        previous_gray = cv2.resize(previous_gray, (current_gray.shape[1], current_gray.shape[0]))
    return float(np.mean(cv2.absdiff(previous_gray, current_gray))) / 255.0


def sample_segment_frames(
    segment: SegmentWindow,
    selected_root: Path,
    rejected_root: Path,
    target_fps: float,
    blur_threshold: float,
    overexposed_ratio: float,
    underexposed_ratio: float,
    duplicate_hash_threshold: int,
    save_rejected: bool,
    repo_root: Path,
    frame_format: str = "jpg",
    frame_source: str = "segment",
    source_video: Path | None = None,
    resize_to: tuple[int, int] | None = None,
    source_rotation_degrees: int = 0,
    duplicate_time_window_sec: float = 2.0,
    max_selected_gap_sec: float = 2.0,
    manifest_selected_root: Path | None = None,
    manifest_rejected_root: Path | None = None,
) -> list[FrameRecord]:
    if segment.path is None:
        raise ValueError("segment.path is required for frame sampling")
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero")
    if frame_format not in {"jpg", "png"}:
        raise ValueError("frame_format must be jpg or png")
    if frame_source not in {"segment", "source"}:
        raise ValueError("frame_source must be segment or source")
    if frame_source == "source" and source_video is None:
        raise ValueError("source_video is required when frame_source is source")

    sampling_path = source_video if frame_source == "source" else segment.path
    cap = cv2.VideoCapture(str(sampling_path))
    try:
        if not cap.isOpened():
            raise PreprocessError(f"OpenCV could not open video for frame sampling: {sampling_path}.")
        if frame_source == "source" and hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

        segment_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        sample_every = 1 if segment_fps <= 0 else max(1, int(round(segment_fps / target_fps)))
        start_frame_index = 0
        end_frame_index: int | None = None
        if frame_source == "source" and segment_fps > 0:
            start_frame_index = max(0, int(round(segment.start_sec * segment_fps)))
            end_frame_index = max(start_frame_index, int(round(segment.end_sec * segment_fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)

        selected_hashes: list[tuple[float, str, np.ndarray]] = []
        records: list[FrameRecord] = []
        selected_dir = selected_root / segment.id
        rejected_dir = rejected_root / segment.id
        selected_dir.mkdir(parents=True, exist_ok=True)
        if save_rejected:
            rejected_dir.mkdir(parents=True, exist_ok=True)

        frame_number = 0
        source_frame_index = start_frame_index
        sample_index = 0
        previous_sampled_frame: np.ndarray | None = None
        last_selected_timestamp: float | None = None
        while True:
            if end_frame_index is not None and source_frame_index >= end_frame_index:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            absolute_frame_index = source_frame_index + 1
            source_frame_index += 1
            if (frame_number - 1) % sample_every != 0:
                continue

            if frame_source == "source":
                frame = rotate_frame(frame, source_rotation_degrees)
            frame = resize_frame(frame, resize_to if frame_source == "source" else None)
            sample_index += 1
            local_timestamp = 0.0 if segment_fps <= 0 else (frame_number - 1) / segment_fps
            timestamp = round_sec(segment.start_sec + local_timestamp)
            blur = blur_score(frame)
            over, under = exposure_ratios(frame)
            current_hash = average_hash(frame)
            current_frame_id = f"{segment.id}_frame_{sample_index:06d}"
            recent_hashes = [
                item for item in selected_hashes if timestamp - item[0] <= duplicate_time_window_sec
            ]
            duplicate_match = min(
                ((hash_distance(current_hash, item_hash), frame_id) for _, frame_id, item_hash in recent_hashes),
                default=None,
            )
            duplicate = duplicate_match[0] if duplicate_match else None
            matched_frame_id = duplicate_match[1] if duplicate_match else None
            motion = motion_difference(previous_sampled_frame, frame)
            previous_sampled_frame = frame.copy()

            reject_reasons = []
            if blur < blur_threshold:
                reject_reasons.append("blur")
            if over > overexposed_ratio:
                reject_reasons.append("overexposed")
            if under > underexposed_ratio:
                reject_reasons.append("underexposed")
            coverage_due = (
                last_selected_timestamp is not None
                and timestamp - last_selected_timestamp >= max_selected_gap_sec
            )
            if duplicate is not None and duplicate <= duplicate_hash_threshold and not coverage_due:
                reject_reasons.append("duplicate")

            selected = not reject_reasons
            filename = frame_filename(sample_index, timestamp, frame_format)
            output_path: Path | None
            if selected:
                output_path = selected_dir / filename
                selected_hashes.append((timestamp, current_frame_id, current_hash))
                last_selected_timestamp = timestamp
                write_image(output_path, frame, frame_format)
            elif save_rejected:
                output_path = rejected_dir / filename
                write_image(output_path, frame, frame_format)
            else:
                output_path = None

            logical_output_path = output_path
            if output_path is not None and selected and manifest_selected_root is not None:
                logical_output_path = manifest_selected_root / segment.id / filename
            elif output_path is not None and not selected and manifest_rejected_root is not None:
                logical_output_path = manifest_rejected_root / segment.id / filename

            records.append(
                FrameRecord(
                    id=current_frame_id,
                    segment_id=segment.id,
                    path=relative_path(logical_output_path, repo_root) if logical_output_path else None,
                    timestamp_sec=timestamp,
                    frame_index=absolute_frame_index if frame_source == "source" else frame_number,
                    selected=selected,
                    blur_score=round(float(blur), 3),
                    overexposed_ratio=round(float(over), 6),
                    underexposed_ratio=round(float(under), 6),
                    duplicate_score=duplicate,
                    reject_reasons=reject_reasons,
                    width=int(frame.shape[1]),
                    height=int(frame.shape[0]),
                    sha256=sha256_file(output_path) if output_path else None,
                    motion_score=round(float(motion), 6) if motion is not None else None,
                    matched_frame_id=matched_frame_id if "duplicate" in reject_reasons else None,
                )
            )
    finally:
        cap.release()
    return records


def _sampling_parameters(
    segment: SegmentWindow,
    target_fps: float,
    frame_source: str,
    source_video: Path | None,
) -> tuple[cv2.VideoCapture, float, int, int, int | None]:
    if segment.path is None:
        raise ValueError("segment.path is required for frame sampling")
    sampling_path = source_video if frame_source == "source" else segment.path
    cap = cv2.VideoCapture(str(sampling_path))
    if not cap.isOpened():
        cap.release()
        raise PreprocessError(f"OpenCV could not open video for frame sampling: {sampling_path}.")
    if frame_source == "source" and hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    segment_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    sample_every = 1 if segment_fps <= 0 else max(1, int(round(segment_fps / target_fps)))
    start_frame_index = 0
    end_frame_index: int | None = None
    if frame_source == "source" and segment_fps > 0:
        start_frame_index = max(0, int(round(segment.start_sec * segment_fps)))
        end_frame_index = max(start_frame_index, int(round(segment.end_sec * segment_fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    return cap, segment_fps, sample_every, start_frame_index, end_frame_index


def _iter_sampled_frames(
    segment: SegmentWindow,
    target_fps: float,
    frame_source: str,
    source_video: Path | None,
    source_rotation_degrees: int,
    resize_to: tuple[int, int] | None,
) -> Iterable[tuple[int, float, int, np.ndarray]]:
    cap, segment_fps, sample_every, start_frame_index, end_frame_index = _sampling_parameters(
        segment, target_fps, frame_source, source_video
    )
    try:
        frame_number = 0
        source_frame_index = start_frame_index
        sample_index = 0
        while True:
            if end_frame_index is not None and source_frame_index >= end_frame_index:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_number += 1
            absolute_frame_index = source_frame_index + 1
            source_frame_index += 1
            if (frame_number - 1) % sample_every != 0:
                continue
            if frame_source == "source":
                frame = rotate_frame(frame, source_rotation_degrees)
            frame = resize_frame(frame, resize_to if frame_source == "source" else None)
            sample_index += 1
            local_timestamp = 0.0 if segment_fps <= 0 else (frame_number - 1) / segment_fps
            yield (
                sample_index,
                round_sec(segment.start_sec + local_timestamp),
                absolute_frame_index if frame_source == "source" else frame_number,
                frame,
            )
    finally:
        cap.release()


def _coverage_segment_quality(
    segment: SegmentWindow,
    scanned: list[pk.ScannedFrame],
    decisions: list[pk.KeyframeDecision],
    policy: pk.KeyframePolicy,
) -> dict[str, Any]:
    selected_pairs = [(frame, decision) for frame, decision in zip(scanned, decisions) if decision.selected]
    selected = [frame for frame, _decision in selected_pairs]
    timestamps = [frame.timestamp_sec for frame in selected]
    max_gap = selected_frame_max_gap(segment, timestamps)
    failed_reasons: list[str] = []
    if not selected:
        failed_reasons.append("no_selected_frames")
    if max_gap > policy.max_gap_sec:
        failed_reasons.append("max_selected_gap_sec")
    bridge_count = sum(decision.bridge for decision in decisions if decision.selected)
    bridge_fraction = float(bridge_count) / float(len(selected)) if selected else 0.0
    max_bridge_window = 0.0
    run_start = 0
    while run_start < len(selected_pairs):
        if not selected_pairs[run_start][1].bridge:
            run_start += 1
            continue
        run_end = run_start
        while run_end + 1 < len(selected_pairs) and selected_pairs[run_end + 1][1].bridge:
            run_end += 1
        left = segment.start_sec if run_start == 0 else selected_pairs[run_start - 1][0].timestamp_sec
        right = segment.end_sec if run_end + 1 == len(selected_pairs) else selected_pairs[run_end + 1][0].timestamp_sec
        max_bridge_window = max(max_bridge_window, right - left)
        run_start = run_end + 1
    motion_model_counts = {"affine": 0, "fundamental": 0}
    for _frame, decision in selected_pairs:
        if decision.motion is not None and decision.motion.model in motion_model_counts:
            motion_model_counts[decision.motion.model] += 1
    if bridge_fraction > policy.max_bridge_fraction:
        failed_reasons.append("bridge_fraction")
    if policy.max_bridge_window_sec > 0 and max_bridge_window > policy.max_bridge_window_sec:
        failed_reasons.append("bridge_window_sec")
    return {
        "segment_id": segment.id,
        "total_frames": len(scanned),
        "selected_frames": len(selected),
        "rejected_frames": len(scanned) - len(selected),
        "component_count": 1 if selected else 0,
        "max_selected_gap_sec": max_gap,
        "bridge_frames": bridge_count,
        "bridge_fraction": round(bridge_fraction, 6),
        "max_consecutive_bridge_window_sec": round_sec(max_bridge_window),
        "motion_model_counts": motion_model_counts,
        "failed_reasons": failed_reasons,
    }


def selected_frame_max_gap(segment: SegmentWindow, timestamps: Iterable[float]) -> float:
    ordered = sorted(timestamps)
    if not ordered:
        return round_sec(segment.duration_sec)
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    gaps.extend([ordered[0] - segment.start_sec, segment.end_sec - ordered[-1]])
    return round_sec(max(gaps, default=0.0))


def sample_segment_frames_coverage(
    segment: SegmentWindow,
    selected_root: Path,
    rejected_root: Path,
    target_fps: float,
    save_rejected: bool,
    repo_root: Path,
    policy: pk.KeyframePolicy,
    frame_format: str = "jpg",
    frame_source: str = "segment",
    source_video: Path | None = None,
    resize_to: tuple[int, int] | None = None,
    source_rotation_degrees: int = 0,
    manifest_selected_root: Path | None = None,
    manifest_rejected_root: Path | None = None,
) -> tuple[list[FrameRecord], dict[str, Any]]:
    scanned: list[pk.ScannedFrame] = []
    for sample_index, timestamp, frame_index, frame in _iter_sampled_frames(
        segment, target_fps, frame_source, source_video, source_rotation_degrees, resize_to
    ):
        scanned.append(
            pk.ScannedFrame(
                id=f"{segment.id}_frame_{sample_index:06d}",
                segment_id=segment.id,
                sample_index=sample_index,
                timestamp_sec=timestamp,
                frame_index=frame_index,
                blur_score=blur_score(frame),
                calibrated_blur_score=pk.calibrated_blur_score(frame, policy.quality_analysis_long_edge),
                overexposed_ratio=exposure_ratios(frame)[0],
                underexposed_ratio=exposure_ratios(frame)[1],
                average_hash_bits=average_hash(frame),
                flow_gray=pk.make_analysis_gray(frame, policy.flow_analysis_long_edge),
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
            )
        )
    decisions = pk.select_coverage_frames(
        scanned,
        policy,
        segment_start_sec=segment.start_sec,
        segment_end_sec=segment.end_sec,
    )
    selected_dir = selected_root / segment.id
    rejected_dir = rejected_root / segment.id
    selected_dir.mkdir(parents=True, exist_ok=True)
    if save_rejected:
        rejected_dir.mkdir(parents=True, exist_ok=True)
    records: list[FrameRecord] = []
    second_pass = iter(
        _iter_sampled_frames(segment, target_fps, frame_source, source_video, source_rotation_degrees, resize_to)
    )
    for scanned_frame, decision in zip(scanned, decisions):
        try:
            sample_index, timestamp, frame_index, frame = next(second_pass)
        except StopIteration as exc:
            raise PreprocessError(
                f"Second-pass frame count mismatch for {segment.id}: expected {len(scanned)}"
            ) from exc
        if sample_index != scanned_frame.sample_index or abs(timestamp - scanned_frame.timestamp_sec) > 0.001:
            raise PreprocessError(
                f"Second-pass frame mismatch for {scanned_frame.id}: "
                f"expected sample {scanned_frame.sample_index} at {scanned_frame.timestamp_sec}, "
                f"got sample {sample_index} at {timestamp}"
            )
        filename = frame_filename(sample_index, timestamp, frame_format)
        output_path: Path | None = None
        if decision.selected:
            output_path = selected_dir / filename
            write_image(output_path, frame, frame_format)
        elif save_rejected:
            output_path = rejected_dir / filename
            write_image(output_path, frame, frame_format)
        logical_output_path = output_path
        if output_path is not None and decision.selected and manifest_selected_root is not None:
            logical_output_path = manifest_selected_root / segment.id / filename
        elif output_path is not None and not decision.selected and manifest_rejected_root is not None:
            logical_output_path = manifest_rejected_root / segment.id / filename
        keyframe = {
            "policy": policy.name,
            "component_id": decision.component_id,
            "selected_by_policy": decision.selected,
            "bridge": decision.bridge,
            "bridge_reason": decision.bridge_reason,
            "reference_frame_id": decision.reference_frame_id,
            "adaptive_blur_threshold": round(decision.adaptive_blur_threshold, 6),
            "quality_score": round(decision.quality_score, 6),
            "motion": None if decision.motion is None else decision.motion.to_manifest(),
        }
        records.append(
            FrameRecord(
                id=scanned_frame.id,
                segment_id=segment.id,
                path=relative_path(logical_output_path, repo_root) if logical_output_path else None,
                timestamp_sec=timestamp,
                frame_index=frame_index,
                selected=decision.selected,
                blur_score=round(scanned_frame.blur_score, 3),
                overexposed_ratio=round(scanned_frame.overexposed_ratio, 6),
                underexposed_ratio=round(scanned_frame.underexposed_ratio, 6),
                duplicate_score=decision.duplicate_score,
                reject_reasons=list(decision.reject_reasons),
                width=scanned_frame.width,
                height=scanned_frame.height,
                sha256=sha256_file(output_path) if output_path else None,
                calibrated_blur_score=round(scanned_frame.calibrated_blur_score, 6),
                keyframe=keyframe,
                matched_frame_id=decision.matched_frame_id if "duplicate" in decision.reject_reasons else None,
            )
        )
    try:
        next(second_pass)
    except StopIteration:
        pass
    else:
        raise PreprocessError(
            f"Second-pass frame count mismatch for {segment.id}: more frames than first pass"
        )
    return records, _coverage_segment_quality(segment, scanned, decisions, policy)


def summarize_frames(
    segments: Iterable[SegmentWindow],
    frames: Iterable[FrameRecord],
    max_selected_gap_sec: float | None = None,
) -> dict[str, Any]:
    segment_list = list(segments)
    segment_counts: dict[str, dict[str, int]] = {
        segment.id: {"total": 0, "selected": 0, "rejected": 0} for segment in segment_list
    }
    segment_by_id = {segment.id: segment for segment in segment_list}
    selected_timestamps: dict[str, list[float]] = {segment_id: [] for segment_id in segment_counts}
    reject_counts: Counter[str] = Counter()
    total = selected = rejected = 0
    for frame in frames:
        total += 1
        segment_counts.setdefault(frame.segment_id, {"total": 0, "selected": 0, "rejected": 0})
        segment_counts[frame.segment_id]["total"] += 1
        if frame.selected:
            selected += 1
            segment_counts[frame.segment_id]["selected"] += 1
            selected_timestamps.setdefault(frame.segment_id, []).append(frame.timestamp_sec)
        else:
            rejected += 1
            segment_counts[frame.segment_id]["rejected"] += 1
            reject_counts.update(frame.reject_reasons)

    coverage_by_segment: dict[str, dict[str, Any]] = {}
    for segment_id, segment in segment_by_id.items():
        observed_gap = selected_frame_max_gap(segment, selected_timestamps.get(segment_id, []))
        coverage_by_segment[segment_id] = {
            "max_selected_gap_sec": observed_gap,
            "exceeds_target": max_selected_gap_sec is not None and observed_gap > max_selected_gap_sec,
        }
    for segment_id, timestamps in selected_timestamps.items():
        if segment_id in coverage_by_segment:
            continue
        ordered = sorted(timestamps)
        gaps = [right - left for left, right in zip(ordered, ordered[1:])]
        observed_gap = round_sec(max(gaps, default=0.0))
        coverage_by_segment[segment_id] = {
            "max_selected_gap_sec": observed_gap,
            "exceeds_target": max_selected_gap_sec is not None and observed_gap > max_selected_gap_sec,
        }

    return {
        "total_segments": len(segment_counts),
        "total_frames": total,
        "selected_frames": selected,
        "rejected_frames": rejected,
        "frames_by_segment": segment_counts,
        "reject_reasons": dict(sorted(reject_counts.items())),
        "coverage_target_max_selected_gap_sec": max_selected_gap_sec,
        "coverage_by_segment": coverage_by_segment,
    }


def segment_to_manifest(
    segment: SegmentWindow,
    repo_root: Path,
    width: int,
    height: int,
) -> dict[str, Any]:
    if segment.path is None:
        raise ValueError("segment.path is required for manifest output")
    return {
        "id": segment.id,
        "path": relative_path(segment.path, repo_root),
        "index": segment.index,
        "start_sec": round_sec(segment.start_sec),
        "end_sec": round_sec(segment.end_sec),
        "duration_sec": round_sec(segment.duration_sec),
        "reason": segment.reason,
        "width": width,
        "height": height,
        "sha256": segment.sha256,
    }


def build_manifest(
    video_id: str,
    source_metadata: VideoMetadata,
    normalized_metadata: VideoMetadata,
    settings: dict[str, Any],
    segments: list[SegmentWindow],
    frames: list[FrameRecord],
    repo_root: Path,
    run: dict[str, Any] | None = None,
    keyframe_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    invalid_frames = [frame.id for frame in frames if frame.width <= 0 or frame.height <= 0]
    if invalid_frames:
        raise PreprocessError(f"Frame records must include positive dimensions: {', '.join(invalid_frames[:3])}")
    coverage_target = (
        settings.get("selection_max_gap_sec")
        if settings.get("keyframe_policy") == "coverage_v1"
        else settings.get("max_selected_gap_sec")
    )
    summary = summarize_frames(segments, frames, coverage_target)
    if keyframe_quality is not None:
        summary["keyframe_quality"] = keyframe_quality
    return {
        "schema_version": SCHEMA_VERSION,
        "video_id": video_id,
        "source": {
            "path": relative_path(source_metadata.path, repo_root),
            "size_bytes": source_metadata.size_bytes,
            "duration_sec": round_sec(source_metadata.duration_sec),
            "fps": round(float(source_metadata.fps), 6),
            "width": source_metadata.width,
            "height": source_metadata.height,
            "codec": source_metadata.codec,
            "sha256": source_metadata.sha256,
            "rotation_degrees": source_metadata.rotation_degrees,
            "is_vfr": source_metadata.is_vfr,
        },
        "normalized": {
            "path": relative_path(normalized_metadata.path, repo_root),
            "duration_sec": round_sec(normalized_metadata.duration_sec),
            "fps": round(float(normalized_metadata.fps), 6),
            "width": normalized_metadata.width,
            "height": normalized_metadata.height,
            "codec": normalized_metadata.codec,
            "sha256": normalized_metadata.sha256,
            "rotation_degrees": normalized_metadata.rotation_degrees,
            "is_vfr": normalized_metadata.is_vfr,
        },
        "settings": settings,
        "segments": [
            segment_to_manifest(segment, repo_root, normalized_metadata.width, normalized_metadata.height)
            for segment in segments
        ],
        "frames": [frame_to_manifest(frame) for frame in frames],
        "summary": summary,
        "run": run or {},
    }


def frame_to_manifest(frame: FrameRecord) -> dict[str, Any]:
    """Keep the legacy schema stable while serializing coverage diagnostics additively."""
    payload = asdict(frame)
    if frame.calibrated_blur_score is None:
        payload.pop("calibrated_blur_score", None)
    if frame.keyframe is None:
        payload.pop("keyframe", None)
    return payload


def write_manifest(manifest: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ensure_repo_path(path: Path, repo_root: Path, label: str) -> Path:
    resolved = path.resolve()
    if not is_relative_to(resolved, repo_root):
        raise PreprocessError(f"{label} must be inside the repository: {resolved}")
    return resolved


def prepare_output_dirs(
    output_root: Path,
    video_id: str,
    force: bool,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    output_root = output_root.resolve()
    staging_root = output_root / ".preprocess-staging" / f"{video_id}-{uuid.uuid4().hex}"
    segments_dir = staging_root / "segments" / video_id
    frames_dir = staging_root / "frames" / video_id
    selected_dir = frames_dir / "selected"
    rejected_dir = frames_dir / "rejected"
    manifests_dir = staging_root / "manifests" / video_id
    final_dirs = [
        output_root / "segments" / video_id,
        output_root / "frames" / video_id,
        output_root / "manifests" / video_id,
    ]

    existing = [path for path in final_dirs if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise PreprocessError(f"Output already exists for video_id '{video_id}': {joined}. Use --force to overwrite.")

    segments_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    return segments_dir, selected_dir, rejected_dir, manifests_dir, frames_dir, staging_root


def publish_output_dirs(
    output_root: Path,
    video_id: str,
    staging_root: Path,
    force: bool,
) -> None:
    final_dirs = [
        output_root / "segments" / video_id,
        output_root / "frames" / video_id,
        output_root / "manifests" / video_id,
    ]
    staged_dirs = [
        staging_root / "segments" / video_id,
        staging_root / "frames" / video_id,
        staging_root / "manifests" / video_id,
    ]
    backup_root = output_root / ".preprocess-backups" / f"{video_id}-{uuid.uuid4().hex}"
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        if force:
            for final_dir in final_dirs:
                if final_dir.exists():
                    backup_dir = backup_root / final_dir.parent.name / video_id
                    backup_dir.parent.mkdir(parents=True, exist_ok=True)
                    final_dir.replace(backup_dir)
                    backups.append((final_dir, backup_dir))
        for staged_dir, final_dir in zip(staged_dirs, final_dirs):
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            staged_dir.replace(final_dir)
            published.append(final_dir)
    except OSError as exc:
        for final_dir in published:
            if final_dir.exists():
                shutil.rmtree(final_dir)
        for final_dir, backup_dir in reversed(backups):
            if backup_dir.exists():
                backup_dir.replace(final_dir)
        raise PreprocessError(f"Could not publish preprocessing outputs atomically: {exc}") from exc
    finally:
        if backup_root.exists():
            shutil.rmtree(backup_root, ignore_errors=True)
        cleanup_staging_root(staging_root)


def cleanup_staging_root(staging_root: Path) -> None:
    staging_parent = staging_root.parent
    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=True)
    if staging_parent.exists() and not any(staging_parent.iterdir()):
        staging_parent.rmdir()


def load_settings_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreprocessError(f"Config file does not exist: {path}") from exc
    except OSError as exc:
        raise PreprocessError(f"Could not read config file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PreprocessError(
            f"Invalid JSON in config file {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(payload, dict):
        raise PreprocessError(f"Config file must contain a JSON object: {path}")

    unknown_keys = sorted(set(payload) - CONFIG_SETTING_KEYS)
    if unknown_keys:
        raise PreprocessError(f"Unknown config setting(s): {', '.join(unknown_keys)}")

    settings = dict(payload)
    for name in FLOAT_CONFIG_SETTINGS:
        if name not in settings:
            continue
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PreprocessError(f"Config setting '{name}' must be a number.")
        settings[name] = float(value)

    for name in INT_CONFIG_SETTINGS:
        if name not in settings:
            continue
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise PreprocessError(f"Config setting '{name}' must be an integer.")

    if "save_rejected" in settings and not isinstance(settings["save_rejected"], bool):
        raise PreprocessError("Config setting 'save_rejected' must be a boolean.")

    for name, choices in CHOICE_CONFIG_SETTINGS.items():
        if name not in settings:
            continue
        value = settings[name]
        if not isinstance(value, str) or value not in choices:
            allowed = ", ".join(sorted(choices))
            raise PreprocessError(f"Config setting '{name}' must be one of: {allowed}.")

    for name in pk.KEYFRAME_DEFAULT_SETTINGS:
        if (
            name in settings
            and name not in FLOAT_CONFIG_SETTINGS
            and name not in INT_CONFIG_SETTINGS
            and name not in CHOICE_CONFIG_SETTINGS
        ):
            value = settings[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PreprocessError(f"Config setting '{name}' must be a number.")

    preset_name = str(settings.get("preset", DEFAULT_PRESET))
    candidate = dict(DEFAULT_SETTINGS)
    candidate.update(PRESETS[preset_name])
    candidate.update(settings)
    try:
        pk.validate_keyframe_settings(candidate)
    except (KeyError, TypeError, ValueError) as exc:
        raise PreprocessError(str(exc)) from exc

    return settings


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config_path = getattr(args, "config", None)
    config = load_settings_config(config_path) if config_path is not None else {}
    preset_name = getattr(args, "preset", None) or config.get("preset", DEFAULT_PRESET)

    settings = dict(DEFAULT_SETTINGS)
    settings.update(PRESETS[preset_name])
    settings.update({name: value for name, value in config.items() if name != "preset"})
    settings["preset"] = preset_name

    for name in CONFIG_SETTING_KEYS - {"preset"}:
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            settings[name] = cli_value

    validate_window_settings(
        settings["segment_length_sec"],
        settings["segment_overlap_sec"],
        settings["min_segment_sec"],
    )
    if settings["target_fps"] <= 0:
        raise PreprocessError("--target-fps must be greater than zero.")
    if settings["max_long_edge"] <= 0:
        raise PreprocessError("--max-long-edge must be greater than zero.")
    for name in ("overexposed_ratio", "underexposed_ratio"):
        if not 0.0 <= settings[name] <= 1.0:
            raise PreprocessError(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if settings["duplicate_hash_threshold"] < 0:
        raise PreprocessError("--duplicate-hash-threshold cannot be negative.")
    for name in ("duplicate_time_window_sec", "max_selected_gap_sec"):
        if settings[name] <= 0:
            raise PreprocessError(f"--{name.replace('_', '-')} must be greater than zero.")
    try:
        pk.validate_keyframe_settings(settings)
    except (KeyError, TypeError, ValueError) as exc:
        raise PreprocessError(str(exc)) from exc
    return settings


def validate_video_id(video_id: str) -> None:
    if not VIDEO_ID_RE.match(video_id):
        raise PreprocessError("--video-id may only contain letters, numbers, underscore, dash, and dot.")
    if video_id in {".", ".."}:
        raise PreprocessError("--video-id cannot be '.' or '..'.")


def read_keyframe_status(manifest_path: Path) -> tuple[str, str]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown", "failed"
    quality = manifest.get("summary", {}).get("keyframe_quality")
    if not quality:
        return "legacy", "not_applicable"
    return str(quality.get("policy", "unknown")), str(quality.get("status", "failed"))


def audit_preprocess_manifest(manifest_path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = (repo_root or Path.cwd()).resolve()
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "failed", "errors": [f"manifest does not exist: {manifest_path}"], "summary": {}}
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "failed", "errors": [f"could not read manifest: {exc}"], "summary": {}}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    frame_ids: set[str] = set()
    timestamps: dict[str, list[float]] = {}
    for frame in manifest.get("frames", []):
        frame_id = frame.get("id")
        if not isinstance(frame_id, str) or frame_id in frame_ids:
            errors.append(f"frame ids must be globally unique: {frame_id}")
        frame_ids.add(str(frame_id))
        timestamps.setdefault(str(frame.get("segment_id")), []).append(float(frame.get("timestamp_sec", -1)))
        if frame.get("selected"):
            path_value = frame.get("path")
            if not isinstance(path_value, str):
                errors.append(f"selected frame {frame_id} has no path")
                continue
            candidate = (repo_root / Path(path_value)).resolve()
            if not is_relative_to(candidate, repo_root):
                errors.append(f"selected frame {frame_id} path is outside the repository")
            elif not candidate.is_file():
                errors.append(f"selected frame {frame_id} path is missing: {path_value}")
            elif cv2.imdecode(np.frombuffer(candidate.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR) is None:
                errors.append(f"selected frame {frame_id} cannot be decoded")
    for segment_id, values in timestamps.items():
        if values != sorted(values) or len(values) != len(set(values)):
            errors.append(f"timestamps are not strictly increasing in {segment_id}")
    quality = manifest.get("summary", {}).get("keyframe_quality")
    if quality and quality.get("status") != "passed":
        errors.append("keyframe quality status is not passed")
    return {"status": "passed" if not errors else "failed", "errors": errors, "summary": manifest.get("summary", {})}


def preprocess(args: argparse.Namespace, command_args: list[str] | None = None) -> Path:
    repo_root = Path.cwd().resolve()
    source = ensure_repo_path(args.source_video, repo_root, "Source video")
    output_root = ensure_repo_path(args.output_root, repo_root, "Output root")
    validate_video_id(args.video_id)
    if not source.exists() or not source.is_file():
        raise PreprocessError(f"Source video does not exist: {source}")
    if args.config is not None:
        args.config = ensure_repo_path(args.config, repo_root, "Config file")

    require_tool("ffmpeg")
    require_tool("ffprobe")
    settings = resolve_settings(args)
    source_metadata = probe_video(source)
    revision, dirty = repository_state(repo_root)
    if settings["frame_source"] == "source" and source_metadata.is_vfr:
        raise PreprocessError(
            "--frame-source source is not timestamp-safe for variable-frame-rate input. "
            "Use --frame-source segment so frames are sampled from the normalized CFR video."
        )

    segments_dir, selected_dir, rejected_dir, manifests_dir, _frames_dir, staging_root = prepare_output_dirs(
        output_root, args.video_id, args.force
    )
    final_segments_dir = output_root / "segments" / args.video_id
    final_frames_dir = output_root / "frames" / args.video_id
    final_manifests_dir = output_root / "manifests" / args.video_id
    normalized_path = segments_dir / "normalized.mp4"
    try:
        staged_normalized_metadata = normalize_video(
            source, normalized_path, source_metadata, settings["max_long_edge"]
        )
        normalized_metadata = replace(staged_normalized_metadata, path=final_segments_dir / "normalized.mp4")
        source_width, source_height = source_metadata.width, source_metadata.height
        if source_metadata.rotation_degrees in {90, 270}:
            source_width, source_height = source_height, source_width
        source_frame_dimensions = scaled_dimensions(source_width, source_height, settings["max_long_edge"])

        planned_segments = build_segment_windows(
            duration_sec=staged_normalized_metadata.duration_sec,
            segment_method=settings["segment_method"],
            segment_length_sec=settings["segment_length_sec"],
            segment_overlap_sec=settings["segment_overlap_sec"],
            min_segment_sec=settings["min_segment_sec"],
            normalized_path=normalized_path,
        )
        if not planned_segments:
            raise PreprocessError("No segments were produced from the normalized video.")

        staged_segments = [
            write_segment(normalized_path, segment, segments_dir / f"{segment.id}.mp4")
            for segment in planned_segments
        ]
        final_segments = [
            replace(segment, path=final_segments_dir / f"{segment.id}.mp4")
            for segment in staged_segments
        ]

        frames: list[FrameRecord] = []
        segment_quality: list[dict[str, Any]] = []
        policy = pk.KeyframePolicy.from_settings(settings)
        for segment in staged_segments:
            if settings["keyframe_policy"] == "coverage_v1":
                segment_frames, quality = sample_segment_frames_coverage(
                    segment=segment,
                    selected_root=selected_dir,
                    rejected_root=rejected_dir,
                    target_fps=settings["target_fps"],
                    save_rejected=settings["save_rejected"],
                    repo_root=repo_root,
                    policy=policy,
                    frame_format=settings["frame_format"],
                    frame_source=settings["frame_source"],
                    source_video=source if settings["frame_source"] == "source" else None,
                    resize_to=source_frame_dimensions if settings["frame_source"] == "source" else None,
                    source_rotation_degrees=source_metadata.rotation_degrees,
                    manifest_selected_root=final_frames_dir / "selected",
                    manifest_rejected_root=final_frames_dir / "rejected",
                )
                segment_quality.append(quality)
            else:
                segment_frames = sample_segment_frames(
                    segment=segment,
                    selected_root=selected_dir,
                    rejected_root=rejected_dir,
                    target_fps=settings["target_fps"],
                    blur_threshold=settings["blur_threshold"],
                    overexposed_ratio=settings["overexposed_ratio"],
                    underexposed_ratio=settings["underexposed_ratio"],
                    duplicate_hash_threshold=settings["duplicate_hash_threshold"],
                    save_rejected=settings["save_rejected"],
                    repo_root=repo_root,
                    frame_format=settings["frame_format"],
                    frame_source=settings["frame_source"],
                    source_video=source if settings["frame_source"] == "source" else None,
                    resize_to=source_frame_dimensions if settings["frame_source"] == "source" else None,
                    source_rotation_degrees=source_metadata.rotation_degrees,
                    duplicate_time_window_sec=settings["duplicate_time_window_sec"],
                    max_selected_gap_sec=settings["max_selected_gap_sec"],
                    manifest_selected_root=final_frames_dir / "selected",
                    manifest_rejected_root=final_frames_dir / "rejected",
                )
            frames.extend(segment_frames)

        keyframe_quality: dict[str, Any] | None = None
        if settings["keyframe_policy"] == "coverage_v1":
            failed_segments = [item for item in segment_quality if item["failed_reasons"]]
            keyframe_quality = {
                "policy": "coverage_v1",
                "status": "failed" if failed_segments else "passed",
                "selected_frames": sum(item["selected_frames"] for item in segment_quality),
                "bridge_frames": sum(item["bridge_frames"] for item in segment_quality),
                "component_count": max((item["component_count"] for item in segment_quality), default=0),
                "max_selected_gap_sec": round_sec(max((item["max_selected_gap_sec"] for item in segment_quality), default=0.0)),
                "max_selected_gap_target_sec": policy.max_gap_sec,
                "min_motion_inlier_ratio": policy.min_motion_inlier_ratio,
                "min_motion_grid_coverage": policy.min_motion_grid_coverage,
                "failed_segments": [item["segment_id"] for item in failed_segments],
            }
            report = {
                "schema_version": SCHEMA_VERSION,
                "video_id": args.video_id,
                "policy": "coverage_v1",
                "status": keyframe_quality["status"],
                "segments": segment_quality,
            }
            write_manifest(report, manifests_dir / "keyframe_quality_report.json")

        settings_fingerprint = stable_hash(settings)
        raw_command_args = command_args or [str(args.source_video), "--video-id", args.video_id]
        run = {
            "id": f"{args.video_id}-{(source_metadata.sha256 or 'unknown')[:12]}-{settings_fingerprint[:12]}",
            "source_sha256": source_metadata.sha256,
            "settings_fingerprint": settings_fingerprint,
            "config": {
                "path": relative_path(args.config, repo_root) if args.config is not None else None,
                "sha256": sha256_file(args.config) if args.config is not None else None,
            },
            "code": {"commit": revision, "working_tree_dirty": dirty},
            "tools": {
                "python": sys.version.split()[0],
                "opencv": cv2.__version__,
                "ffmpeg": tool_version("ffmpeg"),
                "ffprobe": tool_version("ffprobe"),
            },
            "command": manifest_command_args(raw_command_args, repo_root, source),
        }
        manifest = build_manifest(
            video_id=args.video_id,
            source_metadata=source_metadata,
            normalized_metadata=normalized_metadata,
            settings=settings,
            segments=final_segments,
            frames=frames,
            repo_root=repo_root,
            run=run,
            keyframe_quality=keyframe_quality,
        )
        write_manifest(manifest, manifests_dir / MANIFEST_FILENAME)
        publish_output_dirs(output_root, args.video_id, staging_root, args.force)
        return final_manifests_dir / MANIFEST_FILENAME
    except Exception:
        cleanup_staging_root(staging_root)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess exhibition video for downstream 3DGS stages.")
    parser.add_argument("source_video", type=Path, nargs="?", help="Path to the raw source video.")
    parser.add_argument("--output-root", type=Path, default=Path("data"), help="Root output directory.")
    parser.add_argument("--video-id", help="Stable video identifier used in output paths.")
    parser.add_argument("--audit-manifest", type=Path, help="Audit an existing frames_manifest.json and print JSON.")
    parser.add_argument("--config", type=Path, help="JSON file containing preprocess tuning settings.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None, help="Preprocess preset.")
    parser.add_argument("--target-fps", type=float, default=None, help="Frame sampling rate per segment.")
    parser.add_argument("--max-long-edge", type=int, default=None, help="Maximum normalized video long edge.")
    parser.add_argument(
        "--segment-method",
        choices=["scene", "time", "scene,time"],
        default=None,
        help="Segmentation strategy.",
    )
    parser.add_argument("--segment-length-sec", type=float, default=None, help="Maximum segment length.")
    parser.add_argument("--segment-overlap-sec", type=float, default=None, help="Overlap for split time windows.")
    parser.add_argument("--min-segment-sec", type=float, default=None, help="Merge or avoid tiny segments below this.")
    parser.add_argument("--blur-threshold", type=float, default=None, help="Reject frames below this Laplacian variance.")
    parser.add_argument("--overexposed-ratio", type=float, default=None, help="Reject frames above this white-pixel ratio.")
    parser.add_argument("--underexposed-ratio", type=float, default=None, help="Reject frames above this black-pixel ratio.")
    parser.add_argument("--duplicate-hash-threshold", type=int, default=None, help="Reject near-duplicate average hashes.")
    parser.add_argument(
        "--duplicate-time-window-sec",
        type=float,
        default=None,
        help="Compare duplicate hashes only within this recent time window.",
    )
    parser.add_argument(
        "--max-selected-gap-sec",
        type=float,
        default=None,
        help="Preserve coverage by allowing a duplicate after this selected-frame gap.",
    )
    parser.add_argument(
        "--save-rejected",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write rejected frame images as well as manifest rows.",
    )
    parser.add_argument(
        "--frame-format",
        choices=["jpg", "png"],
        default=None,
        help="Image format for selected and rejected frames.",
    )
    parser.add_argument(
        "--frame-source",
        choices=["segment", "source"],
        default=None,
        help="Read sampled frames from generated segments or directly from the source video.",
    )
    parser.add_argument("--keyframe-policy", choices=["legacy", "coverage_v1"], default=None,
                        help="Keyframe selector; legacy preserves the original filtering behavior.")
    parser.add_argument("--quality-analysis-long-edge", type=int, default=None)
    parser.add_argument("--flow-analysis-long-edge", type=int, default=None)
    parser.add_argument("--adaptive-blur-percentile", type=float, default=None)
    parser.add_argument("--selection-min-gap-sec", type=float, default=None)
    parser.add_argument("--selection-target-gap-sec", type=float, default=None)
    parser.add_argument("--selection-max-gap-sec", type=float, default=None)
    parser.add_argument("--duplicate-window-sec", type=float, default=None)
    parser.add_argument("--min-tracked-points", type=int, default=None)
    parser.add_argument("--min-motion-inlier-ratio", type=float, default=None)
    parser.add_argument("--min-motion-grid-coverage", type=float, default=None)
    parser.add_argument("--max-motion-residual-diag-ratio", type=float, default=None)
    parser.add_argument("--max-median-displacement-diag-ratio", type=float, default=None)
    parser.add_argument("--max-affine-rotation-deg", type=float, default=None)
    parser.add_argument("--min-affine-scale", type=float, default=None)
    parser.add_argument("--max-affine-scale", type=float, default=None)
    parser.add_argument("--motion-model", choices=["affine", "auto"], default=None)
    parser.add_argument("--flow-forward-backward-max-error-px", type=float, default=None)
    parser.add_argument("--fundamental-ransac-threshold-px", type=float, default=None)
    parser.add_argument("--fundamental-ransac-confidence", type=float, default=None)
    parser.add_argument("--bridge-min-blur-ratio", type=float, default=None)
    parser.add_argument("--max-bridge-window-sec", type=float, default=None)
    parser.add_argument("--max-bridge-fraction", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directories for this video id.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.audit_manifest is not None:
        preprocess_values = [
            args.source_video, args.video_id, args.config, args.preset, args.target_fps,
            args.max_long_edge, args.segment_method, args.segment_length_sec,
            args.segment_overlap_sec, args.min_segment_sec, args.blur_threshold,
            args.overexposed_ratio, args.underexposed_ratio, args.duplicate_hash_threshold,
            args.duplicate_time_window_sec, args.max_selected_gap_sec, args.save_rejected,
            args.frame_format, args.frame_source, args.keyframe_policy, args.force,
            args.quality_analysis_long_edge, args.flow_analysis_long_edge,
            args.adaptive_blur_percentile, args.selection_min_gap_sec,
            args.selection_target_gap_sec, args.selection_max_gap_sec,
            args.duplicate_window_sec, args.min_tracked_points,
            args.min_motion_inlier_ratio, args.min_motion_grid_coverage,
            args.max_motion_residual_diag_ratio,
            args.max_median_displacement_diag_ratio, args.max_affine_rotation_deg,
            args.min_affine_scale, args.max_affine_scale,
            args.motion_model, args.flow_forward_backward_max_error_px,
            args.fundamental_ransac_threshold_px, args.fundamental_ransac_confidence,
            args.bridge_min_blur_ratio, args.max_bridge_window_sec, args.max_bridge_fraction,
        ]
        if any(value is not None and value is not False for value in preprocess_values):
            parser.error("--audit-manifest cannot be combined with preprocess options")
        result = audit_preprocess_manifest(args.audit_manifest)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "passed" else 2
    if args.source_video is None or args.video_id is None:
        parser.error("source_video and --video-id are required unless --audit-manifest is used")
    try:
        command_args = list(argv) if argv is not None else sys.argv[1:]
        manifest_path = preprocess(args, command_args=command_args)
    except (PreprocessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    policy, status = read_keyframe_status(manifest_path)
    print(f"Wrote manifest: {relative_path(manifest_path)}")
    if policy == "coverage_v1" and status != "passed":
        print(f"Keyframe quality gate failed: {status}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
