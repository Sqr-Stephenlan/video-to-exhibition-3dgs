#!/usr/bin/env python3
"""Preprocess raw exhibition video into segments, selected frames, and a manifest."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


SCHEMA_VERSION = "1.0"
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


@dataclass(frozen=True)
class SegmentWindow:
    id: str
    index: int
    start_sec: float
    end_sec: float
    reason: str
    path: Path | None = None

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
        return resolved.as_posix()


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
    duration = float(stream.get("duration") or format_info.get("duration") or 0.0)
    fps = parse_fraction(stream.get("avg_frame_rate")) or parse_fraction(stream.get("r_frame_rate"))
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if duration <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise PreprocessError(f"Incomplete video metadata for {source}.")

    return VideoMetadata(
        path=source,
        duration_sec=duration,
        fps=fps,
        width=width,
        height=height,
        codec=str(stream.get("codec_name") or ""),
        size_bytes=source.stat().st_size,
        streams=video_streams,
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
    width, height = scaled_dimensions(metadata.width, metadata.height, max_long_edge)
    command = [
        "ffmpeg",
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
        (round_sec(start.get_seconds()), round_sec(end.get_seconds()))
        for start, end in scenes
        if end.get_seconds() > start.get_seconds()
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


def frame_filename(sample_index: int, timestamp_sec: float) -> str:
    return f"frame_{sample_index:06d}_t{timestamp_sec:010.3f}.jpg"


def write_image(path: Path, frame: np.ndarray) -> None:
    """Write an image through Python file I/O so Unicode Windows paths work."""
    encoded_ok, encoded = cv2.imencode(".jpg", frame)
    if not encoded_ok:
        raise PreprocessError(f"OpenCV could not encode frame for {path}.")
    try:
        path.write_bytes(encoded.tobytes())
    except OSError as exc:
        raise PreprocessError(f"Could not write frame image {path}: {exc}") from exc


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
) -> list[FrameRecord]:
    if segment.path is None:
        raise ValueError("segment.path is required for frame sampling")
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero")

    cap = cv2.VideoCapture(str(segment.path))
    if not cap.isOpened():
        raise PreprocessError(f"OpenCV could not open segment {segment.path}.")

    segment_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    sample_every = 1 if segment_fps <= 0 else max(1, int(round(segment_fps / target_fps)))
    selected_hashes: list[np.ndarray] = []
    records: list[FrameRecord] = []
    selected_dir = selected_root / segment.id
    rejected_dir = rejected_root / segment.id
    selected_dir.mkdir(parents=True, exist_ok=True)
    if save_rejected:
        rejected_dir.mkdir(parents=True, exist_ok=True)

    frame_number = 0
    sample_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1
        if (frame_number - 1) % sample_every != 0:
            continue

        sample_index += 1
        local_timestamp = 0.0 if segment_fps <= 0 else (frame_number - 1) / segment_fps
        timestamp = round_sec(segment.start_sec + local_timestamp)
        blur = blur_score(frame)
        over, under = exposure_ratios(frame)
        current_hash = average_hash(frame)
        duplicate = min((hash_distance(current_hash, item) for item in selected_hashes), default=None)

        reject_reasons = []
        if blur < blur_threshold:
            reject_reasons.append("blur")
        if over > overexposed_ratio:
            reject_reasons.append("overexposed")
        if under > underexposed_ratio:
            reject_reasons.append("underexposed")
        if duplicate is not None and duplicate <= duplicate_hash_threshold:
            reject_reasons.append("duplicate")

        selected = not reject_reasons
        filename = frame_filename(sample_index, timestamp)
        output_path: Path | None
        if selected:
            output_path = selected_dir / filename
            selected_hashes.append(current_hash)
            write_image(output_path, frame)
        elif save_rejected:
            output_path = rejected_dir / filename
            write_image(output_path, frame)
        else:
            output_path = None

        records.append(
            FrameRecord(
                id=f"{segment.id}_frame_{sample_index:06d}",
                segment_id=segment.id,
                path=relative_path(output_path, repo_root) if output_path else None,
                timestamp_sec=timestamp,
                frame_index=frame_number,
                selected=selected,
                blur_score=round(float(blur), 3),
                overexposed_ratio=round(float(over), 6),
                underexposed_ratio=round(float(under), 6),
                duplicate_score=duplicate,
                reject_reasons=reject_reasons,
            )
        )

    cap.release()
    return records


def summarize_frames(segments: Iterable[SegmentWindow], frames: Iterable[FrameRecord]) -> dict[str, Any]:
    segment_counts: dict[str, dict[str, int]] = {
        segment.id: {"total": 0, "selected": 0, "rejected": 0} for segment in segments
    }
    reject_counts: Counter[str] = Counter()
    total = selected = rejected = 0
    for frame in frames:
        total += 1
        segment_counts.setdefault(frame.segment_id, {"total": 0, "selected": 0, "rejected": 0})
        segment_counts[frame.segment_id]["total"] += 1
        if frame.selected:
            selected += 1
            segment_counts[frame.segment_id]["selected"] += 1
        else:
            rejected += 1
            segment_counts[frame.segment_id]["rejected"] += 1
            reject_counts.update(frame.reject_reasons)

    return {
        "total_segments": len(segment_counts),
        "total_frames": total,
        "selected_frames": selected,
        "rejected_frames": rejected,
        "frames_by_segment": segment_counts,
        "reject_reasons": dict(sorted(reject_counts.items())),
    }


def segment_to_manifest(segment: SegmentWindow, repo_root: Path) -> dict[str, Any]:
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
    }


def build_manifest(
    video_id: str,
    source_metadata: VideoMetadata,
    normalized_metadata: VideoMetadata,
    settings: dict[str, Any],
    segments: list[SegmentWindow],
    frames: list[FrameRecord],
    repo_root: Path,
) -> dict[str, Any]:
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
        },
        "normalized": {
            "path": relative_path(normalized_metadata.path, repo_root),
            "duration_sec": round_sec(normalized_metadata.duration_sec),
            "fps": round(float(normalized_metadata.fps), 6),
            "width": normalized_metadata.width,
            "height": normalized_metadata.height,
            "codec": normalized_metadata.codec,
        },
        "settings": settings,
        "segments": [segment_to_manifest(segment, repo_root) for segment in segments],
        "frames": [asdict(frame) for frame in frames],
        "summary": summarize_frames(segments, frames),
    }


def write_manifest(manifest: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def prepare_output_dirs(output_root: Path, video_id: str, force: bool) -> tuple[Path, Path, Path, Path, Path]:
    output_root = output_root.resolve()
    segments_parent = output_root / "segments"
    frames_parent = output_root / "frames"
    manifests_parent = output_root / "manifests"
    segments_dir = output_root / "segments" / video_id
    frames_dir = output_root / "frames" / video_id
    selected_dir = frames_dir / "selected"
    rejected_dir = frames_dir / "rejected"
    manifests_dir = output_root / "manifests" / video_id
    managed_dirs = [segments_dir, frames_dir, manifests_dir]
    expected_parents = [segments_parent, frames_parent, manifests_parent]

    for path, parent in zip(managed_dirs, expected_parents):
        if not is_relative_to(path, parent) or path.resolve() == parent.resolve():
            raise PreprocessError(f"Unsafe output path for video_id '{video_id}': {path}")

    existing = [path for path in managed_dirs if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise PreprocessError(f"Output already exists for video_id '{video_id}': {joined}. Use --force to overwrite.")

    if force:
        for path in existing:
            shutil.rmtree(path)

    segments_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    return segments_dir, selected_dir, rejected_dir, manifests_dir, frames_dir


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    preset = PRESETS[args.preset]
    settings = {
        "preset": args.preset,
        "target_fps": args.target_fps if args.target_fps is not None else preset["target_fps"],
        "max_long_edge": args.max_long_edge if args.max_long_edge is not None else preset["max_long_edge"],
        "segment_method": args.segment_method,
        "segment_length_sec": (
            args.segment_length_sec if args.segment_length_sec is not None else preset["segment_length_sec"]
        ),
        "segment_overlap_sec": (
            args.segment_overlap_sec if args.segment_overlap_sec is not None else preset["segment_overlap_sec"]
        ),
        "min_segment_sec": args.min_segment_sec,
        "blur_threshold": args.blur_threshold,
        "overexposed_ratio": args.overexposed_ratio,
        "underexposed_ratio": args.underexposed_ratio,
        "duplicate_hash_threshold": args.duplicate_hash_threshold,
        "save_rejected": args.save_rejected,
    }
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
    return settings


def validate_video_id(video_id: str) -> None:
    if not VIDEO_ID_RE.match(video_id):
        raise PreprocessError("--video-id may only contain letters, numbers, underscore, dash, and dot.")
    if video_id in {".", ".."}:
        raise PreprocessError("--video-id cannot be '.' or '..'.")


def preprocess(args: argparse.Namespace) -> Path:
    repo_root = Path.cwd()
    source = args.source_video.resolve()
    output_root = args.output_root
    validate_video_id(args.video_id)
    if not source.exists() or not source.is_file():
        raise PreprocessError(f"Source video does not exist: {source}")

    require_tool("ffmpeg")
    require_tool("ffprobe")
    settings = resolve_settings(args)
    segments_dir, selected_dir, rejected_dir, manifests_dir, _frames_dir = prepare_output_dirs(
        output_root, args.video_id, args.force
    )

    source_metadata = probe_video(source)
    normalized_path = segments_dir / "normalized.mp4"
    normalized_metadata = normalize_video(source, normalized_path, source_metadata, settings["max_long_edge"])

    planned_segments = build_segment_windows(
        duration_sec=normalized_metadata.duration_sec,
        segment_method=settings["segment_method"],
        segment_length_sec=settings["segment_length_sec"],
        segment_overlap_sec=settings["segment_overlap_sec"],
        min_segment_sec=settings["min_segment_sec"],
        normalized_path=normalized_path,
    )
    if not planned_segments:
        raise PreprocessError("No segments were produced from the normalized video.")

    written_segments = [
        write_segment(normalized_path, segment, segments_dir / f"{segment.id}.mp4")
        for segment in planned_segments
    ]

    frames: list[FrameRecord] = []
    for segment in written_segments:
        frames.extend(
            sample_segment_frames(
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
            )
        )

    manifest = build_manifest(
        video_id=args.video_id,
        source_metadata=source_metadata,
        normalized_metadata=normalized_metadata,
        settings=settings,
        segments=written_segments,
        frames=frames,
        repo_root=repo_root,
    )
    manifest_path = manifests_dir / "preprocess_manifest.json"
    write_manifest(manifest, manifest_path)
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess exhibition video for downstream 3DGS stages.")
    parser.add_argument("source_video", type=Path, help="Path to the raw source video.")
    parser.add_argument("--output-root", type=Path, default=Path("data"), help="Root output directory.")
    parser.add_argument("--video-id", required=True, help="Stable video identifier used in output paths.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="baseline", help="Default preprocess preset.")
    parser.add_argument("--target-fps", type=float, default=None, help="Frame sampling rate per segment.")
    parser.add_argument("--max-long-edge", type=int, default=None, help="Maximum normalized video long edge.")
    parser.add_argument(
        "--segment-method",
        choices=["scene", "time", "scene,time"],
        default="scene,time",
        help="Segmentation strategy.",
    )
    parser.add_argument("--segment-length-sec", type=float, default=None, help="Maximum segment length.")
    parser.add_argument("--segment-overlap-sec", type=float, default=None, help="Overlap for split time windows.")
    parser.add_argument("--min-segment-sec", type=float, default=2.0, help="Merge or avoid tiny segments below this.")
    parser.add_argument("--blur-threshold", type=float, default=40.0, help="Reject frames below this Laplacian variance.")
    parser.add_argument("--overexposed-ratio", type=float, default=0.6, help="Reject frames above this white-pixel ratio.")
    parser.add_argument("--underexposed-ratio", type=float, default=0.6, help="Reject frames above this black-pixel ratio.")
    parser.add_argument("--duplicate-hash-threshold", type=int, default=4, help="Reject near-duplicate average hashes.")
    parser.add_argument("--save-rejected", action="store_true", help="Write rejected frame JPEGs as well as manifest rows.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directories for this video id.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        manifest_path = preprocess(args)
    except (PreprocessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote manifest: {relative_path(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
