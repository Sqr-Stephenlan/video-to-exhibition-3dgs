"""
Split long exhibition video into shorter segments.

Supports two methods:

* ``time``  – fixed-duration segments (FFmpeg segment muxer).
* ``scene`` – scene-change detection via PySceneDetect's Python API, with
  optional re-encoding for keyframe-precise cuts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _common import ensure_output_dir, get_project_root, now_utc_iso

# ---------------------------------------------------------------------------
# Time-based splitting
# ---------------------------------------------------------------------------


def split_by_duration(video_path: str, output_dir: Path, duration: int = 30) -> list[Path]:
    """Split *video_path* into equal-length segments via FFmpeg segment muxer."""
    ensure_output_dir(output_dir)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-c",
        "copy",
        "-map",
        "0",
        "-segment_time",
        str(duration),
        "-f",
        "segment",
        "-reset_timestamps",
        "1",
        str(output_dir / "segment_%03d.mp4"),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"FFmpeg failed:\n{exc.stderr}", file=sys.stderr)
        raise

    segments = sorted(output_dir.glob("segment_*.mp4"))
    if not segments:
        print("Warning: no segments produced.", file=sys.stderr)
    print(f"Time-based split: {len(segments)} segment(s) -> {output_dir}")
    return segments


# ---------------------------------------------------------------------------
# Scene-change-based splitting
# ---------------------------------------------------------------------------


def split_by_scene(
    video_path: str,
    output_dir: Path,
    threshold: float = 30.0,
    min_scene_len: int = 15,
    precision: str = "fast",
) -> list[Path]:
    """Detect scene changes with PySceneDetect and split accordingly.

    Parameters
    ----------
    precision:
        ``"fast"``  – stream-copy (fast, may shift on non-keyframe cuts).
        ``"exact"`` – re-encode with fixed GOP for frame-accurate boundaries.
    """
    try:
        from scenedetect import AdaptiveDetector, SceneManager, open_video
    except ImportError:
        print(
            "PySceneDetect is not installed. Install: pip install scenedetect",
            file=sys.stderr,
        )
        sys.exit(1)

    ensure_output_dir(output_dir)

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=threshold,
            min_scene_len=min_scene_len,
        )
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        print("No scene changes detected. Exporting entire video as one segment.")
        from scenedetect import FrameTimecode

        scene_list = [
            (
                video.base_timecode,
                FrameTimecode(video.duration.frame_num, video.frame_rate),
            )
        ]

    print(f"Detected {len(scene_list)} scene(s).")

    segments: list[Path] = []
    for idx, (start_tc, end_tc) in enumerate(scene_list):
        seg_path = output_dir / f"segment_{idx + 1:03d}.mp4"
        start_sec = start_tc.get_seconds()
        duration_sec = end_tc.get_seconds() - start_sec

        if precision == "exact":
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-ss",
                str(start_sec),
                "-t",
                str(duration_sec),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-g",
                "30",
                "-avoid_negative_ts",
                "make_zero",
                str(seg_path),
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-ss",
                str(start_sec),
                "-to",
                str(end_tc.get_seconds()),
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(seg_path),
            ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            print(f"FFmpeg failed for segment {idx + 1}:\n{exc.stderr}", file=sys.stderr)
            raise

        segments.append(seg_path)
        print(f"  Segment {idx + 1}: {start_tc} -> {end_tc}  ({seg_path.name})")

    meta = {
        "video_path": str(Path(video_path).resolve()),
        "method": "scene",
        "detector": "AdaptiveDetector",
        "threshold": threshold,
        "min_scene_len": min_scene_len,
        "precision": precision,
        "num_scenes": len(scene_list),
        "scenes": [
            {
                "start": str(s),
                "end": str(e),
                "start_sec": s.get_seconds(),
                "end_sec": e.get_seconds(),
            }
            for s, e in scene_list
        ],
        "segments": [str(p.resolve()) for p in segments],
        "created": now_utc_iso(),
    }
    meta_path = output_dir / "_split_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata written to {meta_path}")

    return segments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a video into segments.")
    parser.add_argument("--input", required=True, help="Path to input video.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <project>/data/segments).",
    )
    parser.add_argument(
        "--method",
        choices=["time", "scene"],
        default="time",
        help="Splitting method (default: time).",
    )
    parser.add_argument(
        "--segment_duration",
        type=int,
        default=30,
        help="Segment duration in seconds for --method time (default: 30).",
    )
    parser.add_argument(
        "--scene_threshold",
        type=float,
        default=30.0,
        help="AdaptiveDetector threshold for --method scene (default: 30.0).",
    )
    parser.add_argument(
        "--min_scene_len",
        type=int,
        default=15,
        help="Minimum scene length in frames (default: 15).",
    )
    parser.add_argument(
        "--precision",
        choices=["fast", "exact"],
        default="fast",
        help="Cut precision: fast (stream-copy) or exact (re-encode).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing output directory before processing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing output directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else get_project_root() / "data" / "segments"

    # Pre-validate output directory lifecycle
    ensure_output_dir(output_dir, overwrite=args.overwrite, resume=args.resume)

    if args.method == "scene":
        split_by_scene(
            args.input,
            output_dir,
            threshold=args.scene_threshold,
            min_scene_len=args.min_scene_len,
            precision=args.precision,
        )
    else:
        split_by_duration(args.input, output_dir, duration=args.segment_duration)


if __name__ == "__main__":
    main()
