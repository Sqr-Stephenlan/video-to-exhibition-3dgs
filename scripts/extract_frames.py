"""
Extract frames from video at a configurable FPS and maximum resolution.

Usage::

    python scripts/extract_frames.py --input data/raw_videos/scene.mp4 --fps 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _common import (
    ensure_output_dir,
    get_project_root,
    now_utc_iso,
    safe_parse_fps,
    write_json_atomic,
)


def get_video_info(video_path: str) -> dict:
    """Query video metadata via ffprobe (safe FPS parsing, no eval)."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    info = json.loads(result.stdout)
    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    if not video_stream:
        raise RuntimeError(f"No video stream found in {video_path}")

    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    fps = safe_parse_fps(r_frame_rate)
    if fps <= 0:
        raise RuntimeError(f"Invalid frame rate '{r_frame_rate}' in {video_path}")

    duration_s = float(info.get("format", {}).get("duration", 0))
    if duration_s <= 0:
        print(f"Warning: duration is {duration_s}s for {video_path}", file=sys.stderr)

    return {
        "duration": duration_s,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "codec": video_stream.get("codec_name", "unknown"),
        "r_frame_rate": r_frame_rate,
    }


def extract_frames(
    video_path: str,
    output_dir: Path,
    fps: float = 5.0,
    max_size: int = 1600,
    quality: int = 2,
) -> list[Path]:
    """Extract frames from *video_path* into *output_dir* via ffmpeg."""
    video_info = get_video_info(video_path)
    ensure_output_dir(output_dir)

    scale_filter = (
        f"scale='min({max_size},iw)':'min({max_size},ih)':force_original_aspect_ratio=decrease"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},{scale_filter}",
        "-q:v",
        str(quality),
        "-frame_pts",
        "1",
        str(output_dir / "frame_%06d.jpg"),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"FFmpeg failed:\n{exc.stderr}", file=sys.stderr)
        raise

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        print("Warning: no frames extracted.", file=sys.stderr)

    meta = {
        "video": Path(video_path).name,
        "video_path": str(Path(video_path).resolve()),
        "video_info": video_info,
        "extraction_fps": fps,
        "max_size": max_size,
        "quality": quality,
        "num_frames": len(frames),
        "created": now_utc_iso(),
    }
    write_json_atomic(meta, output_dir / "_extraction_meta.json")

    print(f"Extracted {len(frames)} frame(s) -> {output_dir}")
    return frames


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <project>/data/frames)",
    )
    parser.add_argument("--fps", type=float, default=5.0, help="Extraction frame rate")
    parser.add_argument("--max_size", type=int, default=1600, help="Long-edge max pixels")
    parser.add_argument("--quality", type=int, default=2, help="JPEG quality 2-31 (lower=better)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else get_project_root() / "data" / "frames"

    extract_frames(args.input, output_dir, args.fps, args.max_size, args.quality)


if __name__ == "__main__":
    main()
