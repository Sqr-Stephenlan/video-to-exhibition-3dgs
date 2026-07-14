"""
从视频中提取帧，支持自定义帧率、分辨率、输出格式。
用法:
    python extract_frames.py --input ../data/raw_videos/scene.mp4 --fps 5 --max_size 1600
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_video_info(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 失败: {result.stderr}")

    info = json.loads(result.stdout)
    video_stream = next(
        (s for s in info.get("streams", []) if s["codec_type"] == "video"), {}
    )
    return {
        "duration": float(info.get("format", {}).get("duration", 0)),
        "width": video_stream.get("width", 0),
        "height": video_stream.get("height", 0),
        "fps": eval(video_stream.get("r_frame_rate", "0/1")),
        "codec": video_stream.get("codec_name", "unknown"),
    }


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 5.0,
    max_size: int = 1600,
    quality: int = 2,
) -> list[Path]:
    video_info = get_video_info(video_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scale_filter = f"scale='min({max_size},iw)':'min({max_size},ih)':force_original_aspect_ratio=decrease"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps},{scale_filter}",
        "-q:v", str(quality),
        "-frame_pts", "1",
        f"{out}/frame_%06d.jpg",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    frames = sorted(out.glob("frame_*.jpg"))
    meta = {
        "video": Path(video_path).name,
        "video_info": video_info,
        "extraction_fps": fps,
        "max_size": max_size,
        "num_frames": len(frames),
    }
    with open(out / "_extraction_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"提取完成: {len(frames)} 帧 -> {out}")
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从视频提取帧")
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--output", default="../data/frames", help="输出目录")
    parser.add_argument("--fps", type=float, default=5.0, help="提取帧率")
    parser.add_argument("--max_size", type=int, default=1600, help="长边最大像素")
    parser.add_argument("--quality", type=int, default=2, help="JPEG 质量 2-31，越小越好")
    args = parser.parse_args()

    extract_frames(args.input, args.output, args.fps, args.max_size, args.quality)
