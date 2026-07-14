"""
按时间窗口或场景切换将长视频切分为短片段。
用法:
    python split_video.py --input scene.mp4 --method time --segment_duration 30
    python split_video.py --input scene.mp4 --method scene  （使用 PySceneDetect）
"""

import argparse
import json
import subprocess
from pathlib import Path


def split_by_duration(video_path: str, output_dir: str, duration: int = 30) -> list:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(duration),
        "-f", "segment",
        "-reset_timestamps", "1",
        f"{out}/segment_%03d.mp4",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    segments = sorted(out.glob("segment_*.mp4"))
    print(f"按时间窗口分段完成: {len(segments)} 个片段 -> {out}")
    return segments


def split_by_scene(video_path: str, output_dir: str, threshold: float = 30.0) -> list:
    """使用 PySceneDetect 按内容变化检测场景切换。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 先用 PySceneDetect 检测分割时间点
    detect_cmd = [
        "scenedetect", "-i", str(video_path),
        "detect-adaptive", "-t", str(threshold),
        "list-scenes", "-q",
    ]
    result = subprocess.run(detect_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"PySceneDetect 未检测到切换点，回退到时间分段: {result.stderr}")
        return split_by_duration(video_path, output_dir)

    # 解析时间点并用 FFmpeg 切割
    lines = result.stdout.strip().split("\n")
    time_points = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                time_points.append(parts[0].strip())
            except ValueError:
                continue

    if len(time_points) < 2:
        return split_by_duration(video_path, output_dir)

    segments = []
    prev_t = time_points[0]
    for idx, curr_t in enumerate(time_points[1:], 1):
        seg_path = out / f"segment_{idx:03d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", prev_t,
            "-to", curr_t,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(seg_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        segments.append(seg_path)
        prev_t = curr_t

    print(f"按场景切换分段完成: {len(segments)} 个片段 -> {out}")
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频分段")
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument("--output", default="../data/segments", help="输出目录")
    parser.add_argument("--method", choices=["time", "scene"], default="time",
                        help="分段方式: time=时间窗口, scene=场景切换检测")
    parser.add_argument("--segment_duration", type=int, default=30,
                        help="每段时长(秒)，仅 method=time 时有效")
    parser.add_argument("--scene_threshold", type=float, default=30.0,
                        help="场景切换检测灵敏度，仅 method=scene 时有效")
    args = parser.parse_args()

    if args.method == "time":
        split_by_duration(args.input, args.output, args.segment_duration)
    else:
        split_by_scene(args.input, args.output, args.scene_threshold)
