"""
生成帧序列的 manifest 元数据文件，为后续重建提供统一数据描述。
用法:
    python build_manifest.py --input ../data/frames_filtered/ --segment scene_01
"""

import argparse
import json
from pathlib import Path

import cv2


def build_manifest(
    frames_dir: str,
    segment_name: str = "scene_01",
    output_dir: str = "../data/manifests",
) -> str:
    in_dir = Path(frames_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.png"))
    if not frames:
        frames = sorted(in_dir.glob("frame_*.*"))

    if not frames:
        raise FileNotFoundError(f"在 {frames_dir} 中未找到图像帧")

    entries = []
    for idx, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        h, w = img.shape[:2]
        entries.append({
            "id": idx,
            "file": str(fp.relative_to(in_dir.parent)),
            "width": w,
            "height": h,
        })

    first = frames[0]
    base = Path(output_dir).parent
    paths = {
        "frames_dir": str(in_dir.relative_to(base)),
        "depth_dir": f"{segment_name}/depth",
        "segments_dir": f"{segment_name}/segments",
        "mask_dir": f"{segment_name}/masks",
    }

    manifest = {
        "segment": segment_name,
        "num_frames": len(entries),
        "paths": paths,
        "frames": entries,
    }

    out_path = out_dir / f"{segment_name}_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest 已生成: {out_path}  ({len(entries)} 帧)")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成帧序列 manifest")
    parser.add_argument("--input", required=True, help="输入帧目录")
    parser.add_argument("--segment", default="scene_01", help="段名称")
    parser.add_argument("--output", default="../data/manifests", help="manifest 输出目录")
    args = parser.parse_args()

    build_manifest(args.input, args.segment, args.output)
