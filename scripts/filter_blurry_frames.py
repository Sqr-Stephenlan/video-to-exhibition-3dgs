"""
基于模糊度、曝光、相邻帧相似度过滤低质量帧。
用法:
    python filter_blurry_frames.py --input ../data/frames/ --threshold 100
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def laplacian_variance(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def exposure_score(img: np.ndarray) -> float:
    """过曝/欠曝比例，越接近0越好。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    over = (gray > 250).sum()
    under = (gray < 5).sum()
    total = gray.size
    return (over + under) / total


def perceptual_hash_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """感知哈希汉明距离相似度，越大越相似。"""
    h1 = cv2.img_hash.pHash(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    h2 = cv2.img_hash.pHash(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    hamming = cv2.norm(h1, h2, cv2.NORM_HAMMING)
    max_bits = len(h1) * 8
    return 1.0 - (hamming / max_bits)


def filter_frames(
    input_dir: str,
    output_dir: str,
    blur_threshold: float = 100.0,
    exposure_threshold: float = 0.3,
    similarity_keep_threshold: float = 0.95,
) -> dict:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = out_dir.parent / "rejected_frames"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.png"))
    if not frames:
        frames = sorted(in_dir.glob("frame_*.*"))
    if not frames:
        raise FileNotFoundError(f"在 {input_dir} 中未找到图像帧")

    results = []
    kept_paths = []
    prev_img = None

    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue

        blur = laplacian_variance(img)
        exp = exposure_score(img)

        reject_reason = None
        if blur < blur_threshold:
            reject_reason = f"模糊 ({blur:.1f} < {blur_threshold})"
        elif exp > exposure_threshold:
            reject_reason = f"曝光异常 ({exp:.3f} > {exposure_threshold})"

        if reject_reason is None and prev_img is not None:
            sim = perceptual_hash_similarity(prev_img, img)
            if sim > similarity_keep_threshold:
                reject_reason = f"与前一帧过于相似 ({sim:.4f})"
        else:
            sim = None

        result = {
            "file": f.name,
            "blur_score": round(blur, 2),
            "exposure_score": round(exp, 4),
            "similarity": round(sim, 4) if sim is not None else None,
        }

        if reject_reason:
            result["status"] = "rejected"
            result["reason"] = reject_reason
            shutil.copy2(f, rejected_dir / f.name)
        else:
            result["status"] = "kept"
            shutil.copy2(f, out_dir / f.name)
            kept_paths.append(out_dir / f.name)
            prev_img = img

        results.append(result)

    summary = {
        "input_dir": str(input_dir),
        "total": len(results),
        "kept": sum(1 for r in results if r["status"] == "kept"),
        "rejected": sum(1 for r in results if r["status"] == "rejected"),
        "params": {
            "blur_threshold": blur_threshold,
            "exposure_threshold": exposure_threshold,
            "similarity_keep_threshold": similarity_keep_threshold,
        },
    }

    # 写过滤结果
    with open(out_dir / "_filter_report.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "frames": results},
                  f, indent=2, ensure_ascii=False)

    print(f"过滤完成: {summary['kept']}/{summary['total']} 帧保留")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="过滤低质量帧")
    parser.add_argument("--input", required=True, help="输入帧目录")
    parser.add_argument("--output", default="../data/frames_filtered", help="输出目录")
    parser.add_argument("--blur_threshold", type=float, default=100.0,
                        help="拉普拉斯方差阈值，低于为模糊")
    parser.add_argument("--exposure_threshold", type=float, default=0.3,
                        help="过曝/欠曝像素比例阈值")
    parser.add_argument("--similarity_threshold", type=float, default=0.95,
                        help="pHashing 相似度阈值，高于该值视为重复帧")
    args = parser.parse_args()

    filter_frames(
        args.input, args.output,
        args.blur_threshold, args.exposure_threshold, args.similarity_threshold,
    )
