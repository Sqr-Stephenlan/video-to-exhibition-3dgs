"""
Filter low-quality frames by blur, exposure, and near-duplicate similarity.

Uses ``imagehash.phash`` for perceptual hashing (no OpenCV contrib needed).

Usage::

    python scripts/filter_blurry_frames.py --input data/frames/ --blur_threshold 100
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
from _common import (
    ensure_output_dir,
    get_project_root,
    now_utc_iso,
    write_json_atomic,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def laplacian_variance(img: np.ndarray) -> float:
    """Blur metric. Higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_score(img: np.ndarray) -> float:
    """Fraction of over/under-exposed pixels (0 = perfect)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    over = int((gray > 250).sum())
    under = int((gray < 5).sum())
    return (over + under) / gray.size


def perceptual_hash_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Perceptual hash similarity (0–1, higher = more similar).

    Uses ``imagehash.phash`` (64-bit) with Hamming distance.
    """
    pil1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    pil2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    h1 = imagehash.phash(pil1)
    h2 = imagehash.phash(pil2)
    # h1.hash is a flat numpy bool array; .size gives total bits (64 for pHash)
    max_bits = int(h1.hash.size)
    hamming = int(h1 - h2)  # imagehash supports subtraction for Hamming distance
    return 1.0 - (hamming / max_bits)


# ---------------------------------------------------------------------------
# Main filtering
# ---------------------------------------------------------------------------


def filter_frames(
    input_dir: str,
    output_dir: Path,
    blur_threshold: float = 100.0,
    exposure_threshold: float = 0.3,
    similarity_threshold: float = 0.95,
) -> dict:
    in_dir = Path(input_dir)
    ensure_output_dir(output_dir)
    rejected_dir = output_dir.parent / "rejected_frames"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.png"))
    if not frames:
        frames = sorted(in_dir.glob("frame_*.*"))
    if not frames:
        raise FileNotFoundError(f"No image frames found in {input_dir}")

    results: list[dict] = []
    prev_img: np.ndarray | None = None

    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            results.append({"file": f.name, "status": "rejected", "reason": "unreadable/corrupt"})
            continue

        blur = laplacian_variance(img)
        exp = exposure_score(img)

        reject_reason: str | None = None
        if blur < blur_threshold:
            reject_reason = f"blurry ({blur:.1f} < {blur_threshold})"
        elif exp > exposure_threshold:
            reject_reason = f"bad exposure ({exp:.3f} > {exposure_threshold})"

        sim: float | None = None
        if reject_reason is None and prev_img is not None:
            sim = perceptual_hash_similarity(prev_img, img)
            if sim > similarity_threshold:
                reject_reason = f"near-duplicate (sim={sim:.4f})"

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
            shutil.copy2(f, output_dir / f.name)
            prev_img = img

        results.append(result)

    summary = {
        "input_dir": str(Path(input_dir).resolve()),
        "total": len(results),
        "kept": sum(1 for r in results if r["status"] == "kept"),
        "rejected": sum(1 for r in results if r["status"] == "rejected"),
        "params": {
            "blur_threshold": blur_threshold,
            "exposure_threshold": exposure_threshold,
            "similarity_threshold": similarity_threshold,
            "hash_algorithm": "imagehash.phash (64-bit)",
        },
        "created": now_utc_iso(),
    }

    write_json_atomic(
        {"summary": summary, "frames": results},
        output_dir / "_filter_report.json",
    )

    print(f"Filtering done: {summary['kept']}/{summary['total']} frames kept")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter low-quality frames")
    parser.add_argument("--input", required=True, help="Input frames directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <project>/data/frames_filtered)",
    )
    parser.add_argument(
        "--blur_threshold",
        type=float,
        default=100.0,
        help="Laplacian variance threshold (lower = blurrier)",
    )
    parser.add_argument(
        "--exposure_threshold",
        type=float,
        default=0.3,
        help="Max over/under-exposed pixel fraction",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.95,
        help="pHash similarity threshold (higher = more similar)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = (
        Path(args.output) if args.output else get_project_root() / "data" / "frames_filtered"
    )

    filter_frames(
        args.input,
        output_dir,
        blur_threshold=args.blur_threshold,
        exposure_threshold=args.exposure_threshold,
        similarity_threshold=args.similarity_threshold,
    )


if __name__ == "__main__":
    main()
