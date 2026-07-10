"""
Exposure normalization across frames to reduce auto-exposure flicker.

Algorithm: match each frame's mean/variance to the reference frame's.

Usage::

    python scripts/normalize_exposure.py \\
        --input data/frames_filtered/ \\
        --output data/frames_normalized/ \\
        --reference first
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from _common import (
    ensure_output_dir,
    get_project_root,
    now_utc_iso,
    write_json_atomic,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------


def compute_stats(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel (BGR) mean and std in [0, 1] space."""
    means, stds = [], []
    for c in range(3):
        ch = img[:, :, c].astype(np.float32) / 255.0
        means.append(ch.mean())
        stds.append(ch.std())
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def match_histogram(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Match the histogram of *src* to *ref* per-channel."""
    result = np.empty_like(src)
    for c in range(3):
        src_ch = src[:, :, c]
        ref_ch = ref[:, :, c]
        n_bins = 256
        src_hist, _ = np.histogram(src_ch, n_bins, [0, 256])
        ref_hist, _ = np.histogram(ref_ch, n_bins, [0, 256])
        src_cdf = src_hist.cumsum().astype(np.float64) / src_hist.sum()
        ref_cdf = ref_hist.cumsum().astype(np.float64) / ref_hist.sum()
        lut = np.interp(src_cdf, ref_cdf, np.arange(n_bins)).astype(np.uint8)
        result[:, :, c] = lut[src_ch]
    return result


def simple_normalize(src: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray) -> np.ndarray:
    """Match mean and std of *src* to *ref* per-channel."""
    src_f = src.astype(np.float32) / 255.0
    src_mean, src_std = compute_stats(src)

    for c in range(3):
        if src_std[c] > 1e-8:
            src_f[:, :, c] = (src_f[:, :, c] - src_mean[c]) / src_std[c] * ref_std[c] + ref_mean[c]
        else:
            src_f[:, :, c] = src_f[:, :, c] - src_mean[c] + ref_mean[c]

    return np.clip(src_f * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Exposure normalization")
    parser.add_argument("--input", required=True, help="Input frames directory")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: <project>/data/frames_normalized)",
    )
    parser.add_argument(
        "--reference",
        default="first",
        choices=["first", "median", "path"],
        help="Reference frame selection",
    )
    parser.add_argument(
        "--ref_path",
        default="",
        help="Path to reference image (only for --reference path)",
    )
    parser.add_argument(
        "--method",
        default="simple",
        choices=["simple", "histogram"],
        help="Normalization method",
    )
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.png,.bmp",
        help="Image extensions (comma-separated)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    exts = [e.strip().lower() for e in args.extensions.split(",")]
    frame_paths = sorted(
        str(p) for p in Path(args.input).iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not frame_paths:
        print(f"No image files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    output_dir = (
        Path(args.output) if args.output else get_project_root() / "data" / "frames_normalized"
    )
    ensure_output_dir(output_dir, overwrite=args.overwrite, resume=args.resume)

    # Reference frame
    if args.reference == "first":
        ref_path = frame_paths[0]
    elif args.reference == "median":
        ref_path = frame_paths[len(frame_paths) // 2]
    else:
        ref_path = args.ref_path
        if not ref_path or not Path(ref_path).exists():
            print("Provide a valid --ref_path for --reference path", file=sys.stderr)
            sys.exit(1)

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"Failed to read reference image: {ref_path}", file=sys.stderr)
        sys.exit(1)

    ref_mean, ref_std = compute_stats(ref_img)
    print(f"Reference: {ref_path}  mean={ref_mean}  std={ref_std}")

    skipped = 0
    for fpath in tqdm(frame_paths, desc="Normalizing"):
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [SKIP] Cannot read {fpath}", file=sys.stderr)
            skipped += 1
            continue

        out_img = (
            simple_normalize(img, ref_mean, ref_std)
            if args.method == "simple"
            else match_histogram(img, ref_img)
        )

        out_path = output_dir / Path(fpath).name
        ok = cv2.imwrite(str(out_path), out_img)
        if not ok:
            print(f"  [FAIL] Cannot write {out_path}", file=sys.stderr)
            skipped += 1

    # Provenance
    meta = {
        "input_dir": str(Path(args.input).resolve()),
        "reference": args.reference,
        "ref_path": str(ref_path),
        "method": args.method,
        "total": len(frame_paths),
        "successful": len(frame_paths) - skipped,
        "skipped": skipped,
        "ref_mean": ref_mean.tolist(),
        "ref_std": ref_std.tolist(),
        "created": now_utc_iso(),
    }
    write_json_atomic(meta, output_dir / "_exposure_meta.json")
    print(f"Normalization done: {meta['successful']}/{meta['total']} frames")


if __name__ == "__main__":
    main()
