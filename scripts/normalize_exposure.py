"""
Exposure normalization across frames to reduce auto-exposure flicker.

Algorithm: match each frame's mean/variance to the reference frame's.

Usage:
    python scripts/normalize_exposure.py \
        --input data/frames_filtered/ \
        --output data/frames_normalized/ \
        --reference first
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def compute_stats(img):
    """Compute per-channel mean and std of an image."""
    means = []
    stds = []
    for c in range(3):
        ch = img[:, :, c].astype(np.float32) / 255.0
        means.append(ch.mean())
        stds.append(ch.std())
    return np.array(means), np.array(stds)


def match_histogram(src, ref):
    """Match the histogram of src to ref per-channel. Returns uint8 image."""
    result = np.empty_like(src)
    for c in range(3):
        src_ch = src[:, :, c]
        ref_ch = ref[:, :, c]
        n_bins = 256

        src_hist, _ = np.histogram(src_ch, n_bins, [0, 256])
        ref_hist, _ = np.histogram(ref_ch, n_bins, [0, 256])

        src_cdf = src_hist.cumsum() / src_hist.sum()
        ref_cdf = ref_hist.cumsum() / ref_hist.sum()

        lut = np.interp(src_cdf, ref_cdf, np.arange(n_bins)).astype(np.uint8)
        result[:, :, c] = lut[src_ch]
    return result


def simple_normalize(src, ref_mean, ref_std):
    """Match mean and std of src to ref. Returns uint8 image."""
    src_f = src.astype(np.float32) / 255.0
    src_mean, src_std = compute_stats(src)

    for c in range(3):
        if src_std[c] > 1e-8:
            src_f[:, :, c] = (src_f[:, :, c] - src_mean[c]) / src_std[c] * ref_std[c] + ref_mean[c]
        else:
            src_f[:, :, c] = src_f[:, :, c] - src_mean[c] + ref_mean[c]

    src_f = np.clip(src_f * 255.0, 0, 255).astype(np.uint8)
    return src_f


def main():
    parser = argparse.ArgumentParser(description="Exposure normalization across frames")
    parser.add_argument("--input", required=True, help="Directory of input frames")
    parser.add_argument("--output", required=True, help="Directory for normalized frames")
    parser.add_argument("--reference", default="first", choices=["first", "median", "path"],
                        help="How to choose the reference frame")
    parser.add_argument("--ref_path", default="", help="Path to reference image (only for --reference path)")
    parser.add_argument("--method", default="simple", choices=["simple", "histogram"],
                        help="Normalization method")
    parser.add_argument("--extensions", default=".jpg,.jpeg,.png,.bmp", help="Image extensions")
    args = parser.parse_args()

    exts = [e.strip() for e in args.extensions.split(",")]
    frame_paths = sorted(
        str(p) for p in Path(args.input).iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )

    if not frame_paths:
        print(f"No image files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    # Select reference image
    if args.reference == "first":
        ref_path = frame_paths[0]
    elif args.reference == "median":
        ref_path = frame_paths[len(frame_paths) // 2]
    else:
        ref_path = args.ref_path
        if not ref_path or not os.path.exists(ref_path):
            print("Please provide a valid --ref_path for --reference path", file=sys.stderr)
            sys.exit(1)

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"Failed to read reference: {ref_path}", file=sys.stderr)
        sys.exit(1)

    ref_mean, ref_std = compute_stats(ref_img)
    print(f"Reference: {ref_path} (mean={ref_mean}, std={ref_std})")

    for fpath in tqdm(frame_paths, desc="Normalizing"):
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [SKIP] Cannot read {fpath}")
            continue

        if args.method == "simple":
            out_img = simple_normalize(img, ref_mean, ref_std)
        else:
            out_img = match_histogram(img, ref_img)

        out_path = os.path.join(args.output, os.path.basename(fpath))
        cv2.imwrite(out_path, out_img)


if __name__ == "__main__":
    main()
