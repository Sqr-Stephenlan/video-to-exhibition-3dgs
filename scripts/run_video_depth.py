"""
Batch Video-Depth-Anything inference for 3DGS pre-processing.

Usage:
    python scripts/run_video_depth.py \
        --input data/frames_filtered/ \
        --output data/depths/ \
        --model_size small

Output:
    One *_depth.npy per frame in the output directory.
    depth maps are in [H, W] float32, metric depth (relative scale).

Requirements:
    pip install transformers accelerate timm
    Model weights auto-downloaded from HuggingFace on first run.
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch

MODEL_SIZES = {
    "small": ("depth-anything/Video-Depth-Anything-Small", "apache-2.0"),
    "base": ("depth-anything/Video-Depth-Anything-Base", "non-commercial"),
    "large": ("depth-anything/Video-Depth-Anything-Large", "non-commercial"),
}


def resolve_model_name(model_size: str) -> str:
    if model_size in MODEL_SIZES:
        return MODEL_SIZES[model_size][0]
    if "/" in model_size:
        return model_size
    raise ValueError(f"Unknown model size: {model_size}. Choose: {list(MODEL_SIZES.keys())}")


def load_model(model_name: str, device: str):
    """Load Video-Depth-Anything pipeline. Returns a callable (image) -> depth_np."""
    from transformers import pipeline

    print(f"Loading model: {model_name}")
    pipe = pipeline(
        task="depth-estimation",
        model=model_name,
        device=device,
    )
    return pipe


def process_frames(pipe, frame_paths, output_dir, max_resolution=512):
    """Run depth inference on all frames, writing *_depth.npy to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    for frame_path in tqdm(frame_paths, desc="Depth inference"):
        stem = Path(frame_path).stem
        out_path = os.path.join(output_dir, f"{stem}_depth.npy")

        if os.path.exists(out_path):
            continue

        try:
            from PIL import Image
            image = Image.open(frame_path).convert("RGB")

            prediction = pipe(image)
            depth_np = np.array(prediction["depth"])

            np.save(out_path, depth_np.astype(np.float32))
        except Exception as e:
            print(f"  [WARN] Failed on {frame_path}: {e}", file=sys.stderr)
            continue


def main():
    parser = argparse.ArgumentParser(description="Video-Depth-Anything batch inference")
    parser.add_argument("--input", required=True, help="Directory of input frame images")
    parser.add_argument("--output", required=True, help="Directory for output depth .npy files")
    parser.add_argument("--model_size", default="small", choices=["small", "base", "large"],
                        help="Model variant (default: small, 8GB VRAM compatible)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Torch device string")
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

    print(f"Found {len(frame_paths)} frames.")
    model_name = resolve_model_name(args.model_size)
    pipe = load_model(model_name, args.device)
    process_frames(pipe, frame_paths, args.output)


if __name__ == "__main__":
    main()
