"""
Video-Depth-Anything batch inference using the official VDA API.

Accepts a time-ordered frame sequence, loads the official
VideoDepthAnything model, and runs temporal-consistent depth inference
on the full sequence (not frame-by-frame).

Usage::

    python scripts/run_video_depth.py \\
        --input data/frames_filtered/ \\
        --output data/depths/ \\
        --model vda-small-relative
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from _common import (
    ensure_output_dir,
    get_project_root,
    now_utc_iso,
    write_json_atomic,
)
from _vda_adapter import VDA_MODELS, VDAInference
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Video-Depth-Anything temporal depth inference")
    parser.add_argument("--input", required=True, help="Directory of input frame images")
    parser.add_argument(
        "--output",
        default=None,
        help="Directory for output depth .npy files (default: <project>/data/depth)",
    )
    parser.add_argument(
        "--model",
        default="vda-small-relative",
        choices=list(VDA_MODELS.keys()),
        help="VDA model variant (default: vda-small-relative, 8 GB VRAM)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device string (default: cuda:0 if available, else cpu)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=518,
        help="VDA input resolution (default: 518)",
    )
    parser.add_argument(
        "--max_resolution",
        type=int,
        default=1024,
        help="Max short-side resolution for pre-inference resize",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use half-precision inference",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target FPS for temporal model (default: auto = frame count)",
    )
    parser.add_argument(
        "--max_frames_per_batch",
        type=int,
        default=None,
        help="Max frames per batch (None = all frames, may OOM on long videos)",
    )
    parser.add_argument(
        "--extensions",
        default=".jpg,.jpeg,.png,.bmp",
        help="Image extensions to glob (comma-separated)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear output directory before processing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run",
    )
    args = parser.parse_args()

    # -- Gather frames (sorted by name = temporal order) -------------------
    exts = [e.strip().lower() for e in args.extensions.split(",")]
    frame_paths = sorted(
        str(p) for p in Path(args.input).iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not frame_paths:
        print(f"No image files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    # -- Resolve device (torch-dependent) -----------------------------------
    if args.device == "auto":
        try:
            import torch

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    # -- Output directory --------------------------------------------------
    output_dir = Path(args.output) if args.output else get_project_root() / "data" / "depth"
    ensure_output_dir(output_dir, overwrite=args.overwrite, resume=args.resume)

    # -- Load model ---------------------------------------------------------
    spec = VDA_MODELS[args.model]
    vda = VDAInference(
        checkpoint=spec.checkpoint,
        depth_type=spec.depth_type,
        device=device,
        input_size=args.input_size,
        fp16=args.fp16,
        max_frames_per_batch=args.max_frames_per_batch,
    )
    print(f"Model: {args.model}  depth_type: {spec.depth_type}  device: {device}")

    # -- Preprocess (resize) ------------------------------------------------
    from PIL import Image

    print(f"Loading {len(frame_paths)} frames …")
    images = []
    for fp in tqdm(frame_paths, desc="Loading"):
        images.append(Image.open(fp).convert("RGB"))

    print(f"Resizing (max_resolution={args.max_resolution}) …")
    images = vda.preprocess(images, max_resolution=args.max_resolution)

    # -- Inference ----------------------------------------------------------
    fps = args.fps if args.fps is not None else float(len(frame_paths))
    print(f"Running VDA inference (fps={fps:.1f}, {len(images)} frames) …")

    try:
        depth_maps = vda.infer_video_depth(images, fps=fps)
    except NotImplementedError:
        print(
            "VDA adapter is a stub. "
            "Clone the locked VDA commit under third_party/Video-Depth-Anything "
            "and complete the _vda_adapter.py implementation.",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Save results -------------------------------------------------------
    print(f"Saving depth maps to {output_dir} …")
    for fp, depth in tqdm(
        zip(frame_paths, depth_maps, strict=True), total=len(frame_paths), desc="Saving"
    ):
        stem = Path(fp).stem
        np.save(output_dir / f"{stem}_depth.npy", depth.astype(np.float32))

    # -- Provenance ---------------------------------------------------------
    meta = vda.build_metadata(
        frame_paths=frame_paths,
        output_dir=str(output_dir),
        fps=fps,
        max_resolution=args.max_resolution,
    )
    meta["args"] = {k: str(v) for k, v in vars(args).items()}
    meta["created"] = now_utc_iso()
    write_json_atomic(meta, output_dir / "_depth_meta.json")
    print(f"Done. Metadata: {output_dir / '_depth_meta.json'}")


if __name__ == "__main__":
    main()
