"""
Adapter for Video-Depth-Anything (VDA) temporal depth inference.

Wraps the official VDA model and exposes a clean ``VDAInference`` class
that accepts a time-ordered frame sequence and returns per-frame depth
maps using VDA's temporal attention.

The locked VDA version is recorded in ``VDA_COMMIT`` below.

Reference: https://github.com/DepthAnything/Video-Depth-Anything
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

VDA_COMMIT = "HEAD"  # replaced when third-party clone is available


def _setup_vda_path() -> None:
    """Add third_party/Video-Depth-Anything to sys.path."""
    project_root = Path(__file__).resolve().parent.parent
    vda_root = project_root / "third_party" / "Video-Depth-Anything"
    if str(vda_root) not in sys.path:
        sys.path.insert(0, str(vda_root))


@dataclass(frozen=True)
class VDAModelSpec:
    checkpoint: str
    encoder: Literal["vits", "vitb", "vitl"]
    depth_type: Literal["relative", "metric"]
    license: str
    description: str


VDA_MODELS: dict[str, VDAModelSpec] = {
    "vda-small-relative": VDAModelSpec(
        checkpoint="video_depth_anything_vits",
        encoder="vits",
        depth_type="relative",
        license="apache-2.0",
        description="VDA Small – relative depth (8 GB VRAM compatible)",
    ),
    "vda-base-relative": VDAModelSpec(
        checkpoint="video_depth_anything_vitb",
        encoder="vitb",
        depth_type="relative",
        license="non-commercial",
        description="VDA Base – relative depth",
    ),
    "vda-large-relative": VDAModelSpec(
        checkpoint="video_depth_anything_vitl",
        encoder="vitl",
        depth_type="relative",
        license="non-commercial",
        description="VDA Large – relative depth",
    ),
    "vda-small-metric": VDAModelSpec(
        checkpoint="metric_video_depth_anything_vits",
        encoder="vits",
        depth_type="metric",
        license="apache-2.0",
        description="VDA Small – metric depth",
    ),
    "vda-base-metric": VDAModelSpec(
        checkpoint="metric_video_depth_anything_vitb",
        encoder="vitb",
        depth_type="metric",
        license="non-commercial",
        description="VDA Base – metric depth",
    ),
    "vda-large-metric": VDAModelSpec(
        checkpoint="metric_video_depth_anything_vitl",
        encoder="vitl",
        depth_type="metric",
        license="non-commercial",
        description="VDA Large – metric depth",
    ),
}

# VDA encoder-specific model configs (matches the official run.py)
_VDA_ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _resolve_vda_checkpoint(checkpoint: str) -> str:
    """Resolve a VDA checkpoint name to a local .pth file."""
    if Path(checkpoint).is_file():
        return checkpoint

    # checkpoint is a stem name like "video_depth_anything_vits"
    fname = f"{checkpoint}.pth" if not checkpoint.endswith(".pth") else checkpoint

    project_root = Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "third_party" / "Video-Depth-Anything" / "checkpoints" / fname,
        project_root / "checkpoints" / fname,
        Path("checkpoints") / fname,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    # Fallback: assume it's in the CWD checkpoints dir
    return str(Path("checkpoints") / fname)


class VDAInference:
    """Video-Depth-Anything temporal inference wrapper.

    Loads the official VDA model and runs ``infer_video_depth()`` on a
    time-ordered sequence of frames, making use of VDA's temporal attention.
    """

    def __init__(
        self,
        checkpoint: str,
        depth_type: Literal["relative", "metric"],
        device: str = "cuda:0",
        input_size: int = 518,
        fp16: bool = True,
        max_frames_per_batch: int | None = None,
    ) -> None:
        self.checkpoint = _resolve_vda_checkpoint(checkpoint)
        self.depth_type = depth_type
        self.device = device
        self.input_size = input_size
        self.fp16 = fp16
        self.max_frames_per_batch = max_frames_per_batch
        self._model = None  # lazy-load
        self._encoder: str | None = None

    @property
    def version(self) -> str:
        return f"VDA@{VDA_COMMIT}:{self.checkpoint}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, images: list, max_resolution: int = 1024) -> list:
        """Resize PIL images so the short side <= *max_resolution*.

        This is a local pre-processing step that runs before VDA inference.
        """
        from PIL import Image

        resized: list[Image.Image] = []
        for img in images:
            w, h = img.size
            short = min(w, h)
            if short > max_resolution:
                ratio = max_resolution / short
                new_w, new_h = int(w * ratio), int(h * ratio)
                img = img.resize((new_w, new_h), Image.BICUBIC)
            resized.append(img)
        return resized

    def infer_video_depth(self, images: list, fps: float = 30.0) -> list[np.ndarray]:
        """Run official VDA temporal inference on a frame sequence.

        Parameters
        ----------
        images : list[PIL.Image]
            Time-ordered frame sequence (RGB).
        fps : float
            Frame rate hint for the temporal model.

        Returns
        -------
        list[np.ndarray]
            Per-frame depth maps, each ``(H, W) float32``.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file cannot be found.
        """
        _setup_vda_path()

        import torch

        from video_depth_anything.video_depth import VideoDepthAnything

        # Lazy-load model
        if self._model is None:
            # Auto-detect encoder from checkpoint filename
            ckpt_stem = Path(self.checkpoint).stem.lower()
            if "vits" in ckpt_stem or "small" in ckpt_stem:
                self._encoder = "vits"
            elif "vitb" in ckpt_stem or "base" in ckpt_stem:
                self._encoder = "vitb"
            else:
                self._encoder = "vitl"

            metric = self.depth_type == "metric"
            config = _VDA_ENCODER_CONFIGS[self._encoder]
            self._model = VideoDepthAnything(**config, metric=metric)
            self._model.load_state_dict(
                torch.load(self.checkpoint, map_location="cpu", weights_only=True),
                strict=True,
            )
            self._model = self._model.to(self.device).eval()

        # Convert PIL images to numpy array (T, H, W, 3)
        frames_np = np.stack([np.array(img) for img in images])

        # Batch processing for memory-constrained GPUs
        if self.max_frames_per_batch and len(frames_np) > self.max_frames_per_batch:
            all_depths: list[np.ndarray] = []
            for start in range(0, len(frames_np), self.max_frames_per_batch):
                batch = frames_np[start : start + self.max_frames_per_batch]
                depths, fps_out = self._model.infer_video_depth(
                    batch, fps,
                    input_size=self.input_size,
                    device=self.device,
                    fp32=not self.fp16,
                )
                all_depths.append(depths)
            depths = np.concatenate(all_depths, axis=0)
        else:
            depths, fps_out = self._model.infer_video_depth(
                frames_np, fps,
                input_size=self.input_size,
                device=self.device,
                fp32=not self.fp16,
            )

        return [depths[i].astype(np.float32) for i in range(len(depths))]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def build_metadata(
        self,
        frame_paths: list[str],
        output_dir: str,
        fps: float,
        max_resolution: int,
    ) -> dict:
        return {
            "vda_version": self.version,
            "checkpoint": self.checkpoint,
            "depth_type": self.depth_type,
            "encoder": self._encoder,
            "device": self.device,
            "input_size": self.input_size,
            "fp16": self.fp16,
            "fps": fps,
            "max_resolution": max_resolution,
            "num_frames": len(frame_paths),
            "frame_paths": [str(p) for p in frame_paths],
            "output_dir": str(output_dir),
        }
