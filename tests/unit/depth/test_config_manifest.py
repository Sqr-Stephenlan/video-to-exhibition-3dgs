from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT))

from scripts.depth.config import load_config, validate_config
from scripts.depth.manifest import build_depth_manifest, selected_frames


def test_default_vitb_config_loads() -> None:
    config = load_config(ROOT / "configs" / "depth" / "default_vitb.yaml")
    assert config["backend"]["encoder"] == "vitb"
    assert config["backend"]["depth_type"] == "relative"


def test_validate_config_rejects_bad_encoder() -> None:
    config = load_config(ROOT / "configs" / "depth" / "default_vitb.yaml")
    config["backend"]["encoder"] = "vitx"
    with pytest.raises(ValueError, match="encoder"):
        validate_config(config)


def test_selected_frames_and_depth_manifest() -> None:
    frames_manifest = {
        "video_id": "demo",
        "frames": [
            {"frame_id": "a", "path": "data/frames/a.png", "selected": True},
            {"frame_id": "b", "path": "data/frames/b.png", "selected": False},
        ],
    }
    selected = selected_frames(frames_manifest)
    assert [item["frame_id"] for item in selected] == ["a"]
    depth_manifest = build_depth_manifest(
        frames_manifest=frames_manifest,
        frame_records=[
            {
                "frame_id": "a",
                "rgb_path": "data/frames/a.png",
                "depth_path": "data/depth/a.npz",
                "depth_type": "relative",
                "confidence_path": None,
            }
        ],
        backend={"name": "video-depth-anything", "commit": "4f5ae23", "encoder": "vitb"},
        depth_type="relative",
    )
    assert depth_manifest["backend"]["encoder"] == "vitb"
    assert len(depth_manifest["frames"]) == 1
