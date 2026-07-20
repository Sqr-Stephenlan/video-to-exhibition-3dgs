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
    assert config["schema_version"] == "1.0"


def test_validate_config_rejects_bad_encoder() -> None:
    config = load_config(ROOT / "configs" / "depth" / "default_vitb.yaml")
    config["backend"]["encoder"] = "vitx"
    with pytest.raises(ValueError, match="encoder"):
        validate_config(config)


def test_validate_config_rejects_unsupported_schema_version() -> None:
    config = load_config(ROOT / "configs" / "depth" / "default_vitb.yaml")
    config["schema_version"] = "9.9"
    with pytest.raises(ValueError, match="schema_version"):
        validate_config(config)


def test_selected_frames_and_depth_manifest() -> None:
    frames_manifest = {
        "schema_version": "1.0",
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
        frames_manifest_path="data/manifests/frames_manifest.json",
    )
    assert depth_manifest["backend"]["encoder"] == "vitb"
    assert depth_manifest["source_video_id"] == "demo"
    assert depth_manifest["source_frames_manifest"] == "data/manifests/frames_manifest.json"
    assert depth_manifest["frame_depth_mapping"] == "strict_positional"
    assert depth_manifest["depth_scale"] == {"mode": "relative", "unit": None}
    assert len(depth_manifest["frames"]) == 1


def test_selected_frames_sorts_by_timestamp_sec() -> None:
    frames_manifest = {
        "schema_version": "1.0",
        "frames": [
            {
                "frame_id": "late",
                "path": "data/frames/late.png",
                "timestamp_sec": 1.2,
                "selected": True,
            },
            {
                "frame_id": "early",
                "path": "data/frames/early.png",
                "timestamp_sec": 0.1,
                "selected": True,
            },
        ],
    }
    selected = selected_frames(frames_manifest)
    assert [item["frame_id"] for item in selected] == ["early", "late"]


def test_selected_frames_rejects_mixed_timestamp_presence() -> None:
    frames_manifest = {
        "schema_version": "1.0",
        "frames": [
            {"frame_id": "a", "path": "data/frames/a.png", "timestamp_sec": 0.0},
            {"frame_id": "b", "path": "data/frames/b.png"},
        ],
    }
    with pytest.raises(ValueError, match="timestamp_sec"):
        selected_frames(frames_manifest)


def test_selected_frames_rejects_duplicate_frame_id() -> None:
    with pytest.raises(ValueError, match="Duplicate frame_id"):
        selected_frames(
            {
                "schema_version": "1.0",
                "frames": [
                    {"frame_id": "a", "path": "data/frames/a.png", "selected": True},
                    {"frame_id": "a", "path": "data/frames/a2.png", "selected": True},
                ],
            }
        )


def test_selected_frames_treats_null_timestamp_as_absent() -> None:
    selected = selected_frames(
        {
            "schema_version": "1.0",
            "frames": [
                {
                    "frame_id": "a",
                    "path": "data/frames/a.png",
                    "selected": True,
                    "timestamp_sec": None,
                },
                {"frame_id": "b", "path": "data/frames/b.png", "selected": True},
            ],
        }
    )
    assert [item["frame_id"] for item in selected] == ["a", "b"]
