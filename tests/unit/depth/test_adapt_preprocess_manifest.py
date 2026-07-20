from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.depth.adapt_preprocess_manifest import (
    adapt_preprocess_to_frames_manifest,
    main as adapt_main,
)
from scripts.depth.config import load_config
from scripts.depth.manifest import selected_frames


def test_adapt_preprocess_manifest_maps_id_and_filters() -> None:
    payload = {
        "schema_version": "1.0",
        "video_id": "joint_smoke",
        "source": {"path": "data/raw_videos/joint_smoke.mp4"},
        "frames": [
            {
                "id": "segment_0001_frame_000001",
                "path": "data/frames/joint_smoke/selected/a.jpg",
                "selected": True,
                "timestamp_sec": 0.0,
            },
            {
                "id": "segment_0001_frame_000002",
                "path": "data/frames/joint_smoke/selected/b.jpg",
                "selected": False,
                "timestamp_sec": 0.2,
            },
            {
                "id": "segment_0001_frame_000003",
                "path": "data/frames/joint_smoke/selected/c.jpg",
                "selected": True,
                "timestamp_sec": 0.4,
            },
        ],
    }
    adapted = adapt_preprocess_to_frames_manifest(
        payload,
        frames_manifest_path="data/manifests/joint_smoke/preprocess_manifest.json",
    )
    assert adapted["schema_version"] == "1.0"
    assert adapted["video_id"] == "joint_smoke"
    assert adapted["source_video"] == "data/raw_videos/joint_smoke.mp4"
    assert [f["frame_id"] for f in adapted["frames"]] == [
        "segment_0001_frame_000001",
        "segment_0001_frame_000003",
    ]
    # Round-trip into depth-prior selection rules.
    selected = selected_frames(adapted)
    assert len(selected) == 2


def test_adapt_rejects_null_timestamp_mixed_with_numeric() -> None:
    payload = {
        "schema_version": "1.0",
        "frames": [
            {
                "id": "segment_0001_frame_000001",
                "path": "data/frames/a.jpg",
                "selected": True,
                "timestamp_sec": 0.0,
            },
            {
                "id": "segment_0001_frame_000002",
                "path": "data/frames/b.jpg",
                "selected": True,
                "timestamp_sec": None,
            },
        ],
    }
    with pytest.raises(ValueError, match="timestamp_sec"):
        adapt_preprocess_to_frames_manifest(payload)


def test_adapt_rejects_selected_without_path() -> None:
    payload = {
        "schema_version": "1.0",
        "frames": [
            {
                "id": "segment_0001_frame_000001",
                "path": None,
                "selected": True,
            }
        ],
    }
    with pytest.raises(ValueError, match="selected but path"):
        adapt_preprocess_to_frames_manifest(payload)


def test_adapt_rejects_path_escape() -> None:
    payload = {
        "schema_version": "1.0",
        "frames": [
            {
                "id": "segment_0001_frame_000001",
                "path": "../outside.jpg",
                "selected": True,
            }
        ],
    }
    with pytest.raises(ValueError, match="escapes"):
        adapt_preprocess_to_frames_manifest(payload)


def test_adapt_cli_rejects_output_escape(tmp_path: Path) -> None:
    src = ROOT / "data" / "manifests" / "_tmp_adapt_test" / "preprocess_manifest.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        '{"schema_version":"1.0","frames":[{"id":"segment_0001_frame_000001",'
        '"path":"data/frames/a.jpg","selected":true,"timestamp_sec":0.0}]}',
        encoding="utf-8",
    )
    try:
        code = adapt_main(
            [
                "data/manifests/_tmp_adapt_test/preprocess_manifest.json",
                "--output",
                "../escaped_frames_manifest.json",
            ]
        )
        assert code == 2
    finally:
        src.unlink(missing_ok=True)
        try:
            src.parent.rmdir()
        except OSError:
            pass


def test_smoke_joint_config_loads() -> None:
    config = load_config(ROOT / "configs" / "depth" / "smoke_joint.yaml")
    assert config["backend"]["input_size"] == 308
    assert config["backend"]["max_res"] == 512
    assert config["backend"]["allow_custom_checkpoint"] is False
