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
                "width": 384,
                "height": 512,
                "segment_id": "segment_0001",
                "reason": "sharpest_in_window",
            },
            {
                "id": "segment_0001_frame_000002",
                "path": "data/frames/joint_smoke/selected/b.jpg",
                "selected": False,
                "timestamp_sec": 0.2,
                "width": 384,
                "height": 512,
            },
            {
                "id": "segment_0001_frame_000003",
                "path": "data/frames/joint_smoke/selected/c.jpg",
                "selected": True,
                "timestamp_sec": 0.4,
                "width": 384,
                "height": 512,
                "segment_id": "segment_0001",
                "blur_score": 0.9,
            },
        ],
    }
    adapted = adapt_preprocess_to_frames_manifest(
        payload,
        frames_manifest_path="data/manifests/joint_smoke/frames_manifest.json",
        fill_missing_size=False,
    )
    assert adapted["schema_version"] == "1.0"
    assert adapted["source_preprocess_schema"] == "1.0"
    assert adapted["video_id"] == "joint_smoke"
    assert adapted["source_video"] == "data/raw_videos/joint_smoke.mp4"
    assert [f["frame_id"] for f in adapted["frames"]] == [
        "segment_0001_frame_000001",
        "segment_0001_frame_000003",
    ]
    assert adapted["frames"][0]["segment_id"] == "segment_0001"
    assert adapted["frames"][0]["reason"] == "sharpest_in_window"
    assert adapted["frames"][0]["width"] == 384
    assert adapted["frames"][1]["blur_score"] == 0.9
    # Round-trip into depth-prior selection rules.
    selected = selected_frames(adapted, dedupe_timestamps=False)
    assert len(selected) == 2


def test_adapt_preprocess_schema_v2_maps_and_filters() -> None:
    payload = {
        "schema_version": "2.0",
        "video_id": "contract_v2",
        "source": {"path": "data/raw_videos/contract_v2.mp4"},
        "summary": {"keyframe_quality": {"status": "passed"}},
        "frames": [
            {
                "id": "segment_0001_frame_000001",
                "segment_id": "segment_0001",
                "path": "data/frames/contract_v2/selected/segment_0001/a.png",
                "timestamp_sec": 0.0,
                "frame_index": 1,
                "width": 160,
                "height": 120,
                "selected": True,
                "blur_score": 100.0,
                "reject_reasons": [],
            },
            {
                "id": "segment_0001_frame_000002",
                "segment_id": "segment_0001",
                "path": None,
                "timestamp_sec": 0.5,
                "frame_index": 2,
                "width": 160,
                "height": 120,
                "selected": False,
            },
            {
                "id": "segment_0001_frame_000003",
                "segment_id": "segment_0001",
                "path": "data/frames/contract_v2/selected/segment_0001/c.png",
                "timestamp_sec": 1.0,
                "frame_index": 3,
                "width": 160,
                "height": 120,
                "selected": True,
                "sha256": "abcd",
            },
        ],
    }
    adapted = adapt_preprocess_to_frames_manifest(
        payload,
        frames_manifest_path="data/manifests/contract_v2/frames_manifest.json",
        fill_missing_size=False,
    )
    assert adapted["schema_version"] == "1.0"
    assert adapted["source_preprocess_schema"] == "2.0"
    assert adapted["source_video"] == "data/raw_videos/contract_v2.mp4"
    assert [frame["frame_id"] for frame in adapted["frames"]] == [
        "segment_0001_frame_000001",
        "segment_0001_frame_000003",
    ]
    assert adapted["frames"][0]["segment_id"] == "segment_0001"
    assert adapted["frames"][1]["sha256"] == "abcd"
    selected = selected_frames(adapted, dedupe_timestamps=False)
    assert len(selected) == 2


def test_adapt_rejects_failed_coverage_quality_gate() -> None:
    payload = {
        "schema_version": "2.0",
        "video_id": "bad_coverage",
        "summary": {"keyframe_quality": {"status": "failed"}},
        "frames": [
            {
                "id": "segment_0001_frame_000001",
                "path": "data/frames/a.jpg",
                "selected": True,
                "timestamp_sec": 0.0,
                "width": 64,
                "height": 64,
                "segment_id": "segment_0001",
            }
        ],
    }
    with pytest.raises(ValueError, match="keyframe_quality"):
        adapt_preprocess_to_frames_manifest(payload, fill_missing_size=False)


def test_adapt_rejects_unsupported_schema() -> None:
    with pytest.raises(ValueError, match="Unsupported preprocess schema_version"):
        adapt_preprocess_to_frames_manifest(
            {
                "schema_version": "9.9",
                "frames": [
                    {
                        "id": "segment_0001_frame_000001",
                        "path": "data/frames/a.jpg",
                        "selected": True,
                        "width": 64,
                        "height": 64,
                    }
                ],
            },
            fill_missing_size=False,
        )


def test_adapt_rejects_null_timestamp_mixed_with_numeric() -> None:
    payload = {
        "schema_version": "1.0",
        "frames": [
            {
                "id": "segment_0001_frame_000001",
                "path": "data/frames/a.jpg",
                "selected": True,
                "timestamp_sec": 0.0,
                "width": 64,
                "height": 64,
            },
            {
                "id": "segment_0001_frame_000002",
                "path": "data/frames/b.jpg",
                "selected": True,
                "timestamp_sec": None,
                "width": 64,
                "height": 64,
            },
        ],
    }
    with pytest.raises(ValueError, match="timestamp_sec"):
        adapt_preprocess_to_frames_manifest(payload, fill_missing_size=False)


def test_adapt_rejects_selected_without_path() -> None:
    payload = {
        "schema_version": "1.0",
        "frames": [
            {
                "id": "segment_0001_frame_000001",
                "path": None,
                "selected": True,
                "width": 64,
                "height": 64,
            }
        ],
    }
    with pytest.raises(ValueError, match="selected but path"):
        adapt_preprocess_to_frames_manifest(payload, fill_missing_size=False)


def test_adapt_rejects_path_escape() -> None:
    payload = {
        "schema_version": "1.0",
        "frames": [
            {
                "id": "segment_0001_frame_000001",
                "path": "../outside.jpg",
                "selected": True,
                "width": 64,
                "height": 64,
            }
        ],
    }
    with pytest.raises(ValueError, match="escapes"):
        adapt_preprocess_to_frames_manifest(payload, fill_missing_size=False)


def test_adapt_cli_rejects_output_escape(tmp_path: Path) -> None:
    src = ROOT / "data" / "manifests" / "_tmp_adapt_test" / "preprocess_manifest.json"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        '{"schema_version":"1.0","video_id":"tmp_adapt",'
        '"frames":[{"id":"segment_0001_frame_000001",'
        '"path":"data/frames/a.jpg","selected":true,"timestamp_sec":0.0,'
        '"width":64,"height":64}]}',
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
