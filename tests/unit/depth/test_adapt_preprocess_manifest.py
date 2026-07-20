from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.depth.adapt_preprocess_manifest import adapt_preprocess_to_frames_manifest


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
                "path": None,
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
    assert all("frame_id" in f and "path" in f for f in adapted["frames"])
