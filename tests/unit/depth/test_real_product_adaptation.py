"""Regression coverage for real preprocess product adaptation issues (P1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.depth.adapt_preprocess_manifest import adapt_preprocess_to_frames_manifest
from scripts.depth.config import format_io_template
from scripts.depth.manifest import (
    assert_outputs_not_conflicting,
    prepare_vda_batches,
    selected_frames,
)


def _write_bytes(path: Path, *, payload: bytes = b"same") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_shared_frame_id_namespaces_do_not_collide_across_video_ids() -> None:
    """Same frame_id is safe when video_id + run_id isolate output paths."""
    shared_id = "segment_0001_frame_000001"
    baseline = format_io_template(
        "data/depth/{video_id}/{run_id}",
        video_id="capture",
        run_id="baseline",
    )
    longsplat = format_io_template(
        "data/depth/{video_id}/{run_id}",
        video_id="capture",
        run_id="longsplat",
    )
    assert baseline != longsplat
    assert f"{baseline}/{shared_id}.npz" != f"{longsplat}/{shared_id}.npz"


def test_assert_outputs_refuse_overwrite(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir()
    (depth_dir / "frame_a.npz").write_bytes(b"x")
    manifest = tmp_path / "depth_manifest.json"
    record = tmp_path / "run_record.json"
    frames = [{"frame_id": "frame_a", "path": "data/frames/a.png"}]
    with pytest.raises(FileExistsError, match="overwrite"):
        assert_outputs_not_conflicting(
            root=tmp_path,
            depth_dir=depth_dir,
            depth_manifest_path=manifest,
            run_record_path=record,
            frames=frames,
            overwrite=False,
        )
    assert_outputs_not_conflicting(
        root=tmp_path,
        depth_dir=depth_dir,
        depth_manifest_path=manifest,
        run_record_path=record,
        frames=frames,
        overwrite=True,
    )


def test_adapt_preserves_provenance_and_fills_size(tmp_path: Path) -> None:
    root = tmp_path
    img = root / "data" / "frames" / "a.jpg"
    img.parent.mkdir(parents=True)
    pytest.importorskip("PIL")
    from PIL import Image

    Image.new("RGB", (720, 960), color=(10, 20, 30)).save(img, format="JPEG")

    payload = {
        "schema_version": "1.0",
        "video_id": "real_capture_baseline",
        "source": {"path": "data/raw_videos/demo.mp4"},
        "frames": [
            {
                "id": "segment_0001_frame_000001",
                "path": "data/frames/a.jpg",
                "selected": True,
                "timestamp_sec": 0.0,
                "segment_id": "segment_0001",
                "blur_score": 0.42,
                "overexposed_ratio": 0.01,
                "reject_reasons": [],
            }
        ],
    }
    adapted = adapt_preprocess_to_frames_manifest(payload, root=root)
    frame = adapted["frames"][0]
    assert frame["width"] == 720
    assert frame["height"] == 960
    assert frame["segment_id"] == "segment_0001"
    assert frame["blur_score"] == 0.42
    assert frame["reason"] == "selected"
    assert adapted["video_id"] == "real_capture_baseline"


def test_per_segment_keeps_same_timestamp_across_segments(tmp_path: Path) -> None:
    """Cross-segment same timestamp is allowed; each segment is its own VDA batch."""
    root = tmp_path
    _write_bytes(root / "data" / "frames" / "a.bin", payload=b"aaa")
    _write_bytes(root / "data" / "frames" / "b.bin", payload=b"bbb")
    _write_bytes(root / "data" / "frames" / "c.bin", payload=b"ccc")
    frames_manifest = {
        "schema_version": "1.0",
        "frames": [
            {
                "frame_id": "a",
                "path": "data/frames/a.bin",
                "timestamp_sec": 1.0,
                "segment_id": "segment_0001",
                "selected": True,
            },
            {
                "frame_id": "b",
                "path": "data/frames/b.bin",
                "timestamp_sec": 1.0,
                "segment_id": "segment_0002",
                "selected": True,
            },
            {
                "frame_id": "c",
                "path": "data/frames/c.bin",
                "timestamp_sec": 5.0,
                "segment_id": "segment_0002",
                "selected": True,
            },
        ],
    }
    selected = selected_frames(
        frames_manifest,
        root=root,
        dedupe_timestamps=True,
        infer_per_segment=True,
    )
    assert [item["frame_id"] for item in selected] == ["a", "b", "c"]
    batches = prepare_vda_batches(selected, infer_per_segment=True)
    assert [key for key, _ in batches] == ["segment_0001", "segment_0002"]
    assert [frame["frame_id"] for frame in batches[0][1]] == ["a"]
    assert [frame["frame_id"] for frame in batches[1][1]] == ["b", "c"]


def test_dedupe_within_segment_keeps_one_identical(tmp_path: Path) -> None:
    root = tmp_path
    _write_bytes(root / "data" / "frames" / "a.bin", payload=b"identical")
    _write_bytes(root / "data" / "frames" / "b.bin", payload=b"identical")
    selected = selected_frames(
        {
            "schema_version": "1.0",
            "frames": [
                {
                    "frame_id": "a",
                    "path": "data/frames/a.bin",
                    "timestamp_sec": 1.0,
                    "segment_id": "segment_0001",
                    "selected": True,
                },
                {
                    "frame_id": "b",
                    "path": "data/frames/b.bin",
                    "timestamp_sec": 1.0,
                    "segment_id": "segment_0001",
                    "selected": True,
                },
            ],
        },
        root=root,
        dedupe_timestamps=True,
        infer_per_segment=True,
    )
    assert [item["frame_id"] for item in selected] == ["a"]


def test_dedupe_rejects_same_timestamp_different_bytes_within_batch(tmp_path: Path) -> None:
    root = tmp_path
    _write_bytes(root / "data" / "frames" / "a.bin", payload=b"aaa")
    _write_bytes(root / "data" / "frames" / "b.bin", payload=b"bbb")
    with pytest.raises(ValueError, match="Duplicate timestamp_sec"):
        selected_frames(
            {
                "schema_version": "1.0",
                "frames": [
                    {
                        "frame_id": "a",
                        "path": "data/frames/a.bin",
                        "timestamp_sec": 1.0,
                        "selected": True,
                    },
                    {
                        "frame_id": "b",
                        "path": "data/frames/b.bin",
                        "timestamp_sec": 1.0,
                        "selected": True,
                    },
                ],
            },
            root=root,
            dedupe_timestamps=True,
            infer_per_segment=True,
        )
