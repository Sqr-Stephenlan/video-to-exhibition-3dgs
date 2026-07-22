"""Regression coverage for real preprocess product adaptation issues (P1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.depth.adapt_preprocess_manifest import adapt_preprocess_to_frames_manifest
from scripts.depth.config import format_io_template
from scripts.depth.longsplat_consumer_contract import (
    bind_depth_manifest_to_frame_mapping,
    prepared_depth_npy_name,
)
from scripts.depth.manifest import (
    assert_outputs_not_conflicting,
    selected_frames,
)
from scripts.depth.run_depth_prior import frame_durations_sec


def _write_png(path: Path, *, payload: bytes = b"same") -> None:
    # Minimal identical/different files for hash-based dedupe tests (not real PNG decode).
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
    # Tiny valid JPEG via Pillow if available; otherwise skip fill path and set sizes.
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


def test_dedupe_identical_timestamps_keeps_one(tmp_path: Path) -> None:
    root = tmp_path
    a = root / "data" / "frames" / "a.bin"
    b = root / "data" / "frames" / "b.bin"
    _write_png(a, payload=b"identical-bytes")
    _write_png(b, payload=b"identical-bytes")
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
                "path": "data/frames/a.bin",
                "timestamp_sec": 5.0,
                "segment_id": "segment_0002",
                "selected": True,
            },
        ],
    }
    selected = selected_frames(frames_manifest, root=root, dedupe_timestamps=True)
    assert [item["frame_id"] for item in selected] == ["a", "c"]
    assert selected[0]["segment_id"] == "segment_0001"


def test_dedupe_rejects_same_timestamp_different_bytes(tmp_path: Path) -> None:
    root = tmp_path
    _write_png(root / "data" / "frames" / "a.bin", payload=b"aaa")
    _write_png(root / "data" / "frames" / "b.bin", payload=b"bbb")
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
        )


def test_timestamp_driven_durations_preserve_gaps() -> None:
    durations = frame_durations_sec(3, fps=5.0, timestamps_sec=[0.0, 4.0, 5.9])
    assert durations[0] == pytest.approx(4.0)
    assert durations[1] == pytest.approx(1.9)
    assert durations[2] == pytest.approx(0.2)  # last uses 1/target_fps


def test_longsplat_consumer_contract_fail_closed() -> None:
    depth_manifest = {
        "frames": [
            {
                "frame_id": "segment_0001_frame_000001",
                "rgb_path": "data/frames/a.jpg",
                "depth_path": "data/depth/vid/segment_0001_frame_000001.npz",
            }
        ]
    }
    mapping = {
        "frame_000000.jpg": {"frame_id": "segment_0001_frame_000001"},
        "frame_000001.jpg": {"frame_id": "missing_frame"},
    }
    with pytest.raises(ValueError, match="fail closed"):
        bind_depth_manifest_to_frame_mapping(
            depth_manifest=depth_manifest,
            frame_mapping=mapping,
            require_all=True,
        )
    bindings = bind_depth_manifest_to_frame_mapping(
        depth_manifest=depth_manifest,
        frame_mapping={"frame_000000.jpg": {"frame_id": "segment_0001_frame_000001"}},
        require_all=True,
    )
    assert bindings[0]["prepared_depth_npy"] == prepared_depth_npy_name("frame_000000.jpg")
    assert bindings[0]["prepared_depth_npy"] == "frame_000000_depth.npy"
    assert bindings[0]["depth_path"].endswith(".npz")
