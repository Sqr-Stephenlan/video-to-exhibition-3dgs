"""Integration tests for the preprocess → LongSplat manifest adapter.

Uses fixtures that match the real ``feature/preprocess-video`` schema_version
``"1.0"`` output format.  Verifies the adapter correctly translates producer
manifests into consumer manifests accepted by ``validate_input.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.longsplat.manifest_adapter import AdapterError, adapt_manifest
from scripts.longsplat.validate_input import validate_manifest


# ---------------------------------------------------------------------------
# Real-format preprocess manifest fixture builder
# ---------------------------------------------------------------------------


def _make_producer_manifest(
    *,
    schema_version: str = "1.0",
    segment_id: str = "seg_01",
    frame_count: int = 3,
    include_width_height: bool = True,
    include_rejected: bool = False,
    frame_width: int = 640,
    frame_height: int = 480,
) -> dict:
    """Build a minimal manifest matching the real preprocess producer format.

    The returned manifest MUST be accepted by ``adapt_manifest`` and the
    result MUST pass ``validate_manifest``.
    """
    frames = []
    for i in range(frame_count):
        frames.append(
            {
                "id": f"frame_{i:06d}_t{i:010.3f}",
                "segment_id": segment_id,
                "path": f"frames/test_video/selected/{segment_id}/frame_{i:06d}_t{float(i):010.3f}.jpg",
                "timestamp_sec": float(i),
                "frame_index": i,
                "selected": True,
                "blur_score": 12.0 + i,
                "overexposed_ratio": 0.05,
                "underexposed_ratio": 0.03,
                "duplicate_score": None if i > 0 else None,
                "reject_reasons": [],
            }
        )

    if include_rejected:
        frames.append(
            {
                "id": f"frame_{frame_count:06d}_t{float(frame_count):010.3f}",
                "segment_id": segment_id,
                "path": None,
                "timestamp_sec": float(frame_count),
                "frame_index": frame_count,
                "selected": False,
                "blur_score": 99.0,
                "overexposed_ratio": 0.01,
                "underexposed_ratio": 0.02,
                "duplicate_score": None,
                "reject_reasons": ["blurry"],
            }
        )

    manifest: dict = {
        "schema_version": schema_version,
        "video_id": "test_video",
        "source": {
            "path": "data/raw_videos/test_video.mp4",
            "size_bytes": 1234567,
            "duration_sec": 10.0,
            "fps": 30.0,
            "width": frame_width,
            "height": frame_height,
            "codec": "h264",
        },
        "normalized": {
            "path": "data/segments/test_video/normalized.mp4",
            "duration_sec": 10.0,
            "fps": 10.0,
            "width": frame_width,
            "height": frame_height,
            "codec": "h264",
        },
        "settings": {
            "preset": "longsplat",
            "target_fps": 10.0,
            "max_long_edge": 512,
            "segment_method": "time",
            "segment_length_sec": 30.0,
            "segment_overlap_sec": 10.0,
        },
        "segments": [
            {
                "id": segment_id,
                "path": f"data/segments/test_video/{segment_id}.mp4",
                "index": 0,
                "start_sec": 0.0,
                "end_sec": 10.0,
                "duration_sec": 10.0,
                "reason": "time",
            }
        ],
        "frames": frames,
        "summary": {
            "total_segments": 1,
            "total_frames": frame_count + (1 if include_rejected else 0),
            "selected_frames": frame_count,
            "rejected_frames": 1 if include_rejected else 0,
            "frames_by_segment": {
                segment_id: {
                    "total": frame_count + (1 if include_rejected else 0),
                    "selected": frame_count,
                    "rejected": 1 if include_rejected else 0,
                }
            },
            "reject_reasons": {"blurry": 1} if include_rejected else {},
        },
    }
    return manifest


# ---------------------------------------------------------------------------
# Fixture: real frame files on disk (needed for SHA-256 computation)
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_root_with_frames(tmp_path: Path) -> Path:
    """Create a temporary repo root with real frame files matching the manifest."""
    segment_id = "seg_01"
    frame_dir = tmp_path / "frames" / "test_video" / "selected" / segment_id
    frame_dir.mkdir(parents=True)

    for i in range(3):
        fname = f"frame_{i:06d}_t{float(i):010.3f}.jpg"
        path = frame_dir / fname
        path.write_bytes(b"dummy frame content %d" % i)

    return tmp_path


# ---------------------------------------------------------------------------
# Adapter happy-path tests
# ---------------------------------------------------------------------------


def test_adapt_and_validate_roundtrip(repo_root_with_frames: Path):
    """A real-format preprocess manifest adapts and validates successfully."""
    producer = _make_producer_manifest()
    consumer = adapt_manifest(producer, "seg_01", repo_root_with_frames)

    # Must pass consumer validation
    validated = validate_manifest_dict(consumer)
    assert validated["schema_version"] == 1
    assert validated["segment_id"] == "seg_01"
    assert len(validated["frames"]) == 3

    for f in validated["frames"]:
        assert isinstance(f["frame_id"], int)
        assert isinstance(f["width"], int) and f["width"] == 640
        assert isinstance(f["height"], int) and f["height"] == 480
        assert len(f["sha256"]) == 64


def test_adapter_skips_rejected_frames(repo_root_with_frames: Path):
    """Frames with selected=False are excluded from the consumer manifest."""
    producer = _make_producer_manifest(include_rejected=True, frame_count=3)
    consumer = adapt_manifest(producer, "seg_01", repo_root_with_frames)
    assert len(consumer["frames"]) == 3  # rejected frame excluded


def test_adapter_uses_normalized_dimensions(repo_root_with_frames: Path):
    """Frame dimensions come from normalized metadata, not source."""
    producer = _make_producer_manifest(frame_width=1280, frame_height=720)
    consumer = adapt_manifest(producer, "seg_01", repo_root_with_frames)
    assert consumer["frames"][0]["width"] == 1280
    assert consumer["frames"][0]["height"] == 720


def test_adapter_computes_sha256(repo_root_with_frames: Path):
    """Each frame in the adapted manifest has a real SHA-256."""
    producer = _make_producer_manifest(frame_count=1)
    consumer = adapt_manifest(producer, "seg_01", repo_root_with_frames)
    sha = consumer["frames"][0]["sha256"]
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


def test_adapter_preserves_frame_path(repo_root_with_frames: Path):
    """Frame paths are kept as relative paths from the producer manifest."""
    producer = _make_producer_manifest(frame_count=1)
    consumer = adapt_manifest(producer, "seg_01", repo_root_with_frames)
    path = consumer["frames"][0]["path"]
    assert "frame_000000_t" in path


# ---------------------------------------------------------------------------
# Error / rejection tests
# ---------------------------------------------------------------------------


def test_rejects_unsupported_schema_version(repo_root_with_frames: Path):
    """Only schema_version '1.0' (string) is supported."""
    producer = _make_producer_manifest(schema_version="2.0")
    with pytest.raises(AdapterError, match="schema_version"):
        adapt_manifest(producer, "seg_01", repo_root_with_frames)


def test_rejects_integer_schema_version(repo_root_with_frames: Path):
    """Real producer uses string '1.0', not integer 1."""
    producer = _make_producer_manifest()
    producer["schema_version"] = 1  # type: ignore[assignment]
    with pytest.raises(AdapterError, match="schema_version"):
        adapt_manifest(producer, "seg_01", repo_root_with_frames)


def test_rejects_unknown_segment(repo_root_with_frames: Path):
    producer = _make_producer_manifest()
    with pytest.raises(AdapterError, match="Segment.*not found"):
        adapt_manifest(producer, "nonexistent", repo_root_with_frames)


def test_rejects_missing_normalized_dimensions(repo_root_with_frames: Path):
    producer = _make_producer_manifest()
    producer["normalized"]["width"] = 0
    with pytest.raises(AdapterError, match="normalized.width"):
        adapt_manifest(producer, "seg_01", repo_root_with_frames)


def test_rejects_missing_frame_file(repo_root_with_frames: Path):
    """If a frame path points to a non-existent file, raise an error."""
    producer = _make_producer_manifest(frame_count=1)
    producer["frames"][0]["path"] = "frames/test_video/selected/seg_01/nonexistent.jpg"
    with pytest.raises(AdapterError, match="Frame file not found"):
        adapt_manifest(producer, "seg_01", repo_root_with_frames)


def test_rejects_empty_segment(repo_root_with_frames: Path):
    """A segment with no selected frames should raise."""
    producer = _make_producer_manifest(frame_count=0)
    with pytest.raises(AdapterError, match="No selected frames"):
        adapt_manifest(producer, "seg_01", repo_root_with_frames)


def test_rejects_selected_frame_with_null_path(repo_root_with_frames: Path):
    """A frame marked selected=True but with path=None is a producer error."""
    producer = _make_producer_manifest(frame_count=1)
    producer["frames"][0]["path"] = None
    producer["frames"][0]["selected"] = True
    with pytest.raises(AdapterError, match="has no path"):
        adapt_manifest(producer, "seg_01", repo_root_with_frames)


# ---------------------------------------------------------------------------
# Helper: validate a dict directly (skip file read)
# ---------------------------------------------------------------------------


def validate_manifest_dict(manifest: dict) -> dict:
    """Validate a manifest dict by writing it to a temp file first."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        json.dump(manifest, fh)
        tmp_path = fh.name
    try:
        return validate_manifest(tmp_path)
    finally:
        Path(tmp_path).unlink()
