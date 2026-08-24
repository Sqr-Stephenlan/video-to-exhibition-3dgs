"""Integration tests for the schema-2 preprocess-to-LongSplat adapter."""

from __future__ import annotations

import hashlib
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from scripts.longsplat.manifest_adapter import AdapterError, adapt_manifest
from scripts.longsplat.validate_input import validate_manifest


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_frame(
    root: Path,
    relative_path: str,
    *,
    width: int,
    height: int,
    value: int,
) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        np.full((height, width, 3), value, dtype=np.uint8),
    )
    assert ok
    payload = encoded.tobytes()
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256(payload)


@pytest.fixture
def schema2_handoff(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a contract-shaped synthetic handoff for focused negative tests.

    The frozen ``schema2_coverage_v1`` fixture is the producer-generated
    compatibility source of truth.
    """
    selected_b = "data/frames/video/selected/seg_01/frame_b.jpg"
    rejected = "data/frames/video/rejected/seg_01/frame_rejected.jpg"
    selected_other = "data/frames/video/selected/seg_02/frame_other.jpg"
    selected_a = "data/frames/video/selected/seg_01/frame_a.jpg"

    frame_specs = {
        selected_b: (800, 600, 16),
        rejected: (800, 600, 32),
        selected_other: (640, 360, 48),
        selected_a: (320, 240, 64),
    }
    frame_hashes = {
        path: _write_frame(
            tmp_path,
            path,
            width=spec[0],
            height=spec[1],
            value=spec[2],
        )
        for path, spec in frame_specs.items()
    }

    def frame(
        frame_id: str,
        segment_id: str,
        path: str,
        timestamp: float,
        frame_index: int,
        *,
        selected: bool,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        return {
            "id": frame_id,
            "segment_id": segment_id,
            "path": path,
            "timestamp_sec": timestamp,
            "frame_index": frame_index,
            "selected": selected,
            "blur_score": 42.0,
            "overexposed_ratio": 0.01,
            "underexposed_ratio": 0.02,
            "duplicate_score": None,
            "reject_reasons": [] if selected else ["blur"],
            "width": width,
            "height": height,
            "sha256": frame_hashes[path],
            "calibrated_blur_score": 42.0,
            "keyframe": {
                "policy": "coverage_v1",
                "component_id": 0 if selected else None,
                "selected_by_policy": selected,
                "bridge": False,
                "reference_frame_id": None,
                "adaptive_blur_threshold": 42.0,
                "quality_score": 1.0,
                "motion": None,
            },
        }

    manifest: dict[str, Any] = {
        "schema_version": "2.0",
        "video_id": "video",
        "source": {
            "path": "data/raw_videos/video.mp4",
            "size_bytes": 100,
            "duration_sec": 5.0,
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "codec": "h264",
            "sha256": "a" * 64,
            "rotation_degrees": 0,
            "is_vfr": False,
        },
        "normalized": {
            "path": "data/segments/video/normalized.mp4",
            "duration_sec": 5.0,
            "fps": 10.0,
            "width": 512,
            "height": 288,
            "codec": "h264",
            "sha256": "b" * 64,
            "rotation_degrees": 0,
            "is_vfr": False,
        },
        "settings": {
            "keyframe_policy": "coverage_v1",
            "selection_max_gap_sec": 2.0,
            "min_motion_inlier_ratio": 0.1,
            "min_motion_grid_coverage": 0.1,
        },
        "segments": [
            {
                "id": "seg_01",
                "path": "data/segments/video/seg_01.mp4",
                "index": 0,
                "start_sec": 0.0,
                "end_sec": 3.0,
                "duration_sec": 3.0,
                "reason": "time",
                "width": 512,
                "height": 288,
                "sha256": "c" * 64,
            },
            {
                "id": "seg_02",
                "path": "data/segments/video/seg_02.mp4",
                "index": 1,
                "start_sec": 3.0,
                "end_sec": 5.0,
                "duration_sec": 2.0,
                "reason": "time",
                "width": 512,
                "height": 288,
                "sha256": "d" * 64,
            },
        ],
        "frames": [
            frame(
                "seg_01_frame_000002",
                "seg_01",
                selected_b,
                1.0,
                10,
                selected=True,
                width=800,
                height=600,
            ),
            frame(
                "seg_01_frame_000003",
                "seg_01",
                rejected,
                1.5,
                11,
                selected=False,
                width=800,
                height=600,
            ),
            frame(
                "seg_01_frame_000001",
                "seg_01",
                selected_a,
                2.0,
                12,
                selected=True,
                width=320,
                height=240,
            ),
            frame(
                "seg_02_frame_000001",
                "seg_02",
                selected_other,
                3.5,
                20,
                selected=True,
                width=640,
                height=360,
            ),
        ],
        "summary": {
            "total_segments": 2,
            "total_frames": 4,
            "selected_frames": 3,
            "rejected_frames": 1,
            "frames_by_segment": {
                "seg_01": {"total": 3, "selected": 2, "rejected": 1},
                "seg_02": {"total": 1, "selected": 1, "rejected": 0},
            },
            "reject_reasons": {"blur": 1},
            "coverage_target_max_selected_gap_sec": 2.0,
            "coverage_by_segment": {},
            "keyframe_quality": {
                "policy": "coverage_v1",
                "status": "passed",
                "selected_frames": 3,
                "bridge_frames": 0,
                "component_count": 1,
                "max_selected_gap_sec": 1.5,
                "min_motion_inlier_ratio": 0.1,
                "min_motion_grid_coverage": 0.1,
                "failed_segments": [],
            },
        },
        "run": {
            "id": "video-abc123-settings456",
            "code": {
                "commit": "75b5204fa313c84c613141b89a0ba4462db599a1",
                "working_tree_dirty": False,
            },
        },
    }
    quality_report = {
        "schema_version": "2.0",
        "video_id": "video",
        "policy": "coverage_v1",
        "status": "passed",
        "segments": [
            {
                "segment_id": "seg_01",
                "total_frames": 3,
                "selected_frames": 2,
                "rejected_frames": 1,
                "component_count": 1,
                "max_selected_gap_sec": 1.0,
                "bridge_frames": 0,
                "failed_reasons": [],
            },
            {
                "segment_id": "seg_02",
                "total_frames": 1,
                "selected_frames": 1,
                "rejected_frames": 0,
                "component_count": 1,
                "max_selected_gap_sec": 1.5,
                "bridge_frames": 0,
                "failed_reasons": [],
            },
        ],
    }
    return manifest, quality_report


def _adapt(
    handoff: tuple[dict[str, Any], dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    manifest, report = handoff
    return adapt_manifest(
        manifest,
        "seg_01",
        repo_root,
        quality_report=report,
    )


def _validate_manifest_dict(manifest: dict[str, Any]) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as file:
        json.dump(manifest, file)
        path = Path(file.name)
    try:
        return validate_manifest(path)
    finally:
        path.unlink()


def test_adapts_frozen_real_producer_fixture() -> None:
    fixture_root = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "longsplat"
        / "schema2_coverage_v1"
    )
    manifest_path = (
        fixture_root
        / "data"
        / "manifests"
        / "coverage_fixture"
        / "frames_manifest.json"
    )
    report_path = manifest_path.with_name("keyframe_quality_report.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    consumer = adapt_manifest(
        manifest,
        "segment_0001",
        fixture_root,
        quality_report=report,
    )

    validated = _validate_manifest_dict(consumer)
    assert [frame["producer_frame_id"] for frame in validated["frames"]] == [
        "segment_0001_frame_000001",
        "segment_0001_frame_000002",
    ]


def test_adapts_schema2_synthetic_handoff_and_passes_consumer_validation(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    consumer = _adapt(schema2_handoff, tmp_path)

    validated = _validate_manifest_dict(consumer)
    assert validated["schema_version"] == 1
    assert validated["segment_id"] == "seg_01"


def test_preserves_selected_producer_order_with_contiguous_consumer_ids(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    consumer = _adapt(schema2_handoff, tmp_path)

    assert [frame["frame_id"] for frame in consumer["frames"]] == [0, 1]
    assert [frame["timestamp"] for frame in consumer["frames"]] == [1.0, 2.0]
    assert [Path(frame["path"]).name for frame in consumer["frames"]] == [
        "frame_b.jpg",
        "frame_a.jpg",
    ]


def test_uses_each_selected_frames_dimensions_hash_and_run_id(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, _ = schema2_handoff
    consumer = _adapt(schema2_handoff, tmp_path)

    assert [(frame["width"], frame["height"]) for frame in consumer["frames"]] == [
        (800, 600),
        (320, 240),
    ]
    assert [frame["sha256"] for frame in consumer["frames"]] == [
        manifest["frames"][0]["sha256"],
        manifest["frames"][2]["sha256"],
    ]
    assert [frame["producer_frame_id"] for frame in consumer["frames"]] == [
        "seg_01_frame_000002",
        "seg_01_frame_000001",
    ]
    assert [frame["producer_frame_index"] for frame in consumer["frames"]] == [
        10,
        12,
    ]
    assert {frame["producer_run_id"] for frame in consumer["frames"]} == {
        "video-abc123-settings456"
    }


def test_rejects_producer_sha_mismatch(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["frames"][0]["sha256"] = "0" * 64

    with pytest.raises(AdapterError, match="sha256"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_rejects_selected_path_that_escapes_repo_root(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["frames"][0]["path"] = "../outside.jpg"

    with pytest.raises(AdapterError, match="outside|escape|relative"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_rejects_schema1(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["schema_version"] = "1.0"

    with pytest.raises(AdapterError, match="schema_version"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_coverage_v1_requires_quality_report(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, _ = schema2_handoff

    with pytest.raises(AdapterError, match="quality report"):
        adapt_manifest(manifest, "seg_01", tmp_path)


def test_rejects_failed_inline_quality_status(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["summary"]["keyframe_quality"]["status"] = "failed"

    with pytest.raises(AdapterError, match="status"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_rejects_quality_report_video_mismatch(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    report["video_id"] = "another_video"

    with pytest.raises(AdapterError, match="video_id"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


@pytest.mark.parametrize(
    "field",
    ["selected_frames", "rejected_frames", "total_frames"],
)
def test_rejects_selected_segment_quality_summary_mismatch(
    field: str,
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    report["segments"][0][field] += 1

    with pytest.raises(AdapterError, match=field):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_legacy_schema2_does_not_require_quality_report(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, _ = deepcopy(schema2_handoff)
    manifest["settings"] = {"keyframe_policy": "legacy"}
    manifest["summary"].pop("keyframe_quality")
    for frame in manifest["frames"]:
        frame.pop("calibrated_blur_score")
        frame.pop("keyframe")

    consumer = adapt_manifest(manifest, "seg_01", tmp_path)

    assert len(consumer["frames"]) == 2


def test_rejects_unknown_segment(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = schema2_handoff

    with pytest.raises(AdapterError, match="not found"):
        adapt_manifest(
            manifest,
            "missing",
            tmp_path,
            quality_report=report,
        )


def test_rejects_segment_without_selected_frames(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, _ = deepcopy(schema2_handoff)
    manifest["settings"] = {"keyframe_policy": "legacy"}
    manifest["summary"].pop("keyframe_quality")
    for frame in manifest["frames"]:
        frame.pop("calibrated_blur_score")
        frame.pop("keyframe")
        if frame["segment_id"] == "seg_01":
            frame["selected"] = False

    with pytest.raises(AdapterError, match="No selected frames"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
        )


@pytest.mark.parametrize("path", ["", "   "])
def test_rejects_selected_empty_path(
    path: str,
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["frames"][0]["path"] = path

    with pytest.raises(AdapterError, match="path"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_rejects_missing_selected_frame_file(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["frames"][0]["path"] = "data/frames/video/missing.jpg"

    with pytest.raises(AdapterError, match="Frame file not found"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


@pytest.mark.parametrize("field", ["width", "height"])
def test_rejects_bad_selected_frame_dimensions(
    field: str,
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["frames"][0][field] = 0

    with pytest.raises(AdapterError, match=field):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_rejects_duplicate_selected_producer_frame_id(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["frames"][2]["id"] = manifest["frames"][0]["id"]

    with pytest.raises(AdapterError, match="producer frame id"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_selected_timestamp(
    timestamp: float,
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["frames"][0]["timestamp_sec"] = timestamp

    with pytest.raises(AdapterError, match="timestamp_sec"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "1.0"),
        ("policy", "legacy"),
        ("status", "failed"),
    ],
)
def test_rejects_invalid_quality_report_header(
    field: str,
    value: str,
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    report[field] = value

    with pytest.raises(AdapterError, match=field):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_rejects_non_target_segment_quality_summary_mismatch(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    report["segments"][1]["selected_frames"] += 1

    with pytest.raises(AdapterError, match="seg_02.*selected_frames"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_frames", -1),
        ("selected_frames", 1.5),
        ("rejected_frames", True),
        ("component_count", -1),
        ("bridge_frames", -1),
        ("max_selected_gap_sec", float("nan")),
        ("failed_reasons", "none"),
    ],
)
def test_rejects_invalid_quality_segment_field(
    field: str,
    value: Any,
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    report["segments"][0][field] = value

    with pytest.raises(AdapterError, match=field):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_passed_report_rejects_segment_failed_reasons(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    report["segments"][1]["failed_reasons"] = ["max_selected_gap_sec"]

    with pytest.raises(AdapterError, match="failed_reasons"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


@pytest.mark.parametrize("payload", [b"", b"not a decodable image"])
def test_rejects_undecodable_selected_frame_even_when_sha_matches(
    payload: bytes,
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    relative_path = manifest["frames"][0]["path"]
    (tmp_path / relative_path).write_bytes(payload)
    manifest["frames"][0]["sha256"] = _sha256(payload)

    with pytest.raises(AdapterError, match="decode"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_rejects_report_max_gap_from_stale_bundle(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    report["segments"][0]["max_selected_gap_sec"] = 0.25

    with pytest.raises(AdapterError, match="max_selected_gap_sec"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_rejects_report_bridge_count_inconsistent_with_frame_records(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    report["segments"][0]["bridge_frames"] = 1

    with pytest.raises(AdapterError, match="bridge_frames"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_rejects_inline_quality_aggregate_mismatch(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["summary"]["keyframe_quality"]["selected_frames"] = 99

    with pytest.raises(AdapterError, match="selected_frames"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )


def test_rejects_non_increasing_producer_timestamps(
    schema2_handoff: tuple[dict[str, Any], dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest, report = deepcopy(schema2_handoff)
    manifest["frames"][2]["timestamp_sec"] = 1.0

    with pytest.raises(AdapterError, match="strictly increasing"):
        adapt_manifest(
            manifest,
            "seg_01",
            tmp_path,
            quality_report=report,
        )
