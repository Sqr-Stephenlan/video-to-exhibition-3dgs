from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import preprocess_keyframes as pk  # noqa: E402
import preprocess_video as pv  # noqa: E402


def policy(**overrides: object) -> pk.KeyframePolicy:
    settings: dict[str, object] = {
        "keyframe_policy": "coverage_v1",
        "blur_threshold": 0.0,
        "overexposed_ratio": 1.0,
        "underexposed_ratio": 1.0,
        "duplicate_hash_threshold": 4,
        "target_fps": 5.0,
        **pk.KEYFRAME_DEFAULT_SETTINGS,
    }
    settings.update(overrides)
    return pk.KeyframePolicy.from_settings(settings)


def test_calibrated_blur_is_stable_across_input_resolution() -> None:
    small = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(small, (40, 40), (280, 200), (255, 255, 255), 3)
    large = cv2.resize(small, (1280, 960), interpolation=cv2.INTER_NEAREST)

    small_score = pk.calibrated_blur_score(small, long_edge=512)
    large_score = pk.calibrated_blur_score(large, long_edge=512)

    assert small_score == pytest.approx(large_score, rel=0.08)


def test_scanned_frame_does_not_keep_full_resolution_bgr() -> None:
    fields = {field.name for field in dataclasses.fields(pk.ScannedFrame)}

    assert "frame" not in fields
    assert "frame_bgr" not in fields
    assert "flow_gray" in fields


def test_motion_estimation_reports_trackability() -> None:
    rng = np.random.default_rng(4)
    reference = (rng.random((240, 320)) * 255).astype(np.uint8)
    candidate = cv2.warpAffine(reference, np.float32([[1, 0, 2], [0, 1, 1]]), (320, 240))

    metrics = pk.estimate_pair_motion(reference, candidate, policy())

    assert metrics.valid is True
    assert metrics.tracked_count >= 64
    assert metrics.inlier_ratio > 0.8
    assert pk.motion_passes_hard_gate(metrics, policy())[0] is True


def test_coverage_selection_is_explicit_and_preserves_manifest_additivity() -> None:
    rng = np.random.default_rng(7)
    image = (rng.random((240, 320)) * 255).astype(np.uint8)
    frames = [
        pk.ScannedFrame(
            id=f"segment_0001_frame_{index:06d}",
            segment_id="segment_0001",
            sample_index=index,
            timestamp_sec=index * 0.2,
            frame_index=index,
            blur_score=100.0,
            calibrated_blur_score=100.0,
            overexposed_ratio=0.0,
            underexposed_ratio=0.0,
            average_hash_bits=np.zeros(64, dtype=bool),
            flow_gray=np.roll(image, index, axis=1),
            width=320,
            height=240,
        )
        for index in range(3)
    ]

    decisions = pk.select_coverage_frames(frames, policy())

    assert decisions[0].selected is True
    assert all(decision.reference_frame_id is not None for decision in decisions[1:])
    assert all(decision.component_id == 0 for decision in decisions if decision.selected)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"keyframe_policy": "pose_solver"}, "keyframe_policy.*must be one of"),
        ({"adaptive_blur_percentile": 101}, "adaptive_blur_percentile"),
        ({"selection_min_gap_sec": 0.4, "selection_max_gap_sec": 0.3}, "gap"),
        ({"min_motion_inlier_ratio": -0.1}, "min_motion_inlier_ratio"),
        ({"min_tracked_points": True}, "min_tracked_points.*integer"),
        ({"max_affine_rotation_deg": float("nan")}, "max_affine_rotation_deg.*finite"),
    ],
)
def test_keyframe_config_rejects_invalid_values(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    config_path = tmp_path / "invalid_keyframes.json"
    config_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(pv.PreprocessError, match=message):
        pv.load_settings_config(config_path)


def test_frame_to_manifest_omits_legacy_keyframe_fields() -> None:
    frame = pv.FrameRecord(
        id="f1",
        segment_id="segment_0001",
        path=None,
        timestamp_sec=0.0,
        frame_index=1,
        selected=False,
        blur_score=1.0,
        overexposed_ratio=0.0,
        underexposed_ratio=0.0,
        duplicate_score=None,
        reject_reasons=["blur"],
    )

    payload = pv.frame_to_manifest(frame)

    assert "calibrated_blur_score" not in payload
    assert "keyframe" not in payload


def test_ab_configs_only_change_keyframe_policy() -> None:
    legacy_path = ROOT / "configs/preprocess/coverage_v1_legacy_control.json"
    candidate_path = ROOT / "configs/preprocess/coverage_v1.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    assert legacy["keyframe_policy"] == "legacy"
    assert candidate["keyframe_policy"] == "coverage_v1"
    assert {**legacy, "keyframe_policy": "coverage_v1"} == candidate
    assert pv.load_settings_config(legacy_path)["keyframe_policy"] == "legacy"
    assert pv.load_settings_config(candidate_path)["keyframe_policy"] == "coverage_v1"
