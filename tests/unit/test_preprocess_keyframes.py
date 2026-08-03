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
    assert metrics.model == "affine"
    assert metrics.forward_backward_rmse_px is not None
    assert metrics.forward_backward_rmse_px <= policy().flow_forward_backward_max_error_px
    assert pk.motion_passes_hard_gate(metrics, policy())[0] is True


def test_auto_motion_model_accepts_rigid_two_depth_parallax() -> None:
    xs, ys = np.meshgrid(
        np.linspace(20, 300, 12, dtype=np.float32),
        np.linspace(20, 220, 9, dtype=np.float32),
    )
    reference = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    # Keep the two affine residual modes strictly outside the existing 3 px
    # RANSAC boundary; 2/8 px lands exactly on that implementation boundary.
    disparity = np.where(reference[:, 0] < 160.0, 2.0, 9.0).astype(np.float32)
    candidate = reference.copy()
    candidate[:, 0] += disparity

    metrics = pk.estimate_motion_from_tracks(
        reference,
        candidate,
        (240, 320),
        policy(motion_model="auto"),
    )

    assert metrics.valid is True
    assert metrics.model == "fundamental"
    assert metrics.inlier_ratio >= 0.90
    assert metrics.grid_coverage >= 0.75
    assert pk.motion_passes_hard_gate(metrics, policy(motion_model="auto"))[0] is True


def test_auto_motion_model_prefers_affine_for_planar_similarity() -> None:
    xs, ys = np.meshgrid(
        np.linspace(20, 300, 12, dtype=np.float32),
        np.linspace(20, 220, 9, dtype=np.float32),
    )
    reference = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    angle = np.deg2rad(3.0)
    linear = 1.02 * np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    candidate = reference @ linear.T + np.array([2.0, 1.0], dtype=np.float32)

    metrics = pk.estimate_motion_from_tracks(
        reference,
        candidate,
        (240, 320),
        policy(motion_model="auto"),
    )

    assert metrics.valid is True
    assert metrics.model == "affine"
    assert pk.motion_passes_hard_gate(metrics, policy(motion_model="auto"))[0] is True


def test_fundamental_model_does_not_accept_random_correspondences() -> None:
    rng = np.random.default_rng(20260729)
    reference = rng.uniform([0.0, 0.0], [319.0, 239.0], size=(160, 2)).astype(np.float32)
    candidate = rng.uniform([0.0, 0.0], [319.0, 239.0], size=(160, 2)).astype(np.float32)

    metrics = pk.estimate_motion_from_tracks(
        reference,
        candidate,
        (240, 320),
        policy(motion_model="auto"),
    )

    assert pk.motion_passes_hard_gate(metrics, policy(motion_model="auto"))[0] is False


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


def test_coverage_selection_does_not_reinitialize_after_rejected_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    def frame(index: int, timestamp: float, blur: float) -> pk.ScannedFrame:
        bits = np.zeros(64, dtype=bool)
        bits[: index * 8] = True
        return pk.ScannedFrame(
            id=f"segment_0001_frame_{index:06d}",
            segment_id="segment_0001",
            sample_index=index,
            timestamp_sec=timestamp,
            frame_index=index,
            blur_score=blur,
            calibrated_blur_score=blur,
            overexposed_ratio=0.0,
            underexposed_ratio=0.0,
            average_hash_bits=bits,
            flow_gray=np.full((8, 8), index, dtype=np.uint8),
            width=8,
            height=8,
        )

    def fake_motion(reference: np.ndarray, candidate: np.ndarray, _policy: pk.KeyframePolicy) -> pk.MotionMetrics:
        reference_index = int(reference[0, 0])
        candidate_index = int(candidate[0, 0])
        if reference_index == 1 and candidate_index == 2:
            return pk.MotionMetrics(True, 100, 100, 1.0, 1.0, 0.01, 0.0, 1.0, 0.001)
        return pk.MotionMetrics(False, 100, 0, 0.0, 0.0, 0.0, None, None, None, "motion_break")

    monkeypatch.setattr(pk, "estimate_pair_motion", fake_motion)
    frames = [
        frame(0, 0.0, 100.0),
        frame(1, 0.2, 1.0),
        frame(2, 0.4, 100.0),
    ]

    decisions = pk.select_coverage_frames(
        frames,
        policy(
            blur_threshold=10.0,
            adaptive_blur_percentile=0.0,
            selection_min_gap_sec=0.1,
            selection_target_gap_sec=0.2,
            selection_max_gap_sec=0.3,
        ),
    )

    assert [decision.selected for decision in decisions] == [True, False, False]
    assert "blur" in decisions[1].reject_reasons
    assert decisions[2].reference_frame_id == frames[0].id
    assert decisions[2].bridge is False


def _coverage_frame(index: int, timestamp: float, blur: float) -> pk.ScannedFrame:
    bits = np.zeros(64, dtype=bool)
    bits[: min(64, index * 8)] = True
    return pk.ScannedFrame(
        id=f"segment_0001_frame_{index:06d}",
        segment_id="segment_0001",
        sample_index=index,
        timestamp_sec=timestamp,
        frame_index=index,
        blur_score=blur,
        calibrated_blur_score=blur,
        overexposed_ratio=0.0,
        underexposed_ratio=0.0,
        average_hash_bits=bits,
        flow_gray=np.full((8, 8), index, dtype=np.uint8),
        width=8,
        height=8,
    )


def _adjacent_motion_only(
    reference: np.ndarray,
    candidate: np.ndarray,
    _policy: pk.KeyframePolicy,
) -> pk.MotionMetrics:
    left = int(reference[0, 0])
    right = int(candidate[0, 0])
    if right - left == 1:
        return pk.MotionMetrics(True, 100, 95, 0.95, 0.75, 0.01, 0.0, 1.0, 0.001)
    return pk.MotionMetrics(False, 100, 0, 0.0, 0.0, 0.0, None, None, None, "non_adjacent")


def test_coverage_repair_promotes_trackable_frames_above_bridge_blur_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        _coverage_frame(0, 0.0, blur=100.0),
        _coverage_frame(1, 0.2, blur=70.0),
        _coverage_frame(2, 0.4, blur=70.0),
        _coverage_frame(3, 0.6, blur=100.0),
    ]
    monkeypatch.setattr(pk, "estimate_pair_motion", _adjacent_motion_only)

    decisions = pk.select_coverage_frames(
        frames,
        policy(
            adaptive_blur_percentile=100.0,
            blur_threshold=40.0,
            bridge_min_blur_ratio=0.65,
            selection_max_gap_sec=0.2,
        ),
    )

    assert [item.selected for item in decisions] == [True, True, True, True]
    assert [item.bridge for item in decisions] == [False, True, True, False]
    assert decisions[1].reference_frame_id == frames[0].id
    assert decisions[2].reference_frame_id == frames[1].id
    assert decisions[1].bridge_reason == "coverage_graph"


def test_coverage_repair_rejects_frame_below_absolute_blur_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        _coverage_frame(0, 0.0, blur=100.0),
        _coverage_frame(1, 0.2, blur=39.0),
        _coverage_frame(2, 0.4, blur=100.0),
    ]
    monkeypatch.setattr(pk, "estimate_pair_motion", _adjacent_motion_only)

    decisions = pk.select_coverage_frames(
        frames,
        policy(
            adaptive_blur_percentile=100.0,
            blur_threshold=40.0,
            bridge_min_blur_ratio=0.65,
            selection_max_gap_sec=0.2,
        ),
    )

    assert decisions[1].selected is False
    assert "blur" in decisions[1].reject_reasons


def test_coverage_repair_does_not_cross_failed_geometry_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        _coverage_frame(0, 0.0, blur=100.0),
        _coverage_frame(1, 0.2, blur=70.0),
        _coverage_frame(2, 0.4, blur=70.0),
        _coverage_frame(3, 0.6, blur=100.0),
    ]

    def motion_with_cut(
        reference: np.ndarray,
        candidate: np.ndarray,
        _policy: pk.KeyframePolicy,
    ) -> pk.MotionMetrics:
        left = int(reference[0, 0])
        right = int(candidate[0, 0])
        if (left, right) == (1, 2):
            return pk.MotionMetrics(False, 100, 0, 0.0, 0.0, 0.0, None, None, None, "scene_cut")
        if right - left == 1:
            return pk.MotionMetrics(True, 100, 95, 0.95, 0.75, 0.01, 0.0, 1.0, 0.001)
        return pk.MotionMetrics(False, 100, 0, 0.0, 0.0, 0.0, None, None, None, "non_adjacent")

    monkeypatch.setattr(pk, "estimate_pair_motion", motion_with_cut)
    decisions = pk.select_coverage_frames(
        frames,
        policy(
            adaptive_blur_percentile=100.0,
            blur_threshold=40.0,
            bridge_min_blur_ratio=0.65,
            selection_max_gap_sec=0.2,
        ),
    )

    assert [item.selected for item in decisions] != [True, True, True, True]


def test_selected_bridge_never_references_rejected_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        _coverage_frame(0, 0.0, blur=100.0),
        _coverage_frame(1, 0.2, blur=70.0),
        _coverage_frame(2, 0.4, blur=70.0),
        _coverage_frame(3, 0.6, blur=100.0),
    ]
    monkeypatch.setattr(pk, "estimate_pair_motion", _adjacent_motion_only)
    decisions = pk.select_coverage_frames(
        frames,
        policy(
            adaptive_blur_percentile=100.0,
            blur_threshold=40.0,
            bridge_min_blur_ratio=0.65,
            selection_max_gap_sec=0.2,
        ),
    )
    selected_ids = {item.id for item, decision in zip(frames, decisions) if decision.selected}

    assert all(
        decision.reference_frame_id in selected_ids
        for decision in decisions
        if decision.selected and decision.reference_frame_id is not None
    )


def test_coverage_repair_covers_segment_end_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [
        _coverage_frame(0, 0.0, blur=100.0),
        _coverage_frame(1, 0.2, blur=100.0),
        _coverage_frame(2, 0.4, blur=70.0),
    ]
    monkeypatch.setattr(pk, "estimate_pair_motion", _adjacent_motion_only)
    decisions = pk.select_coverage_frames(
        frames,
        policy(
            adaptive_blur_percentile=100.0,
            blur_threshold=40.0,
            bridge_min_blur_ratio=0.65,
            selection_max_gap_sec=0.2,
        ),
        segment_start_sec=0.0,
        segment_end_sec=0.6,
    )

    assert decisions[2].selected is True
    assert decisions[2].bridge is True


def test_coverage_repair_start_boundary_requires_geometry_to_first_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = [
        _coverage_frame(0, 0.2, blur=70.0),
        _coverage_frame(1, 0.4, blur=100.0),
    ]

    def failed_motion(
        _reference: np.ndarray,
        _candidate: np.ndarray,
        _policy: pk.KeyframePolicy,
    ) -> pk.MotionMetrics:
        return pk.MotionMetrics(False, 100, 0, 0.0, 0.0, 0.0, None, None, None, "scene_cut")

    monkeypatch.setattr(pk, "estimate_pair_motion", failed_motion)
    decisions = pk.select_coverage_frames(
        frames,
        policy(
            adaptive_blur_percentile=100.0,
            blur_threshold=40.0,
            bridge_min_blur_ratio=0.65,
            selection_max_gap_sec=0.2,
        ),
        segment_start_sec=0.0,
        segment_end_sec=0.4,
    )

    assert decisions[0].selected is False
    assert decisions[1].selected is True


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"keyframe_policy": "pose_solver"}, "keyframe_policy.*must be one of"),
        ({"adaptive_blur_percentile": 101}, "adaptive_blur_percentile"),
        ({"selection_min_gap_sec": 0.4, "selection_max_gap_sec": 0.3}, "gap"),
        ({"min_motion_inlier_ratio": -0.1}, "min_motion_inlier_ratio"),
        ({"min_tracked_points": True}, "min_tracked_points.*integer"),
        ({"max_affine_rotation_deg": float("nan")}, "max_affine_rotation_deg.*finite"),
        ({"motion_model": "fundamental"}, "motion_model.*must be one of"),
        ({"bridge_min_blur_ratio": 0.0}, "bridge_min_blur_ratio"),
        ({"max_bridge_fraction": 1.1}, "max_bridge_fraction"),
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
