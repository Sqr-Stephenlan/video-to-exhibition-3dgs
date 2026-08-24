from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.longsplat.formal_executor import (
    ITERATIONS,
    PROFILE,
    FormalAdapterBlocked,
    plan_formal_training,
    plan_automated_formal_training,
    validate_formal_training_plan,
)
from scripts.longsplat.longsplat_input import future_workload_profile
from scripts.longsplat.convergence_smoke import expected_anchor_adjust_iterations, predicted_exposure
from scripts.longsplat.pipeline_contract import sha256_file


def _release(tmp_path: Path, *, delivery_quality: bool = False, held_out: bool = False) -> tuple[Path, str]:
    path = tmp_path / "supervisor-release.json"
    path.write_text(
        json.dumps(
            {
                "decision_source": "supervisor independent visual review",
                "decision": {
                    "EARLY_VISUAL_RECOGNIZABLE": True,
                    "ROUGH_VISUAL_FOR_FORMAL_GATE": "pass",
                    "DELIVERY_QUALITY": delivery_quality,
                    "held_out": held_out,
                },
                "formal_profile": {"profile_id": PROFILE, "iterations": ITERATIONS},
            }
        ),
        encoding="utf-8",
    )
    return path, sha256_file(path)


def test_formal_policy_is_dynamic_and_exactly_binds_30000_sampling() -> None:
    prediction = predicted_exposure(144, ITERATIONS)
    assert prediction["min_exposure_count"] == 208
    assert prediction["max_exposure_count"] == 209
    assert prediction["camera_count_at_min_exposure"] == 96
    assert prediction["camera_count_at_max_exposure"] == 48
    assert expected_anchor_adjust_iterations(ITERATIONS) == list(range(100, 30000, 100))
    assert future_workload_profile(PROFILE)["iterations"] == ITERATIONS


def test_formal_policy_binds_arbitrary_camera_names_and_validates(tmp_path: Path) -> None:
    source = tmp_path / "immutable-input"
    source.mkdir()
    static = source / "static.json"
    static.write_text(json.dumps({"source_video_sha256": "video-sha", "image_count": 3}), encoding="utf-8")
    (source / "camera_contract-v1.json").write_text(json.dumps({"frame_names": ["odd-start.jpg", "middle_17.png", "tail-final.jpeg"]}), encoding="utf-8")
    release, release_sha = _release(tmp_path)
    names = ["odd-start.jpg", "middle_17.png", "tail-final.jpeg"]
    policy = plan_formal_training(
        source=source,
        static_path=static,
        release_path=release,
        camera_names=names,
        camera={"model": "PINHOLE", "width": 1876, "height": 1068, "fx": 1.0, "fy": 1.0},
        release_sha256=release_sha,
    )
    assert policy["identity_binding"]["camera_order"] == names
    assert policy["identity_binding"]["camera_count"] == len(names)
    assert validate_formal_training_plan(policy)["computed_pass"] is True


@pytest.mark.parametrize("delivery_quality,held_out", [(True, False), (False, True)])
def test_formal_policy_rejects_delivery_or_held_out_release(tmp_path: Path, delivery_quality: bool, held_out: bool) -> None:
    source = tmp_path / "immutable-input"
    source.mkdir()
    static = source / "static.json"
    static.write_text(json.dumps({"source_video_sha256": "video-sha", "image_count": 1}), encoding="utf-8")
    (source / "camera_contract-v1.json").write_text(json.dumps({"frame_names": ["one.png"]}), encoding="utf-8")
    release, release_sha = _release(tmp_path, delivery_quality=delivery_quality, held_out=held_out)
    with pytest.raises(FormalAdapterBlocked, match="delivery quality"):
        plan_formal_training(
            source=source,
            static_path=static,
            release_path=release,
            camera_names=["one.png"],
            camera={"model": "PINHOLE", "width": 8, "height": 6, "fx": 4.0, "fy": 4.0},
            release_sha256=release_sha,
        )


def test_formal_policy_never_auto_releases_or_claims_bit_exact_resume(tmp_path: Path) -> None:
    source = tmp_path / "immutable-input"
    source.mkdir()
    static = source / "static.json"
    static.write_text(json.dumps({"source_video_sha256": "video-sha", "image_count": 1}), encoding="utf-8")
    (source / "camera_contract-v1.json").write_text(json.dumps({"frame_names": ["one.png"]}), encoding="utf-8")
    release, release_sha = _release(tmp_path)
    policy = plan_formal_training(
        source=source,
        static_path=static,
        release_path=release,
        camera_names=["one.png"],
        camera={"model": "PINHOLE", "width": 8, "height": 6, "fx": 4.0, "fy": 4.0},
        release_sha256=release_sha,
    )
    assert policy["formal_auto_release"] is False
    assert policy["training_state_resume"]["bit_exact_resume"] is False
    assert policy["policy_boundary"]["conversion"] == "forbidden in this task"


def test_automated_formal_policy_binds_technical_gate_without_human_claim(tmp_path: Path) -> None:
    source = tmp_path / "immutable-input"
    contract = source / "contract"
    contract.mkdir(parents=True)
    static = contract / "static_contract.json"
    static.write_text(json.dumps({"source_video_sha256": "video-sha", "image_count": 2}), encoding="utf-8")
    camera_contract = source / "camera_contract-v1.json"
    camera_contract.write_text(json.dumps({"frame_names": ["odd-a.jpg", "odd-b.png"]}), encoding="utf-8")
    gate = tmp_path / "automated-early-gate.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": "automated-early-gate-v2",
                "automated_technical_gate": True,
                "computed_pass": True,
                "formal_release_eligible": True,
                "formal_auto_release": True,
                "manual_visual_review": False,
                "held_out": False,
                "training_views_only": True,
            }
        ),
        encoding="utf-8",
    )
    policy = plan_automated_formal_training(
        source=source,
        static_path=static,
        gate_path=gate,
        gate_sha256=sha256_file(gate),
        camera_names=["odd-a.jpg", "odd-b.png"],
        camera={"model": "PINHOLE", "width": 19, "height": 13, "fx": 10.0, "fy": 10.0},
    )
    assert policy["policy_mode"] == "automated_technical_v1"
    assert policy["manual_visual_review"] is False
    assert policy["human_visual_claim"] is False
    assert policy["automated_release"]["sha256"] == sha256_file(gate)
    assert validate_formal_training_plan(policy)["computed_pass"] is True
