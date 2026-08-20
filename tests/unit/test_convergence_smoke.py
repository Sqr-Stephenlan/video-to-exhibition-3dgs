from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.longsplat.convergence_smoke import (
    ConvergenceSmokeBlocked,
    expected_anchor_adjust_iterations,
    plan_convergence_smoke,
    predicted_exposure,
    validate_anchor_events,
    validate_convergence_smoke_plan,
    validate_sampling_telemetry,
)
from scripts.longsplat.longsplat_input import future_workload_profile
from third_party.LongSplat.utils.camera_sampling_telemetry import CameraSamplingTelemetry


def test_convergence_prediction_is_dynamic_for_144_and_noncontiguous_names() -> None:
    names = [f"selected_{index * 3}.jpg" for index in range(144)]
    plan = plan_convergence_smoke(
        active_camera_count=144,
        camera_names=names,
        source_video_sha256="source-sha",
        camera={"model": "PINHOLE", "width": 1876, "height": 1068},
    )
    prediction = plan["sampling"]["prediction"]
    assert prediction["complete_rounds"] == 6
    assert prediction["partial_round_size"] == 136
    assert prediction["min_exposure_count"] == 6
    assert prediction["max_exposure_count"] == 7
    assert prediction["camera_count_at_min_exposure"] == 8
    assert prediction["camera_count_at_max_exposure"] == 136
    assert prediction["predicted_unique_camera_count"] == 144
    assert plan["densification"]["planned_adjust_anchor_iterations"] == list(range(100, 1000, 100))
    assert validate_convergence_smoke_plan(plan)["computed_pass"] is True


def test_convergence_profile_is_fixed_and_never_searches_higher_iterations() -> None:
    with pytest.raises(ConvergenceSmokeBlocked, match="fixed at exactly 1000"):
        plan_convergence_smoke(active_camera_count=45, camera_names=[f"x_{i}.png" for i in range(45)], requested_iterations=2000)
    plan = plan_convergence_smoke(active_camera_count=45, camera_names=[f"x_{i}.png" for i in range(45)])
    assert plan["policy_boundary"]["automatic_higher_iteration_search"] is False
    assert plan["formal_gate"] == "closed"
    assert plan["training_state_resume"]["bit_exact_resume"] is False


def test_anchor_schedule_and_runtime_counts_are_fork_specific() -> None:
    assert expected_anchor_adjust_iterations(1000) == [100, 200, 300, 400, 500, 600, 700, 800, 900]
    events = []
    for iteration in expected_anchor_adjust_iterations(1000):
        before = 100 + iteration // 100
        events.append(
            {
                "iteration": iteration,
                "before_anchor_count": before,
                "after_anchor_count": before + 3,
                "added_anchor_count": 3,
                "deleted_anchor_count": 0,
                "net_anchor_delta": 3,
            }
        )
    checked = validate_anchor_events(events, expected_anchor_adjust_iterations(1000))
    assert checked["event_count"] == 9
    broken = dict(events[0])
    broken["after_anchor_count"] += 1
    with pytest.raises(ConvergenceSmokeBlocked, match="arithmetic"):
        validate_anchor_events([broken, *events[1:]], expected_anchor_adjust_iterations(1000))


def test_convergence_telemetry_requires_exact_floor_ceil_distribution(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    internal = [f"stem_{index * 3}" for index in range(144)]
    cameras = [SimpleNamespace(image_name=name) for name in internal]
    contract = {
        "image_names": internal,
        "camera_identity": {
            "basename_to_internal_name": {f"frame_{index}.png": name for index, name in enumerate(internal)},
            "contract_file_sha256": "contract-sha",
        },
    }
    telemetry = CameraSamplingTelemetry(model_path=model, contract=contract, cameras=cameras, iterations=1000)
    for iteration in range(1, 1001):
        telemetry.record(iteration, cameras[(iteration - 1) % 144])
    telemetry.finalize(checkpoint_iteration=1000)
    checked = validate_sampling_telemetry(model_path=model, expected_iterations=1000, expected_internal_names=internal, expected_contract_sha256="contract-sha")
    counts = list(checked["exposure_counts"].values())
    assert counts.count(6) == 8
    assert counts.count(7) == 136
    assert checked["unique_camera_count"] == 144


def test_telemetry_observer_does_not_change_python_random_state(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    cameras = [SimpleNamespace(image_name="a"), SimpleNamespace(image_name="b")]
    random.seed(1234)
    before = random.getstate()
    telemetry = CameraSamplingTelemetry(
        model_path=model,
        contract={"image_names": ["a", "b"], "camera_identity": {"basename_to_internal_name": {"a.png": "a", "b.png": "b"}}},
        cameras=cameras,
        iterations=2,
    )
    telemetry.record(1, cameras[0])
    telemetry.record(2, cameras[1])
    telemetry.finalize(checkpoint_iteration=2)
    assert random.getstate() == before


def test_convergence_plan_keeps_visual_and_formal_decisions_separate() -> None:
    plan = plan_convergence_smoke(active_camera_count=3, camera_names=["a.png", "c.png", "z.png"])
    assert plan["computed_pass"] is True
    assert plan["formal_auto_release"] is False
    assert plan["policy_boundary"]["visual_quality"].startswith("diagnostic")


def test_convergence_profile_is_allowlisted_for_executor_argv() -> None:
    profile = future_workload_profile("convergence1000-v1")
    assert profile["iterations"] == 1000
    assert profile["render_iteration"] == 1000
    with pytest.raises(Exception, match="unsupported future workload"):
        future_workload_profile("convergence2000-v1")
