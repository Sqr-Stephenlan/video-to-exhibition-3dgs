from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.longsplat.coverage_smoke import (
    formal_release_decision,
    plan_coverage_smoke,
    validate_coverage_smoke_plan,
    validate_sampling_telemetry,
)
from scripts.longsplat.coverage_executor import CoverageAdapterBlocked, _coverage_binding, _coverage_training_policy
from scripts.longsplat.reconstruct_pipeline import _render_quality
from scripts.longsplat.smoke_executor import validate_plan_and_static
from scripts.longsplat.longsplat_input import (
    LongSplatInputBlocked,
    build_future_smoke_plan,
    resolve_future_workload_profile,
    validate_future_smoke_plan,
)
from third_party.LongSplat.utils.camera_sampling_telemetry import CameraSamplingTelemetry


def test_coverage_plan_uses_dynamic_n_and_real_anchor_schedule() -> None:
    names = ["wide-angle.jpg", "camera-A.png", "non_contiguous_17.tif"] + [f"view_{i}.png" for i in range(42)]
    plan = plan_coverage_smoke(active_camera_count=45, camera_names=names, frame_selector_max_frames=240)
    assert plan["computed_pass"] is True
    assert plan["requested_iterations"] == 90
    assert plan["coverage_prediction"]["complete_rounds"] == 2
    assert plan["coverage_prediction"]["min_exposure_count"] == 2
    assert plan["coverage_prediction"]["max_exposure_count"] == 2
    assert plan["densification"]["adjust_anchor_iterations"] == []
    assert plan["gpu_invoked"] is False
    assert validate_coverage_smoke_plan(plan)["computed_pass"] is True


@pytest.mark.parametrize("camera_count", [2, 70, 144])
def test_coverage_plan_supports_dynamic_camera_counts(camera_count: int) -> None:
    names = [f"non_contiguous_{index * 3}.png" for index in range(camera_count)]
    plan = plan_coverage_smoke(active_camera_count=camera_count, camera_names=names)
    assert plan["computed_pass"] is True
    assert plan["requested_iterations"] == camera_count * 2
    assert plan["coverage_prediction"]["predicted_unique_camera_count"] == camera_count
    assert plan["coverage_prediction"]["predicted_zero_exposure_camera_count"] == 0


def test_coverage_plan_for_144_cameras_is_not_100_iterations() -> None:
    names = [f"selected_{index * 3}.png" for index in range(144)]
    smoke100 = plan_coverage_smoke(active_camera_count=144, camera_names=names, requested_iterations=100)
    assert smoke100["coverage_prediction"]["predicted_zero_exposure_camera_count"] == 44
    assert smoke100["computed_pass"] is False
    plan = plan_coverage_smoke(active_camera_count=144, camera_names=names)
    assert plan["requested_iterations"] == 288
    assert plan["coverage_prediction"]["predicted_zero_exposure_camera_count"] == 0
    assert plan["coverage_prediction"]["complete_rounds"] == 2
    assert plan["densification"]["adjust_anchor_iterations"] == [100, 200]
    assert plan["formula"]["official_formula_status"] == "not_specified"
    assert plan["policy_status"] == "local_policy"


def test_coverage_telemetry_exact_two_exposures_for_144_cameras(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    internal = [f"camera_{index:03d}" for index in range(144)]
    contract = {
        "image_names": internal,
        "camera_identity": {
            "basename_to_internal_name": {f"selected_{index * 3}.png": name for index, name in enumerate(internal)},
        },
    }
    cameras = [SimpleNamespace(image_name=name) for name in internal]
    telemetry = CameraSamplingTelemetry(model_path=model, contract=contract, cameras=cameras, iterations=288)
    for iteration in range(1, 289):
        telemetry.record(iteration, cameras[(iteration - 1) % 144])
    telemetry.finalize(checkpoint_iteration=288)
    checked = validate_sampling_telemetry(
        model_path=model,
        expected_iterations=288,
        expected_internal_names=internal,
    )
    assert checked["active_camera_count"] == 144
    assert checked["unique_camera_count"] == 144
    assert set(checked["exposure_counts"].values()) == {2}


def test_coverage_adapter_binds_file_and_stable_camera_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    camera_contract = source / "camera_contract-v1.json"
    camera_contract.write_text(
        json.dumps({"frame_names": ["odd_17.jpg", "view-A.png"], "frame_count": 2}),
        encoding="utf-8",
    )
    import hashlib

    camera_sha = hashlib.sha256(camera_contract.read_bytes()).hexdigest()
    coverage = {
        "active_camera_order": ["odd_17.jpg", "view-A.png"],
        "active_camera_count": 2,
        "camera_contract_sha256": camera_sha,
        "requested_iterations": 4,
        "source_video_sha256": "source-sha",
        "densification": {"adjust_anchor_iterations": []},
    }
    static = {
        "image_count": 2,
        "camera_contract_sha256": "stable-sha",
        "parent_binding": {"camera_contract_file_sha256": camera_sha},
        "source_video_sha256": "source-sha",
    }
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    binding = _coverage_binding(
        coverage_path=coverage_path,
        coverage_plan=coverage,
        static=static,
        source=source,
    )
    assert binding["camera_contract_sha256"] == camera_sha
    assert binding["planned_anchor_adjust_iterations"] == []


def test_coverage_plan_structured_stop_respects_selector_cap_and_short_workload() -> None:
    stopped = plan_coverage_smoke(active_camera_count=144, frame_selector_max_frames=100)
    assert stopped["computed_pass"] is False
    assert "active_camera_count_exceeds_frame_selector_max_frames" in stopped["stop"]["reasons"]
    short = plan_coverage_smoke(active_camera_count=45, requested_iterations=100)
    assert short["computed_pass"] is True
    over = plan_coverage_smoke(active_camera_count=45, requested_iterations=500)
    assert over["computed_pass"] is False
    assert "requested_iterations_exceeds_selector_derived_upper_bound" in over["stop"]["reasons"]


@pytest.mark.parametrize("camera_count", [2, 45, 70, 144])
def test_coverage_training_adapter_derives_iterations_from_dynamic_plan(
    tmp_path: Path, camera_count: int
) -> None:
    route = tmp_path / "route"
    selection_path = route / "outputs" / f"selection-{camera_count}.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(
        json.dumps(
            {
                "config": {"max_frames": 200},
                "frames": [{"selected": True} for _ in range(camera_count)],
                "duration_sec": 1.0,
            }
        ),
        encoding="utf-8",
    )
    import hashlib

    plan = plan_coverage_smoke(
        active_camera_count=camera_count,
        camera_names=[f"camera_{index * 3}.png" for index in range(camera_count)],
        frame_selector_max_frames=200,
        frame_selection_path=selection_path,
    )
    plan["frame_selection_path"] = str(selection_path)
    plan["frame_selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    workload = _coverage_training_policy(coverage_plan=plan, route=route)
    assert workload["iterations"] == camera_count * 2
    assert workload["render_iteration"] == camera_count * 2
    assert workload["active_camera_count"] == camera_count


def test_coverage_training_adapter_rejects_selector_bound_and_tampered_iterations(tmp_path: Path) -> None:
    route = tmp_path / "route"
    selection_path = route / "outputs" / "selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(
        json.dumps(
            {
                "config": {"max_frames": 60},
                "frames": [{"selected": True} for _ in range(60)],
                "duration_sec": 1.0,
            }
        ),
        encoding="utf-8",
    )
    import hashlib

    stopped = plan_coverage_smoke(
        active_camera_count=70,
        camera_names=[f"camera_{index}.png" for index in range(70)],
        frame_selector_max_frames=60,
        frame_selection_path=selection_path,
    )
    stopped["frame_selection_path"] = str(selection_path)
    stopped["frame_selection_sha256"] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    with pytest.raises(CoverageAdapterBlocked, match="dynamic policy"):
        _coverage_training_policy(coverage_plan=stopped, route=route)

    valid_selection_path = route / "outputs" / "selection-valid.json"
    valid_selection_path.write_text(
        json.dumps(
            {
                "config": {"max_frames": 100},
                "frames": [{"selected": True} for _ in range(100)],
                "duration_sec": 1.0,
            }
        ),
        encoding="utf-8",
    )
    valid = plan_coverage_smoke(
        active_camera_count=70,
        camera_names=[f"camera_{index}.png" for index in range(70)],
        frame_selector_max_frames=100,
        frame_selection_path=valid_selection_path,
    )
    valid["frame_selector_max_frames"] = 100
    valid["frame_selection_path"] = str(valid_selection_path)
    valid["frame_selection_sha256"] = hashlib.sha256(valid_selection_path.read_bytes()).hexdigest()
    valid["requested_iterations"] = 288
    with pytest.raises(CoverageAdapterBlocked, match="dynamic policy"):
        _coverage_training_policy(coverage_plan=valid, route=route)


def test_coverage_profile_resolves_from_bound_plan_not_fixed_288() -> None:
    plan = {
        "workload_profile": "coverage-smoke-v1",
        "coverage_plan": {"iterations": 140, "active_camera_count": 70},
    }
    spec = resolve_future_workload_profile(plan)
    assert spec["iterations"] == 140
    assert spec["render_iteration"] == 140
    assert spec["active_camera_count"] == 70
    with pytest.raises(LongSplatInputBlocked, match="immutable coverage plan binding"):
        resolve_future_workload_profile({"workload_profile": "coverage-smoke-v1"})


def test_telemetry_is_observational_and_identity_bound(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    internal = ["odd_stem_7", "another-camera", "last"]
    contract = {
        "image_names": internal,
        "camera_identity": {
            "basename_to_internal_name": {
                "odd.stem.7.jpg": internal[0],
                "another-camera.png": internal[1],
                "last.tif": internal[2],
            },
            "contract_file_sha256": "contract-sha",
        },
    }
    cameras = [SimpleNamespace(image_name=name) for name in internal]
    random.seed(19)
    state_before = random.getstate()
    telemetry = CameraSamplingTelemetry(model_path=model, contract=contract, cameras=cameras, iterations=5)
    for iteration, camera in enumerate((cameras[2], cameras[0], cameras[1], cameras[2], cameras[0]), 1):
        telemetry.record(iteration, camera)
    summary = telemetry.finalize(checkpoint_iteration=5)
    assert random.getstate() == state_before
    assert summary["unique_camera_count"] == 3
    checked = validate_sampling_telemetry(
        model_path=model,
        expected_iterations=5,
        expected_internal_names=internal,
        expected_contract_sha256="contract-sha",
    )
    assert checked["exposure_counts"]["odd_stem_7"] == 2
    assert checked["exposure_counts"]["another-camera"] == 1
    assert checked["exposure_counts"]["last"] == 2


def test_telemetry_validator_rejects_identity_or_event_drift(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    internal = ["a", "b"]
    contract = {
        "image_names": internal,
        "camera_identity": {"basename_to_internal_name": {"a.png": "a", "b.png": "b"}},
    }
    cameras = [SimpleNamespace(image_name=name) for name in internal]
    telemetry = CameraSamplingTelemetry(model_path=model, contract=contract, cameras=cameras, iterations=2)
    telemetry.record(1, cameras[0])
    telemetry.record(2, cameras[1])
    telemetry.finalize(checkpoint_iteration=2)
    events = model / "camera_sampling_telemetry-v1.jsonl"
    lines = events.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace('"camera_internal_name":"b"', '"camera_internal_name":"a"')
    events.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="event file identity"):
        validate_sampling_telemetry(model_path=model, expected_iterations=2, expected_internal_names=internal)


def test_smoke100_quality_cannot_auto_release_formal(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    render_root = tmp_path / "render"
    (render_root / "renders").mkdir(parents=True)
    for index in range(3):
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        image[:, :, index] = np.arange(80, dtype=np.uint8).reshape(8, 10)
        assert cv2.imwrite(str(render_root / "renders" / f"view_{index}.png"), image)
    quality = _render_quality(
        {
            "workload_profile": "smoke100-v1",
            "structural": {"structural_pass": True, "render_root": str(render_root)},
        }
    )
    assert quality["quality_status"] == "needs_review"
    assert quality["computed_pass"] is False
    assert quality["formal_auto_release"] is False
    gate = formal_release_decision(smoke_structural_pass=True, coverage_visual_decision="not_run")
    assert gate["formal_release_eligible"] is False


@pytest.mark.parametrize(
    ("kind", "advisory"),
    [("identical", "all_fixed_renders_identical"), ("uniform", "all_fixed_renders_uniform")],
)
def test_default_render_quality_keeps_finite_visual_heuristics_advisory(
    tmp_path: Path, kind: str, advisory: str
) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    render_root = tmp_path / kind
    (render_root / "renders").mkdir(parents=True)
    if kind == "identical":
        images = [
            np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3),
            np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3),
        ]
    else:
        images = [
            np.full((8, 10, 3), value, dtype=np.uint8)
            for value in (32, 64)
        ]
    for index, image in enumerate(images):
        assert cv2.imwrite(str(render_root / "renders" / f"view_{index}.png"), image)

    quality = _render_quality(
        {
            "workload_profile": "formal30000-v1",
            "structural": {"structural_pass": True, "render_root": str(render_root)},
        }
    )

    assert quality["quality_status"] == "needs_review"
    assert quality["computed_pass"] is False
    assert quality["visual_quality_advisory"] == advisory


def test_legacy_plan_without_new_telemetry_contract_remains_valid(tmp_path: Path) -> None:
    route = Path(__file__).resolve().parents[2]
    (tmp_path / "training").mkdir()
    plan = build_future_smoke_plan(
        training_root=tmp_path / "training",
        backend_python=__import__("sys").executable,
        route_root=route,
        containment_root=tmp_path,
        code_identity={"code_identity_sha256": "legacy-code"},
        backend_identity=None,
    )
    legacy = json.loads(json.dumps(plan))
    legacy["frozen_contract"].pop("camera_sampling_telemetry")
    validate_future_smoke_plan(
        legacy,
        route_root=route,
        containment_root=tmp_path,
        training_root=tmp_path / "training",
    )
