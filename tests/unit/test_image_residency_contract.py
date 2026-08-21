from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.longsplat import longsplat_input as input_contract
from scripts.longsplat import smoke_executor as executor
from scripts.longsplat.runner import (
    BackendValidationError,
    LongSplatConfig,
    build_convert_command,
    build_train_command,
)


def _order_sha(names: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(names, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _telemetry(names: list[str], width: int, height: int, *, transfers: int = 1) -> dict[str, object]:
    frame_bytes = 3 * width * height * 4
    return {
        "schema_version": "image-residency-telemetry-v1",
        "phase": "training",
        "strategy": "cpu-stream-v1",
        "camera_count": len(names),
        "camera_order": names,
        "camera_order_sha256": _order_sha(names),
        "dimensions": {"channels": 3, "height": height, "width": width},
        "image_dtype": "torch.float32",
        "cpu_resident_image_bytes": len(names) * frame_bytes,
        "gpu_resident_gt_frame_count": 0,
        "gpu_resident_gt_frame_bytes": 0,
        "gpu_resident_gt_frame_count_peak": 1 if transfers else 0,
        "gpu_resident_gt_frame_bytes_peak": frame_bytes if transfers else 0,
        "safe_alias_camera_count": len(names),
        "transfer_count": transfers,
        "transfer_bytes": transfers * frame_bytes,
        "device_errors": 0,
        "theory_is_advisory": True,
    }


@pytest.mark.parametrize(
    "count,width,height",
    [(1, 8, 6), (3, 11, 7), (17, 5, 13)],
)
def test_root_telemetry_validation_binds_dynamic_nwh_and_transfer_contract(
    tmp_path: Path, count: int, width: int, height: int
) -> None:
    names = [f"camera-{index}" for index in range(count)]
    path = tmp_path / "image_residency_training-v1.json"
    path.write_text(json.dumps(_telemetry(names, width, height, transfers=1000)), encoding="utf-8")

    validated = executor._verify_image_residency_evidence(
        path,
        expected_strategy="cpu-stream-v1",
        expected_phase="training",
        expected_names=names,
        expected_width=width,
        expected_height=height,
        expected_transfer_count=1000,
    )

    assert validated["camera_count"] == count
    assert validated["dimensions"]["width"] == width
    assert validated["dimensions"]["height"] == height
    assert validated["transfer_count"] == 1000


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("camera_order", ["camera-1", "camera-0"], "camera order"),
        ("camera_order_sha256", "drift", "camera order SHA"),
        ("dimensions", {"channels": 3, "height": 99, "width": 8}, "dimensions"),
        ("gpu_resident_gt_frame_count_peak", 2, "more than one"),
        ("transfer_count", 999, "transfer count"),
        ("image_dtype", "", "image_dtype"),
    ],
)
def test_root_telemetry_validation_fails_closed_on_schema_identity_or_runtime_drift(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    names = ["camera-0", "camera-1"]
    payload = _telemetry(names, 8, 6, transfers=1000)
    payload[field] = value
    path = tmp_path / "telemetry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(executor.SmokeExecutorBlocked, match=match):
        executor._verify_image_residency_evidence(
            path,
            expected_strategy="cpu-stream-v1",
            expected_phase="training",
            expected_names=names,
            expected_width=8,
            expected_height=6,
            expected_transfer_count=1000,
        )


def test_runner_policy_is_optional_for_legacy_callers_and_explicit_for_canonical_routes(tmp_path: Path) -> None:
    nested = tmp_path / "LongSplat"
    legacy = LongSplatConfig(source_path="/source", model_path="/model")
    canonical = LongSplatConfig(
        source_path="/source",
        model_path="/model",
        iterations=1000,
        convert_iteration=1000,
        image_residency="cpu-stream-v1",
    )

    assert "--image_residency" not in build_train_command(nested, legacy)
    assert "--image_residency" not in build_convert_command(nested, legacy)
    assert build_train_command(nested, canonical)[-2:] == ["--image_residency", "cpu-stream-v1"]
    assert build_convert_command(nested, canonical)[-2:] == ["--image_residency", "cpu-stream-v1"]
    with pytest.raises(BackendValidationError, match="image_residency"):
        build_train_command(nested, LongSplatConfig(source_path="/source", model_path="/model", image_residency="unknown"))


def _plan_fixture(tmp_path: Path, *, residency: object = "current", include_flag: bool = True) -> tuple[dict[str, object], Path, Path]:
    route = tmp_path / "route"
    source = tmp_path / "source"
    source.mkdir(parents=True)
    train = route / input_contract._TRAIN_ENTRYPOINT
    safe_state = route / input_contract._SAFE_STATE_FILE
    train.parent.mkdir(parents=True)
    safe_state.parent.mkdir(parents=True, exist_ok=True)
    safe_state.write_text(
        "random.seed(0)\nnp.random.seed(0)\ntorch.manual_seed(0)\n"
        'torch.cuda.set_device(torch.device("cuda:0"))\n',
        encoding="utf-8",
    )
    train.write_text("safe_state(args.quiet)\ntraining(lp.extract\n", encoding="utf-8")
    model = tmp_path / "model"
    argv = [
        "python",
        str(train),
        "--source_path",
        str(source),
        "--images",
        "images",
        "--mode",
        "custom",
        "--resolution",
        "1",
        "--iterations",
        "100",
        "--external_colmap_pose",
        "--disable_resize",
        "--depth_source",
        "disabled",
        "--loss_2d_correspondence_weight",
        "0",
        "--depth_loss_weight",
        "0",
        "--rotation_lr_init",
        "0",
        "--translation_lr_init",
        "0",
    ]
    if include_flag:
        argv += ["--image_residency", "cpu-stream-v1"]
    argv += ["--model_path", str(model)]
    frozen: dict[str, object] = {
        "workload_profile": "smoke100",
        "iterations": 100,
        "render_iteration": 100,
        "seed": {
            "value": 0,
            "source": input_contract._SAFE_STATE_SOURCE,
            "application_order": "before_GaussianModel_and_Scene_and_training",
            "python_random": True,
            "numpy": True,
            "torch_manual_seed": True,
            "cli_flag": None,
            "scope": "seeded stochastic initialization/camera sampling",
            "bitwise_cuda_determinism": False,
            "limitation": "CUDA/custom rasterizer kernels may remain nondeterministic",
            "nested_backend_code_identity_sha256": "nested",
        },
    }
    if residency != "legacy":
        frozen["image_residency"] = residency if residency != "current" else {
            "schema_version": "image-residency-telemetry-v1",
            "strategy": "cpu-stream-v1",
            "scope": "external_colmap_pose_depth_disabled",
            "camera_order_binding": "camera_contract-v1",
            "theory_is_advisory": True,
            "hard_gate": "schema_identity_strategy_transfer_device_and_residency_only",
        }
    plan: dict[str, object] = {
        "schema_version": "longsplat-future-smoke-plan-v2",
        "workload_profile": "smoke100",
        "source_path": str(source),
        "model_path": str(model),
        "model_path_exists_at_plan_time": False,
        "argv": argv,
        "frozen_contract": frozen,
        "nested_backend_code_identity": {
            "identity_sha256": "nested",
            "train_entrypoint_sha256": "train",
        },
        "train_script_sha256": "train",
    }
    return plan, route, source


def test_plan_validator_allows_legacy_and_validates_current_residency_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        input_contract,
        "_nested_backend_code_identity",
        lambda route, train: {"identity_sha256": "nested", "train_entrypoint_sha256": "train"},
    )
    current, route, source = _plan_fixture(tmp_path / "current")
    input_contract.validate_future_smoke_plan(
        current,
        route_root=route,
        containment_root=tmp_path / "current",
        training_root=source,
    )

    legacy, legacy_route, legacy_source = _plan_fixture(
        tmp_path / "legacy", residency="legacy", include_flag=False
    )
    input_contract.validate_future_smoke_plan(
        legacy,
        route_root=legacy_route,
        containment_root=tmp_path / "legacy",
        training_root=legacy_source,
    )


def test_plan_validator_rejects_partial_or_malformed_residency_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        input_contract,
        "_nested_backend_code_identity",
        lambda route, train: {"identity_sha256": "nested", "train_entrypoint_sha256": "train"},
    )
    partial, route, source = _plan_fixture(tmp_path / "partial", residency="legacy", include_flag=True)
    with pytest.raises(input_contract.LongSplatInputBlocked, match="without a frozen contract"):
        input_contract.validate_future_smoke_plan(
            partial,
            route_root=route,
            containment_root=tmp_path / "partial",
            training_root=source,
        )

    malformed, route, source = _plan_fixture(
        tmp_path / "malformed",
        residency={"schema_version": "wrong"},
        include_flag=True,
    )
    with pytest.raises(input_contract.LongSplatInputBlocked, match="contract mismatch"):
        input_contract.validate_future_smoke_plan(
            malformed,
            route_root=route,
            containment_root=tmp_path / "malformed",
            training_root=source,
        )
