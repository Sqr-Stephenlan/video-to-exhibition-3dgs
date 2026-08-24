from __future__ import annotations

import json
import copy
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.longsplat.pipeline_contract import RunLedger
from scripts.longsplat.reconstruct_pipeline import _coverage_render_postcheck_stage
from scripts.longsplat.render_postcheck import (
    _fit_with_metadata,
    adapt_existing_postcheck_return,
    run_postcheck,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _fixture(tmp_path: Path, *, count: int, width: int, height: int) -> dict[str, object]:
    run_root = tmp_path / "outputs" / "dynamic-render"
    training_input = run_root / "raw-input" / "longsplat_input"
    _write_json(training_input / "contract" / "static_contract.json", {"camera": {"width": width, "height": height}})
    camera_names = [f"arbitrary-camera-{index * 3}.jpg" for index in range(count)]
    _write_json(training_input / "camera_contract-v1.json", {"frame_names": camera_names})

    model = run_root / "model"
    render_root = model / "train" / "ours_1"
    for directory in (render_root / "renders", render_root / "gt", render_root / "nvs"):
        directory.mkdir(parents=True, exist_ok=True)
    render_paths: list[Path] = []
    for index in range(count):
        image = np.full((height, width, 3), (index * 17 % 251, 40, 90), dtype=np.uint8)
        image[:, index % width, :] = 255
        render_path = render_root / "renders" / f"{index:04d}.png"
        gt_path = render_root / "gt" / f"{index:04d}.png"
        assert cv2.imwrite(str(render_path), image)
        assert cv2.imwrite(str(gt_path), np.clip(image.astype(np.int16) + 2, 0, 255).astype(np.uint8))
        render_paths.append(render_path)

    poses: list[dict[str, object]] = []
    for index in range(count):
        poses.append({"index": len(poses), "source_left_index": index, "source_right_index": index, "source_alpha": 0.0})
        if index + 1 < count:
            poses.append({"index": len(poses), "source_left_index": index, "source_right_index": index + 1, "source_alpha": 0.5})
    for pose in poses:
        index = int(pose["index"])
        left = int(pose["source_left_index"])
        right = int(pose["source_right_index"])
        if pose["source_alpha"] == 0.0:
            image = cv2.imread(str(render_paths[left]), cv2.IMREAD_UNCHANGED)
        else:
            left_image = cv2.imread(str(render_paths[left]), cv2.IMREAD_UNCHANGED)
            right_image = cv2.imread(str(render_paths[right]), cv2.IMREAD_UNCHANGED)
            image = ((left_image.astype(np.uint16) + right_image.astype(np.uint16)) // 2).astype(np.uint8)
        assert cv2.imwrite(str(render_root / "nvs" / f"{index:04d}.png"), image)
    _write_json(render_root / "videos" / "nvs_camera_poses.json", {"pose_mode": "adjacent_midpoint_slerp", "input_count": count, "nvs_count": len(poses), "poses": poses})

    render_attempt = run_root / "stages" / "coverage-smoke-render" / "attempt-0001"
    training_attempt = run_root / "stages" / "coverage-smoke-training" / "attempt-0001"
    _write_json(
        render_attempt / "executor" / "result.json",
        {"exit_code": 0, "structural_pass": True, "model_path": str(model), "render_iteration": 1},
    )
    _write_json(render_attempt / "executor" / "request.json", {"workload_profile": "coverage-smoke-v1", "plan_path": str(run_root / "plan.json")})
    _write_json(
        training_attempt / "executor" / "result.json",
        {"exit_code": 0, "structural_pass": True, "structural": {"checkpoint": {}, "mlp_files": {}}},
    )
    frame_selection = run_root / "frame_selection-v1.json"
    _write_json(frame_selection, {"frames": [{"frame_id": Path(name).stem, "timestamp_sec": float(index)} for index, name in enumerate(camera_names)]})
    segment = run_root / "colmap-result.json"
    _write_json(segment, {"result": {"segment_provenance": {"candidate_segments": [{"frame_names": camera_names, "start_time_sec": 0.0, "end_time_sec": float(count)}], "excluded_time_intervals": []}}})
    return {
        "run_root": run_root,
        "training_input": training_input,
        "render_attempt": render_attempt,
        "training_attempt": training_attempt,
        "frame_selection": frame_selection,
        "segment": segment,
        "render_paths": render_paths,
        "model": model,
    }


@pytest.mark.parametrize("count,width,height", [(2, 8, 6), (45, 7, 5), (70, 6, 4), (144, 5, 3)])
def test_render_postcheck_is_dynamic_and_streaming_contract_is_recorded(tmp_path: Path, count: int, width: int, height: int) -> None:
    fixture = _fixture(tmp_path, count=count, width=width, height=height)
    output = fixture["run_root"] / "postcheck-001"
    result = run_postcheck(
        run_root=fixture["run_root"],
        render_attempt=fixture["render_attempt"],
        training_attempt=fixture["training_attempt"],
        training_input=fixture["training_input"],
        frame_selection=fixture["frame_selection"],
        segment_provenance=fixture["segment"],
        output_dir=output,
    )
    postcheck = json.loads(Path(result["postcheck"]).read_text(encoding="utf-8"))
    assert result["computed_pass"] is True
    assert postcheck["counts"] == {"fixed_render": count, "training_gt": count, "nvs_on_path": 2 * count - 1}
    assert postcheck["camera_contract"]["count"] == count
    assert postcheck["camera_contract"]["dimensions"] == {"width": width, "height": height}
    # Legacy fixtures have no runtime residency telemetry; the advisory peak
    # stays absent instead of being replaced by a theoretical estimate.
    assert postcheck["resident_full_resolution_frame_max"] is None
    assert postcheck["gpu_invoked"] is False
    assert postcheck["held_out"] is False
    assert Path(result["fixed_sheet"]).is_file()
    assert Path(result["on_path_sheet"]).is_file()
    assert Path(result["metrics_path"]).is_file()
    assert Path(result["png_hashes"]).is_file()
    layout = postcheck["contact_sheet_layout"]
    assert layout["policy"] == "aspect-preserving-letterbox-v1"
    assert layout["source_dimensions"] == {"width": width, "height": height}
    assert layout["records"]
    assert all(record["letterbox"] is True for record in layout["records"])
    assert result["contact_sheets"] == postcheck["contact_sheets"]
    assert result["metrics_artifact"] == postcheck["metrics"]
    assert result["png_hashes_artifact"] == postcheck["png_hashes"]


def test_historical_postcheck_adapter_is_exact_and_rejects_contract_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, count=2, width=8, height=6)
    result = run_postcheck(
        run_root=fixture["run_root"],
        render_attempt=fixture["render_attempt"],
        training_attempt=fixture["training_attempt"],
        training_input=fixture["training_input"],
        frame_selection=fixture["frame_selection"],
        segment_provenance=fixture["segment"],
        output_dir=fixture["run_root"] / "postcheck-001",
    )
    postcheck_path = Path(result["postcheck"])
    postcheck = json.loads(postcheck_path.read_text(encoding="utf-8"))
    legacy = {
        key: value
        for key, value in result.items()
        if key not in {"contact_sheets", "metrics_artifact", "png_hashes_artifact", "fixed_view_metrics"}
    }
    normalized = adapt_existing_postcheck_return(postcheck_path=postcheck_path, legacy_return=legacy)
    assert normalized["contact_sheets"] == postcheck["contact_sheets"]
    assert normalized["metrics_artifact"] == postcheck["metrics"]
    assert normalized["png_hashes_artifact"] == postcheck["png_hashes"]

    returned_path_drift = copy.deepcopy(result)
    returned_path_drift["contact_sheets"]["fixed_vs_gt"]["path"] = str(tmp_path / "other.png")
    with pytest.raises(ValueError, match="return contract drift for contact_sheets"):
        adapt_existing_postcheck_return(postcheck_path=postcheck_path, legacy_return=returned_path_drift)

    returned_sha_drift = copy.deepcopy(result)
    returned_sha_drift["metrics_artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="return contract drift for metrics_artifact"):
        adapt_existing_postcheck_return(postcheck_path=postcheck_path, legacy_return=returned_sha_drift)

    on_path = Path(postcheck["contact_sheets"]["on_path"]["path"])
    on_path.unlink()
    with pytest.raises(ValueError, match="on-path artifact is missing or SHA-mutated"):
        adapt_existing_postcheck_return(postcheck_path=postcheck_path, legacy_return=legacy)


@pytest.mark.parametrize(
    "source_shape,tile_shape,expected_resized",
    [
        ((100, 50), (200, 100), (50, 100)),
        ((50, 100), (100, 200), (100, 50)),
        ((60, 60), (200, 100), (100, 100)),
    ],
)
def test_fit_uses_uniform_scale_and_letterbox_for_all_aspects(
    source_shape: tuple[int, int], tile_shape: tuple[int, int], expected_resized: tuple[int, int]
) -> None:
    source = np.zeros((*source_shape, 3), dtype=np.uint8)
    cv2.circle(source, (source_shape[1] // 2, source_shape[0] // 2), max(2, min(source_shape) // 5), (255, 255, 255), -1)
    fitted, metadata = _fit_with_metadata(source, *tile_shape)
    assert fitted.shape[:2] == (tile_shape[1], tile_shape[0])
    assert (metadata["resized_width"], metadata["resized_height"]) == expected_resized
    assert metadata["scale"] == pytest.approx(
        metadata["resized_width"] / source_shape[1]
    )
    assert metadata["scale"] == pytest.approx(
        metadata["resized_height"] / source_shape[0]
    )
    mask = np.any(fitted > 0, axis=2)
    ys, xs = np.where(mask)
    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    assert abs(float(bbox_width) / float(bbox_height) - 1.0) < 0.15


def test_postcheck_rejects_missing_or_endpoint_hash_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "missing", count=2, width=8, height=6)
    render_paths = fixture["render_paths"]
    assert isinstance(render_paths, list)
    render_paths[1].unlink()
    with pytest.raises(ValueError, match="counts differ from the actual NVS pose contract"):
        run_postcheck(
            run_root=fixture["run_root"],
            render_attempt=fixture["render_attempt"],
            training_attempt=fixture["training_attempt"],
            training_input=fixture["training_input"],
            frame_selection=fixture["frame_selection"],
            segment_provenance=fixture["segment"],
            output_dir=fixture["run_root"] / "postcheck-001",
        )

    drift = _fixture(tmp_path / "drift", count=2, width=8, height=6)
    drift_paths = drift["render_paths"]
    assert isinstance(drift_paths, list)
    changed = np.zeros((6, 8, 3), dtype=np.uint8)
    assert cv2.imwrite(str(drift_paths[0]), changed)
    with pytest.raises(ValueError, match="fixed/NVS endpoint equality failed"):
        run_postcheck(
            run_root=drift["run_root"],
            render_attempt=drift["render_attempt"],
            training_attempt=drift["training_attempt"],
            training_input=drift["training_input"],
            frame_selection=drift["frame_selection"],
            segment_provenance=drift["segment"],
            output_dir=drift["run_root"] / "postcheck-001",
        )


def test_failed_postcheck_resumes_cpu_only_without_rerender(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, count=2, width=8, height=6)
    ledger = RunLedger.create_or_resume(output_root=tmp_path / "ledger-outputs", run_id="resume", identity={"code_identity_sha256": "fixture"})
    render_result = {"executor_root": str(fixture["render_attempt"] / "executor"), "training_input": str(fixture["training_input"])}
    training_result = {"executor_root": str(fixture["training_attempt"] / "executor"), "training_input": str(fixture["training_input"])}
    frames = {"raw_result": {"selection_path": str(fixture["frame_selection"])} }
    colmap = {"raw_result_path": str(fixture["segment"])}
    render_paths = fixture["render_paths"]
    assert isinstance(render_paths, list)
    render_paths[1].unlink()
    first, first_status = _coverage_render_postcheck_stage(
        ledger=ledger,
        route=tmp_path,
        coverage_render=render_result,
        coverage_training=training_result,
        training_input={"source_path": str(fixture["training_input"])},
        frames_result=frames,
        colmap_result=colmap,
        plan=False,
    )
    assert first_status == "blocked"
    assert first["render_reused"] is True
    assert (fixture["render_attempt"] / "executor" / "result.json").is_file()
    # Restore the input PNG and use a fresh CPU postcheck attempt. No executor
    # callback is involved, so this explicitly proves that recovery does not
    # rerun the GPU render stage.
    image = np.full((6, 8, 3), (17, 40, 90), dtype=np.uint8)
    image[:, 1, :] = 255
    assert cv2.imwrite(str(render_paths[1]), image)
    second, second_status = _coverage_render_postcheck_stage(
        ledger=ledger,
        route=tmp_path,
        coverage_render=render_result,
        coverage_training=training_result,
        training_input={"source_path": str(fixture["training_input"])},
        frames_result=frames,
        colmap_result=colmap,
        plan=False,
    )
    assert second_status == "passed"
    assert second["gpu_invoked"] is False
    assert second["render_reused"] is True
    summary = json.loads(ledger.summary_path.read_text(encoding="utf-8"))
    attempts = summary["stages"]["coverage-smoke-render-postcheck"]
    assert [item["attempt"] for item in attempts] == ["attempt-0001", "attempt-0002"]
