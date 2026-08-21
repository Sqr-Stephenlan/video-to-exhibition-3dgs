from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.longsplat.converted_eval_postprocess import (
    ConvertedEvalPostprocessError,
    _contract_output_token,
    _expected_converted_names,
    create_candidate_from_postprocess,
    run_postprocess,
)
from tests.unit.test_authority_manifest import _fixture


def _write_reused_eval_evidence(route: Path, name: str, camera_names: list[str], width: int, height: int) -> tuple[Path, Path, Path]:
    root = route / "outputs" / name / "failed-eval"
    eval_root = root / "same_camera_eval"
    gt_root = eval_root / "evaluator_result_gt"
    converted_root = eval_root / "evaluator_result_renders"
    gt_root.mkdir(parents=True)
    converted_root.mkdir(parents=True)
    for index, camera_name in enumerate(camera_names):
        image = np.full((height, width, 3), index + 1, dtype=np.uint8)
        filename = f"{index:04d}_{camera_name}.png"
        assert cv2.imwrite(str(gt_root / filename), image)
        assert cv2.imwrite(str(converted_root / filename), image)
    (root / "evaluation_result.json").write_text(
        json.dumps({"SAME_CAMERA_VISUAL_PASS": "fail", "exit_code": -9}),
        encoding="utf-8",
    )
    (root / "request.json").write_text(json.dumps({"stage": "converted-eval"}), encoding="utf-8")
    (root / "argv.json").write_text(json.dumps({"argv": [], "shell": False}), encoding="utf-8")
    (root / "stdout.log").write_text("", encoding="utf-8")
    (root / "stderr.log").write_text("", encoding="utf-8")
    return root, gt_root, converted_root


def test_streaming_postprocess_uses_dynamic_order_and_marks_cpu_reuse(tmp_path: Path) -> None:
    camera_names = ["clip-7", "angle_final"]
    width, height = 29, 17
    _manifest, authority_path, route = _fixture(
        tmp_path,
        "streaming",
        camera_names=camera_names,
        width=width,
        height=height,
    )
    failed_root, _gt_root, _converted_root = _write_reused_eval_evidence(route, "streaming", camera_names, width, height)
    conversion_root = route / "outputs" / "streaming" / "conversion"
    conversion_root.mkdir(parents=True)
    ply = conversion_root / "point_cloud.ply"
    ply.write_bytes(b"technical-ply")
    digest = hashlib.sha256(ply.read_bytes()).hexdigest()
    conversion_result = conversion_root / "conversion_result.json"
    conversion_result.write_text(
        json.dumps(
            {
                "STRUCTURAL_CONVERSION_PASS": True,
                "structural_pass": True,
                "structural": {
                    "path": str(ply),
                    "sha256": digest,
                    "file_size": ply.stat().st_size,
                    "vertex_count": 3,
                    "attributes": ["x", "y", "z"],
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_postprocess(
        authority_manifest_path=authority_path,
        conversion_result_path=conversion_result,
        converted_ply_path=ply,
        failed_evaluation_root=failed_root,
        output_root=route / "outputs" / "streaming" / "converted-eval-postprocess" / "attempt-0001",
        route_root=route,
    )

    assert result["STRUCTURAL_EVALUATION_PASS"] is True
    assert result["FULL_STREAM_VALIDATION_PASS"] is True
    assert result["SAME_CAMERA_VISUAL_PASS"] == "pass"
    assert result["visual_quality_pass"] is True
    assert result["gpu_invoked"] is False
    assert result["render_reused"] is True
    assert result["cuda_rerun"] is False
    assert result["full_stream_validation"]["triples_processed"] == 2
    assert result["full_stream_validation"]["pass"] is True
    assert result["full_stream_validation"]["structural_pass"] is True
    # Legacy conversion evidence has no runtime residency telemetry, so no
    # theoretical full-resolution-frame estimate is reported.
    assert result["full_stream_validation"]["resident_full_resolution_frame_max"] is None
    assert result["metrics"]["path"].endswith("/metrics.json")
    assert Path(result["contact_sheet"]["path"]).is_file()
    rows = json.loads((Path(result["metrics"]["path"])).read_text(encoding="utf-8"))["per_view"]
    assert [row["camera_name"] for row in rows] == camera_names


def test_visual_quality_advisory_does_not_clear_structural_pass(tmp_path: Path) -> None:
    camera_names = ["first-view", "second-view"]
    width, height = 23, 15
    _manifest, authority_path, route = _fixture(
        tmp_path,
        "visual-advisory",
        camera_names=camera_names,
        width=width,
        height=height,
    )
    failed_root, _gt_root, converted_root = _write_reused_eval_evidence(
        route, "visual-advisory", camera_names, width, height
    )
    assert cv2.imwrite(
        str(converted_root / "0000_first-view.png"),
        np.zeros((height, width, 3), dtype=np.uint8),
    )
    conversion_root = route / "outputs" / "visual-advisory" / "conversion"
    conversion_root.mkdir(parents=True)
    ply = conversion_root / "point_cloud.ply"
    ply.write_bytes(b"technical-ply")
    conversion_result = conversion_root / "conversion_result.json"
    conversion_result.write_text(
        json.dumps(
            {
                "STRUCTURAL_CONVERSION_PASS": True,
                "structural_pass": True,
                "structural": {
                    "path": str(ply),
                    "sha256": hashlib.sha256(ply.read_bytes()).hexdigest(),
                    "file_size": ply.stat().st_size,
                    "vertex_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    result = run_postprocess(
        authority_manifest_path=authority_path,
        conversion_result_path=conversion_result,
        converted_ply_path=ply,
        failed_evaluation_root=failed_root,
        output_root=route / "outputs" / "visual-advisory" / "postprocess",
        route_root=route,
    )

    assert result["STRUCTURAL_EVALUATION_PASS"] is True
    assert result["FULL_STREAM_VALIDATION_PASS"] is True
    assert result["full_stream_validation"]["pass"] is True
    assert result["visual_quality_pass"] is False
    assert result["SAME_CAMERA_VISUAL_PASS"] == "fail"
    assert result["status"] == "needs_review"


def test_manual_candidate_path_still_requires_visual_pass(tmp_path: Path) -> None:
    camera_names = ["manual-view"]
    _manifest, authority_path, route = _fixture(
        tmp_path,
        "manual-gate",
        camera_names=camera_names,
        width=19,
        height=11,
    )
    postprocess_path = route / "outputs" / "manual-gate" / "postprocess_result.json"
    postprocess_path.parent.mkdir(parents=True, exist_ok=True)
    postprocess_path.write_text(
        json.dumps(
            {
                "schema_version": "longsplat-converted-eval-postprocess-v2",
                "STRUCTURAL_EVALUATION_PASS": True,
                "FULL_STREAM_VALIDATION_PASS": True,
                "SAME_CAMERA_VISUAL_PASS": "fail",
                "visual_quality_pass": False,
                "camera_count": 1,
                "camera_order": camera_names,
                "camera_dimensions": {"width": 19, "height": 11},
                "full_stream_validation": {"pass": True, "structural_pass": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConvertedEvalPostprocessError, match="complete passing CPU postprocess"):
        create_candidate_from_postprocess(
            authority_manifest_path=authority_path,
            postprocess_result_path=postprocess_path,
            candidate_output=route / "outputs" / "manual-gate" / "candidate",
            route_root=route,
            manual_visual_decision="pass",
            manual_visual_note="not visually accepted",
        )


def test_explicit_contract_extension_mapping_is_not_fuzzy() -> None:
    manifest = {"run_scope": {"registered_contract_names": ["clip-7.jpg", "angle_final.png"]}}
    order = ["clip-7", "angle_final"]
    assert _contract_output_token(order[0], manifest, 0) == "clip-7"
    assert _contract_output_token(order[1], manifest, 1) == "angle_final"
    assert _expected_converted_names(order, manifest) == ["0000_clip-7.png", "0001_angle_final.png"]


def test_camera_contract_mapping_drift_is_hard_stop() -> None:
    manifest = {"run_scope": {"registered_contract_names": ["different.png"]}}
    with pytest.raises(ConvertedEvalPostprocessError, match="mapping differs"):
        _contract_output_token("camera", manifest, 0)


def test_postprocess_attempt_is_append_only(tmp_path: Path) -> None:
    camera_names = ["only"]
    _manifest, authority_path, route = _fixture(tmp_path, "fresh", camera_names=camera_names, width=13, height=9)
    failed_root, _gt_root, _converted_root = _write_reused_eval_evidence(route, "fresh", camera_names, 13, 9)
    conversion_root = route / "outputs" / "fresh" / "conversion"
    conversion_root.mkdir(parents=True)
    ply = conversion_root / "point_cloud.ply"
    ply.write_bytes(b"technical-ply")
    conversion_result = conversion_root / "conversion_result.json"
    conversion_result.write_text(
        json.dumps(
            {
                "STRUCTURAL_CONVERSION_PASS": True,
                "structural_pass": True,
                "structural": {"path": str(ply), "sha256": hashlib.sha256(ply.read_bytes()).hexdigest(), "file_size": ply.stat().st_size, "vertex_count": 1},
            }
        ),
        encoding="utf-8",
    )
    output = route / "outputs" / "fresh" / "converted-eval-postprocess" / "attempt-0001"
    output.mkdir(parents=True)
    with pytest.raises(ConvertedEvalPostprocessError, match="fresh and append-only"):
        run_postprocess(
            authority_manifest_path=authority_path,
            conversion_result_path=conversion_result,
            converted_ply_path=ply,
            failed_evaluation_root=failed_root,
            output_root=output,
            route_root=route,
        )
