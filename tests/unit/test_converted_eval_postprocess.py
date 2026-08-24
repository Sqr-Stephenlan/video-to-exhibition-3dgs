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
from scripts.longsplat.pipeline_contract import RunLedger
from tests.unit.test_authority_manifest import _fixture


def _write_reused_eval_evidence(
    route: Path,
    name: str,
    camera_names: list[str],
    width: int,
    height: int,
    *,
    exit_code: int = -9,
    visual_marker: str | None = "fail",
) -> tuple[Path, Path, Path]:
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
    evaluator_result = {"exit_code": exit_code, "reason": "fixture evaluator failure"}
    if visual_marker is not None:
        evaluator_result["SAME_CAMERA_VISUAL_PASS"] = visual_marker
    (root / "evaluation_result.json").write_text(json.dumps(evaluator_result), encoding="utf-8")
    (root / "request.json").write_text(json.dumps({"stage": "converted-eval"}), encoding="utf-8")
    (root / "argv.json").write_text(json.dumps({"argv": [], "shell": False}), encoding="utf-8")
    (root / "stdout.log").write_text("", encoding="utf-8")
    (root / "stderr.log").write_text("", encoding="utf-8")
    return root, gt_root, converted_root


def _postprocess_inputs(
    tmp_path: Path,
    name: str,
    camera_names: list[str],
    width: int,
    height: int,
) -> tuple[Path, Path, Path, Path, Path]:
    _manifest, authority_path, route = _fixture(
        tmp_path,
        name,
        camera_names=camera_names,
        width=width,
        height=height,
    )
    failed_root, _gt_root, _converted_root = _write_reused_eval_evidence(
        route, name, camera_names, width, height
    )
    conversion_root = route / "outputs" / name / "conversion"
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
                    "vertex_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    return authority_path, route, failed_root, ply, conversion_result


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
    failed_root, _gt_root, _converted_root = _write_reused_eval_evidence(
        route,
        "streaming",
        camera_names,
        width,
        height,
        exit_code=7,
        visual_marker=None,
    )
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

    run_dir = route / "outputs" / "streaming"
    ledger = RunLedger(run_dir, identity={}, resumed=False)
    attempt = ledger.begin_attempt("converted-eval-postprocess", {"fixture": True})
    evidence_root = attempt / "evidence"

    result = run_postprocess(
        authority_manifest_path=authority_path,
        conversion_result_path=conversion_result,
        converted_ply_path=ply,
        failed_evaluation_root=failed_root,
        output_root=evidence_root,
        route_root=route,
        containment_root=run_dir,
    )

    assert result["STRUCTURAL_EVALUATION_PASS"] is True
    assert result["FULL_STREAM_VALIDATION_PASS"] is True
    assert result["SAME_CAMERA_VISUAL_PASS"] == "pass"
    assert result["visual_quality_pass"] is True
    assert result["gpu_invoked"] is False
    assert result["render_reused"] is True
    assert result["cuda_rerun"] is False
    assert result["original_evaluator_exit_code"] == 7
    assert result["original_evaluator_failure_reason"] == "fixture evaluator failure"
    assert result["full_stream_validation"]["triples_processed"] == 2
    assert result["full_stream_validation"]["pass"] is True
    assert result["full_stream_validation"]["structural_pass"] is True
    # Legacy conversion evidence has no runtime residency telemetry, so no
    # theoretical full-resolution-frame estimate is reported.
    assert result["full_stream_validation"]["resident_full_resolution_frame_max"] is None
    assert result["metrics"]["path"].endswith("/metrics.json")
    assert Path(result["metrics"]["path"]).parent == evidence_root
    assert Path(result["png_hashes"]["path"]).parent == evidence_root
    assert Path(result["contact_sheet"]["path"]).is_file()
    postprocess_path = evidence_root / "postprocess_result.json"
    assert postprocess_path.is_file()
    assert not (attempt / "result.json").exists()
    ledger.finish_attempt(
        stage="converted-eval-postprocess",
        attempt=attempt,
        status="passed",
        result={
            "computed_pass": True,
            "postprocess_result_path": str(postprocess_path.resolve()),
            "artifacts": [
                {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in (
                    postprocess_path,
                    evidence_root / "metrics.json",
                    evidence_root / "png_hashes.json",
                    evidence_root / "fixed_gt_native_converted_contact_sheet.png",
                )
            ],
        },
    )
    assert (attempt / "result.json").is_file()
    assert ledger.latest_result("converted-eval-postprocess") is not None
    rows = json.loads((Path(result["metrics"]["path"])).read_text(encoding="utf-8"))["per_view"]
    assert [row["camera_name"] for row in rows] == camera_names
    context = result["candidate_delivery"]["provenance_context"]
    assert "segment_provenance" not in context
    assert context["quality_advisories"] == result["quality_advisories"]
    serialized = json.dumps(result, sort_keys=True)
    forbidden = ("frame_" + "000140", "2" + ".982", "dynamic-" + "person")
    assert all(token not in serialized for token in forbidden)


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
    output = route / "outputs" / "fresh" / "converted-eval-postprocess" / "attempt-0001" / "evidence"
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


@pytest.mark.parametrize("kind", ["symlink", "escape"])
def test_postprocess_child_symlink_or_escape_is_blocked(tmp_path: Path, kind: str) -> None:
    authority_path, route, failed_root, ply, conversion_result = _postprocess_inputs(
        tmp_path, f"unsafe-{kind}", ["only"], 13, 9
    )
    if kind == "symlink":
        outside = tmp_path / "outside"
        outside.mkdir()
        output = route / "outputs" / f"unsafe-{kind}" / "converted-eval-postprocess" / "attempt-0001" / "evidence"
        output.parent.mkdir(parents=True)
        output.symlink_to(outside, target_is_directory=True)
    else:
        output = tmp_path / "outside" / "converted-eval-postprocess" / "attempt-0001"
    with pytest.raises(ConvertedEvalPostprocessError, match="symlink|below route outputs|escapes"):
        run_postprocess(
            authority_manifest_path=authority_path,
            conversion_result_path=conversion_result,
            converted_ply_path=ply,
            failed_evaluation_root=failed_root,
            output_root=output,
            route_root=route,
        )


@pytest.mark.parametrize(
    ("name", "camera_names", "width", "height", "segment"),
    [
        (
            "dynamic-provenance-a",
            ["cam-west-9", "cam-east-2"],
            31,
            19,
            {"candidate_segments": [{"frame_names": ["cam-west-9"]}], "excluded_time_intervals": []},
        ),
        (
            "dynamic-provenance-b",
            ["view-r", "view-q", "view-p"],
            17,
            23,
            {"candidate_segments": [{"frame_names": ["view-r", "view-q", "view-p"]}], "excluded_time_intervals": []},
        ),
    ],
)
def test_structured_segment_provenance_is_source_bound_and_dynamic(
    tmp_path: Path,
    name: str,
    camera_names: list[str],
    width: int,
    height: int,
    segment: dict[str, object],
) -> None:
    authority_path, route, failed_root, ply, conversion_result = _postprocess_inputs(
        tmp_path, name, camera_names, width, height
    )
    run_dir = route / "outputs" / name
    segment_path = run_dir / "segment-provenance.json"
    segment_path.write_text(
        json.dumps({"schema_version": "segment-fixture-v1", "result": {"segment_provenance": segment}}),
        encoding="utf-8",
    )
    result = run_postprocess(
        authority_manifest_path=authority_path,
        conversion_result_path=conversion_result,
        converted_ply_path=ply,
        failed_evaluation_root=failed_root,
        output_root=run_dir / "postprocess-attempt",
        route_root=route,
        containment_root=run_dir,
        segment_provenance=segment_path,
    )

    context = result["candidate_delivery"]["provenance_context"]
    assert context["segment_provenance"] == segment
    source = context["segment_provenance_source"]
    assert source["path"] == str(segment_path.resolve())
    assert source["sha256"] == hashlib.sha256(segment_path.read_bytes()).hexdigest()
    if context["quality_advisories"]:
        quality_source = context["quality_advisories_source"]
        metrics_path = Path(result["metrics"]["path"])
        assert quality_source["path"] == str(metrics_path.resolve())
        assert quality_source["sha256"] == hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    assert json.loads(Path(result["metrics"]["path"]).read_text())["camera_order"] == camera_names
    serialized = json.dumps(result, sort_keys=True)
    forbidden = ("frame_" + "000140", "2" + ".982", "dynamic-" + "person")
    assert all(token not in serialized for token in forbidden)
