from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.longsplat.acceptance_delivery import AcceptanceDeliveryError, create_candidate_delivery
from scripts.longsplat.pipeline_contract import RunLedger
from scripts.longsplat.reconstruct_pipeline import _automated_technical_delivery
from tests.unit.test_authority_manifest import _fixture


def test_candidate_delivery_is_explicitly_not_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _manifest, authority_path, route = _fixture(
        tmp_path,
        "candidate",
        camera_names=["left", "right"],
        width=29,
        height=17,
    )
    evidence_root = route / "outputs" / "candidate" / "evidence"
    evidence_root.mkdir(parents=True)
    converted = evidence_root / "converted.ply"
    converted.write_bytes(b"technical-ply")
    evaluation = evidence_root / "evaluation.json"
    evaluation.write_text(
        json.dumps({"STRUCTURAL_EVALUATION_PASS": True, "SAME_CAMERA_VISUAL_PASS": "needs_review", "held_out": False}),
        encoding="utf-8",
    )
    conversion = evidence_root / "conversion.json"
    conversion.write_text(json.dumps({"structural_pass": True}), encoding="utf-8")
    comparison = evidence_root / "comparison.png"
    assert cv2.imwrite(str(comparison), np.zeros((17, 29, 3), dtype=np.uint8))

    monkeypatch.setattr("scripts.longsplat.acceptance_delivery._point_count", lambda _path: 3)
    result = create_candidate_delivery(
        converted_ply=converted,
        output_dir=route / "outputs" / "candidate" / "candidate_delivery",
        authority_manifest=authority_path,
        evidence_files={
            "authority_manifest.json": authority_path,
            "conversion_result.json": conversion,
            "evaluation_result.json": evaluation,
        },
        comparison_sheet=comparison,
    )

    root = Path(result["root"])
    manifest = json.loads((root / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert result["accepted"] is False
    assert result["supersplat"] is False
    assert manifest["three_view_manual_acceptance_required"] is True
    assert manifest["point_cloud"]["vertices"] == 3
    assert (root / "PROVENANCE.json").is_file()
    assert (root / "CANDIDATE_REPORT.md").is_file()
    assert json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))["accepted"] is False
    assert "point_cloud.ply" in (root / "SHA256SUMS.txt").read_text(encoding="utf-8")


def test_failed_same_camera_visual_is_advisory_after_structural_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _manifest, authority_path, route = _fixture(
        tmp_path,
        "candidate-fail",
        camera_names=["only"],
        width=13,
        height=9,
    )
    evidence_root = route / "outputs" / "candidate-fail" / "evidence"
    evidence_root.mkdir(parents=True)
    converted = evidence_root / "converted.ply"
    converted.write_bytes(b"technical-ply")
    evaluation = evidence_root / "evaluation.json"
    evaluation.write_text(
        json.dumps({
            "STRUCTURAL_EVALUATION_PASS": True,
            "SAME_CAMERA_VISUAL_PASS": "fail",
            "quality_advisories": ["visual quality is below the local advisory threshold"],
        }),
        encoding="utf-8",
    )
    comparison = evidence_root / "comparison.png"
    assert cv2.imwrite(str(comparison), np.zeros((9, 13, 3), dtype=np.uint8))

    monkeypatch.setattr("scripts.longsplat.acceptance_delivery._point_count", lambda _path: 1)
    result = create_candidate_delivery(
        converted_ply=converted,
        output_dir=route / "outputs" / "candidate-fail" / "candidate_delivery",
        authority_manifest=authority_path,
        evidence_files={"evaluation_result.json": evaluation},
        comparison_sheet=comparison,
    )
    assert result["accepted"] is False
    manifest = json.loads((Path(result["root"]) / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["same_camera_visual_pass"] == "fail"
    assert manifest["quality_advisories"]


def test_candidate_delivery_requires_structural_evaluation_pass(tmp_path: Path) -> None:
    _manifest, authority_path, route = _fixture(
        tmp_path,
        "candidate-no-structural",
        camera_names=["only"],
        width=13,
        height=9,
    )
    evidence_root = route / "outputs" / "candidate-no-structural" / "evidence"
    evidence_root.mkdir(parents=True)
    converted = evidence_root / "converted.ply"
    converted.write_bytes(b"technical-ply")
    evaluation = evidence_root / "evaluation.json"
    evaluation.write_text(json.dumps({"SAME_CAMERA_VISUAL_PASS": "fail"}), encoding="utf-8")
    comparison = evidence_root / "comparison.png"
    assert cv2.imwrite(str(comparison), np.zeros((9, 13, 3), dtype=np.uint8))

    with pytest.raises(AcceptanceDeliveryError, match="structural converted-evaluation pass"):
        create_candidate_delivery(
            converted_ply=converted,
            output_dir=route / "outputs" / "candidate-no-structural" / "candidate_delivery",
            authority_manifest=authority_path,
            evidence_files={"evaluation_result.json": evaluation},
            comparison_sheet=comparison,
        )



def test_candidate_delivery_comparison_sheet_contract_is_identity_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (contract hardened): create_candidate_delivery returns
    comparison_sheet as an _identity dict containing a 'path' key - the
    artifacts builder in _automated_technical_delivery must consume its path."""
    _manifest, authority_path, route = _fixture(tmp_path, "cscontract", camera_names=["left", "right"], width=29, height=17)
    evidence_root = route / "outputs" / "cscontract" / "evidence"
    evidence_root.mkdir(parents=True)
    converted = evidence_root / "converted.ply"
    converted.write_bytes(b"technical-ply")
    evaluation = evidence_root / "evaluation.json"
    evaluation.write_text(json.dumps({"STRUCTURAL_EVALUATION_PASS": True, "SAME_CAMERA_VISUAL_PASS": "needs_review", "held_out": False}), encoding="utf-8")
    comparison = evidence_root / "comparison.png"
    assert cv2.imwrite(str(comparison), np.zeros((17, 29, 3), dtype=np.uint8))

    monkeypatch.setattr("scripts.longsplat.acceptance_delivery._point_count", lambda _path: 3)
    (evidence_root / "conversion.json").write_text(json.dumps({"structural_pass": True}), encoding="utf-8")
    result = create_candidate_delivery(
        converted_ply=converted,
        output_dir=route / "outputs" / "cscontract" / "candidate_delivery",
        authority_manifest=authority_path,
        evidence_files={
            "authority_manifest.json": authority_path,
            "conversion_result.json": evidence_root / "conversion.json",
            "evaluation_result.json": evaluation,
        },
        comparison_sheet=comparison,
    )
    cs = result["comparison_sheet"]
    assert isinstance(cs, dict), cs
    assert "path" in cs
    assert set(cs).issubset({"path", "sha256", "size_bytes"})
    assert Path(cs["path"]).is_file()
    assert Path(cs["path"]).name == "fixed_gt_native_converted_contact_sheet.png"


def _automated_delivery_deps(tmp_path: Path):
    run_dir = tmp_path / "delivery-run"
    run_dir.mkdir()
    ledger = RunLedger(run_dir, {"schema_version": "test", "code_identity": {}}, resumed=True)
    authority_path = run_dir / "authority.json"
    authority_path.write_text(json.dumps({"schema_version": "placeholder"}), encoding="utf-8")
    conversion_root = run_dir / "conversion"
    conversion_root.mkdir(parents=True)
    (conversion_root / "conversion_result.json").write_text(json.dumps({"stage": "conversion"}), encoding="utf-8")
    ply = run_dir / "point_cloud.ply"
    ply.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"end_header\n0 0 0\n"
    )
    eval_root = run_dir / "eval"
    eval_root.mkdir(parents=True)
    comparison_png = run_dir / "comparison.png"
    cv2.imwrite(str(comparison_png), np.zeros((8, 8, 3), dtype=np.uint8))
    evaluation_result = eval_root / "evaluation_result.json"
    evaluation_result.write_text(json.dumps({"candidate_delivery": {"comparison_sheet": str(comparison_png)}}), encoding="utf-8")
    early_gate_path = run_dir / "early_gate.json"
    formal_gate_path = run_dir / "formal_gate.json"
    postcheck_path = run_dir / "postcheck.json"
    for f in (early_gate_path, formal_gate_path, postcheck_path):
        f.write_text(json.dumps({"gate": "pass"}), encoding="utf-8")
    authority = {"authority_manifest_path": str(authority_path.resolve())}
    conversion = {"executor_root": str(conversion_root.resolve()), "executor_result": {"structural": {"path": str(ply.resolve())}}}
    evaluation = {"executor_root": str(eval_root.resolve())}
    postprocess = {"computed_pass": True, "postprocess_result_path": str(evaluation_result.resolve())}
    early_gate = {"computed_pass": True, "automated_gate_result_path": early_gate_path}
    formal_gate = {"computed_pass": True, "automated_gate_result_path": formal_gate_path}
    native_postcheck = {"postcheck_result_path": postcheck_path}
    return run_dir, ledger, authority, conversion, evaluation, postprocess, early_gate, formal_gate, native_postcheck, ply, comparison_png


def test_automated_technical_delivery_artifacts_consume_comparison_sheet_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (dict-as-path bug): a fake create_candidate_delivery returning a
    dict-typed comparison_sheet must not raise TypeError in _artifacts - the
    artifacts list must use comparison_sheet['path'] (a str), not the dict."""
    (run_dir, ledger, authority, conversion, evaluation, postprocess, early_gate, formal_gate, native_postcheck, ply, comparison_png) = _automated_delivery_deps(tmp_path)

    def fake_create(**kwargs):
        manifest_path = run_dir / "candidate_manifest.json"
        provenance = run_dir / "PROVENANCE.json"
        report = run_dir / "CANDIDATE_REPORT.md"
        sums = run_dir / "SHA256SUMS.txt"
        for f in (manifest_path, provenance, report, sums):
            f.write_text("x", encoding="utf-8")
        return {
            "root": str(run_dir / "delivery_root"),
            "point_cloud": {"path": str(ply.resolve()), "sha256": "abc", "size_bytes": ply.stat().st_size, "vertices": 3},
            "vertices": 3,
            "candidate_manifest": str(manifest_path.resolve()),
            "provenance": str(provenance.resolve()),
            "report": str(report.resolve()),
            "comparison_sheet": {"path": str(comparison_png.resolve()), "sha256": "def", "size_bytes": comparison_png.stat().st_size},
            "sha256sums": {"path": str(sums.resolve()), "file_count": 1, "verified": True},
            "accepted": False,
            "supersplat": False,
        }

    monkeypatch.setattr("scripts.longsplat.acceptance_delivery.create_candidate_delivery", fake_create)
    result, status = _automated_technical_delivery(
        ledger=ledger,
        route=run_dir / "route",
        authority=authority,
        conversion=conversion,
        evaluation=evaluation,
        postprocess=postprocess,
        early_gate=early_gate,
        formal_gate=formal_gate,
        native_postcheck=native_postcheck,
        plan=False,
    )
    assert status == "passed"
    artifact_paths = [a["path"] for a in result.get("artifacts", [])]
    assert str(comparison_png.resolve()) in artifact_paths


def test_automated_technical_delivery_publishes_and_receipts_final_ply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir, ledger, authority, conversion, evaluation, postprocess, early_gate, formal_gate, native_postcheck, ply, comparison_png) = _automated_delivery_deps(tmp_path)

    def fake_create(**kwargs):
        manifest_path = run_dir / "candidate_manifest.json"
        provenance = run_dir / "PROVENANCE.json"
        report = run_dir / "CANDIDATE_REPORT.md"
        sums = run_dir / "SHA256SUMS.txt"
        for file_path in (manifest_path, provenance, report, sums):
            file_path.write_text("x", encoding="utf-8")
        return {
            "root": str(run_dir / "delivery_root"),
            "point_cloud": {"path": str(ply.resolve()), "sha256": "unused", "size_bytes": ply.stat().st_size, "vertices": 1},
            "vertices": 1,
            "candidate_manifest": str(manifest_path.resolve()),
            "provenance": str(provenance.resolve()),
            "report": str(report.resolve()),
            "comparison_sheet": {"path": str(comparison_png.resolve()), "sha256": "unused", "size_bytes": comparison_png.stat().st_size},
            "sha256sums": {"path": str(sums.resolve()), "file_count": 1, "verified": True},
            "accepted": False,
            "supersplat": False,
        }

    monkeypatch.setattr("scripts.longsplat.acceptance_delivery.create_candidate_delivery", fake_create)
    result, status = _automated_technical_delivery(
        ledger=ledger,
        route=run_dir / "route",
        authority=authority,
        conversion=conversion,
        evaluation=evaluation,
        postprocess=postprocess,
        early_gate=early_gate,
        formal_gate=formal_gate,
        native_postcheck=native_postcheck,
        plan=False,
        publish_dir=tmp_path / "public",
        delivery_name="gallery",
    )
    assert status == "passed"
    published = result["published_ply"]
    published_path = Path(published["published"]["path"])
    assert published_path.is_file()
    assert result["published_ply_receipt"] == str(run_dir / "published_ply.json")
    assert json.loads((run_dir / "published_ply.json").read_text(encoding="utf-8"))["published"]["path"] == str(published_path)
