from __future__ import annotations

import json
from pathlib import Path

from scripts.longsplat.conversion_executor import execute_conversion, execute_evaluation
from tests.unit.test_authority_manifest import _fixture


def test_validate_only_conversion_uses_dynamic_camera_contract(tmp_path: Path) -> None:
    manifest, manifest_path, route = _fixture(
        tmp_path,
        "dynamic-conversion",
        camera_names=["view-a", "view-b", "view-c"],
        width=37,
        height=23,
    )
    evidence = route / "outputs" / "dynamic-conversion" / "conversion-attempt"
    result = execute_conversion(
        authority_manifest_path=manifest_path,
        evidence_root=evidence,
        route_root=route,
        dry_run=False,
        validate_only=True,
    )
    request = result["request"]
    assert request["camera_count"] == 3
    assert request["camera_order"] == ["view-a", "view-b", "view-c"]
    assert request["camera_dimensions"] == {"width": 37, "height": 23}
    assert request["held_out"] is False
    assert str(manifest["source_video"]["path"]) not in request["argv"]
    assert not evidence.exists()


def test_validate_only_evaluation_uses_all_ordered_cameras(tmp_path: Path) -> None:
    _manifest, manifest_path, route = _fixture(
        tmp_path,
        "dynamic-evaluation",
        camera_names=["first", "second"],
        width=29,
        height=17,
    )
    root = route / "outputs" / "dynamic-evaluation" / "conversion-attempt"
    snapshot = root / "conversion_model_snapshot"
    (snapshot / "converted_3dgs").mkdir(parents=True)
    (root / "conversion_result.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "conversion_result.json").write_text(
        json.dumps(
            {
                "STRUCTURAL_CONVERSION_PASS": True,
                "structural_pass": True,
                "structural": {
                    "converted_ply_path": str(snapshot / "converted_3dgs/point_cloud.ply")
                },
            }
        ),
        encoding="utf-8",
    )
    result = execute_evaluation(
        authority_manifest_path=manifest_path,
        evidence_root=root,
        route_root=route,
        conversion_evidence_root=root,
        dry_run=False,
        validate_only=True,
    )
    request = result["request"]
    assert request["camera_count"] == 2
    assert request["camera_order"] == ["first", "second"]
    assert request["camera_dimensions"] == {"width": 29, "height": 17}
    assert "--max-views" not in request["argv"]
    assert request["held_out"] is False
