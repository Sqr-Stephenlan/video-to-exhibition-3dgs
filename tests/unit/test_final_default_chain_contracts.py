from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.longsplat.colmap_contract import parse_colmap_image_pose_records
from scripts.longsplat.convergence_smoke import ConvergenceSmokeBlocked, plan_convergence_smoke
from scripts.longsplat.converted_eval_postprocess import _output_path as converted_output_path
from scripts.longsplat.coverage_smoke import _output_dir
from scripts.longsplat.pipeline_contract import (
    PipelineBlocked,
    classify_stage_migration,
)
from scripts.longsplat.raw_pipeline import _segment_provenance, _select_mapper_component
from scripts.longsplat.longsplat_input import parse_colmap_images_text
from scripts.longsplat.reconstruct_pipeline import (
    DEFAULT_PIPELINE_PROFILE,
    DEFAULT_STAGES,
    build_arg_parser,
    run_reconstruction,
)
from scripts.longsplat.smoke_executor import SmokeExecutorBlocked, _output_path as smoke_output_path


def _images_text(*, include_points: bool = True, second_header: bool = False) -> str:
    first = "1 1 0 0 0 0 0 0 1 view-a.png\n"
    if not include_points:
        return first
    points = "\n"
    second = "2 1 0 0 0 0 0 0 1 view-b.png\n\n" if second_header else ""
    return first + points + second


def _component(path: Path, names: list[str]) -> Path:
    path.mkdir(parents=True)
    for file_name in ("cameras.bin", "images.bin", "points3D.bin"):
        (path / file_name).write_bytes(b"fixture")
    (path / "images.txt").write_text(
        "".join(
            f"{index + 1} 1 0 0 0 0 0 0 1 {name}\n\n"
            for index, name in enumerate(names)
        ),
        encoding="utf-8",
    )
    return path


def test_dynamic_executor_roots_accept_two_external_roots_and_reject_escape(tmp_path: Path) -> None:
    route = tmp_path / "route-code"
    route.mkdir()
    roots = [tmp_path / "external-a", tmp_path / "external-b"]
    for root in roots:
        root.mkdir()
        run = root / "run"
        run.mkdir()
        path = run / "stage" / "artifact.json"
        path.parent.mkdir()
        path.write_text("{}", encoding="utf-8")
        assert smoke_output_path(path, route, "smoke artifact", containment_root=run) == path
        assert converted_output_path(path, run, "converted artifact", containment_root=run) == path
        output_dir = run / "coverage" / "attempt"
        assert _output_dir(output_dir, route, run) == output_dir
        with pytest.raises(SmokeExecutorBlocked):
            smoke_output_path(root / "outside.json", route, "escape", containment_root=run)


def test_default_cli_policy_is_single_convergence_and_has_no_legacy_gpu_stages() -> None:
    args = build_arg_parser().parse_args(
        ["--input-video", "/tmp/video.mp4", "--output-root", "/tmp/out", "--run-id", "fresh"]
    )
    assert args.stop_after == "automated-technical-delivery"
    assert DEFAULT_PIPELINE_PROFILE == "single-convergence1000-v1"
    assert "smoke100-training" not in DEFAULT_STAGES
    assert "coverage-smoke-training" not in DEFAULT_STAGES
    assert DEFAULT_STAGES[-1] == "automated-technical-delivery"
    assert DEFAULT_STAGES.index("automated-early-gate") < DEFAULT_STAGES.index("formal-training")
    assert DEFAULT_STAGES.index("automated-formal-gate") < DEFAULT_STAGES.index("conversion")


def test_default_plan_records_dynamic_stage_order_without_algorithms(tmp_path: Path) -> None:
    source = tmp_path / "new-video.mp4"
    source.write_bytes(b"new video")
    output = tmp_path / "arbitrary-output-root"
    result = run_reconstruction(
        input_video=source,
        output_root=output,
        run_id="new_run",
        stop_after="automated-technical-delivery",
        plan=True,
    )
    config = json.loads((output / "new_run" / "config.json").read_text(encoding="utf-8"))
    assert config["pipeline_profile"] == DEFAULT_PIPELINE_PROFILE
    assert config["stage_order"] == list(DEFAULT_STAGES)
    assert result["status"] == "planned"
    assert "smoke100-training" not in result["stage_results"]
    assert "coverage-smoke-training" not in result["stage_results"]


@pytest.mark.parametrize("count", [2, 45, 70, 144, 240, 501, 1001])
def test_convergence_profile_dynamic_camera_bound(count: int) -> None:
    plan = plan_convergence_smoke(
        active_camera_count=count,
        camera_names=[f"arbitrary-{index * 3}.jpg" for index in range(count)],
    )
    assert plan["requested_iterations"] == 1000
    assert plan["active_camera_count"] == count


def test_convergence_profile_records_coverage_advisory_without_selector_cap() -> None:
    plan = plan_convergence_smoke(active_camera_count=1001, camera_names=[str(index) for index in range(1001)])
    assert plan["computed_pass"] is True
    assert plan["coverage_advisory"]["predicted_zero_exposure_camera_count"] == 1
    assert plan["coverage_advisory"]["warning"]
    with pytest.raises(ConvergenceSmokeBlocked, match="fixed at exactly 1000"):
        plan_convergence_smoke(active_camera_count=70, camera_names=[str(index) for index in range(70)], requested_iterations=999)


def test_colmap_two_line_contract_is_shared_and_strict(tmp_path: Path) -> None:
    path = tmp_path / "images.txt"
    path.write_text(_images_text(include_points=False), encoding="utf-8")
    with pytest.raises(PipelineBlocked, match="POINTS2D"):
        parse_colmap_image_pose_records(path)
    with pytest.raises(Exception, match="POINTS2D"):
        parse_colmap_images_text(path)
    path.write_text(
        "1 1 0 0 0 0 0 0 1 view-a.png\n"
        "2 1 0 0 0 0 0 0 1 view-b.png\n",
        encoding="utf-8",
    )
    with pytest.raises(PipelineBlocked, match="replaced"):
        parse_colmap_image_pose_records(path)
    with pytest.raises(Exception, match="POINTS2D"):
        parse_colmap_images_text(path)
    path.write_text(_images_text(), encoding="utf-8")
    records = parse_colmap_image_pose_records(path)
    assert [record["name"] for record in records] == ["view-a.png"]


def test_component_inventory_records_overlap_without_blocking_a_clear_winner(tmp_path: Path) -> None:
    selected = [{"staged_name": f"frame_{index:06d}.png"} for index in range(6)]
    first = _component(tmp_path / "mapper" / "1", ["frame_000000.png", "frame_000001.png", "frame_000002.png", "frame_000003.png"])
    second = _component(tmp_path / "mapper" / "2", ["frame_000004.png", "frame_000005.png"])
    chosen, names, inventory = _select_mapper_component([first, second], selected)
    assert chosen == first.resolve()
    assert names == ["frame_000000.png", "frame_000001.png", "frame_000002.png", "frame_000003.png"]
    assert inventory["mapper_component_count"] == 2
    assert inventory["active_model_component_count"] == 1
    overlap = _component(tmp_path / "mapper-overlap" / "1", [
        "frame_000000.png", "frame_000001.png", "frame_000002.png", "frame_000003.png",
    ])
    overlap2 = _component(tmp_path / "mapper-overlap" / "2", [
        "frame_000002.png", "frame_000003.png", "frame_000004.png",
    ])
    chosen_overlap, _, overlap_inventory = _select_mapper_component([overlap, overlap2], selected)
    assert chosen_overlap == overlap.resolve()
    relations = {
        relation["relation"]
        for item in overlap_inventory["components"]
        for relation in item["relationships"]
    }
    assert "non_subset_overlap" in relations


def test_component_inventory_dominates_strict_subset_before_segment_tie_checks(tmp_path: Path) -> None:
    selected = [{"staged_name": f"frame_{index:06d}.png"} for index in range(8)]
    dominant = _component(
        tmp_path / "mapper" / "large",
        [f"frame_{index:06d}.png" for index in range(8)],
    )
    redundant_subset = _component(
        tmp_path / "mapper" / "small",
        ["frame_000002.png", "frame_000006.png"],
    )

    chosen, names, inventory = _select_mapper_component(
        [redundant_subset, dominant],
        selected,
    )

    assert chosen == dominant.resolve()
    assert names == [f"frame_{index:06d}.png" for index in range(8)]
    assert inventory["mapper_component_count"] == 2
    assert inventory["active_candidate_component_count"] == 1
    assert inventory["active_model_component_count"] == 1
    by_name = {item["component_name"]: item for item in inventory["components"]}
    assert by_name[redundant_subset.name]["selection_status"] == "dominated"
    assert by_name[redundant_subset.name]["dominated_by_components"] == [dominant.name]
    assert by_name[redundant_subset.name]["longest_contiguous_run_count_tie"] is True
    assert by_name[dominant.name]["selection_status"] == "selected"
    assert "strict registered-name subsets" in inventory["selection_rule"]


def test_component_inventory_does_not_let_shorter_active_segment_tie_block_winner(tmp_path: Path) -> None:
    selected = [{"staged_name": f"sample_{index:03d}.jpg"} for index in range(8)]
    winner = _component(
        tmp_path / "fragments" / "winner",
        ["sample_000.jpg", "sample_001.jpg", "sample_002.jpg", "sample_003.jpg"],
    )
    shorter_tied = _component(
        tmp_path / "fragments" / "shorter-tied",
        ["sample_005.jpg", "sample_007.jpg"],
    )

    chosen, _, inventory = _select_mapper_component([winner, shorter_tied], selected)

    assert chosen == winner.resolve()
    by_name = {item["component_name"]: item for item in inventory["components"]}
    assert by_name[shorter_tied.name]["longest_contiguous_run_count_tie"] is True
    assert by_name[shorter_tied.name]["selection_status"] == "not_selected"
    assert by_name[shorter_tied.name]["exclusion_reasons"] == [
        "shorter_longest_contiguous_registered_segment"
    ]


def test_component_inventory_records_tie_inside_unique_winner_as_provenance(tmp_path: Path) -> None:
    selected = [{"staged_name": f"sample_{index:03d}.jpg"} for index in range(8)]
    tied_winner = _component(
        tmp_path / "winner-tie" / "component",
        ["sample_000.jpg", "sample_001.jpg", "sample_004.jpg", "sample_005.jpg"],
    )
    shorter = _component(tmp_path / "winner-tie" / "shorter", ["sample_007.jpg"])

    chosen, _, inventory = _select_mapper_component([tied_winner, shorter], selected)
    assert chosen == tied_winner.resolve()
    selected_item = next(item for item in inventory["components"] if item["selection_status"] == "selected")
    assert selected_item["longest_contiguous_run_count_tie"] is True
    assert any("model_retains_all_registered_images" in item for item in selected_item["selection_advisories"])
    provenance = _segment_provenance(
        selected,
        selected_item["registered_image_names"],
        component_count=2,
        component_inventory=inventory,
    )
    assert provenance["selected_segment_tie"] is True
    assert provenance["selected_segment_tie_policy"].startswith("provenance_only")


def test_component_inventory_resolves_equal_candidates_stably_and_preserves_ties(tmp_path: Path) -> None:
    selected = [{"staged_name": f"capture_{index:03d}.jpg"} for index in range(6)]
    first = _component(
        tmp_path / "equal" / "first",
        ["capture_000.jpg", "capture_001.jpg", "capture_002.jpg"],
    )
    second = _component(
        tmp_path / "equal" / "second",
        ["capture_003.jpg", "capture_004.jpg", "capture_005.jpg"],
    )
    chosen, names, inventory = _select_mapper_component([first, second], selected)
    assert chosen == first.resolve()
    assert names == ["capture_000.jpg", "capture_001.jpg", "capture_002.jpg"]
    assert inventory["selection_confidence"] == "low"
    assert inventory["selection_ties"] == [
        {
            "kind": "equal_quality_active_candidates",
            "components": [first.name, second.name],
            "quality": {
                "longest_contiguous_run_count": 3,
                "registered_image_count": 3,
                "coverage_ratio": 0.5,
            },
            "resolution": "stable_component_name_then_path_tiebreak; one_model_no_merge",
        }
    ]
    assert {item["selection_status"] for item in inventory["components"]} == {"selected", "not_selected"}
    reversed_chosen, reversed_names, reversed_inventory = _select_mapper_component(
        [second, first], selected
    )
    assert reversed_chosen == chosen
    assert reversed_names == names
    assert reversed_inventory["selection_ties"] == inventory["selection_ties"]
    assert reversed_inventory["selection_confidence"] == "low"

    overlap_first = _component(
        tmp_path / "equal-overlap" / "first",
        ["capture_000.jpg", "capture_001.jpg", "capture_002.jpg"],
    )
    overlap_second = _component(
        tmp_path / "equal-overlap" / "second",
        ["capture_002.jpg", "capture_003.jpg", "capture_004.jpg"],
    )
    overlap_chosen, _, overlap_inventory = _select_mapper_component(
        [overlap_second, overlap_first], selected
    )
    assert overlap_chosen == overlap_first.resolve()
    assert any(
        relation["relation"] == "non_subset_overlap"
        for item in overlap_inventory["components"]
        for relation in item["relationships"]
    )

    same_first = _component(
        tmp_path / "same" / "first",
        ["capture_000.jpg", "capture_001.jpg"],
    )
    same_second = _component(
        tmp_path / "same" / "second",
        ["capture_000.jpg", "capture_001.jpg"],
    )
    same_forward = _select_mapper_component([same_first, same_second], selected)
    same_reverse = _select_mapper_component([same_second, same_first], selected)
    assert same_forward[0] == same_first.resolve()
    assert same_reverse[0] == same_first.resolve()
    assert same_forward[2]["selection_ties"] == same_reverse[2]["selection_ties"]
    assert same_forward[2]["selection_confidence"] == "low"


def test_component_inventory_preserves_one_component_and_name_validation(tmp_path: Path) -> None:
    selected = [{"staged_name": "single.png"}]
    one = _component(tmp_path / "one" / "component", ["single.png"])
    chosen, names, inventory = _select_mapper_component([one], selected)
    assert chosen == one.resolve()
    assert names == ["single.png"]
    assert inventory["mapper_component_count"] == 1
    assert inventory["active_model_component_count"] == 1

    empty = tmp_path / "empty" / "component"
    empty.mkdir(parents=True)
    for file_name in ("cameras.bin", "images.bin", "points3D.bin"):
        (empty / file_name).write_bytes(b"fixture")
    chosen_empty, names_empty, empty_inventory = _select_mapper_component([empty], selected)
    assert chosen_empty == empty.resolve()
    assert names_empty == []
    assert empty_inventory["components"][0]["registered_image_names"] is None

    duplicate = _component(tmp_path / "duplicate" / "component", ["single.png", "single.png"])
    with pytest.raises(PipelineBlocked, match="duplicate COLMAP image NAME"):
        _select_mapper_component([duplicate], selected)


def test_component_inventory_excludes_invalid_candidate_locally(tmp_path: Path) -> None:
    selected = [{"staged_name": f"frame_{index:03d}.png"} for index in range(5)]
    valid = _component(tmp_path / "valid" / "component", [f"frame_{index:03d}.png" for index in range(5)])
    malformed = _component(tmp_path / "malformed" / "component", ["frame_000.png", "frame_001.png"])
    (malformed / "images.txt").write_text("not a COLMAP image record\n", encoding="utf-8")

    chosen, names, inventory = _select_mapper_component([malformed, valid], selected)

    assert chosen == valid.resolve()
    assert names == [f"frame_{index:03d}.png" for index in range(5)]
    malformed_item = next(item for item in inventory["components"] if item["component_name"] == malformed.name)
    assert malformed_item["selection_status"] == "invalid"
    assert "registered_name_parse_error" in malformed_item["exclusion_reasons"]


def test_component_inventory_blocks_only_after_all_candidates_are_invalid(tmp_path: Path) -> None:
    selected = [{"staged_name": "frame_000.png"}]
    malformed = _component(tmp_path / "malformed-only" / "component", ["frame_000.png"])
    (malformed / "images.txt").write_text("not a COLMAP image record\n", encoding="utf-8")

    with pytest.raises(PipelineBlocked, match="no usable registered") as raised:
        _select_mapper_component([malformed], selected)

    inventory = getattr(raised.value, "inventory", None)
    assert isinstance(inventory, dict)
    assert inventory["active_model_component_count"] == 0
    assert inventory["components"][0]["selection_status"] == "invalid"


def test_component_inventory_records_selected_text_parse_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selected = [{"staged_name": "frame_000.png"}]
    broken = _component(tmp_path / "broken-text" / "component", ["frame_000.png"])

    def raise_parse_error(_component_path: Path) -> list[str]:
        raise PipelineBlocked("synthetic registered-name parse failure")

    monkeypatch.setattr("scripts.longsplat.raw_pipeline._component_image_names", raise_parse_error)
    with pytest.raises(PipelineBlocked, match="no usable registered") as raised:
        _select_mapper_component([broken], selected)

    inventory = getattr(raised.value, "inventory", None)
    assert isinstance(inventory, dict)
    item = inventory["components"][0]
    assert item["selection_status"] == "invalid"
    assert "registered_name_parse_error" in item["exclusion_reasons"]


def test_stage_migration_distinguishes_exact_stale_and_unsafe() -> None:
    expected = {
        "schema_version": "v2",
        "source_sha": "source",
        "camera_sha": "camera",
        "consumer_code_identity_sha256": "new-code",
    }
    exact = {"status": "passed", "reusable": True, **expected}
    assert classify_stage_migration(exact, exact, immutable_fields=("source_sha", "camera_sha"))["classification"] == "reusable_exact"
    stale = {"status": "passed", "reusable": True, "source_sha": "source", "camera_sha": "camera"}
    assert classify_stage_migration(stale, expected, immutable_fields=("source_sha", "camera_sha"))["classification"] == "stale_rebuildable"
    unsafe = {"status": "passed", "reusable": True, "source_sha": "other", "camera_sha": "camera"}
    assert classify_stage_migration(unsafe, expected, immutable_fields=("source_sha", "camera_sha"))["classification"] == "reject_unsafe_drift"
    orphan = {"status": "passed", "reusable": True, "orphan": True, **expected}
    assert classify_stage_migration(orphan, expected, immutable_fields=("source_sha", "camera_sha"))["classification"] == "reject_nonreusable"
