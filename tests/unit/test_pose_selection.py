from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "LongSplat"
    / "utils"
    / "pose_selection.py"
)
SPEC = importlib.util.spec_from_file_location("pose_selection_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _candidate(reference_index: int, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "reference_index": reference_index,
        "hard_reasons": [],
        "quality_reasons": ["reprojection_rmse"],
        "inlier_ratio": 0.81,
        "grid_coverage": 0.5,
        "reprojection_rmse_px": 3.0,
        "state": {"reference": reference_index},
    }
    value.update(overrides)
    return value


def test_strict_quality_pass_beats_first_hard_valid_candidate() -> None:
    first = _candidate(10)
    strict = _candidate(
        11,
        quality_reasons=[],
        inlier_ratio=0.70,
        grid_coverage=0.40,
        reprojection_rmse_px=5.0,
    )

    winner = MODULE.select_pose_candidate([first, strict], gate_mode="observe")

    assert winner is strict


def test_ranking_is_deterministic_and_reference_index_breaks_ties() -> None:
    high_ratio = _candidate(9, quality_reasons=[], inlier_ratio=0.90)
    low_ratio = _candidate(3, quality_reasons=[], inlier_ratio=0.80)
    assert MODULE.select_pose_candidate([low_ratio, high_ratio], gate_mode="off") is high_ratio

    tied_late = _candidate(8, quality_reasons=[], inlier_ratio=0.90)
    tied_early = _candidate(4, quality_reasons=[], inlier_ratio=0.90)
    winner = MODULE.select_pose_candidate([tied_late, tied_early], gate_mode="observe")
    assert winner is tied_early


def test_enforce_rejects_quality_failures_but_permissive_modes_choose_best() -> None:
    rejected = _candidate(2, quality_reasons=["inlier_ratio"])
    assert MODULE.select_pose_candidate([rejected], gate_mode="enforce") is None
    assert MODULE.select_pose_candidate([rejected], gate_mode="observe") is rejected
    assert MODULE.select_pose_candidate([rejected], gate_mode="off") is rejected


def test_selection_does_not_mutate_candidate_or_target_state() -> None:
    candidates = [_candidate(1), _candidate(2, quality_reasons=[])]
    before = copy.deepcopy(candidates)
    target_state = {"pose": "unchanged", "kp0": "unchanged"}

    winner = MODULE.select_pose_candidate(candidates, gate_mode="enforce")

    assert winner is candidates[1]
    assert candidates == before
    assert target_state == {"pose": "unchanged", "kp0": "unchanged"}

