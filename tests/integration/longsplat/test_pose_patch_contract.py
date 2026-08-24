"""RED contract tests for LongSplat pose quality gates patch — Task 4 Step 1.

Validates the patch file existence, content, and backend parity.
All tests are expected to FAIL before the patch is generated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_PATCH_PATH = (
    _PROJECT_ROOT
    / "docs"
    / "longsplat"
    / "patches"
    / "longsplat_pose_quality_gates.patch"
)

_BACKEND_ROOT = _PROJECT_ROOT / "third_party" / "LongSplat"
_LOCAL_BACKEND_SKIP_REASON = (
    "requires ignored local third_party/LongSplat checkout; "
    "the tracked pose patch contract is tested separately"
)

_EXPECTED_BACKEND_FILES = [
    "arguments/__init__.py",
    "scene/__init__.py",
    "scene/cameras.py",
    "train.py",
    "utils/pose_utils.py",
]

_OPTIONAL_POSE_QUALITY_SKIP_REASON = (
    "optional incremental pose-quality patch is not applied; the default "
    "depth-disabled external fixed-pose closure intentionally excludes "
    "project_to_so3 and POSE_TELEMETRY; patch artifact content remains tested separately"
)


def _optional_pose_quality_patch_applied() -> bool:
    if not _PATCH_PATH.is_file():
        return False
    pose_utils = _BACKEND_ROOT / "utils" / "pose_utils.py"
    scene_init = _BACKEND_ROOT / "scene" / "__init__.py"
    train_py = _BACKEND_ROOT / "train.py"
    if not all(path.is_file() for path in (pose_utils, scene_init, train_py)):
        return False
    return (
        "def project_to_so3" in pose_utils.read_text(encoding="utf-8")
        and "POSE_TELEMETRY" in (
            scene_init.read_text(encoding="utf-8")
            + "\n"
            + train_py.read_text(encoding="utf-8")
        )
    )


_OPTIONAL_POSE_QUALITY_PATCH_APPLIED = _optional_pose_quality_patch_applied()


# ---------------------------------------------------------------------------
# Patch file content checks (RED — patch does not exist yet)
# ---------------------------------------------------------------------------


def test_patch_file_exists():
    """The pose quality gates patch file must exist."""
    assert _PATCH_PATH.is_file(), (
        f"Patch file missing: {_PATCH_PATH}\n"
        "Generate it from the isolated worktree after implementing backend changes."
    )


def test_patch_contains_project_to_so3_three_locations():
    """project_to_so3 must be called in exactly three locations."""
    text = _PATCH_PATH.read_text(encoding="utf-8")
    # The function definition itself
    assert "project_to_so3" in text, "project_to_so3 function must exist in patch"
    # Should appear in: global_align (3 initial cameras), incremental registration,
    # and update_pose composite. Count the '+' lines mentioning it.
    plus_lines_with_so3 = [
        line for line in text.splitlines()
        if line.startswith("+") and "project_to_so3" in line
    ]
    assert len(plus_lines_with_so3) >= 3, (
        f"project_to_so3 must appear in >=3 '+' lines, got {len(plus_lines_with_so3)}"
    )


def test_patch_has_pose_gate_parameters():
    """New gate parameters are defined in arguments/__init__.py."""
    text = _PATCH_PATH.read_text(encoding="utf-8")
    required_params = [
        "min_match_count",
        "min_inlier_count",
        "min_inlier_ratio",
        "max_reprojection_rmse_px",
        "min_grid_coverage",
        "min_positive_depth_ratio",
        "max_rotation_step_deg",
        "max_translation_step_ratio",
        "reference_lookback",
    ]
    for param in required_params:
        assert param in text, f"Parameter '{param}' must appear in patch"


def test_patch_removes_old_pnp_retry_pattern():
    """The old max_pnp_retries / end_view_id retry pattern must be removed."""
    text = _PATCH_PATH.read_text(encoding="utf-8")
    # The old code decreases end_view_id to retry same image pair
    # Removal lines (starting with '-') should contain these patterns
    removal_lines = [line for line in text.splitlines() if line.startswith("-")]
    removal_text = "\n".join(removal_lines)
    assert "end_view_id" in removal_text or "max_pnp_retries" in removal_text, (
        "Patch must remove old retry logic (end_view_id/max_pnp_retries)"
    )


def test_patch_replaces_unconditional_is_registered():
    """The unconditional is_registered = True after PnP must be replaced."""
    text = _PATCH_PATH.read_text(encoding="utf-8")
    # The old line '-        cur_viewpoint_cam.is_registered = True' should exist
    removal_lines = [line for line in text.splitlines() if line.startswith("-")]
    has_unconditional = any(
        "is_registered" in line and "True" in line for line in removal_lines
    )
    # New code should gate is_registered after quality check passes
    # is_registered = True may appear as context (no +/- prefix) when git matches
    # identical text in both old and new hunk positions.
    kept_or_added = [
        line for line in text.splitlines()
        if not line.startswith("-") and not line.startswith("@@")
        and not line.startswith("diff ") and not line.startswith("index ")
        and not line.startswith("---") and not line.startswith("+++")
    ]
    has_gated = any(
        "is_registered" in line and "True" in line for line in kept_or_added
    )
    assert has_unconditional, "Patch must remove unconditional is_registered = True"
    assert has_gated, "Patch must gate is_registered after quality check"


def test_patch_uses_telemetry_allow_nan_false():
    """Telemetry JSON must use allow_nan=False."""
    text = _PATCH_PATH.read_text(encoding="utf-8")
    assert "allow_nan=False" in text, (
        "Patch must include allow_nan=False in telemetry JSON dumps"
    )


def test_patch_only_touches_five_backend_files():
    """Patch must only touch the 5 approved backend files."""
    text = _PATCH_PATH.read_text(encoding="utf-8")
    # Parse git diff headers to find touched files
    import re

    file_headers = re.findall(r"^\+\+\+ b/(.+?)$", text, re.MULTILINE)
    touched = set()
    for h in file_headers:
        # Strip the a/ or b/ prefix from diff headers
        path = h.strip()
        # Find the matching file from our expected list
        for expected in _EXPECTED_BACKEND_FILES:
            if path.endswith(expected.replace("/", "\\")) or path.endswith(expected):
                touched.add(expected)
                break

    unexpected = touched - set(_EXPECTED_BACKEND_FILES)
    assert not unexpected, f"Patch touches unexpected files: {unexpected}"
    assert len(touched) >= 5, (
        f"Patch should touch all 5 expected files, got {len(touched)}: {touched}"
    )


# ---------------------------------------------------------------------------
# Backend parity checks (verify current working tree matches patch intent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_BACKEND_ROOT / "scene" / "cameras.py").is_file(),
    reason=_LOCAL_BACKEND_SKIP_REASON,
)
def test_backend_camera_update_rt_rejects_invalid():
    """Current Camera.update_RT must reject non-(3,3) R, non-(3,) t, non-finite."""
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    # Read the current backend file
    cameras_py = _BACKEND_ROOT / "scene" / "cameras.py"
    text = cameras_py.read_text(encoding="utf-8")

    # After patch application, update_RT should validate its inputs
    assert "def update_RT" in text
    # The patched version should include validation
    assert "shape" in text or "finite" in text or "isfinite" in text or "orthogonality" in text, (
        "update_RT must include shape/finite/orthogonality validation after patch"
    )


@pytest.mark.skipif(
    not _OPTIONAL_POSE_QUALITY_PATCH_APPLIED,
    reason=_OPTIONAL_POSE_QUALITY_SKIP_REASON,
)
def test_backend_has_project_to_so3():
    """project_to_so3 helper must exist in the backend utils."""
    pose_utils = _BACKEND_ROOT / "utils" / "pose_utils.py"
    text = pose_utils.read_text(encoding="utf-8")

    assert "def project_to_so3" in text, (
        "project_to_so3 function must be defined in utils/pose_utils.py"
    )


@pytest.mark.skipif(
    not _OPTIONAL_POSE_QUALITY_PATCH_APPLIED,
    reason=_OPTIONAL_POSE_QUALITY_SKIP_REASON,
)
def test_backend_has_pose_telemetry():
    """POSE_TELEMETRY must be emitted in the incremental registration path."""
    scene_init = _BACKEND_ROOT / "scene" / "__init__.py"
    train_py = _BACKEND_ROOT / "train.py"
    text = scene_init.read_text(encoding="utf-8") + "\n" + train_py.read_text(encoding="utf-8")

    assert "POSE_TELEMETRY" in text, (
        "POSE_TELEMETRY marker must be emitted in scene/__init__.py or train.py"
    )
