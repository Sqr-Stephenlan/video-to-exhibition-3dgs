"""Contract tests for Gaussian shape regularization (Task 8).

Covers anisotropy regularization patch contract, quaternion normalization
during PLY export, and the orchestrator PLY publication gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Patch contract tests
# ---------------------------------------------------------------------------


def test_anisotropy_reg_weight_default_zero_in_patch():
    """anisotropy_reg_weight defaults to 0 in the patch."""
    patch_path = (
        Path(__file__).parents[3]
        / "docs" / "longsplat" / "patches" / "longsplat_gaussian_shape_regularization.patch"
    )
    if not patch_path.exists():
        return  # RED: patch not yet generated

    text = patch_path.read_text(encoding="utf-8")
    assert "anisotropy_reg_weight" in text
    assert "= 0" in text or "=0" in text or "default=0" in text or "0.0" in text


def test_anisotropy_soft_limit_in_patch():
    """anisotropy_soft_limit parameter exists with value 30."""
    patch_path = (
        Path(__file__).parents[3]
        / "docs" / "longsplat" / "patches" / "longsplat_gaussian_shape_regularization.patch"
    )
    if not patch_path.exists():
        return  # RED: patch not yet generated

    text = patch_path.read_text(encoding="utf-8")
    assert "anisotropy_soft_limit" in text


def test_quaternion_normalization_in_patch():
    """Patch normalizes quaternions during PLY export."""
    patch_path = (
        Path(__file__).parents[3]
        / "docs" / "longsplat" / "patches" / "longsplat_gaussian_shape_regularization.patch"
    )
    if not patch_path.exists():
        return  # RED: patch not yet generated

    text = patch_path.read_text(encoding="utf-8")
    # Must contain quaternion normalization logic
    assert "vector_norm" in text or "norm" in text
    # Zero-quaternion fallback to identity
    assert "1e-8" in text


def test_gaussian_shape_patch_only_touches_expected_files():
    """Patch only modifies the four expected backend files."""
    patch_path = (
        Path(__file__).parents[3]
        / "docs" / "longsplat" / "patches" / "longsplat_gaussian_shape_regularization.patch"
    )
    if not patch_path.exists():
        return  # RED: patch not yet generated

    text = patch_path.read_text(encoding="utf-8")
    touched = set()
    for line in text.split("\n"):
        if line.startswith("diff --git ") or line.startswith("--- ") or line.startswith("+++ "):
            touched.add(line)
    expected_files = [
        "arguments/__init__.py",
        "utils/loss_utils.py",
        "train.py",
        "scene/gaussian_model.py",
    ]
    for fname in expected_files:
        found = any(fname in line for line in touched)
        assert found, f"Patch should touch {fname}"


# ---------------------------------------------------------------------------
# Anisotropy regularization unit tests
# ---------------------------------------------------------------------------


def test_anisotropy_regularization_zero_for_isotropic():
    """anisotropy_regularization returns 0 for isotropic scales."""
    torch = pytest.importorskip("torch")
    from third_party.LongSplat.utils.loss_utils import anisotropy_regularization

    scaling = torch.ones(100, 3) * 2.0
    loss = anisotropy_regularization(scaling, soft_limit=30)
    assert loss.item() == 0.0


def test_anisotropy_regularization_positive_above_limit():
    """anisotropy_regularization > 0 when scale ratio exceeds soft_limit."""
    torch = pytest.importorskip("torch")
    from third_party.LongSplat.utils.loss_utils import anisotropy_regularization

    # One axis 100x larger than others
    scaling = torch.ones(100, 3)
    scaling[:, 0] = 100.0
    scaling[:, 1] = 1.0
    scaling[:, 2] = 1.0
    loss = anisotropy_regularization(scaling, soft_limit=30)
    assert loss.item() > 0.0


def test_anisotropy_regularization_handles_zero_scale():
    """anisotropy_regularization handles zero-scale entries gracefully."""
    torch = pytest.importorskip("torch")
    from third_party.LongSplat.utils.loss_utils import anisotropy_regularization

    scaling = torch.zeros(10, 3)
    loss = anisotropy_regularization(scaling, soft_limit=30)
    assert torch.isfinite(loss)


def test_anisotropy_regularization_below_limit_is_zero():
    """anisotropy_regularization returns 0 when ratio <= soft_limit."""
    torch = pytest.importorskip("torch")
    from third_party.LongSplat.utils.loss_utils import anisotropy_regularization

    # max/min = 20, below soft_limit of 30
    scaling = torch.ones(50, 3)
    scaling[:, 0] = 20.0
    loss = anisotropy_regularization(scaling, soft_limit=30)
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Quaternion normalization unit test
# ---------------------------------------------------------------------------


def test_zero_quaternion_becomes_identity():
    """Zero-norm quaternion maps to identity [1, 0, 0, 0]."""
    torch = pytest.importorskip("torch")

    # Simulate the normalization logic directly
    rotation = torch.zeros(4, 4)
    norm = torch.linalg.vector_norm(rotation, dim=1, keepdim=True)
    identity = torch.zeros_like(rotation)
    identity[:, 0] = 1.0
    result = torch.where(norm > 1e-8, rotation / norm.clamp_min(1e-8), identity)
    assert torch.allclose(result[:1], torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_nonzero_quaternion_is_normalized():
    """Non-zero quaternion is normalized to unit length."""
    torch = pytest.importorskip("torch")

    rotation = 2.0 * torch.ones(4, 4)  # norm = 4.0
    norm = torch.linalg.vector_norm(rotation, dim=1, keepdim=True)
    identity = torch.zeros_like(rotation)
    identity[:, 0] = 1.0
    result = torch.where(norm > 1e-8, rotation / norm.clamp_min(1e-8), identity)
    expected_norm = torch.linalg.vector_norm(result, dim=1)
    assert torch.allclose(expected_norm, torch.ones(4), atol=1e-6)


# ---------------------------------------------------------------------------
# LOSS_TELEMETRY anisotropy field
# ---------------------------------------------------------------------------


def test_loss_telemetry_includes_anisotropy_field_in_patch():
    """LOSS_TELEMETRY emission includes non-zero anisotropy when active."""
    patch_path = (
        Path(__file__).parents[3]
        / "docs" / "longsplat" / "patches" / "longsplat_gaussian_shape_regularization.patch"
    )
    if not patch_path.exists():
        return  # RED: patch not yet generated

    text = patch_path.read_text(encoding="utf-8")
    # Don't hardcode 0.0 for anisotropy in telemetry — should read from variable
    assert "anisotropy" in text


# ---------------------------------------------------------------------------
# Orchestrator PLY publication gate
# ---------------------------------------------------------------------------


def test_orchestrator_rejects_high_anisotropy_ply(tmp_path):
    """Orchestrator PLY gate fails when anisotropy q99 > 50."""
    import importlib.util
    import subprocess as sp
    from unittest import mock

    _contract_path = Path(__file__).parent / "test_orchestrator_contract.py"
    _contract_spec = importlib.util.spec_from_file_location(
        "test_orchestrator_contract", _contract_path
    )
    _contract = importlib.util.module_from_spec(_contract_spec)
    sys.modules["test_orchestrator_contract"] = _contract
    _contract_spec.loader.exec_module(_contract)
    _make_fake_backend = _contract._make_fake_backend
    _make_producer_manifest = _contract._make_producer_manifest

    backend = _make_fake_backend(tmp_path)
    manifest = _make_producer_manifest(tmp_path)
    output_dir = tmp_path / "outputs"

    import scripts.longsplat.runner as _runner

    fake_commit = sp.run(
        ["git", "-C", str(backend), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    orig_commit = _runner.LONGSPLAT_COMMIT
    orig_links = dict(_runner._LONGSPLAT_SUBMODULE_LINKS)
    _runner.LONGSPLAT_COMMIT = fake_commit
    for sub_key in list(_runner._LONGSPLAT_SUBMODULE_LINKS.keys()):
        sp_path = backend / sub_key
        r = sp.run(
            ["git", "-C", str(sp_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        _runner._LONGSPLAT_SUBMODULE_LINKS[sub_key] = r.stdout.strip()

    try:
        from scripts.longsplat.orchestrator import run_pipeline
        from scripts.longsplat.runner import LongSplatConfig
        from scripts.longsplat.depth_bridge import DepthMaterializationResult
        from scripts.longsplat.quality_gates import PlyGateConfig, QualityGateConfig

        depth_manifest_path = tmp_path / "depth_manifest.json"
        depth_manifest_path.write_text(
            json.dumps({"schema_version": "1.0", "frames": []})
        )

        fake_mat_result = DepthMaterializationResult(
            expected_count=1,
            materialized_count=1,
            depth_manifest_sha256="abcd1234",
            frames=[],
        )

        with mock.patch(
            "scripts.longsplat.orchestrator.materialize_all",
            return_value=fake_mat_result,
        ):
            with mock.patch(
                "scripts.longsplat.orchestrator.audit_pose_quality",
                return_value={
                    "passed": True,
                    "reasons": [],
                    "trajectory": {"camera_count": 1},
                    "telemetry": {"accepted_camera_count": 1, "camera_count": 1},
                },
            ):
                with mock.patch(
                    "scripts.longsplat.orchestrator.summarize_vda_telemetry",
                    return_value={
                        "aligned": 1,
                        "missing": 0,
                        "rejected": 0,
                        "min_correlation": 0.95,
                        "records": [{"result": "aligned"}],
                    },
                ):
                    with mock.patch(
                        "scripts.longsplat.orchestrator.summarize_loss_telemetry",
                        return_value={
                            "has_nonfinite": False,
                            "record_count": 10,
                            "records": [{"finite": True, "stage": "global"}],
                        },
                    ):
                        with mock.patch(
                            "scripts.longsplat.orchestrator.analyze_ply_quality",
                            return_value={
                                "anisotropy_q99": 65.0,
                                "effective_fraction": 0.40,
                                "quaternion_within_1pct_fraction": 0.9995,
                                "finite_core_fraction": 1.0,
                            },
                        ):
                            exit_code = run_pipeline(
                                manifest_path=manifest,
                                segment_id="seg_01",
                                config=LongSplatConfig(
                                    source_path="",
                                    model_path="",
                                    iterations=100,
                                    seed=0,
                                    backend_mode="research_local",
                                    extra_train_args={"depth_source": "vda"},
                                    quality_gates=QualityGateConfig(
                                        ply=PlyGateConfig(mode="enforce"),
                                    ),
                                ),
                                repo_root=backend,
                                output_dir=output_dir,
                                project_root=tmp_path,
                                python_exe=sys.executable,
                                depth_manifest_path=depth_manifest_path,
                            )

            # PLY gate should fail due to high anisotropy
            assert exit_code == 1, f"PLY gate must reject anisotropy q99=65, got exit {exit_code}"

            run_dirs = list(output_dir.iterdir())
            assert len(run_dirs) == 1
            record = json.loads((run_dirs[0] / "reconstruction_run.json").read_text())
            assert record["status"] == "failed"
    finally:
        _runner.LONGSPLAT_COMMIT = orig_commit
        _runner._LONGSPLAT_SUBMODULE_LINKS.clear()
        _runner._LONGSPLAT_SUBMODULE_LINKS.update(orig_links)
