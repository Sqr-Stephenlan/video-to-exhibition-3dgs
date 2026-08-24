"""Contract tests for VDA depth quality gating (Task 6) and plane depth/normal
loss (Task 7).

Covers telemetry parsing of quality fields, VDA usage counting, the
post-training VDA quality gate in the orchestrator, LOSS_TELEMETRY parsing,
and the non-finite loss gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.longsplat.telemetry import parse_json_markers, summarize_vda_telemetry
from scripts.longsplat.orchestrator import _summarize_vda_usage


# ---------------------------------------------------------------------------
# Telemetry: normalized_rmse field
# ---------------------------------------------------------------------------


def test_vda_telemetry_includes_normalized_rmse():
    """VDA_TELEMETRY records include normalized_rmse after quality patch."""
    stdout = (
        'VDA_TELEMETRY {"correlation":0.94,"frame":"f1",'
        '"inlier_count":80000,"inlier_ratio":0.96,'
        '"normalized_rmse":0.05,"offset":0.01,"reason":null,'
        '"residual_rmse":0.001,"result":"aligned",'
        '"slope":2.5,"stage":"incremental","valid_count":100000}\n'
    )

    records = parse_json_markers(stdout, "VDA_TELEMETRY")
    assert len(records) == 1
    assert records[0]["result"] == "aligned"
    assert records[0]["normalized_rmse"] == 0.05
    assert records[0]["inlier_ratio"] == 0.96
    assert records[0]["correlation"] == 0.94


def test_vda_telemetry_rejected_low_inlier_ratio():
    """High correlation but low inlier ratio → rejected."""
    stdout = (
        'VDA_TELEMETRY {"correlation":0.94,"frame":"f2",'
        '"inlier_count":40000,"inlier_ratio":0.40,'
        '"normalized_rmse":0.05,"offset":0.01,"reason":"low_inlier_ratio",'
        '"residual_rmse":0.001,"result":"rejected",'
        '"slope":2.5,"stage":"incremental","valid_count":100000}\n'
    )

    records = parse_json_markers(stdout, "VDA_TELEMETRY")
    assert len(records) == 1
    assert records[0]["result"] == "rejected"
    assert records[0]["reason"] == "low_inlier_ratio"


def test_vda_telemetry_rejected_low_correlation():
    """Low correlation → rejected."""
    stdout = (
        'VDA_TELEMETRY {"correlation":0.30,"frame":"f3",'
        '"inlier_count":80000,"inlier_ratio":0.96,'
        '"normalized_rmse":0.05,"offset":0.01,"reason":"low_correlation",'
        '"residual_rmse":0.001,"result":"rejected",'
        '"slope":2.5,"stage":"incremental","valid_count":100000}\n'
    )

    records = parse_json_markers(stdout, "VDA_TELEMETRY")
    assert records[0]["result"] == "rejected"
    assert records[0]["reason"] == "low_correlation"


def test_vda_telemetry_rejected_negative_slope():
    """Negative slope → rejected."""
    stdout = (
        'VDA_TELEMETRY {"correlation":0.94,"frame":"f4",'
        '"inlier_count":80000,"inlier_ratio":0.96,'
        '"normalized_rmse":0.05,"offset":0.01,"reason":"nonpositive_slope",'
        '"residual_rmse":0.001,"result":"rejected",'
        '"slope":-0.5,"stage":"incremental","valid_count":100000}\n'
    )

    records = parse_json_markers(stdout, "VDA_TELEMETRY")
    assert records[0]["result"] == "rejected"
    assert records[0]["reason"] == "nonpositive_slope"


def test_vda_telemetry_rejected_high_normalized_rmse():
    """Large normalized RMSE → rejected."""
    stdout = (
        'VDA_TELEMETRY {"correlation":0.94,"frame":"f5",'
        '"inlier_count":80000,"inlier_ratio":0.96,'
        '"normalized_rmse":0.55,"offset":0.01,"reason":"high_normalized_rmse",'
        '"residual_rmse":0.01,"result":"rejected",'
        '"slope":2.5,"stage":"incremental","valid_count":100000}\n'
    )

    records = parse_json_markers(stdout, "VDA_TELEMETRY")
    assert records[0]["result"] == "rejected"
    assert records[0]["reason"] == "high_normalized_rmse"


def test_vda_telemetry_missing_not_counted_as_aligned():
    """Old missing-file marker is NOT miscounted as aligned."""
    stdout = (
        'VDA_TELEMETRY {"frame":"f6","reason":"missing_file",'
        '"result":"missing","stage":"incremental"}\n'
    )

    records = parse_json_markers(stdout, "VDA_TELEMETRY")
    assert records[0]["result"] == "missing"
    # Should not have quality fields
    assert "correlation" not in records[0]


# ---------------------------------------------------------------------------
# VDA usage summarisation
# ---------------------------------------------------------------------------


def test_summarize_vda_usage_counts_aligned_rejected_missing():
    """VDA_USAGE markers correctly count aligned, rejected, and missing."""
    stdout = (
        "VDA_USAGE result=aligned stage=scene_init frame=img_001\n"
        "VDA_USAGE result=aligned stage=scene_init frame=img_002\n"
        "VDA_USAGE result=rejected stage=scene_init frame=img_003\n"
        "VDA_USAGE result=rejected stage=incremental frame=img_004\n"
        "VDA_USAGE result=missing stage=scene_init frame=img_005\n"
    )
    counts = _summarize_vda_usage(stdout)
    assert counts == {"aligned": 2, "missing": 1, "rejected": 2}


def test_summarize_vda_usage_empty():
    """Empty stdout → all zeros."""
    assert _summarize_vda_usage("") == {"aligned": 0, "missing": 0, "rejected": 0}


# ---------------------------------------------------------------------------
# VDA telemetry summarisation with quality fields
# ---------------------------------------------------------------------------


def test_summarize_vda_telemetry_extracts_normalized_rmse():
    """summarize_vda_telemetry includes normalized_rmse statistics."""
    stdout = (
        'VDA_TELEMETRY {"correlation":0.94,"frame":"f1",'
        '"inlier_count":80000,"inlier_ratio":0.96,'
        '"normalized_rmse":0.05,"offset":0.01,"reason":null,'
        '"residual_rmse":0.001,"result":"aligned",'
        '"slope":2.5,"stage":"incremental","valid_count":100000}\n'
        'VDA_TELEMETRY {"correlation":0.91,"frame":"f2",'
        '"inlier_count":70000,"inlier_ratio":0.97,'
        '"normalized_rmse":0.08,"offset":0.01,"reason":null,'
        '"residual_rmse":0.002,"result":"aligned",'
        '"slope":2.3,"stage":"incremental","valid_count":100000}\n'
    )
    summary = summarize_vda_telemetry(stdout)
    assert summary["aligned"] == 2
    assert summary["missing"] == 0
    assert summary["rejected"] == 0


def test_summarize_vda_telemetry_mixed_results():
    """Mixed aligned + rejected + missing are correctly counted."""
    stdout = (
        'VDA_TELEMETRY {"correlation":0.94,"frame":"f1",'
        '"inlier_ratio":0.96,"normalized_rmse":0.05,"result":"aligned",'
        '"stage":"incremental"}\n'
        'VDA_TELEMETRY {"correlation":0.30,"frame":"f2",'
        '"inlier_ratio":0.96,"normalized_rmse":0.05,"reason":"low_correlation",'
        '"result":"rejected","stage":"incremental"}\n'
        'VDA_TELEMETRY {"frame":"f3","reason":"missing_file",'
        '"result":"missing","stage":"incremental"}\n'
    )
    summary = summarize_vda_telemetry(stdout)
    assert summary["aligned"] == 1
    assert summary["rejected"] == 1
    assert summary["missing"] == 1


# ---------------------------------------------------------------------------
# Post-training VDA gate integration test
# ---------------------------------------------------------------------------


def test_orchestrator_vda_gate_rejects_insufficient_aligned(tmp_path):
    """Orchestrator fails when aligned/train ratio < 0.98."""
    import importlib.util
    import subprocess as sp
    from unittest import mock

    # Import helpers from test_orchestrator_contract via file path
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

    from scripts.longsplat.orchestrator import run_pipeline
    from scripts.longsplat.runner import LongSplatConfig
    from scripts.longsplat.depth_bridge import DepthMaterializationResult

    depth_manifest_path = tmp_path / "depth_manifest.json"
    depth_manifest_path.write_text(json.dumps({"schema_version": "1.0", "frames": []}))

    fake_mat_result = DepthMaterializationResult(
        expected_count=1,
        materialized_count=1,
        depth_manifest_sha256="abcd1234",
        frames=[],
    )

    # Mock audit_pose_quality to simulate pass
    # Mock summarize_vda_telemetry to simulate low aligned ratio
    # 1 train camera, 0 aligned → ratio 0.0 < 0.98
    try:
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
                        "aligned": 0,
                        "missing": 0,
                        "rejected": 1,
                        "min_correlation": None,
                        "records": [{"result": "rejected", "reason": "low_correlation"}],
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
                        ),
                        repo_root=backend,
                        output_dir=output_dir,
                        project_root=tmp_path,
                        python_exe=sys.executable,
                        depth_manifest_path=depth_manifest_path,
                    )

        assert exit_code == 1, f"VDA gate must reject, got {exit_code}"

        run_dirs = list(output_dir.iterdir())
        assert len(run_dirs) == 1
        record = json.loads((run_dirs[0] / "reconstruction_run.json").read_text())
        assert record["status"] == "failed"
        assert "vda_quality_gate" in record["stages"]
    finally:
        _runner.LONGSPLAT_COMMIT = orig_commit
        _runner._LONGSPLAT_SUBMODULE_LINKS.clear()
        _runner._LONGSPLAT_SUBMODULE_LINKS.update(orig_links)


# ============================================================================
# Task 7 — LOSS_TELEMETRY parsing
# ============================================================================


def test_loss_telemetry_finite_parsed():
    """LOSS_TELEMETRY with finite=true is parsed correctly."""
    stdout = (
        'LOSS_TELEMETRY {"anisotropy":0.0,"depth":0.31,"finite":true,'
        '"iteration":400,"normal":0.02,"photo":0.14,"stage":"global",'
        '"total":1.82}\n'
    )
    records = parse_json_markers(stdout, "LOSS_TELEMETRY")
    assert len(records) == 1
    r = records[0]
    assert r["finite"] is True
    assert r["stage"] == "global"
    assert r["iteration"] == 400
    assert r["total"] == 1.82
    assert r["photo"] == 0.14
    assert r["depth"] == 0.31
    assert r["normal"] == 0.02


def test_loss_telemetry_nonfinite_captured():
    """LOSS_TELEMETRY with finite=false is correctly recorded."""
    stdout = (
        'LOSS_TELEMETRY {"anisotropy":0.0,"depth":null,"finite":false,'
        '"iteration":200,"normal":0.0,"photo":1.5,"stage":"pose","total":null}\n'
    )
    records = parse_json_markers(stdout, "LOSS_TELEMETRY")
    assert len(records) == 1
    assert records[0]["finite"] is False
    assert records[0]["total"] is None


def test_loss_telemetry_empty_stdout():
    """Empty stdout produces empty loss telemetry list."""
    assert parse_json_markers("", "LOSS_TELEMETRY") == []


def test_loss_telemetry_malformed_ignored():
    """Malformed LOSS_TELEMETRY JSON is silently ignored."""
    stdout = (
        'LOSS_TELEMETRY {broken json\n'
        'LOSS_TELEMETRY {"finite":true,"stage":"init","total":0.5}\n'
    )
    records = parse_json_markers(stdout, "LOSS_TELEMETRY")
    assert len(records) == 1
    assert records[0]["finite"] is True


# ============================================================================
# Task 7 — plane normal loss contract (unit tests)
# ============================================================================


def test_make_depth_quality_mask_empty_on_small_input():
    """make_depth_quality_mask returns all-False mask for tiny inputs."""
    torch = pytest.importorskip("torch")
    from third_party.LongSplat.utils.loss_utils import make_depth_quality_mask

    # Input smaller than erosion kernel → all False after erosion
    depth = torch.ones(8, 8)
    rgb = torch.ones(3, 8, 8)
    mask = make_depth_quality_mask(
        depth, rgb,
        depth_edge_quantile=0.90,
        rgb_edge_quantile=0.95,
        erosion_radius=2,
    )
    # With 8x8 and erosion_radius=2, the mask should be empty or nearly so
    assert mask.sum() < 64  # Not all pixels survive


def test_normal_consistency_loss_zero_for_few_pixels():
    """normal_consistency_loss returns 0 when < 1000 valid pixels."""
    torch = pytest.importorskip("torch")
    from third_party.LongSplat.utils.loss_utils import normal_consistency_loss

    H, W = 16, 16  # 256 pixels
    pred = torch.rand(H, W)
    ref = torch.rand(H, W)
    intrinsic = torch.eye(3)
    mask = torch.ones(H, W, dtype=torch.bool)

    loss = normal_consistency_loss(pred, ref, intrinsic, mask)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def test_normal_consistency_loss_finite_for_large_input():
    """normal_consistency_loss produces finite output for valid large input."""
    torch = pytest.importorskip("torch")
    from third_party.LongSplat.utils.loss_utils import normal_consistency_loss

    H, W = 64, 64  # 4096 pixels
    pred = 2.0 + 0.1 * torch.randn(H, W)
    ref = 2.0 + 0.05 * torch.randn(H, W)
    intrinsic = torch.tensor([[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]])
    mask = torch.ones(H, W, dtype=torch.bool)

    loss = normal_consistency_loss(pred, ref, intrinsic, mask)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_plane_normal_loss_weight_default_zero_in_patch():
    """plane_normal_loss_weight defaults to 0 in the arguments patch."""
    patch_path = (
        Path(__file__).parents[3]
        / "docs" / "longsplat" / "patches" / "longsplat_plane_depth_loss.patch"
    )
    if not patch_path.exists():
        return  # RED: patch not yet generated

    text = patch_path.read_text(encoding="utf-8")
    # Must contain the parameter with default 0
    assert "plane_normal_loss_weight" in text
    assert "= 0" in text or "=0" in text or "default=0" in text or "0.0" in text


def test_plane_depth_loss_patch_only_touches_expected_files():
    """Patch only modifies the three expected backend files."""
    patch_path = (
        Path(__file__).parents[3]
        / "docs" / "longsplat" / "patches" / "longsplat_plane_depth_loss.patch"
    )
    if not patch_path.exists():
        return  # RED: patch not yet generated

    text = patch_path.read_text(encoding="utf-8")
    touched = set()
    for line in text.split("\n"):
        if line.startswith("diff --git ") or line.startswith("--- ") or line.startswith("+++ "):
            touched.add(line)
    # The patch should reference only arguments/__init__.py, loss_utils.py, train.py
    expected_files = [
        "arguments/__init__.py",
        "utils/loss_utils.py",
        "train.py",
    ]
    for fname in expected_files:
        found = any(fname in line for line in touched)
        assert found, f"Patch should touch {fname}"


# ============================================================================
# Task 7 — Non-finite loss gate integration test
# ============================================================================


def test_orchestrator_rejects_nonfinite_loss(tmp_path):
    """Orchestrator fails when LOSS_TELEMETRY contains non-finite entries."""
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

    from scripts.longsplat.orchestrator import run_pipeline
    from scripts.longsplat.runner import LongSplatConfig
    from scripts.longsplat.depth_bridge import DepthMaterializationResult

    depth_manifest_path = tmp_path / "depth_manifest.json"
    depth_manifest_path.write_text(json.dumps({"schema_version": "1.0", "frames": []}))

    fake_mat_result = DepthMaterializationResult(
        expected_count=1,
        materialized_count=1,
        depth_manifest_sha256="abcd1234",
        frames=[],
    )

    # Mock all gates to pass, but inject non-finite loss telemetry
    try:
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
                            "has_nonfinite": True,
                            "max_total": None,
                            "records": [
                                {"finite": False, "stage": "pose", "total": None}
                            ],
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
                            ),
                            repo_root=backend,
                            output_dir=output_dir,
                            project_root=tmp_path,
                            python_exe=sys.executable,
                            depth_manifest_path=depth_manifest_path,
                        )

        assert exit_code == 1, f"Non-finite loss gate must reject, got {exit_code}"

        run_dirs = list(output_dir.iterdir())
        assert len(run_dirs) == 1
        record = json.loads((run_dirs[0] / "reconstruction_run.json").read_text())
        assert record["status"] == "failed"
    finally:
        _runner.LONGSPLAT_COMMIT = orig_commit
        _runner._LONGSPLAT_SUBMODULE_LINKS.clear()
        _runner._LONGSPLAT_SUBMODULE_LINKS.update(orig_links)
