from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from scripts.longsplat.conversion_executor import execute_conversion, execute_evaluation
from tests.unit.test_authority_manifest import _fixture


def _write_gaussian_ply(path: Path, vertices: int = 2) -> Path:
    """Write a minimal finite standard-converted 3DGS PLY (62 float32 fields).

    Matches the write-side contract validated by convert.validate_converted_ply:
    all 14 core attributes finite, a non-degenerate quaternion (norm 1), plus
    nx/ny/nz and the 45 SH f_rest_* fields that make up the standard layout.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += ["f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(45)]
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    assert len(names) == 62, len(names)
    dtype = np.dtype([(name, np.float32) for name in names])
    data = np.zeros(vertices, dtype=dtype)
    for name in names:
        data[name] = 0.01
    data["x"] = np.array([0.0, 1.0], dtype=np.float32)[:vertices]
    data["y"] = np.array([0.0, 0.0], dtype=np.float32)[:vertices]
    data["z"] = np.array([0.0, 0.0], dtype=np.float32)[:vertices]
    data["rot_0"] = 1.0
    element = PlyElement.describe(data, "vertex")
    PlyData([element], text=True).write(str(path))
    return path


def _seed_model_snapshot_inputs(route: Path, name: str) -> None:
    """Add the snapshot-ready source files the generic fixture omits (cfg_args)."""
    model = route / "outputs" / name / "training_model"
    (model / "cfg_args").write_text("--sh_degree 3\n", encoding="utf-8")


def _snapshot_ply(conversion_root: Path) -> Path:
    return conversion_root / "conversion_model_snapshot" / "converted_3dgs" / "point_cloud.ply"


class _CapturingRunner:
    """subprocess.run stand-in that records argv and returns exit code 0."""

    def __init__(self, conversion_root: Path) -> None:
        self.calls: list[list[str]] = []
        self.conversion_root = Path(conversion_root)

    def __call__(self, command, **_kwargs):
        self.calls.append(list(command))
        # Simulate the child converter emitting its technical-pass output so
        # execute_conversion can validate it (Bug-B write side).
        _write_gaussian_ply(_snapshot_ply(self.conversion_root))
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def _write_conversion_evidence(conversion_root: Path) -> Path:
    """Write a minimal technical-pass conversion evidence root (Bug-A/B fixture)."""
    conversion_model_snapshot = conversion_root / "conversion_model_snapshot"
    converted_ply = _write_gaussian_ply(conversion_model_snapshot / "converted_3dgs" / "point_cloud.ply")
    (conversion_root / "conversion_result.json").write_text(
        json.dumps(
            {
                "schema_version": "longsplat-generic-conversion-result-v1",
                "stage": "conversion",
                "STRUCTURAL_CONVERSION_PASS": True,
                "structural_pass": True,
                # Authoritative write-side key (written by convert.py);
                # converted_ply_path is intentionally absent to prove the read
                # side consumes structural.path (Bug B).
                "structural": {"path": str(converted_ply)},
                "snapshot_model_path": str(conversion_model_snapshot),
            }
        ),
        encoding="utf-8",
    )
    return converted_ply


def test_dry_run_evaluation_reads_conversion_evidence_from_conversion_root(tmp_path: Path) -> None:
    """Bug A: eval reads conversion_result/snapshot from conversion root, not its own root."""
    _manifest, manifest_path, route = _fixture(
        tmp_path,
        "buga-dry",
        camera_names=["first", "second"],
        width=29,
        height=17,
    )
    conversion_root = route / "outputs" / "buga-dry" / "conversion-attempt"
    conversion_root.mkdir(parents=True)
    converted_ply = _write_conversion_evidence(conversion_root)

    # Distinct eval evidence root; no conversion_result.json / snapshot here.
    eval_root = route / "outputs" / "buga-dry" / "eval-attempt"
    eval_root.mkdir(parents=True)

    result = execute_evaluation(
        authority_manifest_path=manifest_path,
        evidence_root=eval_root,
        route_root=route,
        conversion_evidence_root=conversion_root,
        dry_run=True,
        validate_only=False,
    )

    assert result["validated"] is True
    assert result["stage"] == "converted-eval"
    request = result["request"]
    assert request["converted_ply_path"] == str(converted_ply)
    assert "--ply" in request["argv"]
    assert request["argv"][request["argv"].index("--ply") + 1] == str(converted_ply)
    assert request["snapshot_model_path"] == str(conversion_root / "conversion_model_snapshot")
    # The eval root independently holds no conversion evidence: the read came
    # from conversion_root (Bug A fix), not from the eval's own root.
    assert not (eval_root / "conversion_result.json").exists()
    assert not (eval_root / "conversion_model_snapshot").exists()


def test_validate_only_evaluation_uses_structural_path_without_spawning(tmp_path: Path) -> None:
    """Bug B: structural.path is the authoritative key; validate_only never spawns."""
    _manifest, manifest_path, route = _fixture(
        tmp_path,
        "bugb-validate",
        camera_names=["first", "second"],
        width=29,
        height=17,
    )
    conversion_root = route / "outputs" / "bugb-validate" / "conversion-attempt"
    conversion_root.mkdir(parents=True)
    converted_ply = _write_conversion_evidence(conversion_root)

    eval_root = route / "outputs" / "bugb-validate" / "eval-attempt"
    eval_root.mkdir(parents=True)

    runner = _CapturingRunner(conversion_root)
    result = execute_evaluation(
        authority_manifest_path=manifest_path,
        evidence_root=eval_root,
        route_root=route,
        conversion_evidence_root=conversion_root,
        dry_run=False,
        validate_only=True,
        runner=runner,
    )

    assert result["validated"] is True
    assert result["stage"] == "converted-eval"
    request = result["request"]
    # --ply resolves to structural.path (the conversion root PLY), not a
    # None fallback, proving Bug B read symmetry.
    assert request["converted_ply_path"] == str(converted_ply)
    assert request["argv"][request["argv"].index("--ply") + 1] == str(converted_ply)
    # validate_only returns before subprocess.run: no child was spawned.
    assert runner.calls == []


def test_evaluation_requires_conversion_evidence_root(tmp_path: Path) -> None:
    """Bug A guard: absence of the conversion evidence root is a hard stop."""
    _manifest, manifest_path, route = _fixture(
        tmp_path,
        "buga-missing",
        camera_names=["only"],
        width=13,
        height=9,
    )
    eval_root = route / "outputs" / "buga-missing" / "eval-attempt"
    eval_root.mkdir(parents=True)
    with pytest.raises(Exception, match="conversion evidence root"):
        execute_evaluation(
            authority_manifest_path=manifest_path,
            evidence_root=eval_root,
            route_root=route,
            dry_run=True,
            validate_only=False,
        )


def test_execute_conversion_writes_authoritative_structural_path(tmp_path: Path) -> None:
    """Bug B write side: execute_conversion still records structural.path."""
    _manifest, manifest_path, route = _fixture(
        tmp_path,
        "bugb-write",
        camera_names=["first", "second"],
        width=29,
        height=17,
    )
    _seed_model_snapshot_inputs(route, "bugb-write")

    conversion_root = route / "outputs" / "bugb-write" / "conversion-attempt"
    runner = _CapturingRunner(conversion_root)
    result = execute_conversion(
        authority_manifest_path=manifest_path,
        evidence_root=conversion_root,
        route_root=route,
        dry_run=False,
        validate_only=False,
        runner=runner,
    )

    assert result["structural_pass"] is True
    assert result["STRUCTURAL_CONVERSION_PASS"] is True
    structural = result["structural"]
    assert isinstance(structural, dict)
    assert structural["path"] == str(_snapshot_ply(conversion_root))
    assert result["snapshot_model_path"] == str(conversion_root / "conversion_model_snapshot")
    # The persisted conversion_result.json repeats the same authoritative key.
    persisted = json.loads((conversion_root / "conversion_result.json").read_text(encoding="utf-8"))
    assert persisted["structural"]["path"] == str(_snapshot_ply(conversion_root))
    assert runner.calls, "child conversion was expected to run"


def test_authority_stage_creates_eval_executor_root_and_threads_conversion_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug A runtime: _execute_authority_stage creates the eval executor root
    (which converted-eval hands execute_evaluation as its evidence_root) and
    forwards conversion_evidence_root into the converted-eval child."""
    from scripts.longsplat.pipeline_contract import RunLedger
    from scripts.longsplat.reconstruct_pipeline import _execute_authority_stage

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ledger = RunLedger(run_dir, {"schema_version": "test", "code_identity": {}}, resumed=True)

    route = tmp_path / "route"
    conversion_root = tmp_path / "conversion-executor"
    native_root = tmp_path / "native-executor"

    authority_result = {
        "authority_manifest_path": str(tmp_path / "authority.json"),
        "authority": {
            "manifest": {
                "plan": {"path": str(tmp_path / "plan.json")},
                "static_contract": {"path": str(tmp_path / "static.json")},
            }
        },
    }

    captured: dict = {}

    def fake_execute_stage(**kwargs):
        captured.update(kwargs)
        evidence_root = kwargs["evidence_root"]
        assert Path(evidence_root).is_dir(), "eval executor root must be created before dispatch"
        assert kwargs["conversion_evidence_root"] == conversion_root
        assert kwargs["stage"] == "converted-eval"
        return {"stage": "converted-eval", "exit_code": 0, "SAME_CAMERA_VISUAL_PASS": "needs_review"}

    monkeypatch.setattr("scripts.longsplat.smoke_executor.execute_stage", fake_execute_stage)

    result, status = _execute_authority_stage(
        ledger=ledger,
        stage="converted-eval",
        authority_result=authority_result,
        plan=False,
        execute_gpu=True,
        route=route,
        native_evidence_root=native_root,
        conversion_evidence_root=conversion_root,
    )
    assert status == "passed"
    assert result["executor_root"] == str(run_dir / "stages" / "converted-eval" / "attempt-0001" / "executor")
    assert captured["conversion_evidence_root"] == conversion_root
