from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.longsplat import smoke_executor as executor


def _paths(tmp_path: Path) -> dict[str, Path]:
    route = tmp_path / "route"
    model = tmp_path / "formal-model"
    source = tmp_path / "source"
    nested = route / "third_party/LongSplat"
    backend_env = tmp_path / "backend-env"
    backend_python = backend_env / "bin/python"
    for path in (route, model, source, nested, backend_env, backend_env / "bin"):
        path.mkdir(parents=True, exist_ok=True)
    backend_python.write_text("python", encoding="utf-8")
    backend_python.chmod(0o755)
    return {
        "route": route,
        "source": source,
        "model": model,
        "nested": nested,
        "backend_env": backend_env,
        "backend_python": backend_python,
    }


def _authority(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "plan_sha256": "plan",
        "static_contract_sha256": "static",
        "native_evidence": {"result_path": "native-result"},
        "backend_identity": {},
        # Historical value is explicit regression-fixture metadata, not a
        # production conversion/evaluation default.
        "camera_count": 45,
    }


def _seed_snapshot_source(model: Path) -> None:
    for relative in executor._conversion_source_relatives():
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((relative + "\n").encode("utf-8"))


def test_conversion_snapshot_allowlist_alias_and_source_recheck(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _seed_snapshot_source(paths["model"])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    snapshot, manifest_path, _ = executor._create_conversion_snapshot(
        paths=paths,
        evidence_root=evidence,
        authority=_authority(paths),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["alias"]["reason"] == "conversion eval-false entry"
    assert manifest["alias"]["training_view_count"] == 45
    assert (snapshot / "cameras_all.json").read_bytes() == (snapshot / "cameras_all_train.json").read_bytes()
    assert not (snapshot / "train").exists()
    assert not (snapshot / "test").exists()
    assert not (snapshot / "events").exists()
    executor._verify_conversion_snapshot(
        snapshot=snapshot,
        manifest_path=manifest_path,
        allow_converted_output=False,
    )

    (paths["model"] / "cfg_args").write_text("mutated", encoding="utf-8")
    with pytest.raises(executor.SmokeExecutorBlocked, match="source changed"):
        executor._verify_conversion_snapshot(
            snapshot=snapshot,
            manifest_path=manifest_path,
            allow_converted_output=False,
        )


def test_conversion_snapshot_rejects_unallowlisted_output(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _seed_snapshot_source(paths["model"])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    snapshot, manifest_path, _ = executor._create_conversion_snapshot(
        paths=paths,
        evidence_root=evidence,
        authority=_authority(paths),
    )
    unexpected = snapshot / "events/old.log"
    unexpected.parent.mkdir()
    unexpected.write_text("old", encoding="utf-8")
    with pytest.raises(executor.SmokeExecutorBlocked, match="outside the allowlist"):
        executor._verify_conversion_snapshot(
            snapshot=snapshot,
            manifest_path=manifest_path,
            allow_converted_output=False,
        )


def test_conversion_argv_freezes_all_authorized_parameters(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    evidence = tmp_path / "evidence"
    snapshot = evidence / "conversion_model_snapshot"
    command, nested = executor._build_conversion_argv(
        paths=paths,
        evidence_root=evidence,
        snapshot=snapshot,
    )
    assert command[0:4] == [str(paths["route"] / "dev.sh"), "python", "-m", "scripts.longsplat.reconvert_existing"]
    assert nested[1] == str(paths["nested"] / "convert_3dgs.py")
    for flag, value in (
        ("--iteration", "30000"),
        ("--prune_ratio", "0.6"),
        ("--seed", "0"),
        ("--anisotropy_reg_weight", "0.01"),
        ("--anisotropy_soft_limit", "30.0"),
    ):
        index = nested.index(flag)
        assert nested[index + 1] == value
    assert "--checkpoint-iteration" not in nested
    assert "--conversion-iterations" not in nested


def test_conversion_stage_rejects_training_stage_options() -> None:
    with pytest.raises(executor.SmokeExecutorBlocked, match="does not accept"):
        executor.execute_stage(
            plan_path="/unused/plan",
            static_contract_path="/unused/static",
            evidence_root="/unused/evidence",
            stage="conversion",
            route_root="/unused/route",
            training_evidence_root="/unused/training",
        )


def test_cli_rejects_arbitrary_conversion_parameters() -> None:
    with pytest.raises(SystemExit):
        executor.main(
            [
                "--plan",
                "/frozen/plan.json",
                "--static-contract",
                "/frozen/static.json",
                "--evidence-root",
                "/frozen/evidence",
                "--stage",
                "conversion",
                "--iterations",
                "1",
            ]
        )
