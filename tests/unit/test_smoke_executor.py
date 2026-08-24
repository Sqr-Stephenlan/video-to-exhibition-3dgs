from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.longsplat import smoke_executor as executor


def _write_executable(path: Path, text: str = "#!/bin/sh\nexit 0\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _contract_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, Path, Path, Path]:
    route = tmp_path / "route"
    (route / "outputs").mkdir(parents=True)
    _write_executable(route / "dev.sh")
    _write_executable(route / "third_party/LongSplat/train.py")
    _write_executable(route / "third_party/LongSplat/render.py")
    backend = tmp_path / "backend-env" / "bin" / "python"
    _write_executable(backend)
    source = tmp_path / "source;marker"
    (source / "contract").mkdir(parents=True)
    (source / "staging_manifest.json").write_text(
        json.dumps({"future_smoke_plan_path": "future.json", "image_records": [{"name": "frame_0.png"}]}),
        encoding="utf-8",
    )
    plan_file = source / "future.json"
    static_file = source / "contract/static_contract.json"
    model = tmp_path / "fresh-model"
    argv = [
        str(backend),
        str(route / "third_party/LongSplat/train.py"),
        "--source_path",
        str(source),
        "--images",
        "images",
        "--mode",
        "custom",
        "--resolution",
        "1",
        "--iterations",
        "100",
        "--external_colmap_pose",
        "--disable_resize",
        "--depth_source",
        "disabled",
        "--loss_2d_correspondence_weight",
        "0",
        "--depth_loss_weight",
        "0",
        "--rotation_lr_init",
        "0",
        "--translation_lr_init",
        "0",
        "--model_path",
        str(model),
    ]
    plan = {
        "schema_version": "longsplat-future-smoke-plan-v2",
        "workload_profile": "smoke100",
        "source_path": str(source),
        "model_path": str(model),
        "model_path_exists_at_plan_time": False,
        "argv": argv,
        "frozen_contract": {
            "workload_profile": "smoke100",
            "iterations": 100,
            "render_iteration": 100,
        },
        "backend_identity": {"resolved_path": str(backend.resolve())},
        "route_code_identity_sha256": "route-code",
        "future_smoke_profile": "smoke100",
        "future_smoke_iterations": 100,
    }
    static = {
        "computed_pass": True,
        "accepted": False,
        "delivery_reachable": False,
        "gpu_invoked": False,
        "training_invoked": False,
        "render_invoked": False,
        "route_code_identity_sha256": "route-code",
        "future_smoke_plan_path": "future.json",
        "image_count": 1,
        "camera": {"model": "PINHOLE", "width": 8, "height": 6, "fx": 4.0, "fy": 4.0, "cx": 4.0, "cy": 3.0},
    }
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    static_file.write_text(json.dumps(static), encoding="utf-8")
    monkeypatch.setattr(executor, "validate_future_smoke_plan", lambda *args, **kwargs: None)
    monkeypatch.setattr(executor, "code_identity", lambda route_root: {"code_identity_sha256": "route-code"})
    monkeypatch.setattr(
        executor,
        "_verify_static_input",
        lambda **kwargs: {"image_count": 1, "image_names": ["frame_0"], "camera": static["camera"]},
    )
    return plan, plan_file, static_file, route


def test_executor_builds_exact_wrapper_and_render_argv(tmp_path: Path) -> None:
    route = tmp_path / "route"
    source = tmp_path / "source with ; metachar"
    model = tmp_path / "model"
    argv = [
        "/backend/bin/python",
        str(route / "third_party/LongSplat/train.py"),
        "--source_path",
        str(source),
        "--images",
        "images",
        "--mode",
        "custom",
        "--resolution",
        "1",
        "--depth_source",
        "disabled",
        "--model_path",
        str(model),
        "--external_colmap_pose",
        "--disable_resize",
    ]
    contract = {"argv": argv, "backend_env": "/backend", "render_iteration": 100}
    train = executor.build_training_command(contract, route_root=route)
    render = executor.build_render_command(contract, route_root=route)
    assert train == [str(route / "dev.sh"), "python", *argv[1:]]
    assert render[:3] == [str(route / "dev.sh"), "python", str(route / "third_party/LongSplat/render.py")]
    assert str(source) in render and str(model) in render
    assert render[-6:] == ["--iteration", "100", "--eval", "--skip_test", "--nvs_pose_mode", "adjacent_midpoint_slerp"]
    assert "--skip_train" not in render


def test_executor_builds_formal_render_iteration_without_rebuilding_paths(tmp_path: Path) -> None:
    route = tmp_path / "route"
    source = tmp_path / "source"
    model = tmp_path / "formal-model"
    argv = [
        "/backend/bin/python",
        str(route / "third_party/LongSplat/train.py"),
        "--source_path", str(source), "--images", "images", "--mode", "custom",
        "--resolution", "1", "--depth_source", "disabled", "--model_path", str(model),
        "--external_colmap_pose", "--disable_resize",
    ]
    render = executor.build_render_command({"argv": argv, "render_iteration": 30000}, route_root=route)
    assert render[-6:] == ["--iteration", "30000", "--eval", "--skip_test", "--nvs_pose_mode", "adjacent_midpoint_slerp"]
    assert str(source) in render and str(model) in render


def test_executor_validates_exact_paths_and_code_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, plan_file, static_file, route = _contract_fixture(tmp_path, monkeypatch)
    details = executor.validate_plan_and_static(plan_file, static_file, route_root=route)
    assert details["argv"][details["argv"].index("--source_path") + 1] == plan["source_path"]

    duplicate = dict(plan)
    duplicate["argv"] = list(plan["argv"]) + ["--source_path", plan["source_path"]]
    plan_file.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(executor.SmokeExecutorBlocked, match="exactly one --source_path"):
        executor.validate_plan_and_static(plan_file, static_file, route_root=route)

    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    monkeypatch.setattr(executor, "code_identity", lambda route_root: {"code_identity_sha256": "drift"})
    with pytest.raises(executor.SmokeExecutorBlocked, match="route source identity"):
        executor.validate_plan_and_static(plan_file, static_file, route_root=route)

    monkeypatch.setattr(executor, "code_identity", lambda route_root: {"code_identity_sha256": "route-code"})
    static = json.loads(static_file.read_text(encoding="utf-8"))
    static["route_code_identity_sha256"] = "static-drift"
    static_file.write_text(json.dumps(static), encoding="utf-8")
    with pytest.raises(executor.SmokeExecutorBlocked, match="static/plan route code identity"):
        executor.validate_plan_and_static(plan_file, static_file, route_root=route)


def test_executor_rejects_relative_source_and_existing_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, plan_file, static_file, route = _contract_fixture(tmp_path, monkeypatch)
    relative = dict(plan)
    relative["source_path"] = "relative/source"
    plan_file.write_text(json.dumps(relative), encoding="utf-8")
    with pytest.raises(executor.SmokeExecutorBlocked, match="absolute path"):
        executor.validate_plan_and_static(plan_file, static_file, route_root=route)

    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    Path(plan["model_path"]).mkdir()
    with pytest.raises(executor.SmokeExecutorBlocked, match="fresh model_path"):
        executor.validate_plan_and_static(plan_file, static_file, route_root=route)


def test_executor_fake_runner_gets_list_env_and_shell_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan, plan_file, static_file, route = _contract_fixture(tmp_path, monkeypatch)
    seen: dict[str, object] = {}

    def fake_runner(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 7, stdout="partial", stderr="blocked")

    evidence = route / "outputs" / "evidence"
    result = executor.execute_stage(
        plan_path=plan_file,
        static_contract_path=static_file,
        evidence_root=evidence,
        stage="training",
        route_root=route,
        runner=fake_runner,
    )
    assert result["exit_code"] == 7
    assert isinstance(seen["command"], list)
    assert seen["shell"] is False
    assert seen["env"]["VENV_DIR"] == str(tmp_path / "backend-env")
    assert str(Path(plan["source_path"])) in seen["command"]
    assert not (tmp_path / "marker").exists()
    assert (evidence / "request.json").is_file()
    assert (evidence / "result.json").is_file()


def test_executor_rejects_existing_evidence_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, plan_file, static_file, route = _contract_fixture(tmp_path, monkeypatch)
    evidence = route / "outputs" / "evidence"
    evidence.mkdir()
    with pytest.raises(executor.SmokeExecutorBlocked, match="evidence root must be absent"):
        executor.execute_stage(
            plan_path=plan_file,
            static_contract_path=static_file,
            evidence_root=evidence,
            stage="training",
            route_root=route,
        )


def test_executor_evidence_root_is_strictly_contained_and_symlink_safe(tmp_path: Path) -> None:
    route = tmp_path / "route"
    outputs = route / "outputs"
    outputs.mkdir(parents=True)
    assert executor._output_path(outputs / "nested" / "attempt", route, "evidence") == outputs / "nested" / "attempt"
    with pytest.raises(executor.SmokeExecutorBlocked, match="below route outputs"):
        executor._output_path(tmp_path / "outside", route, "evidence")
    with pytest.raises(executor.SmokeExecutorBlocked, match="strict descendant"):
        executor._output_path(outputs, route, "evidence")
    outside = tmp_path / "outside-target"
    outside.mkdir()
    link = outputs / "escape"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(executor.SmokeExecutorBlocked, match="symlink"):
        executor._output_path(link / "attempt", route, "evidence")


def test_executor_rejects_repeated_worktree_project_prefix(tmp_path: Path) -> None:
    route = tmp_path / "video-to-exhibition-3dgs" / "worktrees" / "longsplat-route"
    (route / "outputs").mkdir(parents=True)
    repeated = route / "outputs" / "video-to-exhibition-3dgs" / "worktrees" / "longsplat-route"
    with pytest.raises(executor.SmokeExecutorBlocked, match="repeated worktree/project prefix"):
        executor._output_path(repeated, route, "evidence")


def test_executor_snapshot_copies_only_render_allowlist_and_preserves_source(tmp_path: Path) -> None:
    route = tmp_path / "route"
    (route / "outputs").mkdir(parents=True)
    model = tmp_path / "training-model"
    iteration = 30000
    relatives = [item.format(iteration=iteration) for item in executor._SNAPSHOT_RELATIVE_FILES]
    for index, relative in enumerate(relatives):
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}".encode())
    (model / f"train/ours_{iteration}").mkdir(parents=True)
    (model / f"train/ours_{iteration}/partial.png").write_bytes(b"must-not-copy")
    plan = {"route_code_identity_sha256": "producer"}
    contract = {
        "model_path": str(model),
        "render_iteration": iteration,
        "plan_path": str(tmp_path / "plan.json"),
        "plan_sha256": "plan-sha",
        "static_contract_path": str(tmp_path / "static.json"),
        "static_contract_sha256": "static-sha",
        "plan": plan,
        "render_consumer_identity": {"files": {
            "nested/render.py": {"sha256": "render-sha"},
            "nested/utils/colmap_utils.py": {"sha256": "colmap-sha"},
        }},
    }
    evidence = route / "outputs" / "attempt"
    evidence.mkdir(parents=True)
    training = {
        "root": str(tmp_path / "training-evidence"),
        "result_path": str(tmp_path / "training-evidence/result.json"),
        "result_sha256": "result-sha",
        "request_path": str(tmp_path / "training-evidence/request.json"),
        "request_sha256": "request-sha",
    }
    snapshot, manifest, manifest_path = executor._snapshot_render_model(
        contract=contract,
        training_evidence=training,
        evidence_root=evidence,
        route=route,
    )
    assert len(manifest["files"]) == len(relatives)
    assert manifest_path.is_file()
    assert not (snapshot / f"train/ours_{iteration}").exists()
    assert not (snapshot / f"test/ours_{iteration}").exists()
    for relative in relatives:
        assert (snapshot / relative).read_bytes() == (model / relative).read_bytes()
        assert not (snapshot / relative).is_symlink()
    assert (model / f"train/ours_{iteration}/partial.png").is_file()
    assert executor._verify_snapshot_manifest(manifest_path)["snapshot_aggregate_sha256"] == manifest["snapshot_aggregate_sha256"]


def test_executor_snapshot_render_command_replaces_only_model_path(tmp_path: Path) -> None:
    route = tmp_path / "route"
    source = tmp_path / "source"
    original = tmp_path / "original-model"
    snapshot = tmp_path / "evidence" / "render_model_snapshot"
    contract = {
        "argv": [
            "/backend/bin/python", str(route / "third_party/LongSplat/train.py"),
            "--source_path", str(source), "--images", "images", "--mode", "custom",
            "--resolution", "1", "--depth_source", "disabled", "--model_path", str(original),
            "--external_colmap_pose", "--disable_resize",
        ],
        "render_iteration": 30000,
    }
    replaced = executor._replace_model_path(contract, snapshot)
    command = executor.build_render_command(replaced, route_root=route)
    assert str(source) in command
    assert str(snapshot) in command
    assert str(original) not in command
    assert [token for token in command if token == "--eval"] == ["--eval"]
    assert [token for token in command if token == "--skip_test"] == ["--skip_test"]
    assert "--skip_train" not in command
    assert command[-6:] == ["--iteration", "30000", "--eval", "--skip_test", "--nvs_pose_mode", "adjacent_midpoint_slerp"]


def test_executor_render_consumer_identity_rejects_render_math_drift(tmp_path: Path) -> None:
    route = tmp_path / "route"
    render = route / "third_party/LongSplat/render.py"
    colmap = route / "third_party/LongSplat/utils/colmap_utils.py"
    render.parent.mkdir(parents=True)
    colmap.parent.mkdir(parents=True)
    render.write_text("render-v1", encoding="utf-8")
    colmap.write_text("colmap-v1", encoding="utf-8")
    plan = {
        "route_code_identity_sha256": "producer",
        "route_code_identity": {"files": [
            {"path": "nested/render.py", "sha256": executor.sha256_file(render)},
            {"path": "nested/utils/colmap_utils.py", "sha256": executor.sha256_file(colmap)},
        ]},
    }
    assert executor._verify_render_consumer_identity(plan, route)["files"]["nested/render.py"]["sha256"] == executor.sha256_file(render)
    render.write_text("render-drift", encoding="utf-8")
    with pytest.raises(executor.SmokeExecutorBlocked, match="render consumer identity drifted"):
        executor._verify_render_consumer_identity(plan, route)


def test_executor_training_evidence_must_be_under_route_outputs(tmp_path: Path) -> None:
    route = tmp_path / "route"
    (route / "outputs").mkdir(parents=True)
    with pytest.raises(executor.SmokeExecutorBlocked, match="below route outputs"):
        executor._verify_training_evidence(tmp_path / "outside-training", {}, route)


def test_executor_snapshot_rejects_existing_snapshot(tmp_path: Path) -> None:
    route = tmp_path / "route"
    (route / "outputs").mkdir(parents=True)
    evidence = route / "outputs" / "attempt"
    evidence.mkdir(parents=True)
    snapshot = evidence / "render_model_snapshot"
    snapshot.mkdir()
    contract = {"model_path": str(tmp_path / "model"), "render_iteration": 30000}
    with pytest.raises(executor.SmokeExecutorBlocked, match="snapshot must be absent"):
        executor._snapshot_render_model(
            contract=contract,
            training_evidence={},
            evidence_root=evidence,
            route=route,
        )


def _training_evidence_fixture(root: Path, route: Path) -> tuple[dict, Path]:
    evidence = root / "stages" / "convergence-smoke-training" / "attempt-0001" / "executor"
    evidence.mkdir(parents=True)
    model = root / "model"
    source = root / "training-input"
    source.mkdir()
    contract = {
        "model_path": str(model),
        "plan_path": str(root / "plan.json"),
        "plan_sha256": "plan-sha",
        "static_contract_path": str(source / "contract" / "static_contract.json"),
        "static_contract_sha256": "static-sha",
        "source_path": str(source),
        "workload_profile": "convergence1000-v1",
        "iterations": 1000,
        "render_iteration": 1000,
        "plan": {"route_code_identity_sha256": "producer"},
    }
    argv = ["/route/dev.sh", "python", "train.py"]
    (evidence / "result.json").write_text(
        json.dumps(
            {
                "stage": "training",
                "exit_code": 0,
                "structural": {"structural_pass": True},
                "model_path": str(model),
                "request_path": str(evidence / "request.json"),
                "argv_path": str(evidence / "argv.json"),
                "static_input_reference": {
                    "source_path": str(source),
                    "plan_sha256": "plan-sha",
                    "static_contract_sha256": "static-sha",
                },
            }
        ),
        encoding="utf-8",
    )
    (evidence / "request.json").write_text(
        json.dumps(
            {
                "stage": "training",
                "model_path": str(model),
                "plan_path": str(root / "plan.json"),
                "plan_sha256": "plan-sha",
                "static_contract_path": str(source / "contract" / "static_contract.json"),
                "static_contract_sha256": "static-sha",
                "source_path": str(source),
                "workload_profile": "convergence1000-v1",
                "iterations": 1000,
                "render_iteration": 1000,
                "route_code_identity_sha256": "producer",
                "argv": argv,
                "shell": False,
            }
        ),
        encoding="utf-8",
    )
    (evidence / "argv.json").write_text(json.dumps({"argv": argv, "shell": False}), encoding="utf-8")
    return contract, evidence


def test_external_training_evidence_uses_dynamic_authority_and_rejects_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = tmp_path / "route"
    route.mkdir()
    roots = [tmp_path / "external-a" / "run", tmp_path / "external-b" / "run", route / "outputs"]
    fixtures: dict[Path, tuple[dict, Path]] = {}
    for root in roots:
        root.mkdir(parents=True)
        contract, evidence = _training_evidence_fixture(root, route)
        fixtures[root] = (contract, evidence)
        monkeypatch.setattr(executor, "_verify_training_outputs", lambda *args, **kwargs: {"structural_pass": True})
        checked = executor._verify_training_evidence(evidence, contract, route, containment_root=root)
        assert checked["root"] == str(evidence.resolve())

    root = roots[0]
    contract, _ = fixtures[root]
    _, other_evidence = fixtures[roots[1]]
    with pytest.raises(executor.SmokeExecutorBlocked, match="escapes"):
        executor._verify_training_evidence(other_evidence, contract, route, containment_root=root)

    outside = tmp_path / "outside-training"
    outside.mkdir()
    link = root / "linked-evidence"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(executor.SmokeExecutorBlocked, match="symlink"):
        executor._verify_training_evidence(link, contract, route, containment_root=root)

    result_path = root / "stages" / "convergence-smoke-training" / "attempt-0001" / "executor" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["stage"] = "render"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    evidence = fixtures[root][1]
    with pytest.raises(executor.SmokeExecutorBlocked, match="successful structural training"):
        executor._verify_training_evidence(evidence, contract, route, containment_root=root)


def test_external_snapshot_source_and_destination_share_dynamic_authority(tmp_path: Path) -> None:
    route = tmp_path / "route"
    route.mkdir()
    root = tmp_path / "external" / "run"
    root.mkdir(parents=True)
    model = root / "training-model"
    iteration = 1000
    relatives = [item.format(iteration=iteration) for item in executor._SNAPSHOT_RELATIVE_FILES]
    for index, relative in enumerate(relatives):
        path = model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact-{index}".encode())
    evidence = root / "stages" / "convergence-smoke-render" / "attempt-0001" / "executor"
    evidence.mkdir(parents=True)
    contract = {
        "model_path": str(model),
        "render_iteration": iteration,
        "plan_path": str(root / "plan.json"),
        "plan_sha256": "plan-sha",
        "static_contract_path": str(root / "static.json"),
        "static_contract_sha256": "static-sha",
        "plan": {"route_code_identity_sha256": "producer"},
        "render_consumer_identity": {"files": {}},
    }
    training = {
        "root": str(root / "stages" / "convergence-smoke-training" / "attempt-0001" / "executor"),
        "result_path": str(root / "training-result.json"),
        "result_sha256": "result-sha",
        "request_path": str(root / "training-request.json"),
        "request_sha256": "request-sha",
    }
    snapshot, _, manifest_path = executor._snapshot_render_model(
        contract=contract,
        training_evidence=training,
        evidence_root=evidence,
        route=route,
        containment_root=root,
    )
    assert snapshot.is_dir()
    assert snapshot.is_relative_to(root)
    assert executor._verify_snapshot_manifest(manifest_path, containment_root=root)["isolated_render_model_path"] == str(snapshot)

    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    outside_model = outside_root / "model"
    outside_model.mkdir()
    with pytest.raises(executor.SmokeExecutorBlocked, match="outside|escapes"):
        executor._snapshot_render_model(
            contract={**contract, "model_path": str(outside_model)},
            training_evidence=training,
            evidence_root=evidence,
            route=route,
            containment_root=root,
        )
