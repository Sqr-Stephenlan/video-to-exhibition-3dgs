from __future__ import annotations

import sys
from types import ModuleType

from scripts.viewer.api.contracts import BackendIdentity
from scripts.viewer.api.worker import run_reconstruction_job


def test_worker_calls_existing_reconstruction_contract(monkeypatch, tmp_path):
    calls = {}

    monkeypatch.setattr(
        "scripts.viewer.api.worker.verify_gpu_backend",
        lambda **kwargs: {"status": "passed"},
    )

    def fake_run_reconstruction(**kwargs):
        calls.update(kwargs)
        return {"status": "complete"}

    fake_module = ModuleType("scripts.longsplat.reconstruct_pipeline")
    fake_module.run_reconstruction = fake_run_reconstruction
    monkeypatch.setitem(sys.modules, "scripts.longsplat.reconstruct_pipeline", fake_module)

    result = run_reconstruction_job(
        job_id="job-1",
        input_video=tmp_path / "input.mp4",
        runtime_root=tmp_path / "runtime",
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        delivery_name="web-job-1",
        source_metadata={"sha256": "a" * 64},
        backend_identity=BackendIdentity(
            superproject_gitlink="c" * 40,
            longsplat_commit="c" * 40,
            provider_identity="provider-sha256",
        ),
    )

    assert result == {"status": "complete"}
    assert calls["run_id"] == "job-1"
    assert calls["stop_after"] == "automated-technical-delivery"
    assert calls["depth_source"] == "disabled"
    assert calls["execute_gpu"] is True
    assert calls["pipeline_profile"] == "single-convergence1000-v1"
    assert calls["acceptance_policy"] == "automated-technical-v1"
    assert calls["publish_dir"] == tmp_path / "outputs"


def test_worker_runs_gpu_preflight_before_reconstruction(monkeypatch, tmp_path):
    calls: list[str] = []

    monkeypatch.setattr(
        "scripts.viewer.api.worker.verify_gpu_backend",
        lambda **kwargs: calls.append("gpu") or {"status": "passed"},
    )

    fake_module = ModuleType("scripts.longsplat.reconstruct_pipeline")
    fake_module.run_reconstruction = lambda **kwargs: calls.append("run") or {"status": "complete"}
    monkeypatch.setitem(sys.modules, "scripts.longsplat.reconstruct_pipeline", fake_module)

    run_reconstruction_job(
        job_id="job-1",
        input_video=tmp_path / "input.mp4",
        runtime_root=tmp_path / "runtime",
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        delivery_name="web-job-1",
        source_metadata={"sha256": "a" * 64},
        backend_identity=BackendIdentity(
            superproject_gitlink="c" * 40,
            longsplat_commit="c" * 40,
            provider_identity="provider-sha256",
        ),
    )

    assert calls == ["gpu", "run"]
