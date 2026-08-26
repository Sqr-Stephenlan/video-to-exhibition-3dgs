from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.viewer.api.contracts import BackendIdentity
from scripts.viewer.api.jobs import JobManager
from scripts.viewer.api.preflight import BackendPreflightError
from scripts.viewer.ply_contract import validate_gaussian_ply
from scripts.longsplat.publisher import _copy_policy


class IdleProcess:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.alive = True
        self.pid = 1234

    def start(self):
        self.started = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        return None


class StartErrorProcess(IdleProcess):
    def start(self):
        raise OSError("worker could not start")


def _upload(name: str, payload: bytes):
    return SimpleNamespace(filename=name, file=BytesIO(payload))


def test_manager_stores_controlled_upload_and_starts_one_worker(tmp_path):
    processes: list[IdleProcess] = []

    def process_factory(**kwargs):
        process = IdleProcess(**kwargs)
        processes.append(process)
        return process

    manager = JobManager(
        runtime_root=tmp_path / "runtime",
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=process_factory,
        probe_video=lambda path: None,
        max_upload_bytes=100,
    )

    response = manager.submit_upload(_upload("gallery.mp4", b"video"), display_name="Gallery")

    assert response.status == "running"
    assert len(processes) == 1
    record = manager.read_job_record(response.job_id)
    assert record["status"] == "running"
    assert record["input"]["size_bytes"] == 5
    assert not Path(record["input_path"]).is_absolute()
    assert (tmp_path / "runtime" / "web-incoming" / response.job_id).is_dir()


def test_manager_reports_worker_start_failure_in_create_response(tmp_path):
    manager = JobManager(
        runtime_root=tmp_path / "runtime",
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=lambda **kwargs: StartErrorProcess(**kwargs),
        probe_video=lambda path: None,
        max_upload_bytes=100,
    )

    response = manager.submit_upload(_upload("gallery.mp4", b"video"))

    assert response.status == "failed"
    record = manager.read_job_record(response.job_id)
    assert record["status"] == "failed"
    assert record["error"]["code"] == "worker_start_failed"


def test_manager_queues_a_second_job_while_worker_is_active(tmp_path):
    processes: list[IdleProcess] = []

    def process_factory(**kwargs):
        process = IdleProcess(**kwargs)
        processes.append(process)
        return process

    manager = JobManager(
        runtime_root=tmp_path / "runtime",
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=process_factory,
        probe_video=lambda path: None,
        max_upload_bytes=100,
    )

    first = manager.submit_upload(_upload("first.mp4", b"one"))
    second = manager.submit_upload(_upload("second.mp4", b"two"))

    assert first.status == "running"
    assert second.status == "queued"
    assert len(processes) == 1
    assert manager.read_job_record(second.job_id)["status"] == "queued"


def test_manager_rejects_empty_oversized_and_non_video_uploads(tmp_path):
    manager = JobManager(
        runtime_root=tmp_path / "runtime",
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=lambda **kwargs: IdleProcess(**kwargs),
        probe_video=lambda path: None,
        max_upload_bytes=3,
    )

    with pytest.raises(ValueError, match="video"):
        manager.submit_upload(_upload("scene.txt", b"1"))
    with pytest.raises(ValueError, match="empty"):
        manager.submit_upload(_upload("scene.mp4", b""))
    with pytest.raises(ValueError, match="size"):
        manager.submit_upload(_upload("scene.mp4", b"1234"))


def test_manager_cleans_an_upload_when_video_probe_fails(tmp_path):
    manager = JobManager(
        runtime_root=tmp_path / "runtime",
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=lambda **kwargs: IdleProcess(**kwargs),
        probe_video=lambda path: (_ for _ in ()).throw(ValueError("ffprobe failed")),
    )

    with pytest.raises(ValueError, match="ffprobe"):
        manager.submit_upload(_upload("scene.mp4", b"video"))

    assert not list((tmp_path / "runtime" / "web-incoming").iterdir())


def test_manager_persists_fixed_route_and_backend_identity(tmp_path):
    identity = BackendIdentity(
        superproject_gitlink="c" * 40,
        longsplat_commit="c" * 40,
        provider_identity="provider-sha256",
    )
    manager = JobManager(
        runtime_root=tmp_path / "runtime",
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=lambda **kwargs: IdleProcess(**kwargs),
        probe_video=lambda path: None,
        backend_preflight=lambda: identity,
    )

    response = manager.submit_upload(_upload("scene.mp4", b"video"))
    record = manager.read_job_record(response.job_id)

    assert record["route"] == {
        "pipeline_profile": "single-convergence1000-v1",
        "depth_source": "disabled",
        "pose_mode": "external-fixed-pose-rgb-only",
        "acceptance_policy": "automated-technical-v1",
    }
    assert record["backend"]["longsplat_commit"] == "c" * 40


def test_manager_blocks_job_when_backend_preflight_fails(tmp_path):
    def fail_preflight():
        raise BackendPreflightError("LongSplat checkout dirty", code="backend_dirty")

    manager = JobManager(
        runtime_root=tmp_path / "runtime",
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=lambda **kwargs: IdleProcess(**kwargs),
        probe_video=lambda path: None,
        backend_preflight=fail_preflight,
    )

    response = manager.submit_upload(_upload("scene.mp4", b"video"))
    record = manager.read_job_record(response.job_id)

    assert response.status == "blocked"
    assert record["status"] == "blocked"
    assert record["error"]["code"] == "backend_dirty"


def test_manager_marks_orphaned_running_job_blocked_after_restart(tmp_path):
    runtime = tmp_path / "runtime"
    job_dir = runtime / "web-jobs" / "web-111111111111111111111111"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(
        '{"schema_version":"web-job-v1","job_id":"web-111111111111111111111111",'
        '"run_id":"web-111111111111111111111111","status":"running",'
        '"created_at":"2026-08-24T00:00:00Z","updated_at":"2026-08-24T00:00:00Z",'
        '"input":{"name":"scene.mp4","size_bytes":5,"sha256":"a"*64}}',
        encoding="utf-8",
    )

    # Replace the invalid JSON shorthand with a real digest while keeping the
    # fixture creation local and deterministic.
    record_path = job_dir / "job.json"
    record = record_path.read_text(encoding="utf-8").replace('"a"*64', '"' + 'a' * 64 + '"')
    record_path.write_text(record, encoding="utf-8")

    manager = JobManager(
        runtime_root=runtime,
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=lambda **kwargs: IdleProcess(**kwargs),
        probe_video=lambda path: None,
    )

    recovered = manager.read_job_record("web-111111111111111111111111")

    assert recovered["status"] == "blocked"
    assert recovered["error"]["code"] == "worker_lost_after_restart"


def test_manager_publishes_receipt_and_gaussian_ply_artifacts(tmp_path):
    runtime = tmp_path / "runtime"
    publish = tmp_path / "outputs"
    manager = JobManager(
        runtime_root=runtime,
        route_root=tmp_path,
        publish_root=publish,
        process_factory=lambda **kwargs: IdleProcess(**kwargs),
        probe_video=lambda path: None,
    )
    response = manager.submit_upload(_upload("scene.mp4", b"video"))
    run_dir = runtime / "longsplat-runs" / response.job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        '{"stage_order":["automated-technical-delivery"],'
        '"pipeline_profile":"single-convergence1000-v1",'
        '"depth_source":"disabled",'
        '"acceptance_policy":"automated-technical-v1"}',
        encoding="utf-8",
    )
    model = publish / "scene.ply"
    publish.mkdir(parents=True, exist_ok=True)
    model.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float opacity\nproperty float scale_0\n"
        "property float rot_0\nend_header\n0 0 0 0 0 0 0\n",
        encoding="ascii",
    )
    receipt = run_dir / "published_ply.json"
    identity = validate_gaussian_ply(model)
    file_identity = {
        "path": str(model.resolve()),
        "sha256": identity.sha256,
        "size_bytes": identity.size_bytes,
        "vertices": identity.vertices,
        "nlink": model.stat().st_nlink,
    }
    receipt.write_text(
        json.dumps(
            {
                "schema_version": "longsplat-published-ply-v2",
                "source": file_identity,
                "published": file_identity,
                "reused": False,
                "copy_policy": _copy_policy(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        '{"run_id":"' + response.job_id + '","status":"complete",'
        '"updated_at":"2026-08-24T00:01:00Z",'
        '"published_ply":{"path":"' + model.as_posix() + '","vertices":1},'
        '"published_ply_receipt":"' + receipt.as_posix() + '",'
        '"accepted_by_automated_policy":true,\n'
        '"supersplat_format_compatible":true}',
        encoding="utf-8",
    )

    snapshot = manager.get_snapshot(response.job_id)

    assert snapshot.status == "complete"
    assert snapshot.quality.gaussian_schema_valid is True
    assert {item.id for item in snapshot.artifacts} >= {
        "published-ply",
        "published-ply-receipt",
    }


def test_manager_reads_bounded_stage_logs_from_the_run_root(tmp_path):
    runtime = tmp_path / "runtime"
    manager = JobManager(
        runtime_root=runtime,
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=lambda **kwargs: IdleProcess(**kwargs),
        probe_video=lambda path: None,
    )
    response = manager.submit_upload(_upload("scene.mp4", b"video"))
    run_dir = runtime / "longsplat-runs" / response.job_id
    stage_dir = run_dir / "stages" / "formal-training" / "attempt-0001"
    stage_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        '{"stage_order":["formal-training"],"pipeline_profile":"single-convergence1000-v1",'
        '"depth_source":"disabled","pose_mode":"external-fixed-pose-rgb-only",'
        '"acceptance_policy":"automated-technical-v1"}',
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        '{"run_id":"' + response.job_id + '","status":"running",'
        '"active_stage":"formal-training","active_attempt":"attempt-0001",'
        '"stages":{"formal-training":[{"attempt":"attempt-0001","status":"running"}]}}',
        encoding="utf-8",
    )
    (stage_dir / "stdout.log").write_text("line-1\nline-2\nline-3\n", encoding="utf-8")
    (stage_dir / "stderr.log").write_text("warning-1\nwarning-2\n", encoding="utf-8")

    logs = manager.get_logs(response.job_id, stage="formal-training", tail=2)

    assert logs.schema_version == "web-job-logs-v1"
    assert len(logs.stages) == 1
    assert logs.stages[0].id == "formal-training"
    assert logs.stages[0].stdout == ["line-2", "line-3"]
    assert logs.stages[0].stderr == ["warning-1", "warning-2"]


def test_manager_aligns_log_status_with_the_selected_attempt(tmp_path):
    runtime = tmp_path / "runtime"
    manager = JobManager(
        runtime_root=runtime,
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=lambda **kwargs: IdleProcess(**kwargs),
        probe_video=lambda path: None,
    )
    response = manager.submit_upload(_upload("scene.mp4", b"video"))
    run_dir = runtime / "longsplat-runs" / response.job_id
    stage_root = run_dir / "stages" / "formal-training"
    first = stage_root / "attempt-0001"
    second = stage_root / "attempt-0002"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (second / "executor").mkdir()
    (second / "executor" / "stdout.log").write_text("retry running\n", encoding="utf-8")
    (run_dir / "config.json").write_text(
        '{"stage_order":["formal-training"],"pipeline_profile":"single-convergence1000-v1",'
        '"depth_source":"disabled","pose_mode":"external-fixed-pose-rgb-only",'
        '"acceptance_policy":"automated-technical-v1"}',
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(
        '{"run_id":"' + response.job_id + '","status":"running",'
        '"active_stage":"formal-training","active_attempt":"attempt-0002",'
        '"stages":{"formal-training":[{"attempt":"attempt-0001","status":"blocked"},'
        '{"attempt":"attempt-0002","status":"running"}]}}',
        encoding="utf-8",
    )

    logs = manager.get_logs(response.job_id, stage="formal-training", tail=2)

    assert logs.stages[0].attempt == "attempt-0002"
    assert logs.stages[0].status == "running"


@pytest.mark.parametrize(
    ("run_status", "blocked_stage", "conversion_status", "expected_sample"),
    [
        ("blocked", "conversion", "blocked", True),
        ("blocked", "converted-eval", "blocked", False),
        ("running", None, "running", False),
    ],
)
def test_manager_exposes_completed_training_ply_only_for_conversion_block(
    tmp_path,
    run_status,
    blocked_stage,
    conversion_status,
    expected_sample,
):
    runtime = tmp_path / "runtime"
    manager = JobManager(
        runtime_root=runtime,
        route_root=tmp_path,
        publish_root=tmp_path / "outputs",
        process_factory=lambda **kwargs: IdleProcess(**kwargs),
        probe_video=lambda path: None,
    )
    response = manager.submit_upload(_upload("scene.mp4", b"video"))
    run_dir = runtime / "longsplat-runs" / response.job_id
    model_root = run_dir / "raw-formal-input" / "model"
    ply = model_root / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    ply.parent.mkdir(parents=True)
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float opacity\nproperty float scale_0\n"
        "property float rot_0\nend_header\n0 0 0 0 0 0 0\n",
        encoding="ascii",
    )
    stage_dir = run_dir / "stages" / "formal-training" / "attempt-0001"
    stage_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        '{"stage_order":["formal-training","conversion"],"pipeline_profile":"single-convergence1000-v1",'
        '"depth_source":"disabled","pose_mode":"external-fixed-pose-rgb-only",'
        '"acceptance_policy":"automated-technical-v1"}',
        encoding="utf-8",
    )
    (stage_dir / "result.json").write_text(
        json.dumps({"status": "passed", "result": {"model_path": str(model_root)}}),
        encoding="utf-8",
    )
    run_record = {
        "run_id": response.job_id,
        "status": run_status,
        "last_stage": blocked_stage or "conversion",
        "stages": {
            "formal-training": [
                {
                    "attempt": "attempt-0001",
                    "status": "passed",
                    "result_path": "stages/formal-training/attempt-0001/result.json",
                }
            ],
            "conversion": [{"attempt": "attempt-0001", "status": conversion_status}],
        },
    }
    if blocked_stage is not None:
        run_record["blocked"] = {"stage": blocked_stage, "error": "conversion failed"}
    (run_dir / "run.json").write_text(json.dumps(run_record), encoding="utf-8")

    snapshot = manager.get_snapshot(response.job_id)

    assert snapshot.status == run_status
    assert any(item.id == "training-sample-ply" for item in snapshot.artifacts) is expected_sample
