from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from scripts.viewer.api.contracts import CreateJobResponse, JobLogs, JobSnapshot
from scripts.viewer.api.app import create_app


class FakeManager:
    def __init__(self, tmp_path: Path):
        self.input_bytes = b""
        self.model = tmp_path / "model.ply"
        self.model.write_bytes(b"ply")

    def submit_upload(self, upload, display_name: str | None = None) -> CreateJobResponse:
        self.input_bytes = upload.file.read()
        return CreateJobResponse(
            schema_version="web-job-v1",
            job_id="job-1",
            status="queued",
            created_at="2026-08-24T00:00:00Z",
            input={
                "name": display_name or upload.filename or "input.mp4",
                "size_bytes": len(self.input_bytes),
                "sha256": "a" * 64,
            },
            status_url="/api/v1/jobs/job-1",
        )

    def get_snapshot(self, job_id: str) -> JobSnapshot:
        if job_id != "job-1":
            raise KeyError(job_id)
        return JobSnapshot.model_validate(
            {
                "schema_version": "web-job-snapshot-v1",
                "job_id": job_id,
                "run_id": job_id,
                "status": "queued",
                "created_at": "2026-08-24T00:00:00Z",
                "updated_at": "2026-08-24T00:00:00Z",
                "stage_order": [],
                "stages": [],
                "artifacts": [],
                "quality": {},
                "next_action": "等待任务启动",
            }
        )

    def get_logs(self, job_id: str, stage: str | None = None, tail: int = 120) -> JobLogs:
        if job_id != "job-1":
            raise KeyError(job_id)
        return JobLogs(
            schema_version="web-job-logs-v1",
            job_id=job_id,
            stages=[
                {
                    "id": stage or "probe",
                    "attempt": "attempt-0001",
                    "status": "passed",
                    "stdout": ["COLMAP ok"],
                    "stderr": [],
                    "updated_at": "2026-08-24T00:01:00Z",
                }
            ],
        )

    def resolve_artifact(self, job_id: str, artifact_id: str) -> Path:
        if job_id != "job-1" or artifact_id != "published-ply":
            raise KeyError(artifact_id)
        return self.model


def test_create_job_accepts_video_multipart_and_returns_202(tmp_path):
    manager = FakeManager(tmp_path)
    client = TestClient(create_app(manager=manager))

    response = client.post(
        "/api/v1/jobs",
        files={"video": ("gallery.mp4", b"video", "video/mp4")},
        data={"name": "gallery"},
    )

    assert response.status_code == 202
    assert response.json()["job_id"] == "job-1"
    assert manager.input_bytes == b"video"
    assert "D:" not in response.text


def test_job_status_and_artifact_routes_are_read_only_and_scoped(tmp_path):
    manager = FakeManager(tmp_path)
    client = TestClient(create_app(manager=manager))

    status = client.get("/api/v1/jobs/job-1")
    assert status.status_code == 200
    assert status.json()["status"] == "queued"

    artifact = client.get("/api/v1/jobs/job-1/artifacts/published-ply")
    assert artifact.status_code == 200
    assert artifact.content == b"ply"

    missing = client.get("/api/v1/jobs/job-2")
    assert missing.status_code == 404
    assert missing.json()["schema_version"] == "web-api-error-v1"


def test_job_logs_route_returns_structured_stage_output(tmp_path):
    client = TestClient(create_app(manager=FakeManager(tmp_path)))

    response = client.get("/api/v1/jobs/job-1/logs", params={"stage": "probe", "tail": 20})

    assert response.status_code == 200
    assert response.json()["schema_version"] == "web-job-logs-v1"
    assert response.json()["stages"][0]["stdout"] == ["COLMAP ok"]
