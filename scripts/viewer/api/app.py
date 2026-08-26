from __future__ import annotations

import mimetypes
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .contracts import ApiError, CreateJobResponse, JobLogs, JobSnapshot


def _error_response(status_code: int, code: str, message: str, job_id: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ApiError(code=code, message=message, job_id=job_id).model_dump(),
    )


def create_app(*, manager: Any | None = None) -> FastAPI:
    if manager is None:
        from .jobs import JobManager

        manager = JobManager.from_environment()

    app = FastAPI(title="Video-to-3DGS Local Web API", version="1.0")

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "schema_version": "web-health-v1"}

    @app.post("/api/v1/jobs", status_code=202, response_model=CreateJobResponse)
    def create_job(
        video: UploadFile = File(...),
        name: str | None = Form(default=None),
    ) -> CreateJobResponse | JSONResponse:
        try:
            return manager.submit_upload(video, display_name=name)
        except ValueError as exc:
            return _error_response(400, "invalid_input", str(exc))
        except RuntimeError as exc:
            return _error_response(409, "job_unavailable", str(exc))

    @app.get("/api/v1/jobs/{job_id}", response_model=JobSnapshot)
    def get_job(job_id: str) -> JobSnapshot | JSONResponse:
        try:
            return manager.get_snapshot(job_id)
        except KeyError:
            return _error_response(404, "job_not_found", "任务不存在", job_id)
        except ValueError as exc:
            return _error_response(409, "job_state_invalid", str(exc), job_id)

    @app.get("/api/v1/jobs/{job_id}/logs", response_model=JobLogs)
    def get_job_logs(
        job_id: str,
        stage: str | None = None,
        tail: int = 120,
    ) -> JobLogs | JSONResponse:
        try:
            return manager.get_logs(job_id, stage=stage, tail=tail)
        except KeyError:
            return _error_response(404, "job_not_found", "任务不存在", job_id)
        except ValueError as exc:
            return _error_response(400, "invalid_log_query", str(exc), job_id)

    @app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_id}", response_model=None)
    def get_artifact(job_id: str, artifact_id: str) -> FileResponse | JSONResponse:
        try:
            path = manager.resolve_artifact(job_id, artifact_id)
        except KeyError:
            return _error_response(404, "artifact_not_found", "artifact 不存在", job_id)
        except RuntimeError as exc:
            return _error_response(409, "artifact_invalid", str(exc), job_id)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=path.name)

    return app
