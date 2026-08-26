from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .artifacts import ArtifactCatalog
from .contracts import BackendIdentity, CreateJobResponse, JobLogs, JobSnapshot, StageLogSnapshot
from .preflight import BackendPreflightError, verify_backend_contract
from .snapshot import build_job_snapshot, normalize_run_status
from .worker import run_reconstruction_job


_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
_JOB_ID_PATTERN = re.compile(r"^web-[0-9a-f]{24}$")
_MAX_CHUNK = 1024 * 1024
_DEFAULT_LOG_TAIL = 120
_MAX_LOG_TAIL = 400
_MAX_LOG_BYTES = 512 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _probe_with_ffprobe(path: Path, executable: str) -> None:
    try:
        result = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_entries",
                "format=format_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"ffprobe 无法验证视频: {exc}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("ffprobe 无法识别上传文件")


def _default_process_factory(**kwargs: Any) -> multiprocessing.Process:
    return multiprocessing.Process(target=_worker_process_entry, kwargs=kwargs, daemon=False)


def _worker_process_entry(
    *,
    job_id: str,
    input_video: Path,
    runtime_root: Path,
    route_root: Path,
    publish_root: Path,
    delivery_name: str,
    source_metadata: Mapping[str, Any],
    backend_identity: BackendIdentity | None,
    result_path: Path,
) -> None:
    try:
        result = run_reconstruction_job(
            job_id=job_id,
            input_video=input_video,
            runtime_root=runtime_root,
            route_root=route_root,
            publish_root=publish_root,
            delivery_name=delivery_name,
            source_metadata=source_metadata,
            backend_identity=backend_identity or BackendIdentity(
                superproject_gitlink="0" * 40,
                longsplat_commit="0" * 40,
                provider_identity="unverified",
            ),
        )
        _write_json_atomic(result_path, {"status": "complete", "result": result})
    except BaseException as exc:
        _write_json_atomic(
            result_path,
            {
                "status": "failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "error_code": getattr(exc, "code", None),
            },
        )
        raise


class JobManager:
    """Single-user local scheduler backed by filesystem job metadata."""

    def __init__(
        self,
        *,
        runtime_root: str | Path,
        route_root: str | Path,
        publish_root: str | Path,
        process_factory: Callable[..., Any] | None = None,
        probe_video: Callable[[Path], None] | None = None,
        max_upload_bytes: int = 8 * 1024 * 1024 * 1024,
        progress_reader: Callable[[Path], Any] | None = None,
        backend_preflight: Callable[[], BackendIdentity] | None = None,
    ):
        self.runtime_root = Path(runtime_root).resolve()
        self.route_root = Path(route_root).resolve()
        self.publish_root = Path(publish_root).resolve()
        self.jobs_root = self.runtime_root / "web-jobs"
        self.incoming_root = self.runtime_root / "web-incoming"
        self.runs_root = self.runtime_root / "longsplat-runs"
        self.max_upload_bytes = max_upload_bytes
        self.probe_video = probe_video or (lambda path: _probe_with_ffprobe(path, "ffprobe"))
        self.process_factory = process_factory or _default_process_factory
        self.progress_reader = progress_reader or self._read_progress
        self.backend_preflight = backend_preflight
        self.backend_identity: BackendIdentity | None = None
        self.backend_error: BackendPreflightError | None = None
        self._queue: list[str] = []
        self._active_job_id: str | None = None
        self._active_process: Any | None = None
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.incoming_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._recover_orphaned_jobs()

    @classmethod
    def from_environment(cls) -> "JobManager":
        route_root = Path(__file__).resolve().parents[3]
        runtime_root = Path(os.environ.get("VIDEO_TO_3DGS_RUNTIME_ROOT", route_root / ".runtime"))
        publish_root = Path(os.environ.get("VIDEO_TO_3DGS_PUBLISH_ROOT", route_root / "outputs"))
        try:
            max_upload_bytes = int(
                os.environ.get("VIDEO_TO_3DGS_MAX_UPLOAD_BYTES", 8 * 1024 * 1024 * 1024)
            )
        except ValueError as exc:
            raise ValueError("VIDEO_TO_3DGS_MAX_UPLOAD_BYTES 必须是整数") from exc
        ffprobe = os.environ.get("VIDEO_TO_3DGS_FFPROBE", "ffprobe")
        return cls(
            runtime_root=runtime_root,
            route_root=route_root,
            publish_root=publish_root,
            max_upload_bytes=max_upload_bytes,
            probe_video=lambda path: _probe_with_ffprobe(path, ffprobe),
            backend_preflight=lambda: verify_backend_contract(
                repo_root=route_root / "third_party" / "LongSplat",
                route_root=route_root,
            ),
        )

    def _recover_orphaned_jobs(self) -> None:
        for record_path in self.jobs_root.glob("*/job.json"):
            record = _read_json(record_path)
            if not isinstance(record, dict) or record.get("status") not in {"queued", "running"}:
                continue
            record["status"] = "blocked"
            record["error"] = {
                "code": "worker_lost_after_restart",
                "message": "API 重启后未恢复后台 worker；请重新提交任务",
            }
            record["updated_at"] = _utc_now()
            _write_json_atomic(record_path, record)

    def _ensure_backend_identity(self) -> BackendIdentity | None:
        if self.backend_identity is not None:
            return self.backend_identity
        if self.backend_error is not None:
            return None
        if self.backend_preflight is None:
            return None
        try:
            identity = self.backend_preflight()
            self.backend_identity = (
                identity if isinstance(identity, BackendIdentity) else BackendIdentity.model_validate(identity)
            )
            return self.backend_identity
        except BackendPreflightError as exc:
            self.backend_error = exc
            return None
        except Exception as exc:
            self.backend_error = BackendPreflightError(str(exc))
            return None

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / job_id

    def _job_record_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "job.json"

    def _run_dir(self, job_id: str) -> Path:
        return self.runs_root / job_id

    def _validate_job_id(self, job_id: str) -> None:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            raise KeyError(job_id)

    def read_job_record(self, job_id: str) -> dict[str, Any]:
        self._validate_job_id(job_id)
        record = _read_json(self._job_record_path(job_id))
        if record is None:
            raise KeyError(job_id)
        return record

    def _new_job_id(self) -> str:
        return f"web-{uuid.uuid4().hex[:24]}"

    def _safe_input_name(self, filename: str) -> str:
        basename = Path(filename).name
        suffix = Path(basename).suffix.lower()
        if suffix not in _VIDEO_EXTENSIONS:
            raise ValueError("only video files are supported")
        return basename[:200] or f"input{suffix}"

    def _store_upload(self, *, job_id: str, filename: str, upload: Any) -> tuple[Path, str, int]:
        safe_name = self._safe_input_name(filename)
        target_dir = self.incoming_root / job_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / f"input{Path(safe_name).suffix.lower()}"
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        digest = hashlib.sha256()
        size = 0
        try:
            with temporary.open("wb") as handle:
                while True:
                    chunk = upload.file.read(_MAX_CHUNK)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise ValueError("video exceeds upload size limit")
                    digest.update(chunk)
                    handle.write(chunk)
            if size == 0:
                raise ValueError("video upload is empty")
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            target_dir.rmdir()
            raise
        return target, digest.hexdigest(), size

    def submit_upload(self, upload: Any, display_name: str | None = None) -> CreateJobResponse:
        filename = str(getattr(upload, "filename", "") or "")
        job_id = self._new_job_id()
        target, sha256, size_bytes = self._store_upload(
            job_id=job_id,
            filename=filename,
            upload=upload,
        )
        try:
            self.probe_video(target)
        except BaseException:
            target.unlink(missing_ok=True)
            try:
                target.parent.rmdir()
            except OSError:
                pass
            raise
        input_relative = target.relative_to(self.runtime_root).as_posix()
        record: dict[str, Any] = {
            "schema_version": "web-job-v1",
            "job_id": job_id,
            "run_id": job_id,
            "status": "queued",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "display_name": (display_name or Path(filename).stem or "video")[:120],
            "input_path": input_relative,
            "input": {
                "name": Path(filename).name[:255] or "video",
                "size_bytes": size_bytes,
                "sha256": sha256,
            },
            "route": {
                "pipeline_profile": "single-convergence1000-v1",
                "depth_source": "disabled",
                "pose_mode": "external-fixed-pose-rgb-only",
                "acceptance_policy": "automated-technical-v1",
            },
        }
        _write_json_atomic(self._job_record_path(job_id), record)
        self._queue.append(job_id)
        self._start_next()
        current = self.read_job_record(job_id)
        response_status = current.get("status")
        if response_status not in {"queued", "running", "complete", "blocked", "failed", "stopped"}:
            response_status = "queued"
        return CreateJobResponse(
            schema_version="web-job-v1",
            job_id=job_id,
            status=response_status,
            created_at=str(current["created_at"]),
            input=current["input"],
            status_url=f"/api/v1/jobs/{job_id}",
        )

    def _start_next(self) -> None:
        if self._active_process is not None:
            if self._active_process.is_alive():
                return
            self._reap_active()
        while self._queue:
            job_id = self._queue.pop(0)
            record = self.read_job_record(job_id)
            if record.get("status") != "queued":
                continue
            backend_identity = self._ensure_backend_identity()
            if self.backend_preflight is not None and backend_identity is None:
                error = self.backend_error or BackendPreflightError(
                    "backend preflight 未返回 identity"
                )
                record["status"] = "blocked"
                record["error"] = {"code": error.code, "message": str(error)}
                record["updated_at"] = _utc_now()
                _write_json_atomic(self._job_record_path(job_id), record)
                continue
            if backend_identity is not None:
                record["backend"] = backend_identity.model_dump()
            input_path = self.runtime_root / str(record["input_path"])
            result_path = self._job_dir(job_id) / "worker-result.json"
            process = self.process_factory(
                job_id=job_id,
                input_video=input_path,
                runtime_root=self.runtime_root,
                route_root=self.route_root,
                publish_root=self.publish_root,
                delivery_name=f"web-{job_id}",
                source_metadata=record["input"],
                backend_identity=backend_identity,
                result_path=result_path,
            )
            try:
                process.start()
            except BaseException as exc:
                record["status"] = "failed"
                record["error"] = {
                    "code": "worker_start_failed",
                    "message": str(exc),
                }
                record["updated_at"] = _utc_now()
                _write_json_atomic(self._job_record_path(job_id), record)
                continue
            record["status"] = "running"
            record["worker_started_at"] = _utc_now()
            record["updated_at"] = _utc_now()
            _write_json_atomic(self._job_record_path(job_id), record)
            self._active_job_id = job_id
            self._active_process = process
            return

    def _reap_active(self) -> None:
        process = self._active_process
        job_id = self._active_job_id
        if process is None or job_id is None:
            self._active_process = None
            self._active_job_id = None
            return
        process.join(timeout=0)
        record = self.read_job_record(job_id)
        run_record = _read_json(self._run_dir(job_id) / "run.json")
        if isinstance(run_record, Mapping) and isinstance(run_record.get("status"), str):
            record["status"] = normalize_run_status(run_record["status"])
        elif getattr(process, "exitcode", 0) not in {0, None}:
            record["status"] = "failed"
            worker_result = _read_json(self._job_dir(job_id) / "worker-result.json")
            if worker_result:
                record["error"] = {
                    "code": worker_result.get("error_code") or "worker_failed",
                    "message": str(worker_result.get("error") or "worker failed"),
                }
        else:
            worker_result = _read_json(self._job_dir(job_id) / "worker-result.json")
            if isinstance(worker_result, Mapping) and worker_result.get("status") == "failed":
                record["status"] = "failed"
                record["error"] = {
                    "code": "worker_failed",
                    "message": str(worker_result.get("error") or "worker failed"),
                }
        record["updated_at"] = _utc_now()
        _write_json_atomic(self._job_record_path(job_id), record)
        self._active_process = None
        self._active_job_id = None
        self._start_next()

    def _read_progress(self, run_dir: Path) -> Any:
        try:
            from scripts.longsplat.terminal_progress import read_progress_snapshot

            return read_progress_snapshot(run_dir)
        except (ImportError, OSError, ValueError):
            return None

    @staticmethod
    def _safe_log_path(path: Path, run_dir: Path) -> Path | None:
        if path.is_symlink() or not path.is_file():
            return None
        try:
            root = run_dir.resolve(strict=True)
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            return None
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return None
        return resolved

    @staticmethod
    def _safe_directory_path(path: Path, run_dir: Path) -> Path | None:
        if path.is_symlink() or not path.is_dir():
            return None
        try:
            root = run_dir.resolve(strict=True)
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            return None
        current = root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return None
        return resolved

    @classmethod
    def _tail_log(cls, path: Path, *, run_dir: Path, tail: int) -> list[str]:
        safe_path = cls._safe_log_path(path, run_dir)
        if safe_path is None:
            return []
        try:
            payload = safe_path.read_bytes()[-_MAX_LOG_BYTES:]
        except OSError:
            return []
        lines = payload.decode("utf-8", errors="replace").splitlines()
        return lines[-tail:]

    @classmethod
    def _stage_log_lines(cls, attempt_dir: Path, *, name: str, run_dir: Path, tail: int) -> list[str]:
        lines: list[str] = []
        for path in (attempt_dir / name, attempt_dir / "executor" / name):
            lines.extend(cls._tail_log(path, run_dir=run_dir, tail=tail))
        return lines[-tail:]

    @staticmethod
    def _latest_attempt(stage_dir: Path) -> Path | None:
        if stage_dir.is_symlink() or not stage_dir.is_dir():
            return None
        attempts = [
            item
            for item in stage_dir.iterdir()
            if item.name.startswith("attempt-") and not item.is_symlink() and item.is_dir()
        ]
        if not attempts:
            return None

        def key(item: Path) -> tuple[int, str]:
            suffix = item.name.rsplit("-", 1)[-1]
            return (int(suffix), item.name) if suffix.isdigit() else (-1, item.name)

        return sorted(attempts, key=key)[-1]

    @staticmethod
    def _stage_status(
        run_record: Mapping[str, Any],
        stage: str,
        *,
        attempt: str | None = None,
    ) -> str:
        stages = run_record.get("stages")
        entries = stages.get(stage) if isinstance(stages, Mapping) else None
        record = None
        if isinstance(entries, list) and entries:
            if attempt is None:
                record = entries[-1]
            else:
                record = next(
                    (
                        candidate
                        for candidate in reversed(entries)
                        if isinstance(candidate, Mapping) and candidate.get("attempt") == attempt
                    ),
                    None,
                )
        value = record.get("status") if isinstance(record, Mapping) else None
        normalized = str(value or "").strip().lower()
        if normalized in {"passed", "complete", "completed", "accepted"}:
            return "passed"
        if normalized in {"running", "started"}:
            return "running"
        if normalized in {"blocked"}:
            return "blocked"
        if normalized in {"failed", "error", "errored"}:
            return "failed"
        if normalized in {"stopped", "cancelled", "canceled"}:
            return "stopped"
        if normalized in {"skipped"}:
            return "skipped"
        if (
            record is None
            and run_record.get("active_stage") == stage
            and run_record.get("active_attempt") == attempt
        ):
            active_status = str(run_record.get("status") or "").strip().lower()
            if active_status in {"running", "started"}:
                return "running"
            if active_status in {"blocked", "failed", "stopped"}:
                return active_status
        return "queued"

    @staticmethod
    def _model_paths(value: Any) -> list[Path]:
        paths: list[Path] = []
        if isinstance(value, Mapping):
            candidate = value.get("model_path")
            if isinstance(candidate, str) and candidate:
                paths.append(Path(candidate))
            for child in value.values():
                paths.extend(JobManager._model_paths(child))
        elif isinstance(value, list):
            for child in value:
                paths.extend(JobManager._model_paths(child))
        return paths

    def _training_sample_ply(self, *, run_dir: Path, run_record: Mapping[str, Any]) -> Path | None:
        stages = run_record.get("stages")
        entries = stages.get("formal-training") if isinstance(stages, Mapping) else None
        if not isinstance(entries, list) or not entries:
            return None
        latest = entries[-1]
        if not isinstance(latest, Mapping) or latest.get("status") != "passed":
            return None
        result_value = latest.get("result_path")
        if not isinstance(result_value, str):
            return None
        result_path = Path(result_value)
        if not result_path.is_absolute():
            result_path = run_dir / result_path
        safe_result = self._safe_log_path(result_path, run_dir)
        result = _read_json(safe_result) if safe_result is not None else None
        for model_path in self._model_paths(result):
            if not model_path.is_absolute():
                model_path = run_dir / model_path
            safe_model_path = self._safe_directory_path(model_path, run_dir)
            if safe_model_path is None:
                continue
            point_cloud_root = self._safe_directory_path(safe_model_path / "point_cloud", run_dir)
            if point_cloud_root is None:
                continue
            try:
                iterations = [
                    item
                    for item in point_cloud_root.iterdir()
                    if item.name.startswith("iteration_") and not item.is_symlink() and item.is_dir()
                ]
            except OSError:
                continue
            iterations.sort(key=lambda item: item.name)
            for iteration in reversed(iterations):
                candidate = iteration / "point_cloud.ply"
                if self._safe_log_path(candidate, run_dir) is not None:
                    return candidate
        return None

    @staticmethod
    def _training_sample_allowed(run_record: Mapping[str, Any]) -> bool:
        if normalize_run_status(run_record.get("status")) != "blocked":
            return False
        blocked = run_record.get("blocked")
        return isinstance(blocked, Mapping) and blocked.get("stage") == "conversion"

    def get_logs(self, job_id: str, *, stage: str | None = None, tail: int = _DEFAULT_LOG_TAIL) -> JobLogs:
        self._validate_job_id(job_id)
        if tail < 1 or tail > _MAX_LOG_TAIL:
            raise ValueError(f"tail 必须在 1 到 {_MAX_LOG_TAIL} 之间")
        run_dir = self._run_dir(job_id)
        config_record = _read_json(run_dir / "config.json") or {}
        stage_order_value = config_record.get("stage_order")
        stage_order = [
            value for value in stage_order_value
            if isinstance(value, str) and value
        ] if isinstance(stage_order_value, list) else []
        if stage is not None and stage not in stage_order:
            raise ValueError("未知流水线阶段")
        run_record = _read_json(run_dir / "run.json") or {}
        selected_stages = [stage] if stage is not None else stage_order
        snapshots: list[StageLogSnapshot] = []
        for stage_id in selected_stages:
            stage_dir = run_dir / "stages" / stage_id
            attempt_dir = self._latest_attempt(stage_dir)
            if attempt_dir is None:
                active_attempt = run_record.get("active_attempt")
                if run_record.get("active_stage") == stage_id and isinstance(active_attempt, str):
                    attempt_name = active_attempt
                    attempt_dir = stage_dir / attempt_name
                else:
                    continue
            stdout_paths = (attempt_dir / "stdout.log", attempt_dir / "executor" / "stdout.log")
            stderr_paths = (attempt_dir / "stderr.log", attempt_dir / "executor" / "stderr.log")
            stdout = self._stage_log_lines(attempt_dir, name="stdout.log", run_dir=run_dir, tail=tail)
            stderr = self._stage_log_lines(attempt_dir, name="stderr.log", run_dir=run_dir, tail=tail)
            if not stdout and not stderr and stage is None:
                continue
            updated_at: str | None = None
            mtimes = [
                path.stat().st_mtime
                for path in (*stdout_paths, *stderr_paths)
                if self._safe_log_path(path, run_dir) is not None
            ]
            if mtimes:
                updated_at = datetime.fromtimestamp(max(mtimes), timezone.utc).isoformat().replace("+00:00", "Z")
            snapshots.append(
                StageLogSnapshot(
                    id=stage_id,
                    attempt=attempt_dir.name,
                    status=self._stage_status(run_record, stage_id, attempt=attempt_dir.name),
                    stdout=stdout,
                    stderr=stderr,
                    updated_at=updated_at,
                )
            )
        return JobLogs(schema_version="web-job-logs-v1", job_id=job_id, stages=snapshots)

    def _artifact_catalog(
        self,
        *,
        job_id: str,
        run_record: Mapping[str, Any],
    ) -> tuple[ArtifactCatalog, Mapping[str, Any] | None]:
        catalog = ArtifactCatalog(
            job_id=job_id,
            allowed_roots=(self.runtime_root, self.publish_root),
        )
        quality_record: Mapping[str, Any] | None = None
        published = run_record.get("published_ply")
        published_path = None
        if isinstance(published, Mapping):
            nested_published = published.get("published")
            if isinstance(nested_published, Mapping):
                published_path = nested_published.get("path")
            else:
                published_path = published.get("path")
        elif isinstance(published, str):
            published_path = published
        receipt_value = run_record.get("published_ply_receipt")
        receipt_path = Path(receipt_value) if isinstance(receipt_value, str) else None
        receipt_valid = False
        if isinstance(published_path, str) and receipt_path is not None and receipt_path.is_file():
            try:
                receipt = _read_json(receipt_path)
                if receipt is None:
                    raise RuntimeError("published PLY receipt is missing or invalid")
                allowed_roots = (self.runtime_root, self.publish_root)

                def is_allowed_regular_file(value: object) -> bool:
                    if not isinstance(value, str):
                        return False
                    candidate = Path(value)
                    if candidate.is_symlink() or not candidate.is_file():
                        return False
                    try:
                        resolved = candidate.resolve(strict=True)
                        return any(
                            resolved.is_relative_to(root.resolve(strict=True))
                            for root in allowed_roots
                        )
                    except (OSError, ValueError):
                        return False

                if not is_allowed_regular_file(str(receipt_path)):
                    raise RuntimeError("published PLY receipt is outside the artifact roots")
                for identity_name in ("source", "published"):
                    identity = receipt.get(identity_name)
                    if not isinstance(identity, Mapping) or not is_allowed_regular_file(identity.get("path")):
                        raise RuntimeError(f"published PLY receipt {identity_name} is outside the artifact roots")
                from scripts.longsplat.publisher import verify_published_ply

                verified = verify_published_ply(receipt)
                published_identity = verified.get("published")
                if not isinstance(published_identity, Mapping):
                    raise RuntimeError("published PLY receipt has no published identity")
                if Path(str(published_identity["path"])).resolve() != Path(published_path).resolve():
                    raise RuntimeError("run record and published PLY receipt point to different files")
                if isinstance(published, Mapping):
                    for field in ("sha256", "size_bytes", "vertices"):
                        if field in published and published[field] != published_identity.get(field):
                            raise RuntimeError(f"run record published PLY {field} differs from receipt")
                receipt_valid = True
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                receipt_valid = False

        if receipt_valid and isinstance(published_path, str):
            try:
                catalog.register(
                    artifact_id="published-ply",
                    kind="model",
                    format="ply",
                    path=published_path,
                    validate_gaussian=True,
                )
            except (OSError, RuntimeError, ValueError):
                pass

        if not receipt_valid and self._training_sample_allowed(run_record):
            sample_path = self._training_sample_ply(run_dir=self._run_dir(job_id), run_record=run_record)
            if sample_path is not None:
                try:
                    catalog.register(
                        artifact_id="training-sample-ply",
                        kind="sample",
                        format="ply",
                        path=sample_path,
                        # LongSplat's native training PLY uses its compact
                        # f_offset/f_anchor_feat schema. It is a useful
                        # training sample, but it is not the viewer-ready
                        # standard Gaussian PLY accepted by the delivery
                        # validator. Keep the normal path/hash containment
                        # checks while deliberately not advertising it as a
                        # converted viewer model.
                        validate_gaussian=False,
                    )
                except (OSError, RuntimeError, ValueError):
                    pass

        if receipt_valid and receipt_path is not None:
            try:
                catalog.register(
                    artifact_id="published-ply-receipt",
                    kind="receipt",
                    format="json",
                    path=receipt_path,
                )
            except (OSError, RuntimeError, ValueError):
                pass

        delivery_root = run_record.get("technical_delivery_root")
        if isinstance(delivery_root, str):
            delivery = Path(delivery_root)
            manifest = delivery / "candidate_manifest.json"
            if manifest.is_file():
                quality_record = _read_json(manifest)
            known_files = (
                ("technical-delivery-manifest", "evidence", "json", manifest),
                ("provenance", "evidence", "json", delivery / "PROVENANCE.json"),
                ("technical-report", "evidence", "md", delivery / "CANDIDATE_REPORT.md"),
                (
                    "comparison-sheet",
                    "evidence",
                    "png",
                    delivery / "fixed_gt_native_converted_contact_sheet.png",
                ),
                ("checksums", "evidence", "txt", delivery / "SHA256SUMS.txt"),
            )
            for artifact_id, kind, format, path in known_files:
                if path.is_file():
                    try:
                        catalog.register(
                            artifact_id=artifact_id,
                            kind=kind,
                            format=format,
                            path=path,
                        )
                    except (OSError, RuntimeError, ValueError):
                        continue
        return catalog, quality_record

    def get_snapshot(self, job_id: str) -> JobSnapshot:
        self._validate_job_id(job_id)
        self._reap_active_if_needed()
        job_record = self.read_job_record(job_id)
        run_dir = self._run_dir(job_id)
        run_record = _read_json(run_dir / "run.json") or {
            "run_id": job_id,
            "status": job_record.get("status", "queued"),
            "created_at": job_record.get("created_at", ""),
            "updated_at": job_record.get("updated_at", ""),
        }
        config_record = _read_json(run_dir / "config.json") or {"stage_order": []}
        catalog, quality_record = self._artifact_catalog(job_id=job_id, run_record=run_record)
        quality_payload = dict(quality_record or {})
        quality_payload.setdefault(
            "gaussian_schema_valid",
            any(item.id == "published-ply" for item in catalog.descriptors()),
        )
        return build_job_snapshot(
            job_record=job_record,
            config_record=config_record,
            run_record=run_record,
            progress=self.progress_reader(run_dir),
            artifacts=catalog.descriptors(),
            quality_record=quality_payload,
        )

    def _reap_active_if_needed(self) -> None:
        if self._active_process is not None and not self._active_process.is_alive():
            self._reap_active()

    def resolve_artifact(self, job_id: str, artifact_id: str) -> Path:
        self._validate_job_id(job_id)
        self._reap_active_if_needed()
        job_record = self.read_job_record(job_id)
        del job_record
        run_record = _read_json(self._run_dir(job_id) / "run.json")
        if run_record is None:
            raise KeyError(artifact_id)
        catalog, _ = self._artifact_catalog(job_id=job_id, run_record=run_record)
        return catalog.resolve(artifact_id)
