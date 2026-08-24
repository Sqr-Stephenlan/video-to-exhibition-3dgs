"""Product-facing local-file/direct-URL entry point for LongSplat."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from .pipeline_contract import PipelineBlocked
from .publisher import PublishError, atomic_noreplace, sanitize_stem
from .terminal_progress import TerminalProgress
from .tool_provider import discover_workspace_root, load_provider_config, resolve_tool_provider


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".webm", ".wmv"}
RUNTIME_DIRNAME = ".runtime"
INCOMING_DIRNAME = "longsplat-incoming"
RUNS_DIRNAME = "longsplat-runs"
DEFAULT_MAX_DOWNLOAD_BYTES = 8 * 1024 * 1024 * 1024
MAX_DOWNLOAD_BYTES_LIMIT = 64 * 1024 * 1024 * 1024
MAX_DOWNLOAD_BYTES_ENV = "LONGSPLAT_MAX_DOWNLOAD_BYTES"


def sanitize_url_origin(value: str) -> str:
    """Return only the scheme/host/port; never retain credentials/query data."""

    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise PipelineBlocked(f"invalid direct video URL: {exc}") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise PipelineBlocked("input must be a local video path or a direct HTTP(S) video URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise PipelineBlocked("direct video URL has an invalid port") from exc
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host if port is None else f"{host}:{port}"
    return f"{parsed.scheme.lower()}://{netloc}"


def _is_url(value: str) -> bool:
    try:
        return urlsplit(value).scheme.lower() in {"http", "https"}
    except ValueError:
        return value.lower().startswith(("http://", "https://"))


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    probe = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        probe /= component
        if probe.is_symlink():
            raise PipelineBlocked(f"{label} traverses a symlink: {probe}")


def _safe_directory(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    raw = raw.absolute()
    _reject_symlink_components(raw, label)
    resolved = raw.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise PipelineBlocked(f"{label} cannot be the filesystem root")
    if resolved.exists() and not resolved.is_dir():
        raise PipelineBlocked(f"{label} is not a directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(resolved, label)
    return resolved


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise PipelineBlocked(f"cannot open incoming directory for fsync: {path}: {exc}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise PipelineBlocked(f"cannot fsync incoming directory: {path}: {exc}") from exc
    finally:
        os.close(fd)


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _source_stem_from_url(value: str) -> str:
    path_name = Path(unquote(urlsplit(value).path)).name
    stem = Path(path_name).stem if path_name else "video"
    return sanitize_stem(stem)


def _download_suffix(value: str, content_type: str | None) -> str:
    suffix = Path(unquote(urlsplit(value).path)).suffix.lower()
    if suffix and re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        return suffix
    guessed = mimetypes.guess_extension((content_type or "").split(";", 1)[0].strip())
    return guessed if guessed and re.fullmatch(r"\.[a-z0-9]{1,8}", guessed) else ".video"


def resolve_max_download_bytes(value: int | str | None = None) -> int:
    """Resolve a positive, bounded URL download limit.

    The default is deliberately finite. An explicit CLI value or environment
    override may raise it only within the hard safety ceiling; there is no
    unlimited mode.
    """

    raw: int | str = value if value is not None else os.environ.get(
        MAX_DOWNLOAD_BYTES_ENV, DEFAULT_MAX_DOWNLOAD_BYTES
    )
    if isinstance(raw, bool):
        raise PipelineBlocked("max download bytes must be a positive integer")
    try:
        limit = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise PipelineBlocked("max download bytes must be a positive integer") from exc
    if limit <= 0 or limit > MAX_DOWNLOAD_BYTES_LIMIT:
        raise PipelineBlocked(
            f"max download bytes must be between 1 and {MAX_DOWNLOAD_BYTES_LIMIT}"
        )
    return limit


def download_direct_url(
    url: str,
    *,
    incoming_root: str | Path,
    opener: Callable[..., Any] = urlopen,
    max_download_bytes: int | str | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> dict[str, Any]:
    """Download a direct video response using an append-safe ``.part`` file."""

    sanitized_origin = sanitize_url_origin(url)
    root = _safe_directory(incoming_root, "URL incoming directory")
    stem = _source_stem_from_url(url)
    size_limit = resolve_max_download_bytes(max_download_bytes)
    temporary: Path | None = None
    response_content_type: str | None = None
    redirected_origin = sanitized_origin
    declared_size: int | None = None
    try:
        request = Request(url, headers={"User-Agent": "video-to-3dgs/1"})
        with opener(request, timeout=60) as response:
            headers = getattr(response, "headers", {})
            response_content_type = headers.get("Content-Type")
            if response_content_type and response_content_type.split(";", 1)[0].strip().lower() in {
                "text/html",
                "application/xhtml+xml",
            }:
                raise PipelineBlocked(
                    "URL returned a webpage, not a direct video file; webpage URLs need an optional yt-dlp adapter"
                )
            try:
                redirect_value = response.geturl()
            except AttributeError:
                redirected_origin = sanitized_origin
            else:
                # A final redirect to file://, javascript:, or another
                # non-HTTP(S) scheme is a hard stop. Do not fall back to the
                # original origin after validation has rejected the target.
                redirected_origin = sanitize_url_origin(str(redirect_value))
            declared_value = headers.get("Content-Length")
            if declared_value is not None:
                try:
                    declared_size = int(str(declared_value).strip())
                except (TypeError, ValueError) as exc:
                    raise PipelineBlocked("direct video URL returned an invalid Content-Length") from exc
                if declared_size < 0:
                    raise PipelineBlocked("direct video URL returned a negative Content-Length")
                if declared_size > size_limit:
                    raise PipelineBlocked(
                        f"direct video URL exceeds the {size_limit}-byte download limit"
                    )
            if progress_callback is not None:
                progress_callback(0, declared_size)
            temporary = root / f".{stem}.{uuid.uuid4().hex}.part"
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(str(temporary), flags, 0o644)
            downloaded_size = 0
            with os.fdopen(fd, "wb") as destination:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise PipelineBlocked("direct video URL returned a non-byte stream")
                    downloaded_size += len(chunk)
                    if downloaded_size > size_limit:
                        raise PipelineBlocked(
                            f"direct video URL exceeded the {size_limit}-byte download limit"
                        )
                    destination.write(chunk)
                    if progress_callback is not None:
                        progress_callback(downloaded_size, declared_size)
                destination.flush()
                os.fsync(destination.fileno())
            if declared_size is not None and declared_size != downloaded_size:
                raise PipelineBlocked(
                    "direct video URL Content-Length does not match the received byte count"
                )
        if temporary.stat().st_size <= 0:
            raise PipelineBlocked(f"direct video URL returned an empty file from {sanitized_origin}")
        sha256, size_bytes = _sha256_and_size(temporary)
        if size_bytes != downloaded_size or size_bytes > size_limit:
            raise PipelineBlocked("direct video URL byte-count verification failed")
        if temporary.stat().st_nlink != 1:
            raise PipelineBlocked(f"incoming temporary download is not an independent file: {temporary}")
        final = root / f"{stem}__{sha256[:12]}{_download_suffix(url, response_content_type)}"
        if final.exists() or final.is_symlink():
            if final.is_symlink() or not final.is_file():
                raise PipelineBlocked(f"download destination is not a regular file: {final}")
            if final.stat().st_nlink != 1:
                raise PipelineBlocked(f"download destination must have st_nlink=1: {final}")
            existing_sha, existing_size = _sha256_and_size(final)
            if (existing_sha, existing_size) != (sha256, size_bytes):
                raise PipelineBlocked(f"download destination collision has different content: {final}")
            temporary.unlink()
            temporary = None
        else:
            try:
                atomic_noreplace(temporary, final)
            except FileExistsError:
                if final.is_symlink() or not final.is_file() or final.stat().st_nlink != 1:
                    raise PipelineBlocked(f"download destination is not an independent regular file: {final}")
                existing_sha, existing_size = _sha256_and_size(final)
                if (existing_sha, existing_size) != (sha256, size_bytes):
                    raise PipelineBlocked(f"download destination collision has different content: {final}")
                temporary.unlink()
            except PublishError as exc:
                raise PipelineBlocked(str(exc)) from exc
            temporary = None
            _fsync_directory(root)
        if final.is_symlink() or not final.is_file() or final.stat().st_nlink != 1:
            raise PipelineBlocked(f"download destination final verification failed: {final}")
        final_sha, final_size = _sha256_and_size(final)
        if (final_sha, final_size) != (sha256, size_bytes):
            raise PipelineBlocked(f"download destination SHA/size verification failed: {final}")
        return {
            "kind": "direct-http-url",
            "source_url_sanitized": sanitized_origin,
            "redirected_url_sanitized": redirected_origin,
            "query_and_fragment_recorded": False,
            "path": str(final.resolve()),
            "download_sha256": sha256,
            "download_size_bytes": size_bytes,
            "content_type": response_content_type,
        }
    except PipelineBlocked:
        raise
    except Exception as exc:
        raise PipelineBlocked(
            f"direct video URL download failed for {sanitized_origin}: {type(exc).__name__}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def validate_downloaded_video(
    path: str | Path,
    *,
    route_root: str | Path,
    tool_paths: Mapping[str, str | Path] | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Run a no-shell ffprobe check before handing a URL file to the chain."""

    raw_source = Path(path)
    if raw_source.is_symlink():
        raise PipelineBlocked(f"downloaded video is missing or symlinked: {raw_source}")
    source = raw_source.resolve()
    if not source.is_file():
        raise PipelineBlocked(f"downloaded video is missing or symlinked: {source}")
    effective, _provider = resolve_tool_provider(route_root, tool_paths)
    argv = [
        effective["ffprobe"],
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "json",
        str(source),
    ]
    try:
        result = runner(
            argv,
            cwd=str(Path(route_root).resolve()),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise PipelineBlocked(f"ffprobe is unavailable for the downloaded video: {type(exc).__name__}") from exc
    if int(getattr(result, "returncode", 1)) != 0:
        raise PipelineBlocked("downloaded URL is not a directly probeable video file")
    try:
        payload = json.loads(str(getattr(result, "stdout", "")))
    except json.JSONDecodeError as exc:
        raise PipelineBlocked("ffprobe returned invalid JSON for the downloaded video") from exc
    streams = payload.get("streams") if isinstance(payload, Mapping) else None
    if not isinstance(streams, list) or not any(
        isinstance(item, Mapping) and item.get("codec_type") == "video" for item in streams
    ):
        raise PipelineBlocked("downloaded URL does not contain a video stream")
    return {"status": "passed", "ffprobe": argv[:-1], "video_stream": True}


def preflight_gpu_backend(
    *,
    route_root: str | Path,
    tool_paths: Mapping[str, str | Path] | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Fail fast on the user machine before the GPU stages are entered."""

    effective, _provider = resolve_tool_provider(route_root, tool_paths)
    commands = [
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        [
            effective["backend_python"],
            "-c",
            "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'",
        ],
    ]
    for argv in commands:
        try:
            result = runner(
                argv,
                cwd=str(Path(route_root).resolve()),
                shell=False,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            raise PipelineBlocked(f"GPU preflight executable is unavailable: {argv[0]}") from exc
        if int(getattr(result, "returncode", 1)) != 0:
            raise PipelineBlocked(f"GPU preflight failed at {argv[0]}")
    return {"status": "passed", "checks": ["nvidia-smi", "backend-python-torch-cuda"]}


def _local_source(path_value: str | Path) -> dict[str, Any]:
    raw_source = Path(path_value).expanduser()
    if raw_source.is_symlink():
        raise PipelineBlocked(f"local input video is missing or symlinked: {raw_source}")
    source = raw_source.resolve()
    if not source.is_file():
        raise PipelineBlocked(f"local input video is missing or symlinked: {source}")
    sha256, size_bytes = _sha256_and_size(source)
    return {
        "kind": "local-path",
        "path": str(source),
        "source_sha256": sha256,
        "source_size_bytes": size_bytes,
        "source_url_sanitized": None,
    }


def _next_run_id(run_root: Path, stem: str, source_sha256: str) -> str:
    run_root = _safe_directory(run_root, "run directory")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = sanitize_stem(f"{stem}__{timestamp}__{source_sha256[:12]}")
    candidate = base
    index = 1
    while (run_root / candidate).exists() or (run_root / candidate).is_symlink():
        candidate = f"{base}__{index:02d}"
        index += 1
    reservation = run_root / f".{candidate}.reservation"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(str(reservation), flags, 0o600)
    except FileExistsError:
        # A concurrent invocation reserved this exact timestamped identity;
        # advance to a fresh suffix rather than sharing a run directory.
        return _next_run_id(run_root, stem, source_sha256)
    os.close(fd)
    return candidate


def _release_run_id_reservation(run_root: Path, run_id: str) -> None:
    reservation = run_root / f".{run_id}.reservation"
    try:
        reservation.unlink()
    except FileNotFoundError:
        pass


def _provider_overrides(route: Path, config_path: str | None) -> dict[str, str]:
    selected = Path(config_path) if config_path else route / "configs/provider.local.json"
    if not selected.is_absolute():
        selected = route / selected
    if not selected.is_file():
        return {}
    return load_provider_config(selected)


def _runtime_root(route: str | Path) -> Path:
    route_path = Path(route).resolve()
    workspace_root = discover_workspace_root(route_path)
    runtime_base = workspace_root or route_path
    return _safe_directory(runtime_base / RUNTIME_DIRNAME, "runtime directory")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video-to-3dgs",
        description="Run the canonical LongSplat chain for a local video or direct HTTP(S) video URL.",
    )
    parser.add_argument("input", help="local video path or direct HTTP(S) video file URL")
    parser.add_argument("--name", help="safe public PLY stem; defaults to the video stem")
    parser.add_argument("--output-dir", default="outputs", help="public PLY directory (default: ./outputs)")
    parser.add_argument("--provider-config", help="optional ignored JSON provider override file")
    parser.add_argument(
        "--max-download-bytes",
        type=int,
        help=f"direct URL byte limit (default: {DEFAULT_MAX_DOWNLOAD_BYTES}; env: {MAX_DOWNLOAD_BYTES_ENV})",
    )
    parser.add_argument("--plan", action="store_true", help="record a CPU-only canonical plan without GPU execution")
    parser.add_argument(
        "--progress",
        choices=("auto", "plain", "off"),
        default="auto",
        help="stderr progress observer mode (default: auto)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    route = Path(__file__).resolve().parents[2]
    runtime_root = _runtime_root(route)
    progress = TerminalProgress(args.progress)
    progress.start()
    incoming_root = runtime_root / INCOMING_DIRNAME
    run_root = runtime_root / RUNS_DIRNAME
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = route / output_dir
    overrides = _provider_overrides(route, args.provider_config)
    run_reservation: tuple[Path, str] | None = None

    try:
        if _is_url(args.input):
            source_metadata = download_direct_url(
                args.input,
                incoming_root=incoming_root,
                max_download_bytes=args.max_download_bytes,
                progress_callback=progress.download_callback,
            )
            source = Path(source_metadata["path"])
            validate_downloaded_video(source, route_root=route, tool_paths=overrides)
            source_sha256 = str(source_metadata["download_sha256"])
            source_stem = _source_stem_from_url(args.input)
            source_metadata = {
                **source_metadata,
                "source_sha256": source_sha256,
                "source_size_bytes": int(source_metadata["download_size_bytes"]),
            }
        else:
            source_metadata = _local_source(args.input)
            source = Path(source_metadata["path"])
            source_sha256 = str(source_metadata["source_sha256"])
            source_stem = sanitize_stem(source.stem)
        run_id = _next_run_id(run_root, source_stem, source_sha256)
        run_reservation = (run_root, run_id)
        progress.bind_run(run_root / run_id)
        delivery_name = sanitize_stem(args.name or source_stem)

        if not args.plan:
            preflight_gpu_backend(route_root=route, tool_paths=overrides)

        from .reconstruct_pipeline import DEFAULT_PIPELINE_PROFILE, run_reconstruction

        result = run_reconstruction(
            input_video=source,
            output_root=run_root,
            run_id=run_id,
            stop_after="automated-technical-delivery",
            plan=args.plan,
            route_root=route,
            depth_source="disabled",
            execute_gpu=not args.plan,
            pipeline_profile=DEFAULT_PIPELINE_PROFILE,
            acceptance_policy="automated-technical-v1",
            tool_paths=overrides,
            publish_dir=output_dir,
            delivery_name=delivery_name,
            source_metadata=source_metadata,
        )
    except KeyboardInterrupt:
        return 130
    except (PipelineBlocked, PublishError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        progress.close()
        if run_reservation is not None:
            _release_run_id_reservation(*run_reservation)

    evidence = Path(str(result.get("run_dir", run_root / run_id))).resolve()
    if args.plan:
        print("PLAN")
        print(f"EVIDENCE: {evidence}")
        return 0
    published = result.get("published_ply")
    if not isinstance(published, Mapping) or not isinstance(published.get("published"), Mapping):
        print("ERROR: canonical chain completed without an authoritative published PLY", file=sys.stderr)
        print(f"EVIDENCE: {evidence}", file=sys.stderr)
        return 2
    published_path = Path(str(published["published"].get("path", ""))).resolve()
    print("SUCCESS")
    print(f"PLY: {published_path}")
    print(f"EVIDENCE: {evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
