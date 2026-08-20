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
from .publisher import PublishError, sanitize_stem
from .tool_provider import discover_workspace_root, load_provider_config, resolve_tool_provider


VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".webm", ".wmv"}
RUNTIME_DIRNAME = ".runtime"
INCOMING_DIRNAME = "longsplat-incoming"
RUNS_DIRNAME = "longsplat-runs"


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
    except OSError:
        return
    try:
        os.fsync(fd)
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


def download_direct_url(
    url: str,
    *,
    incoming_root: str | Path,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Download a direct video response using an append-safe ``.part`` file."""

    sanitized_origin = sanitize_url_origin(url)
    root = _safe_directory(incoming_root, "URL incoming directory")
    stem = _source_stem_from_url(url)
    temporary: Path | None = None
    response_content_type: str | None = None
    redirected_origin = sanitized_origin
    try:
        request = Request(url, headers={"User-Agent": "video-to-3dgs/1"})
        with opener(request, timeout=60) as response:
            response_content_type = response.headers.get("Content-Type")
            if response_content_type and response_content_type.split(";", 1)[0].strip().lower() in {
                "text/html",
                "application/xhtml+xml",
            }:
                raise PipelineBlocked(
                    "URL returned a webpage, not a direct video file; webpage URLs need an optional yt-dlp adapter"
                )
            try:
                redirected_origin = sanitize_url_origin(response.geturl())
            except (AttributeError, PipelineBlocked):
                redirected_origin = sanitized_origin
            temporary = root / f".{stem}.{uuid.uuid4().hex}.part"
            fd = os.open(str(temporary), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "wb") as destination:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
        if temporary.stat().st_size <= 0:
            raise PipelineBlocked(f"direct video URL returned an empty file from {sanitized_origin}")
        sha256, size_bytes = _sha256_and_size(temporary)
        final = root / f"{stem}__{sha256[:12]}{_download_suffix(url, response_content_type)}"
        if final.exists() or final.is_symlink():
            if final.is_symlink() or not final.is_file():
                raise PipelineBlocked(f"download destination is not a regular file: {final}")
            existing_sha, existing_size = _sha256_and_size(final)
            if (existing_sha, existing_size) != (sha256, size_bytes):
                raise PipelineBlocked(f"download destination collision has different content: {final}")
            temporary.unlink()
            temporary = None
        else:
            os.replace(temporary, final)
            temporary = None
            _fsync_directory(root)
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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = sanitize_stem(f"{stem}__{timestamp}__{source_sha256[:12]}")
    candidate = base
    index = 1
    while (run_root / candidate).exists() or (run_root / candidate).is_symlink():
        candidate = f"{base}__{index:02d}"
        index += 1
    return candidate


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
    parser.add_argument("--plan", action="store_true", help="record a CPU-only canonical plan without GPU execution")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    route = Path(__file__).resolve().parents[2]
    runtime_root = _runtime_root(route)
    incoming_root = runtime_root / INCOMING_DIRNAME
    run_root = runtime_root / RUNS_DIRNAME
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = route / output_dir
    overrides = _provider_overrides(route, args.provider_config)

    try:
        if _is_url(args.input):
            source_metadata = download_direct_url(args.input, incoming_root=incoming_root)
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
    except (PipelineBlocked, PublishError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

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
