from __future__ import annotations

import functools
import http.server
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.longsplat.one_click import (
    _local_source,
    _next_run_id,
    _runtime_root,
    download_direct_url,
    sanitize_url_origin,
    validate_downloaded_video,
)
from scripts.longsplat.pipeline_contract import PipelineBlocked
from scripts.longsplat.tool_provider import default_tool_paths


def test_sanitize_url_origin_removes_credentials_query_and_fragment() -> None:
    assert sanitize_url_origin("https://user:secret@example.test:8443/video.mp4?token=abc#frag") == "https://example.test:8443"


def test_direct_url_fixture_uses_part_file_and_records_no_query(tmp_path: Path) -> None:
    fixture_root = tmp_path / "http-root"
    fixture_root.mkdir()
    (fixture_root / "clip.mp4").write_bytes(b"fixture-video")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(fixture_root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/clip.mp4?token=never-record"
        result = download_direct_url(url, incoming_root=tmp_path / "runtime" / "incoming")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    downloaded = Path(result["path"])
    assert downloaded.is_file()
    assert result["source_url_sanitized"] == f"http://127.0.0.1:{server.server_port}"
    assert result["query_and_fragment_recorded"] is False
    assert "never-record" not in json.dumps(result)
    assert not list(downloaded.parent.glob("*.part"))


def test_webpage_url_is_rejected_without_recording_query(tmp_path: Path) -> None:
    class HtmlResponse:
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def geturl(self):
            return "https://video.example/watch?token=never-record"

        def read(self, _size):
            return b"<html>not a direct video</html>"

    with pytest.raises(PipelineBlocked, match="optional yt-dlp adapter") as error:
        download_direct_url(
            "https://video.example/watch?token=never-record",
            incoming_root=tmp_path / "incoming",
            opener=lambda *_args, **_kwargs: HtmlResponse(),
        )
    assert "never-record" not in str(error.value)
    assert not list((tmp_path / "incoming").glob("*.part"))


def test_downloaded_url_probe_uses_argv_without_url(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    seen: dict[str, object] = {}

    def runner(argv, **kwargs):
        seen["argv"] = argv
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [{"codec_type": "video"}]}), stderr="")

    result = validate_downloaded_video(
        video,
        route_root=tmp_path,
        tool_paths={"ffprobe": "/bin/true"},
        runner=runner,
    )
    assert result["video_stream"] is True
    assert seen["shell"] is False
    assert str(video) in seen["argv"]
    assert all("http" not in str(item) for item in seen["argv"])


def test_local_path_identity_and_run_id_are_safe_and_noncolliding(tmp_path: Path) -> None:
    source = tmp_path / "gallery clip.mp4"
    source.write_bytes(b"local")
    metadata = _local_source(source)
    run_root = tmp_path / "runs"
    run_root.mkdir()
    first = _next_run_id(run_root, source.stem, metadata["source_sha256"])
    (run_root / first).mkdir()
    second = _next_run_id(run_root, source.stem, metadata["source_sha256"])
    assert first != second
    assert " " not in first
    assert metadata["source_url_sanitized"] is None


def test_provider_discovery_uses_workspace_ancestor(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    route = workspace / "worktrees" / "production"
    route.mkdir(parents=True)
    media = workspace / "backend-envs" / "media-tools" / "bin"
    backend = workspace / "backend-envs" / "longsplat-cu128" / "bin"
    media.mkdir(parents=True)
    backend.mkdir(parents=True)
    for path in (media / "ffmpeg", media / "ffprobe", backend / "python"):
        path.write_text("#!/bin/sh\n", encoding="utf-8")
        path.chmod(0o755)
    paths = default_tool_paths(route)
    assert paths["ffmpeg"] == str(media / "ffmpeg")
    assert paths["ffprobe"] == str(media / "ffprobe")
    assert paths["backend_python"] == str(backend / "python")
    assert _runtime_root(route) == workspace / ".runtime"
