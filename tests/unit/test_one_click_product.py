from __future__ import annotations

import functools
import http.server
import json
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.longsplat.one_click import (
    _local_source,
    _next_run_id,
    _runtime_root,
    download_direct_url,
    resolve_max_download_bytes,
    sanitize_url_origin,
    validate_downloaded_video,
)
from scripts.longsplat.pipeline_contract import PipelineBlocked
from scripts.longsplat.tool_provider import default_tool_paths


class _FakeResponse:
    def __init__(self, chunks: list[bytes], *, headers: dict[str, str] | None = None, final_url: str = "https://video.example/clip.mp4") -> None:
        self._chunks = iter(chunks)
        self.headers = headers or {"Content-Type": "video/mp4"}
        self._final_url = final_url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return self._final_url

    def read(self, _size):
        return next(self._chunks, b"")


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


def test_final_redirect_to_non_http_scheme_is_a_hard_stop(tmp_path: Path) -> None:
    response = _FakeResponse(
        [b"video"],
        final_url="file:///tmp/redirected.mp4?token=never-record#fragment",
    )
    with pytest.raises(PipelineBlocked, match="local video path or a direct HTTP") as error:
        download_direct_url(
            "https://video.example/start.mp4?token=never-record",
            incoming_root=tmp_path / "incoming",
            opener=lambda *_args, **_kwargs: response,
        )
    assert "never-record" not in str(error.value)
    assert not list((tmp_path / "incoming").glob("*.part"))


def test_declared_content_length_over_limit_is_rejected_before_part_creation(tmp_path: Path) -> None:
    response = _FakeResponse([b"12345"], headers={"Content-Length": "5", "Content-Type": "video/mp4"})
    with pytest.raises(PipelineBlocked, match="4-byte download limit"):
        download_direct_url(
            "https://video.example/large.mp4",
            incoming_root=tmp_path / "incoming",
            max_download_bytes=4,
            opener=lambda *_args, **_kwargs: response,
        )
    assert not list((tmp_path / "incoming").glob("*.part"))


def test_streaming_download_over_limit_cleans_part(tmp_path: Path) -> None:
    response = _FakeResponse([b"123", b"456"], headers={"Content-Type": "video/mp4"})
    with pytest.raises(PipelineBlocked, match="exceeded the 5-byte download limit"):
        download_direct_url(
            "https://video.example/stream.mp4",
            incoming_root=tmp_path / "incoming",
            max_download_bytes=5,
            opener=lambda *_args, **_kwargs: response,
        )
    assert not list((tmp_path / "incoming").glob("*.part"))


def test_content_length_mismatch_cleans_part(tmp_path: Path) -> None:
    response = _FakeResponse(
        [b"1234"],
        headers={"Content-Length": "5", "Content-Type": "video/mp4"},
    )
    with pytest.raises(PipelineBlocked, match="Content-Length"):
        download_direct_url(
            "https://video.example/mismatch.mp4",
            incoming_root=tmp_path / "incoming",
            opener=lambda *_args, **_kwargs: response,
        )
    assert not list((tmp_path / "incoming").glob("*.part"))


def test_download_limit_has_bounded_default_and_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LONGSPLAT_MAX_DOWNLOAD_BYTES", raising=False)
    assert resolve_max_download_bytes() == 8 * 1024 * 1024 * 1024
    monkeypatch.setenv("LONGSPLAT_MAX_DOWNLOAD_BYTES", "1234")
    assert resolve_max_download_bytes() == 1234
    with pytest.raises(PipelineBlocked, match="between 1"):
        resolve_max_download_bytes("0")


def test_concurrent_url_downloads_use_atomic_no_replace_and_clean_parts(tmp_path: Path) -> None:
    url = "https://video.example/concurrent.mp4?token=never-record"

    def opener(*_args, **_kwargs):
        return _FakeResponse([b"same-video"], headers={"Content-Type": "video/mp4"}, final_url=url)

    def download() -> dict[str, object]:
        return download_direct_url(url, incoming_root=tmp_path / "incoming", opener=opener)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _item: download(), range(2)))
    assert {item["path"] for item in results} == {
        results[0]["path"],
    }
    destination = Path(str(results[0]["path"]))
    assert destination.is_file()
    assert destination.stat().st_nlink == 1
    assert not list(destination.parent.glob("*.part"))
    assert all("never-record" not in json.dumps(item) for item in results)


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
