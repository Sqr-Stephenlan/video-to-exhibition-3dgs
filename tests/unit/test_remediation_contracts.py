from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.longsplat.colmap_contract import build_colmap_commands
from scripts.longsplat.pipeline_contract import (
    PipelineBlocked,
    build_run_identity,
    classify_stage_migration,
    preflight_dependencies,
    stable_sha256,
)
from scripts.longsplat.reconstruct_pipeline import _producer_resume_allowed
from scripts.longsplat.tool_provider import resolve_tool_provider


def _sha(letter: str) -> str:
    return letter * 64


def test_supported_camera_matcher_matrix_and_structured_stop(tmp_path: Path) -> None:
    for model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "PINHOLE"):
        for matcher in ("sequential", "exhaustive"):
            plan = build_colmap_commands(
                colmap=tmp_path / "colmap",
                workspace=tmp_path / f"{model}-{matcher}",
                image_dir=tmp_path / "images",
                source_video_sha256=_sha("a"),
                canonical_media_sha256=_sha("b"),
                canonical_width=640,
                canonical_height=480,
                camera_model=model,
                matching=matcher,
            )
            feature = plan["commands"][0]["argv"]
            matching_argv = plan["commands"][1]["argv"]
            assert model in feature
            assert matching_argv[1] == f"{matcher}_matcher"

    with pytest.raises(PipelineBlocked, match="unsupported COLMAP camera model"):
        build_colmap_commands(
            colmap=tmp_path / "colmap",
            workspace=tmp_path / "bad-model",
            image_dir=tmp_path / "images",
            source_video_sha256=_sha("a"),
            canonical_media_sha256=_sha("b"),
            canonical_width=640,
            canonical_height=480,
            camera_model="OPENCV",
        )
    with pytest.raises(PipelineBlocked, match="unsupported COLMAP matching mode"):
        build_colmap_commands(
            colmap=tmp_path / "colmap",
            workspace=tmp_path / "bad-matcher",
            image_dir=tmp_path / "images",
            source_video_sha256=_sha("a"),
            canonical_media_sha256=_sha("b"),
            canonical_width=640,
            canonical_height=480,
            matching="vocab-tree",
        )


def test_two_external_tool_roots_are_explicit_and_argv_safe(tmp_path: Path) -> None:
    roots = []
    records = []
    for index in (1, 2):
        root = tmp_path / f"provider-{index}" / "bin"
        root.mkdir(parents=True)
        for name in ("ffmpeg", "ffprobe", "colmap", "python"):
            path = root / name
            path.write_text(f"provider-{index}-{name}\n", encoding="utf-8")
            path.chmod(0o755)
        roots.append(root)

        def runner(argv, **kwargs):
            assert isinstance(argv, list)
            assert kwargs.get("shell") is False
            return SimpleNamespace(returncode=0, stdout=f"version-{index}\n", stderr="")

        result = preflight_dependencies(
            route_python=root / "python",
            backend_python=root / "python",
            ffmpeg=root / "ffmpeg",
            ffprobe=root / "ffprobe",
            colmap=root / "colmap",
            runner=runner,
        )
        assert result["status"] == "passed"
        assert all(entry["executable"] for entry in result["dependencies"].values())
        records.append(result)
    assert records[0]["tool_identity_sha256"] != records[1]["tool_identity_sha256"]
    assert records[0]["dependencies"]["ffmpeg"]["sha256"] != records[1]["dependencies"]["ffmpeg"]["sha256"]

    _, provider_a = resolve_tool_provider(tmp_path / "route", {"ffmpeg": roots[0] / "ffmpeg"})
    _, provider_b = resolve_tool_provider(tmp_path / "route", {"ffmpeg": roots[1] / "ffmpeg"})
    assert provider_a["provider_identity_sha256"] != provider_b["provider_identity_sha256"]


def test_producer_resume_ignores_docs_but_rejects_provider_and_camera_drift() -> None:
    config = {
        "schema_version": "reconstruct-pipeline-config-v2",
        "camera_model": "SIMPLE_RADIAL",
        "matching": "sequential",
        "tool_provider": {"provider_identity_sha256": _sha("t")},
    }
    files = [
        {"path": "scripts/longsplat/raw_pipeline.py", "sha256": _sha("r"), "size_bytes": 1},
        {"path": "docs/longsplat/README.md", "sha256": _sha("d"), "size_bytes": 1},
    ]
    old_code = {"files": files}
    new_code = {"files": [files[0], {**files[1], "sha256": _sha("e")}]}
    old = build_run_identity(
        source_video_sha256=_sha("a"),
        source_video_path="/tmp/video.mp4",
        source_video_size_bytes=10,
        canonical_config_sha256=stable_sha256(config),
        tool_identity_sha256=_sha("t"),
        code_identity_value=old_code,
    )
    current = build_run_identity(
        source_video_sha256=_sha("a"),
        source_video_path="/tmp/video.mp4",
        source_video_size_bytes=10,
        canonical_config_sha256=stable_sha256(config),
        tool_identity_sha256=_sha("t"),
        code_identity_value=new_code,
    )
    assert _producer_resume_allowed(
        existing_identity=old,
        current_identity=current,
        existing_config=config,
        current_config=config,
    )

    provider_drift = {**config, "tool_provider": {"provider_identity_sha256": _sha("u")}}
    provider_identity = build_run_identity(
        source_video_sha256=_sha("a"),
        source_video_path="/tmp/video.mp4",
        source_video_size_bytes=10,
        canonical_config_sha256=stable_sha256(provider_drift),
        tool_identity_sha256=_sha("u"),
        code_identity_value=new_code,
    )
    assert not _producer_resume_allowed(
        existing_identity=old,
        current_identity=provider_identity,
        existing_config=config,
        current_config=provider_drift,
    )

    camera_drift = {**config, "camera_model": "PINHOLE"}
    camera_identity = build_run_identity(
        source_video_sha256=_sha("a"),
        source_video_path="/tmp/video.mp4",
        source_video_size_bytes=10,
        canonical_config_sha256=stable_sha256(camera_drift),
        tool_identity_sha256=_sha("t"),
        code_identity_value=new_code,
    )
    assert not _producer_resume_allowed(
        existing_identity=old,
        current_identity=camera_identity,
        existing_config=config,
        current_config=camera_drift,
    )


def test_stage_schema_migration_is_explicit() -> None:
    exact = {"status": "passed", "source_sha": "a", "camera_sha": "b", "schema_version": "v1"}
    expected = {**exact}
    assert classify_stage_migration(exact, expected, immutable_fields=("source_sha", "camera_sha"))["classification"] == "reusable_exact"
    stale = {**exact, "consumer_code_identity": "old"}
    expected_stale = {**exact, "consumer_code_identity": "new"}
    assert classify_stage_migration(stale, expected_stale, immutable_fields=("source_sha", "camera_sha"))["classification"] == "stale_rebuildable"
    unsafe = {**exact, "camera_sha": "changed"}
    assert classify_stage_migration(unsafe, expected, immutable_fields=("source_sha", "camera_sha"))["classification"] == "reject_unsafe_drift"
