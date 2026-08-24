from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/longsplat/ONE_CLICK_RECONSTRUCTION_ZH.md"
ROUTE_REQUIREMENTS = ROOT / "requirements-route.txt"
DEFAULT_REQUIREMENTS = ROOT / "requirements.txt"


def _requirement_entries(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _requirement_name(entry: str) -> str:
    name = re.split(r"[<>=!~;\s]", entry, maxsplit=1)[0].strip().lower()
    return re.sub(r"[-_.]+", "-", name)


def test_route_requirements_include_core_and_exclude_optional_stacks() -> None:
    route = _requirement_entries(ROUTE_REQUIREMENTS)
    route_names = {
        _requirement_name(entry)
        for entry in route
        if not entry.startswith("-")
    }
    assert {"numpy", "plyfile"} <= route_names
    assert len(route_names & {"opencv-python", "opencv-python-headless"}) == 1

    default = _requirement_entries(DEFAULT_REQUIREMENTS)
    assert "-r requirements-route.txt" in default
    assert "pytest" in default
    forbidden = {
        "mast3r",
        "dust3r",
        "vda",
        "depth",
        "depth-anything",
        "video-depth-anything",
        "pose-search",
    }
    declared = {
        _requirement_name(entry)
        for entry in route | default
        if not entry.startswith("-")
    }
    assert declared.isdisjoint(forbidden)


def test_route_runtime_packages_are_importable() -> None:
    for module_name in ("numpy", "cv2", "plyfile"):
        importlib.import_module(module_name)


def test_chinese_user_doc_matches_the_real_one_click_parser() -> None:
    text = DOC.read_text(encoding="utf-8")
    for fragment in (
        "./video-to-3dgs /abs/path/video.mp4",
        "./video-to-3dgs 'https://example.invalid/video.mp4'",
        "--progress auto|plain|off",
        "--plan",
        "outputs/<video-stem>__<sha12>.ply",
        "多视频融合",
        "depth disabled",
        "external fixed-pose RGB-only",
    ):
        assert fragment in text
    assert "--depth-manifest" not in text
    assert "smoke_vda.json" not in text
    assert "run_experiment" not in text

    from scripts.longsplat import one_click

    help_text = one_click.build_arg_parser().format_help()
    for option in (
        "--help",
        "--name",
        "--output-dir",
        "--provider-config",
        "--max-download-bytes",
        "--plan",
        "--progress",
    ):
        assert option in help_text


def test_default_import_boundary_does_not_load_legacy_or_optional_modules() -> None:
    before = set(sys.modules)
    importlib.import_module("scripts.longsplat.one_click")
    importlib.import_module("scripts.longsplat.reconstruct_pipeline")
    loaded = set(sys.modules) - before
    forbidden = (
        "scripts.longsplat.coverage_smoke",
        "scripts.longsplat.coverage_executor",
        "scripts.longsplat.depth_bridge",
        "mast3r",
        "dust3r",
        "vda",
        "video_depth_anything",
    )
    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in loaded
        for prefix in forbidden
    )


def test_root_gitlink_and_runner_lock_match_without_network() -> None:
    from scripts.longsplat import runner

    root_gitlink = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "HEAD", "third_party/LongSplat"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert root_gitlink.split()[2] == runner.LONGSPLAT_COMMIT

    nested = ROOT / "third_party/LongSplat"
    assert nested.is_dir()
    for path, expected in runner._LONGSPLAT_SUBMODULE_LINKS.items():
        line = subprocess.run(
            ["git", "-C", str(nested), "ls-tree", "HEAD", path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert line.split()[2] == expected

    gitmodules = ROOT / ".gitmodules"
    submodule_path = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "config",
            "-f",
            str(gitmodules),
            "--get",
            "submodule.third_party/LongSplat.path",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    submodule_url = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "config",
            "-f",
            str(gitmodules),
            "--get",
            "submodule.third_party/LongSplat.url",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert submodule_path == "third_party/LongSplat"
    assert submodule_url
