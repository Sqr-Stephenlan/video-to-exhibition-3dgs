from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORK_URL = "https://github.com/Sqr-Stephenlan/LongSplat.git"
FORK_BRANCH = "research/external-fixed-pose-clean-closure"
NESTED_COMMIT = "bf766eb903c3d9144d64088b8c80b2da67d39411"
NESTED_TREE = "79f232c42e9128092df0520c5b0209d7ac2b08fd"


def test_gitmodules_gitlink_runner_and_manifest_share_nested_distribution() -> None:
    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert '[submodule "third_party/LongSplat"]' in gitmodules
    assert f"\turl = {FORK_URL}" in gitmodules
    assert f"\tbranch = {FORK_BRANCH}" in gitmodules

    runner = (ROOT / "scripts/longsplat/runner.py").read_text(encoding="utf-8")
    assert f'LONGSPLAT_COMMIT = "{NESTED_COMMIT}"' in runner

    gitlink = subprocess.run(
        ["git", "ls-tree", "HEAD", "third_party/LongSplat"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert f"160000 commit {NESTED_COMMIT}\tthird_party/LongSplat" in gitlink

    manifest = (ROOT / "docs/longsplat/REPLAY_MANIFEST_CLEAN_CLOSURE_20260820.md").read_text(
        encoding="utf-8"
    )
    assert f"fork: https://github.com/Sqr-Stephenlan/LongSplat" in manifest
    assert f"fork branch: {FORK_BRANCH}" in manifest
    assert f"published nested commit: {NESTED_COMMIT}" in manifest
    assert f"published nested tree: {NESTED_TREE}" in manifest
