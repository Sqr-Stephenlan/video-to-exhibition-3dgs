from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORK_URL = "https://github.com/Sqr-Stephenlan/LongSplat.git"
FORK_BRANCH = "research/external-fixed-pose-clean-closure"
NESTED_COMMIT = "c6496dc43c4ced6d072e3896fd1b172f6658b259"
NESTED_TREE = "d240c4113a95f632b58b56f3d197160e4dab24cc"


def test_gitmodules_gitlink_runner_and_manifest_share_nested_distribution() -> None:
    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert '[submodule "third_party/LongSplat"]' in gitmodules
    assert f"\turl = {FORK_URL}" in gitmodules
    assert f"\tbranch = {FORK_BRANCH}" in gitmodules

    runner = (ROOT / "scripts/longsplat/runner.py").read_text(encoding="utf-8")
    assert 'LONGSPLAT_REPO_URL = "https://github.com/Sqr-Stephenlan/LongSplat"' in runner
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


def test_fresh_distribution_verifier_is_explicit_and_non_recursive() -> None:
    verifier = (ROOT / "scripts/longsplat/verify_remote_distribution.py").read_text(encoding="utf-8")
    assert '"submodule", "update"' in verifier
    assert "--init" in verifier
    assert "--recursive" not in verifier
    assert "optional_descendants_materialized" in verifier
    assert "_LONGSPLAT_SUBMODULE_LINKS" in verifier
