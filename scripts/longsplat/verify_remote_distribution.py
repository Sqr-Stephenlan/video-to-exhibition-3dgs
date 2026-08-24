"""Materialize and verify the locked LongSplat distribution from Git remotes.

This is intentionally explicit rather than recursive: it initializes the root
LongSplat gitlink and the four direct gitlinks required by the locked runner.
Optional MASt3R/DUSt3R descendants are neither fetched nor silently accepted.
The command is intended for a fresh checkout or a checkout already at the
exact immutable commits; it never resets or force-checks-out an unexpected
working tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .runner import LONGSPLAT_COMMIT, LONGSPLAT_REPO_URL, _LONGSPLAT_SUBMODULE_LINKS


ROOT_SUBMODULE = "third_party/LongSplat"
ROOT_SUBMODULE_URL = f"{LONGSPLAT_REPO_URL}.git"
ROOT_SUBMODULE_BRANCH = "research/external-fixed-pose-clean-closure"
OPTIONAL_DESCENDANTS = (
    "submodules/mast3r/dust3r",
    "submodules/mast3r/dust3r/croco",
)


class DistributionError(RuntimeError):
    """The fresh remote distribution cannot be materialized safely."""


def _git(args: list[str], *, cwd: Path) -> str:
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        # Do not echo command output: a remote helper could include a URL
        # credential or other environment-specific detail.
        raise DistributionError(f"git command failed (exit {result.returncode}): git {' '.join(args[:4])}")
    return result.stdout.strip()


def _config(path: Path, key: str) -> str:
    result = subprocess.run(
        ["git", "config", "--file", str(path), "--get", key],
        cwd=str(path.parent),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise DistributionError(f"missing gitmodules setting: {key}")
    return result.stdout.strip()


def _head(path: Path) -> str:
    return _git(["-C", str(path), "rev-parse", "HEAD"], cwd=path)


def _remote(path: Path) -> str:
    return _git(["-C", str(path), "remote", "get-url", "origin"], cwd=path)


def _normalized_remote(value: str) -> str:
    result = value.strip().rstrip("/")
    if result.endswith(".git"):
        result = result[:-4]
    return result


def _is_local_remote(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        Path(value).is_absolute()
        or value.startswith(("./", "../", "~"))
        or parsed.scheme.lower() == "file"
    )


def _require_remote(actual: str, expected: str, *, label: str) -> None:
    if _is_local_remote(actual):
        raise DistributionError(f"{label} remote is a local path, not a distributable remote")
    if _normalized_remote(actual) != _normalized_remote(expected):
        raise DistributionError(f"{label} remote differs from .gitmodules")


def _gitlink_sha(superproject: Path, path: str) -> str:
    output = _git(["-C", str(superproject), "ls-tree", "HEAD", "--", path], cwd=superproject)
    fields = output.split("\t", 1)[0].split()
    if len(fields) != 3 or fields[0] != "160000" or fields[1] != "commit":
        raise DistributionError(f"{superproject}/{path} is not a gitlink in HEAD")
    return fields[2]


def _prepare_submodule(superproject: Path, path: str) -> Path:
    target = superproject / path
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise DistributionError(f"submodule path is not a normal directory: {target}")
    git_marker = target / ".git"
    if git_marker.is_symlink():
        raise DistributionError(f"submodule git marker is symlinked: {git_marker}")
    if not git_marker.exists():
        if target.exists() and any(target.iterdir()):
            raise DistributionError(f"unexpected non-empty uninitialized submodule path: {target}")
        _git(["-C", str(superproject), "submodule", "sync", "--", path], cwd=superproject)
        _git(
            ["-C", str(superproject), "submodule", "update", "--init", "--", path],
            cwd=superproject,
        )
    if not git_marker.exists():
        raise DistributionError(f"submodule did not materialize as a git repository: {target}")
    return target


def verify_and_materialize(root: str | Path) -> dict[str, Any]:
    """Materialize the exact direct distribution and return safe evidence."""

    checkout = Path(root).resolve()
    if not checkout.is_dir():
        raise DistributionError(f"checkout is not a directory: {checkout}")
    root_gitmodules = checkout / ".gitmodules"
    if not root_gitmodules.is_file() or root_gitmodules.is_symlink():
        raise DistributionError("root .gitmodules is missing or symlinked")
    root_head = _head(checkout)
    root_remote = _remote(checkout)
    if _is_local_remote(root_remote):
        raise DistributionError("root origin is a local path; fresh remote verification is required")
    configured_url = _config(root_gitmodules, "submodule.third_party/LongSplat.url")
    configured_branch = _config(root_gitmodules, "submodule.third_party/LongSplat.branch")
    _require_remote(configured_url, ROOT_SUBMODULE_URL, label="root LongSplat")
    if configured_branch != ROOT_SUBMODULE_BRANCH:
        raise DistributionError("root LongSplat branch differs from the locked distribution")
    root_gitlink = _gitlink_sha(checkout, ROOT_SUBMODULE)
    if root_gitlink != LONGSPLAT_COMMIT:
        raise DistributionError(
            f"root LongSplat gitlink mismatch: expected {LONGSPLAT_COMMIT}, got {root_gitlink}"
        )

    nested = _prepare_submodule(checkout, ROOT_SUBMODULE)
    nested_head = _head(nested)
    if nested_head != LONGSPLAT_COMMIT:
        raise DistributionError(
            f"LongSplat commit mismatch: expected {LONGSPLAT_COMMIT}, got {nested_head}"
        )
    nested_gitmodules = nested / ".gitmodules"
    if not nested_gitmodules.is_file() or nested_gitmodules.is_symlink():
        raise DistributionError("LongSplat .gitmodules is missing or symlinked")
    nested_configured_url = configured_url
    # The nested repository is expected to use the same repository identity as
    # the root gitmodules declaration; its own origin must match that fork.
    nested_remote = _remote(nested)
    _require_remote(nested_remote, ROOT_SUBMODULE_URL, label="LongSplat")
    _require_remote(nested_configured_url, ROOT_SUBMODULE_URL, label="LongSplat .gitmodules")

    direct: list[dict[str, Any]] = []
    for path, expected_sha in _LONGSPLAT_SUBMODULE_LINKS.items():
        configured = _config(nested_gitmodules, f"submodule.{path}.url")
        target = _prepare_submodule(nested, path)
        gitlink = _gitlink_sha(nested, path)
        if gitlink != expected_sha:
            raise DistributionError(
                f"LongSplat gitlink mismatch for {path}: expected {expected_sha}, got {gitlink}"
            )
        actual_sha = _head(target)
        if actual_sha != expected_sha:
            raise DistributionError(
                f"LongSplat submodule SHA mismatch for {path}: expected {expected_sha}, got {actual_sha}"
            )
        actual_remote = _remote(target)
        _require_remote(actual_remote, configured, label=f"LongSplat {path}")
        direct.append(
            {
                "path": path,
                "gitlink_sha": gitlink,
                "head": actual_sha,
                "configured_url": configured,
                "remote": actual_remote,
            }
        )

    optional_materialized = []
    for path in OPTIONAL_DESCENDANTS:
        target = nested / path
        if target.is_symlink() or (target.exists() and (target / ".git").exists()):
            optional_materialized.append(path)
    if optional_materialized:
        raise DistributionError(
            "optional MASt3R/DUSt3R descendants were materialized: "
            + ", ".join(optional_materialized)
        )
    return {
        "schema_version": "longsplat-remote-distribution-v1",
        "root": str(checkout),
        "root_head": root_head,
        "root_origin": root_remote,
        "root_gitlink": {"path": ROOT_SUBMODULE, "sha": root_gitlink},
        "longsplat": {
            "path": str(nested),
            "origin": nested_remote,
            "head": nested_head,
            "configured_url": nested_configured_url,
            "branch": ROOT_SUBMODULE_BRANCH,
        },
        "direct_submodules": direct,
        "optional_descendants_materialized": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize and verify locked LongSplat gitlinks from remote URLs"
    )
    parser.add_argument("--root", default=".", help="root checkout to materialize (default: .)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable evidence")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        evidence = verify_and_materialize(args.root)
    except (DistributionError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        print("REMOTE DISTRIBUTION VERIFIED")
        print(f"ROOT: {evidence['root']}")
        print(f"LONGSPLAT: {evidence['longsplat']['head']}")
        print(f"DIRECT SUBMODULES: {len(evidence['direct_submodules'])}")
        print("OPTIONAL DESCENDANTS: not materialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
