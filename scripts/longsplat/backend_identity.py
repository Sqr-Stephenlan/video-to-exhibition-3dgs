"""Dependency-light identity checks for a LongSplat backend checkout.

This module intentionally imports only the Python standard library.  Raw
conversion/reconversion can bind a backend checkout without importing the
legacy orchestrator, whose module graph includes optional depth, quality, and
telemetry integrations.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class BackendIdentityError(RuntimeError):
    """A backend identity query failed closed."""


def resolve_backend_identity(repo_root: Path, *, backend_mode: str) -> dict[str, Any]:
    """Return commit, dirty, submodule, and local-patch identity evidence.

    The checks deliberately mirror the historical research-local contract,
    including recursive submodule diff fingerprints and containment checks.
    No LongSplat, depth, quality, or telemetry module is imported here.
    """

    commit_result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        raise BackendIdentityError(
            f"Failed to get backend HEAD commit: exit {commit_result.returncode}"
        )
    commit = commit_result.stdout.strip()
    if not _SHA_RE.match(commit):
        raise BackendIdentityError(f"Backend commit is not a valid SHA: {commit!r}")

    dirty_result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if dirty_result.returncode != 0:
        raise BackendIdentityError(
            f"git status failed with exit {dirty_result.returncode}: "
            f"{dirty_result.stderr.strip()[:500]}"
        )
    dirty = bool(dirty_result.stdout.strip())

    sub_list_result = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "submodule",
            "foreach",
            "--recursive",
            "--quiet",
            "echo $displaypath",
        ],
        capture_output=True,
        text=True,
    )
    if sub_list_result.returncode != 0:
        raise BackendIdentityError(
            f"git submodule foreach failed with exit "
            f"{sub_list_result.returncode}: "
            f"{sub_list_result.stderr.strip()[:500]}"
        )
    sub_paths = [
        path.strip()
        for path in sub_list_result.stdout.strip().splitlines()
        if path.strip()
    ]

    canonical_root = repo_root.resolve()
    submodules: dict[str, str] = {}
    resolved_sub_paths: dict[str, Path] = {}
    for sub_path in sub_paths:
        resolved = (canonical_root / sub_path).resolve()
        try:
            resolved.relative_to(canonical_root)
        except ValueError as exc:
            raise BackendIdentityError(
                f"Submodule path {sub_path} resolves outside repo root "
                f"{canonical_root}: {resolved}"
            ) from exc
        if not resolved.is_dir():
            raise BackendIdentityError(
                f"Submodule path reported by git foreach does not exist: {sub_path}"
            )
        sha_result = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if sha_result.returncode != 0:
            raise BackendIdentityError(
                f"Failed to get HEAD for submodule {sub_path}: "
                f"exit {sha_result.returncode} — "
                f"{sha_result.stderr.strip()[:500]}"
            )
        sha = sha_result.stdout.strip()
        if not _SHA_RE.match(sha):
            raise BackendIdentityError(
                f"Submodule {sub_path} HEAD is not a valid SHA: {sha!r}"
            )
        submodules[sub_path] = sha
        resolved_sub_paths[sub_path] = resolved

    diff_sha256: str | None = None
    submodule_diffs: dict[str, str | None] = {}
    if backend_mode == "research_local":
        diff_result = subprocess.run(
            ["git", "-C", str(canonical_root), "diff", "--binary", "HEAD", "--"],
            capture_output=True,
        )
        if diff_result.returncode != 0:
            raise BackendIdentityError("Failed to fingerprint backend diff")
        diff_sha256 = (
            hashlib.sha256(diff_result.stdout).hexdigest()
            if diff_result.stdout
            else None
        )
        if dirty and diff_sha256 is None:
            raise BackendIdentityError(
                "research_local backend is dirty but git diff produced no output"
            )

        for sub_path in sub_paths:
            expected = resolved_sub_paths[sub_path]
            current = (canonical_root / sub_path).resolve()
            try:
                current.relative_to(canonical_root)
            except ValueError as exc:
                raise BackendIdentityError(
                    f"Submodule path {sub_path} resolves outside repo root "
                    f"during diff query: {current}"
                ) from exc
            if current != expected:
                raise BackendIdentityError(
                    f"Submodule path {sub_path} changed between HEAD and diff "
                    f"queries: was {expected}, now {current}"
                )
            if not current.is_dir():
                raise BackendIdentityError(
                    f"Submodule path vanished between HEAD and diff queries: {sub_path}"
                )
            diff = subprocess.run(
                ["git", "-C", str(current), "diff", "--binary", "HEAD", "--"],
                capture_output=True,
            )
            if diff.returncode != 0:
                raise BackendIdentityError(
                    f"Failed to fingerprint submodule diff for {sub_path}"
                )
            submodule_diffs[sub_path] = (
                hashlib.sha256(diff.stdout).hexdigest() if diff.stdout else None
            )

    return {
        "commit": commit,
        "dirty": dirty,
        "submodules": submodules,
        "mode": backend_mode,
        "diff_sha256": diff_sha256,
        "submodule_diffs": submodule_diffs,
    }
