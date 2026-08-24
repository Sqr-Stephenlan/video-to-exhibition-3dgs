"""Provenance helpers: file hashing, directory tree manifests, canonical JSON.

Used by the orchestrator to record complete input/output identity so
every run can be independently audited and reproduced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a single file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_json(obj: dict | list) -> str:
    """SHA-256 of *obj* serialised as canonical JSON.

    Uses ``sort_keys=True``, compact separators, UTF-8 without BOM.
    """
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_tree_manifest(root: Path) -> dict[str, Any]:
    """Write a sorted directory tree manifest.

    Returns a dict with ``files`` (list of ``{path, size, sha256}``
    sorted by relative POSIX path) and a ``tree_sha256`` digest of the
    canonical JSON representation.
    """
    entries: list[dict[str, Any]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            entries.append(
                {
                    "path": rel,
                    "size": p.stat().st_size,
                    "sha256": sha256_file(p),
                }
            )

    tree_sha = sha256_json(entries)
    return {
        "files": entries,
        "tree_sha256": tree_sha,
    }
