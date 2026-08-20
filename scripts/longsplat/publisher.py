"""Immutable publication of the final technical 3DGS PLY.

The reconstruction ledger remains the source of internal evidence.  This
module owns the separate, user-facing copy so that a published PLY can never
be a symlink into an evidence tree and an existing delivery can never be
silently overwritten.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping


class PublishError(RuntimeError):
    """The final PLY could not be safely published."""


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_NAME_LENGTH = 96


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_stem(value: str | Path | None, *, default: str = "video") -> str:
    """Return a safe, stable basename fragment for a user-facing PLY."""

    raw = "" if value is None else unicodedata.normalize("NFKC", str(value)).strip()
    safe = _SAFE_NAME_RE.sub("_", raw).strip("._-")
    return (safe or default)[:_MAX_NAME_LENGTH].rstrip("._-") or default


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = path.absolute()
    probe = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        probe /= component
        if probe.is_symlink():
            raise PublishError(f"{label} traverses a symlink: {probe}")


def _resolve_output_dir(value: str | Path) -> Path:
    raw = Path(value)
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    raw = raw.absolute()
    _reject_symlink_components(raw, "publish directory")
    resolved = raw.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise PublishError("publish directory cannot be the filesystem root")
    if resolved.exists() and not resolved.is_dir():
        raise PublishError(f"publish directory is not a directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(resolved, "publish directory")
    return resolved


def _vertex_count(path: Path) -> int:
    """Read the vertex count from an ASCII PLY header."""

    try:
        with path.open("rb") as handle:
            first = handle.readline().decode("ascii", errors="strict").strip()
            if first != "ply":
                raise ValueError("missing PLY magic")
            count: int | None = None
            for raw_line in handle:
                line = raw_line.decode("ascii", errors="strict").strip()
                if line.startswith("element vertex "):
                    count = int(line.split()[2])
                if line == "end_header":
                    break
            if count is None:
                raise ValueError("PLY header has no vertex element")
            if count < 0:
                raise ValueError("PLY vertex count is negative")
            return count
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PublishError(f"cannot read PLY vertex count: {path}: {exc}") from exc


def _identity(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PublishError(f"{label} is missing or symlinked: {path}")
    _reject_symlink_components(path, label)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "vertices": _vertex_count(path),
    }


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _publish_lock(directory: Path) -> Iterator[None]:
    """Serialize cooperative publishers in one output directory."""

    lock_path = directory / ".longsplat-publish.lock"
    if lock_path.is_symlink():
        raise PublishError(f"publish lock is symlinked: {lock_path}")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(lock_path), flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _copy_file(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("wb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())


def _same_content(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (left.get("sha256"), left.get("size_bytes")) == (right.get("sha256"), right.get("size_bytes"))


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _same_content(left, right) and left.get("vertices") == right.get("vertices")


def publish_ply(
    *,
    source_ply: str | Path,
    output_dir: str | Path,
    name: str | None = None,
) -> dict[str, Any]:
    """Atomically publish one technical PLY and return its receipt.

    The destination name is ``<safe-stem>__<source-sha12>.ply``.  Existing
    identical content is safely reused; any other collision is a hard stop.
    """

    source = Path(source_ply).absolute()
    source_identity = _identity(source, "technical PLY source")
    destination_root = _resolve_output_dir(output_dir)
    stem = sanitize_stem(name if name is not None else source.stem)
    destination = destination_root / f"{stem}__{source_identity['sha256'][:12]}.ply"
    temporary: Path | None = None

    with _publish_lock(destination_root):
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise PublishError(f"published PLY destination is symlinked: {destination}")
            existing = _identity(destination, "published PLY destination")
            if _same_content(existing, source_identity):
                return {
                    "schema_version": "longsplat-published-ply-v1",
                    "source": source_identity,
                    "published": existing,
                    "reused": True,
                    "copy_policy": {
                        "temporary_suffix": ".part",
                        "atomic_publish": "os.replace-under-exclusive-directory-lock",
                        "parent_directory_fsync": True,
                        "destination_reservation": "O_EXCL",
                        "overwrite": "never; identical SHA/size is reused",
                        "source_sha_size_recheck": True,
                    },
                }
            raise PublishError(f"published PLY collision has different content: {destination}")

        temporary = destination_root / f".{destination.name}.{uuid.uuid4().hex}.part"
        try:
            fd = os.open(str(temporary), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            _copy_file(source, temporary)
            temporary_identity = _identity(temporary, "temporary published PLY")
            if not _same_content(temporary_identity, source_identity):
                raise PublishError("technical PLY changed while it was being copied")
            current_source = _identity(source, "technical PLY source after copy")
            if not _same_content(current_source, source_identity):
                raise PublishError("technical PLY source SHA/size drifted during publication")
            reserved_destination = False
            try:
                fd = os.open(str(destination), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.close(fd)
                reserved_destination = True
            except FileExistsError:
                if destination.is_symlink():
                    raise PublishError(f"published PLY destination is symlinked: {destination}")
                existing = _identity(destination, "published PLY destination")
                if _same_content(existing, source_identity):
                    return {
                        "schema_version": "longsplat-published-ply-v1",
                        "source": source_identity,
                        "published": existing,
                        "reused": True,
                        "copy_policy": {
                            "temporary_suffix": ".part",
                            "atomic_publish": "os.replace-under-exclusive-directory-lock",
                            "parent_directory_fsync": True,
                            "destination_reservation": "O_EXCL",
                            "overwrite": "never; identical SHA/size is reused",
                            "source_sha_size_recheck": True,
                        },
                    }
                raise PublishError(f"published PLY collision has different content: {destination}")
            os.replace(temporary, destination)
            temporary = None
            reserved_destination = False
            _fsync_directory(destination_root)
            published = _identity(destination, "published PLY destination")
            if not _same_content(published, source_identity):
                raise PublishError("published PLY SHA/size verification failed")
            return {
                "schema_version": "longsplat-published-ply-v1",
                "source": source_identity,
                "published": published,
                "reused": False,
                "copy_policy": {
                    "temporary_suffix": ".part",
                    "atomic_publish": "os.replace-under-exclusive-directory-lock",
                    "parent_directory_fsync": True,
                    "destination_reservation": "O_EXCL",
                    "overwrite": "never; identical SHA/size is reused",
                    "source_sha_size_recheck": True,
                },
            }
        finally:
            if "reserved_destination" in locals() and reserved_destination:
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def verify_published_ply(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck both sides of a persisted publication receipt."""

    source = receipt.get("source")
    published = receipt.get("published")
    if not isinstance(source, Mapping) or not isinstance(published, Mapping):
        raise PublishError("published PLY receipt is missing source/published identities")
    source_now = _identity(Path(str(source.get("path", ""))), "published receipt source")
    published_now = _identity(Path(str(published.get("path", ""))), "published receipt destination")
    if not _same_identity(source_now, source) or not _same_identity(published_now, published):
        raise PublishError("published PLY receipt SHA/size/vertex identity drifted")
    return {"source": source_now, "published": published_now}
