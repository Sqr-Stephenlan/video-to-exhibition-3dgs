"""Immutable publication of the final technical 3DGS PLY.

The reconstruction ledger remains the source of internal evidence.  This
module owns the separate, user-facing copy so that a published PLY can never
be a symlink or hardlink into an evidence tree and an existing delivery can
never be silently overwritten.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Iterator, Mapping


class PublishError(RuntimeError):
    """The final PLY or its authoritative receipt could not be safely handled."""


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_NAME_LENGTH = 96
_RENAME_NOREPLACE = 1
_PUBLISH_SCHEMA = "longsplat-published-ply-v2"


def _copy_policy() -> dict[str, Any]:
    return {
        "publisher_version": "longsplat-publisher-v2",
        "temporary_suffix": ".part",
        "atomic_publish": "renameat2(RENAME_NOREPLACE)",
        "parent_directory_fsync": True,
        "destination_reservation": "none",
        "overwrite": "never; identical SHA/size/vertices is reused",
        "source_sha_size_recheck": True,
        "final_nlink": 1,
    }


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


def _resolve_output_dir(value: str | Path, *, create: bool) -> Path:
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
    if not resolved.exists():
        if not create:
            raise PublishError(f"publish directory is missing: {resolved}")
        resolved.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(resolved, "publish directory")
    return resolved


def _vertex_count(path: Path) -> int:
    """Read the vertex count from an ASCII or binary PLY header."""

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


def _identity(path: Path, label: str, *, required_nlink: int | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PublishError(f"{label} is missing or symlinked: {path}")
    _reject_symlink_components(path, label)
    stat_result = path.stat()
    if required_nlink is not None and stat_result.st_nlink != required_nlink:
        raise PublishError(
            f"{label} must have st_nlink={required_nlink}; got {stat_result.st_nlink}: {path}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": stat_result.st_size,
        "vertices": _vertex_count(path),
        "nlink": stat_result.st_nlink,
    }


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise PublishError(f"cannot open publication directory for fsync: {directory}: {exc}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise PublishError(f"cannot fsync publication directory: {directory}: {exc}") from exc
    finally:
        os.close(fd)


def atomic_noreplace(source: str | Path, destination: str | Path) -> None:
    """Atomically rename without replacing an existing destination.

    Linux ``renameat2(RENAME_NOREPLACE)`` is required.  A platform without a
    strict no-replace primitive is blocked instead of falling back to a
    check-then-rename race or a hardlink-visible public file.
    """

    source_path = Path(source).absolute()
    destination_path = Path(destination).absolute()
    if source_path.parent != destination_path.parent:
        raise PublishError("atomic no-replace requires source and destination in one directory")
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise PublishError("atomic no-replace is unavailable on this platform") from exc

    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        directory_fd = os.open(str(source_path.parent), flags)
    except OSError as exc:
        raise PublishError(f"cannot open atomic publication directory: {source_path.parent}: {exc}") from exc
    try:
        result = renameat2(
            directory_fd,
            os.fsencode(source_path.name),
            directory_fd,
            os.fsencode(destination_path.name),
            _RENAME_NOREPLACE,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
    finally:
        os.close(directory_fd)
    if error_number == errno.EEXIST:
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), str(destination_path))
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise PublishError("atomic no-replace is unsupported by this filesystem")
    raise PublishError(
        f"atomic no-replace failed for {source_path} -> {destination_path}: "
        f"{os.strerror(error_number)}"
    )


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


def _create_part(path: Path) -> int:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(str(path), flags, 0o644)


def _copy_file(source: Path, destination: Path) -> None:
    flags = os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(destination), flags)
    try:
        with source.open("rb") as source_handle, os.fdopen(fd, "wb", closefd=False) as destination_handle:
            shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    finally:
        os.close(fd)


def _same_content(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (left.get("sha256"), left.get("size_bytes"), left.get("vertices")) == (
        right.get("sha256"),
        right.get("size_bytes"),
        right.get("vertices"),
    )


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _same_content(left, right) and left.get("nlink") == right.get("nlink")


def _destination(output_dir: Path, source: Mapping[str, Any], name: str | None) -> Path:
    source_path = Path(str(source.get("path", "")))
    stem = sanitize_stem(name if name is not None else source_path.stem)
    sha256 = str(source.get("sha256", ""))
    if len(sha256) < 12:
        raise PublishError("published PLY source receipt has an invalid SHA-256")
    return output_dir / f"{stem}__{sha256[:12]}.ply"


def _receipt(source: Mapping[str, Any], published: Mapping[str, Any], *, reused: bool) -> dict[str, Any]:
    return {
        "schema_version": _PUBLISH_SCHEMA,
        "source": dict(source),
        "published": dict(published),
        "reused": reused,
        "copy_policy": _copy_policy(),
    }


def publish_ply(
    *,
    source_ply: str | Path,
    output_dir: str | Path,
    name: str | None = None,
) -> dict[str, Any]:
    """Atomically publish one technical PLY and return its receipt."""

    source = Path(source_ply).absolute()
    source_identity = _identity(source, "technical PLY source")
    destination_root = _resolve_output_dir(output_dir, create=True)
    destination = _destination(destination_root, source_identity, name)
    temporary: Path | None = None

    with _publish_lock(destination_root):
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise PublishError(f"published PLY destination is symlinked: {destination}")
            existing = _identity(destination, "published PLY destination", required_nlink=1)
            if _same_content(existing, source_identity):
                return _receipt(source_identity, existing, reused=True)
            raise PublishError(f"published PLY collision has different content: {destination}")

        temporary = destination_root / f".{destination.name}.{uuid.uuid4().hex}.part"
        try:
            fd = _create_part(temporary)
            os.close(fd)
            _copy_file(source, temporary)
            temporary_identity = _identity(temporary, "temporary published PLY", required_nlink=1)
            if not _same_content(temporary_identity, source_identity):
                raise PublishError("technical PLY changed while it was being copied")
            current_source = _identity(source, "technical PLY source after copy")
            if not _same_content(current_source, source_identity):
                raise PublishError("technical PLY source SHA/size drifted during publication")
            try:
                atomic_noreplace(temporary, destination)
            except FileExistsError:
                if destination.is_symlink():
                    raise PublishError(f"published PLY destination is symlinked: {destination}")
                existing = _identity(destination, "published PLY destination", required_nlink=1)
                if _same_content(existing, source_identity):
                    return _receipt(source_identity, existing, reused=True)
                raise PublishError(f"published PLY collision has different content: {destination}")
            temporary = None
            _fsync_directory(destination_root)
            published = _identity(destination, "published PLY destination", required_nlink=1)
            if not _same_content(published, source_identity):
                raise PublishError("published PLY SHA/size/vertex verification failed")
            return _receipt(source_identity, published, reused=False)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass


def _load_receipt_identity(receipt: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    value = receipt.get(key)
    if not isinstance(value, Mapping):
        raise PublishError(f"published PLY receipt is missing {key} identity")
    for field in ("path", "sha256", "size_bytes", "vertices", "nlink"):
        if field not in value:
            raise PublishError(f"published PLY receipt {key} identity is missing {field}")
    return value


def verify_public_delivery(
    receipt: Mapping[str, Any],
    *,
    output_dir: str | Path,
    name: str | None,
) -> dict[str, Any]:
    """Verify the public PLY against its canonical configured destination."""

    if receipt.get("schema_version") != _PUBLISH_SCHEMA:
        raise PublishError(f"published PLY receipt schema must be {_PUBLISH_SCHEMA}")
    if receipt.get("copy_policy") != _copy_policy():
        raise PublishError("published PLY receipt copy policy is not the current no-replace policy")
    source = _load_receipt_identity(receipt, "source", "published receipt source")
    published = _load_receipt_identity(receipt, "published", "published receipt destination")
    source_now = _identity(Path(str(source["path"])), "published receipt source")
    published_now = _identity(
        Path(str(published["path"])),
        "published receipt destination",
        required_nlink=1,
    )
    if not _same_identity(source_now, source) or not _same_identity(published_now, published):
        raise PublishError("published PLY receipt identity drifted")
    if not _same_content(source_now, published_now):
        raise PublishError("published PLY source and public delivery differ")
    root = _resolve_output_dir(output_dir, create=False)
    expected = _destination(root, source_now, name)
    if published_now["path"] != str(expected):
        raise PublishError(
            f"published PLY path is not the canonical delivery destination: "
            f"expected {expected}, got {published_now['path']}"
        )
    return {"source": source_now, "published": published_now, "copy_policy": _copy_policy()}


def verify_published_ply(
    receipt: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Recheck a persisted receipt, optionally including canonical binding."""

    if output_dir is not None:
        return verify_public_delivery(receipt, output_dir=output_dir, name=name)
    if name is not None:
        raise PublishError("public delivery name requires output-dir for canonical verification")
    if receipt.get("schema_version") != _PUBLISH_SCHEMA:
        raise PublishError(f"published PLY receipt schema must be {_PUBLISH_SCHEMA}")
    if receipt.get("copy_policy") != _copy_policy():
        raise PublishError("published PLY receipt copy policy is not the current no-replace policy")
    source = _load_receipt_identity(receipt, "source", "published receipt source")
    published = _load_receipt_identity(receipt, "published", "published receipt destination")
    source_now = _identity(Path(str(source["path"])), "published receipt source")
    published_now = _identity(Path(str(published["path"])), "published receipt destination", required_nlink=1)
    if not _same_identity(source_now, source) or not _same_identity(published_now, published):
        raise PublishError("published PLY receipt identity drifted")
    return {"source": source_now, "published": published_now, "copy_policy": _copy_policy()}


def write_json_once_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write an immutable JSON receipt with fsync and no-replace commit."""

    target = Path(path).absolute()
    _reject_symlink_components(target.parent, "receipt directory")
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(target.parent, "receipt directory")
    encoded = json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not target.is_file():
            raise PublishError(f"immutable receipt destination is not a regular file: {target}")
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PublishError(f"immutable receipt is unreadable: {target}: {exc}") from exc
        if existing != dict(value):
            raise PublishError(f"immutable receipt differs on resume: {target}")
        return

    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.part"
    committed = False
    try:
        fd = _create_part(temporary)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            atomic_noreplace(temporary, target)
            temporary = None
            committed = True
        except FileExistsError:
            if target.is_symlink() or not target.is_file():
                raise PublishError(f"immutable receipt destination is not a regular file: {target}")
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PublishError(f"concurrent immutable receipt is unreadable: {target}: {exc}") from exc
            if existing != dict(value):
                raise PublishError(f"immutable receipt differs on concurrent commit: {target}")
        if committed:
            try:
                written = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PublishError(f"committed receipt is unreadable: {target}: {exc}") from exc
            if written != dict(value):
                raise PublishError(f"committed receipt verification failed: {target}")
            _fsync_directory(target.parent)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
