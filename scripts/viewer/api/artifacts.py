from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .contracts import ArtifactDescriptor
from ..ply_contract import GaussianPlyError, validate_gaussian_ply


class ArtifactSecurityError(RuntimeError):
    """Raised when an artifact is missing, outside its allowlist, or changed."""


_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _contained_file(path: Path, roots: Iterable[Path]) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ArtifactSecurityError(f"artifact is not a regular file: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ArtifactSecurityError("artifact cannot be resolved") from exc

    for root in roots:
        try:
            resolved_root = root.resolve(strict=True)
            relative = resolved.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        current = resolved_root
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ArtifactSecurityError("artifact path contains a symlink")
        return resolved
    raise ArtifactSecurityError("artifact is outside the configured allowlist")


@dataclass(frozen=True)
class _ArtifactRecord:
    descriptor: ArtifactDescriptor
    path: Path
    validate_gaussian: bool = False


class ArtifactCatalog:
    """In-memory allowlist backed by revalidated files."""

    def __init__(self, *, job_id: str, allowed_roots: Iterable[Path]):
        self.job_id = job_id
        self.allowed_roots = tuple(Path(root) for root in allowed_roots)
        self._records: dict[str, _ArtifactRecord] = {}

    def register(
        self,
        *,
        artifact_id: str,
        kind: str,
        format: str,
        path: str | Path,
        vertices: int | None = None,
        validate_gaussian: bool = False,
    ) -> ArtifactDescriptor:
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ArtifactSecurityError("artifact ID is unsafe")
        verified = _contained_file(Path(path), self.allowed_roots)
        if validate_gaussian:
            try:
                identity = validate_gaussian_ply(verified)
            except GaussianPlyError as exc:
                raise ArtifactSecurityError(str(exc)) from exc
            if vertices is None:
                vertices = identity.vertices
        sha256, size_bytes = _sha256_and_size(verified)
        descriptor = ArtifactDescriptor(
            id=artifact_id,
            kind=kind,
            format=format,
            download_url=f"/api/v1/jobs/{self.job_id}/artifacts/{artifact_id}",
            sha256=sha256,
            size_bytes=size_bytes,
            vertices=vertices,
        )
        self._records[artifact_id] = _ArtifactRecord(
            descriptor=descriptor,
            path=verified,
            validate_gaussian=validate_gaussian,
        )
        return descriptor

    def descriptors(self) -> list[ArtifactDescriptor]:
        return [record.descriptor for record in self._records.values()]

    def resolve(self, artifact_id: str) -> Path:
        record = self._records.get(artifact_id)
        if record is None:
            raise KeyError(artifact_id)
        verified = _contained_file(record.path, self.allowed_roots)
        if record.validate_gaussian:
            try:
                identity = validate_gaussian_ply(verified)
            except GaussianPlyError as exc:
                raise ArtifactSecurityError(str(exc)) from exc
            if record.descriptor.vertices != identity.vertices:
                raise ArtifactSecurityError("artifact Gaussian vertex count changed")
        sha256, size_bytes = _sha256_and_size(verified)
        if sha256 != record.descriptor.sha256 or size_bytes != record.descriptor.size_bytes:
            raise ArtifactSecurityError("artifact identity changed after registration")
        return verified
