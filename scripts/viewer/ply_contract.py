from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class GaussianPlyError(ValueError):
    """Raised when a file is not a standard 3DGS Gaussian PLY."""


@dataclass(frozen=True)
class GaussianPlyIdentity:
    filename: str
    sha256: str
    size_bytes: int
    vertices: int
    properties: tuple[str, ...]
    schema_version: str = "standard-3dgs-gaussian-v1"


_HEADER_LIMIT = 1024 * 1024


def _read_header(path: Path) -> tuple[str, int, tuple[str, ...]]:
    try:
        with path.open("rb") as handle:
            data = handle.read(_HEADER_LIMIT)
    except OSError as exc:
        raise GaussianPlyError(f"cannot read PLY: {path}") from exc
    marker = b"end_header"
    end = data.find(marker)
    if end < 0:
        raise GaussianPlyError("PLY header is missing end_header")
    header = data[: end + len(marker)].decode("ascii", errors="strict")
    lines = [line.strip() for line in header.splitlines() if line.strip()]
    if not lines or lines[0] != "ply":
        raise GaussianPlyError("file is not a PLY")
    if not any(line.startswith("format ") for line in lines[1:]):
        raise GaussianPlyError("PLY format is missing")

    vertex_count: int | None = None
    in_vertex = False
    properties: list[str] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "element":
            in_vertex = fields[1] == "vertex"
            if in_vertex:
                try:
                    vertex_count = int(fields[2])
                except ValueError as exc:
                    raise GaussianPlyError("PLY vertex count is invalid") from exc
                if vertex_count < 1:
                    raise GaussianPlyError("PLY must contain at least one vertex")
        elif in_vertex and len(fields) >= 3 and fields[0] == "property":
            properties.append(fields[-1])

    if vertex_count is None:
        raise GaussianPlyError("PLY vertex element is missing")
    return header, vertex_count, tuple(properties)


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise GaussianPlyError(f"cannot hash PLY: {path}") from exc
    return digest.hexdigest(), size


def validate_gaussian_ply(path: str | Path) -> GaussianPlyIdentity:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise GaussianPlyError("PLY must be a regular file")
    _, vertices, properties = _read_header(candidate)
    property_set = set(properties)
    required_exact = {"x", "y", "z", "opacity"}
    missing = sorted(required_exact - property_set)
    groups = {
        "f_dc": any(name == "f_dc" or name.startswith("f_dc_") for name in properties),
        "scale": any(name == "scale" or name.startswith("scale_") for name in properties),
        "rot": any(name == "rot" or name.startswith("rot_") for name in properties),
    }
    missing.extend(name for name, present in groups.items() if not present)
    if missing:
        raise GaussianPlyError("Gaussian PLY is missing attributes: " + ", ".join(missing))
    sha256, size_bytes = _sha256_and_size(candidate)
    return GaussianPlyIdentity(
        filename=candidate.name,
        sha256=sha256,
        size_bytes=size_bytes,
        vertices=vertices,
        properties=tuple(sorted(property_set)),
    )
