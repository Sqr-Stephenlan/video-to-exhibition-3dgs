from __future__ import annotations

import pytest

from scripts.viewer.ply_contract import GaussianPlyError, validate_gaussian_ply


def _ply(properties: list[str]) -> str:
    lines = [
        "ply",
        "format ascii 1.0",
        "element vertex 1",
    ]
    lines.extend(f"property float {name}" for name in properties)
    lines.extend(["end_header", " ".join("0" for _ in properties)])
    return "\n".join(lines) + "\n"


def test_validate_gaussian_ply_accepts_required_standard_attributes(tmp_path):
    path = tmp_path / "model.ply"
    path.write_text(
        _ply(["x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0", "f_rest_0"]),
        encoding="ascii",
    )

    identity = validate_gaussian_ply(path)

    assert identity.vertices == 1
    assert identity.sha256
    assert identity.schema_version == "standard-3dgs-gaussian-v1"


def test_validate_gaussian_ply_rejects_missing_gaussian_attributes(tmp_path):
    path = tmp_path / "ordinary.ply"
    path.write_text(_ply(["x", "y", "z"]), encoding="ascii")

    with pytest.raises(GaussianPlyError, match="opacity"):
        validate_gaussian_ply(path)
