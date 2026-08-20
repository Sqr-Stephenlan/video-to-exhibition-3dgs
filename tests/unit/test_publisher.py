from __future__ import annotations

from pathlib import Path

import pytest

from scripts.longsplat import publisher
from scripts.longsplat.publisher import PublishError, publish_ply, verify_published_ply


def _ply(vertex_count: int = 1, value: str = "0 0 0") -> bytes:
    return (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {vertex_count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        f"{value}\n"
    ).encode("ascii")


def test_publish_ply_is_independent_and_records_atomic_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(_ply(2, "0 0 0\n1 1 1"))
    result = publish_ply(source_ply=source, output_dir=tmp_path / "outputs", name="Gallery / east")

    destination = Path(result["published"]["path"])
    assert result["reused"] is False
    assert destination.is_file()
    assert not destination.is_symlink()
    assert destination.stat().st_ino != source.stat().st_ino
    assert destination.name.startswith("Gallery_east__")
    assert destination.name.endswith(".ply")
    assert result["published"]["sha256"] == result["source"]["sha256"]
    assert result["published"]["size_bytes"] == result["source"]["size_bytes"]
    assert result["published"]["vertices"] == 2
    assert verify_published_ply(result)["published"]["sha256"] == result["published"]["sha256"]
    tampered = dict(result)
    tampered["published"] = {**result["published"], "vertices": 999}
    with pytest.raises(PublishError, match="vertex identity"):
        verify_published_ply(tampered)


def test_publish_same_content_reuses_and_collision_hard_stops(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(_ply())
    output = tmp_path / "outputs"
    first = publish_ply(source_ply=source, output_dir=output, name="same")
    second = publish_ply(source_ply=source, output_dir=output, name="same")
    assert second["reused"] is True
    assert second["published"]["path"] == first["published"]["path"]

    destination = Path(first["published"]["path"])
    destination.write_bytes(_ply(1, "9 9 9"))
    with pytest.raises(PublishError, match="collision"):
        publish_ply(source_ply=source, output_dir=output, name="same")


def test_publish_rejects_source_sha_drift_and_removes_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(_ply())
    original_copy = publisher._copy_file

    def copy_then_drift(source_path: Path, destination: Path) -> None:
        original_copy(source_path, destination)
        source_path.write_bytes(_ply(1, "8 8 8"))

    monkeypatch.setattr(publisher, "_copy_file", copy_then_drift)
    with pytest.raises(PublishError, match="drifted"):
        publish_ply(source_ply=source, output_dir=tmp_path / "outputs", name="drift")
    assert not list((tmp_path / "outputs").glob("*.part"))
