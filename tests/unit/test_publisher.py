from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path

import pytest

from scripts.longsplat import publisher
from scripts.longsplat.publisher import (
    PublishError,
    publish_ply,
    verify_published_ply,
    verify_public_delivery,
    write_json_once_atomic,
)


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
    assert result["published"]["nlink"] == 1
    assert verify_published_ply(result)["published"]["sha256"] == result["published"]["sha256"]
    tampered = dict(result)
    tampered["published"] = {**result["published"], "vertices": 999}
    with pytest.raises(PublishError, match="identity drifted"):
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


def test_public_delivery_rejects_external_hardlink_and_reuse_requires_nlink_one(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(_ply())
    output = tmp_path / "outputs"
    receipt = publish_ply(source_ply=source, output_dir=output, name="link")
    destination = Path(receipt["published"]["path"])
    external = tmp_path / "external-hardlink.ply"
    os.link(destination, external)
    assert destination.stat().st_nlink == 2
    with pytest.raises(PublishError, match="st_nlink=1"):
        publish_ply(source_ply=source, output_dir=output, name="link")
    with pytest.raises(PublishError, match="st_nlink=1"):
        verify_public_delivery(receipt, output_dir=output, name="link")


def test_stale_part_does_not_reserve_or_create_an_empty_final(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(_ply())
    output = tmp_path / "outputs"
    output.mkdir()
    stale = output / ".stale-copy.part"
    stale.write_bytes(b"stale")
    receipt = publish_ply(source_ply=source, output_dir=output, name="stale")
    destination = Path(receipt["published"]["path"])
    assert destination.is_file()
    assert destination.stat().st_size > 0
    assert stale.is_file()
    assert not list(output.glob(f".{destination.name}.*.part"))


def test_publish_never_overwrites_different_existing_content(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(_ply())
    output = tmp_path / "outputs"
    first = publish_ply(source_ply=source, output_dir=output, name="collision")
    destination = Path(first["published"]["path"])
    original = destination.read_bytes()
    destination.write_bytes(_ply(2, "8 8 8\n9 9 9"))
    collided = destination.read_bytes()
    assert collided != original
    with pytest.raises(PublishError, match="collision"):
        publish_ply(source_ply=source, output_dir=output, name="collision")
    assert destination.read_bytes() == collided
    assert destination.stat().st_size != 0


def test_atomic_failure_cleans_part_and_leaves_no_final(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(_ply())

    def fail_atomic(_source: Path, _destination: Path) -> None:
        raise PublishError("atomic no-replace test failure")

    monkeypatch.setattr(publisher, "atomic_noreplace", fail_atomic)
    output = tmp_path / "outputs"
    with pytest.raises(PublishError, match="atomic no-replace test failure"):
        publish_ply(source_ply=source, output_dir=output, name="atomic-failure")
    assert not list(output.glob("*.ply"))
    assert not list(output.glob("*.part"))


def test_directory_fsync_failure_is_not_reported_as_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(_ply())
    monkeypatch.setattr(publisher, "_fsync_directory", lambda _path: (_ for _ in ()).throw(PublishError("fsync blocked")))
    with pytest.raises(PublishError, match="fsync blocked"):
        publish_ply(source_ply=source, output_dir=tmp_path / "outputs", name="fsync")
    assert list((tmp_path / "outputs").glob("*.ply"))


def test_concurrent_publishers_use_one_no_replace_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(_ply())
    output = tmp_path / "outputs"

    def publish() -> dict[str, object]:
        return publish_ply(source_ply=source, output_dir=output, name="concurrent")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _item: publish(), range(2)))
    assert {bool(item["reused"]) for item in results} == {False, True}
    destinations = {str(item["published"]["path"]) for item in results}
    assert len(destinations) == 1
    assert next(iter(destinations))
    assert next(Path(path) for path in destinations).stat().st_nlink == 1
    assert not list(output.glob("*.part"))


def test_receipt_atomic_commit_is_immutable_and_cleans_failed_part(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path = tmp_path / "run" / "published_ply.json"
    value = {"schema_version": "test", "value": 1}
    write_json_once_atomic(receipt_path, value)
    write_json_once_atomic(receipt_path, value)
    with pytest.raises(PublishError, match="differs"):
        write_json_once_atomic(receipt_path, {"schema_version": "test", "value": 2})

    def fail_atomic(_source: Path, _destination: Path) -> None:
        raise PublishError("receipt atomic failure")

    failed_path = tmp_path / "run" / "failed.json"
    monkeypatch.setattr(publisher, "atomic_noreplace", fail_atomic)
    with pytest.raises(PublishError, match="receipt atomic failure"):
        write_json_once_atomic(failed_path, value)
    assert not failed_path.exists()
    assert not list(failed_path.parent.glob("*.part"))
