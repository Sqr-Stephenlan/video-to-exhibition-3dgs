from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.longsplat.pipeline_contract import RunLedger
from scripts.longsplat.publisher import PublishError, publish_ply
from scripts.longsplat.reconstruct_pipeline import _verify_existing_public_delivery


def _ply(value: str = "0 0 0") -> bytes:
    return (
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        f"{value}\n"
    ).encode("ascii")


def _fixture(tmp_path: Path):
    source = tmp_path / "technical.ply"
    source.write_bytes(_ply())
    public = tmp_path / "outputs"
    receipt = publish_ply(source_ply=source, output_dir=public, name="gallery")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    receipt_path = run_dir / "published_ply.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger = RunLedger(run_dir, {"schema_version": "test"}, resumed=True)
    ledger.summary.update(
        {
            "status": "stopped",
            "computed_pass": True,
            "delivery_reachable": True,
            "published_ply": receipt,
            "published_ply_receipt": str(receipt_path),
        }
    )
    ledger._write_summary()
    return source, public, receipt, receipt_path, ledger


def _assert_blocked(ledger: RunLedger, public: Path, receipt_name: str = "gallery") -> None:
    with pytest.raises(PublishError):
        _verify_existing_public_delivery(
            ledger=ledger,
            publish_dir=public,
            delivery_name=receipt_name,
        )
    summary = json.loads((ledger.run_dir / "run.json").read_text(encoding="utf-8"))
    assert summary["status"] == "blocked"
    assert summary["computed_pass"] is False
    assert summary["delivery_reachable"] is False
    assert summary["published_ply"] is None
    assert summary["published_ply_receipt"] is None


def test_resume_revalidates_canonical_public_delivery(tmp_path: Path) -> None:
    _source, public, _receipt, _receipt_path, ledger = _fixture(tmp_path)
    _verify_existing_public_delivery(ledger=ledger, publish_dir=public, delivery_name="gallery")


def test_resume_blocks_deleted_public_delivery_and_clears_stale_success(tmp_path: Path) -> None:
    _source, public, receipt, _receipt_path, ledger = _fixture(tmp_path)
    Path(receipt["published"]["path"]).unlink()
    _assert_blocked(ledger, public)


def test_resume_blocks_tampered_or_replaced_public_delivery(tmp_path: Path) -> None:
    _source, public, receipt, _receipt_path, ledger = _fixture(tmp_path)
    destination = Path(receipt["published"]["path"])
    destination.write_bytes(_ply("9 9 9"))
    _assert_blocked(ledger, public)


def test_resume_blocks_external_hardlink_public_delivery(tmp_path: Path) -> None:
    _source, public, receipt, _receipt_path, ledger = _fixture(tmp_path)
    destination = Path(receipt["published"]["path"])
    external = tmp_path / "external.ply"
    os.link(destination, external)
    assert destination.stat().st_nlink == 2
    _assert_blocked(ledger, public)


def test_resume_blocks_noncanonical_public_directory_binding(tmp_path: Path) -> None:
    _source, public, _receipt, _receipt_path, ledger = _fixture(tmp_path)
    _assert_blocked(ledger, public / "wrong-directory")
