"""RED tests for isolated LongSplat checkpoint reconversion — Task 2 Step 1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(path: Path, content: bytes | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content or b"mock")


def _make_source_model(base: Path) -> Path:
    """Create a minimal mock source model directory.

    Returns *base*.  The caller is responsible for cleanup (tmp_path).
    """
    base.mkdir(parents=True, exist_ok=True)
    (base / "cfg_args").write_text('{"Images": "images"}', encoding="utf-8")
    _touch(base / "cameras_all_train.json", b"[]")
    _touch(base / "cameras_all_test.json", b"[]")
    chk = base / "point_cloud" / "iteration_50000"
    chk.mkdir(parents=True, exist_ok=True)
    _touch(chk / "point_cloud.ply", b"ply\nheader\nend_header\n")
    for mlp in ("color_mlp.pt", "cov_mlp.pt", "opacity_mlp.pt"):
        _touch(chk / mlp, b"pt-mock")
    return base


# ---------------------------------------------------------------------------
# Test 1: destination already exists → reject
# ---------------------------------------------------------------------------


def test_reject_existing_destination(tmp_path):
    """create_conversion_snapshot fails when destination_model already exists."""
    from scripts.longsplat.reconvert_existing import create_conversion_snapshot

    src = _make_source_model(tmp_path / "src")
    dst = tmp_path / "dst"
    dst.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        create_conversion_snapshot(src, dst, checkpoint_iteration=50000)


# ---------------------------------------------------------------------------
# Test 2: missing MLP file → reject
# ---------------------------------------------------------------------------


def test_reject_missing_mlp(tmp_path):
    """Snapshot fails when an MLP file is missing."""
    from scripts.longsplat.reconvert_existing import create_conversion_snapshot

    src = _make_source_model(tmp_path / "src")
    # Remove one MLP
    (src / "point_cloud" / "iteration_50000" / "color_mlp.pt").unlink()

    with pytest.raises(FileNotFoundError):
        create_conversion_snapshot(src, tmp_path / "dst", checkpoint_iteration=50000)


# ---------------------------------------------------------------------------
# Test 3: source SHA unchanged before/after snapshot
# ---------------------------------------------------------------------------


def test_source_sha_unchanged(tmp_path):
    """Snapshot never modifies source files (SHA preserved)."""
    from scripts.longsplat.reconvert_existing import create_conversion_snapshot

    src = _make_source_model(tmp_path / "src")
    src_files = sorted(
        p.relative_to(src) for p in src.rglob("*") if p.is_file()
    )
    # Hash before
    hashes_before = {}
    for rel in src_files:
        hashes_before[rel] = _sha256_hex(src / rel)

    result = create_conversion_snapshot(src, tmp_path / "dst", checkpoint_iteration=50000)

    # Hash after
    for rel in src_files:
        assert hashes_before[rel] == _sha256_hex(src / rel), f"{rel} modified"

    assert "source_sha256" in result


# ---------------------------------------------------------------------------
# Test 4: only whitelist files copied
# ---------------------------------------------------------------------------


def test_only_whitelist_files_copied(tmp_path):
    """Snapshot copies only the approved file set, nothing else."""
    from scripts.longsplat.reconvert_existing import create_conversion_snapshot

    src = _make_source_model(tmp_path / "src")
    # Add a file that should NOT be copied
    _touch(src / "converted_3dgs" / "point_cloud.ply", b"old-conversion")
    _touch(src / "train.log", b"log")
    _touch(src / "some_other_file.txt", b"junk")

    _ = create_conversion_snapshot(src, tmp_path / "dst", checkpoint_iteration=50000)

    dst_files = sorted(
        p.relative_to(tmp_path / "dst")
        for p in (tmp_path / "dst").rglob("*")
        if p.is_file()
    )
    expected = sorted(
        [
            Path("cfg_args"),
            Path("cameras_all_train.json"),
            Path("cameras_all_test.json"),
            Path("cameras_all.json"),  # copy of cameras_all_train
            Path("point_cloud/iteration_50000/point_cloud.ply"),
            Path("point_cloud/iteration_50000/color_mlp.pt"),
            Path("point_cloud/iteration_50000/cov_mlp.pt"),
            Path("point_cloud/iteration_50000/opacity_mlp.pt"),
        ]
    )
    assert dst_files == expected

    # cameras_all.json is a copy of cameras_all_train.json
    assert _sha256_hex(
        tmp_path / "dst" / "cameras_all.json"
    ) == _sha256_hex(tmp_path / "dst" / "cameras_all_train.json")


# ---------------------------------------------------------------------------
# Test 5: dry-run mode does not launch conversion subprocess
# ---------------------------------------------------------------------------


def test_dry_run_no_subprocess(tmp_path):
    """--dry-run prints the command but does not execute it."""
    from scripts.longsplat.reconvert_existing import main as reconvert_main

    src = _make_source_model(tmp_path / "src")
    dst = tmp_path / "dst"
    src_path = tmp_path / "frames"
    src_path.mkdir()
    (src_path / "images").mkdir()
    (src_path / "images" / "f_000001.png").write_text("mock")

    record_path = tmp_path / "record.json"

    try:
        reconvert_main(
            [
                "--source-model",
                str(src),
                "--destination-model",
                str(dst),
                "--source-path",
                str(src_path),
                "--repo-root",
                str(tmp_path / "fake_repo"),  # won't be validated in dry-run
                "--backend-python",
                "python",
                "--checkpoint-iteration",
                "50000",
                "--conversion-iterations",
                "100",
                "--prune-ratio",
                "0.6",
                "--backend-mode",
                "research_local",
                "--output-record",
                str(record_path),
                "--dry-run",
            ]
        )
    except SystemExit as e:
        assert e.code == 0

    # Output record should exist with dry_run=True
    record = json.loads(record_path.read_text())
    assert record.get("dry_run") is True
    assert "command" in record or "cmd" in record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
