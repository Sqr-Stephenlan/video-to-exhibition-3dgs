"""Tests for _common.py shared utilities."""

from __future__ import annotations

import pytest
from _common import (
    OutputDirectoryError,
    check_imread,
    check_imwrite,
    dict_fingerprint,
    ensure_output_dir,
    file_sha256,
    get_project_root,
    now_utc_iso,
    read_json,
    safe_parse_fps,
    write_json_atomic,
)

# ---------------------------------------------------------------------------
# get_project_root
# ---------------------------------------------------------------------------


def test_get_project_root_returns_valid_path():
    root = get_project_root()
    assert root.exists()
    assert (root / "scripts").is_dir()


def test_get_project_root_is_cached():
    a = get_project_root()
    b = get_project_root()
    assert a is b


# ---------------------------------------------------------------------------
# safe_parse_fps
# ---------------------------------------------------------------------------


def test_safe_parse_fps_common():
    assert safe_parse_fps("30/1") == 30.0
    assert safe_parse_fps("30000/1001") == pytest.approx(29.97002997002997)


def test_safe_parse_fps_integer():
    assert safe_parse_fps("24/1") == 24.0


def test_safe_parse_fps_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        safe_parse_fps("")


def test_safe_parse_fps_zero_denom_raises():
    with pytest.raises(ValueError, match="zero denominator"):
        safe_parse_fps("30/0")


# ---------------------------------------------------------------------------
# ensure_output_dir
# ---------------------------------------------------------------------------


def test_ensure_output_dir_creates(tmp_path):
    d = tmp_path / "new_dir"
    result = ensure_output_dir(d)
    assert result.exists()
    assert result.is_dir()


def test_ensure_output_dir_empty_existing(tmp_path):
    d = tmp_path / "empty_dir"
    d.mkdir()
    result = ensure_output_dir(d)
    assert result == d.resolve()


def test_ensure_output_dir_nonempty_default_raises(tmp_path):
    d = tmp_path / "nonempty"
    d.mkdir()
    (d / "file.txt").write_text("hello")
    with pytest.raises(OutputDirectoryError, match="not empty"):
        ensure_output_dir(d)


def test_ensure_output_dir_overwrite_clears(tmp_path):
    d = tmp_path / "overwrite_test"
    d.mkdir()
    (d / "stale.txt").write_text("old")
    result = ensure_output_dir(d, overwrite=True)
    assert result.exists()
    assert not list(result.iterdir())


def test_ensure_output_dir_resume_keeps(tmp_path):
    d = tmp_path / "resume_test"
    d.mkdir()
    (d / "existing.txt").write_text("keep")
    result = ensure_output_dir(d, resume=True)
    assert (result / "existing.txt").exists()


# ---------------------------------------------------------------------------
# Atomic JSON I/O
# ---------------------------------------------------------------------------


def test_write_and_read_json(tmp_path):
    path = tmp_path / "test.json"
    data = {"key": "value", "nested": [1, 2, 3]}
    write_json_atomic(data, path)
    assert path.exists()
    assert read_json(path) == data


def test_write_json_utf8(tmp_path):
    path = tmp_path / "utf8.json"
    data = {"name": "café", "emoji": "\U0001f4a9"}
    write_json_atomic(data, path)
    result = read_json(path)
    assert result["name"] == "café"


def test_write_json_preserves_ints(tmp_path):
    path = tmp_path / "ints.json"
    data = {"count": 42}
    write_json_atomic(data, path)
    result = read_json(path)
    assert result["count"] == 42


# ---------------------------------------------------------------------------
# File checksum
# ---------------------------------------------------------------------------


def test_file_sha256_deterministic(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world")
    a = file_sha256(f)
    b = file_sha256(f)
    assert a == b
    assert len(a) == 64


def test_file_sha256_different_content(tmp_path):
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"alpha")
    f2 = tmp_path / "b.bin"
    f2.write_bytes(b"beta")
    assert file_sha256(f1) != file_sha256(f2)


# ---------------------------------------------------------------------------
# dict_fingerprint
# ---------------------------------------------------------------------------


def test_dict_fingerprint_deterministic():
    a = dict_fingerprint({"x": 1, "y": 2})
    b = dict_fingerprint({"y": 2, "x": 1})
    assert a == b


def test_dict_fingerprint_different_for_different_dicts():
    a = dict_fingerprint({"x": 1})
    b = dict_fingerprint({"x": 2})
    assert a != b


# ---------------------------------------------------------------------------
# now_utc_iso
# ---------------------------------------------------------------------------


def test_now_utc_iso_format():
    ts = now_utc_iso()
    assert ts.endswith("+00:00") or ts.endswith("Z")
    assert "T" in ts


# ---------------------------------------------------------------------------
# Image I/O skips (cv2-based, tested via integration fixtures)
# ---------------------------------------------------------------------------


def test_check_imread_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cannot read"):
        check_imread(tmp_path / "nonexistent.jpg")


def test_check_imwrite_to_unwritable_dir(tmp_path):
    import numpy as np

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(OSError, match="Failed to write"):
        check_imwrite(tmp_path / "bad" / "out.jpg", img)
