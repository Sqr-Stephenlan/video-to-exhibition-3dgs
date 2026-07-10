"""Tests for _provenance.py: provenance tracking and resume validation."""

from __future__ import annotations

from _provenance import (
    build_provenance,
    check_resume,
    dir_fingerprint,
    file_sha256,
    params_fingerprint,
    publish_outputs,
    read_provenance,
    write_provenance,
)

# ---------------------------------------------------------------------------
# file_sha256
# ---------------------------------------------------------------------------


def test_provenance_file_sha256_deterministic(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("test content", encoding="utf-8")
    assert file_sha256(f) == file_sha256(f)


def test_provenance_file_sha256_changes(tmp_path):
    f = tmp_path / "changing.txt"
    f.write_text("v1", encoding="utf-8")
    h1 = file_sha256(f)
    f.write_text("v2", encoding="utf-8")
    h2 = file_sha256(f)
    assert h1 != h2


# ---------------------------------------------------------------------------
# dir_fingerprint
# ---------------------------------------------------------------------------


def test_dir_fingerprint(tmp_path):
    (tmp_path / "a.txt").write_text("AAA")
    (tmp_path / "b.txt").write_text("BBB")
    fp = dir_fingerprint(tmp_path, "*.txt")
    assert set(fp.keys()) == {"a.txt", "b.txt"}
    assert len(fp["a.txt"]) == 64


def test_dir_fingerprint_empty_dir(tmp_path):
    assert dir_fingerprint(tmp_path) == {}


# ---------------------------------------------------------------------------
# params_fingerprint
# ---------------------------------------------------------------------------


def test_params_fingerprint_stable_order():
    h1 = params_fingerprint({"a": 1, "b": 2})
    h2 = params_fingerprint({"b": 2, "a": 1})
    assert h1 == h2


def test_params_fingerprint_different_values():
    assert params_fingerprint({"x": 1}) != params_fingerprint({"x": 2})


# ---------------------------------------------------------------------------
# build_provenance
# ---------------------------------------------------------------------------


def test_build_provenance_structure():
    prov = build_provenance(
        stage="test_stage",
        inputs={"frame_0001.jpg": "abc123"},
        params={"fps": 5.0},
        tool_versions={"ffmpeg": "6.0"},
        outputs=["output/frame_0001.jpg"],
    )
    assert prov["stage"] == "test_stage"
    assert "timestamp_utc" in prov
    assert prov["inputs"] == {"frame_0001.jpg": "abc123"}
    assert prov["params"] == {"fps": 5.0}
    assert "params_fingerprint" in prov
    assert prov["tool_versions"] == {"ffmpeg": "6.0"}


# ---------------------------------------------------------------------------
# write_provenance / read_provenance roundtrip
# ---------------------------------------------------------------------------


def test_write_and_read_provenance(tmp_path):
    prov = build_provenance(
        stage="test",
        inputs={},
        params={"key": "val"},
        tool_versions={},
        outputs=["a.ply"],
    )
    write_provenance(prov, tmp_path)
    read = read_provenance(tmp_path)
    assert read["stage"] == "test"
    assert read["params_fingerprint"] == prov["params_fingerprint"]


def test_read_provenance_missing_file(tmp_path):
    assert read_provenance(tmp_path / "nonexistent") == {}


# ---------------------------------------------------------------------------
# check_resume
# ---------------------------------------------------------------------------


def test_check_resume_happy_path(tmp_path):
    prov = build_provenance(
        stage="extract",
        inputs={"video.mp4": "hash123"},
        params={"fps": 5.0},
        tool_versions={},
        outputs=["frames/frame_000001.jpg"],
    )
    write_provenance(prov, tmp_path)
    (tmp_path / "_COMPLETE").write_text("2026-07-10T00:00:00+00:00")

    ok = check_resume(
        tmp_path,
        expected_input_fingerprints={"video.mp4": "hash123"},
        expected_params={"fps": 5.0},
    )
    assert ok is True


def test_check_resume_no_complete_marker(tmp_path):
    prov = build_provenance(
        stage="extract",
        inputs={},
        params={},
        tool_versions={},
        outputs=[],
    )
    write_provenance(prov, tmp_path)
    # No _COMPLETE marker
    assert check_resume(tmp_path, {}, {}) is False


def test_check_resume_no_provenance_file(tmp_path):
    (tmp_path / "_COMPLETE").write_text("done")
    assert check_resume(tmp_path, {}, {}) is False


def test_check_resume_inputs_changed(tmp_path):
    prov = build_provenance(
        stage="extract",
        inputs={"video.mp4": "old_hash"},
        params={"fps": 5.0},
        tool_versions={},
        outputs=[],
    )
    write_provenance(prov, tmp_path)
    (tmp_path / "_COMPLETE").write_text("done")

    ok = check_resume(
        tmp_path,
        expected_input_fingerprints={"video.mp4": "new_hash"},
        expected_params={"fps": 5.0},
    )
    assert ok is False


def test_check_resume_params_changed(tmp_path):
    prov = build_provenance(
        stage="extract",
        inputs={"video.mp4": "hash"},
        params={"fps": 5.0},
        tool_versions={},
        outputs=[],
    )
    write_provenance(prov, tmp_path)
    (tmp_path / "_COMPLETE").write_text("done")

    ok = check_resume(tmp_path, {"video.mp4": "hash"}, {"fps": 10.0})
    assert ok is False


# ---------------------------------------------------------------------------
# publish_outputs
# ---------------------------------------------------------------------------


def test_publish_outputs_moves_files(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "output.ply").write_text("ply data")
    (staging / "report.json").write_text('{"ok": true}')

    final = tmp_path / "final"
    publish_outputs(staging, final)

    assert (final / "output.ply").exists()
    assert (final / "report.json").exists()
    assert (final / "_COMPLETE").exists()
    assert not staging.exists() or not list(staging.iterdir())


def test_publish_outputs_overwrites_existing(tmp_path):
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    staging.mkdir()
    final.mkdir()
    (staging / "data.txt").write_text("new")
    (final / "data.txt").write_text("old")

    publish_outputs(staging, final)
    assert (final / "data.txt").read_text() == "new"
