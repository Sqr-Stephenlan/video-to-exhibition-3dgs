"""
Integration tests for the LongSplat input contract.

Pure CPU tests — no Torch, CUDA, LongSplat, MASt3R, VDA, or checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root so scripts.longsplat is importable.
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.longsplat.validate_input import ManifestValidationError, validate_manifest  # noqa: E402
from scripts.longsplat.prepare_input import PreparedFileHashMismatch, prepare_input  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_manifest_dir():
    """Temporary directory containing a manifest and frame images."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        # Create a small fake image file
        img = base / "frames"
        img.mkdir()
        fp = img / "frame_001.jpg"
        fp.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 128)
        sha = _sha256_hex(fp)

        manifest = {
            "schema_version": 1,
            "segment_id": "test_segment_01",
            "base": str(base),
            "frames": [
                {
                    "frame_id": 0,
                    "path": "frames/frame_001.jpg",
                    "width": 1920,
                    "height": 1080,
                    "sha256": sha,
                    "timestamp": 0.0,
                },
                {
                    "frame_id": 1,
                    "path": "frames/frame_001.jpg",
                    "width": 1920,
                    "height": 1080,
                    "sha256": sha,
                    "timestamp": 0.1,
                },
            ],
        }
        mf_path = base / "manifest.json"
        mf_path.write_text(json.dumps(manifest))
        yield base


@pytest.fixture
def valid_manifest_path(tmp_manifest_dir):
    return str(tmp_manifest_dir / "manifest.json")


# ---------------------------------------------------------------------------
# validate_manifest — success
# ---------------------------------------------------------------------------


def test_valid_manifest_passes(valid_manifest_path):
    manifest = validate_manifest(valid_manifest_path)
    assert manifest["schema_version"] == 1
    assert manifest["segment_id"] == "test_segment_01"
    assert len(manifest["frames"]) == 2


# ---------------------------------------------------------------------------
# validate_manifest — top-level rejections
# ---------------------------------------------------------------------------


def test_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        validate_manifest("/nonexistent/manifest.json")


def test_rejects_non_object(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[]")
    with pytest.raises(ManifestValidationError, match="object"):
        validate_manifest(str(p))


def test_rejects_missing_top_keys(tmp_path, valid_manifest_path):
    # Read the valid manifest, remove a top-level key
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    del data["schema_version"]
    p = tmp_path / "no_schema.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="schema_version"):
        validate_manifest(str(p))


def test_rejects_bad_schema_version(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["schema_version"] = 0
    p = tmp_path / "bad_schema.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="schema_version"):
        validate_manifest(str(p))


def test_rejects_empty_segment_id(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["segment_id"] = "   "
    p = tmp_path / "empty_seg.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="segment_id"):
        validate_manifest(str(p))


def test_rejects_empty_frames(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"] = []
    p = tmp_path / "empty_frames.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="frames"):
        validate_manifest(str(p))


# ---------------------------------------------------------------------------
# validate_manifest — frame-level rejections
# ---------------------------------------------------------------------------


def test_rejects_duplicate_frame_id(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][1]["frame_id"] = 0
    p = tmp_path / "dup_id.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="duplicate frame_id"):
        validate_manifest(str(p))


def test_rejects_path_escape(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][0]["path"] = "../../etc/passwd"
    p = tmp_path / "escape.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="escapes"):
        validate_manifest(str(p))


def test_rejects_prefix_escape(tmp_path, valid_manifest_path):
    """Prefix attack: ``../base_evil`` shares prefix with ``/tmp/base``."""
    base = tmp_path / "base"
    base.mkdir()
    # Create a sibling directory that would share a prefix
    evil = tmp_path / "base_evil"
    evil.mkdir()
    (evil / "frame.jpg").write_text("evil")

    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["base"] = str(base)
    data["frames"][0]["path"] = "../base_evil/frame.jpg"
    p = tmp_path / "prefix_escape.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="escapes"):
        validate_manifest(str(p))


def test_rejects_invalid_width(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][0]["width"] = 0
    p = tmp_path / "bad_w.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="width"):
        validate_manifest(str(p))


def test_rejects_invalid_sha256(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][0]["sha256"] = "too_short"
    p = tmp_path / "bad_sha.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="sha256"):
        validate_manifest(str(p))


def test_rejects_unsorted_frame_ids(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    frame_ids = [data["frames"][0]["frame_id"], data["frames"][1]["frame_id"]]
    data["frames"][0]["frame_id"] = frame_ids[1]
    data["frames"][1]["frame_id"] = frame_ids[0]
    p = tmp_path / "unsorted.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="ascending"):
        validate_manifest(str(p))


def test_rejects_unknown_frame_key(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][0]["unknown_field"] = "surprise"
    p = tmp_path / "unknown_key.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="unknown keys"):
        validate_manifest(str(p))


# ---------------------------------------------------------------------------
# validate_manifest — optional fields
# ---------------------------------------------------------------------------


def test_accepts_optional_timestamp(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][0]["timestamp"] = 1.5
    p = tmp_path / "with_ts.json"
    p.write_text(json.dumps(data))
    manifest = validate_manifest(str(p))
    assert manifest["frames"][0]["timestamp"] == 1.5


def test_rejects_non_numeric_timestamp(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][0]["timestamp"] = "not_a_number"
    p = tmp_path / "bad_ts.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="timestamp"):
        validate_manifest(str(p))


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_timestamp(tmp_path, valid_manifest_path, timestamp):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][0]["timestamp"] = timestamp
    p = tmp_path / "bad_finite_ts.json"
    p.write_text(json.dumps(data))
    with pytest.raises(ManifestValidationError, match="timestamp"):
        validate_manifest(str(p))


def test_accepts_producer_trace_fields(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][0].update(
        {
            "producer_frame_id": "segment_0002_frame_000123",
            "producer_frame_index": 123,
            "producer_run_id": "video-source-settings",
            "timestamp": 1.25,
        }
    )
    p = tmp_path / "with_producer_trace.json"
    p.write_text(json.dumps(data))

    manifest = validate_manifest(str(p))

    assert manifest["frames"][0]["producer_frame_id"] == "segment_0002_frame_000123"
    assert manifest["frames"][0]["producer_frame_index"] == 123


# ---------------------------------------------------------------------------
# prepare_input — success and failure
# ---------------------------------------------------------------------------


def test_prepare_input_copies_frames(valid_manifest_path, tmp_path):
    manifest = validate_manifest(valid_manifest_path)
    run_dir = tmp_path / "run_01"
    img_dir = prepare_input(manifest, run_dir)

    assert img_dir.is_dir()
    assert (img_dir / "frame_000000.jpg").is_file()
    assert (img_dir / "frame_000001.jpg").is_file()

    mapping_path = run_dir / "input" / "frame_mapping.json"
    assert mapping_path.is_file()
    with open(mapping_path) as fh:
        mapping = json.load(fh)
    assert len(mapping) == 2
    assert mapping[0]["frame_id"] == 0
    assert mapping[0]["prepared_name"] == "frame_000000.jpg"


def test_prepare_input_hash_mismatch_raises(tmp_path, valid_manifest_path):
    """If manifest SHA-256 is wrong, copy should raise."""
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][0]["sha256"] = "0" * 64
    p = tmp_path / "bad_hash_manifest.json"
    p.write_text(json.dumps(data))

    manifest = validate_manifest(str(p))
    with pytest.raises(PreparedFileHashMismatch):
        prepare_input(manifest, tmp_path / "run_bad")


def test_prepare_input_preserves_producer_trace(tmp_path, valid_manifest_path):
    with open(valid_manifest_path) as fh:
        data = json.load(fh)
    data["frames"][0].update(
        {
            "producer_frame_id": "segment_0002_frame_000123",
            "producer_frame_index": 123,
            "producer_run_id": "video-source-settings",
            "timestamp": 1.25,
        }
    )
    p = tmp_path / "trace_manifest.json"
    p.write_text(json.dumps(data))
    manifest = validate_manifest(p)

    prepare_input(manifest, tmp_path / "run_trace")

    mapping = json.loads(
        (tmp_path / "run_trace" / "input" / "frame_mapping.json").read_text()
    )
    assert mapping[0] == {
        "frame_id": 0,
        "segment_id": "test_segment_01",
        "producer_frame_id": "segment_0002_frame_000123",
        "producer_frame_index": 123,
        "producer_run_id": "video-source-settings",
        "timestamp_sec": 1.25,
        "source_path": "frames/frame_001.jpg",
        "prepared_name": "frame_000000.jpg",
        "sha256": mapping[0]["sha256"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
