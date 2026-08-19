from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT))

from scripts.depth.assemble_depth_manifests import assemble_depth_manifests
from scripts.depth.backend_vda import sha256_file, split_vda_depths_to_frame_files
from scripts.depth.manifest import build_depth_manifest
from scripts.depth.split_selected_manifest import build_chunk_manifest, _parse_range


def _write_frames_manifest(path: Path, n: int = 5) -> dict:
    frames = [
        {
            "frame_id": f"f{i:03d}",
            "path": f"data/frames/f{i:03d}.png",
            "selected": True,
            "timestamp_sec": float(i),
        }
        for i in range(n)
    ]
    data = {
        "schema_version": "1.0",
        "video_id": "demo",
        "frames": frames,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def test_overlap_prefix_is_count_not_start_index() -> None:
    # A=[0,49), B=[39,91) → overlap frame count is 10; start index is 39.
    source = {
        "schema_version": "1.0",
        "video_id": "demo",
        "frames": [
            {"frame_id": f"f{i}", "path": f"p{i}.png", "selected": True}
            for i in range(91)
        ],
    }
    selected = source["frames"]
    chunk_b = build_chunk_manifest(
        source_manifest=source,
        selected=selected,
        chunk_id="b",
        start=39,
        end=91,
        previous_end=49,
        previous_chunk_id="a",
        source_manifest_path="data/manifests/frames_manifest.json",
    )
    assert chunk_b["chunk"]["overlap_prefix_count"] == 10
    assert chunk_b["chunk"]["source_index_start"] == 39
    assert chunk_b["chunk"]["overlap_prefix_count"] != chunk_b["chunk"]["source_index_start"]


def test_parse_range() -> None:
    assert _parse_range("0:49") == (0, 49)
    assert _parse_range("39,91") == (39, 91)
    with pytest.raises(ValueError):
        _parse_range("10")


def test_assemble_requires_sha_and_verifies_order(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    depth_dir = root / "data" / "depth"
    depth_dir.mkdir(parents=True)
    source = _write_frames_manifest(root / "data" / "manifests" / "frames_manifest.json", n=5)
    selected = source["frames"]

    def _chunk_records(slice_frames: list[dict], offset: int, subdir: str) -> list[dict]:
        depths = np.stack(
            [
                np.full((2, 2), float(offset + i), dtype=np.float32)
                for i in range(len(slice_frames))
            ],
            axis=0,
        )
        out_dir = depth_dir / subdir
        return split_vda_depths_to_frame_files(
            depths=depths,
            frames=slice_frames,
            depth_dir=out_dir,
            root=root,
            depth_type="relative",
        )

    frames_a = _chunk_records(selected[0:4], 0, "a")
    frames_b = _chunk_records(selected[2:5], 10, "b")
    frames_manifest_a = {
        **source,
        "chunk": {
            "chunk_id": "a",
            "source_index_start": 0,
            "source_index_end": 4,
            "frame_count": 4,
            "overlap_prefix_count": 0,
            "overlap_with_chunk_id": None,
        },
    }
    frames_manifest_b = {
        **source,
        "chunk": {
            "chunk_id": "b",
            "source_index_start": 2,
            "source_index_end": 5,
            "frame_count": 3,
            "overlap_prefix_count": 2,
            "overlap_with_chunk_id": "a",
        },
    }
    chunk_a = build_depth_manifest(
        frames_manifest=frames_manifest_a,
        frame_records=frames_a,
        backend={"name": "video-depth-anything", "encoder": "vitb", "depth_type": "relative"},
        depth_type="relative",
        frames_manifest_path="data/manifests/frames_manifest.json",
    )
    chunk_b = build_depth_manifest(
        frames_manifest=frames_manifest_b,
        frame_records=frames_b,
        backend={"name": "video-depth-anything", "encoder": "vitb", "depth_type": "relative"},
        depth_type="relative",
        frames_manifest_path="data/manifests/frames_manifest.json",
    )
    assert chunk_b["chunk"]["overlap_prefix_count"] == 2

    assembled, report = assemble_depth_manifests(
        root=root,
        source_frames_manifest=source,
        source_frames_path="data/manifests/frames_manifest.json",
        chunk_a=chunk_a,
        chunk_b=chunk_b,
        drop_b_prefix=2,
        verify_files=True,
    )
    assert len(assembled["frames"]) == 5
    assert assembled["producer_gates"]["file_integrity"]["status"] == "passed"
    assert assembled["producer_gates"]["vda_quality"]["status"] == "not_evaluated"
    assert all(len(frame["sha256"]) == 64 for frame in assembled["frames"])
    assert report["identity_order_hash_ok"] is True

    # Missing chunk metadata must fail closed when dropping overlap.
    bare_b = dict(chunk_b)
    bare_b.pop("chunk", None)
    with pytest.raises(ValueError, match="overlap_prefix_count is required"):
        assemble_depth_manifests(
            root=root,
            source_frames_manifest=source,
            source_frames_path="data/manifests/frames_manifest.json",
            chunk_a=chunk_a,
            chunk_b=bare_b,
            drop_b_prefix=2,
            verify_files=False,
        )

    # Wrong overlap metadata (start index mistaken for count) must fail closed.
    bad = dict(chunk_b)
    bad["chunk"] = dict(chunk_b["chunk"])
    bad["chunk"]["overlap_prefix_count"] = 39
    with pytest.raises(ValueError, match="overlap_prefix_count"):
        assemble_depth_manifests(
            root=root,
            source_frames_manifest=source,
            source_frames_path="data/manifests/frames_manifest.json",
            chunk_a=chunk_a,
            chunk_b=bad,
            drop_b_prefix=2,
            verify_files=False,
        )


def test_build_depth_manifest_copies_chunk() -> None:
    frames_manifest = {
        "schema_version": "1.0",
        "video_id": "demo",
        "chunk": {
            "chunk_id": "b",
            "source_index_start": 39,
            "source_index_end": 40,
            "frame_count": 1,
            "overlap_prefix_count": 0,
            "overlap_with_chunk_id": None,
        },
        "frames": [{"frame_id": "f", "path": "p.png", "selected": True}],
    }
    manifest = build_depth_manifest(
        frames_manifest=frames_manifest,
        frame_records=[
            {
                "frame_id": "f",
                "rgb_path": "p.png",
                "depth_path": "d.npz",
                "sha256": "b" * 64,
                "depth_type": "relative",
                "depth_index": 0,
            }
        ],
        backend={"name": "x", "depth_type": "relative"},
        depth_type="relative",
    )
    assert manifest["chunk"]["chunk_id"] == "b"
    assert manifest["chunk"]["source_index_start"] == 39
    assert manifest["chunk"]["overlap_prefix_count"] == 0
    assert "strict_positional" not in manifest["producer_gates"]["file_integrity"]["checks"]

    # overlap_prefix_count=10 must stay distinct from source_index_start=39
    frames_manifest["chunk"] = {
        "chunk_id": "b",
        "source_index_start": 39,
        "source_index_end": 49,
        "frame_count": 1,
        "overlap_prefix_count": 10,
        "overlap_with_chunk_id": "a",
    }
    with pytest.raises(ValueError, match="frame_count"):
        build_depth_manifest(
            frames_manifest=frames_manifest,
            frame_records=[
                {
                    "frame_id": "f",
                    "rgb_path": "p.png",
                    "depth_path": "d.npz",
                    "sha256": "b" * 64,
                    "depth_type": "relative",
                    "depth_index": 0,
                }
            ],
            backend={"name": "x", "depth_type": "relative"},
            depth_type="relative",
        )


def test_build_depth_manifest_requires_sha256() -> None:
    with pytest.raises(ValueError, match="sha256"):
        build_depth_manifest(
            frames_manifest={"schema_version": "1.0", "video_id": "demo"},
            frame_records=[
                {
                    "frame_id": "f1",
                    "rgb_path": "a.png",
                    "depth_path": "a.npz",
                    "depth_type": "relative",
                    "depth_index": 0,
                }
            ],
            backend={"name": "x"},
            depth_type="relative",
        )


def test_sha256_matches_file_bytes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    depth_dir = root / "data" / "depth"
    depth_dir.mkdir(parents=True)
    frames = [{"frame_id": "f1", "path": "data/frames/f1.png"}]
    depths = np.zeros((1, 3, 3), dtype=np.float32)
    records = split_vda_depths_to_frame_files(
        depths=depths,
        frames=frames,
        depth_dir=depth_dir,
        root=root,
        depth_type="relative",
    )
    assert records[0]["sha256"] == sha256_file(root / records[0]["depth_path"])
