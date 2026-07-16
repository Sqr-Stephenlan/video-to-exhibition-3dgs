from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT))

from scripts.depth.backend_vda import (
    find_vda_depths_npz,
    load_vda_depths_array,
    split_vda_depths_to_frame_files,
)
from scripts.depth.config import resolve_repo_path, to_repo_relative, validate_frame_id


def test_resolve_repo_path_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        resolve_repo_path(root, "../outside.txt")


def test_resolve_repo_path_rejects_absolute(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Absolute"):
        resolve_repo_path(tmp_path, str(tmp_path / "abs.txt"))


def test_validate_frame_id_rejects_path_segments() -> None:
    with pytest.raises(ValueError, match="frame_id"):
        validate_frame_id("../x")
    with pytest.raises(ValueError, match="frame_id"):
        validate_frame_id("a/b")
    assert validate_frame_id("demo_0001") == "demo_0001"


def test_to_repo_relative(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "data" / "depth"
    nested.mkdir(parents=True)
    target = nested / "a.npz"
    target.write_bytes(b"x")
    assert to_repo_relative(root, target) == "data/depth/a.npz"


def test_split_vda_depths_npz_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    depth_dir = root / "data" / "depth"
    depth_dir.mkdir(parents=True)
    vda_out = tmp_path / "vda_out"
    vda_out.mkdir()
    depths = np.stack(
        [
            np.full((2, 3), 1.0, dtype=np.float32),
            np.full((2, 3), 2.0, dtype=np.float32),
            np.full((2, 3), 3.0, dtype=np.float32),
        ],
        axis=0,
    )
    source = vda_out / "input_depths.npz"
    np.savez_compressed(source, depths=depths)

    found = find_vda_depths_npz(vda_out)
    loaded = load_vda_depths_array(found)
    assert loaded.shape == (3, 2, 3)

    frames = [
        {"frame_id": "f1", "path": "data/frames/f1.png"},
        {"frame_id": "f2", "path": "data/frames/f2.png"},
        {"frame_id": "f3", "path": "data/frames/f3.png"},
    ]
    records = split_vda_depths_to_frame_files(
        depths=loaded,
        frames=frames,
        depth_dir=depth_dir,
        root=root,
        depth_type="relative",
    )
    assert len(records) == 3
    assert records[1]["depth_path"] == "data/depth/f2.npz"
    with np.load(root / records[1]["depth_path"]) as data:
        assert "depth" in data
        assert float(data["depth"].mean()) == 2.0


def test_split_vda_depths_rejects_frame_count_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    depth_dir = root / "data" / "depth"
    depth_dir.mkdir(parents=True)
    depths = np.zeros((2, 2, 2), dtype=np.float32)
    frames = [{"frame_id": "only", "path": "data/frames/only.png"}]
    with pytest.raises(ValueError, match="frame count"):
        split_vda_depths_to_frame_files(
            depths=depths,
            frames=frames,
            depth_dir=depth_dir,
            root=root,
            depth_type="relative",
        )
