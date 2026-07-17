from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT))

from scripts.depth.backend_vda import (
    find_vda_depths_npz,
    load_vda_depths_array,
    split_vda_depths_to_frame_files,
    write_run_record,
)
from scripts.depth.config import to_repo_relative
from scripts.depth.manifest import build_depth_manifest, save_json


def test_synthetic_vda_npz_to_depth_manifest(tmp_path: Path) -> None:
    """Integration-style: single VDA *_depths.npz -> per-frame files + manifest + run record."""
    root = tmp_path / "repo"
    depth_dir = root / "data" / "depth"
    manifests = root / "data" / "manifests"
    runs = root / "outputs" / "reconstructions" / "depth_prior"
    vda_out = tmp_path / "vda_out"
    for path in (depth_dir, manifests, runs, vda_out):
        path.mkdir(parents=True)

    frames = [
        {"frame_id": "demo_0001", "path": "data/frames/demo/0001.png", "selected": True},
        {"frame_id": "demo_0002", "path": "data/frames/demo/0002.png", "selected": True},
    ]
    depths = np.stack(
        [
            np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4),
            np.linspace(1, 2, 12, dtype=np.float32).reshape(3, 4),
        ],
        axis=0,
    )
    np.savez_compressed(vda_out / "input_depths.npz", depths=depths)

    loaded = load_vda_depths_array(find_vda_depths_npz(vda_out))
    records = split_vda_depths_to_frame_files(
        depths=loaded,
        frames=frames,
        depth_dir=depth_dir,
        root=root,
        depth_type="relative",
    )
    depth_manifest_path = manifests / "depth_manifest.json"
    save_json(
        depth_manifest_path,
        build_depth_manifest(
            frames_manifest={"schema_version": "1.0", "video_id": "demo"},
            frame_records=records,
            backend={
                "name": "video-depth-anything",
                "commit": "4f5ae23172ba60fd7bc11ef671cca678842c7072",
                "encoder": "vitb",
            },
            depth_type="relative",
            frames_manifest_path="data/manifests/frames_manifest.json",
        ),
    )
    run_record_path = runs / "run_record.json"
    write_run_record(
        run_record_path,
        {
            "module": "depth-prior",
            "config": "configs/depth/default_vitb.yaml",
            "frames_manifest": "data/manifests/frames_manifest.json",
            "backend_commit": "4f5ae23172ba60fd7bc11ef671cca678842c7072",
            "command": ["python", "run.py", "--save_npz"],
            "outputs": {
                "depth_dir": to_repo_relative(root, depth_dir),
                "depth_manifest": to_repo_relative(root, depth_manifest_path),
                "frame_count": len(records),
            },
        },
    )

    payload = json.loads(depth_manifest_path.read_text(encoding="utf-8"))
    assert len(payload["frames"]) == 2
    assert payload["frames"][0]["depth_path"] == "data/depth/demo_0001.npz"
    assert payload["source_video_id"] == "demo"
    assert payload["source_frames_manifest"] == "data/manifests/frames_manifest.json"
    assert payload["frame_depth_mapping"] == "strict_positional"
    run_payload = json.loads(run_record_path.read_text(encoding="utf-8"))
    assert run_payload["frames_manifest"] == "data/manifests/frames_manifest.json"
    assert not Path(run_payload["config"]).is_absolute()
    assert run_payload["outputs"]["frame_count"] == 2


def test_default_vitb_fingerprint_matches_canonical_lf_bytes() -> None:
    import hashlib

    expected = "d0ae1adc17f7ea60ceec74616d53c723a6d28349be6f225c4918121b53e29671"
    # This test targets the proposed worktree content before it is committed.
    worktree = (ROOT / "configs" / "depth" / "default_vitb.yaml").read_bytes()
    canonical_worktree = worktree.replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical_worktree).hexdigest() == expected
    pin = (ROOT / "configs" / "depth" / "backend_pin.md").read_text(encoding="utf-8")
    assert expected in pin
    # Ensure YAML still loads after any checkout conversion.
    data = yaml.safe_load(worktree.decode("utf-8"))
    assert data["backend"]["encoder"] == "vitb"
