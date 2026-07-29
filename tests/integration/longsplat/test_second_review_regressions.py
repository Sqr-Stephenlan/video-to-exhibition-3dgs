"""Permanent regression tests promoted from second-review counterexamples.

These tests encode exact failure modes found during the second review that
the existing 88-test suite did not catch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.longsplat.depth_bridge import (
    DepthContractError,
    materialize_all,
)
from scripts.longsplat.orchestrator import run_pipeline
from scripts.longsplat.runner import LongSplatConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_fake_backend(tmp_path: Path) -> Path:
    """Create a minimal git repo with fake train.py / convert_3dgs.py."""
    repo = tmp_path / "fake_longsplat"
    repo.mkdir()

    import textwrap

    (repo / "train.py").write_text(
        textwrap.dedent("""\
        import json, sys, os
        with open("orchestrator_test_sentinel.json", "w") as fh:
            json.dump({"script": "train", "cwd": os.getcwd(), "argv": sys.argv}, fh)
        model_path = None
        for i, a in enumerate(sys.argv):
            if a == "--model_path" and i + 1 < len(sys.argv):
                model_path = sys.argv[i + 1]
                break
        if model_path:
            os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, "cameras_all_train.json"), "w") as f:
                json.dump([], f)
        sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
    """)
    )

    (repo / "convert_3dgs.py").write_text(
        textwrap.dedent("""\
        import json, sys, os, struct as _struct

        def _write_minimal_ply(path):
            verts = [(
                0.0, 0.0, 0.0,  0.5, 0.3, 0.1,  0.9,
                0.01, 0.01, 0.01,  1.0, 0.0, 0.0, 0.0,
            )]
            lines = [
                "ply",
                "format binary_little_endian 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property float f_dc_0",
                "property float f_dc_1",
                "property float f_dc_2",
                "property float opacity",
                "property float scale_0",
                "property float scale_1",
                "property float scale_2",
                "property float rot_0",
                "property float rot_1",
                "property float rot_2",
                "property float rot_3",
                "end_header",
            ]
            with open(path, "wb") as f:
                f.write("\\n".join(lines).encode() + b"\\n")
                for v in verts:
                    f.write(_struct.pack("<" + "f" * 14, *v))

        with open("orchestrator_test_sentinel_convert.json", "w") as fh:
            json.dump({"script": "convert", "cwd": os.getcwd(), "argv": sys.argv}, fh)
        model_path = None
        for i, a in enumerate(sys.argv):
            if a == "--model_path" and i + 1 < len(sys.argv):
                model_path = sys.argv[i + 1]
                break
        if model_path:
            conv_dir = os.path.join(model_path, "converted_3dgs")
            os.makedirs(conv_dir, exist_ok=True)
            _write_minimal_ply(os.path.join(conv_dir, "point_cloud.ply"))
        sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
    """)
    )

    import subprocess

    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test"],
        cwd=str(repo),
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo),
        capture_output=True,
    )

    for sub in ["mast3r", "diff-gaussian-rasterization", "fused-ssim", "simple-knn"]:
        sub_path = repo / "submodules" / sub
        sub_path.mkdir(parents=True)
        subprocess.run(
            ["git", "-C", str(sub_path), "init", "-b", "main"],
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(sub_path), "config", "user.email", "test@test"],
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(sub_path), "config", "user.name", "Test"],
            capture_output=True,
        )
        (sub_path / "dummy").write_text("")
        subprocess.run(["git", "-C", str(sub_path), "add", "."], capture_output=True)
        subprocess.run(
            ["git", "-C", str(sub_path), "commit", "-m", "init"],
            capture_output=True,
        )

    (repo / ".gitignore").write_text("submodules/\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(repo),
        capture_output=True,
    )

    return repo


def _make_producer_manifest(tmp_path: Path) -> Path:
    """Create a minimal preprocess-video producer manifest with real frame files."""
    seg = "seg_01"
    frame_dir = tmp_path / "frames" / seg
    frame_dir.mkdir(parents=True)

    frames = []
    timestamps = [1.0, 5.0, 9.0]
    for i, timestamp in enumerate(timestamps):
        fname = f"frame_{i:06d}.jpg"
        path = frame_dir / fname
        ok, encoded = cv2.imencode(
            ".jpg",
            np.full((480, 640, 3), 64 + i, dtype=np.uint8),
        )
        assert ok
        payload = encoded.tobytes()
        path.write_bytes(payload)
        frames.append(
            {
                "id": f"frame_{i:06d}",
                "segment_id": seg,
                "path": str(path.relative_to(tmp_path)),
                "timestamp_sec": timestamp,
                "frame_index": i,
                "selected": True,
                "blur_score": 12.0,
                "overexposed_ratio": 0.05,
                "underexposed_ratio": 0.03,
                "duplicate_score": None,
                "reject_reasons": [],
                "width": 640,
                "height": 480,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )

    manifest_path = tmp_path / "frames_manifest.json"
    manifest = {
        "schema_version": "2.0",
        "video_id": "test",
        "source": {
            "path": "raw/test.mp4",
            "size_bytes": 1000,
            "duration_sec": 10,
            "fps": 30,
            "width": 640,
            "height": 480,
            "codec": "h264",
            "sha256": "a" * 64,
            "rotation_degrees": 0,
            "is_vfr": False,
        },
        "normalized": {
            "path": "seg/test/normalized.mp4",
            "duration_sec": 10,
            "fps": 10,
            "width": 640,
            "height": 480,
            "codec": "h264",
            "sha256": "b" * 64,
            "rotation_degrees": 0,
            "is_vfr": False,
        },
        "settings": {
            "preset": "longsplat",
            "target_fps": 10,
            "max_long_edge": 512,
            "segment_method": "time",
            "segment_length_sec": 30,
            "segment_overlap_sec": 10,
            "keyframe_policy": "legacy",
        },
        "segments": [
            {
                "id": seg,
                "path": f"seg/test/{seg}.mp4",
                "index": 0,
                "start_sec": 0,
                "end_sec": 10,
                "duration_sec": 10,
                "reason": "time",
                "width": 640,
                "height": 480,
                "sha256": "c" * 64,
            }
        ],
        "frames": frames,
        "summary": {
            "total_segments": 1,
            "total_frames": 3,
            "selected_frames": 3,
            "rejected_frames": 0,
            "frames_by_segment": {
                seg: {"total": 3, "selected": 3, "rejected": 0},
            },
            "reject_reasons": {},
        },
        "run": {
            "id": "test-source-settings",
            "code": {
                "commit": "75b5204fa313c84c613141b89a0ba4462db599a1",
                "working_tree_dirty": False,
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _make_depth_assets(
    tmp_path: Path,
    *,
    depth_data: np.ndarray | None = None,
    escape: bool = False,
    fake_hash: bool = False,
) -> tuple[Path, Path, Path]:
    """Create depth manifest, NPZ, and frame_mapping for depth-bridge tests."""
    run_dir = tmp_path / "run"
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)

    mapping = [
        {"source_path": "frames/frame_000000.jpg", "prepared_name": "frame_000000.jpg"},
    ]
    (input_dir / "frame_mapping.json").write_text(json.dumps(mapping))

    if depth_data is None:
        depth_data = np.ones((64, 64), dtype=np.float32)

    depth_dir_data = tmp_path / "depths"
    depth_dir_data.mkdir(parents=True, exist_ok=True)
    npz_path = depth_dir_data / "d_000000.npz"
    np.savez(npz_path, depth=depth_data)

    # Compute real SHA-256
    hasher = hashlib.sha256()
    with open(npz_path, "rb") as fh:
        hasher.update(fh.read())
    real_sha = hasher.hexdigest()
    stored_sha = "a" * 64 if fake_hash else real_sha

    if escape:
        depth_path_value = f"../outside/{npz_path.name}"
    else:
        depth_path_value = str(npz_path.relative_to(tmp_path))

    depth_manifest = {
        "frames": [
            {
                "rgb_path": "frames/frame_000000.jpg",
                "depth_path": depth_path_value,
                "sha256": stored_sha,
            },
        ],
    }
    depth_manifest_path = tmp_path / "depth_manifest.json"
    depth_manifest_path.write_text(json.dumps(depth_manifest))

    return depth_manifest_path, run_dir, npz_path


# ---------------------------------------------------------------------------
# Test 1: Path containment
# ---------------------------------------------------------------------------


def test_relative_depth_path_cannot_escape_project_root(tmp_path):
    """A depth_path containing ../ must be rejected even if the file exists."""
    outside = tmp_path / "outside"
    outside.mkdir()
    # Create the NPZ at the escaped location so it "exists" but must still be rejected
    escape_npz = outside / "d_000000.npz"
    np.savez(escape_npz, depth=np.ones((64, 64), dtype=np.float32))

    depth_manifest_path, run_dir, _npz_path = _make_depth_assets(
        tmp_path,
        escape=True,
    )

    with pytest.raises(DepthContractError) as exc_info:
        materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=json.loads(
                (run_dir / "input" / "frame_mapping.json").read_text(),
            ),
            project_root=tmp_path,
            output_depth_dir=run_dir / "input" / "depths",
        )

    msg = str(exc_info.value).lower()
    assert "escape" in msg or "contain" in msg or "traversal" in msg or ".." in msg


# ---------------------------------------------------------------------------
# Test 2: Depth must be 2D
# ---------------------------------------------------------------------------


def test_depth_array_must_be_two_dimensional(tmp_path):
    """A 1D depth array must be rejected with a clear error."""
    depth_1d = np.ones((64,), dtype=np.float32)

    depth_manifest_path, run_dir, _npz_path = _make_depth_assets(
        tmp_path,
        depth_data=depth_1d,
    )

    with pytest.raises(DepthContractError) as exc_info:
        materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=json.loads(
                (run_dir / "input" / "frame_mapping.json").read_text(),
            ),
            project_root=tmp_path,
            output_depth_dir=run_dir / "input" / "depths",
        )

    msg = str(exc_info.value).lower()
    assert "2-d" in msg or "ndim" in msg or "dimension" in msg or "shape" in msg


# ---------------------------------------------------------------------------
# Test 3: Depth must be float32
# ---------------------------------------------------------------------------


def test_depth_array_must_be_float32(tmp_path):
    """An int32 depth array must be rejected."""
    depth_int32 = np.ones((64, 64), dtype=np.int32)

    depth_manifest_path, run_dir, _npz_path = _make_depth_assets(
        tmp_path,
        depth_data=depth_int32,
    )

    with pytest.raises(DepthContractError) as exc_info:
        materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=json.loads(
                (run_dir / "input" / "frame_mapping.json").read_text(),
            ),
            project_root=tmp_path,
            output_depth_dir=run_dir / "input" / "depths",
        )

    msg = str(exc_info.value).lower()
    assert "dtype" in msg or "float32" in msg or "int32" in msg


# ---------------------------------------------------------------------------
# Test 4: SHA-256 must match manifest
# ---------------------------------------------------------------------------


def test_depth_npz_sha256_must_match_manifest(tmp_path):
    """An NPZ whose hash does not match the manifest sha256 is rejected."""
    depth_manifest_path, run_dir, _npz_path = _make_depth_assets(
        tmp_path,
        fake_hash=True,
    )

    with pytest.raises(DepthContractError) as exc_info:
        materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=json.loads(
                (run_dir / "input" / "frame_mapping.json").read_text(),
            ),
            project_root=tmp_path,
            output_depth_dir=run_dir / "input" / "depths",
        )

    msg = str(exc_info.value).lower()
    assert "sha256" in msg or "hash" in msg or "mismatch" in msg


# ---------------------------------------------------------------------------
# Test 5: float64 f_rest is rejected
# ---------------------------------------------------------------------------


def test_float64_f_rest_is_rejected(tmp_path):
    """A PLY with float64 f_rest_0 must fail validation."""
    from plyfile import PlyData, PlyElement

    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
            ("f_rest_0", "f8"),  # float64
        ]
    )
    verts = np.zeros(3, dtype=dtype)
    verts["x"] = np.array([0.0, 1.0, 2.0], dtype="f4")
    verts["y"] = np.array([0.0, 1.0, 2.0], dtype="f4")
    verts["z"] = np.array([0.0, 1.0, 2.0], dtype="f4")
    verts["f_dc_0"] = np.array([0.5, 0.5, 0.5], dtype="f4")
    verts["f_dc_1"] = np.array([0.3, 0.3, 0.3], dtype="f4")
    verts["f_dc_2"] = np.array([0.1, 0.1, 0.1], dtype="f4")
    verts["opacity"] = np.array([0.9, 0.9, 0.9], dtype="f4")
    verts["scale_0"] = verts["scale_1"] = verts["scale_2"] = np.array(
        [0.01, 0.01, 0.01],
        dtype="f4",
    )
    verts["rot_0"] = np.array([1.0, 1.0, 1.0], dtype="f4")
    verts["rot_1"] = np.array([0.0, 0.0, 0.0], dtype="f4")
    verts["rot_2"] = np.array([0.0, 0.0, 0.0], dtype="f4")
    verts["rot_3"] = np.array([0.0, 0.0, 0.0], dtype="f4")
    verts["f_rest_0"] = np.array([0.1, 0.2, 0.3], dtype="f8")

    el = PlyElement.describe(verts, "vertex")
    ply = PlyData([el], text=False)
    ply_path = tmp_path / "test_f64.ply"
    ply.write(ply_path)

    from scripts.longsplat.convert import (
        ConvertedPLYValidationError,
        validate_converted_ply,
    )

    with pytest.raises(ConvertedPLYValidationError) as exc_info:
        validate_converted_ply(ply_path)

    msg = str(exc_info.value).lower()
    assert "dtype" in msg or "f8" in msg or "float64" in msg or "f_rest" in msg


# ---------------------------------------------------------------------------
# Test 6: Backend preflight failure writes a failed record
# ---------------------------------------------------------------------------


def test_backend_preflight_failure_writes_failed_record(tmp_path):
    """When _check_repo raises, a reconstruction_run.json with
    status == "failed" must exist."""
    backend = tmp_path / "not_a_repo"
    backend.mkdir()

    manifest = _make_producer_manifest(tmp_path)
    output_dir = tmp_path / "outputs"
    config = LongSplatConfig(
        source_path="",
        model_path="",
        iterations=100,
        seed=0,
    )

    exit_code = run_pipeline(
        manifest_path=manifest,
        segment_id="seg_01",
        config=config,
        repo_root=backend,
        output_dir=output_dir,
        project_root=tmp_path,
        python_exe=sys.executable,
    )

    assert exit_code == 1
    assert output_dir.is_dir()

    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1, f"Expected exactly 1 run directory, got {len(run_dirs)}"

    record_path = run_dirs[0] / "reconstruction_run.json"
    assert record_path.is_file(), f"No reconstruction_run.json found in {run_dirs[0]}"
    record = json.loads(record_path.read_text())
    assert record["status"] == "failed", (
        f"Expected status 'failed', got {record.get('status')}"
    )

    preflight = record.get("stages", {}).get("preflight", {})
    assert preflight.get("reason") == "BackendValidationError", (
        f"Expected BackendValidationError, got {preflight}"
    )


# ---------------------------------------------------------------------------
# Test 7: Record argv matches executed commands
# ---------------------------------------------------------------------------


def test_record_argv_is_the_executed_train_and_convert_commands(tmp_path):
    """The record must store the exact argv lists passed to subprocess.run."""
    backend = _make_fake_backend(tmp_path)
    manifest = _make_producer_manifest(tmp_path)
    output_dir = tmp_path / "outputs"
    config = LongSplatConfig(
        source_path="",
        model_path="",
        iterations=100,
        seed=0,
    )

    import subprocess

    result = subprocess.run(
        ["git", "-C", str(backend), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    fake_commit = result.stdout.strip()

    import scripts.longsplat.runner as _runner

    orig_commit = _runner.LONGSPLAT_COMMIT
    orig_links = dict(_runner._LONGSPLAT_SUBMODULE_LINKS)
    _runner.LONGSPLAT_COMMIT = fake_commit
    for sub_key in list(_runner._LONGSPLAT_SUBMODULE_LINKS.keys()):
        sp = backend / sub_key
        r = subprocess.run(
            ["git", "-C", str(sp), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        _runner._LONGSPLAT_SUBMODULE_LINKS[sub_key] = r.stdout.strip()

    try:
        exit_code = run_pipeline(
            manifest_path=manifest,
            segment_id="seg_01",
            config=config,
            repo_root=backend,
            output_dir=output_dir,
            project_root=tmp_path,
            python_exe=sys.executable,
        )
    finally:
        _runner.LONGSPLAT_COMMIT = orig_commit
        _runner._LONGSPLAT_SUBMODULE_LINKS.clear()
        _runner._LONGSPLAT_SUBMODULE_LINKS.update(orig_links)

    assert exit_code == 0

    # Read sentinel files
    train_sentinel = json.loads(
        (backend / "orchestrator_test_sentinel.json").read_text(),
    )
    convert_sentinel = json.loads(
        (backend / "orchestrator_test_sentinel_convert.json").read_text(),
    )

    # Read run record
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    record = json.loads(
        (run_dirs[0] / "reconstruction_run.json").read_text(),
    )
    assert record["status"] == "complete"

    # Key assertion: record has "commands" with exact argv
    assert "commands" in record, "record is missing 'commands' key"
    assert "train" in record["commands"], "record.commands is missing 'train'"
    assert "convert" in record["commands"], "record.commands is missing 'convert'"

    recorded_train_argv = record["commands"]["train"]["argv"]
    recorded_convert_argv = record["commands"]["convert"]["argv"]

    executed_train_argv = train_sentinel["argv"]
    executed_convert_argv = convert_sentinel["argv"]

    # recorded argv includes the python interpreter; sys.argv inside the
    # subprocess starts at the script path.  Compare the script+args tail.
    assert recorded_train_argv[1:] == executed_train_argv, (
        f"Train argv mismatch:\n  recorded: {recorded_train_argv}\n"
        f"  executed: {executed_train_argv}"
    )
    assert recorded_convert_argv[1:] == executed_convert_argv, (
        f"Convert argv mismatch:\n  recorded: {recorded_convert_argv}\n"
        f"  executed: {executed_convert_argv}"
    )

    # seed=0 must NOT be passed to the train command
    assert "--seed" not in recorded_train_argv, (
        f"train command must not contain --seed, got: {recorded_train_argv}"
    )


# ---------------------------------------------------------------------------
# Task 2: Depth coverage + VDA lightweight evidence
# ---------------------------------------------------------------------------


def test_surplus_depth_entry_is_rejected(tmp_path):
    """Depth manifest entries not in frame mapping must raise an error."""
    depth_manifest_path, run_dir, _npz_path = _make_depth_assets(tmp_path)

    # Add a surplus entry to the manifest that's not in frame_mapping
    manifest = json.loads(depth_manifest_path.read_text())
    manifest["frames"].append(
        {
            "rgb_path": "frames/extra_surplus.jpg",
            "depth_path": str(_npz_path.relative_to(tmp_path)),
            "sha256": manifest["frames"][0]["sha256"],
        }
    )
    depth_manifest_path.write_text(json.dumps(manifest))

    mapping_path = run_dir / "input" / "frame_mapping.json"
    frame_mapping = json.loads(mapping_path.read_text())

    with pytest.raises(DepthContractError, match="unused depth entries"):
        materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=frame_mapping,
            project_root=tmp_path,
            output_depth_dir=run_dir / "input" / "depths",
        )


def test_duplicate_depth_output_is_rejected(tmp_path):
    """Two entries that map to the same depth output name must raise."""
    depth_manifest_path, run_dir, npz_path = _make_depth_assets(tmp_path)

    # Create a second NPZ for the duplicate
    npz2 = npz_path.parent / "d_000001.npz"
    np.savez(npz2, depth=np.ones((64, 64), dtype=np.float32))
    hasher = hashlib.sha256()
    hasher.update(npz2.read_bytes())
    sha2 = hasher.hexdigest()

    # Add a second manifest entry with same prepared_name stem
    manifest = json.loads(depth_manifest_path.read_text())
    manifest["frames"].append(
        {
            "rgb_path": "frames/frame_000001.jpg",
            "depth_path": str(npz2.relative_to(tmp_path)),
            "sha256": sha2,
        }
    )
    depth_manifest_path.write_text(json.dumps(manifest))

    # frame_mapping has two entries with same stem → duplicate output
    mapping = [
        {"source_path": "frames/frame_000000.jpg", "prepared_name": "frame_000000.jpg"},
        {"source_path": "frames/frame_000001.jpg", "prepared_name": "frame_000000.jpg"},
    ]
    mapping_path = run_dir / "input" / "frame_mapping.json"
    mapping_path.write_text(json.dumps(mapping))

    with pytest.raises(DepthContractError, match="duplicate"):
        materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=mapping,
            project_root=tmp_path,
            output_depth_dir=run_dir / "input" / "depths",
        )


def test_vda_usage_summary():
    """_summarize_vda_usage parses VDA_USAGE markers into exact counts."""
    from scripts.longsplat.orchestrator import _summarize_vda_usage

    stdout = "\n".join(
        [
            "VDA_USAGE result=aligned stage=scene_init frame=frame_000000.jpg",
            "some other output",
            "VDA_USAGE result=missing stage=scene_init frame=frame_000001.jpg",
            "",
            "VDA_USAGE result=rejected stage=incremental frame=frame_000002.jpg",
        ]
    )

    usage = _summarize_vda_usage(stdout)
    assert usage == {"aligned": 1, "missing": 1, "rejected": 1}
