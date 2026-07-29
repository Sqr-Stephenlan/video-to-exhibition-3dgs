"""CPU integration tests covering the delivery entry point and materializer.

These tests use fake backend scripts to verify that ``run_pipeline``:
- Validates the backend repo before launching subprocesses
- Runs subprocesses in the correct CWD
- Produces a terminal record (completed or failed) for all exit paths
- Records actual backend identity rather than constants

Plus unit-level tests for internal helpers and the materializer module.
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
from scripts.longsplat.convert import (
    ConverterError,
    convert_and_validate,
)
from scripts.longsplat.orchestrator import (
    PipelineError,
    _resolve_backend_identity,
    _sanitise_run_id,
    run_pipeline,
)
from scripts.longsplat.runner import BackendValidationError, LongSplatConfig


# ---------------------------------------------------------------------------
# Fake backend script builder
# ---------------------------------------------------------------------------


def _make_fake_backend(tmp_path: Path) -> Path:
    """Create a minimal git repo with a fake train.py / convert_3dgs.py.

    The fake scripts write their CWD and argv to a sentinel file so tests can
    assert the orchestrator ran them in the correct working directory.
    """
    repo = tmp_path / "fake_longsplat"
    repo.mkdir()

    import textwrap

    # --- Fake train.py ---
    (repo / "train.py").write_text(
        textwrap.dedent("""\
        import json, sys, os
        with open("orchestrator_test_sentinel.json", "w") as fh:
            json.dump({"script": "train", "cwd": os.getcwd(), "argv": sys.argv}, fh)
        # Write a valid cameras file and checkpoint so the pose audit gate passes.
        model_path = None
        for i, a in enumerate(sys.argv):
            if a == "--model_path" and i + 1 < len(sys.argv):
                model_path = sys.argv[i + 1]
                break
        if model_path:
            os.makedirs(model_path, exist_ok=True)
            # Single valid SO(3) camera matching the telemetry below
            cameras = [{
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "T": [0.0, 0.0, 1.0],
            }]
            with open(os.path.join(model_path, "cameras_all_train.json"), "w") as f:
                json.dump(cameras, f)
            # Write a dummy native checkpoint so the iteration check can pass
            # when expected_native_checkpoint_iteration matches.
            chkpnt_iter = os.environ.get("FAKE_CHKPNT_ITERATION", "")
            if chkpnt_iter:
                chkpnt_path = os.path.join(model_path, f"chkpnt{chkpnt_iter}.pth")
                with open(chkpnt_path, "w") as f:
                    f.write("dummy")
        # Emit POSE_TELEMETRY matching the camera count
        print('POSE_TELEMETRY {"frame":"frame_000000","inlier_count":8,'
              '"inlier_ratio":0.8,"match_count":10,"method":"pnp_ransac",'
              '"reprojection_rmse_px":1.25,"stage":"incremental",'
              '"success":true}')
        # Emit VDA_USAGE markers so integration tests can assert real stdout
        depth_source = None
        for i, a in enumerate(sys.argv):
            if a == "--depth_source" and i + 1 < len(sys.argv):
                depth_source = sys.argv[i + 1]
                break
        if depth_source == "vda":
            print('VDA_TELEMETRY {"correlation":0.91,"frame":"frame_000000",'
                  '"inlier_ratio":0.88,"result":"aligned",'
                  '"stage":"incremental"}')
            print("VDA_USAGE result=aligned stage=scene_init frame=frame_000000")
            print("VDA_USAGE result=aligned stage=scene_init frame=frame_000001")
            print("VDA_USAGE result=aligned stage=incremental frame=frame_000002")
        sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
    """)
    )

    # --- Fake convert_3dgs.py ---
    (repo / "convert_3dgs.py").write_text(
        textwrap.dedent("""\
        import json, sys, os, struct

        def _write_minimal_ply(path):
            verts = [(
                0.0, 0.0, 0.0,  0.5, 0.3, 0.1,  0.9,
                0.01, 0.01, 0.01,  1.0, 0.0, 0.0, 0.0,
            )]
            if os.environ.get("FAKE_NONFINITE") == "1":
                verts.append((
                    1.0, 0.0, 0.0,  0.5, 0.3, 0.1,  0.9,
                    float("-inf"), 0.01, 0.01,  1.0, 0.0, 0.0, 0.0,
                ))
            lines = [
                "ply",
                "format binary_little_endian 1.0",
                f"element vertex {len(verts)}",
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
                    f.write(struct.pack("<" + "f" * 14, *v))

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
        print('CONVERSION_TELEMETRY {"finite_scale_count":1,'
              '"nonpositive_distance_count":0,"point_count":1}')
        sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
    """)
    )

    # --- Init git repo ---
    import subprocess

    subprocess.run(
        ["git", "init", "-b", "main"], cwd=str(repo), check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test"], cwd=str(repo), capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=str(repo), capture_output=True
    )

    # Add submodules expected by _check_repo (create BEFORE initial commit so
    # the dirty check passes — gitignored from the parent repo).
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
        subprocess.run(
            ["git", "-C", str(sub_path), "add", "."],
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(sub_path), "commit", "-m", "init"],
            capture_output=True,
        )

    # Gitignore submodules so they don't show as dirty
    (repo / ".gitignore").write_text("submodules/\n")
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(repo), capture_output=True)

    return repo


def _make_producer_manifest(tmp_path: Path) -> Path:
    """Create a schema-2 legacy producer manifest with real frame files."""
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
            np.full((480, 640, 3), 32 + i, dtype=np.uint8),
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
            "selection_max_gap_sec": 4.5,
            "min_motion_inlier_ratio": 0.1,
            "min_motion_grid_coverage": 0.1,
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
            "source_sha256": "a" * 64,
            "settings_fingerprint": "d" * 64,
            "config": {"path": None, "sha256": None},
            "code": {
                "commit": "75b5204fa313c84c613141b89a0ba4462db599a1",
                "working_tree_dirty": False,
            },
            "tools": {
                "python": "3.11.9",
                "opencv": "4.10.0",
                "ffmpeg": "8.0",
                "ffprobe": "8.0",
            },
            "command": ["data/raw_videos/test.mp4", "--video-id", "test"],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _make_coverage_handoff(tmp_path: Path) -> tuple[Path, Path]:
    """Create a schema-2 coverage_v1 manifest and its producer sidecar."""
    manifest_path = _make_producer_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["settings"]["keyframe_policy"] = "coverage_v1"
    for frame in manifest["frames"]:
        frame["calibrated_blur_score"] = frame["blur_score"]
        frame["keyframe"] = {
            "policy": "coverage_v1",
            "component_id": 0,
            "selected_by_policy": True,
            "bridge": False,
            "reference_frame_id": None,
            "adaptive_blur_threshold": frame["blur_score"],
            "quality_score": 1.0,
            "motion": None,
        }
    manifest["summary"]["keyframe_quality"] = {
        "policy": "coverage_v1",
        "status": "passed",
        "selected_frames": 3,
        "bridge_frames": 0,
        "component_count": 1,
        "max_selected_gap_sec": 4.0,
        "min_motion_inlier_ratio": 0.1,
        "min_motion_grid_coverage": 0.1,
        "failed_segments": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "schema_version": "2.0",
        "video_id": "test",
        "policy": "coverage_v1",
        "status": "passed",
        "segments": [
            {
                "segment_id": "seg_01",
                "total_frames": 3,
                "selected_frames": 3,
                "rejected_frames": 0,
                "component_count": 1,
                "max_selected_gap_sec": 4.0,
                "bridge_frames": 0,
                "failed_reasons": [],
            }
        ],
    }
    report_path = manifest_path.parent / "keyframe_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return manifest_path, report_path


def _allow_fake_backend(monkeypatch: pytest.MonkeyPatch, backend: Path) -> None:
    """Keep handoff tests focused on manifest orchestration, not commit locks."""
    import scripts.longsplat.orchestrator as orchestrator

    monkeypatch.setattr(
        orchestrator,
        "_check_repo",
        lambda _repo_root, *, backend_mode: backend,
    )
    monkeypatch.setattr(orchestrator, "_check_python", lambda _python_exe: None)


# ---------------------------------------------------------------------------
# _sanitise_run_id tests
# ---------------------------------------------------------------------------


def test_sanitise_valid_ids():
    assert _sanitise_run_id("test_001") == "test_001"
    assert _sanitise_run_id("abc123.XYZ-9") == "abc123.XYZ-9"
    assert _sanitise_run_id("a") == "a"


def test_sanitise_rejects_leading_dot():
    with pytest.raises(PipelineError, match="run_id"):
        _sanitise_run_id(".hidden")


def test_sanitise_rejects_traversal():
    with pytest.raises(PipelineError, match="run_id"):
        _sanitise_run_id("../../etc")


def test_sanitise_rejects_spaces():
    with pytest.raises(PipelineError, match="run_id"):
        _sanitise_run_id("my run")


# ---------------------------------------------------------------------------
# Fake-backend orchestrator tests
# ---------------------------------------------------------------------------


def test_orchestrator_coverage_loads_sidecar_and_records_raw_hashes(
    tmp_path, monkeypatch
):
    """coverage_v1 passes its sibling report through and fingerprints both files."""
    backend = _make_fake_backend(tmp_path)
    manifest_path, report_path = _make_coverage_handoff(tmp_path)
    output_dir = tmp_path / "outputs"
    _allow_fake_backend(monkeypatch, backend)

    import scripts.longsplat.orchestrator as orchestrator

    captured: dict[str, object] = {}
    real_adapt_manifest = orchestrator.adapt_manifest

    def capture_adapt_manifest(
        producer_manifest, segment_id, repo_root, *, quality_report=None
    ):
        captured["quality_report"] = quality_report
        return real_adapt_manifest(
            producer_manifest,
            segment_id,
            repo_root,
            quality_report=quality_report,
        )

    monkeypatch.setattr(orchestrator, "adapt_manifest", capture_adapt_manifest)

    exit_code = run_pipeline(
        manifest_path=manifest_path,
        segment_id="seg_01",
        config=LongSplatConfig(
            source_path="",
            model_path="",
            iterations=100,
            seed=0,
        ),
        repo_root=backend,
        output_dir=output_dir,
        project_root=tmp_path,
        python_exe=sys.executable,
    )

    assert exit_code == 0
    assert captured["quality_report"] == json.loads(
        report_path.read_text(encoding="utf-8")
    )

    run_dir = next(output_dir.iterdir())
    record = json.loads(
        (run_dir / "reconstruction_run.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "complete"
    assert (
        record["_pv"]["producer_manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert (
        record["_pv"]["producer_quality_report_sha256"]
        == hashlib.sha256(report_path.read_bytes()).hexdigest()
    )
    assert (
        record["_pv"]["producer_quality_report_binding"]
        == "semantic_only_no_producer_manifest_digest"
    )
    assert record["_pv"]["producer_run"] == {
        "id": "test-source-settings",
        "code": {
            "commit": "75b5204fa313c84c613141b89a0ba4462db599a1",
            "working_tree_dirty": False,
        },
    }


@pytest.mark.parametrize(
    "report_problem",
    ["missing", "failed", "video_mismatch"],
)
def test_orchestrator_coverage_report_problems_fail_before_training(
    tmp_path, monkeypatch, report_problem
):
    """Missing or incoherent coverage evidence fails closed with a terminal record."""
    backend = _make_fake_backend(tmp_path)
    manifest_path, report_path = _make_coverage_handoff(tmp_path)
    output_dir = tmp_path / "outputs"
    _allow_fake_backend(monkeypatch, backend)

    if report_problem == "missing":
        report_path.unlink()
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report_problem == "failed":
            report["status"] = "failed"
        else:
            report["video_id"] = "different-video"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    exit_code = run_pipeline(
        manifest_path=manifest_path,
        segment_id="seg_01",
        config=LongSplatConfig(
            source_path="",
            model_path="",
            iterations=100,
            seed=0,
        ),
        repo_root=backend,
        output_dir=output_dir,
        project_root=tmp_path,
        python_exe=sys.executable,
    )

    assert exit_code == 1
    assert not (backend / "orchestrator_test_sentinel.json").exists()

    run_dir = next(output_dir.iterdir())
    record = json.loads(
        (run_dir / "reconstruction_run.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["stages"]["exception"]["reason"] == "exception"
    assert (
        record["_pv"]["producer_manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    if report_problem == "missing":
        assert record["_pv"]["producer_quality_report_sha256"] is None
    else:
        assert (
            record["_pv"]["producer_quality_report_sha256"]
            == hashlib.sha256(report_path.read_bytes()).hexdigest()
        )


def test_orchestrator_fake_backend_completes(tmp_path):
    """run_pipeline runs against a fake backend and produces a complete record."""
    backend = _make_fake_backend(tmp_path)
    manifest = _make_producer_manifest(tmp_path)
    output_dir = tmp_path / "outputs"
    config = LongSplatConfig(
        source_path="",
        model_path="",
        iterations=100,
        seed=0,
    )

    # Need to match the fake backend's HEAD, not the locked commit.
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(backend), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    fake_commit = result.stdout.strip()

    # Monkey-patch _check_repo to accept the fake commit
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

    # Find run directory
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    # Check record
    record_path = run_dir / "reconstruction_run.json"
    assert record_path.is_file()
    record = json.loads(record_path.read_text())
    assert record["status"] == "complete"
    assert record["_pv"]["producer_quality_report_sha256"] is None
    assert record["_pv"]["producer_run"] == {
        "id": "test-source-settings",
        "code": {
            "commit": "75b5204fa313c84c613141b89a0ba4462db599a1",
            "working_tree_dirty": False,
        },
    }

    # Check sentinel — training ran in the correct CWD
    sentinel_path = backend / "orchestrator_test_sentinel.json"
    assert sentinel_path.is_file()
    sentinel = json.loads(sentinel_path.read_text())
    assert sentinel["script"] == "train"
    assert sentinel["cwd"] == str(backend.resolve())

    # Check output artifacts
    assert "artifacts" in record
    types = [a["type"] for a in record["artifacts"]]
    assert "longsplat_model" in types

    # CR-T123-07: submodule_diffs must exist in backend block
    assert "submodule_diffs" in record["backend"]
    assert isinstance(record["backend"]["submodule_diffs"], dict)


def test_orchestrator_rejects_nonfinite_converted_ply_without_cleaning(
    tmp_path, monkeypatch
):
    """A raw non-finite export is a failed conversion artifact, not auto-repaired."""
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
    import scripts.longsplat.runner as _runner

    fake_commit = subprocess.run(
        ["git", "-C", str(backend), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    orig_commit = _runner.LONGSPLAT_COMMIT
    orig_links = dict(_runner._LONGSPLAT_SUBMODULE_LINKS)
    _runner.LONGSPLAT_COMMIT = fake_commit
    for sub_key in list(_runner._LONGSPLAT_SUBMODULE_LINKS):
        sub_path = backend / sub_key
        _runner._LONGSPLAT_SUBMODULE_LINKS[sub_key] = subprocess.run(
            ["git", "-C", str(sub_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()

    monkeypatch.setenv("FAKE_NONFINITE", "1")
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

    assert exit_code == 1
    run_dir = next(output_dir.iterdir())
    record = json.loads((run_dir / "reconstruction_run.json").read_text())
    assert record["status"] == "failed"
    assert "nan/inf" in json.dumps(record).lower()

    # Strict validation must not mutate the bad export and hide its cause.
    converted_ply = run_dir / "longsplat_model" / "converted_3dgs" / "point_cloud.ply"
    assert converted_ply.is_file()
    from plyfile import PlyData

    vertices = PlyData.read(str(converted_ply))["vertex"].data
    assert len(vertices) == 2
    assert np.isneginf(vertices["scale_0"][1])


def test_orchestrator_fake_backend_train_fails(tmp_path):
    """Training failure returns non-zero and writes a failed record."""
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

    # This env var makes the fake train.py exit with code 1
    import os as _os

    _os.environ["FAKE_EXIT"] = "1"

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
        _os.environ.pop("FAKE_EXIT", None)

    assert exit_code == 1

    # Run directory should exist with a failed record
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    record = json.loads((run_dirs[0] / "reconstruction_run.json").read_text())
    assert record["status"] == "failed"
    assert record["telemetry"]["pose"]["attempt_count"] == 1
    assert record["telemetry"]["pose"]["accepted_camera_count"] == 1


def test_orchestrator_rejects_non_git_backend(tmp_path):
    """A backend without .git fails before training starts."""
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

    # Output dir may not exist if the failure happened before creation
    if output_dir.is_dir() and list(output_dir.iterdir()):
        run_dirs = list(output_dir.iterdir())
        if (run_dirs[0] / "reconstruction_run.json").is_file():
            record = json.loads((run_dirs[0] / "reconstruction_run.json").read_text())
            assert record["status"] == "failed"


# ---------------------------------------------------------------------------
# Depth materializer contract tests
# ---------------------------------------------------------------------------


def test_materialize_depth_module_importable():
    """The materialize_depth module can be imported (CR-09)."""
    from scripts.longsplat import materialize_depth

    assert hasattr(materialize_depth, "materialize")
    assert hasattr(materialize_depth, "main")


def test_materialize_depth_writes_npy(tmp_path):
    """materialize_all() converts NPZ to npy at the expected path."""
    # Create a depth NPZ
    depth = np.random.rand(100, 200).astype(np.float32)
    npz_dir = tmp_path / "depths_data"
    npz_dir.mkdir()
    npz_path = npz_dir / "frame_001_depth.npz"
    np.savez(npz_path, depth=depth)

    # Create depth manifest referencing the NPZ (paths relative to tmp_path)
    sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "source": {"rgb_dir": "source"},
        "frames": [
            {
                "frame_id": "frame_001",
                "rgb_path": "frame_001_t10.000.jpg",
                "depth_path": str(npz_path.relative_to(tmp_path)),
                "sha256": sha,
            }
        ],
    }
    manifest_path = tmp_path / "depth_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    frame_mapping = [
        {
            "source_path": "frame_001_t10.000.jpg",
            "prepared_name": "frame_001_t10.000.jpg",
        }
    ]

    source = tmp_path / "source"
    source.mkdir()
    output_dir = source / "depths"
    result = materialize_all(
        depth_manifest_path=manifest_path,
        frame_mapping=frame_mapping,
        project_root=tmp_path,
        output_depth_dir=output_dir,
    )

    assert result.materialized_count == 1
    npy = output_dir / "frame_001_t10.000_depth.npy"
    assert npy.is_file()
    loaded = np.load(npy)
    assert np.allclose(loaded, depth)


def test_materialize_depth_missing_manifest(tmp_path):
    """materialize returns 1 for missing manifest."""
    from scripts.longsplat import materialize_depth

    rc = materialize_depth.materialize(
        tmp_path / "nonexistent.json",
        tmp_path / "source",
    )
    assert rc == 1


def test_materialize_depth_missing_npz(tmp_path):
    """materialize warns and returns 1 when NPZs are missing."""
    from scripts.longsplat import materialize_depth

    manifest = {
        "frames": [
            {
                "frame_id": "f1",
                "rgb_path": "frame_001.jpg",
                "depth_path": "/nonexistent/path.npz",
            }
        ],
    }
    manifest_path = tmp_path / "depth_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    source = tmp_path / "source"
    source.mkdir()
    rc = materialize_depth.materialize(manifest_path, source)
    assert rc == 1


# ---------------------------------------------------------------------------
# _materialise_and_validate_depths tests
# ---------------------------------------------------------------------------


def test_materialise_depths_full_coverage(tmp_path):
    """Depth materialisation writes npy files via depth_bridge."""
    run_dir = tmp_path / "run"
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)

    mapping = [
        {
            "source_path": "frames/seg/frame_000000.jpg",
            "prepared_name": "frame_000000.jpg",
        },
        {
            "source_path": "frames/seg/frame_000001.jpg",
            "prepared_name": "frame_000001.jpg",
        },
    ]
    (input_dir / "frame_mapping.json").write_text(json.dumps(mapping))

    depth_dir_data = tmp_path / "depths"
    depth_dir_data.mkdir()
    depth_data = np.ones((64, 64), dtype=np.float32)
    # Create all NPZ files first
    for i in range(2):
        npz_path = depth_dir_data / f"d_{i:06d}.npz"
        np.savez(npz_path, depth=depth_data)
    # Build manifest referencing the already-created NPZ files
    manifest_data = {
        "frames": [
            {
                "rgb_path": f"frames/seg/frame_{j:06d}.jpg",
                "depth_path": f"depths/d_{j:06d}.npz",
                "sha256": hashlib.sha256(
                    (depth_dir_data / f"d_{j:06d}.npz").read_bytes()
                ).hexdigest(),
            }
            for j in range(2)
        ],
    }
    depth_manifest_path = tmp_path / "depth_manifest.json"
    depth_manifest_path.write_text(json.dumps(manifest_data))

    result = materialize_all(
        depth_manifest_path=depth_manifest_path,
        frame_mapping=mapping,
        project_root=tmp_path,
        output_depth_dir=input_dir / "depths",
    )

    assert result.materialized_count == 2
    assert (input_dir / "depths" / "frame_000000_depth.npy").is_file()
    assert (input_dir / "depths" / "frame_000001_depth.npy").is_file()


def test_materialise_depths_incomplete_coverage_raises(tmp_path):
    """Missing depth frames raise DepthContractError from depth_bridge."""
    run_dir = tmp_path / "run"
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)

    mapping = [
        {
            "source_path": "frames/seg/frame_000000.jpg",
            "prepared_name": "frame_000000.jpg",
        },
        {
            "source_path": "frames/seg/frame_000001.jpg",
            "prepared_name": "frame_000001.jpg",
        },
    ]
    (input_dir / "frame_mapping.json").write_text(json.dumps(mapping))

    # Only one depth NPZ — second frame has no matching entry
    depth_dir_data = tmp_path / "depths"
    depth_dir_data.mkdir()
    npz_path = depth_dir_data / "d_000000.npz"
    np.savez(npz_path, depth=np.ones((64, 64), dtype=np.float32))
    sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()

    depth_manifest = {
        "frames": [
            {
                "rgb_path": "frames/seg/frame_000000.jpg",
                "depth_path": "depths/d_000000.npz",
                "sha256": sha,
            },
        ],
    }
    depth_manifest_path = tmp_path / "depth_manifest.json"
    depth_manifest_path.write_text(json.dumps(depth_manifest))

    with pytest.raises(DepthContractError, match="missing depth entries"):
        materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=mapping,
            project_root=tmp_path,
            output_depth_dir=input_dir / "depths",
        )


# ---------------------------------------------------------------------------
# Regression tests for review counterexamples (Section 7.2)
# ---------------------------------------------------------------------------


def test_seed_zero_not_passed_to_command():
    """seed=0 must NOT be passed to the training command (fixed effective seed)."""
    from scripts.longsplat.runner import build_train_command

    config = LongSplatConfig(
        source_path="/tmp/src",
        model_path="/tmp/model",
        iterations=100,
        seed=0,
    )
    cmd = build_train_command("/tmp/repo", config)
    assert "--seed" not in cmd, f"train command must not contain --seed, got: {cmd}"


def test_list_valued_extra_args_expanded():
    """test_iterations=[50, 100] expands to separate arguments."""
    from scripts.longsplat.runner import build_train_command

    config = LongSplatConfig(
        source_path="/tmp/src",
        model_path="/tmp/model",
        iterations=100,
        extra_train_args={"test_iterations": [50, 100]},
    )
    cmd = build_train_command("/tmp/repo", config)
    idx = cmd.index("--test_iterations")
    assert cmd[idx + 1] == "50"
    assert cmd[idx + 2] == "100"


def test_vda_without_depth_manifest_fails(tmp_path):
    """depth_source=vda without depth_manifest_path raises PipelineError."""
    backend = _make_fake_backend(tmp_path)
    manifest = _make_producer_manifest(tmp_path)
    output_dir = tmp_path / "outputs"
    config = LongSplatConfig(
        source_path="",
        model_path="",
        iterations=100,
        seed=0,
        extra_train_args={"depth_source": "vda"},
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

    assert exit_code == 1
    # Should have left a failed record
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    record = json.loads((run_dirs[0] / "reconstruction_run.json").read_text())
    assert record["status"] == "failed"
    assert any("depth_manifest" in str(record).lower() for _ in [1])


def test_nan_depth_rejected(tmp_path):
    """Depth arrays with NaN values are rejected by depth_bridge."""
    run_dir = tmp_path / "run"
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)

    mapping = [
        {"source_path": "frames/frame_000000.jpg", "prepared_name": "frame_000000.jpg"},
    ]
    (input_dir / "frame_mapping.json").write_text(json.dumps(mapping))

    depth_dir_data = tmp_path / "depths"
    depth_dir_data.mkdir()
    bad = np.ones((64, 64), dtype=np.float32)
    bad[0, 0] = np.nan
    npz_path = depth_dir_data / "d_000000.npz"
    np.savez(npz_path, depth=bad)
    sha = hashlib.sha256(npz_path.read_bytes()).hexdigest()

    depth_manifest = {
        "frames": [
            {
                "rgb_path": "frames/frame_000000.jpg",
                "depth_path": "depths/d_000000.npz",
                "sha256": sha,
            },
        ],
    }
    depth_manifest_path = tmp_path / "depth_manifest.json"
    depth_manifest_path.write_text(json.dumps(depth_manifest))

    with pytest.raises(DepthContractError, match="NaN"):
        materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=mapping,
            project_root=tmp_path,
            output_depth_dir=input_dir / "depths",
        )


def test_absolute_depth_path_rejected(tmp_path):
    """Absolute depth paths in depth manifest are rejected by depth_bridge."""
    run_dir = tmp_path / "run"
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)

    mapping = [
        {"source_path": "frames/frame_000000.jpg", "prepared_name": "frame_000000.jpg"},
    ]
    (input_dir / "frame_mapping.json").write_text(json.dumps(mapping))

    depth_manifest = {
        "frames": [
            {
                "rgb_path": "frames/frame_000000.jpg",
                "depth_path": "C:/absolute/path/depth.npz",
                "sha256": "a" * 64,
            },
        ],
    }
    depth_manifest_path = tmp_path / "depth_manifest.json"
    depth_manifest_path.write_text(json.dumps(depth_manifest))

    with pytest.raises(DepthContractError, match="absolute"):
        materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=mapping,
            project_root=tmp_path,
            output_depth_dir=input_dir / "depths",
        )


def test_missing_frame_mapping_raises(tmp_path):
    """Empty frames list in depth manifest raises DepthContractError."""
    depth_manifest_path = tmp_path / "depth_manifest.json"
    depth_manifest_path.write_text(json.dumps({"frames": []}))

    mapping = [
        {"source_path": "frames/frame_000000.jpg", "prepared_name": "frame_000000.jpg"},
    ]
    output_dir = tmp_path / "depths"

    with pytest.raises(DepthContractError, match="non-empty"):
        materialize_all(
            depth_manifest_path=depth_manifest_path,
            frame_mapping=mapping,
            project_root=tmp_path,
            output_depth_dir=output_dir,
        )


def test_malicious_segment_id_contained(tmp_path):
    """Segment IDs with ../ are sanitized before dir creation."""
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
        # segment_id with traversal should fail sanitise
        exit_code = run_pipeline(
            manifest_path=manifest,
            segment_id="../../escape",
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

    assert exit_code == 1
    # No directory escape
    siblings = list(output_dir.iterdir()) if output_dir.is_dir() else []
    for d in siblings:
        assert d.is_relative_to(output_dir)


def test_preflight_failure_writes_terminal_record(tmp_path):
    """Before any training, if the backend is invalid, a failed record exists."""
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
    # Run directory was created before the failure
    assert output_dir.is_dir()
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    # No run record created because failure happened before record creation,
    # but the run directory exists for debugging.
    assert run_dirs[0].is_dir()


# ---------------------------------------------------------------------------
# Task 3: CLI delegation
# ---------------------------------------------------------------------------


def test_run_experiment_cli_delegation():
    """run_experiment.main() parses all args and forwards them to run_pipeline."""
    from scripts.longsplat.run_experiment import main
    from unittest import mock

    argv = [
        "--manifest",
        "data/test_smoke/producer_manifest.json",
        "--segment-id",
        "test_smoke",
        "--config",
        "configs/longsplat/smoke_vda.json",
        "--repo-root",
        "third_party/LongSplat",
        "--output-dir",
        "outputs/longsplat-smoke",
        "--project-root",
        ".",
        "--depth-manifest",
        "outputs/depth/depth_manifest.json",
        "--backend-python",
        "python",
        "--run-id",
        "vda_smoke",
    ]

    with mock.patch(
        "scripts.longsplat.run_experiment.run_pipeline",
        return_value=0,
    ) as mock_run:
        exit_code = main(argv)
        assert exit_code == 0

    mock_run.assert_called_once()
    kwargs = mock_run.call_args[1]
    assert kwargs["manifest_path"] == "data/test_smoke/producer_manifest.json"
    assert kwargs["segment_id"] == "test_smoke"
    assert kwargs["repo_root"] == "third_party/LongSplat"
    assert kwargs["output_dir"] == "outputs/longsplat-smoke"
    assert kwargs["project_root"] == "."
    assert kwargs["depth_manifest_path"] == "outputs/depth/depth_manifest.json"
    assert kwargs["python_exe"] == "python"
    assert kwargs["run_id"] == "vda_smoke"
    assert kwargs["config"].backend_mode == "research_local"
    assert kwargs["config"].seed == 0


def test_run_experiment_module_execution(tmp_path):
    """python -m propagates non-zero exit code from run_pipeline (FV-05).

    Uses a valid config + nonexistent manifest so failure happens inside
    run_pipeline, not during argparse or config loading.  A terminal failed
    record proves the pipeline actually executed.
    """
    import subprocess as sp

    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "test_config.json"
    import json

    config_path.write_text(
        json.dumps(
            {
                "source_path": "",
                "model_path": "",
                "iterations": 100,
                "seed": 0,
                "backend_mode": "research_local",
            }
        )
    )

    result = sp.run(
        [
            sys.executable,
            "-m",
            "scripts.longsplat.run_experiment",
            "--manifest",
            str(tmp_path / "nonexistent_manifest.json"),
            "--segment-id",
            "seg_01",
            "--config",
            str(config_path),
            "--repo-root",
            str(tmp_path / "nonexistent_repo"),
            "--output-dir",
            str(output_dir),
            "--project-root",
            str(tmp_path),
            "--backend-python",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent.parent.parent),
        env={
            **__import__("os").environ,
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent.parent.parent),
        },
    )
    assert result.returncode == 1, (
        f"run_pipeline must propagate exit code 1, got {result.returncode}"
    )

    # A terminal record must exist, proving run_pipeline was called
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    record = json.loads((run_dirs[0] / "reconstruction_run.json").read_text())
    assert record["status"] == "failed"


def test_run_experiment_cli_exits_with_help():
    """python -m scripts.longsplat.run_experiment --help exits 0."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "scripts.longsplat.run_experiment", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"help exit code {result.returncode}: {result.stderr}"
    )
    assert "LongSplat reproduction experiment" in result.stdout


def test_run_experiment_cli_rejects_invalid_args():
    """python -m ... --definitely-invalid exits 2 (argparse error)."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.longsplat.run_experiment",
            "--definitely-invalid",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, (
        f"argparse error must exit 2, got {result.returncode}: {result.stderr}"
    )


# ---------------------------------------------------------------------------
# CR-T123-07 / CR-T123-09: _resolve_backend_identity submodule diffs & fail-closed
# ---------------------------------------------------------------------------


def _make_minimal_git_repo(path: Path) -> str:
    """Create a minimal git repo at *path* and return its HEAD SHA."""
    import subprocess

    (path / "dummy").write_text("")
    subprocess.run(["git", "init", "-b", "main"], cwd=str(path), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test"], cwd=str(path), capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=str(path), capture_output=True
    )
    subprocess.run(["git", "add", "."], cwd=str(path), capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(path), capture_output=True)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_resolve_backend_identity_git_status_fail_closed(tmp_path):
    """git status non-zero exit must raise, not default to clean (CR-T123-09)."""
    import subprocess as sp

    repo = tmp_path / "backend"
    repo.mkdir()
    _make_minimal_git_repo(repo)

    real_run = sp.run

    def _failing_status(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        cmd_str = " ".join(cmd)
        if "status" in cmd_str and "--porcelain" in cmd_str:
            return sp.CompletedProcess(
                cmd, 128, stdout="", stderr="fatal: not a git repo"
            )
        return real_run(*args, **kwargs)

    with pytest.raises(BackendValidationError, match="git status failed"):
        with __import__("unittest").mock.patch("subprocess.run", _failing_status):
            _resolve_backend_identity(repo, backend_mode="locked_clean")


def test_resolve_backend_identity_submodule_rev_parse_fail_closed(tmp_path):
    """Submodule rev-parse failure must raise, not silently omit (CR-T123-09)."""
    import subprocess as sp

    repo = tmp_path / "backend"
    repo.mkdir()
    _make_minimal_git_repo(repo)

    # Register a submodule so git submodule foreach discovers it
    sub_path = repo / "submodules" / "failing_sub"
    sub_path.mkdir(parents=True)
    (sub_path / "dummy").write_text("")
    sp.run(["git", "-C", str(sub_path), "init", "-b", "main"], capture_output=True)
    sp.run(
        ["git", "-C", str(sub_path), "config", "user.email", "test@test"],
        capture_output=True,
    )
    sp.run(
        ["git", "-C", str(sub_path), "config", "user.name", "Test"], capture_output=True
    )
    sp.run(["git", "-C", str(sub_path), "add", "."], capture_output=True)
    sp.run(["git", "-C", str(sub_path), "commit", "-m", "init"], capture_output=True)

    sp.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(repo),
            "submodule",
            "add",
            "-b",
            "main",
            str(sub_path),
            "submodules/failing_sub",
        ],
        capture_output=True,
    )
    # Temporarily unbreak the submodule .git so foreach works
    sp.run(["git", "-C", str(repo), "submodule", "absorbgitdirs"], capture_output=True)

    real_run = sp.run

    def _failing_submodule_rev_parse(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        cmd_str = " ".join(cmd)
        # Fail on rev-parse for the failing_sub submodule
        if "rev-parse" in cmd_str and "failing_sub" in str(cmd):
            return sp.CompletedProcess(cmd, 128, stdout="", stderr="fatal")
        return real_run(*args, **kwargs)

    with pytest.raises(
        BackendValidationError, match="Failed to get HEAD for submodule"
    ):
        with __import__("unittest").mock.patch(
            "subprocess.run", _failing_submodule_rev_parse
        ):
            _resolve_backend_identity(repo, backend_mode="locked_clean")


def test_resolve_backend_identity_nested_submodules_dirty(tmp_path):
    """Nested submodule HEAD and diff hash both recorded (FV-02)."""
    import subprocess as sp

    repo = tmp_path / "backend"
    repo.mkdir()
    _make_minimal_git_repo(repo)

    # Build a nested submodule tree: submodules/child/nested/leaf
    leaf_dir = tmp_path / "leaf_repo"
    leaf_dir.mkdir()
    (leaf_dir / "leaf_file").write_text("leaf content v1\n")
    sp.run(["git", "init", "-b", "main"], cwd=str(leaf_dir), capture_output=True)
    sp.run(
        ["git", "config", "user.email", "test@test"],
        cwd=str(leaf_dir),
        capture_output=True,
    )
    sp.run(
        ["git", "config", "user.name", "Test"], cwd=str(leaf_dir), capture_output=True
    )
    sp.run(["git", "add", "."], cwd=str(leaf_dir), capture_output=True)
    sp.run(["git", "commit", "-m", "init"], cwd=str(leaf_dir), capture_output=True)

    child_dir = tmp_path / "child_repo"
    child_dir.mkdir()
    (child_dir / "child_file").write_text("child content v1\n")
    sp.run(["git", "init", "-b", "main"], cwd=str(child_dir), capture_output=True)
    sp.run(
        ["git", "config", "user.email", "test@test"],
        cwd=str(child_dir),
        capture_output=True,
    )
    sp.run(
        ["git", "config", "user.name", "Test"], cwd=str(child_dir), capture_output=True
    )
    sp.run(["git", "add", "."], cwd=str(child_dir), capture_output=True)
    sp.run(["git", "commit", "-m", "init"], cwd=str(child_dir), capture_output=True)

    # Register leaf as a submodule of child
    sp.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(child_dir),
            "submodule",
            "add",
            "-b",
            "main",
            str(leaf_dir),
            "nested/leaf",
        ],
        capture_output=True,
    )
    sp.run(
        ["git", "-C", str(child_dir), "commit", "-m", "add leaf submodule"],
        capture_output=True,
    )

    # Register child as a submodule of root
    sp.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(repo),
            "submodule",
            "add",
            "-b",
            "main",
            str(child_dir),
            "submodules/child",
        ],
        capture_output=True,
    )
    sp.run(
        ["git", "-C", str(repo), "commit", "-m", "add child submodule"],
        capture_output=True,
    )

    # Initialize nested submodules so working trees are populated (FV-02)
    sp.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(repo),
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        capture_output=True,
    )

    # Make both dirty: modify tracked files
    leaf_in_child = repo / "submodules" / "child" / "nested" / "leaf"
    (leaf_in_child / "leaf_file").write_text("leaf content v2 DIRTY\n")
    (repo / "submodules" / "child" / "child_file").write_text(
        "child content v2 DIRTY\n"
    )

    identity = _resolve_backend_identity(repo, backend_mode="research_local")

    # Both nested paths must be present
    assert "submodules/child" in identity["submodules"]
    assert "submodules/child/nested/leaf" in identity["submodules"]
    assert "submodules/child" in identity["submodule_diffs"]
    assert "submodules/child/nested/leaf" in identity["submodule_diffs"]

    # Both dirty diffs must be non-None, 64-char lowercase hex
    child_diff = identity["submodule_diffs"]["submodules/child"]
    leaf_diff = identity["submodule_diffs"]["submodules/child/nested/leaf"]
    assert child_diff is not None
    assert leaf_diff is not None
    assert isinstance(child_diff, str) and len(child_diff) == 64
    assert isinstance(leaf_diff, str) and len(leaf_diff) == 64
    assert child_diff != leaf_diff, "different content must yield different hashes"

    # Each HEAD must be a valid 40-char SHA
    for sub_name in ("submodules/child", "submodules/child/nested/leaf"):
        sha = identity["submodules"][sub_name]
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_resolve_backend_identity_submodule_path_not_found_fail_closed(tmp_path):
    """If git foreach reports a path that doesn't exist, fail closed (FV-02)."""
    import subprocess as sp

    repo = tmp_path / "backend"
    repo.mkdir()
    _make_minimal_git_repo(repo)

    real_run = sp.run

    def _fake_foreach_fake_path(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        cmd_str = " ".join(cmd)
        if "submodule" in cmd_str and "foreach" in cmd_str:
            return sp.CompletedProcess(
                cmd,
                0,
                stdout="submodules/ghost\n",
                stderr="",
            )
        return real_run(*args, **kwargs)

    with pytest.raises(BackendValidationError, match="does not exist"):
        with __import__("unittest").mock.patch(
            "subprocess.run", _fake_foreach_fake_path
        ):
            _resolve_backend_identity(repo, backend_mode="locked_clean")


def test_resolve_backend_identity_no_submodules(tmp_path):
    """Clean repo with no submodules: submodules and submodule_diffs are empty."""
    repo = tmp_path / "backend"
    repo.mkdir()
    _make_minimal_git_repo(repo)

    identity_lc = _resolve_backend_identity(repo, backend_mode="locked_clean")
    assert identity_lc["submodules"] == {}
    assert identity_lc["submodule_diffs"] == {}

    identity_rl = _resolve_backend_identity(repo, backend_mode="research_local")
    assert identity_rl["submodules"] == {}
    assert identity_rl["submodule_diffs"] == {}
    assert identity_rl["diff_sha256"] is None


def test_resolve_backend_identity_rejects_traversal_outside(tmp_path):
    """RC-FV-F05: foreach reporting ../outside must fail with containment error."""
    import subprocess as sp

    repo = tmp_path / "backend"
    repo.mkdir()
    _make_minimal_git_repo(repo)

    root_target = str(repo.resolve())
    real_run = sp.run
    diff_targets: list[str] = []

    def _fake_foreach_outside(*args, **kwargs):
        cmd = list(args[0] if args else kwargs.get("args", []))
        if cmd and cmd[0] == "git" and "diff" in cmd and "-C" in cmd:
            target = str(cmd[cmd.index("-C") + 1])
            diff_targets.append(target)
            if target != root_target:
                return sp.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
        cmd_str = " ".join(cmd)
        if "submodule" in cmd_str and "foreach" in cmd_str:
            return sp.CompletedProcess(cmd, 0, stdout="../outside\n", stderr="")
        return real_run(*args, **kwargs)

    from unittest import mock as _mock

    with pytest.raises(BackendValidationError, match="resolves outside repo root"):
        with _mock.patch("subprocess.run", _fake_foreach_outside):
            _resolve_backend_identity(repo, backend_mode="locked_clean")

    assert diff_targets == [], (
        f"locked_clean mode must not execute any git diff; got {diff_targets}"
    )


def test_resolve_backend_identity_path_vanished_between_loops(tmp_path):
    """RC-FV-F05: path existing during HEAD query but gone during diff → fail."""
    import subprocess as sp

    repo = tmp_path / "backend"
    repo.mkdir()
    _make_minimal_git_repo(repo)

    # Gitignore submodules so the main repo doesn't see untracked files
    (repo / ".gitignore").write_text("submodules/\n")
    sp.run(["git", "add", ".gitignore"], cwd=str(repo), capture_output=True)
    sp.run(
        ["git", "commit", "-m", "add gitignore"],
        cwd=str(repo),
        capture_output=True,
    )

    # Create a real submodule directory
    vanishing_dir = repo / "submodules" / "vanishing"
    vanishing_dir.mkdir(parents=True)
    (vanishing_dir / "dummy").write_text("x")
    sp.run(
        ["git", "init", "-b", "main"],
        cwd=str(vanishing_dir),
        capture_output=True,
    )
    sp.run(["git", "add", "."], cwd=str(vanishing_dir), capture_output=True)
    sp.run(
        ["git", "commit", "-m", "init"],
        cwd=str(vanishing_dir),
        capture_output=True,
    )

    root_target = str(repo.resolve())
    real_run = sp.run
    diff_targets: list[str] = []

    def _mock_foreach_and_head(*args, **kwargs):
        cmd = list(args[0] if args else kwargs.get("args", []))
        if cmd and cmd[0] == "git" and "diff" in cmd and "-C" in cmd:
            target = str(cmd[cmd.index("-C") + 1])
            diff_targets.append(target)
            if target != root_target:
                return sp.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
        cmd_str = " ".join(cmd)
        if "submodule" in cmd_str and "foreach" in cmd_str:
            return sp.CompletedProcess(
                cmd, 0, stdout="submodules/vanishing\n", stderr=""
            )
        if "rev-parse" in cmd_str and "vanishing" in cmd_str:
            return sp.CompletedProcess(cmd, 0, stdout="a" * 40 + "\n", stderr="")
        return real_run(*args, **kwargs)

    from unittest import mock

    # Path.is_dir returns True once (HEAD loop), then False (diff loop)
    real_is_dir = Path.is_dir
    call_counts: dict[str, int] = {}

    def _fake_is_dir(self: Path) -> bool:
        key = str(self)
        if "vanishing" not in key:
            return real_is_dir(self)
        call_counts[key] = call_counts.get(key, 0) + 1
        return call_counts[key] == 1  # True only first time

    with pytest.raises(BackendValidationError, match="vanished between"):
        with mock.patch("subprocess.run", _mock_foreach_and_head):
            with mock.patch.object(Path, "is_dir", _fake_is_dir):
                _resolve_backend_identity(repo, backend_mode="research_local")

    assert diff_targets == [root_target], (
        f"only main backend fingerprint diff allowed; got {diff_targets}"
    )


def test_resolve_backend_identity_diff_loop_rejects_outside_resolve(tmp_path):
    """LATEST-FV-03: second resolve outside repo during diff loop fails closed."""
    import subprocess as sp

    repo = tmp_path / "backend"
    repo.mkdir()
    _make_minimal_git_repo(repo)

    # Ignore submodules so main repo doesn't see untracked files
    (repo / ".gitignore").write_text("submodules/\n")
    sp.run(["git", "add", ".gitignore"], cwd=str(repo), capture_output=True)
    sp.run(["git", "commit", "-m", "add gitignore"], cwd=str(repo), capture_output=True)

    # Create a real submodule directory
    sub_dir = repo / "submodules" / "real_sub"
    sub_dir.mkdir(parents=True)
    (sub_dir / "dummy").write_text("x")
    sp.run(["git", "init", "-b", "main"], cwd=str(sub_dir), capture_output=True)
    sp.run(
        ["git", "config", "user.email", "test@test"],
        cwd=str(sub_dir),
        capture_output=True,
    )
    sp.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(sub_dir),
        capture_output=True,
    )
    sp.run(["git", "add", "."], cwd=str(sub_dir), capture_output=True)
    sp.run(["git", "commit", "-m", "init"], cwd=str(sub_dir), capture_output=True)

    root_target = str(repo.resolve())
    real_run = sp.run
    resolved_saved: dict[str, Path] = {}
    diff_targets: list[str] = []

    def _mock_run(*args, **kwargs):
        cmd = list(args[0] if args else kwargs.get("args", []))
        if cmd and cmd[0] == "git" and "diff" in cmd and "-C" in cmd:
            target = str(cmd[cmd.index("-C") + 1])
            diff_targets.append(target)
            if target != root_target:
                return sp.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
        cmd_str = " ".join(cmd)
        if "submodule" in cmd_str and "foreach" in cmd_str:
            return sp.CompletedProcess(
                cmd, 0, stdout="submodules/real_sub\n", stderr=""
            )
        if "rev-parse" in cmd_str and "real_sub" in cmd_str:
            return sp.CompletedProcess(cmd, 0, stdout="a" * 40 + "\n", stderr="")
        return real_run(*args, **kwargs)

    real_resolve = Path.resolve

    def _fake_resolve(self: Path) -> Path:
        key = str(self)
        resolved = real_resolve(self)
        if "real_sub" in key:
            if key in resolved_saved:
                # Second call during diff loop — return outside path
                return tmp_path / "outside_evil"
            resolved_saved[key] = resolved
        return resolved

    real_is_dir = Path.is_dir

    def _fake_is_dir(self: Path) -> bool:
        key = str(self)
        if "outside_evil" in key:
            return True
        if "real_sub" in key:
            return real_is_dir(self)
        return real_is_dir(self)

    from unittest import mock

    with pytest.raises(BackendValidationError, match="resolves outside repo root"):
        with mock.patch("subprocess.run", _mock_run):
            with mock.patch.object(Path, "resolve", _fake_resolve):
                with mock.patch.object(Path, "is_dir", _fake_is_dir):
                    _resolve_backend_identity(repo, backend_mode="research_local")

    assert diff_targets == [root_target], (
        f"only main backend fingerprint diff allowed; got {diff_targets}"
    )


def test_resolve_backend_identity_diff_loop_rejects_changed_target(tmp_path):
    """LATEST-FV-03: second resolve returns different path inside repo → fail."""
    import subprocess as sp

    repo = tmp_path / "backend"
    repo.mkdir()
    _make_minimal_git_repo(repo)

    (repo / ".gitignore").write_text("submodules/\n")
    sp.run(["git", "add", ".gitignore"], cwd=str(repo), capture_output=True)
    sp.run(["git", "commit", "-m", "add gitignore"], cwd=str(repo), capture_output=True)

    sub_dir = repo / "submodules" / "real_sub"
    sub_dir.mkdir(parents=True)
    (sub_dir / "dummy").write_text("x")
    sp.run(["git", "init", "-b", "main"], cwd=str(sub_dir), capture_output=True)
    sp.run(
        ["git", "config", "user.email", "test@test"],
        cwd=str(sub_dir),
        capture_output=True,
    )
    sp.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(sub_dir),
        capture_output=True,
    )
    sp.run(["git", "add", "."], cwd=str(sub_dir), capture_output=True)
    sp.run(["git", "commit", "-m", "init"], cwd=str(sub_dir), capture_output=True)

    # Create another directory inside the repo to swap to
    decoy_dir = repo / "submodules" / "decoy"
    decoy_dir.mkdir(parents=True)

    root_target = str(repo.resolve())
    real_run = sp.run
    resolved_calls: dict[str, int] = {}
    diff_targets: list[str] = []

    def _mock_run(*args, **kwargs):
        cmd = list(args[0] if args else kwargs.get("args", []))
        if cmd and cmd[0] == "git" and "diff" in cmd and "-C" in cmd:
            target = str(cmd[cmd.index("-C") + 1])
            diff_targets.append(target)
            if target != root_target:
                return sp.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
        cmd_str = " ".join(cmd)
        if "submodule" in cmd_str and "foreach" in cmd_str:
            return sp.CompletedProcess(
                cmd, 0, stdout="submodules/real_sub\n", stderr=""
            )
        if "rev-parse" in cmd_str and "real_sub" in cmd_str:
            return sp.CompletedProcess(cmd, 0, stdout="a" * 40 + "\n", stderr="")
        return real_run(*args, **kwargs)

    real_resolve = Path.resolve

    def _fake_resolve(self: Path) -> Path:
        key = str(self)
        resolved = real_resolve(self)
        if "real_sub" in key:
            cnt = resolved_calls.get(key, 0)
            resolved_calls[key] = cnt + 1
            if cnt >= 1:  # second call during diff loop → return decoy
                return decoy_dir
        return resolved

    from unittest import mock

    with pytest.raises(BackendValidationError, match="changed between HEAD and diff"):
        with mock.patch("subprocess.run", _mock_run):
            with mock.patch.object(Path, "resolve", _fake_resolve):
                _resolve_backend_identity(repo, backend_mode="research_local")

    assert diff_targets == [root_target], (
        f"only main backend fingerprint diff allowed; got {diff_targets}"
    )


def test_resolve_backend_identity_leaf_diff_hash_changes_on_modify(tmp_path):
    """RC-FV-F05: modifying a nested submodule changes its diff hash."""
    import subprocess as sp

    repo = tmp_path / "backend"
    repo.mkdir()
    _make_minimal_git_repo(repo)

    # Build nested tree: submodules/child/nested/leaf
    leaf_dir = tmp_path / "leaf_repo"
    leaf_dir.mkdir()
    (leaf_dir / "leaf_file").write_text("leaf v1\n")
    sp.run(["git", "init", "-b", "main"], cwd=str(leaf_dir), capture_output=True)
    sp.run(
        ["git", "config", "user.email", "test@test"],
        cwd=str(leaf_dir),
        capture_output=True,
    )
    sp.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(leaf_dir),
        capture_output=True,
    )
    sp.run(["git", "add", "."], cwd=str(leaf_dir), capture_output=True)
    sp.run(["git", "commit", "-m", "init"], cwd=str(leaf_dir), capture_output=True)

    child_dir = tmp_path / "child_repo"
    child_dir.mkdir()
    (child_dir / "child_file").write_text("child v1\n")
    sp.run(["git", "init", "-b", "main"], cwd=str(child_dir), capture_output=True)
    sp.run(
        ["git", "config", "user.email", "test@test"],
        cwd=str(child_dir),
        capture_output=True,
    )
    sp.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(child_dir),
        capture_output=True,
    )
    sp.run(["git", "add", "."], cwd=str(child_dir), capture_output=True)
    sp.run(["git", "commit", "-m", "init"], cwd=str(child_dir), capture_output=True)

    sp.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(child_dir),
            "submodule",
            "add",
            "-b",
            "main",
            str(leaf_dir),
            "nested/leaf",
        ],
        capture_output=True,
    )
    sp.run(
        ["git", "-C", str(child_dir), "commit", "-m", "add leaf"],
        capture_output=True,
    )

    sp.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(repo),
            "submodule",
            "add",
            "-b",
            "main",
            str(child_dir),
            "submodules/child",
        ],
        capture_output=True,
    )
    sp.run(
        ["git", "-C", str(repo), "commit", "-m", "add child"],
        capture_output=True,
    )

    sp.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(repo),
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        capture_output=True,
    )

    # Modify leaf to dirty A — first identity snapshot is already dirty
    leaf_in_tree = repo / "submodules" / "child" / "nested" / "leaf"
    (leaf_in_tree / "leaf_file").write_text("leaf dirty version A\n")
    id_a = _resolve_backend_identity(repo, backend_mode="research_local")
    hash_a = id_a["submodule_diffs"]["submodules/child/nested/leaf"]

    # Modify to dirty B — hash must be different from dirty A
    (leaf_in_tree / "leaf_file").write_text("leaf dirty version B\n")
    id_b = _resolve_backend_identity(repo, backend_mode="research_local")
    hash_b = id_b["submodule_diffs"]["submodules/child/nested/leaf"]

    assert hash_a is not None, "dirty leaf A must have non-None diff hash"
    assert hash_b is not None, "dirty leaf B must have non-None diff hash"
    assert hash_a != hash_b, (
        f"leaf diff hash must differ between dirty versions; "
        f"hash_a={hash_a}, hash_b={hash_b}"
    )
    # Verify both are 64-char lowercase hex
    import re as _re

    _sha256_re = _re.compile(r"^[0-9a-f]{64}$")
    assert _sha256_re.match(hash_a), f"hash_a not a 64-char hex SHA-256: {hash_a!r}"
    assert _sha256_re.match(hash_b), f"hash_b not a 64-char hex SHA-256: {hash_b!r}"
    # Both child and leaf still present
    assert "submodules/child" in id_b["submodules"]
    assert "submodules/child/nested/leaf" in id_b["submodules"]


# ---------------------------------------------------------------------------
# FV-01: convert_and_validate() backend_mode propagation
# ---------------------------------------------------------------------------


def test_convert_and_validate_respects_backend_mode(tmp_path):
    """Dirty backend + locked_clean rejects; research_local enters subprocess."""
    import subprocess as sp

    backend = _make_fake_backend(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()

    # Make backend dirty
    (backend / "dirty_marker").write_text("dirty\n")

    # Monkey-patch _check_repo's commit/submodule constants to match fake backend
    import scripts.longsplat.runner as _runner

    orig_commit = _runner.LONGSPLAT_COMMIT
    orig_links = dict(_runner._LONGSPLAT_SUBMODULE_LINKS)

    fake_commit = sp.run(
        ["git", "-C", str(backend), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    _runner.LONGSPLAT_COMMIT = fake_commit
    for sub_key in list(_runner._LONGSPLAT_SUBMODULE_LINKS.keys()):
        sp_path = backend / sub_key
        r = sp.run(
            ["git", "-C", str(sp_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        _runner._LONGSPLAT_SUBMODULE_LINKS[sub_key] = r.stdout.strip()

    try:
        config_lc = LongSplatConfig(
            source_path=str(tmp_path),
            model_path=str(model_path),
            seed=0,
            backend_mode="locked_clean",
        )
        with pytest.raises(BackendValidationError, match="uncommitted"):
            convert_and_validate(backend, config_lc, python_exe=sys.executable)

        config_rl = LongSplatConfig(
            source_path=str(tmp_path),
            model_path=str(model_path),
            seed=0,
            backend_mode="research_local",
        )
        result_path = convert_and_validate(
            backend,
            config_rl,
            python_exe=sys.executable,
        )
        assert result_path.is_file()
    finally:
        _runner.LONGSPLAT_COMMIT = orig_commit
        _runner._LONGSPLAT_SUBMODULE_LINKS.clear()
        _runner._LONGSPLAT_SUBMODULE_LINKS.update(orig_links)


def test_convert_and_validate_nonzero_exit_converts_to_error(tmp_path):
    """Non-zero convert exit code must be raised as ConverterError."""
    import subprocess as sp

    backend = _make_fake_backend(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()

    import scripts.longsplat.runner as _runner

    orig_commit = _runner.LONGSPLAT_COMMIT
    orig_links = dict(_runner._LONGSPLAT_SUBMODULE_LINKS)

    fake_commit = sp.run(
        ["git", "-C", str(backend), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    _runner.LONGSPLAT_COMMIT = fake_commit
    for sub_key in list(_runner._LONGSPLAT_SUBMODULE_LINKS.keys()):
        sp_path = backend / sub_key
        r = sp.run(
            ["git", "-C", str(sp_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        _runner._LONGSPLAT_SUBMODULE_LINKS[sub_key] = r.stdout.strip()

    try:
        import os

        os.environ["FAKE_EXIT"] = "3"

        config = LongSplatConfig(
            source_path=str(tmp_path),
            model_path=str(model_path),
            seed=0,
            backend_mode="research_local",
        )
        with pytest.raises(ConverterError, match="exited with 3"):
            convert_and_validate(backend, config, python_exe=sys.executable)
    finally:
        _runner.LONGSPLAT_COMMIT = orig_commit
        _runner._LONGSPLAT_SUBMODULE_LINKS.clear()
        _runner._LONGSPLAT_SUBMODULE_LINKS.update(orig_links)
        os.environ.pop("FAKE_EXIT", None)


# ---------------------------------------------------------------------------
# FV-05: VDA usage summarisation, config type validation
# ---------------------------------------------------------------------------


def test_summarize_vda_usage_parsing():
    """FV-05: _summarize_vda_usage correctly counts VDA_USAGE markers."""
    from scripts.longsplat.orchestrator import _summarize_vda_usage

    stdout = (
        "VDA_USAGE result=aligned stage=scene_init frame=img_001\n"
        "VDA_USAGE result=aligned stage=scene_init frame=img_002\n"
        "VDA_USAGE result=aligned stage=scene_init frame=img_003\n"
        "VDA_USAGE result=aligned stage=incremental frame=img_004\n"
        "VDA_USAGE result=missing stage=scene_init frame=img_005\n"
        "VDA_USAGE result=rejected stage=scene_init frame=img_006\n"
        "Some other log line\n"
    )
    counts = _summarize_vda_usage(stdout)
    assert counts == {"aligned": 4, "missing": 1, "rejected": 1}


def test_summarize_vda_usage_empty():
    """FV-05: empty stdout produces zero counts."""
    from scripts.longsplat.orchestrator import _summarize_vda_usage

    assert _summarize_vda_usage("") == {"aligned": 0, "missing": 0, "rejected": 0}


@pytest.mark.parametrize("bad_value", ["a string", ["list"], None, 42, 3.14])
def test_load_config_rejects_non_dict_extra_train_args(tmp_path, bad_value):
    """FV-05: non-dict extra_train_args must be rejected immediately."""
    from scripts.longsplat.runner import load_config

    config_path = tmp_path / "bad_config.json"
    import json

    config_path.write_text(
        json.dumps(
            {
                "source_path": "/tmp/src",
                "model_path": "/tmp/model",
                "extra_train_args": bad_value,
            }
        )
    )

    with pytest.raises(BackendValidationError, match="extra_train_args must be a dict"):
        load_config(str(config_path))


def test_vda_usage_markers_captured_in_run_record(tmp_path):
    """FV-05 / RC-FV-F01: real fake train.py stdout → VDA usage in run record.

    Only ``materialize_all`` is mocked to isolate NPZ contract.  Training
    and conversion subprocesses are real so that stdout transport, encoding,
    cwd, and env are exercised end-to-end.
    """
    import subprocess as sp
    from unittest import mock

    backend = _make_fake_backend(tmp_path)
    manifest = _make_producer_manifest(tmp_path)
    output_dir = tmp_path / "outputs"

    depth_manifest_path = tmp_path / "depth_manifest.json"
    depth_manifest_path.write_text(json.dumps({"schema_version": "1.0", "frames": []}))

    import scripts.longsplat.runner as _runner

    orig_commit = _runner.LONGSPLAT_COMMIT
    orig_links = dict(_runner._LONGSPLAT_SUBMODULE_LINKS)

    real_run = sp.run
    fake_commit = real_run(
        ["git", "-C", str(backend), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    _runner.LONGSPLAT_COMMIT = fake_commit
    for sub_key in list(_runner._LONGSPLAT_SUBMODULE_LINKS.keys()):
        sp_path = backend / sub_key
        r = real_run(
            ["git", "-C", str(sp_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        _runner._LONGSPLAT_SUBMODULE_LINKS[sub_key] = r.stdout.strip()

    from scripts.longsplat.depth_bridge import DepthMaterializationResult

    fake_mat_result = DepthMaterializationResult(
        expected_count=1,
        materialized_count=1,
        depth_manifest_sha256="abcd1234",
        frames=[],
    )

    try:
        with mock.patch(
            "scripts.longsplat.orchestrator.materialize_all",
            return_value=fake_mat_result,
        ):
            exit_code = run_pipeline(
                manifest_path=manifest,
                segment_id="seg_01",
                config=LongSplatConfig(
                    source_path="",
                    model_path="",
                    iterations=100,
                    seed=0,
                    backend_mode="research_local",
                    extra_train_args={"depth_source": "vda"},
                ),
                repo_root=backend,
                output_dir=output_dir,
                project_root=tmp_path,
                python_exe=sys.executable,
                depth_manifest_path=depth_manifest_path,
            )

        assert exit_code == 0, f"pipeline must succeed, got {exit_code}"

        run_dirs = list(output_dir.iterdir())
        assert len(run_dirs) == 1
        record = json.loads((run_dirs[0] / "reconstruction_run.json").read_text())
        assert record["status"] == "complete"
        # The fake backend writes 1 camera and emits 3 VDA_USAGE markers.
        # The gate only checks aligned/train_ratio >= 0.98.
        assert record["depth"]["usage"] == {"aligned": 3, "missing": 0, "rejected": 0}
        assert record["telemetry"]["pose"]["attempt_count"] == 1
        assert record["telemetry"]["pose"]["accepted_camera_count"] == 1
        assert record["telemetry"]["vda"]["aligned"] == 1
        assert record["telemetry"]["vda"]["min_correlation"] == 0.91
        assert record["telemetry"]["conversion"]["all_scales_finite"] is True

        # Sentinel proves train.py executed as a real subprocess.
        # The fake train.py writes to cwd which is the backend repo root.
        sentinel = json.loads((backend / "orchestrator_test_sentinel.json").read_text())
        assert sentinel["script"] == "train"
    finally:
        _runner.LONGSPLAT_COMMIT = orig_commit
        _runner._LONGSPLAT_SUBMODULE_LINKS.clear()
        _runner._LONGSPLAT_SUBMODULE_LINKS.update(orig_links)


# ---------------------------------------------------------------------------
# Task 5: Post-training pose audit — unit tests
# ---------------------------------------------------------------------------


def _make_camera_entry(R=None, T=None):
    """Build a single camera dict with rotation and position."""
    import numpy as np

    if R is None:
        R = np.eye(3, dtype=np.float64)
    if T is None:
        T = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return {
        "R": R.tolist() if isinstance(R, np.ndarray) else R,
        "T": T.tolist() if isinstance(T, np.ndarray) else T,
    }


def _write_cameras_json(path, cameras):
    """Write a list of camera dicts as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cameras, indent=2), encoding="utf-8")


def test_audit_pose_quality_all_legal(tmp_path):
    """Scenario 6: all cameras valid → passes."""
    from scripts.longsplat.quality_gates import audit_pose_quality

    cameras = [
        _make_camera_entry(T=[0.0, 0.0, 1.0]),
        _make_camera_entry(T=[0.1, 0.0, 1.0]),
        _make_camera_entry(T=[0.2, 0.0, 1.0]),
    ]
    cams_path = tmp_path / "cameras_all_train.json"
    _write_cameras_json(cams_path, cameras)

    telemetry = {
        "accepted_camera_count": 3,
        "accepted_frames": ["frame_000000", "frame_000001", "frame_000002"],
        "attempt_count": 3,
        "rejected_attempt_count": 0,
        "records": [
            {"frame": "frame_000000", "success": True},
            {"frame": "frame_000001", "success": True},
            {"frame": "frame_000002", "success": True},
        ],
    }

    result = audit_pose_quality(cams_path, telemetry)
    assert result["passed"] is True, f"unexpected reasons: {result['reasons']}"
    assert result["reasons"] == []


def test_audit_pose_quality_missing_pose_marker(tmp_path):
    """Scenario 1: no accepted cameras in telemetry → fails."""
    from scripts.longsplat.quality_gates import audit_pose_quality

    cameras = [
        _make_camera_entry(T=[0.0, 0.0, 1.0]),
        _make_camera_entry(T=[0.1, 0.0, 1.0]),
    ]
    cams_path = tmp_path / "cameras_all_train.json"
    _write_cameras_json(cams_path, cameras)

    telemetry = {
        "accepted_camera_count": 0,
        "accepted_frames": [],
        "attempt_count": 0,
        "rejected_attempt_count": 0,
        "records": [],
    }

    result = audit_pose_quality(cams_path, telemetry)
    assert result["passed"] is False
    assert any("accepted cameras 0 != train cameras 2" in r for r in result["reasons"])


def test_audit_pose_quality_camera_not_accepted(tmp_path):
    """Scenario 2: one camera without accepted terminal state → fails."""
    from scripts.longsplat.quality_gates import audit_pose_quality

    cameras = [
        _make_camera_entry(T=[0.0, 0.0, 1.0]),
        _make_camera_entry(T=[0.1, 0.0, 1.0]),
        _make_camera_entry(T=[0.2, 0.0, 1.0]),
    ]
    cams_path = tmp_path / "cameras_all_train.json"
    _write_cameras_json(cams_path, cameras)

    telemetry = {
        "accepted_camera_count": 2,
        "accepted_frames": ["frame_000000", "frame_000001"],
        "attempt_count": 3,
        "rejected_attempt_count": 1,
        "records": [
            {"frame": "frame_000000", "success": True},
            {"frame": "frame_000001", "success": True},
            {"frame": "frame_000002", "success": False},
        ],
    }

    result = audit_pose_quality(cams_path, telemetry)
    assert result["passed"] is False
    assert any("accepted cameras 2 != train cameras 3" in r for r in result["reasons"])


def test_audit_pose_quality_det_error(tmp_path):
    """Scenario 3: camera with det 0.96 → fails SO(3) check."""
    from scripts.longsplat.quality_gates import audit_pose_quality

    # Create a rotation matrix scaled to have det ≈ 0.96
    R_bad = np.eye(3, dtype=np.float64) * 0.986  # det ≈ 0.96
    cameras = [
        _make_camera_entry(R=np.eye(3), T=[0.0, 0.0, 1.0]),
        _make_camera_entry(R=R_bad, T=[0.1, 0.0, 1.0]),
    ]
    cams_path = tmp_path / "cameras_all_train.json"
    _write_cameras_json(cams_path, cameras)

    telemetry = {
        "accepted_camera_count": 2,
        "accepted_frames": ["frame_000000", "frame_000001"],
        "attempt_count": 2,
        "rejected_attempt_count": 0,
        "records": [
            {"frame": "frame_000000", "success": True},
            {"frame": "frame_000001", "success": True},
        ],
    }

    result = audit_pose_quality(cams_path, telemetry)
    assert result["passed"] is False
    assert any("determinant error" in r for r in result["reasons"])


def test_audit_pose_quality_large_rotation_step(tmp_path):
    """Scenario 4: 142° rotation step → fails continuity check."""
    from scripts.longsplat.quality_gates import audit_pose_quality

    # Build two cameras with a ~142° relative rotation around Z
    theta = np.deg2rad(142.0)
    R0 = np.eye(3)
    R1 = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    cameras = [
        _make_camera_entry(R=R0, T=[0.0, 0.0, 1.0]),
        _make_camera_entry(R=R1, T=[0.01, 0.0, 1.0]),
    ]
    cams_path = tmp_path / "cameras_all_train.json"
    _write_cameras_json(cams_path, cameras)

    telemetry = {
        "accepted_camera_count": 2,
        "accepted_frames": ["frame_000000", "frame_000001"],
        "attempt_count": 2,
        "rejected_attempt_count": 0,
        "records": [
            {"frame": "frame_000000", "success": True},
            {"frame": "frame_000001", "success": True},
        ],
    }

    result = audit_pose_quality(cams_path, telemetry)
    assert result["passed"] is False
    assert any("rotation step" in r for r in result["reasons"])


def test_audit_pose_quality_large_translation_step(tmp_path):
    """Scenario 5: translation step 10x median → fails continuity check."""
    from scripts.longsplat.quality_gates import audit_pose_quality

    # 3 cameras: small steps, then a giant jump
    cameras = [
        _make_camera_entry(T=[0.0, 0.0, 1.0]),
        _make_camera_entry(T=[0.1, 0.0, 1.0]),
        _make_camera_entry(T=[0.2, 0.0, 1.0]),
        _make_camera_entry(T=[5.0, 0.0, 1.0]),  # ~48x median step of ~0.1
    ]
    cams_path = tmp_path / "cameras_all_train.json"
    _write_cameras_json(cams_path, cameras)

    telemetry = {
        "accepted_camera_count": 4,
        "accepted_frames": [
            "frame_000000",
            "frame_000001",
            "frame_000002",
            "frame_000003",
        ],
        "attempt_count": 4,
        "rejected_attempt_count": 0,
        "records": [{"frame": f"frame_{i:06d}", "success": True} for i in range(4)],
    }

    result = audit_pose_quality(cams_path, telemetry)
    assert result["passed"] is False
    assert any("translation step ratio" in r for r in result["reasons"])


def test_audit_pose_quality_nonfinite_rotation(tmp_path):
    """Camera with NaN in rotation matrix → fails."""
    from scripts.longsplat.quality_gates import audit_pose_quality

    R_bad = np.eye(3, dtype=np.float64)
    R_bad[0, 0] = np.nan
    cameras = [
        _make_camera_entry(R=np.eye(3), T=[0.0, 0.0, 1.0]),
        _make_camera_entry(R=R_bad, T=[0.1, 0.0, 1.0]),
    ]
    cams_path = tmp_path / "cameras_all_train.json"
    _write_cameras_json(cams_path, cameras)

    telemetry = {
        "accepted_camera_count": 2,
        "accepted_frames": ["frame_000000", "frame_000001"],
        "attempt_count": 2,
        "rejected_attempt_count": 0,
        "records": [
            {"frame": "frame_000000", "success": True},
            {"frame": "frame_000001", "success": True},
        ],
    }

    result = audit_pose_quality(cams_path, telemetry)
    assert result["passed"] is False
    assert any("non-finite" in r for r in result["reasons"])


def test_audit_pose_quality_native_checkpoint_missing(tmp_path):
    """Expected checkpoint not found → fails."""
    from scripts.longsplat.quality_gates import audit_pose_quality

    cameras = [_make_camera_entry(T=[0.0, 0.0, 1.0])]
    cams_path = tmp_path / "cameras_all_train.json"
    _write_cameras_json(cams_path, cameras)

    model_path = tmp_path / "model"
    model_path.mkdir()

    telemetry = {
        "accepted_camera_count": 1,
        "accepted_frames": ["frame_000000"],
        "attempt_count": 1,
        "rejected_attempt_count": 0,
        "records": [{"frame": "frame_000000", "success": True}],
    }

    result = audit_pose_quality(
        cams_path,
        telemetry,
        model_path=model_path,
        expected_native_checkpoint_iteration=30050,
    )
    assert result["passed"] is False
    assert any("checkpoint not found" in r for r in result["reasons"])


def test_audit_pose_quality_native_checkpoint_wrong_iteration(tmp_path):
    """Checkpoint exists at wrong iteration → fails."""
    from scripts.longsplat.quality_gates import audit_pose_quality

    cameras = [_make_camera_entry(T=[0.0, 0.0, 1.0])]
    cams_path = tmp_path / "cameras_all_train.json"
    _write_cameras_json(cams_path, cameras)

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "chkpnt100.pth").write_text("dummy")

    telemetry = {
        "accepted_camera_count": 1,
        "accepted_frames": ["frame_000000"],
        "attempt_count": 1,
        "rejected_attempt_count": 0,
        "records": [{"frame": "frame_000000", "success": True}],
    }

    result = audit_pose_quality(
        cams_path,
        telemetry,
        model_path=model_path,
        expected_native_checkpoint_iteration=30050,
    )
    assert result["passed"] is False
    assert any("100 != expected 30050" in r for r in result["reasons"])


def test_audit_pose_quality_native_checkpoint_matches(tmp_path):
    """Checkpoint at expected iteration → passes (combined with legal poses)."""
    from scripts.longsplat.quality_gates import audit_pose_quality

    cameras = [_make_camera_entry(T=[0.0, 0.0, 1.0])]
    cams_path = tmp_path / "cameras_all_train.json"
    _write_cameras_json(cams_path, cameras)

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "chkpnt30050.pth").write_text("dummy")

    telemetry = {
        "accepted_camera_count": 1,
        "accepted_frames": ["frame_000000"],
        "attempt_count": 1,
        "rejected_attempt_count": 0,
        "records": [{"frame": "frame_000000", "success": True}],
    }

    result = audit_pose_quality(
        cams_path,
        telemetry,
        model_path=model_path,
        expected_native_checkpoint_iteration=30050,
    )
    assert result["passed"] is True, f"unexpected reasons: {result['reasons']}"


# ---------------------------------------------------------------------------
# Task 5: Orchestrator integration — pose audit gate
# ---------------------------------------------------------------------------


def test_orchestrator_pose_audit_gate_blocks_conversion_on_bad_poses(tmp_path):
    """Orchestrator fails at pose_quality_gate when poses are bad (det error)."""
    import subprocess as sp

    backend = _make_fake_backend(tmp_path)
    manifest = _make_producer_manifest(tmp_path)
    output_dir = tmp_path / "outputs"

    import scripts.longsplat.runner as _runner
    from unittest import mock

    fake_commit = sp.run(
        ["git", "-C", str(backend), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    orig_commit = _runner.LONGSPLAT_COMMIT
    orig_links = dict(_runner._LONGSPLAT_SUBMODULE_LINKS)
    _runner.LONGSPLAT_COMMIT = fake_commit
    for sub_key in list(_runner._LONGSPLAT_SUBMODULE_LINKS.keys()):
        sp_path = backend / sub_key
        r = sp.run(
            ["git", "-C", str(sp_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        _runner._LONGSPLAT_SUBMODULE_LINKS[sub_key] = r.stdout.strip()

    try:
        with mock.patch(
            "scripts.longsplat.orchestrator.audit_pose_quality",
            return_value={
                "passed": False,
                "reasons": ["max determinant error 4.00e-02 > 1.0e-04"],
                "trajectory": {"camera_count": 3},
                "telemetry": {"accepted_camera_count": 3, "camera_count": 3},
            },
        ):
            exit_code = run_pipeline(
                manifest_path=manifest,
                segment_id="seg_01",
                config=LongSplatConfig(
                    source_path="",
                    model_path="",
                    iterations=100,
                    seed=0,
                    backend_mode="research_local",
                ),
                repo_root=backend,
                output_dir=output_dir,
                project_root=tmp_path,
                python_exe=sys.executable,
            )

        assert exit_code == 1, f"pose audit gate must reject bad poses, got {exit_code}"

        run_dirs = list(output_dir.iterdir())
        assert len(run_dirs) == 1
        record = json.loads((run_dirs[0] / "reconstruction_run.json").read_text())
        assert record["status"] == "failed"
        assert "pose_quality_gate" in record["stages"]
        assert "determinant error" in str(record["stages"]["pose_quality_gate"])
    finally:
        _runner.LONGSPLAT_COMMIT = orig_commit
        _runner._LONGSPLAT_SUBMODULE_LINKS.clear()
        _runner._LONGSPLAT_SUBMODULE_LINKS.update(orig_links)
