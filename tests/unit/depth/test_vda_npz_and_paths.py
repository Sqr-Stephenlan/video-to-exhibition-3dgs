from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(ROOT))

from scripts.depth.backend_vda import (
    build_vda_command,
    doctor_backend,
    ensure_vda_matplotlib_compat,
    find_vda_depths_npz,
    load_vda_depths_array,
    sanitize_command_for_record,
    split_vda_depths_to_frame_files,
    stage_checkpoint_for_vda,
)
from scripts.depth.config import resolve_repo_path, to_repo_relative, validate_frame_id
from scripts.depth.run_depth_prior import build_ffmpeg_concat_command


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


def test_real_command_builder_is_sanitized_for_run_record(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    repo_dir = root / "third_party" / "Video-Depth-Anything"
    temp_dir = tmp_path / "depth_prior_abc"
    repo_dir.mkdir(parents=True)
    temp_dir.mkdir()
    command = build_vda_command(
        repo_dir=repo_dir,
        input_video=temp_dir / "input.mp4",
        output_dir=temp_dir / "vda_out",
        backend={
            "encoder": "vitb",
            "depth_type": "relative",
            "input_size": 518,
            "max_res": 1280,
        },
        python_exe=str(root / ".venv" / "Scripts" / "python.exe"),
    )
    sanitized = sanitize_command_for_record(
        root=root,
        command=command,
        temp_dir=temp_dir,
    )
    assert sanitized[0] == "<project>/.venv/Scripts/python.exe"
    assert sanitized[1] == "<project>/third_party/Video-Depth-Anything/run.py"
    assert "<temp>/input.mp4" in sanitized
    assert "<temp>/vda_out" in sanitized
    assert str(tmp_path) not in " ".join(sanitized)
    assert not any(Path(arg).is_absolute() for arg in sanitized)


def test_custom_checkpoint_is_staged_to_vda_expected_path(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    repo_dir = root / "third_party" / "Video-Depth-Anything"
    custom = root / "weights" / "custom_vitb.pth"
    repo_dir.mkdir(parents=True)
    custom.parent.mkdir(parents=True)
    custom.write_bytes(b"custom-weights")
    backend = {
        "repo_dir": "third_party/Video-Depth-Anything",
        "checkpoint": "weights/custom_vitb.pth",
        "encoder": "vitb",
        "depth_type": "relative",
    }
    target = repo_dir / "checkpoints" / "video_depth_anything_vitb.pth"
    with stage_checkpoint_for_vda(root, backend) as metadata:
        assert target.read_bytes() == b"custom-weights"
        assert metadata["checkpoint_source"] == "weights/custom_vitb.pth"
        assert (
            metadata["checkpoint_vda_path"]
            == "third_party/Video-Depth-Anything/checkpoints/video_depth_anything_vitb.pth"
        )
        assert metadata["checkpoint_staged"] is True
    assert not target.exists()


def test_custom_checkpoint_restore_existing_target_after_failure(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    repo_dir = root / "third_party" / "Video-Depth-Anything"
    custom = root / "weights" / "custom_vitb.pth"
    target = repo_dir / "checkpoints" / "video_depth_anything_vitb.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    custom.parent.mkdir(parents=True)
    custom.write_bytes(b"custom-weights")
    target.write_bytes(b"official-weights")
    backend = {
        "repo_dir": "third_party/Video-Depth-Anything",
        "checkpoint": "weights/custom_vitb.pth",
        "encoder": "vitb",
        "depth_type": "relative",
    }
    with pytest.raises(RuntimeError, match="boom"):
        with stage_checkpoint_for_vda(root, backend):
            assert target.read_bytes() == b"custom-weights"
            raise RuntimeError("boom")
    assert target.read_bytes() == b"official-weights"


def test_backend_repo_and_checkpoint_reject_escape(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        with stage_checkpoint_for_vda(
            root,
            {
                "repo_dir": "../backend",
                "checkpoint": "",
                "encoder": "vitb",
                "depth_type": "relative",
            },
        ):
            pass
    backend = root / "third_party" / "Video-Depth-Anything"
    backend.mkdir(parents=True)
    with pytest.raises(ValueError, match="escapes"):
        with stage_checkpoint_for_vda(
            root,
            {
                "repo_dir": "third_party/Video-Depth-Anything",
                "checkpoint": "../outside.pth",
                "encoder": "vitb",
                "depth_type": "relative",
            },
        ):
            pass


def test_doctor_requires_git_head_and_reports_custom_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    repo_dir = root / "third_party" / "Video-Depth-Anything"
    checkpoint = root / "weights" / "custom_vitb.pth"
    repo_dir.mkdir(parents=True)
    (repo_dir / "run.py").write_text("print('ok')\n", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"custom")
    report = doctor_backend(
        root,
        {
            "backend": {
                "repo_dir": "third_party/Video-Depth-Anything",
                "commit": "4f5ae23172ba60fd7bc11ef671cca678842c7072",
                "checkpoint": "weights/custom_vitb.pth",
                "allow_custom_checkpoint": False,
                "encoder": "vitb",
                "depth_type": "relative",
            }
        },
    )
    assert any("could not be resolved" in issue for issue in report["issues"])
    assert any("allow_custom_checkpoint" in issue for issue in report["issues"])


def test_doctor_allows_custom_checkpoint_with_explicit_opt_in(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    repo_dir = root / "third_party" / "Video-Depth-Anything"
    checkpoint = root / "weights" / "custom_vitb.pth"
    utils = repo_dir / "utils"
    utils.mkdir(parents=True)
    (repo_dir / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_dir / ".git").mkdir()
    (utils / "dc_utils.py").write_text(
        'colormap = np.array(cm.get_cmap("inferno").colors)\n',
        encoding="utf-8",
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"custom")
    report = doctor_backend(
        root,
        {
            "backend": {
                "repo_dir": "third_party/Video-Depth-Anything",
                "commit": "4f5ae23172ba60fd7bc11ef671cca678842c7072",
                "checkpoint": "weights/custom_vitb.pth",
                "allow_custom_checkpoint": True,
                "encoder": "vitb",
                "depth_type": "relative",
            }
        },
    )
    assert report.get("allow_custom_checkpoint") is True
    assert not any("allow_custom_checkpoint" in issue for issue in report["issues"])
    assert any("non-default backend.checkpoint" in note.lower() for note in report.get("notes", []))


def test_doctor_rejects_default_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    repo_dir = root / "third_party" / "Video-Depth-Anything"
    checkpoint = repo_dir / "checkpoints" / "video_depth_anything_vitb.pth"
    repo_dir.mkdir(parents=True)
    (repo_dir / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_dir / ".git").mkdir()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"wrong-default")
    report = doctor_backend(
        root,
        {
            "backend": {
                "repo_dir": "third_party/Video-Depth-Anything",
                "commit": "4f5ae23172ba60fd7bc11ef671cca678842c7072",
                "checkpoint": "",
                "encoder": "vitb",
                "depth_type": "relative",
            }
        },
    )
    assert any("Default checkpoint SHA-256 mismatch" in issue for issue in report["issues"])


def test_ffmpeg_concat_command_truncates_to_exact_frame_count(tmp_path: Path) -> None:
    cmd = build_ffmpeg_concat_command(
        ffmpeg="ffmpeg",
        list_file=tmp_path / "list.txt",
        output_video=tmp_path / "out.mp4",
        frame_count=197,
    )
    assert "-frames:v" in cmd
    assert cmd[cmd.index("-frames:v") + 1] == "197"


def test_ensure_vda_matplotlib_compat_is_idempotent(tmp_path: Path) -> None:
    utils = tmp_path / "utils"
    utils.mkdir()
    target = utils / "dc_utils.py"
    target.write_text(
        'import matplotlib.cm as cm\n'
        'import numpy as np\n'
        'def save_video(frames):\n'
        '    colormap = np.array(cm.get_cmap("inferno").colors)\n'
        '    return colormap\n',
        encoding="utf-8",
    )
    first = ensure_vda_matplotlib_compat(tmp_path)
    assert first["applied"] is True
    text = target.read_text(encoding="utf-8")
    assert 'colormaps["inferno"]' in text
    second = ensure_vda_matplotlib_compat(tmp_path)
    assert second["already_patched"] is True
    assert second["applied"] is False
