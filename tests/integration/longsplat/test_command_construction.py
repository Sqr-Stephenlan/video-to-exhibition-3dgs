"""
Command construction tests for the LongSplat runner.

Uses fake subprocess fixtures — no real GPU, LongSplat, or checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.longsplat.runner import (  # noqa: E402
    BackendValidationError,
    LongSplatConfig,
    _check_repo,
    build_convert_command,
    build_train_command,
    config_to_dict,
    load_config,
    run_training,
    run_conversion,
    _check_python,
)


# ---------------------------------------------------------------------------
# Fake LongSplat repo for commit/submodule checks
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_longsplat_repo(tmp_path):
    """Create a minimal fake LongSplat repo at the locked commit."""
    repo = tmp_path / "LongSplat"
    repo.mkdir()
    (repo / ".git").mkdir()
    # Write a train.py stub
    (repo / "train.py").write_text("# stub")
    (repo / "convert_3dgs.py").write_text("# stub")

    # Init git and set to locked commit (fake)
    import subprocess
    subprocess.run(["git", "-C", str(repo), "init"], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@test"], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], capture_output=True)
    (repo / "dummy").write_text("init")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], capture_output=True)
    (repo / "dummy").write_text("to-locked")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "locked", "--allow-empty"],
        capture_output=True,
    )
    # Get actual commit and patch the module constant (for test purposes)
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    actual_commit = result.stdout.strip()

    # Create submodule directories
    for sub in ["submodules/mast3r", "submodules/diff-gaussian-rasterization",
                "submodules/fused-ssim", "submodules/simple-knn"]:
        sub_dir = repo / sub
        sub_dir.mkdir(parents=True)
        (sub_dir / ".git").mkdir()

    return repo, actual_commit


@pytest.fixture
def smoke_config(tmp_path):
    source = tmp_path / "input" / "images"
    source.mkdir(parents=True)
    model = tmp_path / "model_output"
    return LongSplatConfig(
        source_path=str(source),
        model_path=str(model),
        iterations=100,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def test_build_train_command_args(smoke_config, tmp_path):
    cmd = build_train_command(tmp_path, smoke_config, python_exe="python")
    assert cmd[0] == "python"
    assert str(tmp_path / "train.py") in cmd[1]
    assert "--source_path" in cmd
    assert "--model_path" in cmd
    assert "--iterations" in cmd
    assert "100" in cmd
    assert "--quiet" in cmd


def test_build_train_command_no_shell(smoke_config, tmp_path):
    """All args must be individual list elements — no shell strings."""
    cmd = build_train_command(tmp_path, smoke_config)
    for arg in cmd:
        assert " " not in arg or arg.startswith("--"), \
            f"Arg '{arg}' contains spaces — use separate list elements, not shell strings"


def test_build_train_command_extra_args(smoke_config, tmp_path):
    smoke_config.extra_train_args = {"gpu": "0", "detect_anomaly": True}
    cmd = build_train_command(tmp_path, smoke_config)
    gpu_idx = cmd.index("--gpu")
    assert cmd[gpu_idx + 1] == "0"
    assert "--detect_anomaly" in cmd


def test_build_convert_command_args(smoke_config, tmp_path):
    cmd = build_convert_command(tmp_path, smoke_config)
    assert str(tmp_path / "convert_3dgs.py") in cmd[1]
    assert "--source_path" in cmd
    assert "--model_path" in cmd
    assert "--iteration" in cmd
    idx = cmd.index("--iteration")
    assert cmd[idx + 1] == str(smoke_config.convert_iteration)
    assert "--prune_ratio" in cmd


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_load_config_from_json(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "source_path": "/tmp/images",
        "model_path": "/tmp/model",
        "iterations": 500,
        "seed": 1,
    }))
    config = load_config(str(cfg))
    assert config.source_path == "/tmp/images"
    assert config.model_path == "/tmp/model"
    assert config.iterations == 500


def test_load_config_missing_required_raises(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({"source_path": "/tmp/img"}))
    with pytest.raises(TypeError):
        load_config(str(cfg))


# ---------------------------------------------------------------------------
# Repo validation (needs actual git repo)
# ---------------------------------------------------------------------------


def test_check_repo_missing_raises(tmp_path):
    with pytest.raises(BackendValidationError, match="not found"):
        _check_repo(tmp_path / "nonexistent")


def test_check_repo_commit_mismatch_raises(fake_longsplat_repo):
    """Repo exists but at a different commit than locked."""
    repo, actual_commit = fake_longsplat_repo
    # Create a new commit so the repo diverges from LONGSPLAT_COMMIT
    (repo / "new_file").write_text("diverged")
    import subprocess
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "new"], capture_output=True)
    with pytest.raises(BackendValidationError, match="commit mismatch"):
        _check_repo(repo)


def test_check_repo_missing_submodules_raises(fake_longsplat_repo):
    """Repo at right commit but missing a submodule."""
    repo, actual_commit = fake_longsplat_repo
    # Remove a submodule
    import shutil
    shutil.rmtree(repo / "submodules" / "mast3r")
    # Patch LONGSPLAT_COMMIT to match this fake repo so we reach the
    # submodule check instead of failing on commit mismatch.
    with mock.patch("scripts.longsplat.runner.LONGSPLAT_COMMIT", actual_commit):
        with mock.patch(
            "scripts.longsplat.runner._LONGSPLAT_SUBMODULES",
            ["submodules/mast3r", "submodules/diff-gaussian-rasterization",
             "submodules/fused-ssim", "submodules/simple-knn"],
        ):
            with pytest.raises(BackendValidationError, match="submodules"):
                _check_repo(repo)


# ---------------------------------------------------------------------------
# config_to_dict
# ---------------------------------------------------------------------------


def test_config_to_dict(smoke_config):
    d = config_to_dict(smoke_config)
    assert d["source_path"] == smoke_config.source_path
    assert d["model_path"] == smoke_config.model_path
    assert d["iterations"] == 100
    assert d["seed"] == 42
    assert d["images"] == "images"
    assert d["resolution"] == -1
    assert d["sh_degree"] == 3
    assert d["convert_iteration"] == 30_000
    assert d["convert_prune_ratio"] == 0.6
    assert isinstance(d["extra_train_args"], dict)


def test_config_to_dict_roundtrip(smoke_config, tmp_path):
    """Dump to JSON and re-load; the config must survive serialization."""
    d = config_to_dict(smoke_config)
    p = tmp_path / "roundtrip.json"
    p.write_text(json.dumps(d))
    reloaded = load_config(str(p))
    assert reloaded.source_path == smoke_config.source_path
    assert reloaded.model_path == smoke_config.model_path
    assert reloaded.iterations == smoke_config.iterations
    assert reloaded.seed == smoke_config.seed


# ---------------------------------------------------------------------------
# _check_python
# ---------------------------------------------------------------------------


def test_check_python_valid():
    """sys.executable should always pass."""
    import sys
    _check_python(sys.executable)


def test_check_python_missing():
    with pytest.raises(BackendValidationError, match="not functional"):
        _check_python("/nonexistent/python_exe_xyz_non_existent")


# ---------------------------------------------------------------------------
# run_training / run_conversion (fake subprocess)
# ---------------------------------------------------------------------------


def test_run_training_fake_subprocess(tmp_path, smoke_config):
    """run_training should build command and call subprocess.run."""
    with mock.patch("scripts.longsplat.runner._check_repo"):
        with mock.patch("scripts.longsplat.runner._check_python"):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.MagicMock(
                    returncode=0, stdout="ok", stderr="",
                )
                run_training("/fake/repo", smoke_config, python_exe="python")

    mock_run.assert_called_once()
    train_call_args = mock_run.call_args[0][0]
    assert "train.py" in train_call_args[1] or any("train.py" in a for a in train_call_args)


def test_run_conversion_fake_subprocess(tmp_path, smoke_config):
    """run_conversion should build command and call subprocess.run."""
    with mock.patch("scripts.longsplat.runner._check_repo"):
        with mock.patch("scripts.longsplat.runner._check_python"):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.MagicMock(
                    returncode=0, stdout="ok", stderr="",
                )
                run_conversion("/fake/repo", smoke_config, python_exe="python")

    mock_run.assert_called_once()
    convert_call_args = mock_run.call_args[0][0]
    assert "convert_3dgs.py" in convert_call_args[1] or any(
        "convert_3dgs.py" in a for a in convert_call_args
    )
