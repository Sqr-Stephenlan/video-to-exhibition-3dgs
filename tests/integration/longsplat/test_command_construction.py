"""
Command construction tests for the LongSplat runner.

Uses fake subprocess fixtures — no real GPU, LongSplat, or checkpoint.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import subprocess as _real_subprocess_module

import pytest

_real_run = _real_subprocess_module.run
_real_CompletedProcess = _real_subprocess_module.CompletedProcess

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
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test"],
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "test"], capture_output=True
    )
    (repo / "dummy").write_text("init")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "init"], capture_output=True
    )
    (repo / "dummy").write_text("to-locked")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "locked", "--allow-empty"],
        capture_output=True,
    )
    # Get actual commit and patch the module constant (for test purposes)
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    actual_commit = result.stdout.strip()

    # Create submodule directories as real git repos (needed for SHA checks)
    sub_commits = {}
    for sub in [
        "submodules/mast3r",
        "submodules/diff-gaussian-rasterization",
        "submodules/fused-ssim",
        "submodules/simple-knn",
    ]:
        sub_dir = repo / sub
        sub_dir.mkdir(parents=True)
        subprocess.run(["git", "-C", str(sub_dir), "init"], capture_output=True)
        subprocess.run(
            ["git", "-C", str(sub_dir), "config", "user.email", "test@test"],
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(sub_dir), "config", "user.name", "test"],
            capture_output=True,
        )
        (sub_dir / "stub").write_text("sub stub")
        subprocess.run(["git", "-C", str(sub_dir), "add", "."], capture_output=True)
        subprocess.run(
            ["git", "-C", str(sub_dir), "commit", "-m", "sub init"],
            capture_output=True,
        )
        sha_result = subprocess.run(
            ["git", "-C", str(sub_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        sub_commits[sub] = sha_result.stdout.strip()

    return repo, actual_commit, sub_commits


@pytest.fixture
def smoke_config(tmp_path):
    source = tmp_path / "input" / "images"
    source.mkdir(parents=True)
    model = tmp_path / "model_output"
    return LongSplatConfig(
        source_path=str(source),
        model_path=str(model),
        iterations=100,
        seed=0,
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
    assert "--quiet" not in cmd, (
        "train command must not contain --quiet (stdout is used for VDA telemetry)"
    )


def test_build_train_command_no_shell(smoke_config, tmp_path):
    """All args must be individual list elements — no shell strings."""
    cmd = build_train_command(tmp_path, smoke_config)
    for arg in cmd:
        assert " " not in arg or arg.startswith("--"), (
            f"Arg '{arg}' contains spaces — use separate list elements, not shell strings"
        )


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
    cfg.write_text(
        json.dumps(
            {
                "source_path": "/tmp/images",
                "model_path": "/tmp/model",
                "iterations": 500,
                "seed": 0,
            }
        )
    )
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
    repo, actual_commit, _sub_commits = fake_longsplat_repo
    # Create a new commit so the repo diverges from LONGSPLAT_COMMIT
    (repo / "new_file").write_text("diverged")
    import subprocess

    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "new"], capture_output=True)
    with pytest.raises(BackendValidationError, match="commit mismatch"):
        _check_repo(repo)


def test_check_repo_missing_submodules_raises(fake_longsplat_repo):
    """Repo at right commit but missing a submodule."""
    repo, actual_commit, sub_commits = fake_longsplat_repo
    # Remove a submodule (handle Windows read-only git objects)
    import shutil

    def _on_rm_error(func, path, exc_info):
        os.chmod(path, 0o666)
        func(path)

    shutil.rmtree(repo / "submodules" / "mast3r", onerror=_on_rm_error)
    # Patch LONGSPLAT_COMMIT and submodule links to match this fake repo.
    with mock.patch("scripts.longsplat.runner.LONGSPLAT_COMMIT", actual_commit):
        with mock.patch(
            "scripts.longsplat.runner._LONGSPLAT_SUBMODULE_LINKS",
            sub_commits,
        ):
            with pytest.raises(BackendValidationError, match="not initialized"):
                _check_repo(repo)


# ---------------------------------------------------------------------------
# config_to_dict
# ---------------------------------------------------------------------------


def test_config_to_dict(smoke_config):
    d = config_to_dict(smoke_config)
    assert d["source_path"] == smoke_config.source_path
    assert d["model_path"] == smoke_config.model_path
    assert d["iterations"] == 100
    assert d["seed"] == 0
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
    """run_training should build command with correct cwd and PYTHONPATH."""
    backend = tmp_path / "fake_backend"
    backend.mkdir()
    resolved = backend.resolve()

    with mock.patch("scripts.longsplat.runner._check_repo", return_value=resolved):
        with mock.patch("scripts.longsplat.runner._check_python"):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.MagicMock(
                    returncode=0,
                    stdout="ok",
                    stderr="",
                )
                run_training(str(backend), smoke_config, python_exe="python")

    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["cwd"] == str(resolved), (
        f"cwd must be resolved backend path, got {kwargs.get('cwd')!r}"
    )
    env = kwargs["env"]
    pythonpath_dirs = env["PYTHONPATH"].split(os.pathsep)
    assert str(resolved) in pythonpath_dirs, (
        f"PYTHONPATH must contain backend root {resolved}, "
        f"got {env.get('PYTHONPATH')!r}"
    )
    train_call_args = mock_run.call_args[0][0]
    assert "train.py" in train_call_args[1] or any(
        "train.py" in a for a in train_call_args
    )


def test_run_conversion_fake_subprocess(tmp_path, smoke_config):
    """run_conversion should build command with correct cwd and PYTHONPATH."""
    backend = tmp_path / "fake_backend"
    backend.mkdir()
    resolved = backend.resolve()

    with mock.patch("scripts.longsplat.runner._check_repo", return_value=resolved):
        with mock.patch("scripts.longsplat.runner._check_python"):
            with mock.patch("subprocess.run") as mock_run:
                mock_run.return_value = mock.MagicMock(
                    returncode=0,
                    stdout="ok",
                    stderr="",
                )
                run_conversion(str(backend), smoke_config, python_exe="python")

    mock_run.assert_called_once()
    kwargs = mock_run.call_args.kwargs
    assert kwargs["cwd"] == str(resolved), (
        f"cwd must be resolved backend path, got {kwargs.get('cwd')!r}"
    )
    env = kwargs["env"]
    pythonpath_dirs = env["PYTHONPATH"].split(os.pathsep)
    assert str(resolved) in pythonpath_dirs, (
        f"PYTHONPATH must contain backend root {resolved}, "
        f"got {env.get('PYTHONPATH')!r}"
    )
    convert_call_args = mock_run.call_args[0][0]
    assert "convert_3dgs.py" in convert_call_args[1] or any(
        "convert_3dgs.py" in a for a in convert_call_args
    )


# ---------------------------------------------------------------------------
# Task 1: backend_mode, seed=0, and validation contract (TDD — red first)
# ---------------------------------------------------------------------------


def test_dirty_backend_requires_research_local_mode(fake_longsplat_repo):
    """locked_clean rejects dirty repo; research_local accepts it."""
    repo, actual_commit, sub_commits = fake_longsplat_repo

    # Make the repo dirty by modifying a tracked file
    (repo / "train.py").write_text("# modified")

    # Patch constants to match this fake repo
    with mock.patch("scripts.longsplat.runner.LONGSPLAT_COMMIT", actual_commit):
        with mock.patch(
            "scripts.longsplat.runner._LONGSPLAT_SUBMODULE_LINKS",
            sub_commits,
        ):
            # locked_clean must raise
            with pytest.raises(BackendValidationError, match="uncommitted"):
                _check_repo(repo, backend_mode="locked_clean")

            # research_local must succeed
            result = _check_repo(repo, backend_mode="research_local")
            assert result == repo.resolve()


def test_git_status_failure_is_not_clean(fake_longsplat_repo):
    """A broken git status (exit 128) must fail in BOTH modes."""
    repo, actual_commit, sub_commits = fake_longsplat_repo

    with mock.patch("scripts.longsplat.runner.LONGSPLAT_COMMIT", actual_commit):
        with mock.patch(
            "scripts.longsplat.runner._LONGSPLAT_SUBMODULE_LINKS",
            sub_commits,
        ):
            with mock.patch(
                "subprocess.run",
                side_effect=_make_git_status_failing_run,
            ):
                with pytest.raises(BackendValidationError, match="Failed to inspect"):
                    _check_repo(repo, backend_mode="locked_clean")

                with pytest.raises(BackendValidationError, match="Failed to inspect"):
                    _check_repo(repo, backend_mode="research_local")


def _make_git_status_failing_run(*args, **kwargs):
    """Monkeypatch subprocess.run to fail on git status --porcelain only."""
    cmd = args[0] if args else kwargs.get("args", [])
    cmd_str = " ".join(cmd)
    if "status" in cmd_str and "--porcelain" in cmd_str:
        return _real_CompletedProcess(cmd, 128, stdout="", stderr="fatal")
    return _real_run(*args, **kwargs)


@pytest.mark.parametrize("bad_seed", [True, 1, 42, -1])
def test_nonzero_or_boolean_seed_is_rejected(bad_seed):
    """Seed must be int 0; anything else raises before command construction."""
    config = LongSplatConfig(
        source_path="/tmp/src",
        model_path="/tmp/model",
        seed=bad_seed,
    )
    with pytest.raises(BackendValidationError, match="fixed effective seed of 0"):
        build_train_command("/tmp/repo", config)


def test_invalid_backend_mode_rejected_in_public_builder():
    """Invalid backend_mode must fail in public command construction paths."""
    config = LongSplatConfig(
        source_path="/tmp/src",
        model_path="/tmp/model",
        backend_mode="bad_mode_xyz",
    )
    with pytest.raises(BackendValidationError, match="invalid backend_mode"):
        build_train_command("/tmp/repo", config)


def test_seed_zero_is_recorded_but_not_passed():
    """train command for seed=0 must NOT contain --seed."""
    config = LongSplatConfig(
        source_path="/tmp/src",
        model_path="/tmp/model",
        seed=0,
    )
    cmd = build_train_command("/tmp/repo", config)
    assert "--seed" not in cmd, f"train command must not contain --seed, got: {cmd}"


@pytest.mark.parametrize("field", ["iterations", "convert_iteration"])
def test_boolean_iterations_are_rejected(field):
    """iterations/convert_iteration must be positive int, not bool."""
    config = LongSplatConfig(
        source_path="/tmp/src",
        model_path="/tmp/model",
    )
    setattr(config, field, True)
    with pytest.raises(BackendValidationError, match="must be a positive int"):
        build_train_command("/tmp/repo", config)
