"""
Thin LongSplat runner.

Constructs and executes the backend command for training and conversion
at the locked LongSplat commit.  Uses external subprocess (no shell=True)
and records full provenance.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Locked backend identity
# ---------------------------------------------------------------------------

LONGSPLAT_REPO_URL = "https://github.com/NVlabs/LongSplat"
LONGSPLAT_COMMIT = "19750775a9d19f30aa05a8333c4c6c231b2d5f4a"

# Submodules expected at the locked commit, with their pinned gitlink SHAs.
_LONGSPLAT_SUBMODULE_LINKS = {
    "submodules/mast3r": "f5209afc300cec36239a7ac992263f36847bbba0",
    "submodules/diff-gaussian-rasterization": (
        "401a405b2360677f3a71ab1930af94f8f2bfc1c3"
    ),
    "submodules/fused-ssim": "085e0f36d9009ebd241e019ea762442dd1aaeca9",
    "submodules/simple-knn": "86710c2d4b46680c02301765dd79e465819c8f19",
}

# LongSplat train.py boolean flags that are store_true (no value argument).
_STORE_TRUE_FLAGS = frozenset(
    {
        "quiet",
        "test_iterations",
        "skip_test",
        "skip_train",
        "detect_anomaly",
        "eval",
        "use_wandb",
    }
)

# LongSplat train.py boolean flags that are store_false.
_STORE_FALSE_FLAGS = frozenset(
    {
        "no_save",
    }
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class LongSplatConfig:
    """Version-controlled training configuration."""

    source_path: str
    model_path: str
    images: str = "images"
    resolution: int = -1
    sh_degree: int = 3
    iterations: int = 30_000
    mode: str = "custom"
    seed: int = 0
    # Stage-specific iteration overrides for LongSplat.
    # Defaults (from locked train.py) are long runs; smoke tests need these.
    extra_train_args: dict[str, Any] = field(default_factory=dict)
    # Converter settings
    convert_iteration: int = 30_000
    convert_prune_ratio: float = 0.6


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class BackendValidationError(Exception):
    """Raised when the LongSplat backend environment is misconfigured."""


def _check_repo(repo_root: str | Path) -> Path:
    """Verify the LongSplat repo exists, is at the locked commit, and all
    submodules are at their pinned gitlink SHAs."""
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise BackendValidationError(f"LongSplat repo not found: {root}")

    git_dir = root / ".git"
    if not git_dir.exists():
        raise BackendValidationError(f"Not a git repository: {root}")

    # Check superproject commit
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    actual = result.stdout.strip()
    if actual != LONGSPLAT_COMMIT:
        raise BackendValidationError(
            f"LongSplat commit mismatch: expected {LONGSPLAT_COMMIT}, "
            f"got {actual}"
        )

    # Check submodule init and gitlink SHAs (before dirty check: missing
    # submodules are more critical than uncommitted local changes).
    for sub, expected_sha in _LONGSPLAT_SUBMODULE_LINKS.items():
        sub_path = root / sub
        if not sub_path.is_dir() or not (sub_path / ".git").exists():
            raise BackendValidationError(
                f"LongSplat submodule not initialized: {sub}"
            )

        sha_result = subprocess.run(
            ["git", "-C", str(sub_path), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        actual_sha = sha_result.stdout.strip()
        if actual_sha != expected_sha:
            raise BackendValidationError(
                f"Submodule {sub} SHA mismatch: expected {expected_sha}, "
                f"got {actual_sha}"
            )

    # Check dirty state after submodules — a dirty repo is a warning-worthy
    # condition but missing/wrong submodules are hard failures.
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    if status_result.stdout.strip():
        import sys
        print(
            f"WARNING: LongSplat repo has uncommitted changes:\n"
            f"{status_result.stdout[:500]}",
            file=sys.stderr,
        )

    return root


def _check_python(python_exe: str) -> None:
    """Verify the python executable exists."""
    try:
        result = subprocess.run(
            [python_exe, "--version"],
            capture_output=True, text=True,
        )
    except (FileNotFoundError, NotADirectoryError):
        raise BackendValidationError(
            f"Python executable not functional: {python_exe}"
        )
    if result.returncode != 0:
        raise BackendValidationError(
            f"Python executable not functional: {python_exe}\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def build_train_command(
    repo_root: str | Path,
    config: LongSplatConfig,
    python_exe: str = "python",
) -> list[str]:
    """Build the argument list for ``train.py`` (no shell).

    Returns a list ready for :func:`subprocess.run`.
    """
    root = Path(repo_root)
    train_script = root / "train.py"

    cmd = [
        python_exe,
        str(train_script),
        "--source_path", str(config.source_path),
        "--model_path", str(config.model_path),
        "--images", config.images,
        "--resolution", str(config.resolution),
        "--sh_degree", str(config.sh_degree),
        "--iterations", str(config.iterations),
        "--mode", config.mode,
        "--quiet",
    ]

    # Append extra passthrough args with correct bool handling.
    for key, value in config.extra_train_args.items():
        flag = f"--{key}"
        if key in _STORE_TRUE_FLAGS:
            if value:
                cmd.append(flag)
            # store_true=False means omit the flag
        elif key in _STORE_FALSE_FLAGS:
            if not value:
                cmd.append(flag)
            # store_false=True means omit the flag
        elif isinstance(value, bool):
            # For boolean-valued flags that are NOT store_true/store_false,
            # keep the old behavior: emit flag if value is truthy.
            if value:
                cmd.append(flag)
        else:
            cmd.append(flag)
            cmd.append(str(value))

    return cmd


def build_convert_command(
    repo_root: str | Path,
    config: LongSplatConfig,
    python_exe: str = "python",
) -> list[str]:
    """Build the argument list for ``convert_3dgs.py`` (no shell)."""
    root = Path(repo_root)
    convert_script = root / "convert_3dgs.py"

    return [
        python_exe,
        str(convert_script),
        "--source_path", str(config.source_path),
        "--model_path", str(config.model_path),
        "--iteration", str(config.convert_iteration),
        "--prune_ratio", str(config.convert_prune_ratio),
    ]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_training(
    repo_root: str | Path,
    config: LongSplatConfig,
    python_exe: str = "python",
) -> subprocess.CompletedProcess[str]:
    """Execute LongSplat training.  Blocks until completion.

    Returns the :class:`subprocess.CompletedProcess` with captured stdout/stderr.
    """
    _check_repo(repo_root)
    _check_python(python_exe)

    cmd = build_train_command(repo_root, config, python_exe)
    return subprocess.run(cmd, capture_output=True, text=True)


def run_conversion(
    repo_root: str | Path,
    config: LongSplatConfig,
    python_exe: str = "python",
) -> subprocess.CompletedProcess[str]:
    """Execute LongSplat convert_3dgs.  Blocks until completion."""
    _check_repo(repo_root)
    _check_python(python_exe)

    cmd = build_convert_command(repo_root, config, python_exe)
    return subprocess.run(cmd, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Config serialisation helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> LongSplatConfig:
    """Load a LongSplatConfig from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return LongSplatConfig(**{k: v for k, v in data.items()
                              if k in LongSplatConfig.__dataclass_fields__})


def config_to_dict(config: LongSplatConfig) -> dict[str, Any]:
    """Convert config to a JSON-serialisable dict for run records."""
    from dataclasses import asdict

    return asdict(config)
