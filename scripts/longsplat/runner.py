"""
Thin LongSplat runner.

Constructs and executes the backend command for training and conversion
at the locked LongSplat commit.  Uses external subprocess (no shell=True)
and records full provenance.
"""

from __future__ import annotations

import json
import os as _os
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

# Parameters that MUST NOT be overridden via extra_train_args.
_RESERVED_PARAMS = frozenset(
    {
        "source_path",
        "model_path",
        "images",
        "resolution",
        "sh_degree",
        "iterations",
        "mode",
        "eval",
        "load_pose",
        "white_background",
        "seed",
    }
)

# Valid backend_mode values.
_BACKEND_MODES = frozenset({"locked_clean", "research_local"})


# ---------------------------------------------------------------------------
# Immutable command container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectiveCommands:
    """Train and convert command tuples built once, executed and recorded as-is."""

    train: tuple[str, ...]
    convert: tuple[str, ...]


def build_effective_commands(
    repo_root: str | Path,
    config: LongSplatConfig,
    python_exe: str = "python",
) -> EffectiveCommands:
    """Build both commands in one shot.  The returned object is immutable
    and safe to pass to both the subprocess runner and the run record."""
    _validate_config(config)
    return EffectiveCommands(
        train=tuple(build_train_command(repo_root, config, python_exe)),
        convert=tuple(build_convert_command(repo_root, config, python_exe)),
    )


def backend_subprocess_env(repo_root: str | Path) -> dict[str, str]:
    """Return an env dict with *repo_root* prepended to ``PYTHONPATH``."""
    env = _os.environ.copy()
    root = str(Path(repo_root).resolve())
    previous = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not previous else root + _os.pathsep + previous
    return env


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
    # Backend validation mode: "locked_clean" (production) or "research_local".
    backend_mode: str = "locked_clean"
    # Stage-specific iteration overrides for LongSplat.
    # Defaults (from locked train.py) are long runs; smoke tests need these.
    extra_train_args: dict[str, Any] = field(default_factory=dict)
    # Converter settings
    convert_iteration: int = 30_000
    convert_prune_ratio: float = 0.6


def _validate_config(config: LongSplatConfig) -> None:
    """Validate config value ranges/types before command construction."""
    if config.backend_mode not in _BACKEND_MODES:
        raise BackendValidationError(
            f"invalid backend_mode: {config.backend_mode!r}; "
            f"must be one of {sorted(_BACKEND_MODES)}"
        )
    if type(config.seed) is not int or config.seed != 0:
        raise BackendValidationError(
            "locked LongSplat uses a fixed effective seed of 0"
        )
    for name in ("iterations", "convert_iteration"):
        value = getattr(config, name)
        if type(value) is not int or value <= 0:
            raise BackendValidationError(
                f"{name} must be a positive int, got {value!r}"
            )
    if config.sh_degree not in (0, 1, 2, 3):
        raise BackendValidationError(f"sh_degree must be 0-3, got {config.sh_degree}")
    if config.resolution != -1 and config.resolution <= 0:
        raise BackendValidationError(
            f"resolution must be -1 or positive, got {config.resolution}"
        )
    if not (0.0 <= config.convert_prune_ratio <= 1.0):
        raise BackendValidationError(
            f"convert_prune_ratio must be in [0.0, 1.0], got {config.convert_prune_ratio}"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class BackendValidationError(Exception):
    """Raised when the LongSplat backend environment is misconfigured."""


def _check_repo(repo_root: str | Path, *, backend_mode: str = "locked_clean") -> Path:
    """Verify the LongSplat repo exists, is at the locked commit, and all
    submodules are at their pinned gitlink SHAs.

    In ``locked_clean`` mode uncommitted changes raise an error.  In
    ``research_local`` mode they are allowed (and the diff is recorded
    by the caller via :func:`_resolve_backend_identity`).
    """
    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise BackendValidationError(f"LongSplat repo not found: {root}")

    git_dir = root / ".git"
    if not git_dir.exists():
        raise BackendValidationError(f"Not a git repository: {root}")

    # Check superproject commit
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != LONGSPLAT_COMMIT:
        raise BackendValidationError(
            f"LongSplat commit mismatch: expected {LONGSPLAT_COMMIT}, got {actual}"
        )

    # Check submodule init and gitlink SHAs (before dirty check: missing
    # submodules are more critical than uncommitted local changes).
    for sub, expected_sha in _LONGSPLAT_SUBMODULE_LINKS.items():
        sub_path = root / sub
        if not sub_path.is_dir() or not (sub_path / ".git").exists():
            raise BackendValidationError(f"LongSplat submodule not initialized: {sub}")

        sha_result = subprocess.run(
            ["git", "-C", str(sub_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        actual_sha = sha_result.stdout.strip()
        if actual_sha != expected_sha:
            raise BackendValidationError(
                f"Submodule {sub} SHA mismatch: expected {expected_sha}, "
                f"got {actual_sha}"
            )

    # Check dirty state after submodules.  In locked_clean mode uncommitted
    # changes break reproducibility.  git status failures always raise.
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if status_result.returncode != 0:
        raise BackendValidationError(
            f"Failed to inspect repo cleanliness (exit {status_result.returncode})"
        )
    dirty_out = status_result.stdout.strip()
    if dirty_out and backend_mode == "locked_clean":
        raise BackendValidationError(
            f"LongSplat repo has uncommitted changes:\n{dirty_out[:500]}"
        )

    return root


def _check_python(python_exe: str) -> None:
    """Verify the python executable exists."""
    try:
        result = subprocess.run(
            [python_exe, "--version"],
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, NotADirectoryError):
        raise BackendValidationError(f"Python executable not functional: {python_exe}")
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
    _validate_config(config)
    root = Path(repo_root)
    train_script = root / "train.py"

    cmd = [
        python_exe,
        str(train_script),
        "--source_path",
        str(config.source_path),
        "--model_path",
        str(config.model_path),
        "--images",
        config.images,
        "--resolution",
        str(config.resolution),
        "--sh_degree",
        str(config.sh_degree),
        "--iterations",
        str(config.iterations),
        "--mode",
        config.mode,
    ]

    # Reject reserved parameter overrides
    for key in config.extra_train_args:
        if key in _RESERVED_PARAMS:
            raise BackendValidationError(
                f"extra_train_args must not override reserved param: {key}"
            )

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
        elif isinstance(value, list):
            # nargs="+" / nargs="*" flags — one element per list item.
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
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
    _validate_config(config)
    root = Path(repo_root)
    convert_script = root / "convert_3dgs.py"

    return [
        python_exe,
        str(convert_script),
        "--source_path",
        str(config.source_path),
        "--model_path",
        str(config.model_path),
        "--iteration",
        str(config.convert_iteration),
        "--prune_ratio",
        str(config.convert_prune_ratio),
    ]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def run_training(
    repo_root: str | Path,
    config: LongSplatConfig,
    python_exe: str = "python",
    commands: EffectiveCommands | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute LongSplat training.  Blocks until completion.

    If *commands* is given its ``train`` tuple is executed as-is;
    otherwise the command is built from *config*.
    """
    validated = _check_repo(repo_root, backend_mode=config.backend_mode)
    _check_python(python_exe)
    _validate_config(config)

    cmd = (
        list(commands.train)
        if commands
        else build_train_command(
            validated,
            config,
            python_exe,
        )
    )
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(validated),
        env=backend_subprocess_env(validated),
    )


def run_conversion(
    repo_root: str | Path,
    config: LongSplatConfig,
    python_exe: str = "python",
    commands: EffectiveCommands | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute LongSplat convert_3dgs.  Blocks until completion."""
    validated = _check_repo(repo_root, backend_mode=config.backend_mode)
    _check_python(python_exe)
    _validate_config(config)

    cmd = (
        list(commands.convert)
        if commands
        else build_convert_command(
            validated,
            config,
            python_exe,
        )
    )
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(validated),
        env=backend_subprocess_env(validated),
    )


# ---------------------------------------------------------------------------
# Config serialisation helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> LongSplatConfig:
    """Load a LongSplatConfig from a JSON file.  Unknown keys fail loudly."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    known = set(LongSplatConfig.__dataclass_fields__)
    unknown = set(data) - known - {"extra_train_args"}
    if unknown:
        raise BackendValidationError(
            f"Unknown config keys: {sorted(unknown)}. Known: {sorted(known)}"
        )
    # Validate extra_train_args if present
    if "extra_train_args" in data:
        if not isinstance(data["extra_train_args"], dict):
            raise BackendValidationError(
                f"extra_train_args must be a dict, got "
                f"{type(data['extra_train_args']).__name__}"
            )
        for key in data["extra_train_args"]:
            if key in _RESERVED_PARAMS:
                raise BackendValidationError(
                    f"Config extra_train_args must not override reserved param: {key}"
                )
    cfg = LongSplatConfig(
        **{k: v for k, v in data.items() if k in LongSplatConfig.__dataclass_fields__}
    )
    _validate_config(cfg)
    return cfg


def config_to_dict(config: LongSplatConfig) -> dict[str, Any]:
    """Convert config to a JSON-serialisable dict for run records."""
    from dataclasses import asdict

    return asdict(config)
