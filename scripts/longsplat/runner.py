"""
Thin LongSplat runner.

Constructs and executes the backend command for training and conversion
at the locked LongSplat commit.  Uses external subprocess (no shell=True)
and records full provenance.
"""

from __future__ import annotations

import json
import math
import os as _os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Locked backend identity
# ---------------------------------------------------------------------------

LONGSPLAT_REPO_URL = "https://github.com/NVlabs/LongSplat"
LONGSPLAT_COMMIT = "bf766eb903c3d9144d64088b8c80b2da67d39411"

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

# Passthrough keys become ``--<key>`` argv entries.  Keep the accepted grammar
# deliberately narrow so spellings such as ``model_path=/tmp/escape`` cannot
# smuggle a second representation of a reserved argparse option.
_EXTRA_PARAM_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


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
    convert_anisotropy_reg_weight: float = 0.01
    convert_anisotropy_soft_limit: float = 30.0
    # Quality gate configuration (off by default for backward compatibility).
    quality_gates: Any = None  # QualityGateConfig | None
    # Expected native checkpoint iteration — if set, training must produce
    # this exact iteration (sum of all stage iter counts).
    expected_native_checkpoint_iteration: int | None = None


# Parameter names that are owned by quality gates and must not appear
# in extra_train_args (to prevent configuration drift).
_GATE_OWNED_PARAMS = frozenset(
    {
        "min_match_count",
        "min_inlier_count",
        "min_inlier_ratio",
        "max_reprojection_rmse_px",
        "min_grid_coverage",
        "min_positive_depth_ratio",
        "max_rotation_step_deg",
        "max_translation_step_ratio",
        "reference_lookback",
        "min_correlation",
        "max_normalized_rmse",
        "min_aligned_fraction",
    }
)


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
    if (
        isinstance(config.convert_anisotropy_reg_weight, bool)
        or not isinstance(config.convert_anisotropy_reg_weight, (int, float))
        or not math.isfinite(float(config.convert_anisotropy_reg_weight))
        or config.convert_anisotropy_reg_weight < 0.0
    ):
        raise BackendValidationError(
            "convert_anisotropy_reg_weight must be a finite non-negative number, "
            f"got {config.convert_anisotropy_reg_weight!r}"
        )
    if (
        isinstance(config.convert_anisotropy_soft_limit, bool)
        or not isinstance(config.convert_anisotropy_soft_limit, (int, float))
        or not math.isfinite(float(config.convert_anisotropy_soft_limit))
        or config.convert_anisotropy_soft_limit <= 0.0
    ):
        raise BackendValidationError(
            "convert_anisotropy_soft_limit must be a finite positive number, "
            f"got {config.convert_anisotropy_soft_limit!r}"
        )
    # Validate expected_native_checkpoint_iteration if set
    if config.expected_native_checkpoint_iteration is not None:
        if (
            type(config.expected_native_checkpoint_iteration) is not int
            or config.expected_native_checkpoint_iteration <= 0
        ):
            raise BackendValidationError(
                "expected_native_checkpoint_iteration must be a positive int, "
                f"got {config.expected_native_checkpoint_iteration!r}"
            )
    _validate_extra_train_args(config.extra_train_args)


def _validate_extra_train_args(extra_train_args: Any) -> None:
    """Validate passthrough argv without permitting option injection."""
    if not isinstance(extra_train_args, dict):
        raise BackendValidationError(
            "extra_train_args must be a dict, got "
            f"{type(extra_train_args).__name__}"
        )

    gate_conflicts: list[str] = []
    for key, value in extra_train_args.items():
        if not isinstance(key, str) or _EXTRA_PARAM_RE.fullmatch(key) is None:
            raise BackendValidationError(
                f"invalid passthrough parameter name: {key!r}"
            )
        if key in _RESERVED_PARAMS:
            raise BackendValidationError(
                f"extra_train_args must not override reserved param: {key}"
            )
        if key in _GATE_OWNED_PARAMS:
            gate_conflicts.append(key)
        protected_abbreviations = sorted(
            name
            for name in (_RESERVED_PARAMS | _GATE_OWNED_PARAMS)
            if name != key and name.startswith(key)
        )
        if protected_abbreviations:
            raise BackendValidationError(
                f"passthrough parameter {key!r} abbreviates protected params: "
                f"{protected_abbreviations}"
            )
        if isinstance(value, str) and value.startswith("-"):
            raise BackendValidationError(
                f"extra_train_args[{key!r}] contains option-like value: {value!r}"
            )
        if isinstance(value, list):
            injected = [
                item
                for item in value
                if isinstance(item, str) and item.startswith("-")
            ]
            if injected:
                raise BackendValidationError(
                    f"extra_train_args[{key!r}] contains option-like list value: "
                    f"{injected[0]!r}"
                )

    if gate_conflicts:
        raise BackendValidationError(
            "extra_train_args must not contain gate-owned params: "
            f"{sorted(gate_conflicts)}. Use quality_gates config instead."
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

    # Forward quality gate parameters to the training backend.
    if config.quality_gates is not None:
        qg = config.quality_gates
        if qg.pose.mode != "off":
            for attr_name in [
                "min_match_count", "min_inlier_count", "min_inlier_ratio",
                "max_reprojection_rmse_px", "min_grid_coverage",
                "min_positive_depth_ratio", "max_rotation_step_deg",
                "max_translation_step_ratio", "reference_lookback",
            ]:
                cmd.extend([f"--{attr_name}", str(getattr(qg.pose, attr_name))])
        if qg.vda.mode != "off":
            vda_cli_map = {
                "min_correlation": "min_correlation",
                "min_inlier_ratio": "min_vda_inlier_ratio",
                "max_normalized_rmse": "max_normalized_rmse",
                "min_aligned_fraction": "min_aligned_fraction",
            }
            for attr_name, cli_name in vda_cli_map.items():
                cmd.extend([f"--{cli_name}", str(getattr(qg.vda, attr_name))])

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
        "--seed",
        str(config.seed),
        "--anisotropy_reg_weight",
        str(config.convert_anisotropy_reg_weight),
        "--anisotropy_soft_limit",
        str(config.convert_anisotropy_soft_limit),
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


def _reject_nonfinite(d: Any) -> Any:
    """Recursively scan a JSON-decoded value and reject NaN/Infinity."""
    import math

    if isinstance(d, float):
        if not math.isfinite(d):
            raise BackendValidationError(f"non-finite float value: {d}")
        return d
    if isinstance(d, dict):
        for k, v in d.items():
            _reject_nonfinite(k)
            _reject_nonfinite(v)
        return d
    if isinstance(d, (list, tuple)):
        for item in d:
            _reject_nonfinite(item)
        return d
    return d


def _check_not_bool(value: Any, name: str) -> None:
    """Reject bool where int/float is expected (bool is a subclass of int)."""
    if isinstance(value, bool):
        raise BackendValidationError(
            f"{name} must not be a boolean (got {value!r})"
        )


def load_config(path: str | Path) -> LongSplatConfig:
    """Load a LongSplatConfig from a JSON file.  Unknown keys fail loudly.

    Rejects non-finite numeric values, booleans masquerading as integers,
    and unknown top-level keys.
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    # Reject NaN/Inf anywhere in the parsed data
    _reject_nonfinite(data)

    known = set(LongSplatConfig.__dataclass_fields__)
    allowed_extra = {"extra_train_args", "quality_gates", "expected_native_checkpoint_iteration"}
    unknown = set(data) - known - allowed_extra
    if unknown:
        raise BackendValidationError(
            f"Unknown config keys: {sorted(unknown)}. Known: {sorted(known | allowed_extra)}"
        )

    # Reject bool-as-int for known integer fields
    _int_fields = {"iterations", "convert_iteration", "seed", "resolution", "sh_degree"}
    for int_field in _int_fields:
        if int_field in data:
            _check_not_bool(data[int_field], int_field)

    if "expected_native_checkpoint_iteration" in data:
        _check_not_bool(data["expected_native_checkpoint_iteration"], "expected_native_checkpoint_iteration")

    # Validate extra_train_args if present
    if "extra_train_args" in data:
        _validate_extra_train_args(data["extra_train_args"])

    # Parse quality_gates if present
    quality_gates = None
    if "quality_gates" in data:
        qg_data = data.pop("quality_gates")
        if qg_data is not None:
            if not isinstance(qg_data, dict):
                raise BackendValidationError(
                    f"quality_gates must be a dict or null, got {type(qg_data).__name__}"
                )
            from scripts.longsplat.quality_gates import QualityGateConfig

            quality_gates = QualityGateConfig.from_dict(qg_data)

    expected_checkpoint = data.pop("expected_native_checkpoint_iteration", None)

    cfg = LongSplatConfig(
        **{k: v for k, v in data.items() if k in LongSplatConfig.__dataclass_fields__}
    )
    cfg.quality_gates = quality_gates
    cfg.expected_native_checkpoint_iteration = expected_checkpoint
    _validate_config(cfg)
    return cfg


def config_to_dict(config: LongSplatConfig) -> dict[str, Any]:
    """Convert config to a JSON-serialisable dict for run records."""
    from dataclasses import asdict

    d = asdict(config)
    # Replace quality_gates object with its dict representation
    if config.quality_gates is not None:
        d["quality_gates"] = config.quality_gates.to_dict()
    else:
        d["quality_gates"] = None
    return d
