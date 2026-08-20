"""Shared immutable contracts for the CPU-only raw-video pipeline.

This module deliberately contains no LongSplat or CUDA imports.  It is the
small boundary between probing a video and later GPU/training stages.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PIPELINE_PREFLIGHT_SCHEMA = "pipeline-preflight-v1"
RUN_IDENTITY_SCHEMA = "run-identity-v1"
LEDGER_SCHEMA = "raw-video-run-v1"
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class PipelineContractError(RuntimeError):
    """Base error for a contract violation."""


class PipelineBlocked(PipelineContractError):
    """A safe stop caused by missing evidence or an unavailable dependency."""


class ResumeMismatchError(PipelineContractError):
    """An existing run was requested with different immutable inputs."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize JSON in the stable form used by all identity hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Write a mutable summary atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def write_json_once(path: str | Path, value: Any) -> None:
    """Create an immutable JSON record and refuse to overwrite it."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def resolve_output_root(value: str | Path) -> Path:
    """Resolve a user output root without accepting symlink aliases or ``/``.

    The canonical reconstruct entrypoint may use any ordinary absolute
    directory.  Existing path components are checked before resolution so a
    symlink cannot be hidden by a ``..`` or by ``Path.resolve``.
    """

    raw = Path(value)
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    raw = raw.absolute()
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PipelineBlocked(f"output root contains a symlink component: {raw}")
    resolved = raw.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise PipelineBlocked("output root cannot be the filesystem root")
    if resolved.exists() and not resolved.is_dir():
        raise PipelineBlocked(f"output root must be a directory: {resolved}")
    return resolved


def resolve_containment_root(value: str | Path, *, label: str = "containment root") -> Path:
    """Resolve a verified dynamic run/evidence root.

    ``route_root`` identifies source code; it is never a data containment
    boundary.  Executors use this helper for the caller-selected run root so
    an arbitrary output root remains safe without falling back to
    ``route/outputs``.
    """

    raw = Path(value)
    if not raw.is_absolute():
        raise PipelineBlocked(f"{label} must be an absolute path: {raw}")
    raw = raw.absolute()
    cursor = Path(raw.anchor)
    for part in raw.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise PipelineBlocked(f"{label} contains a symlink component: {raw}")
    resolved = raw.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise PipelineBlocked(f"{label} cannot be the filesystem root")
    if resolved.exists() and not resolved.is_dir():
        raise PipelineBlocked(f"{label} must be a directory: {resolved}")
    return resolved


def resolve_contained_path(
    value: str | Path,
    *,
    root: str | Path,
    label: str,
    must_exist: bool = False,
    directory: bool | None = None,
    allow_root: bool = False,
) -> Path:
    """Resolve an absolute path strictly below a verified dynamic root."""

    containment = resolve_containment_root(root, label=f"{label} containment root")
    raw = Path(value)
    if not raw.is_absolute():
        raise PipelineBlocked(f"{label} must be an absolute path: {raw}")
    raw = raw.absolute()
    try:
        relative = raw.relative_to(containment)
    except ValueError as exc:
        raise PipelineBlocked(f"{label} escapes {containment}: {raw}") from exc
    if not relative.parts and not allow_root:
        raise PipelineBlocked(f"{label} must be a strict descendant of {containment}")
    probe = containment
    for part in relative.parts:
        if part in {".", ".."}:
            raise PipelineBlocked(f"{label} contains traversal: {raw}")
        probe = probe / part
        if probe.is_symlink():
            raise PipelineBlocked(f"{label} traverses a symlink: {probe}")
    resolved = raw.resolve(strict=False)
    try:
        resolved.relative_to(containment)
    except ValueError as exc:
        raise PipelineBlocked(f"{label} resolves outside {containment}: {resolved}") from exc
    if not allow_root and resolved == containment:
        raise PipelineBlocked(f"{label} cannot equal {containment}")
    if must_exist and not resolved.exists():
        raise PipelineBlocked(f"{label} is missing: {resolved}")
    if directory is True and must_exist and not resolved.is_dir():
        raise PipelineBlocked(f"{label} must be a directory: {resolved}")
    if directory is False and must_exist and not resolved.is_file():
        raise PipelineBlocked(f"{label} must be a file: {resolved}")
    return resolved


STAGE_MIGRATION_SCHEMA = "stage-migration-decision-v1"


def classify_stage_migration(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    immutable_fields: Sequence[str],
    rebuildable_fields: Sequence[str] = (
        "schema_version",
        "consumer_code_identity_sha256",
        "stage_consumer_identity",
    ),
) -> dict[str, Any]:
    """Classify an old stage record without silently promoting orphan evidence.

    Missing or changed immutable data is unsafe drift.  A passed record with
    only derivable schema/consumer fields missing or changed is rebuildable
    append-only.  Failed, blocked, orphan, or explicitly mutated records are
    never reusable.
    """

    if not isinstance(observed, Mapping) or not isinstance(expected, Mapping):
        raise PipelineBlocked("stage migration records must be mappings")
    status = observed.get("status")
    if status != "passed" or observed.get("reusable", True) is False or observed.get("orphan") is True:
        return {
            "schema_version": STAGE_MIGRATION_SCHEMA,
            "classification": "reject_nonreusable",
            "reusable_exact": False,
            "stale_rebuildable": False,
            "unsafe_drift": False,
            "reason": "stage status is not a passed reusable producer attempt",
        }
    if observed.get("artifacts_sha_match") is False or observed.get("identity_valid") is False:
        return {
            "schema_version": STAGE_MIGRATION_SCHEMA,
            "classification": "reject_nonreusable",
            "reusable_exact": False,
            "stale_rebuildable": False,
            "unsafe_drift": False,
            "reason": "stage artifacts or identity are not verifiably intact",
        }
    unsafe: list[str] = []
    stale: list[str] = []
    for field in immutable_fields:
        if field not in observed:
            unsafe.append(field)
        elif observed.get(field) != expected.get(field):
            unsafe.append(field)
    if unsafe:
        classification = "reject_unsafe_drift"
        reason = "immutable source/camera/training/producer identity drift"
    else:
        expected_keys = set(expected)
        observed_keys = set(observed)
        for field in sorted(expected_keys | observed_keys):
            if field in immutable_fields or field in {"status", "reusable", "orphan", "artifacts_sha_match", "identity_valid"}:
                continue
            if observed.get(field) != expected.get(field):
                if field in rebuildable_fields or field not in observed_keys or field not in expected_keys:
                    stale.append(field)
                else:
                    stale.append(field)
        classification = "stale_rebuildable" if stale else "reusable_exact"
        reason = "append-only derived stage rebuild required" if stale else "stage identity is exact"
    return {
        "schema_version": STAGE_MIGRATION_SCHEMA,
        "classification": classification,
        "reusable_exact": classification == "reusable_exact",
        "stale_rebuildable": classification == "stale_rebuildable",
        "unsafe_drift": classification == "reject_unsafe_drift",
        "reason": reason,
        "diff": {"unsafe_fields": sorted(unsafe), "stale_fields": sorted(stale)},
    }


def validate_run_id(run_id: str) -> str:
    """Require a single safe basename for a run directory."""

    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or _SAFE_RUN_ID_RE.fullmatch(run_id) is None
    ):
        raise ResumeMismatchError("run-id must be a single safe basename")
    return run_id


def write_text_once(path: str | Path, value: str) -> None:
    """Create an immutable text record without replacing a prior attempt log."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(value)


def _run_capture(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[int, str, str]:
    try:
        result = runner(
            list(argv),
            cwd=None if cwd is None else str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", str(exc)
    return (
        int(getattr(result, "returncode", 1)),
        str(getattr(result, "stdout", "") or ""),
        str(getattr(result, "stderr", "") or ""),
    )


def _version_for(name: str, executable: str, runner: Callable[..., Any]) -> str | None:
    if name.endswith("python"):
        argv = [executable, "-c", "import sys; print(sys.version.split()[0])"]
    elif name == "colmap":
        argv = [executable, "--help"]
    else:
        argv = [executable, "-version"]
    code, stdout, stderr = _run_capture(argv, runner=runner)
    if code != 0:
        return None
    first_line = (stdout or stderr).strip().splitlines()
    return first_line[0][:500] if first_line else "unknown"


def _resolve_requested(requested: str | Path | None, name: str) -> str | None:
    if requested is None:
        requested_text = name
    else:
        requested_text = str(requested)
        separators = {os.sep, "\\"}
        if os.altsep:
            separators.add(os.altsep)
        if any(separator in requested_text for separator in separators):
            candidate = Path(requested_text)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
            return None
    resolved = shutil.which(requested_text)
    return None if resolved is None else str(Path(resolved).resolve())


def _executable_identity(path: str | Path) -> dict[str, Any]:
    """Return the identity of the resolved executable, including its bytes."""

    resolved = Path(path).resolve()
    try:
        stat = resolved.stat()
        if not resolved.is_file():
            raise OSError("resolved executable is not a regular file")
        return {
            "resolved_target": str(resolved),
            "size_bytes": int(stat.st_size),
            "sha256": sha256_file(resolved),
            "identity_status": "verified",
        }
    except (OSError, ValueError) as exc:
        return {
            "resolved_target": str(resolved),
            "size_bytes": None,
            "sha256": None,
            "identity_status": "blocked",
            "identity_error": str(exc),
        }


@dataclass(frozen=True)
class DependencySpec:
    name: str
    requested: str | None
    required_for: tuple[str, ...]


def preflight_dependencies(
    *,
    route_python: str | Path | None = None,
    backend_python: str | Path | None = None,
    ffmpeg: str | Path | None = None,
    ffprobe: str | Path | None = None,
    colmap: str | Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Resolve and version dependencies without installing anything.

    COLMAP is recorded even when absent.  A missing dependency is a
    structured ``blocked`` result, never a request to mutate the environment.
    """

    requested = {
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "colmap": colmap,
        "route_python": route_python or sys.executable,
        "backend_python": backend_python,
    }
    required_for = {
        "ffmpeg": ("canonical-media", "frames"),
        "ffprobe": ("probe",),
        "colmap": ("colmap", "camera-staging"),
        "route_python": ("all-cpu-stages",),
        "backend_python": ("backend-preflight",),
    }
    executable_names = {
        "ffmpeg": "ffmpeg",
        "ffprobe": "ffprobe",
        "colmap": "colmap",
        "route_python": "python",
        "backend_python": "python",
    }
    dependencies: dict[str, Any] = {}
    blocked: list[dict[str, Any]] = []
    for name, requested_path in requested.items():
        requested_text = None if requested_path is None else str(requested_path)
        resolved = _resolve_requested(requested_path, executable_names[name])
        executable = bool(resolved and os.access(resolved, os.X_OK))
        version = _version_for(name, resolved, runner) if executable and resolved else None
        entry = {
            "requested_path": requested_text,
            "resolved_path": resolved,
            "version": version,
            "executable": executable,
            "required_for": list(required_for[name]),
            "resolved_target": resolved,
            "size_bytes": None,
            "sha256": None,
        }
        if requested_text is not None and _resolve_requested(requested_path, executable_names[name]) is not None:
            requested_candidate = Path(requested_text)
            if any(separator in requested_text for separator in {os.sep, "\\"}):
                entry["symlink"] = requested_candidate.is_symlink()
        else:
            entry["symlink"] = False
        if executable and resolved:
            executable_identity = _executable_identity(resolved)
            entry["executable_identity"] = executable_identity
            entry["resolved_target"] = executable_identity.get("resolved_target")
            entry["size_bytes"] = executable_identity.get("size_bytes")
            entry["sha256"] = executable_identity.get("sha256")
        if executable and version is None:
            entry["reason"] = "version_probe_failed"
            blocked.append({"dependency": name, "reason": "version_probe_failed"})
        elif not executable:
            entry["reason"] = "missing_or_not_executable"
            blocked.append({"dependency": name, "reason": "missing_or_not_executable"})
        elif entry["executable_identity"]["identity_status"] != "verified":
            entry["reason"] = "executable_identity_unreadable"
            blocked.append({"dependency": name, "reason": "executable_identity_unreadable"})
        dependencies[name] = entry

    result = {
        "schema_version": PIPELINE_PREFLIGHT_SCHEMA,
        "status": "passed" if not blocked else "blocked",
        "dependencies": dependencies,
        "gpu": {
            "status": "deferred",
            "required_for": ["training", "conversion", "delivery"],
            "cuda_invoked": False,
            "note": "CPU-only first slice; GPU capability is recorded, not probed or used.",
        },
        "blocked": blocked,
    }
    result["tool_identity_sha256"] = stable_sha256(dependencies)
    return result


_IDENTITY_SUFFIXES = {
    ".py", ".pyi", ".sh", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".cfg"
}
_IDENTITY_EXCLUDED_PARTS = {
    ".git", ".codegraph", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "outputs", "output", "data", "datasets", "models", "checkpoints", "cache", "caches",
    ".venv", "venv", "env", "build", "dist", "submodules",
}


def _identity_files(root: Path, *, category: str, relative_prefix: str = "") -> list[dict[str, Any]]:
    """Hash relevant source files while excluding generated/data-heavy trees."""

    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(
            name for name in dirnames
            if name not in _IDENTITY_EXCLUDED_PARTS and not name.startswith(".")
        )
        for filename in sorted(filenames):
            path = Path(directory) / filename
            if path.suffix.lower() not in _IDENTITY_SUFFIXES and filename not in {"dev.sh", "Makefile"}:
                continue
            try:
                size = path.stat().st_size
                # A source identity must never accidentally hash a model or dump.
                if size > 16 * 1024 * 1024:
                    continue
                digest = sha256_file(path)
            except OSError:
                continue
            relative = path.relative_to(root).as_posix()
            records.append({
                "category": category,
                "path": f"{relative_prefix}{relative}",
                "sha256": digest,
                "size_bytes": int(size),
            })
    return records


def _git_identity(root: Path, scope: Sequence[str]) -> dict[str, Any]:
    commit = "unknown"
    diff_sha = stable_sha256({"git": "unavailable", "root": str(root), "scope": list(scope)})
    staged_diff_sha = stable_sha256({"git": "unavailable", "root": str(root), "scope": list(scope)})
    if not root.is_dir():
        return {"commit": commit, "working_tree_diff_sha256": diff_sha, "index_diff_sha256": staged_diff_sha}
    commit_code, commit_out, _ = _run_capture(["git", "rev-parse", "HEAD"], cwd=root)
    if commit_code == 0 and commit_out.strip():
        commit = commit_out.strip()
    diff_args = ["git", "diff", "--binary", "--", *scope] if scope else ["git", "diff", "--binary"]
    diff_code, diff_out, diff_err = _run_capture(diff_args, cwd=root)
    if diff_code == 0:
        diff_sha = hashlib.sha256(diff_out.encode("utf-8")).hexdigest()
    elif diff_err:
        diff_sha = stable_sha256({"git_diff_error": diff_err})
    staged_args = ["git", "diff", "--cached", "--binary", "--", *scope] if scope else ["git", "diff", "--cached", "--binary"]
    staged_code, staged_out, staged_err = _run_capture(staged_args, cwd=root)
    if staged_code == 0:
        staged_diff_sha = hashlib.sha256(staged_out.encode("utf-8")).hexdigest()
    elif staged_err:
        staged_diff_sha = stable_sha256({"git_cached_diff_error": staged_err})
    return {
        "commit": commit,
        "working_tree_diff_sha256": diff_sha,
        "index_diff_sha256": staged_diff_sha,
        "scope": list(scope),
    }


def code_identity(route_root: str | Path, nested_root: str | Path | None = None) -> dict[str, Any]:
    """Capture route and nested LongSplat source identity, including untracked files.

    The identity is intentionally limited to source/tests/docs/configuration.
    Generated outputs, data, caches, ``.codegraph`` and model trees are excluded.
    """

    root = Path(route_root).resolve()
    nested = Path(nested_root).resolve() if nested_root is not None else root / "third_party" / "LongSplat"
    route_scope = ("scripts", "tests", "docs/longsplat", "dev.sh")
    nested_scope = ("scene", "utils", "arguments", "train.py", "render.py", "setup.py")
    files = _identity_files(root / "scripts", category="route_source", relative_prefix="scripts/")
    files += _identity_files(root / "tests", category="route_test", relative_prefix="tests/")
    files += _identity_files(root / "docs" / "longsplat", category="route_doc", relative_prefix="docs/longsplat/")
    for name in ("dev.sh", "AGENTS.md", "CONTRIBUTING.md"):
        path = root / name
        if path.is_file():
            try:
                files.append({
                    "category": "route_config",
                    "path": name,
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                })
            except OSError:
                pass
    for scope in nested_scope:
        path = nested / scope
        if path.is_dir():
            files += _identity_files(path, category="nested_source", relative_prefix=f"nested/{scope}/")
        elif path.is_file() and path.suffix.lower() in _IDENTITY_SUFFIXES:
            try:
                files.append({
                    "category": "nested_source",
                    "path": f"nested/{scope}",
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                })
            except OSError:
                pass
    # Include any other source-like nested file as well; the explicit scope
    # above documents the primary LongSplat path, while this pass protects
    # resume against a new source module appearing elsewhere in the tree.
    files += _identity_files(nested, category="nested_source", relative_prefix="nested/")
    files = sorted({record["path"]: record for record in files}.values(), key=lambda item: item["path"])
    route_git = _git_identity(root, route_scope)
    nested_git = _git_identity(nested, (".",))
    record = {
        "schema_version": "source-tree-identity-v1",
        "route_root": str(root),
        "route_git": route_git,
        "nested_root": str(nested),
        "nested_git": nested_git,
        "files": files,
    }
    record["code_identity_sha256"] = stable_sha256(record)
    return record


def build_run_identity(
    *,
    source_video_sha256: str,
    canonical_config_sha256: str,
    tool_identity_sha256: str,
    code_identity_value: str | Mapping[str, Any],
    source_video_path: str | Path | None = None,
    source_video_size_bytes: int | None = None,
    output_root_input: str | Path | None = None,
    output_root_resolved: str | Path | None = None,
    run_root_resolved: str | Path | None = None,
) -> dict[str, Any]:
    """Build the immutable identity tuple and its aggregate digest."""

    code_value = (
        code_identity_value
        if isinstance(code_identity_value, str)
        else dict(code_identity_value)
    )
    fields = {
        "source_video_sha256": source_video_sha256,
        "canonical_config_sha256": canonical_config_sha256,
        "tool_identity_sha256": tool_identity_sha256,
        "code_identity": code_value,
    }
    if source_video_path is not None:
        fields["source_video_path"] = str(Path(source_video_path).resolve())
    if source_video_size_bytes is not None:
        fields["source_video_size_bytes"] = int(source_video_size_bytes)
    if output_root_input is not None:
        fields["output_root_input"] = str(output_root_input)
    if output_root_resolved is not None:
        fields["output_root_resolved"] = str(Path(output_root_resolved).resolve())
    if run_root_resolved is not None:
        fields["run_root_resolved"] = str(Path(run_root_resolved).resolve())
    return {
        "schema_version": RUN_IDENTITY_SCHEMA,
        **fields,
        "run_identity_sha256": stable_sha256(fields),
    }


def _assert_identity_matches(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    keys = (
        "source_video_sha256",
        "source_video_path",
        "source_video_size_bytes",
        "canonical_config_sha256",
        "tool_identity_sha256",
        "code_identity",
        "run_identity_sha256",
    )
    mismatches = [key for key in keys if existing.get(key) != expected.get(key)]
    if mismatches:
        raise ResumeMismatchError(
            "resume identity mismatch for " + ", ".join(mismatches)
        )


class RunLedger:
    """Append-only stage attempts plus a mutable summary for one run."""

    def __init__(self, run_dir: Path, identity: Mapping[str, Any], resumed: bool):
        self.run_dir = run_dir
        self.identity = dict(identity)
        self.resumed = resumed
        self.summary_path = run_dir / "run.json"
        self.summary: dict[str, Any] = {
            "schema_version": LEDGER_SCHEMA,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "status": "planned",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "identity": self.identity,
            "stages": {},
            "computed_pass": False,
            "accepted": False,
            "delivery_reachable": False,
            "gpu_invoked": False,
            "acceptance": {
                "accepted": False,
                "reason": "CPU-only first slice cannot finalize delivery",
            },
        }

    @classmethod
    def create_or_resume(
        cls,
        *,
        output_root: str | Path,
        run_id: str,
        identity: Mapping[str, Any],
    ) -> "RunLedger":
        root = resolve_output_root(output_root)
        validate_run_id(run_id)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / run_id
        if run_dir.is_symlink():
            raise ResumeMismatchError(f"run root is symlinked: {run_dir}")
        try:
            run_dir.relative_to(root)
        except ValueError as exc:
            raise ResumeMismatchError(f"run root escapes output root: {run_dir}") from exc
        identity_path = run_dir / "identity.json"
        if run_dir.exists():
            if not identity_path.is_file():
                raise ResumeMismatchError(f"existing run has no immutable identity: {run_dir}")
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
            _assert_identity_matches(existing, identity)
            ledger = cls(run_dir, existing, resumed=True)
            if ledger.summary_path.is_file():
                ledger.summary = json.loads(ledger.summary_path.read_text(encoding="utf-8"))
            # A mutable summary can retain an active stage when a previous
            # invocation died between begin_attempt and finish_attempt.  That
            # state is historical evidence, not the intent of this invocation.
            # Preserve it in an append-only inventory and clear the live
            # fields before any new stage or outer exception is attributed.
            stale_stage = ledger.summary.get("active_stage")
            stale_attempt = ledger.summary.get("active_attempt")
            stale_intent = ledger.summary.get("current_intended_stage")
            if stale_stage or stale_attempt or stale_intent:
                inventory = ledger.summary.setdefault("orphan_inventory", [])
                if not isinstance(inventory, list):
                    raise ResumeMismatchError("existing run orphan inventory is malformed")
                record = {
                    "schema_version": "orphan-inventory-v1",
                    "stage": stale_stage if isinstance(stale_stage, str) else stale_intent,
                    "attempt": stale_attempt if isinstance(stale_attempt, str) else None,
                    "status": "stale_from_prior_invocation",
                    "reason": "unclosed active stage at invocation boundary",
                }
                if not any(
                    isinstance(item, Mapping)
                    and item.get("stage") == record["stage"]
                    and item.get("attempt") == record["attempt"]
                    and item.get("status") == record["status"]
                    for item in inventory
                ):
                    inventory.append(record)
                ledger.summary.pop("active_stage", None)
                ledger.summary.pop("active_attempt", None)
                ledger.summary.pop("current_intended_stage", None)
                ledger.summary["invocation_boundary"] = {
                    "schema_version": "invocation-boundary-v1",
                    "stale_active_stage": record["stage"],
                    "stale_active_attempt": record["attempt"],
                }
                ledger._write_summary()
            return ledger
        run_dir.mkdir(parents=False)
        write_json_once(identity_path, dict(identity))
        ledger = cls(run_dir, identity, resumed=False)
        ledger._write_summary()
        return ledger

    def _write_summary(self) -> None:
        self.summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(self.summary_path, self.summary)

    def begin_attempt(self, stage: str, request: Mapping[str, Any]) -> Path:
        stage_root = self.run_dir / "stages" / stage
        stage_root.mkdir(parents=True, exist_ok=True)
        existing = sorted(path for path in stage_root.glob("attempt-*" ) if path.is_dir())
        number = len(existing) + 1
        attempt = stage_root / f"attempt-{number:04d}"
        attempt.mkdir()
        self.summary["current_intended_stage"] = stage
        self.summary["active_stage"] = stage
        self.summary["active_attempt"] = attempt.name
        self._write_summary()
        write_json_once(
            attempt / "request.json",
            {
                "schema_version": "stage-attempt-request-v1",
                "stage": stage,
                "attempt": number,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "identity": self.identity,
                "request": dict(request),
            },
        )
        return attempt

    def finish_attempt(
        self,
        *,
        stage: str,
        attempt: Path,
        status: str,
        result: Mapping[str, Any],
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        if result.get("stage_result_conflict"):
            status = "blocked"
        write_json_once(
            attempt / "result.json",
            {
                "schema_version": "stage-attempt-result-v1",
                "stage": stage,
                "status": status,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "identity": self.identity,
                "result": dict(result),
            },
        )
        write_text_once(attempt / "stdout.log", stdout)
        write_text_once(attempt / "stderr.log", stderr)
        self.summary.setdefault("stages", {}).setdefault(stage, []).append(
            {
                "attempt": attempt.name,
                "status": status,
                "result_path": str((attempt / "result.json").relative_to(self.run_dir)),
            }
        )
        self.summary["last_stage"] = stage
        self.summary.pop("active_stage", None)
        self.summary.pop("active_attempt", None)
        if self.summary.get("current_intended_stage") == stage:
            self.summary.pop("current_intended_stage", None)
        if status == "blocked":
            self._set_blocked_summary(
                stage=stage,
                error=str(result.get("reason", "stage blocked")),
                exit_code=result.get("exit_code", 2),
            )
        self._write_summary()

    def _set_blocked_summary(
        self,
        *,
        stage: str | None,
        error: str,
        exit_code: Any = 2,
    ) -> None:
        try:
            normalized_exit_code = int(exit_code)
        except (TypeError, ValueError):
            normalized_exit_code = 2
        self.summary["status"] = "blocked"
        self.summary["computed_pass"] = False
        self.summary["accepted"] = False
        self.summary["delivery_reachable"] = False
        self.summary["blocked"] = {
            "stage": stage,
            "error": str(error),
            "exit_code": normalized_exit_code,
        }
        self.summary["acceptance"] = {
            "accepted": False,
            "reason": str(error),
            "final_delivery_stage": stage,
        }

    def mark_blocked(
        self,
        *,
        stage: str | None,
        error: str,
        exit_code: Any = 2,
    ) -> None:
        """Persist an authoritative blocked terminal state for this run."""

        self._set_blocked_summary(stage=stage, error=error, exit_code=exit_code)
        self.summary["last_stage"] = stage
        self.summary.pop("active_stage", None)
        self.summary.pop("active_attempt", None)
        self.summary.pop("current_intended_stage", None)
        self._write_summary()

    def set_current_stage(self, stage: str) -> None:
        """Record the stage this invocation is about to inspect or execute."""

        if not isinstance(stage, str) or not stage:
            raise ValueError("current stage must be a non-empty string")
        self.summary["current_intended_stage"] = stage
        self._write_summary()

    def stop_cpu(self, *, status: str, computed_pass: bool, reason: str) -> None:
        """Finalize a CPU stop while making delivery unreachable in code."""

        self.summary["status"] = status
        self.summary["computed_pass"] = bool(computed_pass)
        self.summary["accepted"] = False
        self.summary["delivery_reachable"] = False
        # Preserve a prior real executor invocation when a resumed run is
        # finalized after a GPU stage.  CPU-only stops remain false because
        # the initial value is false; this is not a success claim.
        self.summary["gpu_invoked"] = bool(self.summary.get("gpu_invoked", False))
        self.summary["acceptance"] = {
            "accepted": False,
            "reason": reason,
            "final_delivery_stage": "unreachable_in_cpu_slice",
        }
        self.summary.pop("current_intended_stage", None)
        self.summary.pop("active_stage", None)
        self.summary.pop("active_attempt", None)
        self._write_summary()

    def mark_gpu_invoked(self) -> None:
        """Persist the run-level fact that a GPU executor was actually started."""

        self.summary["gpu_invoked"] = True
        self._write_summary()

    @staticmethod
    def _artifacts_valid(result: Mapping[str, Any], *, root: Path | None = None) -> bool:
        artifacts = result.get("artifacts")
        if artifacts is None:
            return True
        if not isinstance(artifacts, list):
            return False
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                return False
            path_value = artifact.get("path")
            expected_sha = artifact.get("sha256")
            if not isinstance(path_value, str):
                return False
            path = Path(path_value)
            if path.is_symlink() or not path.is_file():
                return False
            if root is not None:
                try:
                    path.resolve(strict=True).relative_to(Path(root).resolve(strict=True))
                except (OSError, ValueError):
                    return False
            if expected_sha and sha256_file(path) != expected_sha:
                return False
        return True

    def latest_attempt(self, stage: str, *, require_artifacts: bool = True) -> dict[str, Any] | None:
        """Return the latest attempt with status and an explicit reuse decision."""

        attempts = self.summary.get("stages", {}).get(stage, [])
        if not attempts:
            return None
        path = self.run_dir / attempts[-1]["result_path"]
        if not path.is_file():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        result = record.get("result", {})
        artifacts_valid = self._artifacts_valid(result, root=self.run_dir)
        return {
            "attempt": attempts[-1].get("attempt"),
            "status": record.get("status"),
            "result": result,
            "result_path": str(path),
            "artifacts_valid": artifacts_valid,
            "reusable": record.get("status") == "passed" and (not require_artifacts or artifacts_valid),
        }

    def latest_result(self, stage: str, *, require_artifacts: bool = True) -> dict[str, Any] | None:
        attempt = self.latest_attempt(stage, require_artifacts=require_artifacts)
        return None if attempt is None or not attempt["reusable"] else attempt["result"]
