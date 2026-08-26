from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

from .contracts import BackendIdentity


class BackendPreflightError(RuntimeError):
    """Raised when the locked backend cannot be used safely."""

    def __init__(self, message: str, *, code: str = "backend_preflight_failed") -> None:
        super().__init__(message)
        self.code = code


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise BackendPreflightError(
            f"无法读取 LongSplat 版本信息: git {args[0]}",
            code="backend_git_inspection_failed",
        )
    return result.stdout.strip()


def _gitlink(cwd: Path, path: str) -> str:
    line = _git(cwd, "ls-tree", "HEAD", "--", path)
    fields = line.split("\t", 1)[0].split()
    if len(fields) != 3 or fields[0] != "160000" or fields[1] != "commit":
        raise BackendPreflightError(
            f"缺少 LongSplat gitlink: {path}", code="backend_gitlink_missing"
        )
    return fields[2]


def verify_backend_contract(
    *,
    repo_root: str | Path,
    route_root: str | Path,
    tool_paths: Mapping[str, str | Path] | None = None,
) -> BackendIdentity:
    """Read-only verification of the locked checkout and provider identity.

    This deliberately does not call ``verify_and_materialize``: the Web API
    must never initialize, update, or reset a dirty third-party checkout.
    """

    route = Path(route_root).resolve()
    backend = Path(repo_root).resolve()
    if not backend.is_dir():
        raise BackendPreflightError("LongSplat 目录不存在", code="backend_missing")
    try:
        from scripts.longsplat.runner import LONGSPLAT_COMMIT, _LONGSPLAT_SUBMODULE_LINKS

        root_gitlink = _gitlink(route, "third_party/LongSplat")
        actual_commit = _git(backend, "rev-parse", "HEAD")
        if root_gitlink != LONGSPLAT_COMMIT:
            raise BackendPreflightError(
                "LongSplat root gitlink 与 runner lock 不一致",
                code="backend_commit_mismatch",
            )
        if actual_commit != LONGSPLAT_COMMIT:
            raise BackendPreflightError(
                "LongSplat checkout HEAD 与锁定 commit 不一致: "
                f"expected {LONGSPLAT_COMMIT}, got {actual_commit}",
                code="backend_commit_mismatch",
            )
        for path, expected in _LONGSPLAT_SUBMODULE_LINKS.items():
            if _gitlink(backend, path) != expected:
                raise BackendPreflightError(
                    f"LongSplat direct gitlink 不一致: {path}",
                    code="backend_submodule_mismatch",
                )
        status = _git(backend, "status", "--porcelain")
        if status:
            raise BackendPreflightError(
                "LongSplat checkout 有未提交修改，不能作为 locked backend",
                code="backend_dirty",
            )
        from scripts.longsplat.tool_provider import resolve_tool_provider

        _, provider = resolve_tool_provider(route, tool_paths)
        provider_identity = provider.get("provider_identity_sha256")
        if not isinstance(provider_identity, str) or not provider_identity:
            raise BackendPreflightError(
                "工具提供方 identity 缺失", code="provider_identity_missing"
            )
        return BackendIdentity(
            superproject_gitlink=root_gitlink,
            longsplat_commit=actual_commit,
            provider_identity=provider_identity,
        )
    except BackendPreflightError:
        raise
    except (ImportError, OSError, ValueError, TypeError) as exc:
        raise BackendPreflightError(str(exc)) from exc


def verify_gpu_backend(
    *,
    route_root: str | Path,
    tool_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    from scripts.longsplat.one_click import preflight_gpu_backend

    try:
        return preflight_gpu_backend(route_root=route_root, tool_paths=tool_paths)
    except Exception as exc:
        raise BackendPreflightError(str(exc), code="gpu_preflight_failed") from exc
