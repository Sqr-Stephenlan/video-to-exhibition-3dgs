from __future__ import annotations

from pathlib import Path

import pytest

import scripts.viewer.api.preflight as preflight
from scripts.viewer.api.preflight import BackendPreflightError, verify_backend_contract


def test_backend_contract_rejects_a_missing_locked_submodule(tmp_path: Path):
    with pytest.raises(BackendPreflightError, match="LongSplat"):
        verify_backend_contract(repo_root=tmp_path, route_root=tmp_path)


def test_backend_contract_reports_a_checkout_commit_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    backend = tmp_path / "LongSplat"
    backend.mkdir()

    from scripts.longsplat.runner import LONGSPLAT_COMMIT

    monkeypatch.setattr(
        preflight,
        "_gitlink",
        lambda _cwd, _path: LONGSPLAT_COMMIT,
    )
    monkeypatch.setattr(
        preflight,
        "_git",
        lambda _cwd, *args: (
            "19750775a9d19f30aa05a8333c4c6c231b2d5f4a"
            if args == ("rev-parse", "HEAD")
            else ""
        ),
    )

    with pytest.raises(
        BackendPreflightError,
        match="LongSplat checkout HEAD 与锁定 commit 不一致",
    ):
        verify_backend_contract(repo_root=backend, route_root=tmp_path)
