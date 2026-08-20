from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.longsplat import authority_manifest as am
from scripts.longsplat.conversion_executor import execute_evaluation
from scripts.longsplat.evaluate_converted_ply import main as evcp_main
from tests.unit.test_authority_manifest import _fixture
from tests.unit.test_conversion_executor import _write_conversion_evidence


class _ManifestLoaded(Exception):
    """Raised by the sentinel wrapper right after a successful manifest load."""

    def __init__(self, containment_root: object) -> None:
        self.containment_root = containment_root
        super().__init__("manifest loaded")


def _evcp_verbose_argv(manifest_path: Path, run_root: Path, source_path: Path, *, with_containment: bool) -> list[str]:
    argv = [
        "--authority-manifest", str(manifest_path),
    ]
    if with_containment:
        argv += ["--containment-root", str(run_root)]
    argv += [
        "--ply", str(run_root / "point_cloud.ply"),
        "--model-path", str(run_root / "conv_model_snapshot"),
        "--source-path", str(source_path),
        "--output", str(run_root / "out.json"),
        "--contact-sheet", str(run_root / "cs.png"),
        "--iteration", "30000",
    ]
    return argv


def test_evcp_main_accepts_external_manifest_with_containment_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive: with --containment-root=<run root>, an external (non-worktree)
    authority manifest whose bound artifacts all live under the run root loads
    successfully and main() never raises 'outside route root'."""
    _manifest, manifest_path, route = _fixture(tmp_path, "cp", camera_names=["a", "b"], width=16, height=12)
    run_root = manifest_path.parent
    source_path = _manifest["training_input"]["path"]

    real_load = am.load_authority_manifest
    called: dict[str, object] = {}

    def wrapped_load(path, *, route_root=None, containment_root=None):
        called["route_root"] = route_root
        called["containment_root"] = containment_root
        result = real_load(path, route_root=route_root, containment_root=containment_root)
        # Stop main() immediately after a real, successful manifest load so the
        # GPU render body is never reached in this CPU-only test.
        raise _ManifestLoaded(str(containment_root))

    monkeypatch.setattr("scripts.longsplat.evaluate_converted_ply.load_authority_manifest", wrapped_load)

    with pytest.raises(_ManifestLoaded) as eval_info:
        evcp_main(_evcp_verbose_argv(manifest_path, run_root, source_path, with_containment=True))

    assert eval_info.value.containment_root == str(run_root.resolve())
    assert called["route_root"] is None
    # main() routed to containment_root (not the worktree route_root).
    assert Path(str(called["containment_root"])).resolve() == run_root.resolve()


def test_evcp_main_rejects_external_manifest_without_containment_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Negative: without --containment-root the historical worktree route_root
    guard is preserved, so an external manifest is still refused - the safety
    contract is not weakened."""
    _manifest, manifest_path, route = _fixture(tmp_path, "cp-neg", camera_names=["a", "b"], width=16, height=12)
    run_root = manifest_path.parent
    source_path = _manifest["training_input"]["path"]

    real_load = am.load_authority_manifest

    def wrapped_load(path, *, route_root=None, containment_root=None):
        # No sentinel here: the real loader raises 'outside route root' and must
        # propagate so main() turns it into parser.error -> SystemExit.
        return real_load(path, route_root=route_root, containment_root=containment_root)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("scripts.longsplat.evaluate_converted_ply.load_authority_manifest", wrapped_load)
    try:
        with pytest.raises(SystemExit):
            evcp_main(_evcp_verbose_argv(manifest_path, run_root, source_path, with_containment=False))
        captured = capsys.readouterr()
        assert "authority manifest is outside route root" in captured.err
    finally:
        monkeypatch.undo()


def test_evcp_main_real_containment_root_accepts_manifest(tmp_path: Path) -> None:
    """Direct real-loader proof: the external manifest validates under
    containment_root=<run root> and is rejected under the worktree route_root."""
    _manifest, manifest_path, route = _fixture(tmp_path, "cp-real", camera_names=["a", "b"], width=16, height=12)
    run_root = manifest_path.parent
    # containment_root accepts it (all bound artifacts under run root)
    ok = am.load_authority_manifest(manifest_path, containment_root=run_root)
    assert ok["manifest_path"] == str(manifest_path.resolve())
    # worktree route_root still rejects an external manifest (safety preserved)
    worktree = Path("/home/stephenlan/workspaces/video-to-exhibition-3dgs/worktrees/longsplat-route")
    with pytest.raises(am.AuthorityManifestError, match="outside route root"):
        am.load_authority_manifest(manifest_path, route_root=worktree)


def test_execute_evaluation_dry_run_adds_containment_root_to_argv(tmp_path: Path) -> None:
    _manifest, manifest_path, route = _fixture(tmp_path, "eval", camera_names=["a", "b"], width=16, height=12)
    run_root = manifest_path.parent
    conversion_root = run_root / "conversion-attempt"
    conversion_root.mkdir(parents=True)
    _write_conversion_evidence(conversion_root)
    eval_root = run_root / "eval-attempt"
    eval_root.mkdir(parents=True)

    result = execute_evaluation(
        authority_manifest_path=manifest_path,
        evidence_root=eval_root,
        route_root=route,
        containment_root=run_root,
        conversion_evidence_root=conversion_root,
        dry_run=True,
        validate_only=False,
    )
    argv = result["command"]
    assert "--containment-root" in argv
    assert argv[argv.index("--containment-root") + 1] == str(run_root.resolve())
    # request.json argv stays in sync with the command.
    assert result["request"]["argv"] == argv


def test_execute_evaluation_validate_only_adds_containment_root_without_spawning(tmp_path: Path) -> None:
    _manifest, manifest_path, route = _fixture(tmp_path, "eval-vo", camera_names=["a", "b"], width=16, height=12)
    run_root = manifest_path.parent
    conversion_root = run_root / "conversion-attempt"
    conversion_root.mkdir(parents=True)
    _write_conversion_evidence(conversion_root)
    eval_root = run_root / "eval-attempt"
    eval_root.mkdir(parents=True)

    class _NeverRun:
        def __call__(self, *_a, **_k):
            raise AssertionError("child must not spawn for validate_only")

    result = execute_evaluation(
        authority_manifest_path=manifest_path,
        evidence_root=eval_root,
        route_root=route,
        containment_root=run_root,
        conversion_evidence_root=conversion_root,
        dry_run=False,
        validate_only=True,
        runner=_NeverRun(),
    )
    argv = result["command"]
    assert "--containment-root" in argv
    assert argv[argv.index("--containment-root") + 1] == str(run_root.resolve())


def test_execute_evaluation_without_containment_root_omits_flag(tmp_path: Path) -> None:
    _manifest, manifest_path, route = _fixture(tmp_path, "eval-none", camera_names=["a", "b"], width=16, height=12)
    # No containment_root: evidence paths resolve under route outputs.
    conversion_root = route / "outputs" / "eval-none" / "conversion-attempt"
    conversion_root.mkdir(parents=True)
    _write_conversion_evidence(conversion_root)
    eval_root = route / "outputs" / "eval-none" / "eval-attempt"
    eval_root.mkdir(parents=True)

    result = execute_evaluation(
        authority_manifest_path=manifest_path,
        evidence_root=eval_root,
        route_root=route,
        conversion_evidence_root=conversion_root,
        dry_run=True,
        validate_only=False,
    )
    argv = result["command"]
    assert "--containment-root" not in argv
