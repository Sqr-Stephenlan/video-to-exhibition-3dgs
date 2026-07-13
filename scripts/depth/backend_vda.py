"""Thin Video Depth Anything environment checks and subprocess runner."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


CHECKPOINT_NAMES = {
    ("relative", "vits"): "video_depth_anything_vits.pth",
    ("relative", "vitb"): "video_depth_anything_vitb.pth",
    ("relative", "vitl"): "video_depth_anything_vitl.pth",
    ("metric", "vits"): "metric_video_depth_anything_vits.pth",
    ("metric", "vitb"): "metric_video_depth_anything_vitb.pth",
    ("metric", "vitl"): "metric_video_depth_anything_vitl.pth",
}


def expected_checkpoint_name(depth_type: str, encoder: str) -> str:
    key = (depth_type, encoder)
    if key not in CHECKPOINT_NAMES:
        raise ValueError(f"Unsupported depth_type/encoder: {depth_type}/{encoder}")
    return CHECKPOINT_NAMES[key]


def resolve_checkpoint(repo_dir: Path, backend: dict[str, Any]) -> Path:
    configured = backend.get("checkpoint") or ""
    if configured:
        path = Path(configured)
        if path.is_absolute():
            raise ValueError("backend.checkpoint must be repository-relative")
        # Allow either repo-relative or project-relative; prefer under repo_dir/checkpoints.
        candidate = repo_dir / configured
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Configured checkpoint not found under backend repo: {configured}")
    name = expected_checkpoint_name(backend.get("depth_type", "relative"), backend["encoder"])
    return repo_dir / "checkpoints" / name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo_dir: Path) -> str | None:
    if not (repo_dir / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def doctor_backend(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    backend = config["backend"]
    repo_dir = (root / backend["repo_dir"]).resolve()
    report: dict[str, Any] = {
        "repo_dir": repo_dir.as_posix(),
        "repo_exists": repo_dir.is_dir(),
        "run_py": (repo_dir / "run.py").is_file() if repo_dir.is_dir() else False,
        "pinned_commit": backend.get("commit"),
        "actual_commit": None,
        "checkpoint": None,
        "checkpoint_exists": False,
        "checkpoint_sha256": None,
        "torch_importable": False,
        "cuda_available": False,
        "issues": [],
    }

    if not report["repo_exists"]:
        report["issues"].append(
            f"Missing backend clone at {backend['repo_dir']}. See configs/depth/backend_pin.md."
        )
        return report

    if not report["run_py"]:
        report["issues"].append("Backend repo is missing run.py")

    actual = git_head(repo_dir)
    report["actual_commit"] = actual
    pinned = str(backend.get("commit") or "")
    if actual and pinned and not actual.startswith(pinned) and pinned not in actual:
        report["issues"].append(
            f"Backend commit mismatch: expected pin starting with {pinned}, got {actual}"
        )

    try:
        ckpt = resolve_checkpoint(repo_dir, backend)
        report["checkpoint"] = ckpt.as_posix()
        report["checkpoint_exists"] = ckpt.is_file()
        if ckpt.is_file():
            report["checkpoint_sha256"] = sha256_file(ckpt)
        else:
            report["issues"].append(
                f"Missing checkpoint: {ckpt.as_posix()}. Download the vitb relative weights."
            )
    except (OSError, ValueError) as exc:
        report["issues"].append(str(exc))

    try:
        import torch  # type: ignore

        report["torch_importable"] = True
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["torch_version"] = torch.__version__
    except Exception as exc:  # noqa: BLE001 - doctor must surface any import failure
        report["issues"].append(f"torch import failed: {exc}")

    ffmpeg = shutil.which("ffmpeg")
    report["ffmpeg"] = ffmpeg
    if not ffmpeg:
        report["issues"].append("ffmpeg not found on PATH (needed to assemble frames into a temp video)")

    return report


def run_vda_on_video(
    *,
    repo_dir: Path,
    input_video: Path,
    output_dir: Path,
    backend: dict[str, Any],
    python_exe: str | None = None,
) -> subprocess.CompletedProcess[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe or sys.executable,
        str(repo_dir / "run.py"),
        "--input_video",
        str(input_video),
        "--output_dir",
        str(output_dir),
        "--encoder",
        str(backend["encoder"]),
        "--input_size",
        str(backend.get("input_size", 518)),
        "--max_res",
        str(backend.get("max_res", 1280)),
    ]
    if backend.get("depth_type") == "metric":
        cmd.append("--metric")
    if backend.get("fp16", True) is False:
        cmd.append("--fp32")
    cmd.extend(["--save_npz", "--grayscale"])

    return subprocess.run(
        cmd,
        cwd=str(repo_dir),
        check=False,
        capture_output=True,
        text=True,
    )


def write_run_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
