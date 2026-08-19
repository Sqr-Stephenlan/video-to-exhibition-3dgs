"""Thin Video Depth Anything environment checks and subprocess runner."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from scripts.depth.config import resolve_repo_path, to_repo_relative

CHECKPOINT_NAMES = {
    ("relative", "vits"): "video_depth_anything_vits.pth",
    ("relative", "vitb"): "video_depth_anything_vitb.pth",
    ("relative", "vitl"): "video_depth_anything_vitl.pth",
    ("metric", "vits"): "metric_video_depth_anything_vits.pth",
    ("metric", "vitb"): "metric_video_depth_anything_vitb.pth",
    ("metric", "vitl"): "metric_video_depth_anything_vitl.pth",
}

DEFAULT_CHECKPOINT_SHA256 = {
    ("relative", "vitb"): "775e578e8f9431ec0496514aa466bd0a1f67c28d0f518267809f35a43c04329b",
}

# matplotlib>=3.9 removed matplotlib.cm.get_cmap; pinned VDA still uses it when
# saving visualization videos (which happens before --save_npz in run.py).
_VDA_GET_CMAP_SNIPPET = 'colormap = np.array(cm.get_cmap("inferno").colors)'
_VDA_GET_CMAP_PATCH = """try:
            colormap = np.array(cm.get_cmap("inferno").colors)
        except AttributeError:  # matplotlib >= 3.9
            from matplotlib import colormaps
            colormap = np.array(colormaps["inferno"].colors)"""


def ensure_vda_matplotlib_compat(repo_dir: Path) -> dict[str, Any]:
    """
    Idempotently patch utils/dc_utils.py for matplotlib 3.9+ colormap API.

    Returns metadata for doctor / run_record. Does not modify files outside repo_dir.
    """
    target = repo_dir / "utils" / "dc_utils.py"
    meta: dict[str, Any] = {
        "patch": "matplotlib_colormap_compat",
        "path": "utils/dc_utils.py",
        "present": target.is_file(),
        "already_patched": False,
        "applied": False,
    }
    if not target.is_file():
        meta["error"] = "utils/dc_utils.py missing"
        return meta
    text = target.read_text(encoding="utf-8")
    if "colormaps[\"inferno\"]" in text or "colormaps['inferno']" in text:
        meta["already_patched"] = True
        return meta
    if _VDA_GET_CMAP_SNIPPET not in text:
        meta["error"] = "expected cm.get_cmap snippet not found; refuse to patch blindly"
        return meta
    patched = text.replace(_VDA_GET_CMAP_SNIPPET, _VDA_GET_CMAP_PATCH, 1)
    if patched == text:
        meta["error"] = "patch replace produced no change"
        return meta
    target.write_text(patched, encoding="utf-8", newline="\n")
    meta["applied"] = True
    return meta


def expected_checkpoint_name(depth_type: str, encoder: str) -> str:
    key = (depth_type, encoder)
    if key not in CHECKPOINT_NAMES:
        raise ValueError(f"Unsupported depth_type/encoder: {depth_type}/{encoder}")
    return CHECKPOINT_NAMES[key]


def expected_vda_checkpoint_path(repo_dir: Path, backend: dict[str, Any]) -> Path:
    """Path hardcoded by pinned VDA run.py under the backend repo."""
    name = expected_checkpoint_name(backend.get("depth_type", "relative"), backend["encoder"])
    return repo_dir / "checkpoints" / name


def resolve_checkpoint(root: Path, backend: dict[str, Any]) -> Path:
    """Resolve the configured or default checkpoint; must stay inside the project root."""
    repo_dir = resolve_repo_path(root, backend["repo_dir"])
    configured = str(backend.get("checkpoint") or "").strip()
    if configured:
        return resolve_repo_path(root, configured)
    return expected_vda_checkpoint_path(repo_dir, backend)


def expected_default_checkpoint_sha256(backend: dict[str, Any]) -> str | None:
    return DEFAULT_CHECKPOINT_SHA256.get(
        (backend.get("depth_type", "relative"), backend["encoder"])
    )


@contextmanager
def stage_checkpoint_for_vda(root: Path, backend: dict[str, Any]):
    """Temporarily stage the configured/default checkpoint to the pinned VDA path, then restore it."""
    repo_dir = resolve_repo_path(root, backend["repo_dir"])
    source = resolve_checkpoint(root, backend)
    target = expected_vda_checkpoint_path(repo_dir, backend)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {to_repo_relative(root, source)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = source.resolve() != target.resolve()
    backup_path: Path | None = None
    target_existed = target.exists()
    try:
        if staged:
            if target_existed:
                fd, tmp_name = tempfile.mkstemp(
                    prefix=f"{target.stem}.backup.",
                    suffix=target.suffix,
                    dir=target.parent,
                )
                try:
                    os.close(fd)
                except OSError:
                    pass
                Path(tmp_name).unlink(missing_ok=True)
                backup_path = Path(tmp_name)
                shutil.copy2(target, backup_path)
            shutil.copy2(source, target)
        yield {
            "checkpoint_source": to_repo_relative(root, source),
            "checkpoint_vda_path": to_repo_relative(root, target),
            "checkpoint_staged": staged,
            "checkpoint_sha256": sha256_file(source),
            "checkpoint_target_restored": staged,
            "checkpoint_target_originally_present": target_existed,
        }
    finally:
        if staged:
            if backup_path is not None and backup_path.exists():
                shutil.copy2(backup_path, target)
                backup_path.unlink(missing_ok=True)
            elif not target_existed and target.exists():
                target.unlink()


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


def _path_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def sanitize_command_for_record(
    *,
    root: Path,
    command: list[str],
    temp_dir: Path | None = None,
) -> list[str]:
    """Rewrite absolute machine/temp paths to <project>/<temp> placeholders for run records."""
    root_resolved = root.resolve()
    temp_resolved = temp_dir.resolve() if temp_dir is not None else None
    sanitized: list[str] = []
    for arg in command:
        if arg.startswith("-"):
            sanitized.append(arg)
            continue
        path = Path(arg)
        if not path.is_absolute():
            sanitized.append(arg.replace("\\", "/"))
            continue
        try:
            resolved = path.resolve()
        except OSError:
            sanitized.append(f"<external>/{path.name}")
            continue
        if temp_resolved is not None and _path_under(resolved, temp_resolved):
            rel = resolved.relative_to(temp_resolved).as_posix()
            sanitized.append("<temp>" if rel == "." else f"<temp>/{rel}")
            continue
        if _path_under(resolved, root_resolved):
            sanitized.append(f"<project>/{resolved.relative_to(root_resolved).as_posix()}")
            continue
        sanitized.append(f"<external>/{resolved.name}")
    return sanitized


def doctor_backend(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    backend = config["backend"]
    report: dict[str, Any] = {
        "repo_dir": backend.get("repo_dir"),
        "repo_exists": False,
        "run_py": False,
        "pinned_commit": backend.get("commit"),
        "actual_commit": None,
        "checkpoint": None,
        "checkpoint_exists": False,
        "checkpoint_sha256": None,
        "checkpoint_vda_path": None,
        "torch_importable": False,
        "cuda_available": False,
        "issues": [],
    }

    try:
        repo_dir = resolve_repo_path(root, backend["repo_dir"])
    except ValueError as exc:
        report["issues"].append(str(exc))
        return report

    report["repo_exists"] = repo_dir.is_dir()
    report["run_py"] = (repo_dir / "run.py").is_file() if repo_dir.is_dir() else False

    if not report["repo_exists"]:
        report["issues"].append(
            f"Missing backend clone at {backend['repo_dir']}. See configs/depth/backend_pin.md."
        )
        return report

    if not report["run_py"]:
        report["issues"].append("Backend repo is missing run.py")

    patch_meta = ensure_vda_matplotlib_compat(repo_dir)
    report["local_patch_matplotlib"] = patch_meta
    if patch_meta.get("error"):
        report["issues"].append(
            "VDA matplotlib compat patch failed: "
            f"{patch_meta['error']}. See configs/depth/backend_pin.md."
        )

    actual = git_head(repo_dir)
    report["actual_commit"] = actual
    pinned = str(backend.get("commit") or "")
    if not actual:
        report["issues"].append(
            "Backend commit could not be resolved from the local clone; "
            "the pinned VDA source tree must be a git checkout with a readable HEAD."
        )
    if actual and pinned and not actual.startswith(pinned) and pinned not in actual:
        report["issues"].append(
            f"Backend commit mismatch: expected pin starting with {pinned}, got {actual}"
        )

    try:
        ckpt = resolve_checkpoint(root, backend)
        vda_ckpt = expected_vda_checkpoint_path(repo_dir, backend)
        report["checkpoint"] = to_repo_relative(root, ckpt)
        report["checkpoint_vda_path"] = to_repo_relative(root, vda_ckpt)
        report["checkpoint_exists"] = ckpt.is_file()
        if ckpt.is_file():
            report["checkpoint_sha256"] = sha256_file(ckpt)
            expected_sha = expected_default_checkpoint_sha256(backend)
            if not str(backend.get("checkpoint") or "").strip() and expected_sha:
                if report["checkpoint_sha256"] != expected_sha:
                    report["issues"].append(
                        "Default checkpoint SHA-256 mismatch: "
                        f"expected {expected_sha}, got {report['checkpoint_sha256']}"
                    )
            elif str(backend.get("checkpoint") or "").strip():
                report["custom_checkpoint"] = True
                allowed = bool(backend.get("allow_custom_checkpoint"))
                report["allow_custom_checkpoint"] = allowed
                if not allowed:
                    report["issues"].append(
                        "Non-default backend.checkpoint is configured but "
                        "backend.allow_custom_checkpoint is not true. "
                        "Set allow_custom_checkpoint: true only after explicit "
                        "reviewer confirmation of that weight source."
                    )
                else:
                    report["notes"] = list(report.get("notes") or [])
                    report["notes"].append(
                        "Using non-default backend.checkpoint with "
                        "allow_custom_checkpoint=true; default SHA-256 pin is skipped."
                    )
        else:
            ckpt_name = expected_checkpoint_name(
                str(backend.get("depth_type") or "relative"),
                str(backend["encoder"]),
            )
            report["issues"].append(
                f"Missing checkpoint: {report['checkpoint']}. "
                f"Download {ckpt_name} into third_party/Video-Depth-Anything/checkpoints/."
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
        report["issues"].append(
            "ffmpeg not found on PATH (needed to assemble frames into a temp video)"
        )

    return report


def build_vda_command(
    *,
    repo_dir: Path,
    input_video: Path,
    output_dir: Path,
    backend: dict[str, Any],
    python_exe: str | None = None,
) -> list[str]:
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
    return cmd


def run_vda_on_video(
    *,
    root: Path,
    repo_dir: Path,
    input_video: Path,
    output_dir: Path,
    backend: dict[str, Any],
    python_exe: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str], dict[str, Any]]:
    """Run VDA after staging checkpoint to the path run.py hardcodes."""
    patch_meta = ensure_vda_matplotlib_compat(repo_dir)
    if patch_meta.get("error"):
        raise RuntimeError(
            "VDA matplotlib compat patch failed: "
            f"{patch_meta['error']}. See configs/depth/backend_pin.md."
        )
    with stage_checkpoint_for_vda(root, backend) as checkpoint_meta:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_vda_command(
            repo_dir=repo_dir,
            input_video=input_video,
            output_dir=output_dir,
            backend=backend,
            python_exe=python_exe,
        )
        result = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        checkpoint_meta = {
            **checkpoint_meta,
            "local_patch_matplotlib": patch_meta,
        }
        return result, cmd, checkpoint_meta


def find_vda_depths_npz(output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("*_depths.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No VDA *_depths.npz found under {output_dir.as_posix()}"
        )
    if len(candidates) > 1:
        raise ValueError(
            "Multiple depth NPZ files found; expected a single VDA *_depths.npz. "
            f"Found: {[path.name for path in candidates]}"
        )
    return candidates[0]


def load_vda_depths_array(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        if "depths" not in data:
            raise ValueError(
                f"VDA NPZ missing 'depths' key: {npz_path.as_posix()} "
                f"(keys={list(data.keys())})"
            )
        depths = np.asarray(data["depths"])
    if depths.ndim < 3:
        raise ValueError(
            f"VDA depths array must be (N,H,W[+C]); got shape {depths.shape} "
            f"from {npz_path.as_posix()}"
        )
    return depths


def split_vda_depths_to_frame_files(
    *,
    depths: np.ndarray,
    frames: list[dict[str, Any]],
    depth_dir: Path,
    root: Path,
    depth_type: str,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """
    Write one NPZ per frame and return producer frame records.

    Each record includes ``sha256`` (64 lowercase hex of the NPZ bytes) so route
    consumers can verify materialization without an out-of-band hash patch.
    File integrity (float32 / finite / 2-D) is enforced here; VDA geometric
    quality (correlation / inlier / nRMSE) is intentionally not claimed.
    """
    if depths.shape[0] != len(frames):
        raise ValueError(
            f"VDA depths frame count {depths.shape[0]} != selected frames {len(frames)}"
        )
    depth_dir.mkdir(parents=True, exist_ok=True)
    frame_records: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        frame_id = frame["frame_id"]
        dest = depth_dir / f"{frame_id}.npz"
        if dest.is_file() and not overwrite:
            raise FileExistsError(
                f"Depth output already exists (set runtime.overwrite: true): {dest.as_posix()}"
            )
        depth = np.asarray(depths[index], dtype=np.float32)
        if depth.ndim != 2 or depth.shape[0] == 0 or depth.shape[1] == 0:
            raise ValueError(
                f"frame {frame_id}: depth must be non-empty 2-D, got shape {depth.shape}"
            )
        if not np.all(np.isfinite(depth)):
            raise ValueError(f"frame {frame_id}: depth contains NaN or Inf values")
        np.savez_compressed(dest, depth=depth)
        digest = sha256_file(dest)
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise RuntimeError(f"internal sha256 encoding error for {dest.as_posix()}")
        record: dict[str, Any] = {
            "frame_id": frame_id,
            "rgb_path": frame["path"],
            "depth_path": to_repo_relative(root, dest),
            "sha256": digest,
            "depth_type": depth_type,
            "confidence_path": None,
            "depth_index": index,
            "integrity": {
                "status": "passed",
                "dtype": "float32",
                "shape": [int(depth.shape[0]), int(depth.shape[1])],
                "finite": True,
            },
        }
        for key in (
            "segment_id",
            "timestamp_sec",
            "width",
            "height",
            "reason",
            "blur_score",
            "frame_index",
        ):
            if key in frame and frame[key] is not None:
                record[key] = frame[key]
        frame_records.append(record)
    return frame_records


def write_run_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
