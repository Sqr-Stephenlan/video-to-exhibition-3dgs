#!/usr/bin/env python3
"""Depth-prior CLI: environment doctor and Video Depth Anything orchestration."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.depth.backend_vda import doctor_backend, run_vda_on_video, write_run_record
from scripts.depth.config import load_config, resolve_repo_path
from scripts.depth.manifest import (
    build_depth_manifest,
    load_json,
    save_json,
    selected_frames,
)


def project_root() -> Path:
    return ROOT


def cmd_doctor(config_path: Path) -> int:
    root = project_root()
    config = load_config(config_path)
    report = doctor_backend(root, config)
    print("depth-prior doctor")
    print(f"  config: {config_path.as_posix()}")
    print(f"  encoder: {config['backend']['encoder']} ({config['backend']['depth_type']})")
    print(f"  repo_exists: {report['repo_exists']}")
    print(f"  pinned_commit: {report['pinned_commit']}")
    print(f"  actual_commit: {report['actual_commit']}")
    print(f"  checkpoint_exists: {report['checkpoint_exists']}")
    if report.get("checkpoint"):
        print(f"  checkpoint: {report['checkpoint']}")
    if report.get("checkpoint_sha256"):
        print(f"  checkpoint_sha256: {report['checkpoint_sha256']}")
    print(f"  torch_importable: {report['torch_importable']}")
    print(f"  cuda_available: {report['cuda_available']}")
    print(f"  ffmpeg: {report.get('ffmpeg')}")
    if report["issues"]:
        print("issues:")
        for issue in report["issues"]:
            print(f"  - {issue}")
        return 1
    print("status: ok (environment ready; sample frames still required for run)")
    return 0


def _assemble_temp_video(frame_paths: list[Path], fps: float, output_video: Path) -> None:
    if not frame_paths:
        raise ValueError("No frame paths to assemble")
    missing = [path for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing frame files (sample frames not available yet):\n  "
            + "\n  ".join(path.as_posix() for path in missing[:5])
        )
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to assemble frames into a temp video")

    list_file = output_video.with_suffix(".txt")
    with list_file.open("w", encoding="utf-8") as handle:
        duration = 1.0 / float(fps)
        for path in frame_paths:
            handle.write(f"file '{path.resolve().as_posix()}'\n")
            handle.write(f"duration {duration:.6f}\n")
        handle.write(f"file '{frame_paths[-1].resolve().as_posix()}'\n")

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vsync",
        "vfr",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def cmd_run(config_path: Path, dry_run: bool = False) -> int:
    root = project_root()
    config = load_config(config_path)
    backend = config["backend"]
    io = config["io"]
    runtime = config.get("runtime") or {}

    frames_manifest_path = resolve_repo_path(root, io["frames_manifest"])
    if not frames_manifest_path.is_file():
        example = root / "configs" / "depth" / "frames_manifest.example.json"
        print(
            "frames_manifest not found. Place selected frames and a manifest first.\n"
            f"  expected: {io['frames_manifest']}\n"
            f"  example schema: {example.as_posix()}"
        )
        return 2

    doctor = doctor_backend(root, config)
    if doctor["issues"] and not dry_run:
        print("Environment is not ready:")
        for issue in doctor["issues"]:
            print(f"  - {issue}")
        return 1

    frames_manifest = load_json(frames_manifest_path)
    frames = selected_frames(frames_manifest)
    frame_paths = [resolve_repo_path(root, frame["path"]) for frame in frames]

    depth_dir = resolve_repo_path(root, io["depth_dir"])
    depth_manifest_path = resolve_repo_path(root, io["depth_manifest"])
    run_record_path = resolve_repo_path(root, io["run_record"])
    repo_dir = resolve_repo_path(root, backend["repo_dir"])

    if dry_run:
        print("dry-run ok")
        print(f"  selected_frames: {len(frames)}")
        print(f"  depth_dir: {depth_dir.as_posix()}")
        print(f"  depth_manifest: {depth_manifest_path.as_posix()}")
        return 0

    with tempfile.TemporaryDirectory(prefix="depth_prior_") as tmp:
        tmp_dir = Path(tmp)
        temp_video = tmp_dir / "input.mp4"
        vda_out = tmp_dir / "vda_out"
        _assemble_temp_video(frame_paths, float(runtime.get("target_fps", 5)), temp_video)
        result = run_vda_on_video(
            repo_dir=repo_dir,
            input_video=temp_video,
            output_dir=vda_out,
            backend=backend,
        )
        write_run_record(
            run_record_path,
            {
                "module": "depth-prior",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "config": config_path.as_posix(),
                "backend_commit": doctor.get("actual_commit"),
                "encoder": backend["encoder"],
                "depth_type": backend["depth_type"],
                "command_returncode": result.returncode,
                "stdout_tail": (result.stdout or "")[-4000:],
                "stderr_tail": (result.stderr or "")[-4000:],
                "checkpoint_sha256": doctor.get("checkpoint_sha256"),
            },
        )
        if result.returncode != 0:
            print("Video Depth Anything failed. See run_record for stdout/stderr tails.")
            print(run_record_path.as_posix())
            return result.returncode

        # VDA writes NPZ under output_dir; map by sorted order to selected frames.
        npz_files = sorted(vda_out.rglob("*.npz"))
        if len(npz_files) < len(frames):
            print(
                f"VDA produced {len(npz_files)} depth files for {len(frames)} frames; "
                "refusing to write a partial depth_manifest."
            )
            return 3

        depth_dir.mkdir(parents=True, exist_ok=True)
        frame_records = []
        for frame, npz_path in zip(frames, npz_files[: len(frames)], strict=True):
            dest = depth_dir / f"{frame['frame_id']}.npz"
            shutil.copy2(npz_path, dest)
            frame_records.append(
                {
                    "frame_id": frame["frame_id"],
                    "rgb_path": frame["path"],
                    "depth_path": dest.relative_to(root).as_posix(),
                    "depth_type": backend["depth_type"],
                    "confidence_path": None,
                }
            )

        depth_manifest = build_depth_manifest(
            frames_manifest=frames_manifest,
            frame_records=frame_records,
            backend={**backend, "commit": doctor.get("actual_commit") or backend.get("commit")},
            depth_type=backend["depth_type"],
        )
        save_json(depth_manifest_path, depth_manifest)
        print(f"wrote {len(frame_records)} depth maps")
        print(f"depth_manifest: {depth_manifest_path.as_posix()}")
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exhibition 3DGS depth-prior orchestration")
    parser.add_argument(
        "--config",
        default="configs/depth/default_vitb.yaml",
        help="Repository-relative YAML config (default: vitb relative)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check VDA clone, vitb weights, torch, ffmpeg")
    run_parser = sub.add_parser("run", help="Run depth prior from frames_manifest")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and manifest only; do not call VDA",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = project_root() / args.config
    if args.command == "doctor":
        return cmd_doctor(config_path)
    if args.command == "run":
        return cmd_run(config_path, dry_run=bool(args.dry_run))
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
