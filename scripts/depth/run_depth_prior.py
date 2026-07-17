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

from scripts.depth.backend_vda import (
    doctor_backend,
    find_vda_depths_npz,
    load_vda_depths_array,
    run_vda_on_video,
    sanitize_command_for_record,
    split_vda_depths_to_frame_files,
    stage_checkpoint_for_vda,
    write_run_record,
)
from scripts.depth.config import load_config, resolve_repo_path, to_repo_relative
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
    print(f"  config: {to_repo_relative(root, config_path)}")
    print(f"  encoder: {config['backend']['encoder']} ({config['backend']['depth_type']})")
    print(f"  repo_exists: {report['repo_exists']}")
    print(f"  pinned_commit: {report['pinned_commit']}")
    print(f"  actual_commit: {report['actual_commit']}")
    print(f"  checkpoint_exists: {report['checkpoint_exists']}")
    if report.get("checkpoint"):
        print(f"  checkpoint: {report['checkpoint']}")
    if report.get("checkpoint_sha256"):
        print(f"  checkpoint_sha256: {report['checkpoint_sha256']}")
    if report.get("allow_custom_checkpoint"):
        print(f"  allow_custom_checkpoint: {report['allow_custom_checkpoint']}")
    for note in report.get("notes") or []:
        print(f"  note: {note}")
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
        # ffmpeg concat demuxer requires a trailing file entry so the last
        # duration applies; that would otherwise emit N+1 frames.
        handle.write(f"file '{frame_paths[-1].resolve().as_posix()}'\n")

    cmd = build_ffmpeg_concat_command(
        ffmpeg=ffmpeg,
        list_file=list_file,
        output_video=output_video,
        frame_count=len(frame_paths),
    )
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def build_ffmpeg_concat_command(
    *,
    ffmpeg: str,
    list_file: Path,
    output_video: Path,
    frame_count: int,
) -> list[str]:
    """Build ffmpeg concat command that emits exactly frame_count frames."""
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    return [
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
        # Truncate the duplicated trailing concat entry so VDA sees N frames,
        # matching selected_frames / strict positional depth mapping.
        "-frames:v",
        str(frame_count),
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]


def cmd_run(config_path: Path, dry_run: bool = False) -> int:
    root = project_root()
    config = load_config(config_path)
    backend = config["backend"]
    io = config["io"]
    runtime = config.get("runtime") or {}

    config_rel = to_repo_relative(root, config_path)
    frames_manifest_rel = io["frames_manifest"]
    frames_manifest_path = resolve_repo_path(root, frames_manifest_rel)
    if not frames_manifest_path.is_file():
        example = root / "configs" / "depth" / "frames_manifest.example.json"
        print(
            "frames_manifest not found. Place selected frames and a manifest first.\n"
            f"  expected: {frames_manifest_rel}\n"
            f"  example schema: {to_repo_relative(root, example)}"
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
        print(f"  depth_dir: {to_repo_relative(root, depth_dir)}")
        print(f"  depth_manifest: {to_repo_relative(root, depth_manifest_path)}")
        return 0

    started_at = datetime.now(timezone.utc).isoformat()
    command_record: list[str] = []
    result_code = 1
    stdout_tail = ""
    stderr_tail = ""
    vda_npz_rel: str | None = None
    frame_count_written = 0
    checkpoint_meta: dict = {}

    try:
        with tempfile.TemporaryDirectory(prefix="depth_prior_") as tmp:
            tmp_dir = Path(tmp)
            temp_video = tmp_dir / "input.mp4"
            vda_out = tmp_dir / "vda_out"
            _assemble_temp_video(frame_paths, float(runtime.get("target_fps", 5)), temp_video)
            result, command, checkpoint_meta = run_vda_on_video(
                root=root,
                repo_dir=repo_dir,
                input_video=temp_video,
                output_dir=vda_out,
                backend=backend,
            )
            command_record = sanitize_command_for_record(
                root=root,
                command=command,
                temp_dir=tmp_dir,
            )
            result_code = result.returncode
            stdout_tail = (result.stdout or "")[-4000:]
            stderr_tail = (result.stderr or "")[-4000:]
            if result.returncode != 0:
                print("Video Depth Anything failed. See run_record for stdout/stderr tails.")
                print(to_repo_relative(root, run_record_path))
                return result.returncode

            # VDA writes one *_depths.npz with depths shaped (N,H,W).
            vda_npz = find_vda_depths_npz(vda_out)
            depths = load_vda_depths_array(vda_npz)
            # Keep a repo-relative note of the source artifact name only (temp path is not retained).
            vda_npz_rel = vda_npz.name
            frame_records = split_vda_depths_to_frame_files(
                depths=depths,
                frames=frames,
                depth_dir=depth_dir,
                root=root,
                depth_type=backend["depth_type"],
            )
            frame_count_written = len(frame_records)
            depth_manifest = build_depth_manifest(
                frames_manifest=frames_manifest,
                frame_records=frame_records,
                backend={
                    **backend,
                    "commit": doctor.get("actual_commit") or backend.get("commit"),
                },
                depth_type=backend["depth_type"],
                frames_manifest_path=frames_manifest_rel,
            )
            save_json(depth_manifest_path, depth_manifest)
            print(f"wrote {frame_count_written} depth maps")
            print(f"depth_manifest: {to_repo_relative(root, depth_manifest_path)}")
            return 0
    finally:
        write_run_record(
            run_record_path,
            {
                "module": "depth-prior",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "config": config_rel,
                "frames_manifest": frames_manifest_rel,
                "backend_name": backend.get("name"),
                "backend_repo_dir": backend.get("repo_dir"),
                "backend_commit": doctor.get("actual_commit"),
                "encoder": backend["encoder"],
                "depth_type": backend["depth_type"],
                "checkpoint": checkpoint_meta.get("checkpoint_source") or doctor.get("checkpoint"),
                "checkpoint_vda_path": checkpoint_meta.get("checkpoint_vda_path")
                or doctor.get("checkpoint_vda_path"),
                "checkpoint_staged": checkpoint_meta.get("checkpoint_staged"),
                "checkpoint_target_restored": checkpoint_meta.get("checkpoint_target_restored"),
                "checkpoint_target_originally_present": checkpoint_meta.get(
                    "checkpoint_target_originally_present"
                ),
                "checkpoint_sha256": checkpoint_meta.get("checkpoint_sha256")
                or doctor.get("checkpoint_sha256"),
                "local_patch_matplotlib": checkpoint_meta.get("local_patch_matplotlib")
                or doctor.get("local_patch_matplotlib"),
                "command": command_record,
                "random_seed": None,
                "command_returncode": result_code,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "vda_depths_npz": vda_npz_rel,
                "outputs": {
                    "depth_dir": to_repo_relative(root, depth_dir),
                    "depth_manifest": to_repo_relative(root, depth_manifest_path),
                    "run_record": to_repo_relative(root, run_record_path),
                    "frame_count": frame_count_written,
                },
            },
        )


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
    config_path = resolve_repo_path(project_root(), args.config)
    if args.command == "doctor":
        return cmd_doctor(config_path)
    if args.command == "run":
        return cmd_run(config_path, dry_run=bool(args.dry_run))
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
