from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path for scripts.longsplat imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.longsplat.run_record import create as create_record, mark_complete
from scripts.longsplat.materialize_depth import materialize as materialize_depth


def project_root() -> Path:
    return _REPO_ROOT


def _find_python() -> str:
    candidates = [
        os.path.join(project_root(), "venv", "Scripts", "python.exe"),
        os.path.join(project_root(), ".venv", "Scripts", "python.exe"),
        sys.executable,
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return sys.executable


def cmd_prepare(args: argparse.Namespace) -> int:
    """Materialize depth and validate input directories."""
    root = project_root()

    if args.depth_manifest:
        manifest_path = root / args.depth_manifest
        source_path = root / args.source_path
        print(f"Materializing depth from {args.depth_manifest}")
        rc = materialize_depth(manifest_path, source_path)
        if rc != 0:
            print("ERROR: depth materialization failed")
            return rc

    images_dir = root / args.source_path / args.images
    if not images_dir.is_dir():
        print(f"ERROR: images directory not found: {images_dir}")
        return 1
    frame_count = len([f for f in os.listdir(images_dir) if not f.startswith(".")])
    print(f"Ready: {frame_count} frames in {images_dir}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Run LongSplat training."""
    root = project_root()
    python_exe = _find_python()
    train_script = root / "third_party" / "LongSplat" / "train.py"

    model_path = args.model_path
    if model_path is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        model_path = f"outputs/reconstructions/longsplat_{ts}"
    model_path = root / model_path

    if args.prepare:
        rc = cmd_prepare(args)
        if rc != 0:
            return rc

    # Build command
    cmd = [python_exe, str(train_script)]
    if args.eval:
        cmd.append("--eval")
    cmd += [
        "--source_path", str(root / args.source_path),
        "--model_path", str(model_path),
        "--images", args.images,
        "--mode", args.mode,
        "--resolution", str(args.resolution),
        "--port", str(args.port),
    ]
    if args.depth_source and args.depth_source != "mast3r":
        cmd += ["--depth_source", args.depth_source]

    env = os.environ.copy()
    env["PYTHONPATH"] = ";".join([
        str(root / "third_party" / "LongSplat"),
        str(root / "third_party" / "LongSplat" / "submodules" / "mast3r"),
        str(root / "third_party" / "LongSplat" / "submodules" / "mast3r" / "dust3r"),
    ])

    if args.dry_run:
        print(f"Would launch: {' '.join(cmd)}")
        print(f"PYTHONPATH={env['PYTHONPATH']}")
        return 0

    # Write run record
    record = create_record(
        source_path=root / args.source_path,
        model_path=model_path,
        images=args.images,
        mode=args.mode,
        resolution=args.resolution,
        depth_source=args.depth_source,
        extra_train_args=None,
    )
    save_record = record  # capture for mark_complete

    model_path.mkdir(parents=True, exist_ok=True)
    log_path = model_path / "train.log"
    log = open(log_path, "w", buffering=1)

    print(f"Launching training to {model_path}")
    print(f"Log: {log_path}")

    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"PID: {proc.pid}")

    proc.wait()
    duration = time.monotonic() - t0
    log.close()

    mark_complete(
        save_record, model_path,
        stage="training",
        exit_code=proc.returncode,
        duration_s=duration,
    )

    status = "completed" if proc.returncode == 0 else "FAILED"
    print(f"Training {status} ({duration:.0f}s)")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="LongSplat training orchestrator")
    sub = parser.add_subparsers(dest="command")

    # --- prepare ---
    prep = sub.add_parser("prepare", help="Materialize depth and validate input")
    prep.add_argument("--source-path", default="data/frames/wall_test")
    prep.add_argument("--images", default="selected/segment_0002")
    prep.add_argument("--depth-manifest", default=None,
                      help="Path to depth_manifest.json relative to repo root")

    # --- train ---
    tr = sub.add_parser("train", help="Run LongSplat training")
    tr.add_argument("--source-path", default="data/frames/wall_test")
    tr.add_argument("--model-path", default=None,
                    help="Output directory (auto-generated if omitted)")
    tr.add_argument("--images", default="selected/segment_0002")
    tr.add_argument("--mode", default="custom")
    tr.add_argument("--resolution", type=int, default=-1)
    tr.add_argument("--depth-source", default=None,
                    help="Depth supervision source (e.g. 'vda')")
    tr.add_argument("--depth-manifest", default=None,
                    help="Path to depth_manifest.json for materialization")
    tr.add_argument("--port", type=int, default=6009)
    tr.add_argument("--eval", action="store_true", default=True,
                    help="Evaluation mode (default: True)")
    tr.add_argument("--prepare", action="store_true",
                    help="Run input preparation before training")
    tr.add_argument("--dry-run", action="store_true",
                    help="Print command without executing")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "prepare":
        return cmd_prepare(args)
    elif args.command == "train":
        return cmd_train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
