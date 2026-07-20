"""Launch LongSplat training with VDA depth prior for wall_test."""
import subprocess
import sys
import os
import time

os.chdir(r"D:\video-to-exhibition-3dgs")

env = os.environ.copy()
env["PYTHONPATH"] = "third_party/LongSplat;third_party/LongSplat/submodules/mast3r;third_party/LongSplat/submodules/mast3r/dust3r"

model_dir = "outputs/wall_test/longsplat_segment_0002_vda_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
os.makedirs(model_dir, exist_ok=True)

log_path = os.path.join(model_dir, "train.log")
log = open(log_path, "w", buffering=1)

cmd = [
    r"D:\video-to-exhibition-3dgs\venv\Scripts\python.exe",
    "third_party/LongSplat/train.py",
    "--eval",
    "--source_path", "data/frames/wall_test",
    "--model_path", model_dir,
    "--images", "selected/segment_0002",
    "--mode", "custom",
    "--resolution", "-1",
    "--depth_source", "vda",
]

print(f"Launching to {model_dir}")
print(f"Log: {log_path}")
print(f"Command: {' '.join(cmd)}")

proc = subprocess.Popen(
    cmd,
    stdout=log, stderr=subprocess.STDOUT,
    env=env,
)

print(f"Launched PID: {proc.pid}")
log.close()
print("Done. Check log for progress.")
