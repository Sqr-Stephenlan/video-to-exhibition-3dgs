from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NESTED = ROOT / "third_party" / "LongSplat"
BACKEND = Path("/home/stephenlan/workspaces/video-to-exhibition-3dgs/backend-envs/longsplat-cu128")


def _backend_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["VENV_DIR"] = str(BACKEND)
    return subprocess.run(
        ["./dev.sh", "python", "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _imports(tree: ast.AST) -> list[ast.Import | ast.ImportFrom]:
    return [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]


def test_production_modules_do_not_have_unconditional_dust3r_imports():
    colmap_tree = ast.parse((NESTED / "utils" / "colmap_utils.py").read_text(encoding="utf-8"))
    render_tree = ast.parse((NESTED / "render.py").read_text(encoding="utf-8"))
    convert_tree = ast.parse((NESTED / "convert_3dgs.py").read_text(encoding="utf-8"))

    def is_dust_import(node: ast.Import | ast.ImportFrom) -> bool:
        if isinstance(node, ast.Import):
            return any(alias.name == "dust3r" or alias.name.startswith("dust3r.") for alias in node.names)
        return bool(node.module and (node.module == "dust3r" or node.module.startswith("dust3r.")))

    assert not any(is_dust_import(node) for node in _imports(colmap_tree))
    assert not any(is_dust_import(node) for node in _imports(render_tree))
    assert not any(is_dust_import(node) for node in _imports(convert_tree))

    helper = next(
        node
        for node in colmap_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_require_dust3r_to_numpy"
    )
    assert any(isinstance(node, ast.ImportFrom) and node.module == "dust3r.utils.device" for node in ast.walk(helper))
    assert not any(isinstance(node, ast.ImportFrom) and node.module == "dust3r.utils.device" for node in _imports(colmap_tree))


def test_render_camera_text_helpers_keep_contract_without_optional_packages():
    code = r'''
import importlib.util
import os
import tempfile
from pathlib import Path

os.chdir("third_party/LongSplat")
assert importlib.util.find_spec("mast3r") is None
assert importlib.util.find_spec("dust3r") is None

from utils.colmap_utils import get_pc, save_cameras, save_imagestxt
import numpy as np

with tempfile.TemporaryDirectory() as tmp:
    focals = np.asarray([[10.0]], dtype=float)
    principal_points = np.asarray([[2.0, 1.0]], dtype=float)
    save_cameras(focals, principal_points, tmp, imgs_shape=(3, 4, 5))
    world2cam = np.eye(4, dtype=float)[None, ...]
    save_imagestxt(world2cam, tmp, np.asarray(["frame_000000"]))
    cameras = Path(tmp, "cameras.txt").read_text()
    images = Path(tmp, "images.txt").read_text()
    assert "0 PINHOLE 5 4 10.0 10.0 2.0 1.0\n" in cameras
    assert "0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0 frame_000000.png\n\n" in images

try:
    get_pc([], [])
except ImportError as exc:
    assert "optional legacy dependency" in str(exc)
    assert "external fixed-pose training/render" in str(exc)
else:
    raise AssertionError("legacy get_pc unexpectedly succeeded without DUSt3R")

print("production_optional_import_smoke=ok")
'''
    result = _backend_python(code)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "production_optional_import_smoke=ok" in result.stdout


def test_external_scene_and_convert_graph_have_no_new_optional_dependency_edges():
    scene_tree = ast.parse((NESTED / "scene" / "__init__.py").read_text(encoding="utf-8"))
    train_tree = ast.parse((NESTED / "train.py").read_text(encoding="utf-8"))
    convert_tree = ast.parse((NESTED / "convert_3dgs.py").read_text(encoding="utf-8"))

    def is_mast3r_import(node: ast.AST) -> bool:
        return isinstance(node, ast.ImportFrom) and node.module == "utils.mast3r_utils"

    assert not any(is_mast3r_import(node) for node in _imports(scene_tree))
    assert not any(is_mast3r_import(node) for node in _imports(train_tree))
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "mast3r" in ast.unparse(node).lower()
        for node in _imports(convert_tree)
    )

    # The external route's explicit import boundary remains in the existing
    # scene/training code, while the optional matcher is only a nested branch.
    source_scene = (NESTED / "scene" / "__init__.py").read_text(encoding="utf-8")
    source_train = (NESTED / "train.py").read_text(encoding="utf-8")
    assert "mast3r_global_align_skipped" in source_scene
    assert "mast3r_global_align_skipped" in source_train
