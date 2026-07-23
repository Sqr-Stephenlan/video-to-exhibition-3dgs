"""Contracts for the local MASt3R low-host-memory loader patch."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_SOURCE = (
    PROJECT_ROOT
    / "third_party"
    / "LongSplat"
    / "submodules"
    / "mast3r"
    / "mast3r"
    / "model.py"
)
PATCH_PATH = (
    PROJECT_ROOT / "docs" / "longsplat" / "patches" / "mast3r_low_memory_load.patch"
)


def _load_model_function() -> ast.FunctionDef:
    module = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )


def test_mast3r_checkpoint_is_released_before_cuda_transfer():
    function = _load_model_function()
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]

    torch_load = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "torch"
        and call.func.attr == "load"
    )
    mmap_keyword = next(
        (keyword for keyword in torch_load.keywords if keyword.arg == "mmap"), None
    )
    assert mmap_keyword is not None
    assert isinstance(mmap_keyword.value, ast.Constant)
    assert mmap_keyword.value.value is True

    delete_ckpt = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Delete)
        and any(
            isinstance(target, ast.Name) and target.id == "ckpt"
            for target in node.targets
        )
    )
    collect_call = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "gc"
        and call.func.attr == "collect"
    )
    cuda_transfer = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "net"
        and call.func.attr == "to"
    )

    assert delete_ckpt.lineno < cuda_transfer.lineno
    assert collect_call.lineno < cuda_transfer.lineno


def test_nested_mast3r_patch_preserves_low_memory_loader_contract():
    patch = PATCH_PATH.read_text(encoding="utf-8")

    assert "diff --git a/mast3r/model.py b/mast3r/model.py" in patch
    assert "+import gc" in patch
    assert "+        mmap=True," in patch
    assert "+    del ckpt" in patch
    assert "+    gc.collect()" in patch
