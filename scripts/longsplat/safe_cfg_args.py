"""Parse generated LongSplat ``cfg_args`` without executing Python."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


class SafeCfgArgsError(ValueError):
    """Raised when a cfg_args file is not a literal Namespace expression."""


def parse_cfg_args(text: str) -> argparse.Namespace:
    """Parse exactly one ``Namespace(...)`` call containing literals only."""

    if not isinstance(text, str):
        raise SafeCfgArgsError("cfg_args must be text")
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError as exc:
        raise SafeCfgArgsError(f"invalid cfg_args syntax: {exc.msg}") from exc

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise SafeCfgArgsError("cfg_args must contain one top-level Namespace call")
    expression = tree.body[0].value
    if not isinstance(expression, ast.Call):
        raise SafeCfgArgsError("cfg_args must be a Namespace call")
    if not isinstance(expression.func, ast.Name) or expression.func.id != "Namespace":
        raise SafeCfgArgsError("cfg_args must call Namespace directly")
    if expression.args:
        raise SafeCfgArgsError("cfg_args Namespace call cannot have positional arguments")

    values: dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise SafeCfgArgsError("cfg_args Namespace call cannot use ** expansion")
        if keyword.arg in values:
            raise SafeCfgArgsError(f"duplicate cfg_args keyword: {keyword.arg}")
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
            raise SafeCfgArgsError(
                f"cfg_args value for {keyword.arg!r} is not a literal"
            ) from exc
    return argparse.Namespace(**values)


def load_cfg_args(path: str | Path) -> argparse.Namespace:
    """Read and safely parse a generated cfg_args file."""

    cfg_path = Path(path)
    return parse_cfg_args(cfg_path.read_text(encoding="utf-8"))
