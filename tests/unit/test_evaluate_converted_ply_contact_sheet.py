from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

import scripts.longsplat.evaluate_converted_ply as evaluator


@pytest.mark.parametrize("width,height", [(31, 17), (17, 31), (29, 23)])
def test_contact_sheet_streams_all_views_with_aspect_preserving_tiles(
    tmp_path: Path,
    width: int,
    height: int,
) -> None:
    rows: list[dict[str, object]] = []
    for ordinal in range(7):
        render_path = tmp_path / f"render-{ordinal}.png"
        gt_path = tmp_path / f"gt-{ordinal}.png"
        render_color = (10 + ordinal, 30 + ordinal, 50 + ordinal)
        gt_color = (70 + ordinal, 90 + ordinal, 110 + ordinal)
        assert cv2.imwrite(
            str(render_path),
            np.full((height, width, 3), render_color, dtype=np.uint8),
        )
        assert cv2.imwrite(
            str(gt_path),
            np.full((height, width, 3), gt_color, dtype=np.uint8),
        )
        rows.append({"render_path": str(render_path), "gt_path": str(gt_path)})

    output = tmp_path / "contact-sheet.png"
    layout = evaluator._write_contact_sheet_from_pngs(
        rows=rows,
        output=output,
        source_dimensions=(width, height),
        columns=3,
        tile_width=20,
        max_tile_height=20,
        padding=2,
    )

    sheet = cv2.imread(str(output), cv2.IMREAD_COLOR)
    assert sheet is not None
    tile_height = min(20, max(1, round(20 * height / width)))
    assert sheet.shape[:2] == (
        2 + 3 * (tile_height + 2),
        2 + 3 * (40 + 2),
    )
    assert layout["source_count"] == len(rows)
    assert layout["columns"] == 3
    assert layout["tile_width"] == 20
    assert layout["tile_height"] == tile_height
    assert layout["resident_full_resolution_frame_max"] <= 2
    assert layout["source_dimensions"] == {"width": width, "height": height}
    assert cv2.imread(str(output), cv2.IMREAD_UNCHANGED) is not None

    for ordinal in range(len(rows)):
        row_index, column_index = divmod(ordinal, 3)
        x = 2 + column_index * (40 + 2)
        y = 2 + row_index * (tile_height + 2)
        assert tuple(int(value) for value in sheet[y + tile_height // 2, x + 10]) == (
            10 + ordinal,
            30 + ordinal,
            50 + ordinal,
        )
        assert tuple(int(value) for value in sheet[y + tile_height // 2, x + 20 + 10]) == (
            70 + ordinal,
            90 + ordinal,
            110 + ordinal,
        )

    first_tile = sheet[2 : 2 + tile_height, 2 : 2 + 20]
    non_white = np.any(first_tile != 255, axis=2)
    ys, xs = np.where(non_white)
    fitted_width = int(xs.max() - xs.min() + 1)
    fitted_height = int(ys.max() - ys.min() + 1)
    assert fitted_width / fitted_height == pytest.approx(width / height, rel=0.08)


def test_contact_sheet_uses_png_streaming_without_full_resolution_pairs() -> None:
    source = inspect.getsource(evaluator)
    assert "make_grid" not in source
    assert "pairs.append" not in source
    assert "torch.cat([rendered.cpu(), target.cpu()], dim=2)" not in source
    assert "_write_contact_sheet_from_pngs" in source
