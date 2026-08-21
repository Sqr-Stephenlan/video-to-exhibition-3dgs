"""Render and score a standard 3DGS PLY against all ordered training views.

The evaluator deliberately loads the converted PLY into LongSplat's standard
3DGS rasterizer instead of loading the native LongSplat checkpoint.  A supplied
authority manifest binds the complete camera order and dimensions; no random
or evenly spaced subset is allowed.  The current segment has no held-out
camera set, so these are training-view reconstruction metrics, not
generalization metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.longsplat.authority_manifest import (
    AuthorityManifestError,
    conversion_profile,
    load_authority_manifest,
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_converted_ply(path: Path, gaussians: GaussianModel) -> None:
    """Replace native model tensors with tensors decoded from a 3DGS PLY."""
    import numpy as np
    import torch
    from plyfile import PlyData
    from torch import nn

    vertex = PlyData.read(str(path))["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
    required |= {"rot_0", "rot_1", "rot_2", "rot_3"}
    required |= {"f_dc_0", "f_dc_1", "f_dc_2"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"converted PLY missing required fields: {missing}")

    def column(name: str) -> np.ndarray:
        return np.asarray(vertex[name], dtype=np.float32)

    count = len(vertex.data)
    xyz = np.stack([column(axis) for axis in ("x", "y", "z")], axis=1)
    f_dc = np.stack([column(f"f_dc_{i}") for i in range(3)], axis=1)
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if len(rest_names) % 3 != 0:
        raise ValueError(f"f_rest field count is not divisible by 3: {len(rest_names)}")
    f_rest = (
        np.stack([column(name) for name in rest_names], axis=1)
        .reshape(count, 3, -1)
        .transpose(0, 2, 1)
    )
    opacity = column("opacity")[:, None]
    scaling = np.stack([column(f"scale_{i}") for i in range(3)], axis=1)
    rotation = np.stack([column(f"rot_{i}") for i in range(4)], axis=1)

    device = torch.device("cuda")
    gaussians._xyz = nn.Parameter(torch.from_numpy(xyz).to(device), requires_grad=False)
    gaussians._features_dc = nn.Parameter(
        torch.from_numpy(f_dc[:, None, :]).to(device), requires_grad=False
    )
    gaussians._features_rest = nn.Parameter(
        torch.from_numpy(f_rest).to(device), requires_grad=False
    )
    gaussians._opacity = nn.Parameter(torch.from_numpy(opacity).to(device), requires_grad=False)
    gaussians._scaling = nn.Parameter(torch.from_numpy(scaling).to(device), requires_grad=False)
    gaussians._rotation = nn.Parameter(torch.from_numpy(rotation).to(device), requires_grad=False)
    gaussians.active_sh_degree = 3


def _load_args(model_path: Path, source_path: Path) -> argparse.Namespace:
    """Read the checkpoint's camera/model args and redirect paths safely."""
    from scripts.longsplat.safe_cfg_args import load_cfg_args

    cfg = load_cfg_args(model_path / "cfg_args")
    cfg.source_path = str(source_path.resolve())
    cfg.model_path = str(model_path.resolve())
    cfg.load_pose = True
    cfg.eval = False
    return cfg


def _write_contact_sheet_from_pngs(
    *,
    rows: Sequence[Mapping[str, object]],
    output: Path,
    source_dimensions: tuple[int, int] | None,
    columns: int = 4,
    tile_width: int = 320,
    max_tile_height: int = 320,
    padding: int = 4,
) -> dict[str, object]:
    """Build the render/GT sheet by streaming the already-written PNGs.

    The evaluator's metrics and full-resolution PNGs remain unchanged.  The
    contact sheet is only presentation evidence, so it uses fixed-size
    letterboxed thumbnails and keeps no full-resolution image collection in
    memory.  A row contains the rendered PNG followed by its GT PNG, matching
    the former per-camera ``cat(render, target)`` ordering.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - route environment supplies cv2
        raise RuntimeError(f"OpenCV/numpy are required for contact sheet: {exc}") from exc

    if not rows:
        raise ValueError("cannot build a contact sheet without evaluated views")
    if columns <= 0 or tile_width <= 0 or max_tile_height <= 0 or padding < 0:
        raise ValueError("contact sheet layout dimensions must be positive")
    columns = min(columns, len(rows))

    if source_dimensions is not None:
        source_width, source_height = source_dimensions
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"invalid source dimensions: {source_dimensions!r}")
        tile_height = min(
            max_tile_height,
            max(1, round(tile_width * source_height / source_width)),
        )
    else:
        source_width = source_height = None
        tile_height = min(max_tile_height, 240)

    pair_width = tile_width * 2
    row_count = (len(rows) + columns - 1) // columns
    canvas_width = padding + columns * (pair_width + padding)
    canvas_height = padding + row_count * (tile_height + padding)
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    def fit_into_tile(image: np.ndarray) -> np.ndarray:
        image_height, image_width = image.shape[:2]
        scale = min(tile_width / image_width, tile_height / image_height)
        resized_width = max(1, round(image_width * scale))
        resized_height = max(1, round(image_height * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        tile = np.full((tile_height, tile_width, 3), 255, dtype=np.uint8)
        offset_x = (tile_width - resized_width) // 2
        offset_y = (tile_height - resized_height) // 2
        tile[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
        del resized
        return tile

    for ordinal, row in enumerate(rows):
        row_index, column_index = divmod(ordinal, columns)
        x = padding + column_index * (pair_width + padding)
        y = padding + row_index * (tile_height + padding)
        for side, key in enumerate(("render_path", "gt_path")):
            path = Path(str(row[key]))
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"contact sheet cannot decode {key}: {path}")
            tile = fit_into_tile(image)
            tile_x = x + side * tile_width
            canvas[y : y + tile_height, tile_x : tile_x + tile_width] = tile
            del image, tile

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise ValueError(f"cannot write contact sheet: {output}")
    del canvas
    return {
        "schema": "converted-evaluator-contact-sheet-v1",
        "policy": "cpu-streaming-png-thumbnails-v1",
        "ordering": "camera_order_row_major_render_then_ground_truth",
        "source_count": len(rows),
        "columns": columns,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "padding": padding,
        "source_dimensions": (
            None
            if source_width is None or source_height is None
            else {"width": source_width, "height": source_height}
        ),
        "source": "per_view_png_on_disk",
        "resident_full_resolution_frame_max": 1,
        "path": str(output),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a converted standard 3DGS PLY")
    parser.add_argument("--authority-manifest", type=Path)
    parser.add_argument(
        "--containment-root",
        type=Path,
        help="optional external run/evidence containment root; when supplied the authority "
        "manifest and all bound artifacts must live under it (used for external delivery roots)",
    )
    parser.add_argument("--ply", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contact-sheet", required=True, type=Path)
    parser.add_argument(
        "--max-views",
        type=int,
        help="deprecated compatibility flag; when supplied it must equal the full training-camera count",
    )
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists():
        print(f"ERROR: output already exists: {output}", file=sys.stderr)
        raise SystemExit(1)
    if args.max_views is not None and args.max_views <= 0:
        parser.error("--max-views must be positive")

    authority = None
    if args.authority_manifest is not None:
        try:
            # When an explicit external containment root is supplied (delivery
            # run_dir outside the worktree), validate the manifest and every
            # bound artifact against that root.  Otherwise keep the historical
            # worktree route_root bound so an external manifest is still refused
            # (the safety contract is never weakened).
            if args.containment_root is not None:
                authority = load_authority_manifest(
                    args.authority_manifest,
                    containment_root=args.containment_root,
                )
            else:
                authority = load_authority_manifest(
                    args.authority_manifest,
                    route_root=Path(__file__).resolve().parents[2],
                )
        except AuthorityManifestError as exc:
            parser.error(str(exc))
        expected_source = Path(str(authority["manifest"]["training_input"]["path"])).resolve()
        if args.source_path.resolve() != expected_source:
            parser.error("--source-path differs from the authority training input")
        if args.iteration is not None and args.iteration != authority["profile"]["checkpoint_iteration"]:
            parser.error("--iteration differs from the authority conversion profile")
        iteration = int(authority["profile"]["checkpoint_iteration"])
    else:
        profile = conversion_profile("standard30000-v1")
        if args.iteration is not None and args.iteration != profile["checkpoint_iteration"]:
            parser.error("--iteration must match the whitelisted standard30000-v1 profile")
        iteration = int(profile["checkpoint_iteration"])

    import numpy as np
    import torch
    from torchvision.utils import save_image

    from gaussian_renderer import render3dgs
    from scene import GaussianModel, Scene
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    try:
        from lpipsPyTorch import lpips
    except Exception:  # pragma: no cover - optional runtime dependency
        lpips = None

    ply_path = args.ply.resolve()
    model_path = args.model_path.resolve()
    source_path = args.source_path.resolve()
    config = _load_args(model_path, source_path)

    gaussians = GaussianModel(
        config.feat_dim,
        config.n_offsets,
        config.voxel_size,
        config.update_depth,
        config.update_init_factor,
        config.update_hierachy_factor,
        config.use_feat_bank,
        config.appearance_dim,
        config.ratio,
        config.add_opacity_dist,
        config.add_cov_dist,
        config.add_color_dist,
    )
    scene = Scene(gaussians=gaussians, args=config, load_iteration=iteration)
    _load_converted_ply(ply_path, gaussians)
    gaussians.eval()

    views = scene.getTrainCameras()
    if not views:
        parser.error("training model contains no cameras")
    if args.max_views is not None and args.max_views != len(views):
        parser.error("evaluation is full ordered training-camera evaluation; --max-views must equal camera_count")
    indices = np.arange(len(views), dtype=int)
    selected = list(views)
    if authority is not None:
        expected_order = authority["camera_order"]
        if len(expected_order) != len(selected):
            parser.error("authority camera count differs from loaded training cameras")
        for ordinal, view in enumerate(selected):
            if Path(str(view.image_name)).stem != Path(str(expected_order[ordinal])).stem:
                parser.error(f"camera identity differs at ordinal {ordinal}")
    pipeline = SimpleNamespace(compute_cov3D_python=False, debug=False)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    render_dir = output.parent / (output.stem + "_renders")
    gt_dir = output.parent / (output.stem + "_gt")
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    image_dimensions: tuple[int, int] | None = None
    lpips_available = lpips is not None

    with torch.no_grad():
        for ordinal, view in enumerate(selected):
            if authority is not None:
                expected_dimensions = authority["camera_dimensions"]
                actual_dimensions = (int(view.original_image.shape[-1]), int(view.original_image.shape[-2]))
                if actual_dimensions != (expected_dimensions["width"], expected_dimensions["height"]):
                    parser.error(f"camera dimensions differ at ordinal {ordinal}")
            else:
                actual_dimensions = (int(view.original_image.shape[-1]), int(view.original_image.shape[-2]))
            if image_dimensions is None:
                image_dimensions = actual_dimensions
            torch.cuda.synchronize()
            started = time.perf_counter()
            rendering = render3dgs(view, gaussians, pipeline, background)["render"]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            rendered = rendering.clamp(0.0, 1.0)
            target = view.original_image[:3].to("cuda").clamp(0.0, 1.0)
            psnr_value = float(psnr(rendered, target).mean().item())
            ssim_value = float(ssim(rendered.unsqueeze(0), target.unsqueeze(0)).item())
            lpips_value = None
            if lpips_available:
                try:
                    lpips_value = float(
                        lpips(rendered.unsqueeze(0), target.unsqueeze(0), net_type="vgg")
                        .mean()
                        .item()
                    )
                except Exception as exc:  # preserve PSNR/SSIM if optional metric fails
                    lpips_available = False
                    lpips_error = f"{type(exc).__name__}: {exc}"
            else:
                lpips_error = "lpipsPyTorch import unavailable"

            stem = f"{ordinal:04d}_{view.image_name}"
            render_path = render_dir / f"{stem}.png"
            gt_path = gt_dir / f"{stem}.png"
            rendered_cpu = rendered.detach().cpu()
            save_image(rendered_cpu, str(render_path))
            del rendered_cpu
            target_cpu = target.detach().cpu()
            save_image(target_cpu, str(gt_path))
            del target_cpu
            records.append(
                {
                    "ordinal": ordinal,
                    "camera_index": int(indices[ordinal]),
                    "image_name": view.image_name,
                    "render_path": str(render_path),
                    "gt_path": str(gt_path),
                    "render_seconds": elapsed,
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "lpips": lpips_value,
                }
            )
            del rendering, rendered, target

    contact_layout = _write_contact_sheet_from_pngs(
        rows=records,
        output=args.contact_sheet,
        source_dimensions=image_dimensions,
    )

    def mean_metric(name: str) -> float | None:
        values = [record[name] for record in records if record[name] is not None]
        return float(np.mean(values)) if values else None

    result = {
        "metric_schema": "converted-ply-render-v1",
        "quality_boundary": "training_views_only_no_heldout_cameras",
        "inputs": {
            "ply": {"path": str(ply_path), "sha256": _sha256(ply_path)},
            "model_path": str(model_path),
            "source_path": str(source_path),
        },
        "camera_count": len(views),
        "evaluated_count": len(selected),
        "camera_indices": [int(index) for index in indices],
        "camera_order": [str(view.image_name) for view in selected],
        "authority_manifest": None if authority is None else str(Path(args.authority_manifest).resolve()),
        "held_out": False if authority is None else authority["status"]["held_out"],
        "metrics": {
            "psnr_mean": mean_metric("psnr"),
            "ssim_mean": mean_metric("ssim"),
            "lpips_mean": mean_metric("lpips"),
            "lpips_available": lpips_available,
            "lpips_error": locals().get("lpips_error"),
        },
        "contact_sheet": str(args.contact_sheet.resolve()),
        "contact_sheet_layout": contact_layout,
        "renders_dir": str(render_dir),
        "ground_truth_dir": str(gt_dir),
        "per_view": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Render evaluation written to {output}")


if __name__ == "__main__":
    main()
