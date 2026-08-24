"""CPU-only postcheck and evidence builder for one completed smoke render."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


CONTACT_SHEET_LAYOUT_POLICY = "aspect-preserving-letterbox-v1"
NORMALIZED_POSTCHECK_SCHEMA = "render-postcheck-evidence-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact is missing or symlinked: {path}")
    try:
        relative = str(path.relative_to(root.resolve()))
    except ValueError:
        relative = str(path)
    return {
        "path": str(path),
        "relative_to_run": relative,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON artifact is missing or symlinked: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _write_json_once(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"postcheck output already exists: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _numeric_pngs(path: Path) -> list[Path]:
    files = sorted(path.glob("*.png"), key=lambda item: int(item.stem))
    if any(item.is_symlink() or not item.is_file() for item in files):
        raise ValueError(f"render PNG list contains a symlink or non-file: {path}")
    indices = [int(item.stem) for item in files]
    if indices != list(range(len(files))):
        raise ValueError(f"render PNG indices are not exact and contiguous: {path}")
    return files


def _decode_one(path: Path, *, width: int, height: int) -> tuple[Any, dict[str, Any]]:
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim < 2:
        raise ValueError(f"PNG decode failed: {path}")
    if image.shape[1] != width or image.shape[0] != height:
        raise ValueError(f"PNG dimensions differ from PINHOLE contract: {path}: {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"PNG contains non-finite values: {path}")
    return image, {
        "name": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "channels": 1 if image.ndim == 2 else int(image.shape[2]),
        "finite": True,
    }


def _decode(paths: Iterable[Path], *, width: int, height: int) -> tuple[list[Any], list[dict[str, Any]]]:
    """Compatibility helper; production ``run`` uses ``_decode_one`` streaming."""

    images: list[Any] = []
    records: list[dict[str, Any]] = []
    for path in paths:
        image, record = _decode_one(path, width=width, height=height)
        images.append(image)
        records.append(record)
    return images, records


def _psnr(render: Any, target: Any) -> float | None:
    import numpy as np

    delta = render.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(delta * delta))
    if mse == 0.0:
        return None
    return float(10.0 * math.log10((255.0 * 255.0) / mse))


def _ssim(render: Any, target: Any) -> float:
    import cv2
    import numpy as np

    if render.ndim == 3:
        render = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY)
    if target.ndim == 3:
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    x = render.astype(np.float64) / 255.0
    y = target.astype(np.float64) / 255.0
    window = (11, 11)
    sigma = 1.5
    mu_x = cv2.GaussianBlur(x, window, sigma)
    mu_y = cv2.GaussianBlur(y, window, sigma)
    sigma_x = cv2.GaussianBlur(x * x, window, sigma) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, window, sigma) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, window, sigma) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return float(np.mean(score))


def _fit_with_metadata(image: Any, width: int, height: int) -> tuple[Any, dict[str, Any]]:
    import cv2
    import numpy as np

    if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("contact-sheet source image must have positive H/W")
    if width <= 0 or height <= 0:
        raise ValueError("contact-sheet tile dimensions must be positive")
    source_height = int(image.shape[0])
    source_width = int(image.shape[1])
    scale = min(float(width) / source_width, float(height) / source_height)
    resized_width = max(1, min(width, int(round(source_width * scale))))
    resized_height = max(1, min(height, int(round(source_height * scale))))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        canvas = np.zeros((height, width), dtype=image.dtype)
    else:
        canvas = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    padding_left = (width - resized_width) // 2
    padding_top = (height - resized_height) // 2
    canvas[padding_top : padding_top + resized_height, padding_left : padding_left + resized_width] = resized
    metadata = {
        "policy": CONTACT_SHEET_LAYOUT_POLICY,
        "source_width": source_width,
        "source_height": source_height,
        "tile_width": int(width),
        "tile_height": int(height),
        "scale": float(scale),
        "resized_width": resized_width,
        "resized_height": resized_height,
        "padding": {
            "left": padding_left,
            "right": width - padding_left - resized_width,
            "top": padding_top,
            "bottom": height - padding_top - resized_height,
        },
        "letterbox": True,
    }
    return canvas, metadata


def _fit(image: Any, width: int, height: int) -> Any:
    """Fit an image into a tile without changing its aspect ratio."""

    fitted, _ = _fit_with_metadata(image, width, height)
    return fitted


def _label(canvas: Any, text: str, x: int, y: int) -> None:
    import cv2

    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)


def _write_fixed_sheet(path: Path, indices: list[int], renders: Mapping[int, Any], gts: Mapping[int, Any]) -> None:
    import cv2
    import numpy as np

    tile_width = 360
    image_height = 205
    tile_height = image_height * 2 + 42
    columns = 3
    rows = math.ceil(len(indices) / columns)
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for slot, index in enumerate(indices):
        row, column = divmod(slot, columns)
        x = column * tile_width
        y = row * tile_height
        render = renders[index]
        target = gts[index]
        if render.ndim == 2:
            render = cv2.cvtColor(render, cv2.COLOR_GRAY2BGR)
        if target.ndim == 2:
            target = cv2.cvtColor(target, cv2.COLOR_GRAY2BGR)
        sheet[y : y + image_height, x : x + tile_width] = _fit(render, tile_width, image_height)
        _label(sheet, f"R fixed[{index:03d}]", x + 8, y + image_height - 8)
        second_y = y + image_height + 24
        sheet[second_y : second_y + image_height, x : x + tile_width] = _fit(target, tile_width, image_height)
        _label(sheet, f"GT train[{index:03d}]", x + 8, second_y + image_height - 8)
    if not cv2.imwrite(str(path), sheet):
        raise ValueError(f"failed to write fixed-vs-GT contact sheet: {path}")


def _write_on_path_sheet(path: Path, indices: list[int], nvs: Mapping[int, Any], pose_records: list[dict[str, Any]], cross_gap_index: int | None) -> None:
    import cv2
    import numpy as np

    tile_width = 320
    image_height = 182
    tile_height = image_height + 34
    columns = 5
    rows = math.ceil(len(indices) / columns)
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for slot, index in enumerate(indices):
        row, column = divmod(slot, columns)
        x = column * tile_width
        y = row * tile_height
        image = nvs[index]
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        sheet[y : y + image_height, x : x + tile_width] = _fit(image, tile_width, image_height)
        record = pose_records[index]
        marker = " CROSS-GAP" if index == cross_gap_index else ""
        _label(
            sheet,
            f"NVS[{index:03d}] {record['source_left_index']}->{record['source_right_index']} a={record['source_alpha']:.1f}{marker}",
            x + 5,
            y + image_height + 22,
        )
    if not cv2.imwrite(str(path), sheet):
        raise ValueError(f"failed to write on-path contact sheet: {path}")


def _write_ab_sheet(
    path: Path,
    indices: list[int],
    current: Mapping[int, Any],
    baseline: Mapping[int, Any],
    baseline_label: str,
    current_label: str = "coverage",
) -> None:
    """Write the deterministic fixed-view A/B comparison without judging quality."""

    import cv2
    import numpy as np

    tile_width = 360
    image_height = 205
    tile_height = image_height * 2 + 42
    columns = 3
    rows = math.ceil(len(indices) / columns)
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for slot, index in enumerate(indices):
        row, column = divmod(slot, columns)
        x = column * tile_width
        y = row * tile_height
        current_image = current[index]
        baseline_image = baseline[index]
        if current_image.ndim == 2:
            current_image = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)
        if baseline_image.ndim == 2:
            baseline_image = cv2.cvtColor(baseline_image, cv2.COLOR_GRAY2BGR)
        sheet[y : y + image_height, x : x + tile_width] = _fit(current_image, tile_width, image_height)
        _label(sheet, f"{current_label}[{index:03d}]", x + 8, y + image_height - 8)
        second_y = y + image_height + 24
        sheet[second_y : second_y + image_height, x : x + tile_width] = _fit(baseline_image, tile_width, image_height)
        _label(sheet, f"{baseline_label}[{index:03d}]", x + 8, second_y + image_height - 8)
    if not cv2.imwrite(str(path), sheet):
        raise ValueError(f"failed to write fixed-view A/B contact sheet: {path}")


def _write_three_way_sheet(
    path: Path,
    indices: list[int],
    current: Mapping[int, Any],
    coverage: Mapping[int, Any],
    smoke: Mapping[int, Any],
) -> None:
    """Write deterministic convergence/coverage/smoke rows for the same views."""

    import cv2
    import numpy as np

    if not indices:
        raise ValueError("three-way comparison requires at least one fixed view")
    tile_width = 320
    image_height = 180
    gap = 24
    row_height = image_height + gap
    sheet = np.zeros((3 * row_height + 24, tile_width * len(indices), 3), dtype=np.uint8)
    labels = (("convergence1000", current), ("coverage288", coverage), ("smoke100", smoke))
    for column, index in enumerate(indices):
        x = column * tile_width
        for row, (label, images) in enumerate(labels):
            image = images[index]
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            y = row * row_height
            sheet[y : y + image_height, x : x + tile_width] = _fit(image, tile_width, image_height)
            _label(sheet, f"{label}[{index:03d}]", x + 8, y + image_height - 8)
    if not cv2.imwrite(str(path), sheet):
        raise ValueError(f"failed to write three-way contact sheet: {path}")


def _spread_indices(count: int, requested: int, extra: Iterable[int] = ()) -> list[int]:
    if count <= 0:
        return []
    values = {int(round(index * (count - 1) / max(requested - 1, 1))) for index in range(requested)}
    values.update(int(value) for value in extra if 0 <= int(value) < count)
    return sorted(values)


def _optional_lpips_status() -> dict[str, Any]:
    """Report optional LPIPS availability without installing or downloading weights."""

    import importlib.util

    available = importlib.util.find_spec("lpips") is not None
    return {
        "available_in_route_environment": available,
        "computed": False,
        "status": "available_not_run_no_weight_download" if available else "unavailable",
        "reason": "optional metric; deterministic PSNR/SSIM are authoritative for this CPU postcheck",
    }


def _cross_gap(
    *,
    camera_names: list[str],
    segment_provenance: dict[str, Any],
    pose_records: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    segments = segment_provenance.get("candidate_segments")
    if not isinstance(segments, list):
        raise ValueError("segment provenance has no candidate_segments")
    selection_records = selection.get("frames")
    if not isinstance(selection_records, list):
        raise ValueError("frame selection has no frames")
    timestamps = {
        str(record.get("frame_id")): record.get("timestamp_sec")
        for record in selection_records
        if isinstance(record, dict) and isinstance(record.get("frame_id"), str)
    }
    for left_segment, right_segment in zip(segments, segments[1:]):
        left_names = left_segment.get("frame_names", [])
        right_names = right_segment.get("frame_names", [])
        excluded = [
            item
            for interval in segment_provenance.get("excluded_time_intervals", [])
            if isinstance(interval, dict)
            for item in interval.get("frame_names", [])
        ]
        if not left_names or not right_names or not excluded:
            continue
        left_name = str(left_names[-1])
        right_name = str(right_names[0])
        if left_name not in camera_names or right_name not in camera_names:
            continue
        left_index = camera_names.index(left_name)
        right_index = camera_names.index(right_name)
        matches = [
            pose
            for pose in pose_records
            if pose.get("source_left_index") == left_index
            and pose.get("source_right_index") == right_index
            and math.isclose(float(pose.get("source_alpha", -1.0)), 0.5, abs_tol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError("adjacent midpoint pose for the registered gap is missing or ambiguous")
        midpoint = matches[0]
        left_id = Path(left_name).stem
        right_id = Path(right_name).stem
        left_time = float(left_segment["end_time_sec"])
        right_time = float(right_segment["start_time_sec"])
        selection_left = timestamps.get(left_id)
        selection_right = timestamps.get(right_id)
        return {
            "cross_gap": True,
            "left_registered_name": left_name,
            "right_registered_name": right_name,
            "left_camera_index": left_index,
            "right_camera_index": right_index,
            "excluded_selected_names": excluded,
            "excluded_frame_count": len(excluded),
            "left_segment_end_time_sec": left_time,
            "right_segment_start_time_sec": right_time,
            "registered_boundary_time_gap_sec": right_time - left_time,
            "selection_timestamp_left_sec": selection_left,
            "selection_timestamp_right_sec": selection_right,
            "selection_timestamp_gap_sec": None if selection_left is None or selection_right is None else float(selection_right) - float(selection_left),
            "nvs_midpoint_index": int(midpoint["index"]),
            "nvs_source_left_index": int(midpoint["source_left_index"]),
            "nvs_source_right_index": int(midpoint["source_right_index"]),
            "nvs_source_alpha": float(midpoint["source_alpha"]),
            "reason": "adjacent_midpoint_slerp explicitly interleaves a midpoint across the registered component boundary; it is retained and not silently deleted",
        }
    return {"cross_gap": False, "reason": "no excluded interval between candidate segments"}


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import numpy as np

    run_root = args.run_root.resolve()
    render_attempt = args.render_attempt.resolve()
    training_attempt = args.training_attempt.resolve()
    training_input = args.training_input.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"postcheck output directory must be absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    render_result_path = render_attempt / "executor" / "result.json"
    render_request_path = render_attempt / "executor" / "request.json"
    training_result_path = training_attempt / "executor" / "result.json"
    render_result = _load(render_result_path)
    render_request = _load(render_request_path)
    training_result = _load(training_result_path)
    if render_result.get("exit_code") != 0 or render_result.get("structural_pass") is not True:
        raise ValueError("render executor result is not a successful structural pass")
    if training_result.get("exit_code") != 0 or training_result.get("structural_pass") is not True:
        raise ValueError("training executor result is not a successful structural pass")
    model = Path(str(render_result["model_path"])).resolve()
    iteration = int(render_result["render_iteration"])
    workload_profile = str(render_request.get("workload_profile", render_result.get("workload_profile", "smoke100-v1")))
    if workload_profile == "coverage-smoke-v1":
        postcheck_schema = "coverage-smoke-render-postcheck-v1"
        stage_name = "coverage-smoke-render"
    elif workload_profile == "convergence1000-v1":
        postcheck_schema = "convergence-smoke-render-postcheck-v1"
        stage_name = "convergence-smoke-render"
    elif workload_profile == "formal30000-v1":
        postcheck_schema = "formal30000-native-render-postcheck-v1"
        stage_name = "formal-native-render"
    else:
        postcheck_schema = "smoke100-render-postcheck-v1"
        stage_name = "smoke100-render"
    static = _load(training_input / "contract" / "static_contract.json")
    camera_contract = _load(training_input / "camera_contract-v1.json")
    camera_names = [str(name) for name in camera_contract["frame_names"]]
    width = int(static["camera"]["width"])
    height = int(static["camera"]["height"])
    training_image_residency = training_result.get("structural", {}).get("image_residency")
    render_image_residency = render_result.get("structural", {}).get("image_residency")
    image_residency_peak = (
        render_image_residency.get("gpu_resident_gt_frame_count_peak")
        if isinstance(render_image_residency, dict)
        else None
    )
    render_root = model / "train" / f"ours_{iteration}"
    render_paths = _numeric_pngs(render_root / "renders")
    gt_paths = _numeric_pngs(render_root / "gt")
    nvs_paths = _numeric_pngs(render_root / "nvs")
    pose_path = render_root / "videos" / "nvs_camera_poses.json"
    pose = _load(pose_path)
    pose_records = pose.get("poses")
    if not isinstance(pose_records, list) or not pose_records:
        raise ValueError("NVS pose contract has no poses")
    input_count = int(pose["input_count"])
    nvs_count = int(pose["nvs_count"])
    if len(render_paths) != input_count or len(gt_paths) != input_count or len(nvs_paths) != nvs_count:
        raise ValueError("render/GT/NVS counts differ from the actual NVS pose contract")
    if len(camera_names) != input_count:
        raise ValueError("camera contract count differs from NVS input_count")
    if [int(record.get("index", -1)) for record in pose_records] != list(range(nvs_count)):
        raise ValueError("NVS pose order/index contract is not exact")
    for record in pose_records:
        if not isinstance(record, dict) or not _finite_tree(record):
            raise ValueError("NVS pose contract contains non-finite values")
        left = record.get("source_left_index")
        right = record.get("source_right_index")
        alpha = record.get("source_alpha")
        if not isinstance(left, int) or not isinstance(right, int) or not 0 <= left < input_count or not 0 <= right < input_count:
            raise ValueError("NVS pose source camera index is outside the exact camera contract")
        if not isinstance(alpha, (int, float)) or not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("NVS pose interpolation alpha is invalid")

    fixed_indices = _spread_indices(input_count, 9)
    fixed_render_tiles: dict[int, Any] = {}
    fixed_gt_tiles: dict[int, Any] = {}
    render_hashes: list[dict[str, Any]] = []
    gt_hashes: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    contact_sheet_layouts: list[dict[str, Any]] = []

    def remember_layout(role: str, image: Any, tile_width: int, tile_height: int) -> None:
        _, metadata = _fit_with_metadata(image, tile_width, tile_height)
        metadata = {"role": role, **metadata}
        key = (
            metadata["role"],
            metadata["source_width"],
            metadata["source_height"],
            metadata["tile_width"],
            metadata["tile_height"],
        )
        if not any(
            (
                item.get("role"),
                item.get("source_width"),
                item.get("source_height"),
                item.get("tile_width"),
                item.get("tile_height"),
            )
            == key
            for item in contact_sheet_layouts
        ):
            contact_sheet_layouts.append(metadata)

    fixed_means: list[float] = []
    fixed_stds: list[float] = []
    for index, (render_path, gt_path) in enumerate(zip(render_paths, gt_paths)):
        render, render_record = _decode_one(render_path, width=width, height=height)
        target, target_record = _decode_one(gt_path, width=width, height=height)
        render_hashes.append(render_record)
        gt_hashes.append(target_record)
        metrics.append({"index": index, "psnr_db": _psnr(render, target), "ssim": _ssim(render, target)})
        fixed_means.append(float(render.mean()))
        fixed_stds.append(float(render.std()))
        if index in fixed_indices:
            fixed_render_tiles[index], _ = _fit_with_metadata(render, 360, 205)
            fixed_gt_tiles[index], _ = _fit_with_metadata(target, 360, 205)
            remember_layout("fixed_render", render, 360, 205)
            remember_layout("fixed_gt", target, 360, 205)
        del render, target

    segment_path = args.segment_provenance.resolve()
    selection_path = args.frame_selection.resolve()
    segment_document = _load(segment_path)
    if isinstance(segment_document.get("result"), dict):
        segment_provenance = segment_document["result"].get("segment_provenance", {})
    else:
        segment_provenance = segment_document.get("segment_provenance", segment_document)
    if not isinstance(segment_provenance, dict):
        raise ValueError("segment provenance document is malformed")
    selection = _load(selection_path)
    cross_gap = _cross_gap(
        camera_names=camera_names,
        segment_provenance=segment_provenance,
        pose_records=pose_records,
        selection=selection,
    )
    cross_gap_index = cross_gap.get("nvs_midpoint_index") if cross_gap.get("cross_gap") else None
    on_path_indices = _spread_indices(nvs_count, 24, () if cross_gap_index is None else (int(cross_gap_index),))
    nvs_hashes: list[dict[str, Any]] = []
    nvs_tiles: dict[int, Any] = {}
    for index, nvs_path in enumerate(nvs_paths):
        nvs, nvs_record = _decode_one(nvs_path, width=width, height=height)
        nvs_hashes.append(nvs_record)
        if index in on_path_indices:
            nvs_tiles[index], _ = _fit_with_metadata(nvs, 320, 182)
            remember_layout("on_path_nvs", nvs, 320, 182)
        del nvs

    endpoint_indices: list[int] = []
    endpoint_decoded = 0
    endpoint_bytes = 0
    for input_index in range(input_count):
        candidates = [
            pose_record
            for pose_record in pose_records
            if pose_record.get("source_left_index") == input_index
            and pose_record.get("source_right_index") == input_index
            and math.isclose(float(pose_record.get("source_alpha", -1.0)), 0.0, abs_tol=1e-12)
        ]
        if len(candidates) != 1:
            raise ValueError(f"NVS pose endpoint is missing or ambiguous for input index {input_index}")
        endpoint_index = int(candidates[0]["index"])
        endpoint_indices.append(endpoint_index)
        if render_paths[input_index].read_bytes() == nvs_paths[endpoint_index].read_bytes():
            endpoint_bytes += 1
        render, _ = _decode_one(render_paths[input_index], width=width, height=height)
        nvs, _ = _decode_one(nvs_paths[endpoint_index], width=width, height=height)
        if np.array_equal(render, nvs):
            endpoint_decoded += 1
        del render, nvs
    if endpoint_decoded != input_count or endpoint_bytes != input_count:
        raise ValueError("fixed/NVS endpoint equality failed")

    finite_psnr = [item["psnr_db"] for item in metrics if item["psnr_db"] is not None]
    ssim_values = [float(item["ssim"]) for item in metrics]
    fixed_sheet = output_dir / "fixed_vs_gt_contact_sheet.png"
    on_path_sheet = output_dir / "on_path_contact_sheet.png"
    _write_fixed_sheet(fixed_sheet, fixed_indices, fixed_render_tiles, fixed_gt_tiles)
    _write_on_path_sheet(on_path_sheet, on_path_indices, nvs_tiles, pose_records, cross_gap_index)

    baseline_comparison: dict[str, Any] | None = None
    baseline_sheet: Path | None = None
    three_way_comparison: dict[str, Any] | None = None
    if args.baseline_render_attempt is not None:
        baseline_attempt = args.baseline_render_attempt.resolve()
        baseline_result_path = baseline_attempt / "executor" / "result.json"
        baseline_result = _load(baseline_result_path)
        if baseline_result.get("exit_code") != 0 or baseline_result.get("structural_pass") is not True:
            raise ValueError("baseline render executor result is not a successful structural pass")
        baseline_model = Path(str(baseline_result["model_path"])).resolve()
        baseline_iteration = int(baseline_result["render_iteration"])
        baseline_paths = _numeric_pngs(baseline_model / "train" / f"ours_{baseline_iteration}" / "renders")
        if len(baseline_paths) != input_count:
            raise ValueError("baseline render count differs from current camera contract")
        baseline_hashes: list[dict[str, Any]] = []
        baseline_metrics = []
        baseline_tiles: dict[int, Any] = {}
        for index in fixed_indices:
            current, _ = _decode_one(render_paths[index], width=width, height=height)
            baseline, baseline_record = _decode_one(baseline_paths[index], width=width, height=height)
            while len(baseline_hashes) <= index:
                baseline_hashes.append({})
            baseline_hashes[index] = baseline_record
            baseline_tiles[index], _ = _fit_with_metadata(baseline, 360, 205)
            remember_layout("baseline_fixed", baseline, 360, 205)
            delta = np.abs(current.astype(np.float64) - baseline.astype(np.float64))
            baseline_metrics.append(
                {
                    "index": index,
                    "current_sha256": render_hashes[index]["sha256"],
                    "baseline_sha256": baseline_hashes[index]["sha256"],
                    "mean_abs_pixel_delta": float(delta.mean()),
                    "current_vs_baseline_psnr_db": _psnr(current, baseline),
                }
            )
            del current, baseline
        formal_baseline = workload_profile == "formal30000-v1"
        current_label = "formal30000" if formal_baseline else ("convergence1000" if workload_profile == "convergence1000-v1" else "coverage")
        baseline_label = "convergence1000" if formal_baseline else "smoke100"
        baseline_filename = "formal30000_vs_convergence1000_contact_sheet.png" if formal_baseline else "coverage_vs_smoke100_contact_sheet.png"
        baseline_sheet = output_dir / baseline_filename
        _write_ab_sheet(
            baseline_sheet,
            fixed_indices,
            fixed_render_tiles,
            baseline_tiles,
            baseline_label,
            current_label=current_label,
        )
        baseline_comparison = {
            "scope": "same deterministic fixed-view indices",
            "baseline_render_attempt": str(baseline_attempt),
            "baseline_result": _artifact(baseline_result_path, root=run_root),
            "baseline_model_path": str(baseline_model),
            "baseline_iteration": baseline_iteration,
            "indices": fixed_indices,
            "views": baseline_metrics,
            "contact_sheet": _artifact(baseline_sheet, root=run_root),
            "quality_decision": "not_automated; inspect current and baseline views together",
        }
    if workload_profile == "convergence1000-v1" and args.comparison_render_attempt is not None:
        if baseline_comparison is None:
            raise ValueError("convergence three-way comparison requires --baseline-render-attempt for smoke100")
        comparison_attempt = args.comparison_render_attempt.resolve()
        comparison_result_path = comparison_attempt / "executor" / "result.json"
        comparison_result = _load(comparison_result_path)
        if comparison_result.get("exit_code") != 0 or comparison_result.get("structural_pass") is not True:
            raise ValueError("comparison render executor result is not a successful structural pass")
        comparison_model = Path(str(comparison_result["model_path"])).resolve()
        comparison_iteration = int(comparison_result["render_iteration"])
        comparison_paths = _numeric_pngs(comparison_model / "train" / f"ours_{comparison_iteration}" / "renders")
        if len(comparison_paths) != input_count:
            raise ValueError("comparison render count differs from current camera contract")
        comparison_hashes: list[dict[str, Any]] = []
        three_metrics = []
        comparison_tiles: dict[int, Any] = {}
        smoke_tiles: dict[int, Any] = {}
        for index in fixed_indices:
            current, _ = _decode_one(render_paths[index], width=width, height=height)
            comparison, comparison_record = _decode_one(comparison_paths[index], width=width, height=height)
            baseline, baseline_record = _decode_one(baseline_paths[index], width=width, height=height)
            while len(comparison_hashes) <= index:
                comparison_hashes.append({})
            comparison_hashes[index] = comparison_record
            comparison_tiles[index], _ = _fit_with_metadata(comparison, 320, 180)
            smoke_tiles[index], _ = _fit_with_metadata(baseline, 320, 180)
            remember_layout("comparison_fixed", comparison, 320, 180)
            remember_layout("comparison_baseline", baseline, 320, 180)
            three_metrics.append(
                {
                    "index": index,
                    "convergence_vs_coverage_psnr_db": _psnr(current, comparison),
                    "convergence_vs_smoke100_psnr_db": _psnr(current, baseline),
                    "convergence_vs_coverage_mean_abs_pixel_delta": float(
                        np.abs(current.astype(np.float64) - comparison.astype(np.float64)).mean()
                    ),
                    "convergence_vs_smoke100_mean_abs_pixel_delta": float(
                        np.abs(current.astype(np.float64) - baseline.astype(np.float64)).mean()
                    ),
                    "convergence_sha256": render_hashes[index]["sha256"],
                    "coverage_sha256": comparison_hashes[index]["sha256"],
                    "smoke100_sha256": baseline_record["sha256"],
                }
            )
            del current, comparison, baseline
        three_sheet = output_dir / "convergence_vs_coverage_vs_smoke100_contact_sheet.png"
        _write_three_way_sheet(three_sheet, fixed_indices, fixed_render_tiles, comparison_tiles, smoke_tiles)
        three_way_comparison = {
            "scope": "same deterministic fixed-view indices; visual inspection only",
            "indices": fixed_indices,
            "coverage_render_attempt": str(comparison_attempt),
            "coverage_result": _artifact(comparison_result_path, root=run_root),
            "coverage_model_path": str(comparison_model),
            "coverage_iteration": comparison_iteration,
            "views": three_metrics,
            "contact_sheet": _artifact(three_sheet, root=run_root),
            "training_views_only": True,
            "quality_decision": "not_automated; inspect recognizable structure and low-frequency-to-geometry change",
        }

    png_hashes = {
        "schema_version": "smoke-render-png-hashes-v1",
        "dimensions": {"width": width, "height": height},
        "groups": {
            "fixed_renders": render_hashes,
            "training_gt": gt_hashes,
            "nvs_on_path": nvs_hashes,
        },
        "counts": {"fixed_renders": len(render_hashes), "training_gt": len(gt_hashes), "nvs_on_path": len(nvs_hashes)},
    }
    png_hashes_path = output_dir / "png_hashes.json"
    _write_json_once(png_hashes_path, png_hashes)

    metrics_payload = {
        "schema_version": "render-postcheck-metrics-v1",
        "metric_scope": "training_views_only",
        "held_out": False,
        "psnr": {
            "definition": "10*log10(255^2/MSE) over decoded uint8 image values; exact equality is null",
            "views": metrics,
            "mean_db": None if not finite_psnr else float(sum(finite_psnr) / len(finite_psnr)),
            "min_db": None if not finite_psnr else float(min(finite_psnr)),
        },
        "ssim": {
            "definition": "11x11 Gaussian window, sigma=1.5, grayscale uint8 values normalized to [0,1], C1=0.01^2, C2=0.03^2",
            "views": [{"index": item["index"], "ssim": item["ssim"]} for item in metrics],
            "mean": float(sum(ssim_values) / len(ssim_values)),
            "min": float(min(ssim_values)),
        },
        "deterministic_fixed_indices": fixed_indices,
    }
    metrics_path = output_dir / "metrics.json"
    _write_json_once(metrics_path, metrics_payload)

    fixed_sha_set = {item["sha256"] for item in render_hashes}
    if max(fixed_means) < 1e-4:
        visual_health = "fail"
    elif len(fixed_sha_set) == 1 or max(fixed_stds) < 1e-5:
        # Identical/uniform finite renders are a quality warning.  They do
        # not by themselves invalidate the ordered render contract or make
        # automated technical delivery unreachable.
        visual_health = "needs_review"
    else:
        visual_health = "needs_review"

    checkpoint = training_result.get("structural", {}).get("checkpoint", {})
    segment_artifact = _artifact(segment_path, root=run_root)
    selection_artifact = _artifact(selection_path, root=run_root)
    postcheck = {
        "schema_version": postcheck_schema,
        "normalized_schema_version": NORMALIZED_POSTCHECK_SCHEMA,
        "stage": stage_name,
        "workload_profile": workload_profile,
        "render_iteration": iteration,
        "training_views_only": True,
        "held_out": False,
        "gpu_invoked": False,
        "render_reused": True,
        "cuda_rerun": False,
        "resident_full_resolution_frame_max": image_residency_peak,
        "image_residency": {
            "training": training_image_residency,
            "render": render_image_residency,
            "source": "nested image-residency-telemetry-v1 when available",
        },
        "run_root": str(run_root),
        "render_attempt": str(render_attempt),
        "render_result": _artifact(render_result_path, root=run_root),
        "render_request": _artifact(render_request_path, root=run_root),
        "training_result": _artifact(training_result_path, root=run_root),
        "plan": {
            "path": render_request.get("plan_path"),
            "sha256": render_request.get("plan_sha256"),
            "route_code_identity_sha256": render_request.get("route_code_identity_sha256"),
            "static_contract_sha256": render_request.get("static_contract_sha256"),
        },
        "checkpoint": checkpoint,
        "mlp_files": training_result.get("structural", {}).get("mlp_files", {}),
        "camera_sampling_telemetry": training_result.get("structural", {}).get("camera_sampling_telemetry", {}),
        "anchor_schedule": training_result.get("structural", {}).get("anchor_schedule", {}),
        "camera_contract": {
            "path": str(training_input / "camera_contract-v1.json"),
            "sha256": _sha256(training_input / "camera_contract-v1.json"),
            "count": len(camera_names),
            "order": camera_names,
            "dimensions": {"width": width, "height": height},
        },
        "pose_contract": {
            "path": str(pose_path),
            "sha256": _sha256(pose_path),
            "pose_mode": pose.get("pose_mode"),
            "input_count": input_count,
            "nvs_count": nvs_count,
            "pose_convention": pose.get("pose_convention"),
            "finite": True,
            "endpoint_indices": endpoint_indices,
        },
        "counts": {
            "fixed_render": len(render_paths),
            "training_gt": len(gt_paths),
            "nvs_on_path": len(nvs_paths),
        },
        "structural": {
            "render_exit_code": int(render_result["exit_code"]),
            "fixed_render_count_exact": len(render_paths) == input_count,
            "training_gt_count_exact": len(gt_paths) == input_count,
            "nvs_count_from_actual_pose_contract": len(nvs_paths) == nvs_count,
            "all_pngs_decoded_finite_and_dimensions_exact": True,
            "endpoint_decoded_pixel_equal_count": endpoint_decoded,
            "endpoint_file_byte_equal_count": endpoint_bytes,
            "camera_order_exact": True,
        },
        "fixed_view_metrics": {
            "scope": "training_views_only",
            "held_out": False,
            "views": metrics,
            "deterministic_sample_indices": fixed_indices,
            "sample": [metrics[index] for index in fixed_indices],
            "mean_psnr_db": None if not finite_psnr else float(sum(finite_psnr) / len(finite_psnr)),
            "min_psnr_db": None if not finite_psnr else float(min(finite_psnr)),
            "mean_ssim": float(sum(ssim_values) / len(ssim_values)),
            "min_ssim": float(min(ssim_values)),
        },
        "metrics": _artifact(metrics_path, root=run_root),
        "optional_metrics": {
            "lpips": _optional_lpips_status(),
        },
        "segment_provenance": segment_artifact,
        "frame_selection": selection_artifact,
        "segment_provenance_summary": segment_provenance.get("segment_provenance", segment_provenance),
        "cross_gap": cross_gap,
        "contact_sheets": {
            "fixed_vs_gt": _artifact(fixed_sheet, root=run_root),
            "on_path": _artifact(on_path_sheet, root=run_root),
        },
        "contact_sheet_layout": {
            "schema_version": "contact-sheet-layout-v1",
            "policy": CONTACT_SHEET_LAYOUT_POLICY,
            "source_dimensions": {"width": width, "height": height},
            "records": contact_sheet_layouts,
        },
        "png_hashes": _artifact(png_hashes_path, root=run_root),
        "rough_visual": {
            "automatic_health": visual_health,
            "suggestion": "needs_review" if visual_health == "needs_review" else visual_health,
            "visual_acceptance_claimed": False,
            "notes": "Automatic CPU checks cannot certify upright orientation, recognizable subject/background, or absence of flying points; inspect both deterministic contact sheets.",
        },
        "baseline_comparison": baseline_comparison,
        "three_way_comparison": three_way_comparison,
        "limitations": [
            (
                "formal30000 is the local versioned 30000-iteration policy; it is not the official complete final schedule or an adaptive formula"
                if workload_profile == "formal30000-v1"
                else f"{workload_profile} iteration-{iteration} render only; no formal training"
            ),
            "training-view metrics only; held_out=false",
            "cross-gap midpoint is explicitly retained and separately attributed",
            "no conversion, converted evaluation, candidate delivery, or SuperSplat acceptance",
            "convergence1000 remains a one-shot local diagnostic; no automatic higher-iteration ladder",
        ],
    }
    postcheck_path = output_dir / "postcheck.json"
    _write_json_once(postcheck_path, postcheck)
    legacy_return = {
        "computed_pass": True,
        "postcheck": str(postcheck_path),
        "png_hashes": str(png_hashes_path),
        "metrics_path": str(metrics_path),
        "fixed_sheet": str(fixed_sheet),
        "on_path_sheet": str(on_path_sheet),
        "counts": postcheck["counts"],
        "metrics": postcheck["fixed_view_metrics"],
        "metrics_summary": postcheck["fixed_view_metrics"],
        # The adapter contract deliberately exposes the exact artifact
        # records already written to postcheck.json.  Keep the historical
        # metrics summary above and the path-oriented fields below for
        # callers that use the pre-v2 return shape.
        "fixed_view_metrics": postcheck["fixed_view_metrics"],
        "contact_sheets": postcheck["contact_sheets"],
        "metrics_artifact": postcheck["metrics"],
        "png_hashes_artifact": postcheck["png_hashes"],
        "cross_gap": cross_gap,
        "rough_visual": postcheck["rough_visual"],
        "contact_sheet_layout": postcheck["contact_sheet_layout"],
        "baseline_comparison": baseline_comparison,
        "three_way_comparison": three_way_comparison,
        "resident_full_resolution_frame_max": image_residency_peak,
        "image_residency": {
            "training": training_image_residency,
            "render": render_image_residency,
            "source": "nested image-residency-telemetry-v1 when available",
        },
        "gpu_invoked": False,
        "render_reused": True,
        "cuda_rerun": False,
    }
    return adapt_existing_postcheck_return(postcheck_path=postcheck_path, legacy_return=legacy_return)


def run_postcheck(
    *,
    run_root: Path,
    render_attempt: Path,
    training_attempt: Path,
    training_input: Path,
    frame_selection: Path,
    segment_provenance: Path,
    output_dir: Path,
    baseline_render_attempt: Path | None = None,
    comparison_render_attempt: Path | None = None,
) -> dict[str, Any]:
    """Run the append-only CPU postcheck without spawning another process."""

    return run(
        argparse.Namespace(
            run_root=run_root,
            render_attempt=render_attempt,
            training_attempt=training_attempt,
            training_input=training_input,
            frame_selection=frame_selection,
            segment_provenance=segment_provenance,
            output_dir=output_dir,
            baseline_render_attempt=baseline_render_attempt,
            comparison_render_attempt=comparison_render_attempt,
        )
    )


def adapt_existing_postcheck_return(*, postcheck_path: Path, legacy_return: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly materialize the v2 return contract for an older ledger result.

    This is an append-only compatibility adapter, not a replacement for the
    producer.  It verifies every path-oriented field from the historical
    return against the immutable postcheck JSON and its current file hashes,
    then exposes the exact artifact records already written by that producer.
    It never recomputes metrics or changes evidence bytes.
    """

    stored = _load(postcheck_path.resolve())
    if not isinstance(legacy_return, Mapping):
        raise ValueError("historical postcheck return must be an object")
    stored_sheets = stored.get("contact_sheets")
    stored_metrics = stored.get("metrics")
    stored_png_hashes = stored.get("png_hashes")
    if not isinstance(stored_sheets, Mapping) or not isinstance(stored_metrics, Mapping) or not isinstance(stored_png_hashes, Mapping):
        raise ValueError("postcheck JSON lacks the immutable visual evidence artifact contract")
    stored_structural = stored.get("structural")
    if not isinstance(stored_structural, Mapping):
        raise ValueError("postcheck JSON lacks structural evidence")

    def verify_artifact(value: Any, label: str) -> dict[str, Any]:
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str) or not isinstance(value.get("sha256"), str):
            raise ValueError(f"postcheck JSON {label} artifact is incomplete")
        path = Path(str(value["path"])).resolve()
        if path.is_symlink() or not path.is_file() or _sha256(path) != value["sha256"]:
            raise ValueError(f"postcheck JSON {label} artifact is missing or SHA-mutated")
        return dict(value)

    fixed = verify_artifact(stored_sheets.get("fixed_vs_gt"), "fixed-vs-GT")
    on_path = verify_artifact(stored_sheets.get("on_path"), "on-path")
    metrics = verify_artifact(stored_metrics, "metrics")
    png_hashes = verify_artifact(stored_png_hashes, "PNG hashes")
    # A current producer already returns these records.  Compare them before
    # normalizing so a producer/ledger return disagreement is a hard stop;
    # only genuinely older returns (which lack the fields) use this adapter.
    for key, expected in (
        ("structural", stored_structural),
        ("contact_sheets", stored_sheets),
        ("metrics_artifact", stored_metrics),
        ("png_hashes_artifact", stored_png_hashes),
    ):
        if key in legacy_return and legacy_return.get(key) != expected:
            raise ValueError(f"postcheck return contract drift for {key}")
    expected_paths = {
        "postcheck": str(postcheck_path.resolve()),
        "fixed_sheet": fixed["path"],
        "on_path_sheet": on_path["path"],
        "metrics_path": metrics["path"],
        "png_hashes": png_hashes["path"],
    }
    for key, expected in expected_paths.items():
        if legacy_return.get(key) != expected:
            raise ValueError(f"historical postcheck return path drift for {key}")

    normalized = dict(legacy_return)
    normalized.update(
        {
            "return_schema": "render-postcheck-return-v3",
            "normalized_schema_version": NORMALIZED_POSTCHECK_SCHEMA,
            "schema_version": stored.get("schema_version", normalized.get("schema_version")),
            "structural": dict(stored_structural),
            "counts": dict(stored.get("counts", normalized.get("counts", {}))),
            "camera_contract": dict(stored.get("camera_contract", {})),
            "pose_contract": dict(stored.get("pose_contract", {})),
            "checkpoint": dict(stored.get("checkpoint", {})),
            "mlp_files": dict(stored.get("mlp_files", {})),
            "camera_sampling_telemetry": dict(stored.get("camera_sampling_telemetry", {})),
            "anchor_schedule": dict(stored.get("anchor_schedule", {})),
            "image_residency": dict(stored.get("image_residency", {})),
            "cross_gap": dict(stored.get("cross_gap", {})),
            "rough_visual": dict(stored.get("rough_visual", {})),
            "contact_sheet_layout": dict(stored.get("contact_sheet_layout", {})),
            # Keep the historical path aliases consumed by the stage
            # orchestrator.  The hash-bound records remain available under
            # their explicit *_artifact fields, and are the canonical fields
            # used by validators.
            "metrics": legacy_return.get("metrics", stored_metrics),
            "png_hashes": legacy_return.get("png_hashes", png_hashes["path"]),
            "metrics_record": dict(stored_metrics),
            "png_hashes_record": dict(stored_png_hashes),
            "contact_sheets": dict(stored_sheets),
            "metrics_artifact": metrics,
            "png_hashes_artifact": png_hashes,
            "fixed_view_metrics": stored.get("fixed_view_metrics", legacy_return.get("metrics")),
            "metrics_summary": stored.get("fixed_view_metrics", legacy_return.get("metrics_summary", legacy_return.get("metrics"))),
            "legacy_return_adapter": {
                "schema_version": "render-postcheck-return-adapter-v1",
                "source_postcheck_path": str(postcheck_path.resolve()),
                "source_postcheck_sha256": _sha256(postcheck_path.resolve()),
                "producer_bytes_unchanged": True,
            },
        }
    )
    return normalized


def normalize_postcheck_evidence(
    postcheck: Mapping[str, Any],
    *,
    run_root: Path | None = None,
) -> dict[str, Any]:
    """Return one strict postcheck evidence shape for every consumer.

    Current stage results may contain the historical return adapter while the
    immutable ``postcheck.json`` contains the full structural object.  This
    function is the only compatibility boundary: it verifies the stored JSON,
    exact artifact hashes, and optional run containment, then returns the same
    normalized fields used by gates and delivery.  It never recomputes image
    metrics or rewrites producer evidence.
    """

    if not isinstance(postcheck, Mapping):
        raise ValueError("postcheck evidence must be an object")
    candidate = postcheck.get("postcheck_result")
    if not isinstance(candidate, Mapping):
        candidate = postcheck
    postcheck_value = postcheck.get("postcheck_result_path")
    if not isinstance(postcheck_value, str):
        postcheck_value = candidate.get("postcheck")
    if isinstance(postcheck_value, str):
        raw_path = Path(postcheck_value)
        if raw_path.exists() or raw_path.is_symlink():
            if raw_path.is_symlink() or not raw_path.is_file():
                raise ValueError(f"postcheck JSON is missing or symlinked: {raw_path}")
            resolved = raw_path.resolve()
            if run_root is not None:
                root = run_root.resolve()
                try:
                    resolved.relative_to(root)
                except ValueError as exc:
                    raise ValueError(f"postcheck JSON escapes run root: {resolved}") from exc
            legacy = dict(candidate)
            legacy["postcheck"] = str(resolved)
            stored = _load(resolved)
            sheets = stored.get("contact_sheets", {})
            metrics = stored.get("metrics", {})
            hashes = stored.get("png_hashes", {})
            if isinstance(sheets, Mapping):
                legacy["fixed_sheet"] = sheets.get("fixed_vs_gt", {}).get("path") if isinstance(sheets.get("fixed_vs_gt"), Mapping) else legacy.get("fixed_sheet")
                legacy["on_path_sheet"] = sheets.get("on_path", {}).get("path") if isinstance(sheets.get("on_path"), Mapping) else legacy.get("on_path_sheet")
                legacy["contact_sheets"] = sheets
            if isinstance(metrics, Mapping):
                legacy["metrics_path"] = metrics.get("path", legacy.get("metrics_path"))
                legacy["metrics_artifact"] = metrics
            if isinstance(hashes, Mapping):
                legacy["png_hashes"] = hashes.get("path", legacy.get("png_hashes"))
                legacy["png_hashes_artifact"] = hashes
            normalized = adapt_existing_postcheck_return(postcheck_path=resolved, legacy_return=legacy)
            normalized["postcheck_result_path"] = str(resolved)
            return normalized
    if not isinstance(candidate.get("structural"), Mapping):
        raise ValueError("postcheck evidence has no normalized structural object")
    normalized = dict(candidate)
    normalized.setdefault("normalized_schema_version", NORMALIZED_POSTCHECK_SCHEMA)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="Build CPU-only deterministic render postcheck evidence")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--render-attempt", type=Path, required=True)
    parser.add_argument("--training-attempt", type=Path, required=True)
    parser.add_argument("--training-input", type=Path, required=True)
    parser.add_argument("--frame-selection", type=Path, required=True)
    parser.add_argument("--segment-provenance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-render-attempt", type=Path)
    parser.add_argument("--comparison-render-attempt", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
