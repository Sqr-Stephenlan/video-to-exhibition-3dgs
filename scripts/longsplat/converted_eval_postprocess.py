"""CPU-only recovery for a completed converted-evaluation render.

The CUDA evaluator may be interrupted after writing its render pairs.  This
module consumes those immutable PNGs, the failed stage record, and the
authority manifest without importing LongSplat or torch.  It processes one
ordered GT/native/converted triple at a time and keeps candidate delivery
separate from SuperSplat acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .acceptance_delivery import AcceptanceDeliveryError, create_candidate_delivery
from .authority_manifest import AuthorityManifestError, load_authority_manifest
from .pipeline_contract import PipelineBlocked, resolve_contained_path, resolve_containment_root


SCHEMA_VERSION = "longsplat-converted-eval-cpu-postprocess-v1"
DEFAULT_CONTACT_INDICES = (0, 18, 36, 54, 72, 89, 107, 125, 143)
_PNG_NAME_RE = re.compile(r"^(?P<ordinal>[0-9]{4})_(?P<token>.+)\.png$")


class ConvertedEvalPostprocessError(ValueError):
    """The immutable converted-evaluation evidence violates its contract."""


def _fail(message: str) -> None:
    raise ConvertedEvalPostprocessError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail(f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def _reject_symlink_components(path: Path, label: str) -> None:
    probe = Path(path.anchor)
    for component in path.parts[1:]:
        probe /= component
        if probe.is_symlink():
            _fail(f"{label} traverses a symlink: {probe}")


def _identity(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is missing or symlinked: {path}")
    _reject_symlink_components(path, label)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _identity(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not valid JSON: {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object: {path}")
    return value


def _route_outputs(
    route_root: str | Path,
    *,
    containment_root: str | Path | None = None,
) -> tuple[Path, Path]:
    route = Path(route_root).resolve()
    if containment_root is not None:
        try:
            return route, resolve_containment_root(containment_root)
        except PipelineBlocked as exc:
            _fail(str(exc))
    outputs = route / "outputs"
    if outputs.is_symlink() or not outputs.is_dir():
        _fail(f"route outputs directory is missing or symlinked: {outputs}")
    return route, outputs.resolve()


def _output_path(
    value: str | Path,
    outputs: Path,
    label: str,
    *,
    directory: bool = False,
    must_exist: bool = True,
    containment_root: str | Path | None = None,
) -> Path:
    if containment_root is not None:
        try:
            return resolve_contained_path(
                value,
                root=resolve_containment_root(containment_root),
                label=label,
                must_exist=must_exist,
                directory=directory if must_exist else None,
            )
        except PipelineBlocked as exc:
            _fail(str(exc))
    raw = Path(value)
    if not raw.is_absolute():
        _fail(f"{label} must be absolute: {raw}")
    try:
        relative = raw.relative_to(outputs)
    except ValueError:
        _fail(f"{label} must be below route outputs: {raw}")
    if not relative.parts or any(part in {".", ".."} for part in relative.parts):
        _fail(f"{label} must be a strict descendant of route outputs: {raw}")
    _reject_symlink_components(raw, label)
    resolved = raw.resolve(strict=False)
    try:
        resolved.relative_to(outputs.resolve())
    except ValueError:
        _fail(f"{label} escapes route outputs: {raw}")
    if must_exist:
        if directory and not raw.is_dir():
            _fail(f"{label} directory is missing: {raw}")
        if not directory and not raw.is_file():
            _fail(f"{label} file is missing: {raw}")
    return raw


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_manifest_evidence(authority: Mapping[str, Any], outputs: Path) -> None:
    """Contain all consumed manifest evidence in this route's outputs."""

    manifest = authority["manifest"]
    for field in ("plan", "static_contract", "training_input", "training_model"):
        value = manifest[field]
        _output_path(value["path"], outputs, f"authority {field}", directory=field in {"training_input", "training_model"})
    checkpoint = manifest.get("checkpoint", {})
    for name, value in checkpoint.get("files", {}).items():
        _output_path(value["path"], outputs, f"authority checkpoint {name}")
    cameras = manifest.get("cameras", {})
    for name, value in (("train", cameras.get("train")), ("test", cameras.get("test"))):
        _output_path(value["path"], outputs, f"authority cameras {name}")
    _output_path(manifest["pose_contract"]["path"], outputs, "authority pose contract")
    native = manifest["native_render_evidence"]
    _output_path(native["render_root"], outputs, "authority native render root", directory=True)
    for name in ("result", "postcheck"):
        _output_path(native[name]["path"], outputs, f"authority native {name}")
    run_scope = manifest.get("run_scope", {})
    for name in ("camera_contract", "segment_provenance"):
        value = run_scope.get(name)
        if isinstance(value, Mapping) and isinstance(value.get("path"), str):
            _output_path(value["path"], outputs, f"authority run_scope {name}")


def _conversion_inputs(
    conversion_result_path: Path,
    conversion_result: Mapping[str, Any],
    converted_ply: Path,
) -> dict[str, Any]:
    if conversion_result.get("STRUCTURAL_CONVERSION_PASS") is not True or conversion_result.get("structural_pass") is not True:
        _fail("CPU postprocess requires the unique technical conversion pass")
    structural = conversion_result.get("structural")
    if not isinstance(structural, Mapping):
        _fail("conversion result has no structural report")
    declared_path = structural.get("path") or structural.get("converted_ply_path")
    if not isinstance(declared_path, str):
        _fail("conversion result has no technical PLY path")
    if Path(declared_path).resolve() != converted_ply.resolve():
        _fail("technical PLY differs from the conversion result path")
    ply_identity = _identity(converted_ply, "technical converted PLY")
    for key in ("sha256",):
        declared = structural.get(key)
        if declared is not None and declared != ply_identity[key]:
            _fail(f"technical PLY identity differs from conversion result at {key}")
    declared_size = structural.get("file_size")
    if declared_size is not None and int(declared_size) != ply_identity["size_bytes"]:
        _fail("technical PLY size differs from conversion result")
    vertex_count = structural.get("vertex_count")
    if not isinstance(vertex_count, int) or vertex_count <= 0:
        _fail("conversion result has no positive technical PLY vertex count")
    return {
        "conversion_result": _identity(conversion_result_path, "conversion result"),
        "technical_ply": ply_identity,
        "vertex_count": vertex_count,
        "attributes": structural.get("attributes", []),
        "quality": structural.get("quality", {}),
    }


def _failed_evaluation_inputs(failed_root: Path, outputs: Path) -> dict[str, Any]:
    root = _output_path(failed_root, outputs, "failed evaluator root", directory=True)
    result_path = _output_path(root / "evaluation_result.json", outputs, "failed evaluator result")
    result = _load_json(result_path, "failed evaluator result")
    if result.get("exit_code") != -9:
        _fail(f"failed evaluator result must preserve exit_code -9, got {result.get('exit_code')!r}")
    if result.get("SAME_CAMERA_VISUAL_PASS") != "fail":
        _fail("failed evaluator result is not the preserved failed result")
    eval_root = root / "same_camera_eval"
    if not eval_root.is_dir():
        _fail(f"failed evaluator render root is missing: {eval_root}")
    eval_root = _output_path(eval_root, outputs, "failed evaluator render root", directory=True)
    input_files: dict[str, dict[str, Any]] = {
        "failed_evaluation_result": _identity(result_path, "failed evaluator result")
    }
    for name in ("request.json", "argv.json", "stdout.log", "stderr.log"):
        for candidate in (root / name, eval_root / name):
            if candidate.is_file() and not candidate.is_symlink():
                input_files[name] = _identity(candidate, f"failed evaluator {name}")
                break
    gt_root = _output_path(eval_root / "evaluator_result_gt", outputs, "failed evaluator GT root", directory=True)
    converted_root = _output_path(eval_root / "evaluator_result_renders", outputs, "failed evaluator converted root", directory=True)
    return {
        "root": root,
        "eval_root": eval_root,
        "result_path": result_path,
        "result": result,
        "gt_root": gt_root,
        "converted_root": converted_root,
        "input_files": input_files,
        "original_exit_code": -9,
        "original_failure_reason": "unknown (SIGKILL; no cause inferred)",
    }


def _contract_output_token(camera_name: str, manifest: Mapping[str, Any], index: int) -> str:
    """Return the exact recorded evaluator basename token.

    Current cameras are stored as stems while the immutable COLMAP contract
    records ``frame_*.png``.  This is an explicit positional mapping, not a
    substring or fuzzy extension strip.  Ambiguous mappings are rejected.
    """

    if Path(camera_name).name != camera_name or camera_name in {"", ".", ".."}:
        _fail(f"camera identity is not a basename at ordinal {index}: {camera_name!r}")
    run_scope = manifest.get("run_scope", {})
    registered = run_scope.get("registered_contract_names")
    if not isinstance(registered, list) or len(registered) <= index:
        return camera_name
    contract_name = registered[index]
    if not isinstance(contract_name, str) or Path(contract_name).name != contract_name:
        _fail(f"registered contract name is not a basename at ordinal {index}")
    camera_stem = Path(camera_name).stem if Path(camera_name).suffix else camera_name
    contract_stem = Path(contract_name).stem
    if contract_stem != camera_stem:
        _fail(f"registered contract/name mapping differs at ordinal {index}: {contract_name!r} != {camera_name!r}")
    return contract_stem


def _expected_converted_names(camera_order: Sequence[str], manifest: Mapping[str, Any]) -> list[str]:
    expected = [f"{index:04d}_{_contract_output_token(name, manifest, index)}.png" for index, name in enumerate(camera_order)]
    if len(set(expected)) != len(expected):
        _fail("converted evaluator output names are not unique")
    return expected


def _validate_png_set(root: Path, names: Sequence[str], dimensions: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    entries = list(root.iterdir())
    if any(entry.is_symlink() for entry in entries):
        _fail(f"{label} contains a symlink")
    if any(not entry.is_file() for entry in entries):
        _fail(f"{label} contains a non-file entry")
    actual = sorted(entry.name for entry in entries)
    expected = list(names)
    if actual != sorted(expected) or len(actual) != len(expected):
        _fail(f"{label} names/count differ from the authority: {len(actual)} != {len(expected)}")
    expected_shape = (int(dimensions["height"]), int(dimensions["width"]))
    rows = []
    for ordinal, name in enumerate(expected):
        path = root / name
        identity = _identity(path, f"{label} {name}")
        rows.append({"ordinal": ordinal, "filename": name, "path": str(path.resolve()), "identity": identity, "expected_shape": expected_shape})
    return rows


def _read_rgb(path: Path, expected_shape: tuple[int, int], label: str) -> tuple[Any, bool]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - route environment supplies these
        _fail(f"OpenCV/numpy are required for CPU postprocess: {exc}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        _fail(f"{label} cannot be decoded: {path}")
    if tuple(image.shape[:2]) != expected_shape:
        _fail(f"{label} dimensions differ at {path}: {image.shape[:2]} != {expected_shape}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if not np.all(np.isfinite(rgb)):
        _fail(f"{label} contains non-finite pixels: {path}")
    empty = bool(rgb.size == 0 or float(np.max(rgb)) <= 1e-6)
    return rgb, empty


def _metrics(left: Any, right: Any) -> dict[str, float | None]:
    import numpy as np

    difference = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    mse = float(np.mean(difference * difference))
    psnr = None if mse == 0.0 else float(10.0 * np.log10(1.0 / mse))
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    ma, mb = float(a.mean()), float(b.mean())
    va, vb = float(((a - ma) ** 2).mean()), float(((b - mb) ** 2).mean())
    covariance = float(((a - ma) * (b - mb)).mean())
    c1, c2 = 0.01**2, 0.03**2
    denominator = (ma * ma + mb * mb + c1) * (va + vb + c2)
    ssim = float(((2 * ma * mb + c1) * (2 * covariance + c2)) / denominator) if denominator else 0.0
    return {"mse": mse, "psnr_db": psnr, "ssim": ssim, "mean_pixel_diff": float(np.mean(np.abs(difference)))}


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    import numpy as np

    output: dict[str, float | int | None] = {"count": len(records)}
    for field in ("mse", "psnr_db", "ssim", "mean_pixel_diff"):
        values = [float(item[field]) for item in records if item.get(field) is not None]
        output[field + "_mean"] = float(np.mean(values)) if values else None
        output[field + "_min"] = float(np.min(values)) if values else None
        output[field + "_p10"] = float(np.percentile(values, 10)) if values else None
        output[field + "_max"] = float(np.max(values)) if values else None
    return output


def _contact_sheet(
    *,
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    dimensions: tuple[int, int],
    indices: Sequence[int],
) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        _fail(f"OpenCV/numpy are required for contact sheet: {exc}")
    width, height = dimensions[1], dimensions[0]
    tile_width = 320
    tile_height = max(1, round(tile_width * height / width))
    label_height = 34
    margin = 12
    header_height = 30
    canvas_width = len(indices) * (tile_width + margin) + margin
    canvas_height = header_height + 3 * (tile_height + label_height + margin) + margin
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)
    row_by_ordinal = {int(row["ordinal"]): row for row in rows}
    labels = ("GT", "native", "converted")
    for column, ordinal in enumerate(indices):
        if ordinal not in row_by_ordinal:
            _fail(f"contact sheet ordinal is absent from postprocess rows: {ordinal}")
        row = row_by_ordinal[ordinal]
        x = margin + column * (tile_width + margin)
        cv2.putText(canvas, f"{ordinal:03d} {row['camera_name']}", (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        for series, key in enumerate(("gt_path", "native_path", "converted_path")):
            path = Path(str(row[key]))
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or tuple(image.shape[:2]) != (height, width):
                _fail(f"contact sheet cannot decode expected {key}: {path}")
            resized = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
            y = header_height + margin + series * (tile_height + label_height + margin)
            canvas[y : y + tile_height, x : x + tile_width] = resized
            cv2.putText(canvas, labels[series], (x + 6, y + tile_height + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            del image, resized
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        _fail(f"cannot write contact sheet: {output}")
    return _identity(output, "postprocess contact sheet")


def _contact_indices(camera_count: int) -> list[int]:
    if camera_count <= 0:
        _fail("contact sheet requires at least one camera")
    selected = [index for index in DEFAULT_CONTACT_INDICES if index < camera_count]
    if not selected:
        selected = [0]
    if camera_count > 1 and camera_count - 1 not in selected:
        selected.append(camera_count - 1)
    return selected


def run_postprocess(
    *,
    authority_manifest_path: str | Path,
    conversion_result_path: str | Path,
    converted_ply_path: str | Path,
    failed_evaluation_root: str | Path,
    output_root: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    native_visual_review: str | Path | None = None,
    conversion_release_decision: str | Path | None = None,
    segment_provenance: str | Path | None = None,
) -> dict[str, Any]:
    route, outputs = _route_outputs(route_root, containment_root=containment_root)
    manifest_path = _output_path(authority_manifest_path, outputs, "authority manifest", containment_root=containment_root)
    try:
        authority = load_authority_manifest(manifest_path, route_root=route, containment_root=containment_root)
    except AuthorityManifestError as exc:
        _fail(str(exc))
    _validate_manifest_evidence(authority, outputs)
    manifest = authority["manifest"]

    conversion_path = _output_path(conversion_result_path, outputs, "conversion result", containment_root=containment_root)
    converted_ply = _output_path(converted_ply_path, outputs, "technical converted PLY", containment_root=containment_root)
    conversion_result = _load_json(conversion_path, "conversion result")
    conversion_inputs = _conversion_inputs(conversion_path, conversion_result, converted_ply)
    failed = _failed_evaluation_inputs(Path(failed_evaluation_root), outputs)
    output = _output_path(output_root, outputs, "postprocess output root", directory=True, must_exist=False, containment_root=containment_root)
    if output.exists() or output.is_symlink():
        _fail(f"postprocess attempt must be fresh and append-only: {output}")
    output.mkdir(parents=True, exist_ok=False)

    count = authority["camera_count"]
    order = authority["camera_order"]
    dimensions = authority["camera_dimensions"]
    shape = (int(dimensions["height"]), int(dimensions["width"]))
    eval_names = _expected_converted_names(order, manifest)
    native = authority["native_render_evidence"]
    native_names = native.get("render_files")
    if not isinstance(native_names, list) or len(native_names) != count:
        _fail("authority native render file list is not the exact camera count")
    native_root = _output_path(native["render_root"], outputs, "native render root", directory=True)
    failed_gt_rows = _validate_png_set(failed["gt_root"], eval_names, dimensions, "failed evaluator GT set")
    failed_converted_rows = _validate_png_set(failed["converted_root"], eval_names, dimensions, "failed evaluator converted set")
    native_rows = _validate_png_set(native_root, native_names, dimensions, "native render set")

    png_hashes = {
        "schema_version": "longsplat-converted-eval-png-hashes-v1",
        "camera_count": count,
        "camera_order": order,
        "camera_dimensions": dimensions,
        "sets": {"failed_evaluator_gt": failed_gt_rows, "failed_evaluator_converted": failed_converted_rows, "native": native_rows},
    }
    rows: list[dict[str, Any]] = []
    converted_metrics: list[dict[str, float | None]] = []
    native_metrics: list[dict[str, float | None]] = []
    empty_views: list[dict[str, Any]] = []
    for ordinal, camera_name in enumerate(order):
        gt_path = Path(failed_gt_rows[ordinal]["path"])
        converted_path = Path(failed_converted_rows[ordinal]["path"])
        native_path = Path(native_rows[ordinal]["path"])
        gt, gt_empty = _read_rgb(gt_path, shape, f"failed evaluator GT ordinal {ordinal}")
        converted, converted_empty = _read_rgb(converted_path, shape, f"failed evaluator converted ordinal {ordinal}")
        native_image, native_empty = _read_rgb(native_path, shape, f"native ordinal {ordinal}")
        converted_metric = _metrics(converted, gt)
        native_metric = _metrics(native_image, converted)
        converted_metrics.append(converted_metric)
        native_metrics.append(native_metric)
        flags = {"gt": gt_empty, "converted": converted_empty, "native": native_empty}
        if any(flags.values()):
            empty_views.append({"ordinal": ordinal, "camera_name": camera_name, "empty": flags})
        rows.append(
            {
                "ordinal": ordinal,
                "camera_name": camera_name,
                "gt_path": str(gt_path),
                "native_path": str(native_path),
                "converted_path": str(converted_path),
                "empty_or_black": flags,
                "converted_vs_gt": converted_metric,
                "native_vs_converted": native_metric,
            }
        )
        del gt, converted, native_image

    metrics = {
        "schema_version": "longsplat-converted-eval-cpu-metrics-v1",
        "metrics_scope": "training_views_only",
        "held_out": False,
        "camera_count": count,
        "camera_order": order,
        "camera_dimensions": dimensions,
        "color_contract": "PNG decoded by cv2 IMREAD_COLOR, BGR converted to RGB, float32 divided by 255.0",
        "psnr": {"formula": "10*log10(1/MSE)", "data_range": 1.0},
        "ssim": {"formula": "global scalar SSIM over RGB pixels/channels; C1=0.01^2, C2=0.03^2; no windowed library", "data_range": 1.0},
        "converted_vs_gt": _aggregate(converted_metrics),
        "native_vs_converted": _aggregate(native_metrics),
        "per_view": rows,
    }
    metrics_path = output / "metrics.json"
    png_hashes_path = output / "png_hashes.json"
    _write_json(metrics_path, metrics)
    _write_json(png_hashes_path, png_hashes)
    contact_path = output / "fixed_gt_native_converted_contact_sheet.png"
    contact = _contact_sheet(rows=rows, output=contact_path, dimensions=shape, indices=_contact_indices(count))

    severe_mse_threshold = 0.25
    severe_converted = any(float(record["mse"]) > severe_mse_threshold for record in converted_metrics)
    severe_native = any(float(record["mse"]) > severe_mse_threshold for record in native_metrics)
    structural_pass = len(rows) == count and not empty_views
    full_pass = structural_pass and not severe_converted and not severe_native
    optional_evidence: dict[str, str] = {}
    for name, value in (
        ("supervisor_conversion_release_decision.json", conversion_release_decision),
        ("formal_native_visual_review.json", native_visual_review),
        ("segment_provenance.json", segment_provenance),
    ):
        if value is not None:
            path = _output_path(value, outputs, name)
            optional_evidence[name] = str(path.resolve())
    evidence_files = {
        "authority_manifest.json": str(manifest_path.resolve()),
        "conversion_result.json": str(conversion_path.resolve()),
        "failed_evaluation_result.json": str(failed["result_path"].resolve()),
        "failed_evaluation_argv.json": str(Path(failed["input_files"]["argv.json"]["path"]).resolve()) if "argv.json" in failed["input_files"] else str(failed["result_path"].resolve()),
        "postprocess_result.json": str((output / "postprocess_result.json").resolve()),
        "postprocess_metrics.json": str(metrics_path.resolve()),
        "postprocess_png_hashes.json": str(png_hashes_path.resolve()),
        **optional_evidence,
    }
    known_limitations = [
        "training views only; no held-out evaluation",
        "dynamic-person ghosting",
        "local breakage/ghosting around the 2.982s frame_000140 to frame_000146 cross-gap",
        "the original converted evaluator exited with SIGKILL (-9); the cause is unknown and was not inferred as OOM",
        "the converted render PNGs were reused; no CUDA evaluator rerun was performed",
        "no real SuperSplat three-view manual acceptance has been imported",
    ]
    run_scope = manifest.get("run_scope", {})
    provenance_context = {
        "current_video_only": True,
        "selected_camera_count": run_scope.get("selected_camera_count"),
        "registered_camera_count": run_scope.get("registered_camera_count", count),
        "selected_not_registered": run_scope.get("selected_not_registered", []),
        "segment_provenance": run_scope.get("segment_provenance"),
        "historical_exclusions": manifest.get("historical_exclusions", []),
        "postprocess_recovery": {
            "gpu_invoked": False,
            "render_reused": True,
            "cuda_rerun": False,
            "original_evaluator_exit_code": -9,
            "original_evaluator_failure_reason": "unknown",
        },
        "known_limitations": known_limitations,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "converted-eval-postprocess",
        "status": "technical_pass" if full_pass else "needs_review",
        "STRUCTURAL_CONVERSION_PASS": True,
        "FULL_STREAM_VALIDATION_PASS": full_pass,
        "SAME_CAMERA_VISUAL_PASS": "pass" if full_pass else "fail",
        "accepted": False,
        "supersplat": False,
        "three_view_manual_acceptance_required": True,
        "gpu_invoked": False,
        "render_reused": True,
        "cuda_rerun": False,
        "original_evaluator_exit_code": -9,
        "original_evaluator_failure_reason": "unknown",
        "metrics_scope": "training_views_only",
        "held_out": False,
        "camera_count": count,
        "camera_order": order,
        "camera_dimensions": dimensions,
        "authority_manifest": _identity(manifest_path, "authority manifest"),
        "conversion": conversion_inputs,
        "failed_evaluation": {
            "root": str(failed["eval_root"]),
            "result": _identity(failed["result_path"], "failed evaluator result"),
            "render_reused": True,
            "cuda_rerun": False,
            "original_failure_reason": "unknown",
            "input_files": failed["input_files"],
        },
        "native_render": {"root": str(native_root), "camera_order": native_names},
        "full_stream_validation": {
            "pass": full_pass,
            "triples_processed": len(rows),
            "expected_triples": count,
            "resident_full_resolution_frame_max": 3,
            "all_pngs_decoded": structural_pass,
            "all_pngs_finite": structural_pass,
            "all_dimensions_exact": structural_pass,
            "all_names_and_order_exact": structural_pass,
            "empty_or_black_views": empty_views,
        },
        "metrics": {"path": str(metrics_path.resolve()), "sha256": _sha256(metrics_path), "size_bytes": metrics_path.stat().st_size},
        "png_hashes": {"path": str(png_hashes_path.resolve()), "sha256": _sha256(png_hashes_path), "size_bytes": png_hashes_path.stat().st_size},
        "contact_sheet": contact,
        "severe_degradation_policy": {"mse_threshold": severe_mse_threshold, "converted_vs_gt": severe_converted, "native_vs_converted": severe_native},
        "candidate_delivery": {"eligible_after_manual_visual_review": full_pass, "evidence_files": evidence_files, "comparison_sheet": str(contact_path.resolve()), "provenance_context": provenance_context},
        "known_limitations": known_limitations,
    }
    result_path = output / "postprocess_result.json"
    _write_json(result_path, result)
    return result


def create_candidate_from_postprocess(
    *,
    authority_manifest_path: str | Path,
    postprocess_result_path: str | Path,
    candidate_output: str | Path,
    route_root: str | Path,
    containment_root: str | Path | None = None,
    manual_visual_decision: str,
    manual_visual_note: str,
) -> dict[str, Any]:
    route, outputs = _route_outputs(route_root, containment_root=containment_root)
    manifest_path = _output_path(authority_manifest_path, outputs, "authority manifest", containment_root=containment_root)
    post_path = _output_path(postprocess_result_path, outputs, "postprocess result", containment_root=containment_root)
    result = _load_json(post_path, "postprocess result")
    if result.get("SAME_CAMERA_VISUAL_PASS") != "pass" or result.get("FULL_STREAM_VALIDATION_PASS") is not True:
        _fail("candidate delivery requires a complete passing CPU postprocess")
    if manual_visual_decision != "pass":
        _fail("candidate delivery requires an explicit manual visual pass; no automatic acceptance is provided")
    candidate_info = result.get("candidate_delivery")
    if not isinstance(candidate_info, Mapping):
        _fail("postprocess result has no candidate delivery inputs")
    converted = candidate_info.get("provenance_context")
    evidence = candidate_info.get("evidence_files")
    comparison = candidate_info.get("comparison_sheet")
    if not isinstance(evidence, Mapping) or not isinstance(comparison, str):
        _fail("postprocess result candidate inputs are malformed")
    evidence_files = {}
    for name, value in evidence.items():
        if not isinstance(name, str) or not isinstance(value, str):
            _fail("candidate evidence mapping is malformed")
        path = _output_path(value, outputs, f"candidate evidence {name}")
        evidence_files[name] = path
    # The existing candidate helper uses the stable semantic key
    # ``evaluation_result`` for its technical gate.  Bind it to the new CPU
    # result, while retaining the explicit postprocess filename above and the
    # preserved failed evaluator result as separate evidence.
    evidence_files.setdefault("evaluation_result", post_path)
    comparison_path = _output_path(comparison, outputs, "candidate comparison sheet")
    candidate_root = _output_path(candidate_output, outputs, "candidate output root", directory=True, must_exist=False)
    if candidate_root.exists() or candidate_root.is_symlink():
        _fail(f"candidate delivery must be fresh and append-only: {candidate_root}")
    review_path = post_path.parent / "manual_visual_review-v1.json"
    if review_path.is_symlink():
        _fail(f"manual visual review evidence is symlinked: {review_path}")
    if review_path.is_file():
        review = _load_json(review_path, "existing manual visual review")
        if review.get("decision") != manual_visual_decision:
            _fail("existing manual visual review decision differs")
    elif review_path.exists():
        _fail(f"manual visual review evidence is not a regular file: {review_path}")
    else:
        review = {
            "schema_version": "longsplat-converted-eval-manual-visual-review-v1",
            "decision": manual_visual_decision,
            "delivery_quality": False,
            "accepted": False,
            "supersplat": False,
            "note": manual_visual_note,
            "scope": "technical same-camera conversion review only; not SuperSplat acceptance",
            "postprocess_result": _identity(post_path, "postprocess result"),
            "comparison_sheet": _identity(comparison_path, "candidate comparison sheet"),
        }
        _write_json(review_path, review)
    evidence_files["manual_visual_review.json"] = review_path
    context = dict(converted) if isinstance(converted, Mapping) else {}
    context["manual_visual_review"] = _identity(review_path, "manual visual review")
    context["known_limitations"] = list(result.get("known_limitations", []))
    try:
        return create_candidate_delivery(
            converted_ply=result["conversion"]["technical_ply"]["path"],
            output_dir=candidate_root,
            authority_manifest=manifest_path,
            evidence_files=evidence_files,
            comparison_sheet=comparison_path,
            provenance_context=context,
        )
    except AcceptanceDeliveryError as exc:
        _fail(str(exc))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only streaming recovery for converted-evaluation PNG evidence")
    parser.add_argument("--route-root", required=True, type=Path)
    parser.add_argument("--containment-root", type=Path, help="verified dynamic run root for all evidence paths")
    parser.add_argument("--authority-manifest", required=True, type=Path)
    parser.add_argument("--candidate-only", action="store_true", help="package an already passing postprocess result; do not decode renders")
    parser.add_argument("--postprocess-result", type=Path)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument("--manual-visual-decision", choices=("pass", "needs_review", "fail"), default="needs_review")
    parser.add_argument("--manual-visual-note", default="")
    parser.add_argument("--conversion-result", type=Path)
    parser.add_argument("--converted-ply", type=Path)
    parser.add_argument("--failed-evaluation-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--native-visual-review", type=Path)
    parser.add_argument("--conversion-release-decision", type=Path)
    parser.add_argument("--segment-provenance", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.candidate_only:
            if args.postprocess_result is None or args.candidate_output is None:
                _fail("--candidate-only requires --postprocess-result and --candidate-output")
            result = create_candidate_from_postprocess(
                authority_manifest_path=args.authority_manifest,
                postprocess_result_path=args.postprocess_result,
                candidate_output=args.candidate_output,
                route_root=args.route_root,
                containment_root=args.containment_root,
                manual_visual_decision=args.manual_visual_decision,
                manual_visual_note=args.manual_visual_note,
            )
        else:
            required = {"--conversion-result": args.conversion_result, "--converted-ply": args.converted_ply, "--failed-evaluation-root": args.failed_evaluation_root, "--output-root": args.output_root}
            missing = [name for name, value in required.items() if value is None]
            if missing:
                _fail(f"missing required recovery arguments: {', '.join(missing)}")
            result = run_postprocess(
                authority_manifest_path=args.authority_manifest,
                conversion_result_path=args.conversion_result,
                converted_ply_path=args.converted_ply,
                failed_evaluation_root=args.failed_evaluation_root,
                output_root=args.output_root,
                route_root=args.route_root,
                containment_root=args.containment_root,
                native_visual_review=args.native_visual_review,
                conversion_release_decision=args.conversion_release_decision,
                segment_provenance=args.segment_provenance,
            )
    except (ConvertedEvalPostprocessError, AuthorityManifestError, AcceptanceDeliveryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
