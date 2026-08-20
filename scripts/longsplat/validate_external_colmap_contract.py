#!/usr/bin/env python3
"""CPU-only validation for a dynamic per-video external-COLMAP contract.

Historical Set A values are fixtures, not pipeline defaults.  This validator
reads the current staging manifest/camera contract and therefore accepts any
positive canonical dimensions and any evidence-backed intrinsics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROUTE = Path(__file__).resolve().parents[2]
if __package__:
    from .camera_staging import CameraParameters, validate_centered_pinhole
    from .colmap_contract import parse_cameras_text
    from .longsplat_input import (
        compare_colmap_models,
        parse_colmap_model_binary,
        parse_colmap_model_text,
        _validate_parent_binding_digest,
        validate_future_smoke_plan,
        validate_colmap_model,
        model_counts,
    )
    from .pipeline_contract import stable_sha256
else:  # Direct script invocation is kept working for the existing runbook.
    sys.path.insert(0, str(ROUTE))
    from scripts.longsplat.camera_staging import CameraParameters, validate_centered_pinhole  # type: ignore  # noqa: E402
    from scripts.longsplat.colmap_contract import parse_cameras_text  # type: ignore  # noqa: E402
    from scripts.longsplat.longsplat_input import (  # type: ignore  # noqa: E402
        compare_colmap_models,
        parse_colmap_model_binary,
        parse_colmap_model_text,
        _validate_parent_binding_digest,
        validate_future_smoke_plan,
        validate_colmap_model,
        model_counts,
    )
    from scripts.longsplat.pipeline_contract import stable_sha256  # type: ignore  # noqa: E402

sys.path.insert(0, str(ROUTE / "third_party/LongSplat"))
from utils.external_colmap_pose import (  # noqa: E402
    camera_center_from_c2w,
    parse_colmap_image_names_binary,
    parse_colmap_image_names_text,
    parse_colmap_poses_text,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    staging_path = root / "staging_manifest.json"
    if not staging_path.is_file():
        raise AssertionError(f"missing staging manifest: {staging_path}")
    manifest = json.loads(staging_path.read_text(encoding="utf-8"))
    contract = manifest.get("camera_contract")
    contract_path = manifest.get("camera_contract_path")
    if contract is None and isinstance(contract_path, str):
        contract = json.loads((root / contract_path).read_text(encoding="utf-8"))
    if contract is None:
        candidate = root / "camera_staging" / "camera_contract-v1.json"
        if candidate.is_file():
            contract = json.loads(candidate.read_text(encoding="utf-8"))
    return manifest, contract or {}


def _records(manifest: Mapping[str, Any], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    # The final contract is authoritative: it must refer to undistorted output
    # pixels, while manifest image_records may still describe earlier stages.
    values = contract.get("frames") or manifest.get("image_records") or manifest.get("frames")
    if not isinstance(values, list) or not values:
        raise AssertionError("staging contract has no image/frame records")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise AssertionError(f"frame record {index} is not an object")
        name = value.get("staged_name", value.get("name", value.get("frame_id")))
        if not isinstance(name, str) or not name:
            raise AssertionError(f"frame record {index} has no dynamic name")
        result.append(dict(value, staged_name=name))
    names = [record["staged_name"] for record in result]
    if len(set(names)) != len(names):
        raise AssertionError("staged image names are not unique")
    return result


def _camera_from_contract(sparse: Path, contract: Mapping[str, Any]) -> CameraParameters:
    cameras = parse_cameras_text(sparse / "cameras.txt")
    if len(cameras) != 1:
        raise AssertionError(f"expected exactly one sparse camera, found {len(cameras)}")
    camera = CameraParameters.from_colmap_record(cameras[0])
    declared = contract.get("camera")
    if not isinstance(declared, Mapping):
        raise AssertionError("camera contract must declare one camera")
    for key in ("model", "width", "height"):
        if declared.get(key) != camera.to_dict().get(key):
            raise AssertionError(f"camera {key} mismatch: sparse={camera.to_dict().get(key)}, contract={declared.get(key)}")
    for key in ("fx", "fy", "cx", "cy"):
        if not np.isclose(float(declared.get(key)), float(camera.to_dict()[key]), rtol=0.0, atol=1e-6):
            raise AssertionError(f"camera {key} mismatch")
    if list(declared.get("distortion", [])) != list(camera.distortion):
        raise AssertionError("camera distortion mismatch")
    return camera


def _find_model_dir(root: Path) -> Path:
    candidates: list[Path] = []
    for base in (root / "input" / "sparse", root / "sparse", root / "undistorted" / "sparse"):
        if (base / "cameras.txt").is_file():
            candidates.append(base)
        elif base.is_dir():
            children = sorted(
                child for child in base.iterdir()
                if child.is_dir() and (child / "cameras.txt").is_file()
            )
            if len(children) == 1:
                candidates.append(children[0])
    if len(candidates) != 1:
        raise AssertionError(f"expected one TXT sparse model, found {len(candidates)}")
    return candidates[0]


def _training_root_path(root: Path, value: str | Path, *, label: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise AssertionError(f"{label} escapes training input root: {resolved}") from exc
    return resolved


def _verify_code_identity(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AssertionError(f"{label} must be an object")
    declared = value.get("code_identity_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise AssertionError(f"{label} code_identity_sha256 is required")
    unsigned = dict(value)
    unsigned.pop("code_identity_sha256", None)
    if stable_sha256(unsigned) != declared:
        raise AssertionError(f"{label} code identity hash failed")
    return declared


def validate_training_input(
    root: Path,
    *,
    containment_root: str | Path | None = None,
    allow_producer_code_drift: bool = False,
) -> dict[str, Any]:
    """Validate the CPU-produced training-specific reference input only."""

    manifest, contract = _load_contract(root)
    if manifest.get("scope") != "LongSplat training-specific model":
        raise AssertionError("training validator requires the LongSplat training-specific scope")
    if manifest.get("observation_rewrite") is not True:
        raise AssertionError("training input must explicitly declare observation_rewrite=true")
    if manifest.get("accepted") is not False or manifest.get("delivery_reachable") is not False:
        raise AssertionError("training input cannot be accepted or delivery reachable in this stage")
    if manifest.get("gpu_invoked") is not False:
        raise AssertionError("training input stage must declare gpu_invoked=false")
    source_video_sha = manifest.get("source_video_sha256")
    binding_sha = manifest.get("canonical_media_binding_sha256")
    pixels_sha = manifest.get("canonical_media_pixels_sha256")
    for label, value in (
        ("source_video_sha256", source_video_sha),
        ("canonical_media_binding_sha256", binding_sha),
        ("canonical_media_pixels_sha256", pixels_sha),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise AssertionError(f"{label} must be a 64-character SHA")
    parent_binding = manifest.get("parent_binding")
    if not isinstance(parent_binding, Mapping):
        raise AssertionError("training parent_binding is required")
    try:
        parent_binding_sha = _validate_parent_binding_digest(parent_binding)
    except Exception as exc:
        raise AssertionError(str(exc)) from exc
    if manifest.get("parent_binding_sha256") != parent_binding_sha:
        raise AssertionError("training parent_binding SHA binding failed")
    if parent_binding.get("source_video_sha256") != source_video_sha:
        raise AssertionError("training parent_binding source SHA differs from manifest")
    if parent_binding.get("canonical_media_binding_sha256") != binding_sha:
        raise AssertionError("training parent_binding canonical binding differs from manifest")
    if parent_binding.get("canonical_media_pixels_sha256") != pixels_sha:
        raise AssertionError("training parent_binding canonical pixels differ from manifest")
    contract_sha = contract.get("contract_sha256")
    if not isinstance(contract_sha, str) or manifest.get("camera_contract_sha256") != contract_sha:
        raise AssertionError("training camera contract stable SHA binding failed")
    unsigned_contract = dict(contract)
    unsigned_contract.pop("contract_sha256", None)
    if stable_sha256(unsigned_contract) != contract_sha:
        raise AssertionError("training camera contract stable SHA failed")
    records = manifest.get("image_records")
    if not isinstance(records, list) or not records:
        raise AssertionError("training input has no copied image records")
    names = [str(record.get("name", record.get("staged_name", ""))) for record in records]
    if len(names) != len(set(names)) or any(Path(name).name != name for name in names):
        raise AssertionError("training image names are not unique staged basenames")
    stems = [Path(name).stem for name in names]
    if len(stems) != len(set(stems)):
        raise AssertionError("training image names have duplicate stems across extensions")
    contract_names = list(contract.get("frame_names", []))
    if sorted(names) != sorted(contract_names):
        raise AssertionError("training image names differ from camera contract")
    camera_declared = contract.get("camera")
    if not isinstance(camera_declared, Mapping):
        raise AssertionError("training camera contract has no final camera")
    width = int(camera_declared.get("width", 0))
    height = int(camera_declared.get("height", 0))
    if width <= 0 or height <= 0 or camera_declared.get("model") != "PINHOLE":
        raise AssertionError("training camera must be positive PINHOLE")
    try:
        import cv2
    except ImportError as exc:
        raise AssertionError("OpenCV is required for training image validation") from exc
    verified_pixel_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise AssertionError("training image record is not an object")
        path = _training_root_path(root, str(record.get("path", "")), label="training image")
        if path.is_symlink() or not path.is_file():
            raise AssertionError(f"training image must be an independent file: {path}")
        expected_file_sha = record.get("file_sha256", record.get("sha256"))
        if not isinstance(expected_file_sha, str) or sha256_file(path) != expected_file_sha:
            raise AssertionError(f"training image file SHA failed: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim < 2 or image.shape[1] != width or image.shape[0] != height:
            raise AssertionError(f"training image dimensions failed: {path}")
        pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
        if pixel_sha != record.get("decoded_pixel_sha256"):
            raise AssertionError(f"training decoded pixel SHA failed: {path}")
        verified_pixel_records.append({"name": record["name"], "decoded_pixel_sha256": pixel_sha})
    aggregate = stable_sha256(sorted(verified_pixel_records, key=lambda item: item["name"]))
    if aggregate != manifest.get("final_staged_pixel_aggregate_sha256"):
        raise AssertionError("training image aggregate SHA failed")
    if aggregate != contract.get("staging", {}).get("final_staged_pixel_aggregate_sha256"):
        raise AssertionError("training image aggregate differs from camera contract")

    sparse = root / "sparse" / "0"
    if not sparse.is_dir():
        raise AssertionError(f"training sparse/0 is missing: {sparse}")
    text_model = parse_colmap_model_text(sparse)
    binary_model = parse_colmap_model_binary(sparse)
    validate_colmap_model(text_model, image_width=width, image_height=height, require_pinhole=True)
    validate_colmap_model(binary_model, image_width=width, image_height=height, require_pinhole=True)
    comparison = compare_colmap_models(text_model, binary_model, tolerance=1e-9)
    if len(text_model.cameras) != 1 or len(text_model.images) != len(records):
        raise AssertionError("training sparse model camera/image counts are wrong")
    model_names = sorted(image["name"] for image in text_model.images.values())
    if model_names != sorted(names):
        raise AssertionError("training sparse model names differ from copied images")
    camera = next(iter(text_model.cameras.values()))
    expected_params = [camera_declared.get(key) for key in ("fx", "fy", "cx", "cy")]
    if any(not np.isclose(float(left), float(right), rtol=0.0, atol=1e-8) for left, right in zip(camera["params"], expected_params)):
        raise AssertionError("training sparse K differs from camera contract")
    roundtrip_value = manifest.get("roundtrip_txt_evidence_path")
    if not isinstance(roundtrip_value, str):
        raise AssertionError("roundtrip TXT evidence path is required")
    roundtrip_dir = _training_root_path(root, roundtrip_value, label="roundtrip TXT")
    roundtrip_model = parse_colmap_model_text(roundtrip_dir)
    roundtrip_comparison = compare_colmap_models(text_model, roundtrip_model, tolerance=1e-9)
    if (root / "cameras_all_train.json").exists() or (root / "external_colmap_pose_contract.json").exists():
        raise AssertionError("training input stage must not create active Camera contracts")
    future_plan_value = manifest.get("future_smoke_plan_path")
    if not isinstance(future_plan_value, str):
        raise AssertionError("future smoke plan path is required")
    future_plan_path = _training_root_path(root, future_plan_value, label="future smoke plan")
    if not future_plan_path.is_file():
        raise AssertionError(f"future smoke plan is missing: {future_plan_path}")
    future_plan = json.loads(future_plan_path.read_text(encoding="utf-8"))
    validate_future_smoke_plan(
        future_plan,
        route_root=ROUTE,
        containment_root=containment_root,
        training_root=root,
        allow_producer_code_drift=allow_producer_code_drift,
    )
    if future_plan.get("parent_binding") != dict(parent_binding) or future_plan.get("parent_binding_sha256") != parent_binding_sha:
        raise AssertionError("future smoke plan parent_binding differs from training manifest")
    workload_profile = future_plan.get("workload_profile")
    if not isinstance(workload_profile, str):
        raise AssertionError("future smoke workload_profile is required")
    if manifest.get("future_smoke_profile") != workload_profile:
        raise AssertionError("training manifest future workload profile differs from plan")
    workload_iterations = future_plan.get("frozen_contract", {}).get("iterations")
    if manifest.get("future_smoke_iterations") != workload_iterations:
        raise AssertionError("training manifest future workload iterations differ from plan")
    code_identity_sha = _verify_code_identity(manifest.get("code_identity"), label="route code")
    tool_identity = manifest.get("tool_identity")
    if not isinstance(tool_identity, Mapping) or not isinstance(tool_identity.get("sha256"), str):
        raise AssertionError("COLMAP tool identity is required")
    static = {
        "status": "STATIC_LONGSPLAT_INPUT_REFERENCE_PASS",
        "computed_pass": True,
        "accepted": False,
        "delivery_reachable": False,
        "gpu_invoked": False,
        "validation_scope": "LongSplat training-specific model; static staging/COLMAP reference only; no active Camera objects",
        "scope": "LongSplat training-specific model",
        "observation_rewrite": True,
        "source_video_sha256": source_video_sha,
        "parent_binding": dict(parent_binding),
        "parent_binding_sha256": parent_binding_sha,
        "canonical_media_binding_sha256": binding_sha,
        "canonical_media_pixels_sha256": pixels_sha,
        "final_staged_pixel_aggregate_sha256": aggregate,
        "camera_contract_sha256": contract_sha,
        "camera": dict(camera_declared),
        "image_count": len(records),
        "image_names_verified": True,
        "independent_pixel_files_verified": len(records),
        "decoded_pixel_aggregate_verified": True,
        "txt_bin_roundtrip_verified": True,
        "model_counts": {
            "txt": model_counts(text_model),
            "bin": model_counts(binary_model),
            "roundtrip_txt": model_counts(roundtrip_model),
        },
        "txt_bin_comparison": comparison,
        "roundtrip_comparison": roundtrip_comparison,
        "track_referential_integrity_verified": True,
        "crop_provenance": manifest.get("crop_provenance"),
        "tool_identity": dict(tool_identity),
        "route_code_identity_sha256": code_identity_sha,
        "future_smoke_plan_path": manifest.get("future_smoke_plan_path"),
        "future_smoke_plan_schema_version": future_plan["schema_version"],
        "future_smoke_profile": workload_profile,
        "future_smoke_iterations": workload_iterations,
        "seed_contract_verified": True,
        "nested_backend_code_identity": dict(future_plan["nested_backend_code_identity"]),
        "cameras_all_train_created": False,
        "external_colmap_pose_contract_created": False,
        "active_camera_objects_created": False,
        "training_invoked": False,
        "render_invoked": False,
        "conversion_invoked": False,
        "quality_acceptance": "not_claimed; requires separately authorized GPU 100-it fixed native render",
    }
    return static


def validate(root: Path) -> dict[str, Any]:
    manifest, contract = _load_contract(root)
    if manifest.get("scope") == "LongSplat training-specific model":
        # The direct validator is a narrowly scoped legacy entrypoint.  Its
        # source and planned sibling model are both contained by this immutable
        # LongSplat-input attempt directory; canonical reconstruct callers pass
        # their signed run root explicitly instead.
        return validate_training_input(root, containment_root=root.parent)
    records = _records(manifest, contract)
    sparse = _find_model_dir(root)
    expected_names = [record["staged_name"] for record in records]
    source_video_sha = (
        manifest.get("source_video_sha256")
        or manifest.get("source", {}).get("sha256")
        or contract.get("source_video_sha256")
    )
    canonical_sha = manifest.get("canonical_media_sha256") or contract.get("canonical_media_sha256")
    if not isinstance(source_video_sha, str) or len(source_video_sha) != 64:
        raise AssertionError("current source video SHA is required")
    if not isinstance(canonical_sha, str) or len(canonical_sha) != 64:
        raise AssertionError("current canonical media SHA is required")

    for record in records:
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise AssertionError(f"final frame path missing for {record['staged_name']}")
        staged = Path(path_value)
        if not staged.is_file():
            staged = root / path_value
        if not staged.is_file():
            fallback = root / "input" / "images" / record["staged_name"]
            staged = fallback if fallback.is_file() else root / "images" / record["staged_name"]
        if not staged.is_file():
            raise AssertionError(f"final staged image missing: {staged}")
        expected_sha = record.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise AssertionError(f"final staged image SHA missing: {record['staged_name']}")
        if sha256_file(staged) != expected_sha:
            raise AssertionError(f"final staged image SHA failed: {staged}")
        try:
            import cv2
            image = cv2.imread(str(staged), cv2.IMREAD_UNCHANGED)
        except ImportError as exc:
            raise AssertionError("OpenCV is required for staged image validation") from exc
        declared_width = int(record.get("width", 0))
        declared_height = int(record.get("height", 0))
        if image is None or image.ndim < 2 or image.shape[1] != declared_width or image.shape[0] != declared_height:
            raise AssertionError(f"final staged image dimensions failed: {staged}")

    declared_contract_sha = contract.get("contract_sha256")
    if not isinstance(declared_contract_sha, str):
        raise AssertionError("camera contract SHA is required")
    unsigned_contract = dict(contract)
    unsigned_contract.pop("contract_sha256", None)
    if stable_sha256(unsigned_contract) != declared_contract_sha:
        raise AssertionError("camera contract SHA failed")

    camera = _camera_from_contract(sparse, contract)
    expected_width = int(contract.get("camera", {}).get("width", 0))
    expected_height = int(contract.get("camera", {}).get("height", 0))
    if expected_width <= 0 or expected_height <= 0:
        raise AssertionError("declared camera dimensions are required")
    camera_result = validate_centered_pinhole(
        camera,
        source_video_sha256=source_video_sha,
        canonical_media_sha256=canonical_sha,
        frame_names=expected_names,
        expected_width=expected_width,
        expected_height=expected_height,
    )
    txt_names = sorted(parse_colmap_image_names_text(sparse / "images.txt"))
    bin_names = sorted(parse_colmap_image_names_binary(sparse / "images.bin"))
    if txt_names != sorted(expected_names) or bin_names != sorted(expected_names):
        raise AssertionError("COLMAP images.bin/TXT names do not match dynamic staging order")
    poses = parse_colmap_poses_text(sparse / "images.txt")
    ordered_poses = [poses[name] for name in expected_names]
    centers = np.asarray([
        camera_center_from_c2w(pose["R_c2w"], pose["T_w2c"])
        for pose in ordered_poses
    ])
    if centers.shape != (len(records), 3) or not np.isfinite(centers).all():
        raise AssertionError("reference camera centers failed")
    output = {
        "status": "STATIC_INPUT_REFERENCE_PASS",
        "computed_pass": True,
        "accepted": False,
        "delivery_reachable": False,
        "validation_scope": "staging/COLMAP input/reference only; no active Camera objects",
        "external_colmap_reference_present": True,
        "mast3r_global_align_expected_skipped": True,
        "reference_pose_finite": True,
        "reference_camera_centers_finite": True,
        "reference_camera_count": len(records),
        "staged_image_sha_verified": len(records),
        "staged_image_names_verified": True,
        "staged_image_time_order_verified": True,
        "colmap_images_bin_txt_names_verified": True,
        "source_video_sha256": source_video_sha,
        "canonical_media_sha256": canonical_sha,
        "width": camera.width,
        "height": camera.height,
        "model": camera.model,
        "focal_x_px": camera.fx,
        "focal_y_px": camera.fy,
        "cx_px": camera.cx,
        "cy_px": camera.cy,
        "distortion": list(camera.distortion),
        "reference_pose_source": "COLMAP images.txt",
        "external_depth_reference": "disabled",
        "camera_contract_sha256": contract.get("contract_sha256"),
    }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the generic staged external-COLMAP input contract"
    )
    parser.add_argument("staging_root", type=Path)
    arguments = sys.argv[1:] if argv is None else argv
    parsed = parser.parse_args(arguments)
    root = parsed.staging_root.resolve()
    output = validate(root)
    output_path = root / "contract" / "static_contract.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != output:
            raise AssertionError(f"immutable contract output differs: {output_path}")
    else:
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
            handle.write("\n")
    print(json.dumps({
        **output,
        "static_contract_path": str(output_path),
        "static_contract_sha256": sha256_file(output_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
