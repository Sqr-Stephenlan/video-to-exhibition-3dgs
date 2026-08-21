"""Append-only import and sealing of an explicitly accepted candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from .authority_manifest import AuthorityManifestError, validate_authority_manifest
from .conversion_evidence_schema import (
    ConvertedEvaluationSchemaError,
    normalize_converted_evaluation,
)


class AcceptanceDeliveryError(ValueError):
    """The acceptance import cannot be safely sealed."""


def _fail(message: str) -> None:
    raise AcceptanceDeliveryError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_components(path: Path, label: str) -> None:
    probe = Path(path.anchor) if path.is_absolute() else Path.cwd()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for component in parts:
        probe /= component
        if probe.is_symlink():
            _fail(f"{label} traverses a symlink: {probe}")


def _identity(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is missing or symlinked: {path}")
    _reject_symlink_components(path, label)
    return {"path": str(path.resolve()), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _copy_verified(source: Path, destination: Path, label: str) -> dict[str, Any]:
    source_identity = _identity(source, f"{label} source")
    if destination.exists() or destination.is_symlink():
        _fail(f"accepted delivery destination must be fresh: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination_identity = _identity(destination, f"{label} destination")
    if (source_identity["sha256"], source_identity["size_bytes"]) != (
        destination_identity["sha256"],
        destination_identity["size_bytes"],
    ):
        _fail(f"{label} source/destination identity differs")
    return {"label": label, "source": source_identity, "destination": destination_identity}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not valid JSON: {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must be an object: {path}")
    return value


def _point_count(path: Path) -> int:
    try:
        from plyfile import PlyData

        ply = PlyData.read(str(path))
        return int(ply["vertex"].count)
    except Exception as exc:  # pragma: no cover - exercised by real evidence
        _fail(f"cannot read accepted PLY vertex count: {exc}")


def _contact_sheet(sources: Sequence[Path], output: Path) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - route environment supplies cv2
        _fail(f"OpenCV is required for the SuperSplat contact sheet: {exc}")
    images = []
    for source in sources:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            _fail(f"SuperSplat screenshot cannot be decoded: {source}")
        images.append(image)
    tile_width = 640
    margin = 16
    label_height = 34
    tiles = []
    for index, image in enumerate(images, start=1):
        height, width = image.shape[:2]
        scale = min(tile_width / width, 1.0)
        resized = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        tile = np.full((resized.shape[0] + label_height, tile_width, 3), 255, dtype=np.uint8)
        tile[label_height : label_height + resized.shape[0], : resized.shape[1]] = resized
        cv2.putText(tile, f"SuperSplat view {index:02d}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    height = max(tile.shape[0] for tile in tiles)
    canvas = np.full((height + 2 * margin, len(tiles) * tile_width + (len(tiles) + 1) * margin, 3), 255, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        x = margin + index * (tile_width + margin)
        canvas[margin : margin + tile.shape[0], x : x + tile.shape[1]] = tile
    if not cv2.imwrite(str(output), canvas):
        _fail(f"cannot write SuperSplat contact sheet: {output}")
    return {"path": str(output.resolve()), "sha256": _sha256(output), "size_bytes": output.stat().st_size}


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_sums(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS.txt":
            rows.append(f"{_sha256(path)}  {path.relative_to(root)}")
    sums = "\n".join(rows) + "\n"
    (root / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    for line in sums.splitlines():
        digest, relative = line.split("  ", 1)
        path = root / relative
        if _sha256(path) != digest:
            _fail(f"SHA256SUMS verification failed: {relative}")
    return {"path": str((root / "SHA256SUMS.txt").resolve()), "file_count": len(rows), "verified": True}


def create_accepted_delivery(
    *,
    candidate_ply: str | Path,
    output_dir: str | Path,
    authority_manifest: Mapping[str, Any] | str | Path,
    screenshots: Sequence[str | Path],
    evidence_files: Mapping[str, str | Path],
    user_acceptance_source: str,
    provenance_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy and seal one accepted candidate without changing its source.

    ``evidence_files`` is intentionally an explicit allowlist.  The function
    never recursively copies an attempt directory and never overwrites an
    existing accepted package.
    """

    try:
        authority = validate_authority_manifest(authority_manifest)
    except AuthorityManifestError as exc:
        _fail(str(exc))
    if not user_acceptance_source.strip():
        _fail("user_acceptance_source is required")
    if len(screenshots) != 3:
        _fail("exactly three SuperSplat screenshots are required")
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        _fail(f"accepted delivery must be a fresh sibling directory: {root}")
    root = root.absolute()
    root.mkdir(parents=True, exist_ok=False)
    bindings = []
    bindings.append(_copy_verified(Path(candidate_ply), root / "point_cloud.ply", "candidate PLY"))
    for name, source_value in sorted(evidence_files.items()):
        if Path(name).name != name or name in {"point_cloud.ply", "SHA256SUMS.txt"}:
            _fail(f"unsafe accepted evidence destination: {name}")
        bindings.append(_copy_verified(Path(source_value), root / name, f"evidence {name}"))

    screenshot_records = []
    screenshot_paths = [Path(value) for value in screenshots]
    for index, source in enumerate(screenshot_paths, start=1):
        destination = root / f"supersplat_view_{index:02d}.png"
        screenshot_records.append(_copy_verified(source, destination, f"SuperSplat screenshot {index:02d}"))
    contact = _contact_sheet(screenshot_paths, root / "SUPERSPLAT_EVIDENCE.png")

    point_cloud = root / "point_cloud.ply"
    ply_identity = _identity(point_cloud, "accepted point_cloud.ply")
    points = _point_count(point_cloud)
    dimensions = authority["camera_dimensions"]
    profile = authority["profile"]
    status = authority["status"]
    if status["held_out"]:
        _fail("accepted delivery cannot claim held-out evidence in this slice")
    acceptance = {
        "schema_version": "longsplat-supersplat-acceptance-v1",
        "accepted": True,
        "supersplat": True,
        "viewer": "SuperSplat v2.32.3",
        "three_view_manual_acceptance": True,
        "user_acceptance_source": user_acceptance_source,
        "identity_binding_method": "user assertion + exact point count match",
        "screenshot_embeds_ply_sha": False,
        "held_out": False,
        "metrics_scope": "training_views_only",
        "point_cloud": {**ply_identity, "vertices": points},
        "camera_count": authority["camera_count"],
        "camera_order": authority["camera_order"],
        "camera_dimensions": dimensions,
        "screenshot_records": [record["destination"] for record in screenshot_records],
        "contact_sheet": contact,
        "limitations": [
            "outer/peripheral stretching",
            "local floating points",
            "some highlights are over-bright",
            "screenshots include auxiliary grid/axes",
            "viewer object may have been renamed by the user",
            "no held-out evaluation",
        ],
    }
    _write_json(root / "SUPERSPLAT_ACCEPTANCE.json", acceptance)

    structural = _load_json(root / "structural_report.json", "structural report") if (root / "structural_report.json").is_file() else {}
    ab_metrics = _load_json(root / "ab_metrics.json", "A/B metrics") if (root / "ab_metrics.json").is_file() else {}
    provenance = {
        "schema_version": "longsplat-final-delivery-provenance-v1",
        "accepted": True,
        "supersplat": True,
        "held_out": False,
        "training_views_only": True,
        "training_source": {
            "source_video": authority["manifest"]["source_video"],
            "plan": authority["manifest"]["plan"],
            "static_contract": authority["manifest"]["static_contract"],
            "training_input": authority["manifest"]["training_input"],
            "training_model": authority["manifest"]["training_model"],
            "camera_count": authority["camera_count"],
            "camera_order": authority["camera_order"],
            "camera_dimensions": dimensions,
            "pose_contract": authority["manifest"]["pose_contract"],
            "colmap": "per-video COLMAP",
            "camera_model": "centered PINHOLE",
            "external_fixed_pose": True,
            "depth_source": "disabled",
            "mast3r_dust3r_vda": "not in production chain",
            **dict(provenance_context or {}),
        },
        "checkpoint": authority["manifest"]["checkpoint"],
        "mlp_identities": authority["manifest"]["checkpoint"]["files"],
        "conversion": {
            "profile_id": authority["profile_id"],
            "profile": profile,
            "frozen_single_conversion": True,
            "converted_point_cloud": ply_identity,
        },
        "native_render_evidence": authority["manifest"]["native_render_evidence"],
        "manual_acceptance": {
            "source": user_acceptance_source,
            "viewer": acceptance["viewer"],
            "identity_binding_method": acceptance["identity_binding_method"],
            "screenshot_embeds_ply_sha": False,
            "point_count_exact_match": True,
        },
        "known_limitations": acceptance["limitations"],
        "source_destination_bindings": bindings + screenshot_records,
    }
    _write_json(root / "PROVENANCE.json", provenance)
    report = "\n".join(
        [
            "# ACCEPTED_DELIVERY",
            "",
            "This package is an append-only final delivery for the explicitly accepted candidate.",
            "",
            f"- point cloud: `{point_cloud}`",
            f"- SHA-256: `{ply_identity['sha256']}`",
            f"- size: `{ply_identity['size_bytes']}` bytes",
            f"- vertices: `{points}`",
            f"- standard 3DGS fields: 62 float32 fields; structural report: `{structural.get('schema', 'present')}`",
            f"- conversion profile: `{authority['profile_id']}`; iteration: `{profile['checkpoint_iteration']}`",
            f"- training cameras: `{authority['camera_count']}`; dimensions: `{dimensions['width']}x{dimensions['height']}`",
            "- held-out: `false`; metrics scope: `training_views_only`",
            "",
            "## SuperSplat manual evidence",
            "",
            "User explicitly accepted three different free-view screenshots in SuperSplat v2.32.3. Identity is bound by the user assertion and exact point-count match; the screenshots do not embed a cryptographic PLY SHA.",
            f"- acceptance record: `{root / 'SUPERSPLAT_ACCEPTANCE.json'}`",
            f"- contact sheet: `{root / 'SUPERSPLAT_EVIDENCE.png'}`",
            f"- A/B evidence: `{root / 'ab_metrics.json'}`",
            f"- converted-vs-GT PSNR mean/min: `{ab_metrics.get('converted_vs_gt', {}).get('psnr_db_mean')}` / `{ab_metrics.get('converted_vs_gt', {}).get('psnr_db_min')}`",
            f"- converted-vs-GT SSIM mean/min: `{ab_metrics.get('converted_vs_gt', {}).get('ssim_mean')}` / `{ab_metrics.get('converted_vs_gt', {}).get('ssim_min')}`",
            f"- native-vs-converted PSNR mean: `{ab_metrics.get('native_vs_converted', {}).get('psnr_db_mean')}`",
            f"- native-vs-converted SSIM mean: `{ab_metrics.get('native_vs_converted', {}).get('ssim_mean')}`",
            "",
            "## Known limitations",
            "",
            *[f"- {item}" for item in acceptance["limitations"]],
            "",
            "This is an ACCEPTED_DELIVERY, not a claim of perfection or held-out generalization.",
        ]
    ) + "\n"
    (root / "FINAL_DELIVERY_REPORT.md").write_text(report, encoding="utf-8")
    sums = _write_sums(root)
    return {
        "root": str(root),
        "point_cloud": ply_identity,
        "vertices": points,
        "acceptance": str(root / "SUPERSPLAT_ACCEPTANCE.json"),
        "provenance": str(root / "PROVENANCE.json"),
        "report": str(root / "FINAL_DELIVERY_REPORT.md"),
        "contact_sheet": contact,
        "sha256sums": sums,
    }


def create_user_asserted_accepted_delivery(
    *,
    candidate_ply: str | Path,
    output_dir: str | Path,
    authority_manifest: Mapping[str, Any] | str | Path,
    evidence_files: Mapping[str, str | Path],
    user_acceptance_source: str,
    provenance_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal a user-accepted candidate when screenshot files are unavailable.

    The explicit user declaration is retained as acceptance state, while the
    absent screenshot archive remains a first-class, machine-readable
    limitation.  This function never fabricates screenshots or a contact
    sheet and never changes the candidate source PLY.
    """

    try:
        authority = validate_authority_manifest(authority_manifest)
    except AuthorityManifestError as exc:
        _fail(str(exc))
    if not user_acceptance_source.strip():
        _fail("user_acceptance_source is required")
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        _fail(f"accepted delivery must be a fresh sibling directory: {root}")
    root = root.absolute()
    root.mkdir(parents=True, exist_ok=False)

    bindings = [_copy_verified(Path(candidate_ply), root / "point_cloud.ply", "candidate PLY")]
    for name, source_value in sorted(evidence_files.items()):
        if Path(name).name != name or name in {"point_cloud.ply", "SHA256SUMS.txt"}:
            _fail(f"unsafe accepted evidence destination: {name}")
        bindings.append(_copy_verified(Path(source_value), root / name, f"evidence {name}"))

    point_cloud = root / "point_cloud.ply"
    ply_identity = _identity(point_cloud, "accepted point_cloud.ply")
    points = _point_count(point_cloud)
    dimensions = authority["camera_dimensions"]
    profile = authority["profile"]
    status = "ACCEPTED_BY_USER_PENDING_SCREENSHOT_ARCHIVE"
    limitations = [
        "dynamic-person ghosting",
        "local breakage/ghosting around the 2.982s frame_000140 to frame_000146 cross-gap",
        "training-view-only evaluation; held-out is false",
        "the original converted evaluator exited with SIGKILL (-9); the cause is unknown, while its complete PNG output passed CPU streaming postprocess",
        "screenshot file evidence for this candidate is missing; no screenshot or contact sheet was fabricated",
    ]
    acceptance = {
        "schema_version": "longsplat-supersplat-acceptance-v2",
        "status": status,
        "accepted": True,
        "supersplat": True,
        "user_asserted_manual_acceptance": True,
        "viewer": "SuperSplat v2.32.3",
        "three_view_manual_acceptance": True,
        "user_acceptance_source": user_acceptance_source,
        "identity_binding_method": "user assertion + exact candidate PLY SHA/size/point-count match",
        "screenshot_file_evidence": "missing",
        "screenshot_evidence_complete": False,
        "screenshot_embeds_ply_sha": False,
        "screenshot_records": [],
        "contact_sheet": None,
        "held_out": False,
        "metrics_scope": "training_views_only",
        "point_cloud": {**ply_identity, "vertices": points},
        "camera_count": authority["camera_count"],
        "camera_order": authority["camera_order"],
        "camera_dimensions": dimensions,
        "known_limitations": limitations,
    }
    _write_json(root / "SUPERSPLAT_ACCEPTANCE.json", acceptance)

    context = dict(provenance_context or {})
    provenance = {
        "schema_version": "longsplat-final-delivery-provenance-v2",
        "status": status,
        "accepted": True,
        "supersplat": True,
        "user_asserted_manual_acceptance": True,
        "screenshot_file_evidence": "missing",
        "screenshot_evidence_complete": False,
        "held_out": False,
        "training_views_only": True,
        "source_video": authority["manifest"]["source_video"],
        "plan": authority["manifest"]["plan"],
        "static_contract": authority["manifest"]["static_contract"],
        "training_input": authority["manifest"]["training_input"],
        "training_model": authority["manifest"]["training_model"],
        "checkpoint": authority["manifest"]["checkpoint"],
        "mlp_identities": authority["manifest"]["checkpoint"]["files"],
        "cameras": {
            "count": authority["camera_count"],
            "order": authority["camera_order"],
            "dimensions": dimensions,
            "camera_model": "centered PINHOLE",
        },
        "pose_contract": authority["manifest"]["pose_contract"],
        "training_semantics": {
            "colmap": "per-video COLMAP",
            "external_fixed_pose": True,
            "rgb_only": True,
            "depth_source": "disabled",
            "mast3r_dust3r_vda": "not in production chain",
        },
        "conversion": {
            "profile_id": authority["profile_id"],
            "profile": profile,
            "frozen_single_conversion": True,
            "converted_point_cloud": ply_identity,
        },
        "native_render_evidence": authority["manifest"]["native_render_evidence"],
        "manual_acceptance": {
            "source": user_acceptance_source,
            "viewer": acceptance["viewer"],
            "identity_binding_method": acceptance["identity_binding_method"],
            "point_count_exact_match": True,
            "screenshot_file_evidence": "missing",
        },
        "known_limitations": limitations,
        "source_destination_bindings": bindings,
        **context,
    }
    _write_json(root / "PROVENANCE.json", provenance)

    report = "\n".join(
        [
            "# ACCEPTED_DELIVERY",
            "",
            f"Status: `{status}`.",
            "",
            "The user explicitly declared that this exact candidate was manually accepted in real SuperSplat. The screenshot files were not found in the authoritative route and are therefore not represented or reconstructed here.",
            "",
            f"- point cloud: `{point_cloud}`",
            f"- SHA-256: `{ply_identity['sha256']}`",
            f"- size: `{ply_identity['size_bytes']}` bytes",
            f"- vertices: `{points}`",
            f"- conversion profile: `{authority['profile_id']}`; iteration: `{profile['checkpoint_iteration']}`",
            f"- training cameras: `{authority['camera_count']}`; dimensions: `{dimensions['width']}x{dimensions['height']}`",
            "- held-out: `false`; metrics scope: `training_views_only`",
            "- user_asserted_manual_acceptance: `true`",
            "- screenshot_file_evidence: `missing`",
            "- screenshot_evidence_complete: `false`",
            "",
            "## Known limitations",
            "",
            *[f"- {item}" for item in limitations],
            "",
            "This package records user acceptance without claiming that the screenshot archive is complete.",
        ]
    ) + "\n"
    (root / "FINAL_DELIVERY_REPORT.md").write_text(report, encoding="utf-8")
    sums = _write_sums(root)
    return {
        "root": str(root),
        "status": status,
        "point_cloud": ply_identity,
        "vertices": points,
        "acceptance": str(root / "SUPERSPLAT_ACCEPTANCE.json"),
        "provenance": str(root / "PROVENANCE.json"),
        "report": str(root / "FINAL_DELIVERY_REPORT.md"),
        "screenshot_file_evidence": "missing",
        "sha256sums": sums,
        "accepted": True,
        "supersplat": True,
    }


def create_candidate_delivery(
    *,
    converted_ply: str | Path,
    output_dir: str | Path,
    authority_manifest: Mapping[str, Any] | str | Path,
    evidence_files: Mapping[str, str | Path],
    comparison_sheet: str | Path,
    provenance_context: Mapping[str, Any] | None = None,
    route_root: str | Path | None = None,
    containment_root: str | Path | None = None,
    automated_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal a technical candidate while leaving application acceptance open.

    This is deliberately separate from :func:`create_accepted_delivery`:
    conversion/evaluation evidence can make a candidate technically
    deliverable, but cannot manufacture SuperSplat or user acceptance.
    ``evidence_files`` is an explicit allowlist and the destination is always
    a fresh append-only directory.
    """

    try:
        authority = validate_authority_manifest(
            authority_manifest,
            route_root=route_root,
            containment_root=containment_root,
        )
    except AuthorityManifestError as exc:
        _fail(str(exc))
    evaluation_value = evidence_files.get("evaluation_result") or evidence_files.get("evaluation_result.json")
    if evaluation_value is None:
        _fail("candidate delivery requires evaluation_result evidence")
    evaluation = _load_json(Path(evaluation_value), "converted evaluation result")
    try:
        normalized_evaluation = normalize_converted_evaluation(
            evaluation,
            expected_identity={
                "camera_count": authority["camera_count"],
                "camera_order": authority["camera_order"],
                "camera_dimensions": authority["camera_dimensions"],
            },
        )
    except ConvertedEvaluationSchemaError as exc:
        _fail(f"candidate delivery requires a structural converted-evaluation pass: {exc}")
    if normalized_evaluation["STRUCTURAL_EVALUATION_PASS"] is not True:
        _fail("candidate delivery requires a structural converted-evaluation pass; visual quality remains advisory")
    root = Path(output_dir)
    if root.exists() or root.is_symlink():
        _fail(f"candidate delivery must be a fresh sibling directory: {root}")
    root = root.absolute()
    root.mkdir(parents=True, exist_ok=False)

    bindings = [_copy_verified(Path(converted_ply), root / "point_cloud.ply", "candidate PLY")]
    for name, source_value in sorted(evidence_files.items()):
        if Path(name).name != name or name in {"point_cloud.ply", "SHA256SUMS.txt"}:
            _fail(f"unsafe candidate evidence destination: {name}")
        bindings.append(_copy_verified(Path(source_value), root / name, f"candidate evidence {name}"))
    comparison = _copy_verified(Path(comparison_sheet), root / "fixed_gt_native_converted_contact_sheet.png", "candidate comparison sheet")
    bindings.append(comparison)

    point_cloud = root / "point_cloud.ply"
    ply_identity = _identity(point_cloud, "candidate point_cloud.ply")
    points = _point_count(point_cloud)
    manifest = authority["manifest"]
    dimensions = authority["camera_dimensions"]
    profile = authority["profile"]
    context = dict(provenance_context or {})
    known_limitations = list(
        context.pop(
            "known_limitations",
            [
                "training-view-only evaluation; held-out is false",
                "dynamic-person ghosting",
                "local breakage/ghosting around the 2.982s frame_000140 to frame_000146 cross-gap",
                "no real SuperSplat three-view manual acceptance has been imported",
            ],
        )
    )
    automated = automated_policy is not None
    candidate = {
        "schema_version": "longsplat-candidate-delivery-v2",
        "status": "AUTOMATED_TECHNICAL_DELIVERY" if automated else "TECHNICAL_CANDIDATE_PENDING_SUPERSPLAT",
        "accepted": False,
        "supersplat": False,
        "accepted_by_automated_policy": automated,
        "manual_visual_review": False if automated else True,
        "supersplat_format_compatible": True,
        "supersplat_runtime_verified": False,
        "screenshot_evidence": "not_required_by_current_user" if automated else "required",
        "three_view_manual_acceptance_required": not automated,
        "held_out": False,
        "training_views_only": True,
        "point_cloud": {**ply_identity, "vertices": points},
        "camera_count": authority["camera_count"],
        "camera_order": authority["camera_order"],
        "camera_dimensions": dimensions,
        "conversion_profile_id": authority["profile_id"],
        "same_camera_visual_pass": evaluation.get("SAME_CAMERA_VISUAL_PASS"),
        "visual_quality_pass": normalized_evaluation["visual_quality_pass"],
        "structural_evaluation_pass": normalized_evaluation["STRUCTURAL_EVALUATION_PASS"],
        "quality_advisories": evaluation.get("quality_advisories", []),
        "known_limitations": known_limitations,
        "comparison_sheet": comparison["destination"],
        "authority_manifest": authority["manifest_path"] or str(authority_manifest),
    }
    if automated:
        candidate["automated_policy"] = dict(automated_policy)
    _write_json(root / "candidate_manifest.json", candidate)

    provenance = {
        "schema_version": "longsplat-candidate-provenance-v2",
        "accepted": False,
        "supersplat": False,
        "accepted_by_automated_policy": automated,
        "manual_visual_review": False if automated else True,
        "supersplat_format_compatible": True,
        "supersplat_runtime_verified": False,
        "screenshot_evidence": "not_required_by_current_user" if automated else "required",
        "three_view_manual_acceptance_required": not automated,
        "held_out": False,
        "training_views_only": True,
        "source_video": manifest["source_video"],
        "plan": manifest["plan"],
        "static_contract": manifest["static_contract"],
        "training_input": manifest["training_input"],
        "training_model": manifest["training_model"],
        "checkpoint": manifest["checkpoint"],
        "mlp_identities": manifest["checkpoint"]["files"],
        "cameras": {
            "count": authority["camera_count"],
            "order": authority["camera_order"],
            "dimensions": dimensions,
            "camera_model": "centered PINHOLE",
        },
        "pose_contract": manifest["pose_contract"],
        "training_semantics": {
            "colmap": "per-video COLMAP",
            "external_fixed_pose": True,
            "rgb_only": True,
            "depth_source": "disabled",
            "mast3r_dust3r_vda": "not in production chain",
        },
        "conversion": {
            "profile_id": authority["profile_id"],
            "profile": profile,
            "frozen_single_conversion": True,
            "converted_point_cloud": ply_identity,
        },
        "native_render_evidence": manifest["native_render_evidence"],
        "evaluation": {
            "result": _identity(Path(evaluation_value), "evaluation result"),
            "same_camera_visual_pass": evaluation.get("SAME_CAMERA_VISUAL_PASS"),
            "visual_quality_pass": normalized_evaluation["visual_quality_pass"],
            "structural_evaluation_pass": normalized_evaluation["STRUCTURAL_EVALUATION_PASS"],
            "quality_advisories": evaluation.get("quality_advisories", []),
            "held_out": False,
            "metrics_scope": "training_views_only",
        },
        "known_limitations": known_limitations,
        "source_destination_bindings": bindings,
        **context,
    }
    if automated:
        provenance["automated_policy"] = dict(automated_policy)
    _write_json(root / "PROVENANCE.json", provenance)

    report_lines = [
        "# AUTOMATED_TECHNICAL_DELIVERY" if automated else "# CANDIDATE_DELIVERY",
        "",
        "Technical delivery sealed by the versioned automated policy; this does not claim a SuperSplat runtime load or human visual acceptance." if automated else "Technical candidate package; application acceptance remains explicitly pending real SuperSplat review.",
        "",
        f"- point cloud: `{point_cloud}`",
        f"- SHA-256: `{ply_identity['sha256']}`",
        f"- size: `{ply_identity['size_bytes']}` bytes",
        f"- vertices: `{points}`",
        f"- conversion profile: `{authority['profile_id']}` (local policy; official converter default is not asserted)",
        f"- cameras: `{authority['camera_count']}` ordered training cameras at `{dimensions['width']}x{dimensions['height']}`",
        "- held-out: `false`; metrics scope: `training_views_only`",
        f"- same-camera evaluation: `{evaluation.get('SAME_CAMERA_VISUAL_PASS')}`",
        f"- accepted_by_automated_policy: `{automated}`; manual_visual_review: `{False if automated else True}`",
        "- supersplat_format_compatible: `true`; supersplat_runtime_verified: `false`",
        f"- screenshot evidence: `{'not_required_by_current_user' if automated else 'required'}`",
        "",
        "## Evidence",
        "",
        f"- authority manifest: `{root / 'authority_manifest.json' if (root / 'authority_manifest.json').is_file() else authority['manifest_path'] or authority_manifest}`",
        f"- fixed/GT/native/converted comparison: `{root / 'fixed_gt_native_converted_contact_sheet.png'}`",
        "",
        "## Known limitations",
        "",
        *[f"- {item}" for item in known_limitations],
        "",
        "This package is an automated technical delivery, not a claim of SuperSplat runtime verification." if automated else "This package is not an accepted delivery and must not be represented as SuperSplat-reviewed.",
    ]
    (root / "CANDIDATE_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    sums = _write_sums(root)
    return {
        "root": str(root),
        "point_cloud": ply_identity,
        "vertices": points,
        "candidate_manifest": str(root / "candidate_manifest.json"),
        "provenance": str(root / "PROVENANCE.json"),
        "report": str(root / "CANDIDATE_REPORT.md"),
        "comparison_sheet": comparison["destination"],
        "sha256sums": sums,
        "accepted": False,
        "supersplat": False,
    }
