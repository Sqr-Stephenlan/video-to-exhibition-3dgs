from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.longsplat.authority_manifest import (
    AuthorityManifestError,
    build_authority_manifest,
    load_authority_manifest,
    validate_authority_manifest,
    validate_profile,
    write_manifest_once,
)


def _fixture(tmp_path: Path, name: str, *, camera_names: list[str], width: int, height: int) -> tuple[dict, Path, Path]:
    root = tmp_path / name
    route = root / "route"
    (route / "outputs").mkdir(parents=True)
    (route / "third_party/LongSplat").mkdir(parents=True)
    source = root / f"{name}.mp4"
    source.write_bytes(f"source-{name}".encode())
    training_input = route / "outputs" / name / "training_input"
    model = route / "outputs" / name / "training_model"
    training_input.mkdir(parents=True)
    model.mkdir(parents=True)
    plan = training_input / "plan.json"
    static = training_input / "static.json"
    plan.write_text("{}\n", encoding="utf-8")
    static.write_text("{}\n", encoding="utf-8")
    train = model / "cameras_all_train.json"
    test = model / "cameras_all_test.json"
    train.write_text(
        json.dumps(
            [
                {"uid": index, "image_name": camera, "width": width, "height": height}
                for index, camera in enumerate(camera_names)
            ]
        ),
        encoding="utf-8",
    )
    test.write_text("[]\n", encoding="utf-8")
    pose = model / "external_colmap_pose_contract.json"
    pose.write_text(json.dumps({"image_names": camera_names}), encoding="utf-8")
    checkpoint = model / "point_cloud/iteration_30000"
    checkpoint.mkdir(parents=True)
    for filename in ("point_cloud.ply", "color_mlp.pt", "cov_mlp.pt", "opacity_mlp.pt"):
        (checkpoint / filename).write_bytes((name + filename).encode())
    native = route / "outputs" / name / "native"
    render_root = native / "renders"
    render_root.mkdir(parents=True)
    for index, _camera in enumerate(camera_names):
        image = np.full((height, width, 3), index + 1, dtype=np.uint8)
        assert cv2.imwrite(str(render_root / f"{index:05d}.png"), image)
    native_result = native / "result.json"
    native_postcheck = native / "postcheck.json"
    native_result.write_text("{}\n", encoding="utf-8")
    native_postcheck.write_text("{}\n", encoding="utf-8")
    manifest = build_authority_manifest(
        source_video=source,
        plan=plan,
        static_contract=static,
        training_input=training_input,
        training_model=model,
        pose_contract=pose,
        native_result=native_result,
        native_postcheck=native_postcheck,
        native_render_root=render_root,
    )
    path = write_manifest_once(route / "outputs" / name / "authority.json", manifest)
    return manifest, path, route


def test_two_synthetic_authority_fixtures_are_dynamic() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        root = Path(directory)
        first, first_path, first_route = _fixture(
            root, "fixture-one", camera_names=["alpha", "bravo", "charlie"], width=32, height=24
        )
        second, second_path, second_route = _fixture(
            root, "fixture-two", camera_names=["left-view", "right-view"], width=19, height=13
        )
        first_checked = load_authority_manifest(first_path, route_root=first_route)
        second_checked = load_authority_manifest(second_path, route_root=second_route)
        assert first_checked["camera_count"] == 3
        assert second_checked["camera_count"] == 2
        assert first_checked["camera_order"] == ["alpha", "bravo", "charlie"]
        assert second_checked["camera_order"] == ["left-view", "right-view"]
        assert first_checked["camera_dimensions"] == {"width": 32, "height": 24}
        assert second_checked["camera_dimensions"] == {"width": 19, "height": 13}
        assert first["source_video"]["sha256"] != second["source_video"]["sha256"]
        assert first["cameras"]["train"]["sha256"] != second["cameras"]["train"]["sha256"]


def test_source_mutation_is_a_hard_stop(tmp_path: Path) -> None:
    _manifest, path, _route = _fixture(tmp_path, "mutation", camera_names=["one", "two"], width=20, height=12)
    source = Path(_manifest["source_video"]["path"])
    source.write_bytes(b"mutated")
    with pytest.raises(AuthorityManifestError, match="source_video identity drifted"):
        load_authority_manifest(path)


def test_camera_identity_and_order_drift_are_rejected(tmp_path: Path) -> None:
    manifest, _path, _route = _fixture(tmp_path, "camera-drift", camera_names=["one", "two"], width=20, height=12)
    manifest["cameras"]["order"] = ["two", "one"]
    with pytest.raises(AuthorityManifestError, match="cameras.order"):
        validate_authority_manifest(manifest)
    manifest, _path, _route = _fixture(tmp_path, "native-drift", camera_names=["one", "two"], width=20, height=12)
    manifest["native_render_evidence"]["camera_order"] = ["two", "one"]
    with pytest.raises(AuthorityManifestError, match="native render camera order"):
        validate_authority_manifest(manifest)


def test_symlink_and_manifest_path_escape_are_rejected(tmp_path: Path) -> None:
    manifest, path, route = _fixture(tmp_path, "symlink", camera_names=["one"], width=20, height=12)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    link = tmp_path / "source-link.mp4"
    link.symlink_to(outside)
    manifest["source_video"]["path"] = str(link)
    with pytest.raises(AuthorityManifestError, match="source_video must not be a symlink"):
        validate_authority_manifest(manifest)
    escaped = tmp_path / "escaped-authority.json"
    write_manifest_once(escaped, manifest if False else _fixture(tmp_path, "escape-second", camera_names=["x"], width=10, height=8)[0])
    with pytest.raises(AuthorityManifestError, match="outside route root"):
        load_authority_manifest(escaped, route_root=route)


def test_symlinked_parent_escape_is_rejected(tmp_path: Path) -> None:
    manifest, _path, route = _fixture(tmp_path, "parent-symlink", camera_names=["one"], width=20, height=12)
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    link = route / "outputs" / "escaped-parent"
    link.symlink_to(outside, target_is_directory=True)
    escaped_input = link / "training_input"
    escaped_input.mkdir()
    manifest["training_input"] = {"path": str(escaped_input)}
    with pytest.raises(AuthorityManifestError, match="training_input traverses a symlink"):
        validate_authority_manifest(manifest)


def test_arbitrary_profile_and_parameter_overrides_are_rejected(tmp_path: Path) -> None:
    manifest, _path, _route = _fixture(tmp_path, "profile", camera_names=["one"], width=20, height=12)
    manifest["conversion_profile_id"] = "arbitrary-search-v1"
    with pytest.raises(AuthorityManifestError, match="unknown conversion profile"):
        validate_authority_manifest(manifest)
    manifest, _path, _route = _fixture(tmp_path, "profile-parameter", camera_names=["one"], width=20, height=12)
    manifest["conversion_profile"]["seed"] = 7
    with pytest.raises(AuthorityManifestError, match="parameter seed"):
        validate_authority_manifest(manifest)
    with pytest.raises(AuthorityManifestError, match="unknown conversion profile"):
        validate_profile("not-whitelisted")
