from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_GEOMETRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "LongSplat"
    / "utils"
    / "mast3r_geometry.py"
)
_SPEC = importlib.util.spec_from_file_location("mast3r_geometry_contract", _GEOMETRY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_legacy_and_capped_shapes_are_patch_aligned():
    assert _MODULE.patch_aligned_shape(288, 512, max_side=512) == (288, 512)
    assert _MODULE.patch_aligned_shape(479, 511, max_side=512) == (480, 512)
    assert _MODULE.patch_aligned_shape(720, 1280, max_side=512) == (288, 512)
    assert _MODULE.patch_aligned_shape(720, 1280, max_side=None) == (720, 1280)


def test_inputs_already_within_512_keep_legacy_shape():
    source = (271, 479)
    assert _MODULE.patch_aligned_shape(source[0], source[1], max_side=512) == _MODULE.patch_aligned_shape(
        source[0], source[1], max_side=None
    )


def test_non_integer_resize_preserves_aspect_with_explicit_scales():
    source = (901, 1500)
    target = _MODULE.patch_aligned_shape(*source, max_side=512)
    assert target[0] % 16 == 0 and target[1] % 16 == 0
    source_ratio = source[1] / source[0]
    target_ratio = target[1] / target[0]
    assert abs(target_ratio / source_ratio - 1.0) < 0.02


def test_intrinsic_and_keypoint_mapping_use_independent_xy_scales():
    source = (901, 1500)
    target = (304, 512)
    intrinsic = np.array(
        [[900.0, 4.0, 749.5], [0.0, 910.0, 450.25], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    original = intrinsic.copy()
    scaled = _MODULE.scale_intrinsic(intrinsic, source, target)
    sx = target[1] / source[1]
    sy = target[0] / source[0]
    np.testing.assert_allclose(scaled[0], original[0] * sx)
    np.testing.assert_allclose(scaled[1], original[1] * sy)
    np.testing.assert_array_equal(intrinsic, original)

    internal_points = np.array([[0.0, 0.0], [256.0, 152.0], [511.0, 303.0]])
    normalized = _MODULE.keypoints_to_normalized(
        internal_points, target, source
    )
    expected = internal_points / np.array([target[1], target[0]])
    expected = expected * 2.0 - 1.0
    np.testing.assert_allclose(normalized, expected, rtol=0.0, atol=1e-6)


def test_camera_center_uses_c2w_rotation_and_w2c_translation():
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translation = np.array([2.0, 3.0, 4.0])
    expected = -(rotation @ translation)
    np.testing.assert_allclose(
        _MODULE.camera_center_from_c2w(rotation, translation), expected
    )


def test_grid_sample_mapping_adds_pixel_center_offset():
    keypoints = np.array([[0.0, 0.0], [511.0 / 512.0 * 2 - 1, 287.0 / 288.0 * 2 - 1]])
    mapped = _MODULE.normalized_to_grid_sample(keypoints, (288, 512))
    np.testing.assert_allclose(mapped[0], [1.0 / 512, 1.0 / 288])
    np.testing.assert_allclose(
        mapped[1], [1.0 - 1.0 / 512, 1.0 - 1.0 / 288]
    )
