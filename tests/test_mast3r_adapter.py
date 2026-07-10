"""Tests for _mast3r_adapter.py: Sim(3) estimation and utilities."""

from __future__ import annotations

import numpy as np
import pytest
from _mast3r_adapter import (
    InsufficientMatchesError,
    Mast3rMatcher,
    _is_collinear,
    _resolve_checkpoint,
    _umeyama,
    estimate_sim3_ransac,
)

# ---------------------------------------------------------------------------
# _is_collinear
# ---------------------------------------------------------------------------


def test_is_collinear_two_points():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    assert _is_collinear(pts) is True


def test_is_collinear_three_collinear():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    assert _is_collinear(pts) is True


def test_is_collinear_three_triangle():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    assert _is_collinear(pts) is False


def test_is_collinear_near_collinear():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1e-12, 0.0], [2.0, 2e-12, 0.0]], dtype=np.float64)
    assert _is_collinear(pts) is True


# ---------------------------------------------------------------------------
# _umeyama — known transform recovery
# ---------------------------------------------------------------------------


def test_umeyama_identity():
    pts = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float64)
    scale, r_mat, t = _umeyama(pts, pts)
    assert scale == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(r_mat, np.eye(3), atol=1e-5)
    np.testing.assert_allclose(t, np.zeros(3), atol=1e-5)


def test_umeyama_known_scale():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    dst = src * 2.0
    scale, _, _ = _umeyama(src, dst)
    assert scale == pytest.approx(2.0, rel=1e-4)


def test_umeyama_known_translation():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    dst = src + np.array([5.0, -3.0, 1.0])
    _, _, t = _umeyama(src, dst)
    np.testing.assert_allclose(t, [5.0, -3.0, 1.0], atol=1e-4)


# ---------------------------------------------------------------------------
# estimate_sim3_ransac
# ---------------------------------------------------------------------------


def make_transformed(src, scale, r_mat, t, noise_std=0.0):
    rng = np.random.default_rng(123)
    dst = scale * (r_mat @ src.T).T + t
    if noise_std > 0:
        dst += rng.normal(0, noise_std, dst.shape).astype(dst.dtype)
    return dst.astype(np.float64)


def test_ransac_identity_no_noise():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(100, 3)).astype(np.float64)
    dst = src.copy()
    scale, r_mat, t, n_inliers = estimate_sim3_ransac(
        src,
        dst,
        thresh=0.01,
        max_iters=500,
        min_inliers=50,
    )
    assert scale == pytest.approx(1.0, abs=0.01)
    np.testing.assert_allclose(r_mat, np.eye(3), atol=0.01)
    np.testing.assert_allclose(t, np.zeros(3), atol=0.01)
    assert n_inliers >= 50


def test_ransac_known_sim3():
    rng = np.random.default_rng(2)
    src = rng.normal(size=(200, 3)).astype(np.float64)
    true_scale = 2.5
    true_rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    true_t = np.array([10.0, -5.0, 2.0], dtype=np.float64)
    dst = make_transformed(src, true_scale, true_rot, true_t, noise_std=0.001)
    scale, r_mat, t, n_inliers = estimate_sim3_ransac(
        src,
        dst,
        thresh=0.05,
        max_iters=2000,
        min_inliers=100,
    )
    assert scale == pytest.approx(true_scale, rel=0.05)
    assert n_inliers >= 100
    # The transform should approximately recover the input
    recovered = scale * (r_mat @ src.T).T + t
    np.testing.assert_allclose(recovered, dst, atol=0.1)


def test_ransac_fails_with_few_points():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    dst = src.copy()
    with pytest.raises(InsufficientMatchesError):
        estimate_sim3_ransac(src, dst, min_inliers=5)


def test_ransac_fails_insufficient_inliers():
    rng = np.random.default_rng(9)
    src = rng.normal(size=(20, 3)).astype(np.float64)
    # Random noise → no meaningful correspondence
    dst = rng.normal(size=(20, 3)).astype(np.float64)
    with pytest.raises(InsufficientMatchesError):
        estimate_sim3_ransac(src, dst, thresh=0.001, max_iters=200, min_inliers=15)


# ---------------------------------------------------------------------------
# Mast3rMatcher
# ---------------------------------------------------------------------------


def test_mast3r_matcher_version_property():
    m = Mast3rMatcher()
    assert "MASt3R" in m.version
    assert m.checkpoint in m.version


def test_resolve_checkpoint_finds_bundled_model():
    """The bundled MASt3R checkpoint under LongSplat should resolve."""
    path = _resolve_checkpoint("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    from pathlib import Path

    assert Path(path).is_file()
    assert path.endswith("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")


def test_mast3r_matcher_default_checkpoint_exists():
    """Default checkpoint should be a real file on disk."""
    m = Mast3rMatcher()
    from pathlib import Path

    assert Path(m.checkpoint).is_file()


def test_mast3r_matcher_match_pair_with_fake_paths():
    """Non-existent image paths raise FileNotFoundError before model load."""
    m = Mast3rMatcher()
    with pytest.raises(FileNotFoundError):
        m.match_pair(["nonexistent_a.jpg"], ["nonexistent_b.jpg"])
