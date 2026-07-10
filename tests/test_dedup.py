"""Tests for _dedup.py spatial deduplication."""

from __future__ import annotations

import numpy as np
from _dedup import deduplicate_spatial


def make_cloud(n, seed=42):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3)).astype(np.float32)
    features = {
        "opacity": rng.uniform(0.01, 0.99, size=n).astype(np.float32),
    }
    return xyz, features


# ---------------------------------------------------------------------------
# Single-segment (no-op)
# ---------------------------------------------------------------------------


def test_single_segment_returns_unchanged():
    xyz, features = make_cloud(50)
    merged_xyz, merged_feat = deduplicate_spatial([xyz], [features], radius=0.01)
    assert merged_xyz.shape == xyz.shape
    np.testing.assert_array_equal(merged_xyz, xyz)
    np.testing.assert_array_equal(merged_feat["opacity"], features["opacity"])


# ---------------------------------------------------------------------------
# Two identical segments: one should mostly survive
# ---------------------------------------------------------------------------


def test_two_identical_segments_dedup():
    xyz, features = make_cloud(100, seed=99)
    # Identical copy
    merged_xyz, merged_feat = deduplicate_spatial(
        [xyz, xyz.copy()], [features, features], radius=0.1
    )
    # After dedup there should be fewer than 200 points
    assert len(merged_xyz) < 200
    assert len(merged_xyz) > 50


# ---------------------------------------------------------------------------
# No overlap (far apart segments)
# ---------------------------------------------------------------------------


def test_far_apart_segments_no_dedup():
    xyz_a = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    xyz_b = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    feat = {"opacity": np.array([0.5], dtype=np.float32)}
    merged_xyz, _ = deduplicate_spatial([xyz_a, xyz_b], [feat, feat], radius=0.01)
    assert len(merged_xyz) == 2


# ---------------------------------------------------------------------------
# Opacity-based retention
# ---------------------------------------------------------------------------


def test_opacity_based_retention():
    # Two overlapping points, segment 1 has higher opacity
    xyz_a = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    xyz_b = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    feat_a = {"opacity": np.array([0.9], dtype=np.float32)}
    feat_b = {"opacity": np.array([0.2], dtype=np.float32)}
    merged_xyz, merged_feat = deduplicate_spatial([xyz_a, xyz_b], [feat_a, feat_b], radius=0.5)
    assert len(merged_xyz) == 1
    assert merged_feat["opacity"][0] == 0.9


def test_opacity_based_first_lower():
    xyz_a = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    xyz_b = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    feat_a = {"opacity": np.array([0.2], dtype=np.float32)}
    feat_b = {"opacity": np.array([0.9], dtype=np.float32)}
    merged_xyz, merged_feat = deduplicate_spatial([xyz_a, xyz_b], [feat_a, feat_b], radius=0.5)
    assert len(merged_xyz) == 1
    # Second segment has higher opacity, should be kept
    assert merged_feat["opacity"][0] == 0.9


# ---------------------------------------------------------------------------
# No opacity attribute: first occurrence kept
# ---------------------------------------------------------------------------


def test_no_opacity_attr_first_wins():
    xyz_a = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    xyz_b = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    feat_a = {"dummy": np.array([1.0], dtype=np.float32)}
    feat_b = {"dummy": np.array([2.0], dtype=np.float32)}
    merged_xyz, merged_feat = deduplicate_spatial(
        [xyz_a, xyz_b],
        [feat_a, feat_b],
        radius=0.5,
        opacity_attr=None,
    )
    assert len(merged_xyz) == 1
    # First occurrence kept (original feat_a value)
    assert merged_feat["dummy"][0] == 1.0


# ---------------------------------------------------------------------------
# Chunked processing doesn't change results
# ---------------------------------------------------------------------------


def test_chunked_processing_result():
    xyz, features = make_cloud(500, seed=7)
    merged_full, feat_full = deduplicate_spatial(
        [xyz],
        [features],
        radius=0.01,
        chunk_size=100_000,
    )
    merged_chunked, feat_chunked = deduplicate_spatial(
        [xyz],
        [features],
        radius=0.01,
        chunk_size=50,
    )
    np.testing.assert_array_equal(merged_full, merged_chunked)
