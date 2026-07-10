"""Tests for _vda_adapter.py: VDA model spec and preprocessing."""

from __future__ import annotations

import pytest
from _vda_adapter import VDA_MODELS, VDAInference

# ---------------------------------------------------------------------------
# VDA_MODELS catalog
# ---------------------------------------------------------------------------


def test_vda_models_has_all_variants():
    expected = {
        "vda-small-relative",
        "vda-base-relative",
        "vda-large-relative",
        "vda-small-metric",
        "vda-base-metric",
        "vda-large-metric",
    }
    assert set(VDA_MODELS.keys()) == expected


def test_vda_models_depth_type_consistency():
    for name, spec in VDA_MODELS.items():
        if "relative" in name:
            assert spec.depth_type == "relative", f"{name} should be relative"
        elif "metric" in name:
            assert spec.depth_type == "metric", f"{name} should be metric"


def test_vda_models_have_checkpoints():
    for name, spec in VDA_MODELS.items():
        assert spec.checkpoint, f"{name} has empty checkpoint"
        # Checkpoint is a filename stem (e.g. "video_depth_anything_vits")
        assert spec.checkpoint.endswith("vits") or spec.checkpoint.endswith("vitb") or spec.checkpoint.endswith("vitl")


# ---------------------------------------------------------------------------
# VDAInference constructor + metadata
# ---------------------------------------------------------------------------


def test_vda_constructor_defaults():
    # Use a path that is_file() == True to bypass _resolve_vda_checkpoint
    import sys
    vda = VDAInference(
        checkpoint=sys.executable,  # a real file, passed through as-is
        depth_type="relative",
    )
    assert "python" in vda.checkpoint.lower()
    assert vda.depth_type == "relative"
    assert vda.device == "cuda:0"
    assert vda.input_size == 518
    assert vda.fp16 is True
    assert vda.max_frames_per_batch is None
    assert vda._model is None


def test_vda_constructor_custom():
    import sys

    vda = VDAInference(
        checkpoint=sys.executable,
        depth_type="metric",
        device="cpu",
        input_size=384,
        fp16=False,
        max_frames_per_batch=32,
    )
    assert vda.device == "cpu"
    assert vda.input_size == 384
    assert vda.fp16 is False
    assert vda.max_frames_per_batch == 32


def test_vda_version_string():
    import sys

    vda = VDAInference(checkpoint=sys.executable, depth_type="relative")
    ver = vda.version
    assert "VDA" in ver


# ---------------------------------------------------------------------------
# VDAInference.preprocess
# ---------------------------------------------------------------------------


def test_preprocess_no_resize_needed():
    """Small images (smaller than max_resolution) pass through unchanged."""
    from PIL import Image

    vda = VDAInference(checkpoint="test", depth_type="relative")
    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    result = vda.preprocess([img], max_resolution=1024)
    assert len(result) == 1
    assert result[0].size == (256, 256)


def test_preprocess_resize_short_side():
    from PIL import Image

    vda = VDAInference(checkpoint="test", depth_type="relative")
    img = Image.new("RGB", (1920, 1080), color=(100, 150, 200))
    result = vda.preprocess([img], max_resolution=1024)
    # Short side = 1080, ratio = 1024/1080
    # New size: (1920 * ratio, 1024) = (1820, 1024)
    w, h = result[0].size
    assert h == 1024
    assert w == pytest.approx(1920 * 1024 / 1080, abs=2)


def test_preprocess_multiple_images():
    from PIL import Image

    vda = VDAInference(checkpoint="test", depth_type="relative")
    images = [Image.new("RGB", (100, 200), color=(i, i, i)) for i in range(5)]
    result = vda.preprocess(images, max_resolution=1024)
    assert len(result) == 5
    for img in result:
        assert img.size == (100, 200)  # no resize needed


# ---------------------------------------------------------------------------
# VDAInference.infer_video_depth stub
# ---------------------------------------------------------------------------


def test_infer_video_depth_is_stub():
    """infer_video_depth now requires a real checkpoint file."""
    from PIL import Image

    vda = VDAInference(checkpoint="nonexistent_checkpoint.pth", depth_type="relative")
    img = Image.new("RGB", (64, 64))
    with pytest.raises(FileNotFoundError):
        vda.infer_video_depth([img])


# ---------------------------------------------------------------------------
# build_metadata
# ---------------------------------------------------------------------------


def test_build_metadata_structure():
    import sys

    vda = VDAInference(
        checkpoint=sys.executable,
        depth_type="relative",
    )
    meta = vda.build_metadata(
        frame_paths=["/tmp/a.jpg", "/tmp/b.jpg"],
        output_dir="/tmp/depth",
        fps=30.0,
        max_resolution=1024,
    )
    assert meta["vda_version"] == vda.version
    assert meta["checkpoint"] == vda.checkpoint
    assert meta["depth_type"] == "relative"
    assert meta["num_frames"] == 2
    assert meta["fps"] == 30.0
    assert meta["max_resolution"] == 1024
    assert len(meta["frame_paths"]) == 2
