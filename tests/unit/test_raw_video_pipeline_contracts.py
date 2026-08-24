from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.longsplat.camera_staging import (
    CameraParameters,
    CameraStagingBlocked,
    plan_centered_integer_crop,
    stage_centered_pinhole,
    transform_intrinsics,
    validate_centered_pinhole,
)
from scripts.longsplat.colmap_contract import (
    CameraPriorError,
    _count_colmap_images,
    _parse_colmap_poses,
    build_colmap_commands,
    collect_intrinsics_evidence,
    enumerate_model_components,
    evaluate_sparse_geometry,
    load_camera_prior,
    require_single_model_component,
)
from scripts.longsplat.frame_contract import FrameMetric, FrameSelectionConfig, select_frames
from scripts.longsplat.longsplat_input import (
    ColmapModel,
    LongSplatInputBlocked,
    build_future_smoke_plan,
    future_workload_profile,
    compare_colmap_models,
    load_parent_camera_staging_evidence,
    materialize_training_images,
    parse_colmap_images_text,
    parse_colmap_model_text,
    prepare_converter_output_dirs,
    rewrite_colmap_model_for_crop,
    validate_future_smoke_plan,
    validate_colmap_model,
    write_colmap_model_text,
)
from scripts.longsplat.pipeline_contract import (
    PipelineBlocked,
    ResumeMismatchError,
    RunLedger,
    stable_sha256,
    build_run_identity,
    preflight_dependencies,
    code_identity,
    sha256_file,
)
from scripts.longsplat.raw_pipeline import run_raw_video_pipeline
from scripts.longsplat.video_contract import (
    UnsupportedDisplayMetadata,
    canonical_media_record,
    canonical_filter_graph,
    parse_ffprobe_payload,
    build_frame_extract_command,
    candidate_sampling_plan,
)


def _sha(letter: str) -> str:
    return letter * 64


def _probe(rotation: int = 0, *, sar: str = "1:1", width: int = 640, height: int = 480) -> dict:
    stream = {
        "codec_type": "video",
        "width": width,
        "height": height,
        "coded_width": width,
        "coded_height": height,
        "avg_frame_rate": "30/1",
        "r_frame_rate": "30/1",
        "duration": "2.0",
        "nb_frames": "60",
        "sample_aspect_ratio": sar,
        "display_aspect_ratio": "4:3",
        "codec_name": "synthetic",
        "pix_fmt": "yuv420p",
        "tags": {"rotate": str(rotation)} if rotation else {},
    }
    return {"streams": [stream], "format": {"duration": "2.0", "format_name": "synthetic"}}


def _fixture_version_probe_runner(argv, **kwargs):
    """Return a harmless version result without executing any external tool."""
    return subprocess.CompletedProcess(argv, 0, stdout="fixture-tool 1.0\n", stderr="")


def _formatted_display_matrix_payload(*, matrix: str, rotation: int) -> dict:
    payload = _probe(width=1920, height=1080, sar="1:1")
    payload["streams"][0]["side_data_list"] = [
        {
            "side_data_type": "Display Matrix",
            "displaymatrix": matrix,
            "rotation": rotation,
        }
    ]
    return payload


_B_DISPLAY_MATRIX = (
    "00000000:            0       65536           0\n"
    "00000001:       -65536           0           0\n"
    "00000002:            0           0  1073741824\n"
)


def test_video_probe_normalizes_b_ffprobe_display_matrix_and_extracts_once_without_autorotate():
    probe = parse_ffprobe_payload(
        _formatted_display_matrix_payload(matrix=_B_DISPLAY_MATRIX, rotation=-90),
        source_path="input-b.mp4",
        source_sha256=_sha("b"),
        source_size_bytes=123,
    )
    assert probe["display"]["rotation_degrees"] == 90
    assert probe["display"]["rotation_semantics"] == "applied_to_noautorotate_pixels"
    assert probe["display"]["orientation_source"] == "display_matrix"
    assert probe["display"]["raw_rotation_degrees"] == -90.0
    assert (probe["stream"]["visible_width"], probe["stream"]["visible_height"]) == (1080, 1920)
    assert (probe["canonical"]["width"], probe["canonical"]["height"]) == (1080, 1920)
    assert canonical_filter_graph(probe) == "transpose=1,setsar=1"
    command = build_frame_extract_command(
        "ffmpeg",
        "input-b.mp4",
        "/tmp/frame_%06d.png",
        probe,
        candidate_fps=2.0,
        candidate_cap=12,
    )
    assert "-noautorotate" in command
    filter_graph = command[command.index("-vf") + 1]
    assert filter_graph.count("transpose=1") == 1
    assert filter_graph == "transpose=1,setsar=1,fps=2"


def test_video_probe_accepts_opposite_formatted_positive_90_matrix():
    matrix = (
        "00000000:            0      -65536           0\n"
        "00000001:        65536           0           0\n"
        "00000002:            0           0  1073741824\n"
    )
    probe = parse_ffprobe_payload(
        _formatted_display_matrix_payload(matrix=matrix, rotation=90),
        source_path="input.mp4",
        source_sha256=_sha("a"),
        source_size_bytes=123,
    )
    assert probe["display"]["rotation_degrees"] == 270
    assert probe["display"]["rotation_semantics"] == "applied_to_noautorotate_pixels"
    assert probe["display"]["raw_rotation_degrees"] == 90.0
    assert canonical_filter_graph(probe) == "transpose=2,setsar=1"


@pytest.mark.parametrize(
    "matrix",
    [
        "00000000: 0 65536 0\n00000001: -65536 0 0\n",
        "00000000: 0 65536 0\n00000001: -65536 0 0\n00000002: 0 0 1073741824 1\n",
        "00000000: 0 65536 nope\n00000001: -65536 0 0\n00000002: 0 0 1073741824\n",
        "00000000: 65536 13107 0\n00000001: 0 65536 0\n00000002: 0 0 1073741824\n",
    ],
)
def test_video_probe_rejects_malformed_or_sheared_display_matrix(matrix):
    with pytest.raises(UnsupportedDisplayMetadata):
        parse_ffprobe_payload(
            _formatted_display_matrix_payload(matrix=matrix, rotation=-90),
            source_path="input.mp4",
            source_sha256=_sha("a"),
            source_size_bytes=123,
        )


def test_video_probe_rejects_matrix_tag_conflict_for_formatted_matrix():
    with pytest.raises(UnsupportedDisplayMetadata, match="disagree"):
        parse_ffprobe_payload(
            _formatted_display_matrix_payload(matrix=_B_DISPLAY_MATRIX, rotation=90),
            source_path="input.mp4",
            source_sha256=_sha("a"),
            source_size_bytes=123,
        )


@pytest.mark.parametrize(
    "matrix",
    [
        "1 0 0 0 1 0 0 0 1",
        [1, 0, 0, 0, 1, 0, 0, 0, 1],
        (1, 0, 0, 0, 1, 0, 0, 0, 1),
    ],
)
def test_video_probe_accepts_flat_identity_display_matrix_forms(matrix):
    probe = parse_ffprobe_payload(
        _formatted_display_matrix_payload(matrix=matrix, rotation=0),
        source_path="input.mp4",
        source_sha256=_sha("a"),
        source_size_bytes=123,
    )
    assert probe["display"]["rotation_degrees"] == 0


@pytest.mark.parametrize(
    ("metadata_rotation", "applied_rotation", "width", "height", "filter_graph"),
    [
        (0, 0, 640, 480, "scale=round(iw*1.33333333333):ih,setsar=1"),
        (90, 270, 480, 640, "scale=round(iw*1.33333333333):ih,transpose=2,setsar=1"),
        (180, 180, 640, 480, "scale=round(iw*1.33333333333):ih,hflip,vflip,setsar=1"),
        (270, 90, 480, 640, "scale=round(iw*1.33333333333):ih,transpose=1,setsar=1"),
    ],
)
def test_video_probe_records_applied_rotation_sar_and_dynamic_visible_dimensions(
    metadata_rotation, applied_rotation, width, height, filter_graph
):
    probe = parse_ffprobe_payload(
        _probe(metadata_rotation, sar="4:3"),
        source_path="/tmp/input.mp4",
        source_sha256=_sha("a"),
        source_size_bytes=123,
    )
    assert probe["display"]["rotation_degrees"] == applied_rotation
    assert probe["display"]["rotation_semantics"] == "applied_to_noautorotate_pixels"
    assert probe["stream"]["visible_width"] == width
    assert probe["stream"]["visible_height"] == height
    assert probe["display"]["sample_aspect_ratio"]["canonical"] == "1:1"
    assert canonical_filter_graph(probe) == filter_graph
    assert probe["timing"]["frame_count"] == 60


@pytest.mark.parametrize(
    ("matrix", "tag", "applied_rotation"),
    [
        (_B_DISPLAY_MATRIX, -90, 90),
        (
            "00000000: 0 -65536 0\n00000001: 65536 0 0\n00000002: 0 0 1073741824\n",
            90,
            270,
        ),
        (
            "00000000: 65536 0 0\n00000001: 0 65536 0\n00000002: 0 0 1073741824\n",
            0,
            0,
        ),
        (
            "00000000: -65536 0 0\n00000001: 0 -65536 0\n00000002: 0 0 1073741824\n",
            180,
            180,
        ),
    ],
)
def test_display_matrix_and_signed_tag_compare_in_applied_transform_semantics(
    matrix, tag, applied_rotation
):
    probe = parse_ffprobe_payload(
        _formatted_display_matrix_payload(matrix=matrix, rotation=tag),
        source_path="input.mp4",
        source_sha256=_sha("a"),
        source_size_bytes=123,
    )
    assert probe["display"]["rotation_degrees"] == applied_rotation
    assert probe["display"]["raw_rotation_degrees"] == float(tag)


def test_transpose_one_is_the_measured_clockwise_pixel_transform():
    probe = parse_ffprobe_payload(
        _formatted_display_matrix_payload(matrix=_B_DISPLAY_MATRIX, rotation=-90),
        source_path="input.mp4",
        source_sha256=_sha("b"),
        source_size_bytes=123,
    )
    assert canonical_filter_graph(probe) == "transpose=1,setsar=1"
    # Four corner labels make the pixel-direction convention explicit without
    # depending on a video filename or fixed production dimensions.
    pixels = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    clockwise = np.rot90(pixels, k=-1)
    assert clockwise.tolist() == [[4, 1], [5, 2], [6, 3]]


def test_video_probe_rejects_unsupported_display_matrix():
    payload = _probe()
    payload["streams"][0].pop("tags")
    payload["streams"][0]["side_data_list"] = [{
        "side_data_type": "Display Matrix",
        "displaymatrix": "1 0 0 0.1 1 0 0 0 1",
    }]
    with pytest.raises(UnsupportedDisplayMetadata):
        parse_ffprobe_payload(payload, source_path="input.mp4", source_sha256=_sha("a"), source_size_bytes=1)


def test_video_probe_rejects_malformed_sar():
    with pytest.raises(UnsupportedDisplayMetadata):
        parse_ffprobe_payload(_probe(sar="0:1"), source_path="input.mp4", source_sha256=_sha("a"), source_size_bytes=1)


def test_frame_budget_is_duration_derived_and_records_quality_reasons():
    metrics = [
        FrameMetric(
            frame_id=f"frame_{index:03d}",
            timestamp_sec=index * 0.5,
            frame_index=index,
            path=f"frame_{index:03d}.png",
            width=640,
            height=480,
            sharpness=100.0 + index,
            overexposed_ratio=0.0,
            underexposed_ratio=0.0,
            coverage_score=1.0,
            duplicate_score=10,
            provenance="synthetic",
        )
        for index in range(10)
    ]
    metrics[2] = FrameMetric(**{**metrics[2].__dict__, "duplicate_score": 0})
    result = select_frames(
        metrics,
        duration_sec=5.0,
        source_video_sha256=_sha("a"),
        canonical_media_sha256=_sha("b"),
        canonical_width=640,
        canonical_height=480,
        config=FrameSelectionConfig(target_fps=1.0, min_frames=3, max_frames=5),
    )
    assert result["frame_budget"] == 5
    assert result["selected_count"] == 5
    assert result["max_time_gap_sec"] >= result["max_internal_gap_sec"]
    rejected = next(frame for frame in result["frames"] if frame["frame_id"] == "frame_002")
    assert "duplicate" in rejected["rejected_reasons"]
    assert result["binding"]["canonical_media_sha256"] == _sha("b")


def test_colmap_without_prior_never_emits_camera_params_and_prior_is_source_bound():
    plan = build_colmap_commands(
        colmap="colmap",
        workspace="/tmp/run/colmap",
        image_dir="/tmp/run/colmap/images",
        source_video_sha256=_sha("a"),
        canonical_media_sha256=_sha("b"),
        canonical_width=640,
        canonical_height=480,
    )
    feature = plan["commands"][0]["argv"]
    assert "--ImageReader.camera_params" not in feature
    assert plan["canonical_media_sha256"] == _sha("b")
    assert plan["canonical_width"] == 640
    prior = load_camera_prior(
        {
            "source_video_sha256": _sha("a"),
            "model": "SIMPLE_RADIAL",
            "params": [400, 320, 240, 0.01],
            "calibration_width": 640,
            "calibration_height": 480,
            "coordinate_space": "canonical_pixels_v1",
        },
        source_video_sha256=_sha("a"),
    )
    prior_plan = build_colmap_commands(
        colmap="colmap",
        workspace="/tmp/run/colmap",
        image_dir="/tmp/run/colmap/images",
        source_video_sha256=_sha("a"),
        canonical_media_sha256=_sha("b"),
        canonical_width=640,
        canonical_height=480,
        camera_prior=prior,
    )
    assert "--ImageReader.camera_params" in prior_plan["commands"][0]["argv"]
    with pytest.raises(CameraPriorError):
        load_camera_prior(
            {
                "source_video_sha256": _sha("b"),
                "model": "SIMPLE_RADIAL",
                "params": [400, 320, 240, 0.01],
                "calibration_width": 640,
                "calibration_height": 480,
                "coordinate_space": "canonical_pixels_v1",
            },
            source_video_sha256=_sha("a"),
        )


@pytest.mark.parametrize(
    ("matching", "matcher_name"),
    [("sequential", "sequential_matcher"), ("exhaustive", "exhaustive_matcher")],
)
def test_colmap_cpu_sift_flags_are_exactly_recorded_for_each_matcher(matching, matcher_name):
    plan = build_colmap_commands(
        colmap="colmap",
        workspace="/tmp/run/colmap",
        image_dir="/tmp/run/colmap/images",
        source_video_sha256=_sha("a"),
        canonical_media_sha256=_sha("b"),
        canonical_width=640,
        canonical_height=480,
        matching=matching,
    )
    feature = plan["commands"][0]["argv"]
    matcher = plan["commands"][1]["argv"]
    assert feature.count("--SiftExtraction.use_gpu") == 1
    assert feature[feature.index("--SiftExtraction.use_gpu") + 1] == "0"
    assert matcher[1] == matcher_name
    assert matcher.count("--SiftMatching.use_gpu") == 1
    assert matcher[matcher.index("--SiftMatching.use_gpu") + 1] == "0"
    assert plan["compute_mode"] == "cpu"
    assert plan["compute_contract"] == {
        "sift_extraction_use_gpu": 0,
        "sift_matching_use_gpu": 0,
        "gpu_hardware_capability": "not_probed",
    }


def test_colmap_images_parser_counts_headers_not_long_points2d_lines(tmp_path):
    def header(index: int, name: str | None = None) -> str:
        return f"{index} 1 0 0 0 {index * 0.1} 0 0 1 {name or f'frame_{index:06d}.png'}"

    lines: list[str] = []
    for index in range(1, 46):
        lines.extend([header(index), " ".join(["0.25", "0.75", "-1"] * 40)])
    path = tmp_path / "images.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    count, names = _count_colmap_images(path)
    pose_names, centers, quaternions = _parse_colmap_poses(path)
    assert count == 45
    assert names == pose_names == [f"frame_{index:06d}.png" for index in range(1, 46)]
    assert len(centers) == len(quaternions) == 45
    assert quaternions[0] == (1.0, 0.0, 0.0, 0.0)


def test_colmap_images_parser_accepts_empty_points2d_lines(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text(
        "1 1 0 0 0 0 0 0 1 frame_000001.png\n\n"
        "2 1 0 0 0 1 0 0 1 frame_000002.png\n\n",
        encoding="utf-8",
    )
    assert _count_colmap_images(path)[0] == 2
    assert len(_parse_colmap_poses(path)[1]) == 2


@pytest.mark.parametrize(
    "bad_header",
    [
        "2 1 0 0 0 0 0 0 1",  # missing NAME
        "2 nan 0 0 0 0 0 0 1 frame_000002.png",  # non-finite quaternion
        "2 1 0 0 0 0 0 0 0 frame_000002.png",  # non-positive camera id
        "2 1 0 0 0 0 0 0 1 nested/frame_000002.png",  # not a basename
        "2 1 0 0 0 0 0 0 1 frame_000002.txt",  # not an image extension
    ],
)
def test_colmap_images_parser_rejects_malformed_headers(tmp_path, bad_header):
    path = tmp_path / "images.txt"
    path.write_text(
        "1 1 0 0 0 0 0 0 1 frame_000001.png\n\n"
        f"{bad_header}\n\n",
        encoding="utf-8",
    )
    with pytest.raises(PipelineBlocked):
        _count_colmap_images(path)


@pytest.mark.parametrize(
    "second_header",
    [
        "1 1 0 0 0 1 0 0 1 frame_000002.png",  # duplicate IMAGE_ID
        "2 1 0 0 0 1 0 0 1 frame_000001.png",  # duplicate NAME
    ],
)
def test_colmap_images_parser_rejects_duplicate_pose_identity(tmp_path, second_header):
    path = tmp_path / "images.txt"
    path.write_text(
        "1 1 0 0 0 0 0 0 1 frame_000001.png\n\n"
        f"{second_header}\n\n",
        encoding="utf-8",
    )
    with pytest.raises(PipelineBlocked, match="duplicate"):
        _parse_colmap_poses(path)


def test_centered_pinhole_math_and_strict_validator():
    camera = CameraParameters("PINHOLE", 640, 480, 400, 402, 320, 240)
    K, width, height, _ = transform_intrinsics(camera, orientation="rotate90_cw")
    assert (width, height) == (480, 640)
    np.testing.assert_allclose(K, np.asarray([[402, 0, 240], [0, 400, 320], [0, 0, 1]], dtype=float))
    contract = validate_centered_pinhole(
        camera,
        source_video_sha256=_sha("a"),
        canonical_media_sha256=_sha("b"),
        frame_names=["frame_000000.png", "frame_000001.png"],
        expected_width=640,
        expected_height=480,
    )
    assert contract["longsplat_compatible"] is True
    assert contract["accepted"] is False
    with pytest.raises(CameraStagingBlocked):
        validate_centered_pinhole(
            CameraParameters("PINHOLE", 640, 480, 400, 402, 322, 240),
            source_video_sha256=_sha("a"),
            canonical_media_sha256=_sha("b"),
            frame_names=["frame_000000.png"],
            expected_width=640,
            expected_height=480,
        )


def test_centered_integer_crop_identity_offsets_and_retention_gate():
    identity = plan_centered_integer_crop(
        CameraParameters("PINHOLE", 641, 481, 400, 402, 320.5, 240.5)
    )
    assert identity["strategy"] == "identity"
    assert identity["width"] == 641 and identity["height"] == 481
    assert identity["center_error_px"] == {"cx": 0.0, "cy": 0.0}

    left_top = plan_centered_integer_crop(
        CameraParameters("PINHOLE", 100, 80, 400, 402, 39.8, 30.2)
    )
    right_bottom = plan_centered_integer_crop(
        CameraParameters("PINHOLE", 100, 80, 400, 402, 60.2, 49.8)
    )
    assert left_top["x0"] == 0 and left_top["y0"] == 0
    assert right_bottom["x0"] > 0 and right_bottom["y0"] > 0
    assert left_top["center_error_px"]["cx"] <= 0.5
    assert right_bottom["center_error_px"]["cy"] <= 0.5
    assert left_top["retained_width_fraction"] >= 0.75
    with pytest.raises(CameraStagingBlocked, match="retained fraction"):
        plan_centered_integer_crop(
            CameraParameters("PINHOLE", 100, 80, 400, 402, 10.2, 40.0)
        )


def test_b_offcenter_crop_materializes_exact_roi_and_resumes(tmp_path):
    import cv2

    input_width, input_height = 1063, 1891
    yy, xx = np.indices((input_height, input_width), dtype=np.int32)
    image = np.empty((input_height, input_width, 3), dtype=np.uint8)
    image[..., 0] = (xx + 3 * yy) % 256
    image[..., 1] = (2 * xx + yy) % 256
    image[..., 2] = (xx + yy) % 256
    input_path = tmp_path / "undistorted" / "frame_000000.png"
    input_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(input_path), image)

    media_record = {
        "canonical": {"pixels_sha256": _sha("p")},
        "binding": {"canonical_media_sha256": _sha("b")},
    }
    media_record_path = tmp_path / "canonical_media-v1.json"
    media_record_path.write_text(json.dumps(media_record), encoding="utf-8")
    raw = CameraParameters("SIMPLE_RADIAL", 1080, 1920, 1800, 1800, 540, 960, (0.01,))
    undistorted = CameraParameters(
        "PINHOLE", input_width, input_height, 1806.1725, 1806.1725, 522.2024973583, 874.4475810315
    )
    frame_record = {
        "staged_name": input_path.name,
        "path": str(input_path),
        "sha256": sha256_file(input_path),
        "width": input_width,
        "height": input_height,
        "provenance": "undistorted_output_frame",
    }

    result = stage_centered_pinhole(
        raw_camera=raw,
        undistorted_camera=undistorted,
        source_video_sha256=_sha("a"),
        canonical_media_sha256=_sha("b"),
        frame_records=[frame_record],
        output_dir=tmp_path / "camera_staging",
        canonical_media_record=media_record,
        canonical_media_record_path=media_record_path,
        undistorter_argv=["colmap", "image_undistorter", "--output_path", "undistorted"],
    )
    crop = result["staging"]["crop_roi"]
    assert (crop["x0"], crop["y0"], crop["width"], crop["height"]) == (0, 0, 1045, 1749)
    assert result["staging"]["strategy"] == "maximum-area-integer-crop-centered-v1"
    assert result["staging"]["canonical_camera"]["width"] == 1045
    assert result["staging"]["canonical_camera"]["height"] == 1749
    assert abs(result["staging"]["canonical_camera"]["cx"] - 1045 / 2) <= 0.5
    assert abs(result["staging"]["canonical_camera"]["cy"] - 1749 / 2) <= 0.5
    assert result["canonical_media_binding_sha256"] == _sha("b")
    assert result["canonical_media_pixels_sha256"] == _sha("p")
    assert result["canonical_media_record_sha256"] == sha256_file(media_record_path)
    assert result["accepted"] is False
    assert result["staging"]["pixel_transform"]["interpolation"] == "none"
    assert result["staging"]["pixel_transform"]["padding"] is False

    output_path = Path(result["frames"][0]["path"])
    output = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)
    assert output is not None
    np.testing.assert_array_equal(output, image[:1749, :1045])
    assert result["frames"][0]["input_decoded_pixel_sha256"] != result["frames"][0]["decoded_pixel_sha256"]
    assert result["frames"][0]["roi"] == {"x0": 0, "y0": 0, "width": 1045, "height": 1749}
    assert len(result["pixel_provenance"]["undistorted_output_frames"]) == 1
    output_bytes_before_resume = output_path.read_bytes()

    resumed = stage_centered_pinhole(
        raw_camera=raw,
        undistorted_camera=undistorted,
        source_video_sha256=_sha("a"),
        canonical_media_sha256=_sha("b"),
        frame_records=[frame_record],
        output_dir=tmp_path / "camera_staging",
        canonical_media_record=media_record,
        canonical_media_record_path=media_record_path,
        undistorter_argv=["colmap", "image_undistorter", "--output_path", "undistorted"],
    )
    assert resumed["contract_sha256"] == result["contract_sha256"]
    assert output_path.read_bytes() == output_bytes_before_resume


def test_staging_records_raw_undistorted_canonical_K_and_bindings(tmp_path):
    raw = CameraParameters("SIMPLE_RADIAL", 640, 480, 400, 400, 320, 240, (0.01,))
    undistorted = CameraParameters("PINHOLE", 640, 480, 400, 402, 320, 240)
    import cv2
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), np.zeros((480, 640, 3), dtype=np.uint8))
    result = stage_centered_pinhole(
        raw_camera=raw,
        undistorted_camera=undistorted,
        source_video_sha256=_sha("a"),
        canonical_media_sha256=_sha("b"),
        frame_records=[
            {"staged_name": "frame_000000.png", "path": str(frame), "sha256": sha256_file(frame), "width": 640, "height": 480}
        ],
        undistorter_argv=["colmap", "image_undistorter"],
    )
    assert result["staging"]["K_raw"][0][2] == 320
    assert result["staging"]["K_undistorted"][1][1] == 402
    assert result["source_video_sha256"] == _sha("a")
    assert result["canonical_media_sha256"] == _sha("b")
    assert result["frame_count"] == 1


def test_staging_rejects_unapplied_pixel_transform_even_with_string_sha():
    camera = CameraParameters("PINHOLE", 640, 480, 400, 402, 320, 240)
    with pytest.raises(CameraStagingBlocked, match="pixel_transform"):
        stage_centered_pinhole(
            raw_camera=camera,
            undistorted_camera=camera,
            source_video_sha256=_sha("a"),
            canonical_media_sha256=_sha("b"),
            frame_records=[],
            pixel_transform={"orientation": "identity"},
            warp_artifact_sha256=_sha("c"),
        )


def test_camera_prior_requires_canonical_dimensions_and_coordinate_space():
    payload = {
        "source_video_sha256": _sha("a"),
        "model": "PINHOLE",
        "params": [400, 402, 320, 240],
        "calibration_width": 640,
        "calibration_height": 480,
        "coordinate_space": "raw_pixels",
    }
    with pytest.raises(CameraPriorError, match="canonical_pixels_v1"):
        load_camera_prior(payload, source_video_sha256=_sha("a"))
    payload["coordinate_space"] = "canonical_pixels_v1"
    prior = load_camera_prior(payload, source_video_sha256=_sha("a"))
    with pytest.raises(CameraPriorError, match="dimensions"):
        load_camera_prior(payload, source_video_sha256=_sha("a"), canonical_width=800, canonical_height=600)
    assert prior.calibration_width == 640


def test_candidate_sampling_is_bounded_and_vfr_is_hard_stop():
    probe = parse_ffprobe_payload(_probe(), source_path="input.mp4", source_sha256=_sha("a"), source_size_bytes=1)
    sampling = candidate_sampling_plan(probe, target_fps=2.0, min_frames=4, max_frames=11)
    command = build_frame_extract_command("ffmpeg", "input.mp4", "/tmp/frame_%06d.png", probe, candidate_fps=sampling["candidate_fps"], candidate_cap=sampling["candidate_cap"])
    assert sampling["candidate_budget"] <= 11
    assert command[command.index("-frames:v") + 1] == "11"
    vfr = _probe()
    vfr["streams"][0]["avg_frame_rate"] = "29/1"
    vfr_probe = parse_ffprobe_payload(vfr, source_path="input.mp4", source_sha256=_sha("a"), source_size_bytes=1)
    with pytest.raises(UnsupportedDisplayMetadata, match="VFR"):
        candidate_sampling_plan(vfr_probe, target_fps=2.0, min_frames=4, max_frames=11)


def test_tool_binary_bytes_change_identity(tmp_path):
    tool = tmp_path / "tool"
    tool.write_bytes(b"one")
    tool.chmod(0o755)
    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="tool 1", stderr="")
    first = preflight_dependencies(
        route_python=str(tool), backend_python=str(tool), ffmpeg=str(tool), ffprobe=str(tool), colmap=str(tool), runner=runner
    )
    tool.write_bytes(b"two")
    second = preflight_dependencies(
        route_python=str(tool), backend_python=str(tool), ffmpeg=str(tool), ffprobe=str(tool), colmap=str(tool), runner=runner
    )
    assert first["tool_identity_sha256"] != second["tool_identity_sha256"]


def test_preflight_resolves_bare_tool_name_through_path_and_records_identity(tmp_path, monkeypatch):
    tool = tmp_path / "colmap-from-path"
    tool.write_bytes(b"bare-colmap")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="tool 1", stderr="")

    result = preflight_dependencies(
        route_python=sys.executable,
        backend_python=sys.executable,
        ffmpeg=sys.executable,
        ffprobe=sys.executable,
        colmap=tool.name,
        runner=runner,
    )
    entry = result["dependencies"]["colmap"]
    assert result["status"] == "passed"
    assert entry["resolved_path"] == str(tool.resolve())
    assert entry["executable_identity"]["resolved_target"] == str(tool.resolve())
    assert entry["executable_identity"]["sha256"] == hashlib.sha256(b"bare-colmap").hexdigest()


def test_preflight_keeps_explicit_existing_tool_path(tmp_path):
    tool = tmp_path / "colmap"
    tool.write_bytes(b"explicit-colmap")
    tool.chmod(0o755)

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="tool 1", stderr="")

    result = preflight_dependencies(
        route_python=sys.executable,
        backend_python=sys.executable,
        ffmpeg=sys.executable,
        ffprobe=sys.executable,
        colmap=tool,
        runner=runner,
    )
    entry = result["dependencies"]["colmap"]
    assert result["status"] == "passed"
    assert entry["resolved_path"] == str(tool.resolve())
    assert entry["sha256"] == hashlib.sha256(b"explicit-colmap").hexdigest()


def test_preflight_does_not_fallback_from_missing_explicit_path_to_path(tmp_path, monkeypatch):
    path_dir = tmp_path / "path"
    path_dir.mkdir()
    path_tool = path_dir / "colmap"
    path_tool.write_bytes(b"path-colmap")
    path_tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(path_dir))
    explicit = tmp_path / "explicit" / "colmap"

    def runner(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout="tool 1", stderr="")

    result = preflight_dependencies(
        route_python=sys.executable,
        backend_python=sys.executable,
        ffmpeg=sys.executable,
        ffprobe=sys.executable,
        colmap=explicit,
        runner=runner,
    )
    entry = result["dependencies"]["colmap"]
    assert result["status"] == "blocked"
    assert entry["resolved_path"] is None
    assert entry["reason"] == "missing_or_not_executable"
    assert {item["dependency"] for item in result["blocked"]} == {"colmap"}


def test_preflight_missing_bare_tool_is_structured_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))

    def runner(argv, **kwargs):
        raise AssertionError("missing dependencies must not be executed")

    result = preflight_dependencies(
        route_python="missing-route-python",
        backend_python="missing-backend-python",
        ffmpeg="missing-ffmpeg",
        ffprobe="missing-ffprobe",
        colmap="missing-colmap",
        runner=runner,
    )
    assert result["status"] == "blocked"
    assert len(result["blocked"]) == 5
    assert result["dependencies"]["colmap"]["resolved_path"] is None
    assert result["dependencies"]["colmap"]["reason"] == "missing_or_not_executable"


def test_code_identity_changes_for_untracked_route_and_nested_sources(tmp_path):
    route = tmp_path / "route"
    nested = route / "third_party" / "LongSplat"
    (route / "scripts").mkdir(parents=True)
    (route / "tests").mkdir()
    (route / "docs" / "longsplat").mkdir(parents=True)
    (nested / "scene").mkdir(parents=True)
    (route / "scripts" / "pipeline.py").write_text("x=1\n")
    (nested / "scene" / "new.py").write_text("x=1\n")
    first = code_identity(route, nested)
    (route / "scripts" / "pipeline.py").write_text("x=2\n")
    second = code_identity(route, nested)
    assert first["code_identity_sha256"] != second["code_identity_sha256"]
    (nested / "scene" / "new.py").write_text("x=3\n")
    third = code_identity(route, nested)
    assert second["code_identity_sha256"] != third["code_identity_sha256"]


def test_ledger_does_not_reuse_failed_attempt_and_logs_are_write_once(tmp_path):
    identity = build_run_identity(source_video_sha256=_sha("a"), canonical_config_sha256=_sha("b"), tool_identity_sha256=_sha("c"), code_identity_value={"fixture": True})
    ledger = RunLedger.create_or_resume(output_root=tmp_path, run_id="retry", identity=identity)
    attempt = ledger.begin_attempt("colmap", {"partial": True})
    ledger.finish_attempt(stage="colmap", attempt=attempt, status="failed", result={"command_results": [{"exit_code": 1}]}, stdout="first")
    assert ledger.latest_result("colmap") is None
    retry = ledger.begin_attempt("colmap", {"retry": True})
    assert retry.name == "attempt-0002"
    ledger.finish_attempt(stage="colmap", attempt=retry, status="passed", result={"artifacts": []}, stdout="second")
    assert ledger.latest_attempt("colmap")["status"] == "passed"
    assert (attempt / "stdout.log").read_text() == "first"


def test_blocked_stage_can_retry_when_bound_artifact_becomes_ready(tmp_path):
    identity = build_run_identity(source_video_sha256=_sha("a"), canonical_config_sha256=_sha("b"), tool_identity_sha256=_sha("c"), code_identity_value={"fixture": True})
    ledger = RunLedger.create_or_resume(output_root=tmp_path, run_id="blocked-retry", identity=identity)
    blocked = ledger.begin_attempt("camera-staging", {"needs": "undistorted-images"})
    ledger.finish_attempt(stage="camera-staging", attempt=blocked, status="blocked", result={"reason": "missing", "artifacts": []})
    latest = ledger.latest_attempt("camera-staging")
    assert latest["status"] == "blocked"
    assert ledger.latest_result("camera-staging") is None
    artifact = ledger.run_dir / "ready.txt"
    artifact.write_text("ready")
    retry = ledger.begin_attempt("camera-staging", {"needs": "undistorted-images"})
    ledger.finish_attempt(
        stage="camera-staging",
        attempt=retry,
        status="passed",
        result={"contract": {"computed_pass": True, "accepted": False}, "artifacts": [{"path": str(artifact), "sha256": sha256_file(artifact)}]},
    )
    assert ledger.latest_result("camera-staging")["contract"]["accepted"] is False


def test_model_components_and_geometry_gate_use_real_fixture(tmp_path):
    root = tmp_path / "mapper"
    component = root / "7"
    component.mkdir(parents=True)
    (component / "cameras.txt").write_text("1 PINHOLE 640 480 400 402 320 240\n")
    (component / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame_000000.png\n\n2 1 0 0 0 0 0 0 1 frame_000001.png\n\n")
    (component / "points3D.txt").write_text("1 0 0 1 255 255 255 0.5 1 0 2 1\n")
    assert [path.name for path in enumerate_model_components(root)] == ["7"]
    evidence = collect_intrinsics_evidence(component, source_video_sha256=_sha("a"), canonical_media_sha256=_sha("b"), expected_image_names=["frame_000000.png", "frame_000001.png"], refine_argv=["colmap", "bundle_adjuster", "--BundleAdjustment.refine_focal_length", "1"])
    geometry = evaluate_sparse_geometry(evidence, expected_image_names=["frame_000000.png", "frame_000001.png"])
    assert geometry["computed_pass"] is True
    assert evidence["component_count"] == 1
    (root / "8").mkdir()
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        (root / "8" / name).write_text((component / name).read_text())
    with pytest.raises(Exception, match="exactly one"):
        require_single_model_component(root)


def test_run_identity_resume_mismatch_and_cpu_stop_cannot_accept(tmp_path):
    identity = build_run_identity(
        source_video_sha256=_sha("a"),
        canonical_config_sha256=_sha("b"),
        tool_identity_sha256=_sha("c"),
        code_identity_value={"commit": "fixture"},
    )
    ledger = RunLedger.create_or_resume(output_root=tmp_path, run_id="fixture", identity=identity)
    ledger.stop_cpu(status="stopped", computed_pass=True, reason="fixture")
    assert ledger.summary["computed_pass"] is True
    assert ledger.summary["accepted"] is False
    assert ledger.summary["delivery_reachable"] is False
    with pytest.raises(ResumeMismatchError):
        RunLedger.create_or_resume(
            output_root=tmp_path,
            run_id="fixture",
            identity={**identity, "source_video_sha256": _sha("d"), "run_identity_sha256": _sha("e")},
        )


def test_missing_dependency_is_structured_blocked_without_install():
    calls: list[list[str]] = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        raise AssertionError("missing dependencies must not be executed")

    result = preflight_dependencies(
        route_python="/does/not/exist/python",
        backend_python="/does/not/exist/backend",
        ffmpeg="/does/not/exist/ffmpeg",
        ffprobe="/does/not/exist/ffprobe",
        colmap="/does/not/exist/colmap",
        runner=runner,
    )
    assert result["status"] == "blocked"
    assert len(result["blocked"]) == 5
    assert calls == []


def test_cli_preflight_stop_does_not_touch_training(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"synthetic video")
    result = run_raw_video_pipeline(
        input_video=source,
        output_root=tmp_path / "runs",
        run_id="cpu-preflight",
        stop_after="preflight",
        tool_paths={
            "route_python": sys.executable,
            "backend_python": sys.executable,
            "ffmpeg": sys.executable,
            "ffprobe": sys.executable,
            "colmap": sys.executable,
        },
        runner=_fixture_version_probe_runner,
        code_identity_override={"fixture": True},
    )
    assert result["status"] == "stopped"
    assert result["accepted"] is False
    assert result["gpu_invoked"] is False
    run_record = json.loads((tmp_path / "runs" / "cpu-preflight" / "run.json").read_text())
    assert "training" not in json.dumps(run_record).lower()


def test_preprocess_manifest_bypass_is_explicitly_blocked(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"synthetic video")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"frames": []}))
    with pytest.raises(Exception, match="preprocess-manifest"):
        run_raw_video_pipeline(
            input_video=source,
            output_root=tmp_path / "runs",
            run_id="manifest-disabled",
            stop_after="preflight",
            preprocess_manifest=manifest,
            tool_paths={
                "route_python": sys.executable,
                "backend_python": sys.executable,
                "ffmpeg": sys.executable,
                "ffprobe": sys.executable,
                "colmap": sys.executable,
            },
            code_identity_override={"fixture": True},
        )


def test_pipeline_probe_stop_records_dynamic_canonical_probe_without_training(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"synthetic video")
    result = run_raw_video_pipeline(
        input_video=source,
        output_root=tmp_path / "runs",
        run_id="cpu-probe",
        stop_after="probe",
        ffprobe_payload=_probe(90, width=800, height=600),
        tool_paths={
            "route_python": sys.executable,
            "backend_python": sys.executable,
            "ffmpeg": sys.executable,
            "ffprobe": sys.executable,
            "colmap": sys.executable,
        },
        runner=_fixture_version_probe_runner,
        code_identity_override={"fixture": True},
    )
    assert result["status"] == "stopped"
    assert result["accepted"] is False
    probe = json.loads((tmp_path / "runs" / "cpu-probe" / "video_probe-v1.json").read_text())
    assert probe["stream"]["visible_width"] == 600
    assert probe["stream"]["visible_height"] == 800
    assert probe["source"]["sha256"] == hashlib.sha256(b"synthetic video").hexdigest()


def test_probe_contract_failure_keeps_blocked_attempt_record(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"synthetic video")
    payload = _probe()
    payload["streams"][0].pop("tags")
    payload["streams"][0]["side_data_list"] = [{
        "side_data_type": "Display Matrix",
        "displaymatrix": "1 0 0 0.1 1 0 0 0 1",
    }]
    result = run_raw_video_pipeline(
        input_video=source,
        output_root=tmp_path / "runs",
        run_id="blocked-probe",
        stop_after="probe",
        ffprobe_payload=payload,
        tool_paths={
            "route_python": sys.executable,
            "backend_python": sys.executable,
            "ffmpeg": sys.executable,
            "ffprobe": sys.executable,
            "colmap": sys.executable,
        },
        runner=_fixture_version_probe_runner,
        code_identity_override={"fixture": True},
    )
    assert result["status"] == "blocked"
    attempt = tmp_path / "runs" / "blocked-probe" / "stages/probe/attempt-0001/result.json"
    record = json.loads(attempt.read_text())
    assert record["status"] == "blocked"
    assert "display matrix" in record["result"]["reason"]


def test_validator_source_keeps_set_a_as_fixture_only():
    path = Path(__file__).parents[2] / "scripts/longsplat/validate_external_colmap_contract.py"
    source = path.read_text(encoding="utf-8")
    assert "778.038926108633" not in source
    assert "range(91)" not in source
    assert "1280" not in source


def test_nested_external_route_error_describes_dynamic_camera_contract():
    path = Path(__file__).parents[2] / "third_party/LongSplat/scene/__init__.py"
    source = path.read_text(encoding="utf-8")
    assert "per-video centered-PINHOLE contract" in source
    assert "1280x720 contract" not in source


def test_camera_staging_binds_only_undistorted_output_pixels_and_keeps_provenance(tmp_path):
    import cv2

    source = tmp_path / "input.mp4"
    source.write_bytes(b"synthetic video")
    original_dir = tmp_path / "original"
    undistorted_dir = tmp_path / "undistorted" / "images"
    original_dir.mkdir()
    undistorted_dir.mkdir(parents=True)
    original = original_dir / "frame_000000.png"
    undistorted = undistorted_dir / "frame_000000.png"
    cv2.imwrite(str(original), np.full((480, 640, 3), 40, dtype=np.uint8))
    cv2.imwrite(str(undistorted), np.full((480, 640, 3), 180, dtype=np.uint8))
    raw = CameraParameters("SIMPLE_RADIAL", 640, 480, 400, 400, 320, 240, (0.01,))
    pinhole = CameraParameters("PINHOLE", 640, 480, 402, 403, 320, 240)
    result = stage_centered_pinhole(
        raw_camera=raw,
        undistorted_camera=pinhole,
        source_video_sha256=_sha("a"),
        canonical_media_sha256=_sha("b"),
        frame_records=[
            {
                "staged_name": "frame_000000.png",
                "path": str(undistorted),
                "sha256": sha256_file(undistorted),
                "width": 640,
                "height": 480,
            }
        ],
    )
    assert result["frames"][0]["path"] == str(undistorted)
    assert result["frames"][0]["sha256"] == sha256_file(undistorted)
    assert result["frames"][0]["sha256"] != sha256_file(original)


def test_mock_colmap_order_layout_evidence_and_camera_staging_are_cpu_only(tmp_path):
    import cv2

    source = tmp_path / "input.mp4"
    source.write_bytes(b"synthetic video")
    calls: list[list[str]] = []

    def mock_runner(argv, cwd=None, **kwargs):
        calls.append(list(argv))
        if "-frames:v" in argv:
            pattern = Path(argv[-1])
            for index in range(int(argv[argv.index("-frames:v") + 1])):
                output = Path(str(pattern).replace("%06d", f"{index:06d}"))
                output.parent.mkdir(parents=True, exist_ok=True)
                image = np.full((480, 640, 3), 30 + index * 70, dtype=np.uint8)
                image[100 + index : 220 + index, 100:300] = 190
                cv2.imwrite(str(output), image)
            return subprocess.CompletedProcess(argv, 0, "", "")
        if len(argv) > 1 and argv[1] == "mapper":
            output = Path(argv[argv.index("--output_path") + 1]) / "7"
            output.mkdir(parents=True, exist_ok=True)
            for name in ("cameras.bin", "images.bin", "points3D.bin"):
                (output / name).write_bytes(b"binary-placeholder")
        elif len(argv) > 1 and argv[1] == "model_converter":
            output = Path(argv[argv.index("--output_path") + 1])
            output.mkdir(parents=True, exist_ok=True)
            (output / "cameras.txt").write_text("1 PINHOLE 640 480 400 402 320 240\n")
            (output / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 frame_000000.png\n\n"
                "2 1 0 0 0 1 0 0 1 frame_000001.png\n\n"
            )
            (output / "points3D.txt").write_text(
                "1 0 0 1 255 255 255 0.5 1 0 2 1\n"
            )
        elif len(argv) > 1 and argv[1] == "image_undistorter":
            output = Path(argv[argv.index("--output_path") + 1])
            images = output / "images"
            images.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                image = np.full((480, 640, 3), 100 + index * 30, dtype=np.uint8)
                image[80:260, 90 + index : 310 + index] = 220
                cv2.imwrite(str(images / f"frame_{index:06d}.png"), image)
            (output / "sparse").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    result = run_raw_video_pipeline(
        input_video=source,
        output_root=tmp_path / "runs",
        run_id="mock-camera-staging",
        stop_after="camera-staging",
        selection_config=FrameSelectionConfig(target_fps=1.0, min_frames=2, max_frames=2, duplicate_hash_threshold=0),
        ffprobe_payload=_probe(width=640, height=480),
        tool_paths={
            "route_python": sys.executable,
            "backend_python": sys.executable,
            "ffmpeg": sys.executable,
            "ffprobe": sys.executable,
            "colmap": sys.executable,
        },
        runner=mock_runner,
        code_identity_override={"fixture": "mock-colmap"},
    )
    assert result["status"] == "stopped"
    assert result["computed_pass"] is True
    assert result["accepted"] is False
    assert [argv[1] for argv in calls if len(argv) > 1 and argv[1] in {"feature_extractor", "sequential_matcher", "mapper", "bundle_adjuster", "model_converter", "image_undistorter"}] == [
        "feature_extractor", "sequential_matcher", "mapper", "bundle_adjuster", "model_converter", "image_undistorter", "model_converter"
    ]
    run_dir = tmp_path / "runs" / "mock-camera-staging"
    attempts = sorted((run_dir / "stages" / "camera-staging").glob("attempt-*/camera_staging/camera_contract-v1.json"))
    contract = json.loads(attempts[-1].read_text())
    assert contract["frames"][0]["provenance"] == "undistorted_output_frame"
    assert contract["pixel_provenance"]["undistorted_output_frames"][0]["path"] == contract["frames"][0]["path"]
    assert contract["pixel_provenance"]["selected_canonical_frames"][0]["path"] != contract["frames"][0]["path"]
    unsigned = dict(contract)
    unsigned.pop("contract_sha256")
    from scripts.longsplat.pipeline_contract import stable_sha256
    assert stable_sha256(unsigned) == contract["contract_sha256"]


def test_external_validator_requires_exact_camera_and_final_frame_sha(tmp_path):
    import struct
    import cv2
    from scripts.longsplat.validate_external_colmap_contract import validate

    root = tmp_path / "validator"
    image_dir = root / "input" / "images"
    sparse = root / "input" / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse.mkdir(parents=True)
    frame = image_dir / "frame_000000.png"
    cv2.imwrite(str(frame), np.full((480, 640, 3), 80, dtype=np.uint8))
    raw = CameraParameters("SIMPLE_RADIAL", 640, 480, 400, 400, 320, 240, (0.01,))
    pinhole = CameraParameters("PINHOLE", 640, 480, 401, 402, 320, 240)
    contract = stage_centered_pinhole(
        raw_camera=raw,
        undistorted_camera=pinhole,
        source_video_sha256=_sha("a"),
        canonical_media_sha256=_sha("b"),
        frame_records=[{"staged_name": frame.name, "path": str(frame), "sha256": sha256_file(frame), "width": 640, "height": 480}],
    )
    (sparse / "cameras.txt").write_text("1 PINHOLE 640 480 401 402 320 240\n")
    (sparse / "images.txt").write_text("1 1 0 0 0 0 0 0 1 frame_000000.png\n\n")
    (sparse / "points3D.txt").write_text("# no points\n")
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<idddddddi", 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1))
        handle.write(frame.name.encode("utf-8") + b"\x00")
        handle.write(struct.pack("<Q", 0))
    (root / "staging_manifest.json").write_text(json.dumps({
        "source_video_sha256": _sha("a"),
        "canonical_media_sha256": _sha("b"),
        "camera_contract": contract,
    }))
    result = validate(root)
    assert result["computed_pass"] is True
    assert result["accepted"] is False
    assert result["staged_image_sha_verified"] == 1


def _synthetic_rewritable_model() -> ColmapModel:
    return ColmapModel(
        cameras={
            1: {
                "camera_id": 1,
                "model": "PINHOLE",
                "width": 10,
                "height": 8,
                "params": [500.0, 501.0, 5.0, 4.0],
            }
        },
        images={
            1: {
                "image_id": 1,
                "qvec": (1.0, 0.0, 0.0, 0.0),
                "tvec": (0.0, 0.0, 0.0),
                "camera_id": 1,
                "name": "frame_000000.png",
                "points2D": [
                    {"x": 1.0, "y": 1.0, "point3D_id": -1},
                    {"x": 2.0, "y": 1.0, "point3D_id": 10},
                    {"x": 7.5, "y": 5.5, "point3D_id": 10},
                    {"x": 8.0, "y": 5.0, "point3D_id": 11},
                ],
            },
            2: {
                "image_id": 2,
                "qvec": (1.0, 0.0, 0.0, 0.0),
                "tvec": (1.0, 0.0, 0.0),
                "camera_id": 1,
                "name": "frame_000001.png",
                "points2D": [
                    {"x": 2.0, "y": 1.0, "point3D_id": 10},
                    {"x": 9.0, "y": 2.0, "point3D_id": 12},
                    {"x": 3.0, "y": 6.0, "point3D_id": 13},
                ],
            },
        },
        points3D={
            10: {"point3D_id": 10, "xyz": (0.0, 0.0, 3.0), "rgb": (10, 20, 30), "error": 0.25, "track": [(1, 1), (1, 2), (2, 0)]},
            11: {"point3D_id": 11, "xyz": (1.0, 0.0, 3.0), "rgb": (30, 20, 10), "error": 0.5, "track": [(1, 3)]},
            12: {"point3D_id": 12, "xyz": (2.0, 0.0, 3.0), "rgb": (40, 50, 60), "error": 0.75, "track": [(2, 1)]},
            13: {"point3D_id": 13, "xyz": (3.0, 0.0, 3.0), "rgb": (70, 80, 90), "error": 1.0, "track": [(2, 2)]},
        },
    )


def test_longsplat_input_rewrites_nonzero_crop_and_reindexes_tracks(tmp_path):
    model = _synthetic_rewritable_model()
    rewritten, stats = rewrite_colmap_model_for_crop(
        model,
        crop={"x0": 2, "y0": 1, "width": 6, "height": 5},
        final_camera={
            "model": "PINHOLE",
            "width": 6,
            "height": 5,
            "fx": 500.0,
            "fy": 501.0,
            "cx": 3.0,
            "cy": 3.0,
        },
    )
    assert stats["observation_rewrite"] is True
    assert stats["dropped_observation_count"] == 4
    assert stats["points_dropped_no_valid_track"] == 3
    assert rewritten.images[1]["points2D"] == [
        {"x": 0.0, "y": 0.0, "point3D_id": 10},
        {"x": 5.5, "y": 4.5, "point3D_id": 10},
    ]
    assert rewritten.images[2]["points2D"] == [{"x": 0.0, "y": 0.0, "point3D_id": 10}]
    assert rewritten.points3D[10]["track"] == [(1, 0), (1, 1), (2, 0)]
    assert set(rewritten.points3D) == {10}
    validate_colmap_model(rewritten, image_width=6, image_height=5, require_pinhole=True)
    write_colmap_model_text(rewritten, tmp_path / "sparse")
    parsed = parse_colmap_model_text(tmp_path / "sparse")
    assert compare_colmap_models(rewritten, parsed, tolerance=1e-12)["track_pairs_equal"]


def test_longsplat_input_rejects_dangling_and_nonfinite_model_references():
    model = _synthetic_rewritable_model()
    model.points3D[10]["track"] = [(1, 999)]
    with pytest.raises(LongSplatInputBlocked, match="dangling"):
        validate_colmap_model(model)
    model = _synthetic_rewritable_model()
    model.images[1]["points2D"][1]["x"] = float("nan")
    with pytest.raises(LongSplatInputBlocked, match="non-finite"):
        validate_colmap_model(model)


def test_longsplat_input_materializes_independent_pixel_identity(tmp_path):
    import cv2

    source = tmp_path / "parent" / "frame_000000.png"
    source.parent.mkdir()
    pixels = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    assert cv2.imwrite(str(source), pixels)
    records, aggregate = materialize_training_images(
        [{"staged_name": source.name, "path": str(source)}],
        tmp_path / "training" / "images",
        expected_width=5,
        expected_height=4,
    )
    destination = Path(records[0]["path"])
    assert destination.is_file() and not destination.is_symlink()
    assert destination.read_bytes() == source.read_bytes()
    assert records[0]["source_file_sha256"] == records[0]["file_sha256"]
    assert aggregate


def _write_synthetic_colmap_binary(model: ColmapModel, root: Path) -> None:
    import struct

    root.mkdir(parents=True, exist_ok=True)
    with (root / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(model.cameras)))
        for camera_id, camera in sorted(model.cameras.items()):
            assert camera["model"] == "PINHOLE"
            handle.write(struct.pack("<iiQQ", camera_id, 1, camera["width"], camera["height"]))
            handle.write(struct.pack("<dddd", *camera["params"]))
    with (root / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(model.images)))
        for image_id, image in sorted(model.images.items()):
            handle.write(struct.pack("<idddddddi", image_id, *image["qvec"], *image["tvec"], image["camera_id"]))
            handle.write(image["name"].encode("utf-8") + b"\x00")
            handle.write(struct.pack("<Q", len(image["points2D"])))
            for point in image["points2D"]:
                handle.write(struct.pack("<ddq", point["x"], point["y"], point["point3D_id"]))
    with (root / "points3D.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(model.points3D)))
        for point_id, point in sorted(model.points3D.items()):
            handle.write(struct.pack("<QdddBBBd", point_id, *point["xyz"], *point["rgb"], point["error"]))
            handle.write(struct.pack("<Q", len(point["track"])))
            for image_id, point2d_idx in point["track"]:
                handle.write(struct.pack("<ii", image_id, point2d_idx))


def _make_parent_binding_fixture(
    root: Path,
    *,
    source_sha: str,
    frame_count: int,
    width: int,
    height: int,
    fresh_identity: bool = True,
) -> dict:
    import cv2

    root.mkdir(parents=True)
    source_video_path = root / "fixture.mp4"
    source_video_path.write_bytes(b"fresh parent source")
    names = [f"capture_{index:03d}.png" for index in range(frame_count)]
    image_dir = root / "stages" / "camera-staging" / "attempt-0002" / "camera_staging" / "canonical_images"
    image_dir.mkdir(parents=True)
    frame_records = []
    for index, name in enumerate(names):
        pixels = np.full((height, width, 3), index + 20, dtype=np.uint8)
        path = image_dir / name
        assert cv2.imwrite(str(path), pixels)
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        frame_records.append({
            "staged_name": name,
            "path": str(path),
            "sha256": sha256_file(path),
            "decoded_pixel_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
            "width": width,
            "height": height,
        })
    from scripts.longsplat.pipeline_contract import stable_sha256

    aggregate = stable_sha256([
        {"name": item["staged_name"], "decoded_pixel_sha256": item["decoded_pixel_sha256"]}
        for item in frame_records
    ])
    media = {
        "schema_version": "canonical-media-v1",
        "status": "ready",
        "source": {"path": "fixture.mp4", "sha256": source_sha},
        "binding": {"source_video_sha256": source_sha, "canonical_media_sha256": _sha("m")},
        "canonical": {
            "width": width,
            "height": height,
            "pixels_sha256": _sha("p"),
            "filter_graph": "setsar=1",
        },
    }
    media_path = root / "canonical_media-v1.json"
    media_path.write_text(json.dumps(media, sort_keys=True), encoding="utf-8")
    final_camera = {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "fx": 300.0 + width,
        "fy": 301.0 + height,
        "cx": width / 2,
        "cy": height / 2,
        "distortion": [],
    }
    und_camera = {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "fx": final_camera["fx"],
        "fy": final_camera["fy"],
        "cx": final_camera["cx"],
        "cy": final_camera["cy"],
        "distortion": [],
    }
    contract = {
        "schema_version": "camera-contract-v1",
        "status": "computed_pass",
        "accepted": False,
        "delivery_reachable": False,
        "source_video_sha256": source_sha,
        "canonical_media_binding_sha256": media["binding"]["canonical_media_sha256"],
        "canonical_media_pixels_sha256": media["canonical"]["pixels_sha256"],
        "canonical_media_record_path": str(media_path),
        "canonical_media_record_sha256": sha256_file(media_path),
        "camera": final_camera,
        "frame_count": frame_count,
        "frame_names": names,
        "frames": frame_records,
        "staging": {
            "final_staged_pixel_aggregate_sha256": aggregate,
            "undistorted_camera": und_camera,
            "crop_roi": {"x0": 0, "y0": 0, "width": width, "height": height},
        },
    }
    contract["contract_sha256"] = stable_sha256(contract)
    contract_path = image_dir.parent / "camera_contract-v1.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True), encoding="utf-8")

    model = ColmapModel(
        cameras={1: {"camera_id": 1, "model": "PINHOLE", "width": width, "height": height, "params": [final_camera["fx"], final_camera["fy"], final_camera["cx"], final_camera["cy"]]}},
        images={
            index + 1: {
                "image_id": index + 1,
                "qvec": (1.0, 0.0, 0.0, 0.0),
                "tvec": (float(index), 0.0, 0.0),
                "camera_id": 1,
                "name": name,
                "points2D": [],
            }
            for index, name in enumerate(names)
        },
        points3D={},
    )
    txt_dir = root / "stages" / "colmap" / "attempt-0003" / "workspace" / "evidence" / "undistorted_txt"
    bin_dir = root / "stages" / "colmap" / "attempt-0003" / "workspace" / "undistorted" / "sparse"
    write_colmap_model_text(model, txt_dir)
    _write_synthetic_colmap_binary(model, bin_dir)
    tool_sha = _sha("c")
    tool_identity = {"resolved_path": "/usr/bin/colmap", "sha256": tool_sha}
    command_result = {"stage": "matching", "argv": ["/usr/bin/colmap", "sequential_matcher"], "exit_code": 0, "tool_identity": {"sha256": tool_sha}}
    colmap_result = {
        "plan": {
            "source_video_sha256": source_sha,
            "canonical_media_sha256": media["canonical"]["pixels_sha256"],
            "canonical_width": width,
            "canonical_height": height,
            "camera_model_requested": "SIMPLE_RADIAL",
            "camera_prior": None,
            "single_camera": True,
            "paths": {"undistorted_sparse_txt": str(txt_dir), "undistorted_sparse_binary": str(bin_dir)},
        },
        "execution_pass": True,
        "computed_pass": True,
        "command_results": [command_result],
        "evidence": {"component_count": 1, "registered_image_count": frame_count, "registered_image_names": names},
        "artifacts": [{"path": str(txt_dir / name), "sha256": sha256_file(txt_dir / name)} for name in ("cameras.txt", "images.txt", "points3D.txt")],
    }

    config = {"depth_source": "disabled", "camera_model": "SIMPLE_RADIAL", "matching": "sequential", "camera_prior": None}
    identity_kwargs = {}
    if fresh_identity:
        identity_kwargs = {
            "source_video_path": source_video_path,
            "source_video_size_bytes": source_video_path.stat().st_size,
            "output_root_input": root / "output-input",
            "output_root_resolved": root / "output-resolved",
            "run_root_resolved": root / "run-root",
        }
    identity = build_run_identity(
        source_video_sha256=source_sha,
        canonical_config_sha256=stable_sha256(config),
        tool_identity_sha256=stable_sha256({"fixture": tool_sha}),
        code_identity_value={"code_identity_sha256": _sha("i")},
        **identity_kwargs,
    )
    (root / "identity.json").write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    (root / "config.json").write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    (root / "preflight.json").write_text(json.dumps({"status": "passed", "dependencies": {"colmap": tool_identity}}, sort_keys=True), encoding="utf-8")

    def stage_record(stage: str, attempt: str, status: str, result: dict) -> dict:
        attempt_dir = root / "stages" / stage / attempt
        attempt_dir.mkdir(parents=True, exist_ok=True)
        record = {"schema_version": "stage-attempt-result-v1", "stage": stage, "status": status, "identity": identity, "result": result}
        path = attempt_dir / "result.json"
        path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        return {"attempt": attempt, "status": status, "result_path": str(path.relative_to(root))}

    failed_camera = stage_record("camera-staging", "attempt-0001", "blocked", {"reason": "fixture older failure", "artifacts": []})
    camera_result = {"contract_path": str(contract_path), "contract": contract, "artifacts": [{"path": str(contract_path), "sha256": sha256_file(contract_path)}] + [{"path": item["path"], "sha256": item["sha256"]} for item in frame_records]}
    passed_camera = stage_record("camera-staging", "attempt-0002", "passed", camera_result)
    passed_colmap = stage_record("colmap", "attempt-0003", "passed", colmap_result)
    failed_colmap = stage_record("colmap", "attempt-0004", "blocked", {"reason": "newer fixture failure", "artifacts": []})
    run = {"status": "stopped", "computed_pass": True, "identity": identity, "stages": {"camera-staging": [failed_camera, passed_camera], "colmap": [passed_colmap, failed_colmap]}}
    (root / "run.json").write_text(json.dumps(run, sort_keys=True), encoding="utf-8")
    return {
        "source_sha": source_sha,
        "tool_sha": tool_sha,
        "names": names,
        "contract": contract,
        "contract_path": contract_path,
        "identity": identity,
        "identity_path": root / "identity.json",
        "run_path": root / "run.json",
        "result_paths": [
            root / "stages" / "camera-staging" / "attempt-0001" / "result.json",
            root / "stages" / "camera-staging" / "attempt-0002" / "result.json",
            root / "stages" / "colmap" / "attempt-0003" / "result.json",
            root / "stages" / "colmap" / "attempt-0004" / "result.json",
        ],
    }


def _rewrite_parent_identity(fixture: dict, identity: dict) -> None:
    fixture["identity_path"].write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    run = json.loads(fixture["run_path"].read_text(encoding="utf-8"))
    run["identity"] = identity
    fixture["run_path"].write_text(json.dumps(run, sort_keys=True), encoding="utf-8")
    for result_path in fixture["result_paths"]:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["identity"] = identity
        result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")


def test_parent_binding_is_dynamic_and_uses_latest_reusable_attempt(tmp_path):
    first = _make_parent_binding_fixture(tmp_path / "parent-a", source_sha=_sha("a"), frame_count=2, width=8, height=6)
    second = _make_parent_binding_fixture(tmp_path / "parent-b", source_sha=_sha("b"), frame_count=3, width=10, height=7)
    first_evidence = load_parent_camera_staging_evidence(
        tmp_path / "parent-a", expected_source_video_sha256=_sha("a"), current_colmap_identity={"sha256": first["tool_sha"]}
    )
    second_evidence = load_parent_camera_staging_evidence(
        tmp_path / "parent-b", expected_source_video_sha256=_sha("b"), current_colmap_identity={"sha256": second["tool_sha"]}
    )
    assert first_evidence["parent_binding"]["camera_staging_attempt"] == "attempt-0002"
    assert first_evidence["parent_binding"]["colmap_attempt"] == "attempt-0003"
    assert first_evidence["parent_binding"]["frame_count"] == 2
    assert second_evidence["parent_binding"]["frame_count"] == 3
    assert first_evidence["parent_binding_sha256"] != second_evidence["parent_binding_sha256"]
    assert first_evidence["parent_binding"]["final_camera"]["width"] == 8
    assert second_evidence["parent_binding"]["final_camera"]["width"] == 10
    assert first["identity"]["source_video_path"] == str((tmp_path / "parent-a" / "fixture.mp4").resolve())
    assert first["identity"]["source_video_size_bytes"] == len(b"fresh parent source")
    assert first["identity"]["output_root_input"] == str(tmp_path / "parent-a" / "output-input")
    assert first["identity"]["output_root_resolved"] == str((tmp_path / "parent-a" / "output-resolved").resolve())
    assert first["identity"]["run_root_resolved"] == str((tmp_path / "parent-a" / "run-root").resolve())


def test_parent_binding_accepts_legacy_run_identity_payload(tmp_path):
    fixture = _make_parent_binding_fixture(
        tmp_path / "legacy-parent",
        source_sha=_sha("a"),
        frame_count=2,
        width=8,
        height=6,
        fresh_identity=False,
    )
    evidence = load_parent_camera_staging_evidence(
        tmp_path / "legacy-parent",
        expected_source_video_sha256=_sha("a"),
        current_colmap_identity={"sha256": fixture["tool_sha"]},
    )
    assert set(fixture["identity"]) == {
        "schema_version",
        "source_video_sha256",
        "canonical_config_sha256",
        "tool_identity_sha256",
        "code_identity",
        "run_identity_sha256",
    }
    assert evidence["identity"] == fixture["identity"]


def test_parent_binding_rejects_run_identity_extension_tamper(tmp_path):
    fixture = _make_parent_binding_fixture(tmp_path / "tampered-parent", source_sha=_sha("a"), frame_count=2, width=8, height=6)
    tampered = dict(fixture["identity"])
    tampered["source_video_size_bytes"] += 1
    _rewrite_parent_identity(fixture, tampered)
    with pytest.raises(LongSplatInputBlocked, match="parent run identity stable SHA"):
        load_parent_camera_staging_evidence(
            tmp_path / "tampered-parent",
            expected_source_video_sha256=_sha("a"),
            current_colmap_identity={"sha256": fixture["tool_sha"]},
        )


def test_parent_binding_rejects_missing_run_identity_base_field(tmp_path):
    fixture = _make_parent_binding_fixture(tmp_path / "missing-base-parent", source_sha=_sha("a"), frame_count=2, width=8, height=6)
    tampered = dict(fixture["identity"])
    tampered.pop("canonical_config_sha256")
    unsigned = dict(tampered)
    unsigned.pop("schema_version", None)
    unsigned.pop("run_identity_sha256", None)
    tampered["run_identity_sha256"] = stable_sha256(unsigned)
    _rewrite_parent_identity(fixture, tampered)
    with pytest.raises(LongSplatInputBlocked, match="parent run identity is missing canonical_config_sha256"):
        load_parent_camera_staging_evidence(
            tmp_path / "missing-base-parent",
            expected_source_video_sha256=_sha("a"),
            current_colmap_identity={"sha256": fixture["tool_sha"]},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("digest", "parent run identity stable SHA"),
        ("missing", "parent run identity SHA is required"),
        ("digest_type", "parent run identity SHA is required"),
        ("schema", "parent run identity schema"),
    ],
)
def test_parent_binding_rejects_run_identity_digest_contract(tmp_path, mutation, message):
    fixture = _make_parent_binding_fixture(tmp_path / f"invalid-{mutation}", source_sha=_sha("a"), frame_count=2, width=8, height=6)
    tampered = dict(fixture["identity"])
    if mutation == "digest":
        tampered["run_identity_sha256"] = _sha("z")
    elif mutation == "missing":
        tampered.pop("run_identity_sha256")
    elif mutation == "digest_type":
        tampered["run_identity_sha256"] = ["not-a-digest"]
    else:
        tampered["schema_version"] = "run-identity-legacy"
    _rewrite_parent_identity(fixture, tampered)
    with pytest.raises(LongSplatInputBlocked, match=message):
        load_parent_camera_staging_evidence(
            tmp_path / f"invalid-{mutation}",
            expected_source_video_sha256=_sha("a"),
            current_colmap_identity={"sha256": fixture["tool_sha"]},
        )


def test_parent_binding_rejects_source_pixel_path_and_tool_drift(tmp_path):
    fixture = _make_parent_binding_fixture(tmp_path / "parent", source_sha=_sha("a"), frame_count=2, width=8, height=6)
    with pytest.raises(LongSplatInputBlocked, match="source video SHA"):
        load_parent_camera_staging_evidence(tmp_path / "parent", expected_source_video_sha256=_sha("b"), current_colmap_identity={"sha256": fixture["tool_sha"]})
    with pytest.raises(LongSplatInputBlocked, match="COLMAP executable bytes"):
        load_parent_camera_staging_evidence(tmp_path / "parent", expected_source_video_sha256=_sha("a"), current_colmap_identity={"sha256": _sha("z")})
    image = Path(fixture["contract"]["frames"][0]["path"])
    image.write_bytes(b"mutated")
    with pytest.raises(LongSplatInputBlocked, match="reusable successful camera-staging"):
        load_parent_camera_staging_evidence(tmp_path / "parent", expected_source_video_sha256=_sha("a"), current_colmap_identity={"sha256": fixture["tool_sha"]})


def test_parent_binding_rejects_contract_stable_sha_and_path_escape(tmp_path):
    fixture = _make_parent_binding_fixture(tmp_path / "parent", source_sha=_sha("a"), frame_count=2, width=8, height=6)
    contract_path = fixture["contract_path"]
    contract = json.loads(contract_path.read_text())
    contract["camera"]["width"] = 99
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    camera_result_path = tmp_path / "parent" / "stages" / "camera-staging" / "attempt-0002" / "result.json"
    camera_result = json.loads(camera_result_path.read_text())
    camera_result["result"]["contract"] = contract
    camera_result["result"]["artifacts"][0]["sha256"] = sha256_file(contract_path)
    camera_result_path.write_text(json.dumps(camera_result, sort_keys=True))
    with pytest.raises(LongSplatInputBlocked, match="camera contract stable SHA"):
        load_parent_camera_staging_evidence(tmp_path / "parent", expected_source_video_sha256=_sha("a"), current_colmap_identity={"sha256": fixture["tool_sha"]})

    fixture = _make_parent_binding_fixture(tmp_path / "parent-escape", source_sha=_sha("a"), frame_count=2, width=8, height=6)
    result_path = tmp_path / "parent-escape" / "stages" / "camera-staging" / "attempt-0002" / "result.json"
    result = json.loads(result_path.read_text())
    result["result"]["artifacts"][0]["path"] = str(tmp_path / "outside.json")
    result_path.write_text(json.dumps(result, sort_keys=True))
    with pytest.raises(LongSplatInputBlocked, match="reusable successful camera-staging"):
        load_parent_camera_staging_evidence(tmp_path / "parent-escape", expected_source_video_sha256=_sha("a"), current_colmap_identity={"sha256": fixture["tool_sha"]})


def test_longsplat_input_production_has_no_sample_specific_parent_allowlist():
    raw_source = (Path(__file__).resolve().parents[2] / "scripts" / "longsplat" / "raw_pipeline.py").read_text()
    input_source = (Path(__file__).resolve().parents[2] / "scripts" / "longsplat" / "longsplat_input.py").read_text()
    assert "_B_V5_" not in raw_source
    assert "B v5" not in raw_source
    assert "B-v5" not in input_source


def test_longsplat_input_precreates_only_private_converter_destinations(tmp_path):
    binary = tmp_path / "attempt" / "_model_bin"
    roundtrip = tmp_path / "attempt" / "_model_roundtrip_txt"
    final = tmp_path / "attempt" / "sparse" / "0"
    prepare_converter_output_dirs(binary, roundtrip, final)
    assert binary.is_dir()
    assert roundtrip.is_dir()
    assert final.is_dir()
    assert not (tmp_path / "attempt" / "source").exists()


def test_future_smoke_plan_freezes_backend_seed_without_cli_seed(tmp_path):
    route_root = Path(__file__).resolve().parents[2]
    (tmp_path / "training").mkdir()
    plan = build_future_smoke_plan(
        training_root=tmp_path / "training",
        backend_python=sys.executable,
        route_root=route_root,
        containment_root=tmp_path,
        code_identity={"code_identity_sha256": _sha("a")},
        backend_identity=None,
    )
    seed = plan["frozen_contract"]["seed"]
    assert plan["schema_version"] == "longsplat-future-smoke-plan-v2"
    assert "--seed" not in plan["argv"]
    assert seed["value"] == 0
    assert seed["source"] == "third_party/LongSplat/utils/general_utils.py::safe_state"
    assert seed["application_order"] == "before_GaussianModel_and_Scene_and_training"
    assert seed["python_random"] is True
    assert seed["numpy"] is True
    assert seed["torch_manual_seed"] is True
    assert seed["cli_flag"] is None
    assert seed["bitwise_cuda_determinism"] is False
    assert "nondeterministic" in seed["limitation"]
    validate_future_smoke_plan(
        plan,
        route_root=route_root,
        containment_root=tmp_path,
        training_root=tmp_path / "training",
    )


def test_future_smoke_plan_rejects_seed_cli_nonzero_and_identity_drift(tmp_path):
    route_root = Path(__file__).resolve().parents[2]
    (tmp_path / "training").mkdir()
    plan = build_future_smoke_plan(
        training_root=tmp_path / "training",
        backend_python=sys.executable,
        route_root=route_root,
        containment_root=tmp_path,
        code_identity={"code_identity_sha256": _sha("a")},
        backend_identity=None,
    )

    missing_seed = json.loads(json.dumps(plan))
    missing_seed["frozen_contract"].pop("seed")
    with pytest.raises(LongSplatInputBlocked, match="frozen_contract.seed"):
        validate_future_smoke_plan(missing_seed, route_root=route_root, containment_root=tmp_path, training_root=tmp_path / "training")

    nonzero_seed = json.loads(json.dumps(plan))
    nonzero_seed["frozen_contract"]["seed"]["value"] = 7
    with pytest.raises(LongSplatInputBlocked, match="seed contract mismatch"):
        validate_future_smoke_plan(nonzero_seed, route_root=route_root, containment_root=tmp_path, training_root=tmp_path / "training")

    seed_cli = json.loads(json.dumps(plan))
    seed_cli["argv"].extend(["--seed", "7"])
    with pytest.raises(LongSplatInputBlocked, match="must not invent a --seed"):
        validate_future_smoke_plan(seed_cli, route_root=route_root, containment_root=tmp_path, training_root=tmp_path / "training")

    identity_drift = json.loads(json.dumps(plan))
    identity_drift["nested_backend_code_identity"]["safe_state_file_sha256"] = _sha("b")
    with pytest.raises(LongSplatInputBlocked, match="nested backend code identity"):
        validate_future_smoke_plan(identity_drift, route_root=route_root, containment_root=tmp_path, training_root=tmp_path / "training")


def test_future_smoke_plan_enforces_dynamic_containment_for_external_and_route_named_roots(tmp_path):
    route = Path(__file__).resolve().parents[2]
    roots = (
        tmp_path / "external-a" / "run-a",
        tmp_path / "external-b" / "run-b",
        tmp_path / "route-instance" / "outputs" / "run-c",
    )
    plans = []
    for root in roots:
        training = root / "training"
        training.mkdir(parents=True)
        plan = build_future_smoke_plan(
            training_root=training,
            backend_python=sys.executable,
            route_root=route,
            containment_root=root,
            code_identity={"code_identity_sha256": _sha("a")},
            backend_identity=None,
        )
        validate_future_smoke_plan(
            plan,
            route_root=route,
            containment_root=root,
            training_root=training,
        )
        plans.append(plan)

    with pytest.raises(LongSplatInputBlocked, match="explicit containment_root"):
        validate_future_smoke_plan(
            plans[0],
            route_root=route,
            training_root=roots[0] / "training",
        )

    cross_root = json.loads(json.dumps(plans[0]))
    cross_root["source_path"] = str(roots[1] / "training")
    source_index = cross_root["argv"].index("--source_path")
    cross_root["argv"][source_index + 1] = str(roots[1] / "training")
    with pytest.raises(LongSplatInputBlocked, match="escapes"):
        validate_future_smoke_plan(
            cross_root,
            route_root=route,
            containment_root=roots[0],
            training_root=roots[1] / "training",
        )

    cross_run = json.loads(json.dumps(plans[0]))
    cross_run["model_path"] = str(roots[1] / "foreign-model")
    model_index = cross_run["argv"].index("--model_path")
    cross_run["argv"][model_index + 1] = str(roots[1] / "foreign-model")
    with pytest.raises(LongSplatInputBlocked, match="model_path escapes"):
        validate_future_smoke_plan(
            cross_run,
            route_root=route,
            containment_root=roots[0],
            training_root=roots[0] / "training",
        )

    symlink_source = roots[0] / "symlink-training"
    symlink_source.symlink_to(roots[1] / "training", target_is_directory=True)
    symlink_plan = json.loads(json.dumps(plans[0]))
    symlink_plan["source_path"] = str(symlink_source)
    symlink_plan["argv"][source_index + 1] = str(symlink_source)
    with pytest.raises(LongSplatInputBlocked, match="symlink"):
        validate_future_smoke_plan(
            symlink_plan,
            route_root=route,
            containment_root=roots[0],
            training_root=symlink_source,
        )


def test_future_workload_profiles_are_fixed_and_formal_plan_is_30000(tmp_path):
    route_root = Path(__file__).resolve().parents[2]
    assert future_workload_profile("smoke100")["iterations"] == 100
    assert future_workload_profile("formal30000")["iterations"] == 30000
    formal_root = tmp_path / "formal-training"
    formal_root.mkdir()
    plan = build_future_smoke_plan(
        training_root=formal_root,
        backend_python=sys.executable,
        route_root=route_root,
        containment_root=tmp_path,
        code_identity={"code_identity_sha256": _sha("a")},
        backend_identity=None,
        workload_profile="formal30000",
    )
    assert plan["workload_profile"] == "formal30000"
    assert plan["frozen_contract"]["iterations"] == 30000
    assert plan["frozen_contract"]["render_iteration"] == 30000
    assert plan["argv"][plan["argv"].index("--iterations") + 1] == "30000"
    assert plan["accepted"] is False
    assert plan["delivery_reachable"] is False
    validate_future_smoke_plan(plan, route_root=route_root, containment_root=tmp_path, training_root=formal_root)

    arbitrary = json.loads(json.dumps(plan))
    arbitrary["frozen_contract"]["iterations"] = 30001
    arbitrary["argv"][arbitrary["argv"].index("--iterations") + 1] = "30001"
    with pytest.raises(LongSplatInputBlocked, match="iterations are fixed"):
        validate_future_smoke_plan(arbitrary, route_root=route_root, containment_root=tmp_path, training_root=formal_root)

    unknown = json.loads(json.dumps(plan))
    unknown["workload_profile"] = "formal30001"
    with pytest.raises(LongSplatInputBlocked, match="unsupported future workload profile"):
        validate_future_smoke_plan(unknown, route_root=route_root, containment_root=tmp_path, training_root=formal_root)


def test_nested_mast3r_imports_are_lazy_and_nonexternal_only():
    nested = Path(__file__).resolve().parents[2] / "third_party" / "LongSplat"
    train_tree = ast.parse((nested / "train.py").read_text(encoding="utf-8"))
    scene_tree = ast.parse((nested / "scene" / "__init__.py").read_text(encoding="utf-8"))

    def is_mast3r_import(node):
        return isinstance(node, ast.ImportFrom) and node.module == "utils.mast3r_utils"

    assert not any(is_mast3r_import(node) for node in train_tree.body)
    assert not any(is_mast3r_import(node) for node in scene_tree.body)

    training = next(node for node in train_tree.body if isinstance(node, ast.FunctionDef) and node.name == "training")
    training_import = next(node for node in ast.walk(training) if is_mast3r_import(node))
    external_branch = next(
        node
        for node in ast.walk(training)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "getattr"
        and any(isinstance(arg, ast.Constant) and arg.value == "external_colmap_pose" for arg in node.test.args)
    )
    assert training_import.lineno > external_branch.lineno

    scene_init = next(node for node in scene_tree.body if isinstance(node, ast.ClassDef) and node.name == "Scene")
    scene_constructor = next(node for node in scene_init.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    scene_import = next(node for node in ast.walk(scene_constructor) if is_mast3r_import(node))
    scene_external_branch = next(
        node
        for node in ast.walk(scene_constructor)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "getattr"
        and any(isinstance(arg, ast.Constant) and arg.value == "external_colmap_pose" for arg in node.test.args)
    )
    assert scene_import.lineno > scene_external_branch.lineno
