from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import preprocess_video as pv  # noqa: E402


def parse_settings(*arguments: str) -> dict[str, object]:
    args = pv.build_arg_parser().parse_args(["source.mp4", "--video-id", "sample", *arguments])
    return pv.resolve_settings(args)


def test_resolve_settings_keeps_existing_defaults_without_config() -> None:
    settings = parse_settings()

    assert settings == {
        "preset": "baseline",
        "target_fps": 5.0,
        "max_long_edge": 1600,
        "segment_method": "scene,time",
        "segment_length_sec": 30.0,
        "segment_overlap_sec": 10.0,
        "min_segment_sec": 2.0,
        "blur_threshold": 40.0,
        "overexposed_ratio": 0.6,
        "underexposed_ratio": 0.6,
        "duplicate_hash_threshold": 4,
        "save_rejected": False,
        "frame_format": "jpg",
        "frame_source": "segment",
    }


def test_resolve_settings_loads_json_config(tmp_path: Path) -> None:
    config_path = tmp_path / "tuning.json"
    config_path.write_text(
        json.dumps(
            {
                "preset": "longsplat",
                "target_fps": 8,
                "blur_threshold": 55.0,
                "save_rejected": True,
                "frame_format": "png",
            }
        ),
        encoding="utf-8",
    )

    settings = parse_settings("--config", str(config_path))

    assert settings["preset"] == "longsplat"
    assert settings["target_fps"] == 8.0
    assert settings["max_long_edge"] == 512
    assert settings["blur_threshold"] == 55.0
    assert settings["save_rejected"] is True
    assert settings["frame_format"] == "png"


def test_explicit_cli_settings_override_json_config(tmp_path: Path) -> None:
    config_path = tmp_path / "tuning.json"
    config_path.write_text(
        json.dumps(
            {
                "preset": "longsplat",
                "target_fps": 8.0,
                "blur_threshold": 55.0,
                "save_rejected": True,
            }
        ),
        encoding="utf-8",
    )

    settings = parse_settings(
        "--config",
        str(config_path),
        "--preset",
        "baseline",
        "--target-fps",
        "3",
        "--no-save-rejected",
    )

    assert settings["preset"] == "baseline"
    assert settings["target_fps"] == 3.0
    assert settings["max_long_edge"] == 1600
    assert settings["blur_threshold"] == 55.0
    assert settings["save_rejected"] is False


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"blur_treshold": 50.0}, "Unknown config setting"),
        ({"target_fps": "fast"}, "target_fps.*must be a number"),
        ({"frame_format": "webp"}, "frame_format.*must be one of"),
    ],
)
def test_load_settings_config_rejects_invalid_values(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(pv.PreprocessError, match=message):
        pv.load_settings_config(config_path)


def test_build_time_windows_with_overlap() -> None:
    windows = pv.build_time_windows(
        duration_sec=95.0,
        segment_length_sec=30.0,
        segment_overlap_sec=10.0,
        min_segment_sec=5.0,
    )

    assert windows == [
        (0.0, 30.0),
        (20.0, 50.0),
        (40.0, 70.0),
        (60.0, 90.0),
        (80.0, 95.0),
    ]


def test_build_time_windows_merges_tiny_tail() -> None:
    windows = pv.build_time_windows(
        duration_sec=53.0,
        segment_length_sec=30.0,
        segment_overlap_sec=10.0,
        min_segment_sec=15.0,
    )

    assert windows == [(0.0, 30.0), (23.0, 53.0)]
    assert all(end - start <= 30.0 for start, end in windows)


def test_build_time_windows_supports_offset() -> None:
    windows = pv.build_time_windows(
        duration_sec=70.0,
        segment_length_sec=30.0,
        segment_overlap_sec=10.0,
        min_segment_sec=5.0,
        offset_sec=100.0,
    )

    assert windows == [(100.0, 130.0), (120.0, 150.0), (140.0, 170.0)]


def test_build_segment_windows_uses_time_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pv, "detect_scene_windows", lambda _path, _min_segment_sec: [])

    segments = pv.build_segment_windows(
        duration_sec=10.0,
        segment_method="scene,time",
        segment_length_sec=30.0,
        segment_overlap_sec=10.0,
        min_segment_sec=2.0,
        normalized_path=Path("normalized.mp4"),
    )

    assert [(segment.start_sec, segment.end_sec, segment.reason) for segment in segments] == [
        (0.0, 10.0, "time_fallback")
    ]


def test_build_segment_windows_splits_scene_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pv, "detect_scene_windows", lambda _path, _min_segment_sec: [(0.0, 75.0)])

    segments = pv.build_segment_windows(
        duration_sec=75.0,
        segment_method="scene,time",
        segment_length_sec=30.0,
        segment_overlap_sec=10.0,
        min_segment_sec=5.0,
        normalized_path=Path("normalized.mp4"),
    )

    assert [segment.reason for segment in segments] == ["scene_split", "scene_split", "scene_split", "scene_split"]
    assert all(segment.duration_sec <= 30.0 for segment in segments)


def test_validate_video_id_rejects_parent_or_current_directory() -> None:
    with pytest.raises(pv.PreprocessError):
        pv.validate_video_id(".")
    with pytest.raises(pv.PreprocessError):
        pv.validate_video_id("..")


def test_scene_dependency_missing_is_not_treated_as_no_cuts(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = __import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("scenedetect"):
            raise ImportError("missing scenedetect")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(pv.PreprocessError, match="PySceneDetect is required"):
        pv.detect_scene_windows(Path("video.mp4"), min_segment_sec=2.0)


def test_quality_metrics_detect_blur_and_exposure() -> None:
    checker = np.zeros((80, 80, 3), dtype=np.uint8)
    checker[::2, ::2] = 255
    checker[1::2, 1::2] = 255
    blurred = cv2.GaussianBlur(checker, (21, 21), 0)

    assert pv.blur_score(checker) > pv.blur_score(blurred)

    white = np.full((20, 20, 3), 255, dtype=np.uint8)
    black = np.zeros((20, 20, 3), dtype=np.uint8)
    mid = np.full((20, 20, 3), 127, dtype=np.uint8)

    assert pv.exposure_ratios(white) == (1.0, 0.0)
    assert pv.exposure_ratios(black) == (0.0, 1.0)
    assert pv.exposure_ratios(mid) == (0.0, 0.0)


def test_average_hash_distance() -> None:
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    inverse = 255 - image

    image_hash = pv.average_hash(image)
    assert pv.hash_distance(image_hash, pv.average_hash(image.copy())) == 0
    assert pv.hash_distance(image_hash, pv.average_hash(inverse)) > 0


def test_write_image_supports_unicode_paths(tmp_path: Path) -> None:
    destination = tmp_path / "中文帧" / "选中.jpg"
    destination.parent.mkdir()
    image = np.full((16, 16, 3), 127, dtype=np.uint8)

    pv.write_image(destination, image)

    assert destination.exists()
    decoded = cv2.imdecode(np.frombuffer(destination.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == image.shape


def test_write_image_supports_png(tmp_path: Path) -> None:
    destination = tmp_path / "selected.png"
    image = np.full((16, 16, 3), 127, dtype=np.uint8)

    pv.write_image(destination, image, frame_format="png")

    assert destination.exists()
    decoded = cv2.imdecode(np.frombuffer(destination.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == image.shape


def test_manifest_writes_stable_relative_paths(tmp_path: Path) -> None:
    source = ROOT / "data" / "raw_videos" / "sample.mp4"
    normalized = ROOT / "data" / "segments" / "sample" / "normalized.mp4"
    segment_path = ROOT / "data" / "segments" / "sample" / "segment_0001.mp4"
    frame_path = "data/frames/sample/selected/segment_0001/frame_000001_t000000.000.jpg"
    source_metadata = pv.VideoMetadata(source, 12.3, 30.0, 640, 480, "h264", 1234, [])
    normalized_metadata = pv.VideoMetadata(normalized, 12.3, 30.0, 640, 480, "h264", 4321, [])
    segments = [pv.SegmentWindow("segment_0001", 1, 0.0, 12.3, "time", segment_path)]
    frames = [
        pv.FrameRecord(
            id="segment_0001_frame_000001",
            segment_id="segment_0001",
            path=frame_path,
            timestamp_sec=0.0,
            frame_index=1,
            selected=True,
            blur_score=100.0,
            overexposed_ratio=0.0,
            underexposed_ratio=0.0,
            duplicate_score=None,
            reject_reasons=[],
        )
    ]

    manifest = pv.build_manifest(
        video_id="sample",
        source_metadata=source_metadata,
        normalized_metadata=normalized_metadata,
        settings={"preset": "baseline"},
        segments=segments,
        frames=frames,
        repo_root=ROOT,
    )
    destination = tmp_path / "preprocess_manifest.json"
    pv.write_manifest(manifest, destination)

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "1.0"
    assert loaded["source"]["path"] == "data/raw_videos/sample.mp4"
    assert loaded["segments"][0]["path"] == "data/segments/sample/segment_0001.mp4"
    assert loaded["frames"][0]["path"] == frame_path
    assert loaded["summary"]["selected_frames"] == 1


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for preprocess smoke tests",
)
def test_preprocess_smoke_with_synthetic_video(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.mp4"
    output_root = tmp_path / "data"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10:duration=2",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = pv.main(
        [
            str(source),
            "--video-id",
            "synthetic",
            "--output-root",
            str(output_root),
            "--segment-method",
            "time",
            "--segment-length-sec",
            "1.0",
            "--segment-overlap-sec",
            "0.25",
            "--target-fps",
            "2",
            "--blur-threshold",
            "0",
            "--force",
        ]
    )

    manifest_path = output_root / "manifests" / "synthetic" / "preprocess_manifest.json"
    assert result == 0
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["video_id"] == "synthetic"
    assert manifest["segments"]
    assert manifest["frames"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for preprocess smoke tests",
)
def test_preprocess_smoke_with_source_png_frames(tmp_path: Path) -> None:
    source = tmp_path / "synthetic.mp4"
    output_root = tmp_path / "data"
    config_path = tmp_path / "source_png.json"
    config_path.write_text(
        json.dumps(
            {
                "segment_method": "time",
                "segment_length_sec": 1.0,
                "segment_overlap_sec": 0.25,
                "target_fps": 2.0,
                "blur_threshold": 0.0,
                "frame_source": "source",
                "frame_format": "png",
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=10:duration=2",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = pv.main(
        [
            str(source),
            "--video-id",
            "synthetic_png",
            "--output-root",
            str(output_root),
            "--config",
            str(config_path),
            "--force",
        ]
    )

    manifest_path = output_root / "manifests" / "synthetic_png" / "preprocess_manifest.json"
    assert result == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["settings"]["frame_format"] == "png"
    assert manifest["settings"]["frame_source"] == "source"
    assert manifest["frames"]
    selected_paths = [frame["path"] for frame in manifest["frames"] if frame["selected"]]
    assert selected_paths
    assert all(path.endswith(".png") for path in selected_paths)
    for relative_path in selected_paths:
        output_path = Path(relative_path)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        assert output_path.exists()
