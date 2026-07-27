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
        "duplicate_time_window_sec": 2.0,
        "max_selected_gap_sec": 2.0,
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


def test_write_segment_uses_quiet_ffmpeg_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(command: list[str], description: str) -> None:
        captured["command"] = command
        captured["description"] = description

    monkeypatch.setattr(pv, "require_tool", lambda _name: "ffmpeg")
    monkeypatch.setattr(pv, "run_command", fake_run_command)
    segment = pv.SegmentWindow(id="segment_0001", index=1, start_sec=0.0, end_sec=5.0, reason="time")
    destination = tmp_path / "segment_0001.mp4"

    result = pv.write_segment(Path("normalized.mp4"), segment, destination)

    assert captured["command"][:5] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    assert captured["description"] == "Writing segment_0001"
    assert result.path == destination


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
            width=640,
            height=480,
            sha256="a" * 64,
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
    destination = tmp_path / "frames_manifest.json"
    pv.write_manifest(manifest, destination)

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "2.0"
    assert loaded["source"]["path"] == "data/raw_videos/sample.mp4"
    assert loaded["segments"][0]["path"] == "data/segments/sample/segment_0001.mp4"
    assert loaded["frames"][0]["path"] == frame_path
    assert loaded["frames"][0]["width"] == 640
    assert loaded["frames"][0]["height"] == 480
    assert loaded["summary"]["selected_frames"] == 1
    assert loaded["summary"]["coverage_by_segment"]["segment_0001"]["max_selected_gap_sec"] == 0.0


def test_relative_path_rejects_paths_outside_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()

    with pytest.raises(pv.PreprocessError, match="outside the repository"):
        pv.relative_path(tmp_path / "external.mp4", repository)


def test_manifest_command_normalizes_paths_when_options_precede_source(tmp_path: Path) -> None:
    repository = tmp_path
    source = repository / "data" / "raw_videos" / "sample.mp4"
    config = repository / "configs" / "preprocess" / "sample.json"
    arguments = ["--config", str(config), "--video-id", "sample", str(source)]

    command = pv.manifest_command_args(arguments, repository, source)

    assert command[-5:] == [
        "--config",
        "configs/preprocess/sample.json",
        "--video-id",
        "sample",
        "data/raw_videos/sample.mp4",
    ]


def test_force_prepares_staging_without_deleting_existing_outputs(tmp_path: Path) -> None:
    output_root = tmp_path / "data"
    existing = output_root / "segments" / "sample"
    existing.mkdir(parents=True)
    sentinel = existing / "keep.txt"
    sentinel.write_text("old", encoding="utf-8")

    *_, staging_root = pv.prepare_output_dirs(output_root, "sample", force=True)

    assert sentinel.read_text(encoding="utf-8") == "old"
    shutil.rmtree(staging_root)


def test_probe_failure_keeps_previous_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("data/raw_videos/source.mp4")
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    sentinel = Path("data/manifests/sample/frames_manifest.json")
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("old", encoding="utf-8")
    args = pv.build_arg_parser().parse_args(
        [str(source), "--video-id", "sample", "--force", "--segment-method", "time"]
    )
    monkeypatch.setattr(pv, "require_tool", lambda name: name)
    monkeypatch.setattr(
        pv,
        "probe_video",
        lambda _source: (_ for _ in ()).throw(pv.PreprocessError("probe failed")),
    )

    with pytest.raises(pv.PreprocessError, match="probe failed"):
        pv.preprocess(args)

    assert sentinel.read_text(encoding="utf-8") == "old"
    assert not Path("data/.preprocess-staging").exists()


def test_sample_segment_frames_releases_capture_when_frame_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def get(self, _property_id: int) -> float:
            return 1.0

        def set(self, _property_id: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.full((8, 8, 3), 127, dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()

    def fail_write(_path: Path, _frame: np.ndarray, _frame_format: str = "jpg") -> None:
        raise RuntimeError("write failed")

    monkeypatch.setattr(pv.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(pv, "write_image", fail_write)
    segment = pv.SegmentWindow(
        id="segment_0001",
        index=1,
        start_sec=0.0,
        end_sec=1.0,
        reason="time",
        path=tmp_path / "segment_0001.mp4",
    )

    with pytest.raises(RuntimeError, match="write failed"):
        pv.sample_segment_frames(
            segment=segment,
            selected_root=tmp_path / "selected",
            rejected_root=tmp_path / "rejected",
            target_fps=1.0,
            blur_threshold=0.0,
            overexposed_ratio=1.0,
            underexposed_ratio=1.0,
            duplicate_hash_threshold=4,
            save_rejected=False,
            repo_root=tmp_path,
        )

    assert capture.released is True


def test_duplicate_filter_is_time_bounded_and_preserves_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, _property_id: int) -> float:
            return 1.0

        def set(self, _property_id: int, _value: float) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index >= 3:
                return False, None
            self.index += 1
            return True, np.full((8, 8, 3), 127, dtype=np.uint8)

        def release(self) -> None:
            pass

    monkeypatch.setattr(pv.cv2, "VideoCapture", lambda _path: FakeCapture())
    segment = pv.SegmentWindow(
        id="segment_0001",
        index=1,
        start_sec=0.0,
        end_sec=3.0,
        reason="time",
        path=tmp_path / "segment_0001.mp4",
    )

    records = pv.sample_segment_frames(
        segment=segment,
        selected_root=tmp_path / "selected",
        rejected_root=tmp_path / "rejected",
        target_fps=1.0,
        blur_threshold=0.0,
        overexposed_ratio=1.0,
        underexposed_ratio=1.0,
        duplicate_hash_threshold=4,
        duplicate_time_window_sec=2.0,
        max_selected_gap_sec=2.0,
        save_rejected=False,
        repo_root=tmp_path,
    )

    assert [record.selected for record in records] == [True, False, True]
    assert records[1].matched_frame_id == records[0].id
    assert records[1].duplicate_score == 0
    assert records[0].width == 8
    assert records[0].height == 8
    assert records[0].sha256 is not None
    assert records[1].motion_score == 0.0


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for preprocess smoke tests",
)
def test_preprocess_smoke_with_synthetic_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("synthetic.mp4")
    output_root = Path("data")
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

    manifest_path = output_root / "manifests" / "synthetic" / "frames_manifest.json"
    assert result == 0
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["video_id"] == "synthetic"
    assert manifest["segments"]
    assert manifest["frames"]
    assert all(frame["width"] == 160 and frame["height"] == 120 for frame in manifest["frames"])
    assert manifest["source"]["sha256"]
    assert manifest["run"]["settings_fingerprint"]
    assert manifest["run"]["command"][:3] == ["./dev.sh", "python", "scripts/preprocess_video.py"]


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for preprocess smoke tests",
)
def test_preprocess_smoke_with_source_png_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("synthetic.mp4")
    output_root = Path("data")
    config_path = Path("source_png.json")
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

    manifest_path = output_root / "manifests" / "synthetic_png" / "frames_manifest.json"
    assert result == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["settings"]["frame_format"] == "png"
    assert manifest["settings"]["frame_source"] == "source"
    assert manifest["frames"]
    selected_paths = [frame["path"] for frame in manifest["frames"] if frame["selected"]]
    assert selected_paths
    assert all(path.endswith(".png") for path in selected_paths)
    for relative_path in selected_paths:
        output_path = tmp_path / relative_path
        assert output_path.exists()


def test_preprocess_rejects_vfr_source_frame_sampling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    args = pv.build_arg_parser().parse_args(
        [str(source), "--video-id", "sample", "--output-root", str(tmp_path / "data"), "--frame-source", "source"]
    )
    metadata = pv.VideoMetadata(source, 1.0, 30.0, 16, 16, "h264", 5, [], "a" * 64, 0, True)
    monkeypatch.setattr(pv, "ensure_repo_path", lambda path, _root, _label: path.resolve())
    monkeypatch.setattr(pv, "require_tool", lambda name: name)
    monkeypatch.setattr(pv, "probe_video", lambda _source: metadata)

    with pytest.raises(pv.PreprocessError, match="variable-frame-rate"):
        pv.preprocess(args)


@pytest.mark.parametrize(
    ("rotation", "expected_shape"),
    [(90, (3, 2, 3)), (180, (2, 3, 3)), (270, (3, 2, 3))],
)
def test_rotate_frame_applies_metadata_orientation(rotation: int, expected_shape: tuple[int, int, int]) -> None:
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    assert pv.rotate_frame(frame, rotation).shape == expected_shape


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for compatibility fixtures",
)
def test_probe_video_detects_vfr_fixture(tmp_path: Path) -> None:
    destination = tmp_path / "vfr.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=5:duration=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-fps_mode",
            "vfr",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert pv.probe_video(destination).is_vfr is True


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for compatibility fixtures",
)
def test_probe_and_normalize_rotation_metadata(tmp_path: Path) -> None:
    base = tmp_path / "base.mp4"
    rotated = tmp_path / "rotated.mp4"
    normalized = tmp_path / "normalized.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=5:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(base),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-display_rotation", "90", "-i", str(base), "-c", "copy", str(rotated)],
        check=True,
        capture_output=True,
        text=True,
    )

    metadata = pv.probe_video(rotated)
    normalized_metadata = pv.normalize_video(rotated, normalized, metadata, max_long_edge=64)

    assert metadata.rotation_degrees == 90
    assert (normalized_metadata.width, normalized_metadata.height) == (48, 64)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for compatibility fixtures",
)
def test_scene_detection_finds_real_cut_fixture(tmp_path: Path) -> None:
    destination = tmp_path / "scene_cut.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:size=64x48:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:size=64x48:rate=10:duration=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    windows = pv.detect_scene_windows(destination, min_segment_sec=0.1)

    assert len(windows) == 2
    assert windows[0][1] == pytest.approx(windows[1][0], abs=0.2)
