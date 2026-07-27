"""Contract tests for dependency-free LongSplat stdout telemetry parsing."""

from __future__ import annotations

from scripts.longsplat.telemetry import (
    parse_json_markers,
    summarize_conversion_telemetry,
    summarize_pose_telemetry,
    summarize_vda_telemetry,
)


def test_summarize_pose_telemetry_keeps_frame_metrics():
    stdout = (
        'POSE_TELEMETRY {"frame":"f4","inlier_count":80,'
        '"inlier_ratio":0.8,"match_count":100,"method":"pnp_ransac",'
        '"reprojection_rmse_px":1.25,"stage":"incremental",'
        '"success":true}\n'
    )

    summary = summarize_pose_telemetry(stdout)

    assert summary["attempt_count"] == 1
    assert summary["accepted_camera_count"] == 1
    assert summary["rejected_attempt_count"] == 0
    assert summary["accepted_frames"] == ["f4"]
    assert summary["min_inlier_ratio"] == 0.8
    assert summary["max_reprojection_rmse_px"] == 1.25
    assert summary["records"][0]["frame"] == "f4"


def test_summarize_vda_telemetry_reports_alignment_quality():
    stdout = (
        'VDA_TELEMETRY {"correlation":0.91,"frame":"f4",'
        '"inlier_ratio":0.88,"result":"aligned","stage":"incremental"}\n'
        'VDA_TELEMETRY {"frame":"f5","reason":"missing_file",'
        '"result":"missing","stage":"incremental"}\n'
        'VDA_TELEMETRY {"correlation":0.1,"frame":"f6",'
        '"result":"rejected","stage":"incremental"}\n'
    )

    summary = summarize_vda_telemetry(stdout)

    assert summary["aligned"] == 1
    assert summary["missing"] == 1
    assert summary["rejected"] == 1
    assert summary["min_correlation"] == 0.1
    assert summary["mean_inlier_ratio"] == 0.88


def test_summarize_conversion_telemetry_reports_floored_distances():
    stdout = (
        'CONVERSION_TELEMETRY {"finite_scale_count":40,'
        '"nonpositive_distance_count":12,"point_count":40}\n'
    )

    summary = summarize_conversion_telemetry(stdout)

    assert summary["records"][0]["nonpositive_distance_count"] == 12
    assert summary["all_scales_finite"] is True


def test_marker_parser_ignores_malformed_or_non_object_json():
    stdout = (
        'POSE_TELEMETRY {"success":true}\n'
        "POSE_TELEMETRY not-json\n"
        "POSE_TELEMETRY [1, 2, 3]\n"
        'POSE_TELEMETRY {"inlier_ratio":NaN}\n'
        'unrelated {"success":false}\n'
    )

    assert parse_json_markers(stdout, "POSE_TELEMETRY") == [{"success": True}]


def test_marker_parser_accepts_longsplat_safe_state_timestamp_suffix():
    stdout = (
        'POSE_TELEMETRY {"frame":"f4","success":true} [22/07 16:02:03]\n'
        'VDA_TELEMETRY {"frame":"f4","result":"aligned"} '
        "[22/07 16:04:05]\n"
    )

    assert parse_json_markers(stdout, "POSE_TELEMETRY") == [
        {"frame": "f4", "success": True}
    ]
    assert parse_json_markers(stdout, "VDA_TELEMETRY") == [
        {"frame": "f4", "result": "aligned"}
    ]


def test_marker_parser_still_rejects_unknown_trailing_text():
    stdout = 'POSE_TELEMETRY {"success":true} arbitrary-tail\n'

    assert parse_json_markers(stdout, "POSE_TELEMETRY") == []


def test_numeric_summaries_ignore_booleans_strings_and_missing_values():
    stdout = (
        'POSE_TELEMETRY {"inlier_ratio":true,"reprojection_rmse_px":"bad",'
        '"success":false}\n'
        'POSE_TELEMETRY {"inlier_ratio":null,"success":true}\n'
    )

    summary = summarize_pose_telemetry(stdout)

    assert summary["attempt_count"] == 2
    assert summary["accepted_camera_count"] == 0
    assert summary["rejected_attempt_count"] == 1
    assert summary["accepted_frames"] == []  # no records have a string frame field
    assert summary["min_inlier_ratio"] is None
    assert summary["max_reprojection_rmse_px"] is None


def test_conversion_without_valid_marker_is_not_reported_finite():
    assert summarize_conversion_telemetry("")["all_scales_finite"] is False
