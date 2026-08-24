from __future__ import annotations

from scripts.longsplat.automated_policy import evaluate_early_gate, evaluate_formal_gate


def _training(*, count: int, iterations: int) -> dict[str, object]:
    low, high = divmod(iterations, count)
    exposures = {f"camera-{index}.png": low + (index < high) for index in range(count)}
    unique = sum(value > 0 for value in exposures.values())
    return {
        "computed_pass": True,
        "executor_result": {
            "structural": {
                "structural_pass": True,
                "active_camera_count": count,
                "camera_sampling_telemetry": {
                    "iterations": iterations,
                    "active_camera_count": count,
                    "unique_camera_count": unique,
                    "exposure_counts": exposures,
                },
                "anchor_schedule": {"observed_from_runtime": True, "observed_events": [100, 200]},
                "checkpoint": {"finite": True},
            }
        },
    }


def _postcheck(*, cross_gap: bool = False, psnr: float = 18.0) -> dict[str, object]:
    evidence = {
        "computed_pass": True,
        "structural": {
            "render_exit_code": 0,
            "fixed_render_count_exact": True,
            "training_gt_count_exact": True,
            "nvs_count_from_actual_pose_contract": True,
            "all_pngs_decoded_finite_and_dimensions_exact": True,
            "camera_order_exact": True,
        },
        "counts": {"fixed_render": 2, "training_gt": 2, "nvs_on_path": 3},
        "rough_visual": {"automatic_health": "needs_review"},
        "fixed_view_metrics": {
            "mean_psnr_db": psnr,
            "min_psnr_db": psnr,
            "mean_ssim": 0.70,
            "min_ssim": 0.55,
        },
        "contact_sheets": {
            "fixed_vs_gt": {"path": "/tmp/fixed.png", "sha256": "fixed"},
            "on_path": {"path": "/tmp/path.png", "sha256": "path"},
        },
        "metrics": {"path": "/tmp/metrics.json", "sha256": "metrics"},
        "png_hashes": {"path": "/tmp/png-hashes.json", "sha256": "hashes"},
        "cross_gap": {"cross_gap": cross_gap},
    }
    return {"postcheck_result": evidence, "postcheck_result_path": "/tmp/postcheck.json"}


def test_early_policy_records_short_coverage_as_advisory() -> None:
    decision = evaluate_early_gate(training=_training(count=1200, iterations=1000), postcheck=_postcheck())
    assert decision["computed_pass"] is True
    assert decision["manual_visual_review"] is False
    assert decision["runtime"]["zero_exposure_camera_count"] == 200
    assert any("zero exposure" in warning for warning in decision["warnings"])


def test_early_policy_records_cross_gap_and_threshold_warning_without_blocking() -> None:
    cross_gap = evaluate_early_gate(training=_training(count=2, iterations=1000), postcheck=_postcheck(cross_gap=True))
    assert cross_gap["computed_pass"] is True
    assert cross_gap["cross_gap"]["warning"] is True
    assert any("cross-gap" in warning for warning in cross_gap["warnings"])
    weak = evaluate_early_gate(training=_training(count=2, iterations=1000), postcheck=_postcheck(psnr=12.0))
    assert weak["computed_pass"] is True
    assert any("thresholds" in warning for warning in weak["warnings"])


def test_formal_policy_records_cross_gap_without_human_acceptance() -> None:
    postcheck = _postcheck(cross_gap=True, psnr=22.0)
    postcheck["postcheck_result"]["fixed_view_metrics"]["mean_ssim"] = 0.80
    postcheck["postcheck_result"]["fixed_view_metrics"]["min_ssim"] = 0.80
    decision = evaluate_formal_gate(training=_training(count=2, iterations=30000), postcheck=postcheck)
    assert decision["computed_pass"] is True
    assert decision["cross_gap"]["present"] is True
    assert decision["manual_visual_review"] is False
