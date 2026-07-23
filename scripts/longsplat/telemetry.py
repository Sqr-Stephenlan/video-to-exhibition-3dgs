"""Parse structured LongSplat telemetry emitted on subprocess stdout."""

from __future__ import annotations

import json
import math
import re
from numbers import Real
from typing import Any


_SAFE_STATE_TIMESTAMP_SUFFIX = re.compile(
    r" \[(?:0[1-9]|[12][0-9]|3[01])/(?:0[1-9]|1[0-2]) "
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\]$"
)


def _reject_nonfinite_constant(value: str) -> None:
    """Reject JavaScript-style NaN/Infinity accepted by ``json.loads``."""
    raise ValueError(f"non-finite JSON constant: {value}")


def parse_json_markers(stdout: str, marker: str) -> list[dict[str, Any]]:
    """Return ordered JSON objects from lines prefixed by ``marker``.

    Malformed JSON, non-object values, and JSON containing non-finite numeric
    constants are ignored so a diagnostic line cannot fail the pipeline.
    """
    prefix = f"{marker} "
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith(prefix):
            continue
        payload = _SAFE_STATE_TIMESTAMP_SUFFIX.sub("", line[len(prefix) :])
        try:
            value = json.loads(
                payload,
                parse_constant=_reject_nonfinite_constant,
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _finite_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _finite_number(record.get(key))
        if value is not None:
            values.append(value)
    return values


def summarize_pose_telemetry(stdout: str) -> dict[str, Any]:
    """Summarize incremental PnP attempts while retaining per-frame records."""
    records = parse_json_markers(stdout, "POSE_TELEMETRY")
    successes = sum(record.get("success") is True for record in records)
    inlier_ratios = _finite_values(records, "inlier_ratio")
    reprojection_errors = _finite_values(records, "reprojection_rmse_px")
    return {
        "pnp_attempts": len(records),
        "pnp_successes": successes,
        "pnp_failures": len(records) - successes,
        "min_inlier_ratio": min(inlier_ratios) if inlier_ratios else None,
        "max_reprojection_rmse_px": (
            max(reprojection_errors) if reprojection_errors else None
        ),
        "records": records,
    }


def summarize_vda_telemetry(stdout: str) -> dict[str, Any]:
    """Summarize VDA alignment outcomes and quality metrics."""
    records = parse_json_markers(stdout, "VDA_TELEMETRY")
    correlations = _finite_values(records, "correlation")
    inlier_ratios = _finite_values(records, "inlier_ratio")
    return {
        "aligned": sum(record.get("result") == "aligned" for record in records),
        "missing": sum(record.get("result") == "missing" for record in records),
        "rejected": sum(record.get("result") == "rejected" for record in records),
        "min_correlation": min(correlations) if correlations else None,
        "mean_inlier_ratio": (
            sum(inlier_ratios) / len(inlier_ratios) if inlier_ratios else None
        ),
        "records": records,
    }


def summarize_conversion_telemetry(stdout: str) -> dict[str, Any]:
    """Report whether every emitted conversion record has finite scales."""
    records = parse_json_markers(stdout, "CONVERSION_TELEMETRY")

    def _all_record_scales_finite(record: dict[str, Any]) -> bool:
        point_count = _finite_number(record.get("point_count"))
        finite_count = _finite_number(record.get("finite_scale_count"))
        return (
            point_count is not None
            and finite_count is not None
            and point_count >= 0
            and finite_count == point_count
        )

    return {
        "all_scales_finite": bool(records)
        and all(_all_record_scales_finite(record) for record in records),
        "records": records,
    }
