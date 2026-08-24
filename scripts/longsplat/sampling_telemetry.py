"""Dependency-light validation for emitted camera sampling telemetry.

The validator is shared by the default convergence profile and the explicit
legacy coverage profile.  Keeping it separate prevents the default import
graph from loading the coverage planner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pipeline_contract import PipelineBlocked, sha256_file


TELEMETRY_SCHEMA = "camera-sampling-telemetry-v1"
TELEMETRY_EVENTS_NAME = "camera_sampling_telemetry-v1.jsonl"
TELEMETRY_SUMMARY_NAME = "camera_sampling_telemetry-v1.json"


class SamplingTelemetryBlocked(PipelineBlocked):
    """Sampling evidence is absent, malformed, or inconsistent."""


def _fail(message: str) -> None:
    raise SamplingTelemetryBlocked(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is missing or symlinked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"{label} is not valid JSON: {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must be an object: {path}")
    return value


def validate_sampling_telemetry(
    *,
    model_path: str | Path,
    expected_iterations: int,
    expected_internal_names: Sequence[str],
    expected_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate emitted sampling events against the immutable pose contract."""

    model = Path(model_path).resolve()
    if model.is_symlink() or not model.is_dir():
        _fail(f"telemetry model path is missing or symlinked: {model}")
    summary_path = model / TELEMETRY_SUMMARY_NAME
    events_path = model / TELEMETRY_EVENTS_NAME
    if (
        not summary_path.is_file()
        or summary_path.is_symlink()
        or not events_path.is_file()
        or events_path.is_symlink()
    ):
        _fail("camera sampling telemetry summary/events are required")
    summary = _load_json(summary_path, "camera sampling telemetry summary")
    if summary.get("schema_version") != TELEMETRY_SCHEMA:
        _fail("camera sampling telemetry schema mismatch")
    if summary.get("selection_policy") != "one_camera_per_iteration_random_pop_without_replacement_per_full_stack":
        _fail("camera sampling telemetry selection policy is not the audited stack policy")
    internal_names = [str(value) for value in expected_internal_names]
    if len(internal_names) != len(set(internal_names)) or not internal_names:
        _fail("expected internal camera names must be unique")
    if summary.get("active_camera_internal_order") != internal_names:
        _fail("telemetry internal camera order differs from immutable pose contract")
    if (
        summary.get("iterations") != expected_iterations
        or summary.get("sampled_iteration_count") != expected_iterations
    ):
        _fail("telemetry iteration count differs from checkpoint contract")
    if (
        expected_contract_sha256 is not None
        and summary.get("camera_contract_file_sha256") != expected_contract_sha256
    ):
        _fail("telemetry camera contract SHA differs from immutable input")
    records = summary.get("exposure_counts")
    if not isinstance(records, list) or len(records) != len(internal_names):
        _fail("telemetry exposure_counts are incomplete")
    counts: dict[str, int] = {}
    for index, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or record.get("camera_internal_name") != internal_names[index]
            or record.get("camera_contract_index") != index
        ):
            _fail("telemetry exposure record order/identity differs from pose contract")
        count = record.get("exposure_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _fail("telemetry exposure count is invalid")
        counts[internal_names[index]] = count
    if sum(counts.values()) != expected_iterations:
        _fail("telemetry exposure counts do not sum to iterations")
    raw_events = events_path.read_bytes()
    if (
        summary.get("events_sha256") != hashlib.sha256(raw_events).hexdigest()
        or summary.get("events_size_bytes") != len(raw_events)
    ):
        _fail("telemetry event file identity differs from summary")
    observed: dict[str, int] = {name: 0 for name in internal_names}
    lines = raw_events.decode("utf-8").splitlines()
    if len(lines) != expected_iterations:
        _fail("telemetry event line count differs from iterations")
    for expected_iteration, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"telemetry event {expected_iteration} is invalid: {exc}")
        if (
            not isinstance(event, Mapping)
            or event.get("schema_version") != TELEMETRY_SCHEMA
            or event.get("iteration") != expected_iteration
        ):
            _fail("telemetry event sequence/schema is invalid")
        name = event.get("camera_internal_name")
        index = event.get("camera_contract_index")
        if not isinstance(name, str) or name not in observed or index != internal_names.index(name):
            _fail("telemetry event camera identity is outside the exact contract")
        observed[name] += 1
        if (
            event.get("exposure_count") != observed[name]
            or event.get("cumulative_unique_count") != sum(value > 0 for value in observed.values())
        ):
            _fail("telemetry event cumulative exposure is inconsistent")
    if observed != counts:
        _fail("telemetry event exposures differ from summary exposures")
    return {
        "schema_version": TELEMETRY_SCHEMA,
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "summary_size_bytes": summary_path.stat().st_size,
        "events_path": str(events_path),
        "events_sha256": hashlib.sha256(raw_events).hexdigest(),
        "events_size_bytes": len(raw_events),
        "active_camera_count": len(internal_names),
        "iterations": expected_iterations,
        "unique_camera_count": sum(value > 0 for value in observed.values()),
        "coverage_fraction": sum(value > 0 for value in observed.values()) / len(observed),
        "exposure_counts": counts,
        "bit_exact_resume": False,
    }
