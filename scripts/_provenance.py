"""
Stage provenance: fingerprinting, resume validation, and atomic publish.

Used by every pipeline script to guarantee that re-runs either cleanly
overwrite or safely resume without mixing old and new artifacts.

Relies on ``_common`` for low-level primitives (SHA-256, JSON I/O, etc.).
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from _common import dict_fingerprint as params_fingerprint
from _common import file_sha256, write_json_atomic

# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def dir_fingerprint(dir_path: Path, pattern: str = "*") -> dict[str, str]:
    """Return ``{filename: sha256}`` for files matching *pattern* in a directory."""
    result: dict[str, str] = {}
    for fp in sorted(dir_path.glob(pattern)):
        if fp.is_file():
            result[fp.name] = file_sha256(fp)
    return result


# ---------------------------------------------------------------------------
# Provenance records
# ---------------------------------------------------------------------------


def build_provenance(
    stage: str,
    inputs: dict[str, str],
    params: dict[str, Any],
    tool_versions: dict[str, str],
    outputs: list[str],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "inputs": inputs,
        "params": params,
        "params_fingerprint": params_fingerprint(params),
        "tool_versions": tool_versions,
        "outputs": outputs,
    }


def write_provenance(
    prov: dict[str, Any], output_dir: Path, filename: str = "_provenance.json"
) -> Path:
    """Atomically write a provenance record (delegates to _common)."""
    write_json_atomic(prov, output_dir / filename)
    return output_dir / filename


def read_provenance(output_dir: Path, filename: str = "_provenance.json") -> dict[str, Any]:
    """Read an existing provenance record, or return {} if missing."""
    path = output_dir / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Resume validation
# ---------------------------------------------------------------------------


def check_resume(
    output_dir: Path,
    expected_input_fingerprints: dict[str, str],
    expected_params: dict[str, Any],
    provenance_filename: str = "_provenance.json",
) -> bool:
    """Return True if a previous run can be safely resumed.

    Requires that:
    - The completion marker exists.
    - Input fingerprints match.
    - Parameter fingerprints match.
    """
    comp_marker = output_dir / "_COMPLETE"
    if not comp_marker.exists():
        return False

    prov = read_provenance(output_dir, provenance_filename)
    if not prov:
        return False

    prev_inputs = prov.get("inputs", {})
    if prev_inputs != expected_input_fingerprints:
        return False

    expected_hash = params_fingerprint(expected_params)
    return prov.get("params_fingerprint") == expected_hash


# ---------------------------------------------------------------------------
# Atomic output publish
# ---------------------------------------------------------------------------


def publish_outputs(
    temp_dir: Path,
    final_dir: Path,
    completion_marker: str = "_COMPLETE",
) -> None:
    """Move all files from *temp_dir* into *final_dir*, then write a completion marker."""
    final_dir.mkdir(parents=True, exist_ok=True)
    for item in temp_dir.iterdir():
        dest = final_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))
    # Write completion marker
    (final_dir / completion_marker).write_text(datetime.now(UTC).isoformat())
