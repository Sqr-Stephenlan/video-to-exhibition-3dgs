"""Run record for LongSplat training sessions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LONGSPLAT_BACKEND = {
    "repo_url": "https://github.com/NVlabs/LongSplat",
    "commit": "19750775a9d19f30aa05a8333c4c6c231b2d5f4a",
}


def create(
    *,
    source_path: str | Path,
    model_path: str | Path,
    images: str,
    mode: str,
    resolution: int,
    depth_source: str | None,
    extra_train_args: dict[str, Any] | None,
) -> dict[str, Any]:
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "status": "running",
        "created_at": ts,
        "updated_at": ts,
        "backend": _LONGSPLAT_BACKEND,
        "config": {
            "source_path": str(Path(source_path).resolve()),
            "model_path": str(Path(model_path).resolve()),
            "images": images,
            "resolution": resolution,
            "mode": mode,
            "depth_source": depth_source or "mast3r",
            "extra_train_args": extra_train_args or {},
        },
        "stages": {},
    }


def save(record: dict[str, Any], model_path: Path) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    path = model_path / "reconstruction_run.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def mark_complete(
    record: dict[str, Any],
    model_path: Path,
    *,
    stage: str,
    exit_code: int,
    duration_s: float,
    artifacts: list[dict[str, Any]] | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    record["updated_at"] = now
    record["stages"][stage] = {
        "status": "completed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "duration_s": duration_s,
    }
    if artifacts:
        record.setdefault("artifacts", []).extend(artifacts)
    record["status"] = (
        "complete"
        if all(s.get("status") == "completed" for s in record["stages"].values())
        else "failed"
    )
    save(record, model_path)
