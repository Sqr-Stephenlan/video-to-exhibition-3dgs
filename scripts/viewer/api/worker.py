from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import BackendIdentity
from .preflight import verify_gpu_backend


def run_reconstruction_job(
    *,
    job_id: str,
    input_video: str | Path,
    runtime_root: str | Path,
    route_root: str | Path,
    publish_root: str | Path,
    delivery_name: str,
    source_metadata: Mapping[str, Any],
    backend_identity: BackendIdentity,
) -> dict[str, Any]:
    """Run the existing canonical pipeline in a worker process."""

    verify_gpu_backend(route_root=route_root)

    from scripts.longsplat.reconstruct_pipeline import run_reconstruction

    return run_reconstruction(
        input_video=Path(input_video),
        output_root=Path(runtime_root) / "longsplat-runs",
        run_id=job_id,
        stop_after="automated-technical-delivery",
        plan=False,
        route_root=Path(route_root),
        depth_source="disabled",
        execute_gpu=True,
        pipeline_profile="single-convergence1000-v1",
        acceptance_policy="automated-technical-v1",
        publish_dir=Path(publish_root),
        delivery_name=delivery_name,
        source_metadata=dict(source_metadata),
    )
