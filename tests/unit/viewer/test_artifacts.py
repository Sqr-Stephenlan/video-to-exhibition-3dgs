from __future__ import annotations

import pytest

from scripts.viewer.api.artifacts import ArtifactCatalog, ArtifactSecurityError


def test_artifact_catalog_rejects_a_file_outside_allowed_roots(tmp_path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside.ply"
    allowed.mkdir()
    outside.write_bytes(b"ply")
    catalog = ArtifactCatalog(job_id="job-1", allowed_roots=[allowed])

    with pytest.raises(ArtifactSecurityError):
        catalog.register(
            artifact_id="published-ply",
            kind="model",
            format="ply",
            path=outside,
        )


def test_artifact_catalog_revalidates_sha_before_download(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    model = allowed / "model.ply"
    model.write_bytes(b"ply")
    catalog = ArtifactCatalog(job_id="job-1", allowed_roots=[allowed])
    descriptor = catalog.register(
        artifact_id="published-ply",
        kind="model",
        format="ply",
        path=model,
        vertices=1,
    )

    assert descriptor.download_url == "/api/v1/jobs/job-1/artifacts/published-ply"
    model.write_bytes(b"changed")

    with pytest.raises(ArtifactSecurityError):
        catalog.resolve("published-ply")


def test_artifact_catalog_validates_published_gaussian_ply(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    model = allowed / "model.ply"
    model.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float opacity\nproperty float scale_0\n"
        "property float rot_0\nend_header\n0 0 0 0 0 0 0\n",
        encoding="ascii",
    )
    catalog = ArtifactCatalog(job_id="job-1", allowed_roots=[allowed])

    descriptor = catalog.register(
        artifact_id="published-ply",
        kind="model",
        format="ply",
        path=model,
        validate_gaussian=True,
    )

    assert descriptor.vertices == 1
    assert catalog.resolve("published-ply") == model.resolve()
