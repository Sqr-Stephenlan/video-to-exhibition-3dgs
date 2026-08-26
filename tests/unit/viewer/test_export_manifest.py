from __future__ import annotations

import json

import pytest

from scripts.viewer.export import (
    ModelExportError,
    build_supersplat_editor_link,
    export_web_model,
)


def test_export_web_model_binds_source_and_target_identity(tmp_path):
    source = tmp_path / "technical.ply"
    source.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float opacity\nproperty float scale_0\n"
        "property float rot_0\nend_header\n"
        "0 0 0 0 0 0 0\n1 1 1 0 0 0 0\n",
        encoding="utf-8",
    )
    destination = tmp_path / "web-model"

    manifest = export_web_model(
        source_ply=source,
        destination_root=destination,
        target_format="ply",
        coordinate_system={
            "name": "model-local",
            "version": "viewer-v1",
            "up_axis": "y",
            "units": "meters",
        },
    )

    assert manifest["schema_version"] == "viewer-model-v1"
    assert manifest["source"]["sha256"] == manifest["target"]["sha256"]
    assert manifest["source"]["vertices"] == 2
    assert manifest["target"]["format"] == "ply"
    assert manifest["format_compatible"] is True
    assert manifest["runtime_verified"] is False
    assert (destination / "technical.ply").is_file()
    json.dumps(manifest)


def test_export_web_model_rejects_a_non_gaussian_ply(tmp_path):
    source = tmp_path / "ordinary.ply"
    source.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ModelExportError, match="Gaussian"):
        export_web_model(
            source_ply=source,
            destination_root=tmp_path / "web-model",
            target_format="ply",
            coordinate_system={"name": "model-local", "version": "viewer-v1"},
        )


def test_export_web_model_reports_unsupported_converter_explicitly(tmp_path):
    source = tmp_path / "technical.ply"
    source.write_bytes(b"ply")

    with pytest.raises(ModelExportError, match="converter"):
        export_web_model(
            source_ply=source,
            destination_root=tmp_path / "web-model",
            target_format="sog",
            coordinate_system={"name": "model-local", "version": "viewer-v1"},
        )


def test_supersplat_editor_link_uses_load_query_parameter():
    link = build_supersplat_editor_link(
        "https://example.test/models/job-1.ply",
        viewer_base="https://superspl.at/editor",
    )

    assert link == (
        "https://superspl.at/editor"
        "?load=https%3A%2F%2Fexample.test%2Fmodels%2Fjob-1.ply"
    )


def test_supersplat_editor_link_rejects_local_path():
    with pytest.raises(ValueError):
        build_supersplat_editor_link("/api/v1/jobs/job-1/artifacts/published-ply")
