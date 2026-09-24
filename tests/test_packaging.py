from __future__ import annotations

import tomllib
from pathlib import Path


def test_mcp_legacy_and_v2_packaging_lines_are_separate() -> None:
    document = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    project = document["project"]
    extras = project["optional-dependencies"]
    assert extras["mcp-legacy"] == ["mcp>=1.26,<2"]
    assert extras["mcp-v2"] == ["mcp==2.0.0"]
    assert all(not dependency.startswith("mcp") for dependency in project["dependencies"])
    assert project["scripts"]["healthcare-mcp-v2"] == "healthcare.mcp_v2_entrypoint:main"
