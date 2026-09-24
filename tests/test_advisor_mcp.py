"""stdio MCP server + MCP Apps workbench for 个人健康管理顾问."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from healthcare.advisor import mcp_app  # noqa: E402
from healthcare.advisor.workbench import render_workbench  # noqa: E402

ROOT = Path(__file__).parents[1]
URI = "ui://personal-health-advisor/workbench.html"


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "data"
    monkeypatch.setenv("HEALTH_ADVISOR_DATA_DIR", str(home))
    monkeypatch.delenv("HEALTHCARE_PASSPHRASE", raising=False)
    monkeypatch.delenv("HEALTHCARE_VAULT", raising=False)
    return home


def test_tool_catalog_marks_ui_resources_and_app_only_tools() -> None:
    by_name = {tool.name: tool for tool in mcp_app.TOOLS}
    assert by_name["health_open_workbench"].meta == {"ui": {"resourceUri": URI}, "ui/resourceUri": URI}
    assert by_name["health_ui_model"].meta["ui"]["visibility"] == ["app"]
    assert "health_ui_record" not in by_name
    assert "health_confirm" not in by_name
    for name, tool in by_name.items():
        assert tool.inputSchema["properties"]["mode"]["enum"] == ["live", "demo"], name


def test_embedded_page_has_no_data_and_defers_csp_to_host() -> None:
    page = render_workbench(None, ui_token="test-token")
    assert "const EMBEDDED=true" in page and '"ui_token":"test-token"' in page
    assert "Content-Security-Policy" not in page
    assert "ui/initialize" in page and "ui/notifications/size-changed" in page and "tools/call" in page
    assert "health_ui_record" not in page and "tip.innerHTML" not in page
    assert "if(e.source!==window.parent)return" in page
    assert "查看虚构示例" in page and "showOnboarding" in page
    assert not re.search(r"https?://", page)
    assert "<form" not in page, "sandboxed MCP hosts block form submission; the quick-record bar must not use <form>"


def test_handle_tool_demo_flow_and_errors(data_dir: Path) -> None:
    opened = mcp_app.handle_tool("health_open_workbench", {"mode": "demo"})
    assert not opened.isError and opened.structuredContent["local_only"] and opened.structuredContent["tiles"]
    assert "bp" in opened.structuredContent and "medications" not in opened.structuredContent

    denied = mcp_app.handle_tool("health_ui_model", {"mode": "demo", "days": 30})
    assert denied.isError and "工作台" in denied.content[0].text
    token = mcp_app._issue_ui_token()
    model = mcp_app.handle_tool("health_ui_model", {"mode": "demo", "days": 30, "ui_token": token})
    assert not model.isError and model.meta["healthData"]["range"]["days"] == 30
    assert model.content[0].text == "本地界面数据已更新."[:-1] + "。"

    today = model.meta["healthData"]["today"]
    bad = mcp_app.handle_tool("health_record_vital", {"mode": "demo", "at": f"{today}T07:00"})
    assert bad.isError and "systolic" in bad.content[0].text
    status = mcp_app.handle_tool("health_status", {"mode": "live"})
    assert not status.isError and status.structuredContent["status"] == "setup_required"
    assert status.structuredContent["demo_available"] is True
    assert status.structuredContent["passphrase_source"] == "none"
    first_open = mcp_app.handle_tool("health_open_workbench", {})
    assert not first_open.isError and first_open.structuredContent["status"] == "setup_required"
    assert first_open.structuredContent["mode"] == "live", "demo must not be silently presented as personal data"
    ui_setup = mcp_app.handle_tool("health_ui_model", {"ui_token": token})
    assert not ui_setup.isError and ui_setup.structuredContent["status"] == "setup_required"


async def _stdio_round_trip(env: dict[str, str]) -> dict[str, object]:
    params = StdioServerParameters(command=sys.executable, args=["-m", "healthcare.advisor", "mcp"], cwd=ROOT, env=env)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            resources = await session.list_resources()
            page = await session.read_resource(URI)
            opened = await session.call_tool("health_open_workbench", {"mode": "demo"})
            recent = await session.call_tool("health_recent", {"mode": "demo", "days": 1})
            return {
                "server": init.serverInfo.name,
                "tools": {tool.name for tool in tools.tools},
                "resource": (str(resources.resources[0].uri), resources.resources[0].mimeType),
                "page_meta": page.contents[0].meta,
                "page_mime": page.contents[0].mimeType,
                "embedded": "const EMBEDDED=true" in page.contents[0].text,
                "opened_error": opened.isError,
                "missing": opened.structuredContent["missing_today"],
                "recent_error": recent.isError,
            }


def test_stdio_round_trip_serves_tools_and_workbench_resource(data_dir: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "HEALTH_ADVISOR_DATA_DIR": str(data_dir)}
    result = asyncio.run(_stdio_round_trip(env))
    assert result["server"] == "personal-health-advisor"
    assert {"health_open_workbench", "health_ui_model", "health_record_vital", "health_candidates"} <= result["tools"]
    assert "health_ui_record" not in result["tools"] and "health_confirm" not in result["tools"]
    assert result["resource"] == (URI, "text/html;profile=mcp-app") and result["page_mime"] == "text/html;profile=mcp-app"
    assert result["page_meta"] == {"ui": {"csp": {"connectDomains": [], "resourceDomains": []}}}
    assert result["embedded"] and not result["opened_error"] and not result["recent_error"]
    assert isinstance(result["missing"], list)
