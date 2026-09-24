from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from healthcare.vault import VaultStore
from healthcare.control import ControlSession


def _unix_socket_available() -> bool:
    path = f"/tmp/healthcare-mcp-test-{os.getpid()}.sock"
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        connection.bind(path)
        return True
    except OSError:
        return False
    finally:
        connection.close()
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


async def _round_trip(session_file: Path, root: Path, observation_id: str | None = None) -> tuple[set[str], object, object, object | None]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "healthcare.mcp_server", "--session-file", str(session_file)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            timeline = await session.call_tool("health_get_timeline", {"requested_person_id": "me"})
            series = await session.call_tool(
                "health_get_observation_series",
                {"requested_person_id": "me", "field": "creatinine"},
            )
            evidence = None
            if observation_id:
                evidence = await session.call_tool(
                    "health_get_source_evidence",
                    {"requested_person_id": "me", "observation_id": observation_id},
                )
            return {tool.name for tool in tools.tools}, timeline, series, evidence


async def _request_import(session_file: Path, root: Path) -> object:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "healthcare.mcp_server", "--session-file", str(session_file)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await session.call_tool(
                "health_request_document_import",
                {
                    "requested_person_id": "me",
                    "purpose": "add a lab report",
                    "file_types": ["text/plain"],
                },
            )


def test_mcp_stdio_round_trip(tmp_path: Path) -> None:
    if not _unix_socket_available():
        pytest.skip("sandbox does not permit ephemeral session sockets")
    vault_path = tmp_path / "pilot.vault"
    session_path = tmp_path / "session.json"
    store = VaultStore.create(vault_path, "secret")
    store.ensure_person("me")
    store.issue_session(session_path, "me", "stdio-test", ["observations.read"])
    names, result, _series, _evidence = asyncio.run(_round_trip(session_path, Path(__file__).parents[1]))
    assert {
        "health_search_records",
        "health_get_timeline",
        "health_get_observation_series",
        "health_get_source_evidence",
        "health_get_vitals",
        "health_get_medications",
        "health_get_activities",
        "health_get_sleep_records",
        "health_get_emotions",
    } <= names
    assert result.isError is False
    assert result.structuredContent["data"]["person_id"] == "me"
    assert result.structuredContent["scope_used"] == ["observations.read"]
    assert result.structuredContent["request_id"].startswith("req_")


def test_mcp_stdio_reads_confirmed_data_and_source_evidence(tmp_path: Path) -> None:
    if not _unix_socket_available():
        pytest.skip("sandbox does not permit ephemeral session sockets")
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", "报告日期：2026-08-01\n肌酐 88.4 umol/L\n")
    ControlSession(store).review_job(job.id, accept_all=True)
    observation = store.observations("me", "creatinine")[0]
    session_path = store.issue_session(tmp_path / "session.json", "me", "stdio-test", ["observations.read"])
    names, timeline, series, evidence = asyncio.run(
        _round_trip(session_path, Path(__file__).parents[1], observation["id"])
    )
    assert "health_get_timeline" in names
    assert timeline.isError is False
    assert timeline.structuredContent["data"]["items"][0]["value"] == 88.4
    assert series.structuredContent["data"]["items"][0]["field"] == "creatinine"
    assert evidence is not None and evidence.isError is False
    assert evidence.structuredContent["data"]["evidence"]["locator"].startswith("line:")


def test_mcp_stdio_agent_can_request_control_import_without_a_path(tmp_path: Path) -> None:
    if not _unix_socket_available():
        pytest.skip("sandbox does not permit ephemeral session sockets")
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    session_path = store.issue_session(
        tmp_path / "session.json",
        "me",
        "stdio-test",
        ["observations.read", "documents.ingest"],
    )
    result = asyncio.run(_request_import(session_path, Path(__file__).parents[1]))
    assert result.isError is False
    payload = result.structuredContent
    assert payload["scope_used"] == ["documents.ingest"]
    assert payload["data"]["status"] == "awaiting_control"
    assert "path" not in str(payload["data"])
