from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from healthcare.mcp_server import create_daemon_server, create_server
from healthcare.control import ControlSession
from healthcare.vault import VaultStore


def test_mcp_server_exposes_only_read_tools(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    server = create_server(store, "me", ["observations.read"])
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "health_search_records",
        "health_get_timeline",
        "health_get_observation_series",
        "health_get_source_evidence",
        "health_get_vitals",
        "health_get_medications",
            "health_get_activities",
            "health_get_emotions",
            "health_get_sleep_records",
            "health_get_encounters",
            "health_get_diagnosis_mentions",
            "health_get_medication_plans",
            "health_get_reminder_rules",
            "health_get_trend_summary",
            "health_prepare_visit_summary",
        }


def test_mcp_server_returns_daily_health_records(tmp_path: Path) -> None:
    from healthcare.csv_import import parse_blood_pressure, parse_exercise, parse_medication

    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    bp = tmp_path / "bp.csv"
    bp.write_text(
        "date,time,period,systolic_mmHg,diastolic_mmHg,heart_rate_bpm,measurement_position,medication_taken,sleep_hours,steps,alcohol,stress_level,symptoms,note\n"
        "2026-03-20,22:11,睡前,128,81,,,未说明,7,,,,,\n",
        encoding="utf-8",
    )
    store.import_vitals("me", parse_blood_pressure(bp), "bp:1")
    server = create_server(store, "me", ["observations.read"])
    vitals = asyncio.run(server.call_tool("health_get_vitals", {"requested_person_id": "me"}))
    assert vitals[1]["data"]["items"][0]["systolic_mmHg"] == 128
    assert vitals[1]["scope_used"] == ["observations.read"]


def test_mcp_server_returns_emotion_records(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    store.record_emotion(
        "me",
        "2026-08-29T09:30",
        "紧张",
        duration_minutes=25,
        feelings="胸口发紧，注意力变窄",
        reflection="先把下一步写下来会更踏实",
    )
    server = create_server(store, "me", ["observations.read"])
    result = asyncio.run(server.call_tool("health_get_emotions", {"requested_person_id": "me"}))
    item = result[1]["data"]["items"][0]
    assert item["name"] == "紧张"
    assert item["duration_minutes"] == 25
    assert item["feelings"] == "胸口发紧，注意力变窄"
    assert item["reflection"] == "先把下一步写下来会更踏实"


def test_mcp_server_returns_sleep_records(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    store.record_sleep(
        "me",
        "2026-09-12",
        bedtime="23:30",
        wake_time="06:45",
        quality=4,
        note="夜里醒过一次",
    )
    server = create_server(store, "me", ["observations.read"])
    result = asyncio.run(server.call_tool("health_get_sleep_records", {"requested_person_id": "me"}))
    item = result[1]["data"]["items"][0]
    assert item["date"] == "2026-09-12"
    assert item["duration_minutes"] == 435.0
    assert item["bedtime"] == "23:30"
    assert item["wake_time"] == "06:45"
    assert item["quality"] == 4


def test_mcp_server_exposes_authorized_read_resources(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", "报告日期：2026-08-01\n肌酐 88.4 umol/L\n")
    ControlSession(store).review_job(job.id, accept_all=True)
    server = create_server(store, "me", ["observations.read"])
    templates = asyncio.run(server.list_resource_templates())
    assert {template.uriTemplate for template in templates} == {
        "health://profiles/{requested_person_id}/timeline",
        "health://profiles/{requested_person_id}/observations/{field}",
        "health://documents/{document_id}/evidence/{evidence_id}",
    }
    contents = list(asyncio.run(server.read_resource("health://profiles/me/timeline")))
    assert len(contents) == 1
    assert '"person_id": "me"' in contents[0].content
    observation = store.observations("me", "creatinine")[0]
    evidence_contents = list(
        asyncio.run(
            server.read_resource(
                f"health://documents/{observation['document_id']}/evidence/{observation['evidence_id']}"
            )
        )
    )
    assert '"locator": "line:' in evidence_contents[0].content


def test_mcp_server_exposes_medication_update_with_write_scope() -> None:
    calls = []

    class FakeClient:
        def call(self, method, params, *, person_id):
            calls.append((method, params, person_id))
            return {"status": "updated", "type": "medication", "medication_id": params["medication_id"]}

    server = create_daemon_server(FakeClient(), "me", ["observations.read", "records.write"])
    tools = asyncio.run(server.list_tools())
    assert "health_update_medication" in {tool.name for tool in tools}
    result = asyncio.run(server.call_tool(
        "health_update_medication",
        {
            "requested_person_id": "me",
            "medication_id": "med_1",
            "medication": "匹伐他汀",
        },
    ))
    assert result[1]["data"]["status"] == "updated"
    assert calls == [("health_update_medication", {"medication_id": "med_1", "changes": {"medication": "匹伐他汀"}}, "me")]


def test_mcp_server_exposes_sleep_recording_with_write_scope() -> None:
    calls = []

    class FakeClient:
        def call(self, method, params, *, person_id):
            calls.append((method, params, person_id))
            return {"status": "recorded", "type": "sleep", "person_id": person_id}

    server = create_daemon_server(FakeClient(), "me", ["observations.read", "records.write"])
    tools = asyncio.run(server.list_tools())
    assert "health_record_sleep" in {tool.name for tool in tools}
    result = asyncio.run(server.call_tool(
        "health_record_sleep",
        {
            "requested_person_id": "me",
            "date": "2026-09-12",
            "bedtime": "23:30",
            "wake_time": "06:45",
            "quality": 4,
        },
    ))
    assert result[1]["data"]["status"] == "recorded"
    assert calls == [(
        "health_record_sleep",
        {
            "date": "2026-09-12",
            "duration_minutes": None,
            "bedtime": "23:30",
            "wake_time": "06:45",
            "quality": 4,
            "note": None,
        },
        "me",
    )]


def test_ingest_scope_exposes_request_and_status_tools(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    server = create_server(store, "me", ["observations.read", "documents.ingest"], owner_id="session-1")
    tools = asyncio.run(server.list_tools())
    names = {tool.name for tool in tools}
    assert {"health_request_document_import", "health_get_import_status"}.issubset(names)
    result = asyncio.run(server.call_tool(
        "health_request_document_import",
        {"requested_person_id": "me", "purpose": "add a report", "file_types": ["text/plain"]},
    ))
    payload = result[1]
    assert payload["data"]["status"] == "awaiting_control"
    assert payload["scope_used"] == ["documents.ingest"]


def test_mcp_rejects_cross_person_queries_before_data_access(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    server = create_server(store, "me", ["observations.read"])
    with pytest.raises(Exception, match="outside the session capability"):
        asyncio.run(server.call_tool("health_get_timeline", {"requested_person_id": "other"}))


def test_generic_record_edit_scope_and_forwarding():
    calls = []

    class FakeClient:
        def call(self, method, params, *, person_id):
            calls.append((method, params, person_id))
            return {'status': 'updated'}

    readonly = create_daemon_server(FakeClient(), 'me', ['observations.read'])
    assert 'health_update_record' not in {t.name for t in asyncio.run(readonly.list_tools())}
    server = create_daemon_server(FakeClient(), 'me', ['observations.read', 'records.write'])
    result = asyncio.run(server.call_tool('health_update_record', {
        'requested_person_id': 'me', 'record_type': 'vital', 'record_id': 'vital_1',
        'changes': {'systolic_mmHg': 125, 'note': None},
    }))
    assert result[1]['data']['status'] == 'updated'
    assert calls == [('health_update_record', {'record_type': 'vital', 'record_id': 'vital_1', 'changes': {'systolic_mmHg': 125, 'note': None}}, 'me')]
