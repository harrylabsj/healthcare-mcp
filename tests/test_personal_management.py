from __future__ import annotations

import asyncio

import pytest

from healthcare.mcp_server import create_server
from healthcare.summaries import render_visit_summary_markdown
from healthcare.vault import VaultError, VaultStore


def make_store(tmp_path):
    store = VaultStore.create(tmp_path / "personal.vault", "secret")
    store.ensure_person("me")
    store.ensure_person("other")
    return store


def test_encounter_and_verbatim_diagnosis_keep_context_and_person_boundary(tmp_path) -> None:
    store = make_store(tmp_path)
    encounter = store.create_encounter("me", "2026-09-01", facility="示例医院", encounter_type="outpatient")
    mention = store.record_diagnosis_mention(
        "me", "原文：疑似某项问题", "suspected", occurred_on="2026-09-01", encounter_id=encounter.id,
    )
    assert store.encounters("me")[0]["facility"] == "示例医院"
    assert store.diagnoses("me")[0]["context"] == "suspected"
    with pytest.raises(VaultError, match="encounter is not assigned"):
        store.record_diagnosis_mention("other", "原文", "current", encounter_id=encounter.id)
    assert mention.text.startswith("原文：")


def test_active_medication_plan_and_reminder_require_explicit_confirmation(tmp_path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(VaultError, match="requires user confirmation"):
        store.create_medication_plan("me", "示例药", "每日 08:00", status="active")
    plan = store.create_medication_plan("me", "示例药", "每日 08:00", status="active", user_confirmed=True)
    rule = store.create_reminder_rule("me", "medication_plan", "08:00", "记录实际服药", medication_plan_id=plan.id)
    assert rule.medication_plan_id == plan.id
    assert store.reminder_rules("me")[0]["title"] == "记录实际服药"


def test_reminder_evaluation_is_idempotent_and_does_not_create_an_intake_event(tmp_path) -> None:
    store = make_store(tmp_path)
    plan = store.create_medication_plan("me", "示例药", "每日 08:00", status="active", user_confirmed=True)
    rule = store.create_reminder_rule("me", "medication_plan", "08:00", "记录实际服药", medication_plan_id=plan.id)
    first = store.due_reminder_occurrences("me", "2026-09-11T08:30:00+08:00")
    second = store.due_reminder_occurrences("me", "2026-09-11T09:00:00+08:00")
    assert len(first["items"]) == len(second["items"]) == 1
    assert first["items"][0]["id"] == second["items"][0]["id"]
    assert store.medications("me") == []
    assert store.complete_reminder_occurrence("me", first["items"][0]["id"])
    assert store.state["reminder_occurrences"][first["items"][0]["id"]]["status"] == "completed"
    assert rule.id == first["items"][0]["rule_id"]


def test_plan_versioning_links_intake_events_and_pauses_reminders(tmp_path) -> None:
    store = make_store(tmp_path)
    plan = store.create_medication_plan("me", "示例药", "每日 08:00", status="active", user_confirmed=True)
    assert store.record_medication("me", "2026-09-11T08:05", "示例药", medication_plan_id=plan.id)
    intake = store.medications("me")[0]
    assert intake["medication_plan_id"] == plan.id
    assert store.update_medication_plan("me", plan.id, {"schedule": "09:00"})
    assert store.medication_plans("me")[0]["revision"] == 2
    rule = store.create_reminder_rule("me", "medication_plan", "09:00", "记录实际服药", medication_plan_id=plan.id)
    assert store.set_reminder_rule_status("me", rule.id, "paused")
    assert store.reminder_rules("me") == []
    assert store.reminder_rules("me", include_paused=True)[0]["status"] == "paused"


def test_vital_trend_preserves_same_day_measurements_and_returns_daily_means(tmp_path) -> None:
    store = make_store(tmp_path)
    store.record_vital("me", "2026-09-01T08:00", systolic_mmHg=120, diastolic_mmHg=80)
    store.record_vital("me", "2026-09-01T20:00", systolic_mmHg=130, diastolic_mmHg=85)
    store.record_vital("me", "2026-09-02T08:00", systolic_mmHg=125, diastolic_mmHg=82)
    trend = store.trend_summary("me", "vital", "systolic_mmHg")
    assert trend["point_count"] == 3
    assert trend["coverage_days"] == 2
    assert trend["daily_means"] == [
        {"date": "2026-09-01", "value": 125.0, "sample_count": 2},
        {"date": "2026-09-02", "value": 125.0, "sample_count": 1},
    ]


def test_visit_summary_is_a_draft_with_limits_and_selected_time_range(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_encounter("me", "2026-08-01", facility="旧资料")
    store.create_encounter("me", "2026-09-01", facility="本期资料")
    store.record_medication("me", "2026-09-02T08:00", "示例药", dose="1", unit="片")
    summary = store.visit_summary("me", start="2026-09-01", end="2026-09-30")
    assert [item["facility"] for item in summary["encounters"]] == ["本期资料"]
    assert len(summary["medication_intake_events"]) == 1
    assert "不构成诊断" in summary["limitations"][0]
    markdown = render_visit_summary_markdown(summary)
    assert "# 就医资料摘要（草稿）" in markdown
    assert "本期资料" in markdown


def test_mcp_exposes_read_only_personal_management_views(tmp_path) -> None:
    store = make_store(tmp_path)
    store.create_encounter("me", "2026-09-01", facility="本期资料")
    store.create_medication_plan("me", "示例药", "每日 08:00", status="active", user_confirmed=True)
    server = create_server(store, "me", ["observations.read"])
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert {
        "health_get_encounters", "health_get_diagnosis_mentions", "health_get_medication_plans",
        "health_get_reminder_rules", "health_get_trend_summary", "health_prepare_visit_summary",
    }.issubset(names)
    result = asyncio.run(server.call_tool("health_get_encounters", {"requested_person_id": "me"}))
    assert result[1]["data"]["items"][0]["facility"] == "本期资料"


def test_mcp_rejects_cross_person_trends_and_visit_summaries(tmp_path) -> None:
    store = make_store(tmp_path)
    server = create_server(store, "me", ["observations.read"])
    for name, arguments in (
        ("health_get_trend_summary", {"requested_person_id": "other", "source": "vital", "field": "weight_kg"}),
        ("health_prepare_visit_summary", {"requested_person_id": "other"}),
    ):
        with pytest.raises(Exception, match="outside the session capability"):
            asyncio.run(server.call_tool(name, arguments))
