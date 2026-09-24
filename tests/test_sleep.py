from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from healthcare.cli import build_parser
from healthcare.vault import VaultError, VaultStore


def test_record_sleep_roundtrip_deduplicates_and_appears_in_recent(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")

    values = {
        "person_id": "me",
        "date": datetime.now().date().isoformat(),
        "duration_minutes": 435.0,
        "bedtime": "23:30",
        "wake_time": "06:45",
        "quality": 4,
        "note": "  夜里醒过一次  ",
    }
    assert store.record_sleep(**values) is True
    assert store.record_sleep(**values) is False

    item = store.sleep_records("me")[0]
    assert item["date"] == values["date"]
    assert item["duration_minutes"] == 435.0
    assert item["bedtime"] == "23:30"
    assert item["wake_time"] == "06:45"
    assert item["quality"] == 4
    assert item["note"] == "夜里醒过一次"
    assert store.recent("me", days=1)["sleep_records"] == [item]

    reopened = VaultStore.open(tmp_path / "pilot.vault", "secret")
    assert reopened.sleep_records("me") == [item]
    assert any(event["event"] == "sleep.recorded" for event in reopened.state["audit_events"])


def test_record_sleep_derives_duration_from_bedtime_and_wake_time(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")

    # A night that crosses midnight belongs to the wake-up date.
    assert store.record_sleep("me", "2026-09-12", bedtime="23:30", wake_time="06:45") is True
    # An after-midnight bedtime stays inside the same night.
    assert store.record_sleep("me", "2026-09-13", bedtime="00:30", wake_time="07:30") is True
    assert [r["duration_minutes"] for r in store.sleep_records("me")] == [435.0, 420.0]

    # A derived duration is the same night as an explicit one.
    assert store.record_sleep(
        "me", "2026-09-12", duration_minutes=435, bedtime="23:30", wake_time="06:45"
    ) is False


def test_record_sleep_rejects_invalid_input(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    with pytest.raises(VaultError, match="date is required"):
        store.record_sleep("me", "")
    with pytest.raises(VaultError, match="date must be an ISO date"):
        store.record_sleep("me", "昨晚", duration_minutes=420)
    with pytest.raises(VaultError, match="duration_minutes or both bedtime and wake_time"):
        store.record_sleep("me", "2026-09-12", bedtime="23:30")
    with pytest.raises(VaultError, match="duration_minutes must be positive"):
        store.record_sleep("me", "2026-09-12", duration_minutes=0)
    with pytest.raises(VaultError, match="duration_minutes must be positive"):
        store.record_sleep("me", "2026-09-12", bedtime="08:00", wake_time="08:00")
    with pytest.raises(VaultError, match="must not exceed 24 hours"):
        store.record_sleep("me", "2026-09-12", duration_minutes=1441)
    with pytest.raises(VaultError, match="quality"):
        store.record_sleep("me", "2026-09-12", duration_minutes=420, quality=9)
    with pytest.raises(VaultError, match="bedtime must use HH:MM"):
        store.record_sleep("me", "2026-09-12", duration_minutes=420, bedtime="11:30pm")
    with pytest.raises(VaultError, match="unknown person"):
        store.record_sleep("other", "2026-09-12", duration_minutes=420)


def test_record_sleep_cli_contract() -> None:
    args = build_parser().parse_args([
        "record-sleep",
        "--vault", "pilot.vault",
        "--person", "me",
        "--date", "2026-09-12",
        "--duration", "435",
        "--bedtime", "23:30",
        "--wake-time", "06:45",
        "--quality", "4",
        "--note", "夜里醒过一次",
    ])
    assert args.command == "record-sleep"
    assert args.date == "2026-09-12"
    assert args.duration == 435.0
    assert args.bedtime == "23:30"
    assert args.wake_time == "06:45"
    assert args.quality == 4


def test_update_record_patches_a_sleep_record(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    store.record_sleep("me", "2026-09-12", duration_minutes=435, bedtime="23:30", wake_time="06:45")
    record_id = store.sleep_records("me")[0]["id"]

    assert store.update_record("me", "sleep", record_id, {"duration_minutes": 400}) is True
    assert store.sleep_records("me")[0]["duration_minutes"] == 400

    assert store.update_record("me", "sleep", record_id, {"wake_time": None}) is True
    assert store.sleep_records("me")[0]["wake_time"] is None

    # The night cannot be left with neither a duration nor a bedtime/wake pair.
    with pytest.raises(VaultError, match="must retain a duration"):
        store.update_record("me", "sleep", record_id, {"duration_minutes": None, "bedtime": None})
    assert store.update_record("me", "sleep", "sleep_missing", {"duration_minutes": 400}) is False
