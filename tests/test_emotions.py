from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from healthcare.cli import build_parser
from healthcare.vault import VaultError, VaultStore


def test_record_emotion_roundtrip_deduplicates_and_appears_in_recent(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")

    values = {
        "person_id": "me",
        "occurred_at": f"{datetime.now().date().isoformat()}T09:30",
        "name": "  紧张  ",
        "duration_minutes": 25.0,
        "feelings": "  胸口发紧，注意力变窄  ",
        "reflection": "  先把下一步写下来会更踏实  ",
    }
    assert store.record_emotion(**values) is True
    assert store.record_emotion(**values) is False

    item = store.emotions("me")[0]
    assert item["name"] == "紧张"
    assert item["duration_minutes"] == 25.0
    assert item["feelings"] == "胸口发紧，注意力变窄"
    assert item["reflection"] == "先把下一步写下来会更踏实"
    assert store.recent("me", days=1)["emotions"] == [item]

    reopened = VaultStore.open(tmp_path / "pilot.vault", "secret")
    assert reopened.emotions("me") == [item]
    assert any(event["event"] == "emotion.recorded" for event in reopened.state["audit_events"])


def test_record_emotion_rejects_invalid_input(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    with pytest.raises(VaultError, match="occurred_at and name"):
        store.record_emotion("me", "", "紧张")
    with pytest.raises(VaultError, match="positive"):
        store.record_emotion("me", "2026-08-29T09:30", "紧张", duration_minutes=0)


def test_record_emotion_cli_contract() -> None:
    args = build_parser().parse_args([
        "record-emotion",
        "--vault", "pilot.vault",
        "--person", "me",
        "--occurred-at", "2026-08-29T09:30",
        "--name", "平静",
        "--duration", "45",
        "--feelings", "呼吸变慢",
        "--reflection", "停下来后更容易看清重点",
    ])
    assert args.command == "record-emotion"
    assert args.name == "平静"
    assert args.duration == 45.0
    assert args.feelings == "呼吸变慢"
    assert args.reflection == "停下来后更容易看清重点"
