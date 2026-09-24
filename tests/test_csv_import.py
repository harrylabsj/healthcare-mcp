from __future__ import annotations

from pathlib import Path

import pytest

from healthcare.csv_import import (
    CsvImportError,
    parse_blood_pressure,
    parse_exercise,
    parse_medication,
    parse_weight,
)
from healthcare.vault import VaultStore

BP = """date,time,period,systolic_mmHg,diastolic_mmHg,heart_rate_bpm,measurement_position,medication_taken,sleep_hours,steps,alcohol,stress_level,symptoms,note
2025-10-27,22:33,睡前,123,87,,未说明,unknown,,,unknown,,,Apple健康导入
2025-10-28,07:12,晨起,131,90,72,左臂,未说明,7,,none,,,
"""
MED = """date,time,medication,dose,unit,taken,missed_reason,side_effects,blood_pressure_before,blood_pressure_after,note
2026-02-19,21:56,匹伐他汀,1,片,yes,,,111/68,,已服
2026-02-22,09:00,阿利沙坦酯片,1,片,no,漏服,,,,,
"""
EX = """date,activity_type,duration_minutes,distance_km,steps,avg_heart_rate_bpm,max_heart_rate_bpm,intensity_1_5,strength_training,calories,note
2026-05-03,跑步,11.42,1.5,,,,,,,手动记录
2026-05-03,平板支撑,2.33,,,,,,yes,,用时2分20秒
"""
WEIGHT = """date,time,weight_kg,height_cm,bmi,measurement_context,note
2026-05-01,,63.55,163.5,23.77,家用体重秤,用户今日称重
2026-08-25,,64.8,163.5,24.24,家用体重秤,用户今日称重
"""


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_blood_pressure(tmp_path: Path) -> None:
    records = parse_blood_pressure(_write(tmp_path, BP))
    assert len(records) == 2
    assert records[0]["measured_at"] == "2025-10-27T22:33"
    assert records[0]["systolic_mmHg"] == 123
    assert records[0]["diastolic_mmHg"] == 87
    assert records[0]["context"]["period"] == "睡前"
    assert records[1]["heart_rate_bpm"] == 72


def test_parse_medication(tmp_path: Path) -> None:
    records = parse_medication(_write(tmp_path, MED))
    assert len(records) == 2
    assert records[0]["medication"] == "匹伐他汀"
    assert records[0]["taken"] is True
    assert records[1]["taken"] is False


def test_parse_exercise(tmp_path: Path) -> None:
    records = parse_exercise(_write(tmp_path, EX))
    assert len(records) == 2
    assert records[0]["activity_type"] == "跑步"
    assert records[0]["duration_minutes"] == 11.42
    assert records[0]["distance_km"] == 1.5


def test_parse_weight(tmp_path: Path) -> None:
    records = parse_weight(_write(tmp_path, WEIGHT))
    assert len(records) == 2
    assert records[0]["measured_at"] == "2026-05-01"  # no time -> bare date
    assert records[0]["weight_kg"] == 63.55
    assert records[0]["context"]["height_cm"] == "163.5"
    assert records[0]["context"]["bmi"] == "23.77"
    assert records[0]["context"]["measurement_context"] == "家用体重秤"
    assert records[1]["weight_kg"] == 64.8


def test_weight_csv_import_roundtrip(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    records = parse_weight(_write(tmp_path, WEIGHT))
    assert store.import_vitals("me", records, "csv:weight:weight.csv:42") == 2
    assert store.import_vitals("me", records, "csv:weight:weight.csv:42") == 0  # idempotent
    vitals = store.vitals("me")
    assert len(vitals) == 2
    assert vitals[0]["weight_kg"] == 63.55
    assert vitals[1]["weight_kg"] == 64.8
    assert vitals[1]["context"]["bmi"] == "24.24"


def test_record_vital_weight(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    assert store.record_vital("me", "2026-08-25", weight_kg=64.8) is True
    assert store.record_vital("me", "2026-08-25", weight_kg=64.8) is False  # duplicate
    assert store.record_vital("me", "2026-08-25", 128, 81, weight_kg=64.8) is True  # BP + weight
    assert len(store.vitals("me")) == 2


def test_header_mismatch_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(CsvImportError, match="headers"):
        parse_blood_pressure(bad)


def test_vault_record_methods_dedupe(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    assert store.record_vital("me", "2026-08-20T08:00", 128, 81, 72) is True
    assert store.record_vital("me", "2026-08-20T08:00", 128, 81, 72) is False  # duplicate
    assert store.record_medication("me", "2026-08-20T08:00", "匹伐他汀", "1", "片") is True
    assert store.record_medication("me", "2026-08-20T08:00", "匹伐他汀", "1", "片") is False
    assert store.record_activity("me", "2026-08-20", "跑步", 30.0) is True
    assert len(store.vitals("me")) == 1
    assert len(store.medications("me")) == 1
    assert len(store.activities("me")) == 1


def test_update_medication_replaces_a_mistaken_entry_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "pilot.vault"
    store = VaultStore.create(path, "secret")
    store.ensure_person("me")
    assert store.record_medication("me", "2026-09-07T08:00", "阿利沙坦酯片", "1", "片") is True
    record = store.medications("me")[0]

    assert store.update_medication("me", record["id"], {"medication": "匹伐他汀"}) is True
    store.save()

    reopened = VaultStore.open(path, "secret")
    updated = reopened.medications("me")[0]
    assert updated["id"] == record["id"]
    assert updated["medication"] == "匹伐他汀"
    assert any(event["event"] == "medication.updated" for event in reopened.state["audit_events"])


def test_update_medication_rejects_duplicate_target(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    store.record_medication("me", "2026-09-07T08:00", "阿利沙坦酯片", "1", "片")
    store.record_medication("me", "2026-09-07T08:00", "匹伐他汀", "1", "片")
    first_id = store.medications("me")[0]["id"]
    with pytest.raises(Exception, match="conflicts"):
        store.update_medication("me", first_id, {"medication": "匹伐他汀"})


def test_vault_import_and_idempotency(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    assert store.import_vitals("me", parse_blood_pressure(_write(tmp_path, BP)), "bp:1") == 2
    assert store.import_vitals("me", parse_blood_pressure(_write(tmp_path, BP)), "bp:1") == 0  # idempotent
    assert len(store.vitals("me")) == 2
    assert store.import_medications("me", parse_medication(_write(tmp_path, MED)), "med:1") == 2
    assert store.import_activities("me", parse_exercise(_write(tmp_path, EX)), "ex:1") == 2
    assert len(store.medications("me")) == 2
    assert len(store.activities("me")) == 2
    assert store.medications("me")[0]["medication"] == "匹伐他汀"
