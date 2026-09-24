"""Skill runtime for 个人健康管理顾问: CLI contract, demo Vault, analytics, workbench."""

from __future__ import annotations

import json
import os
import re
import stat
from datetime import date, datetime
from pathlib import Path

import pytest

from healthcare.advisor import analytics
from healthcare.advisor.auth import DEMO_PASSPHRASE
from healthcare.advisor.cli import build_parser, main
from healthcare.advisor.demo import DEMO_PERSON, build_demo_vault
from healthcare.advisor.paths import write_private_text
from healthcare.advisor.settings import SettingsError, load_settings, parse_bp_target, validate_goals
from healthcare.advisor.workbench import render_workbench
from healthcare.vault import VaultStore

ROOT = Path(__file__).parents[1]
PASSPHRASE = "correct horse battery"


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "data"
    monkeypatch.setenv("HEALTH_ADVISOR_DATA_DIR", str(home))
    monkeypatch.setenv("HEALTH_ADVISOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("HEALTHCARE_PASSPHRASE", raising=False)
    monkeypatch.delenv("HEALTHCARE_VAULT", raising=False)
    return home


@pytest.fixture
def passphrase_file(tmp_path: Path) -> Path:
    path = tmp_path / "pp.txt"
    path.write_text(PASSPHRASE + "\n", encoding="utf-8")
    return path


def run(capsys, *argv: str) -> tuple[int, dict, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    payload = json.loads(captured.out) if captured.out.strip() else {}
    return code, payload, captured.err


@pytest.fixture
def live(data_dir: Path, passphrase_file: Path, capsys) -> Path:
    code, payload, _ = run(capsys, "auth", "init", str(passphrase_file), "--person", "me", "--display-name", "测试者")
    assert code == 0 and payload["status"] == "created"
    return data_dir


# ---------- settings ----------

def test_bp_target_parsing_and_validation() -> None:
    assert parse_bp_target("90-130/60-90") == {"systolic": [90, 130], "diastolic": [60, 90]}
    with pytest.raises(SettingsError):
        parse_bp_target("130-90/60-90")
    with pytest.raises(SettingsError):
        parse_bp_target("abc")


def test_goals_validation_rejects_unknown_kind_and_missing_threshold() -> None:
    with pytest.raises(SettingsError):
        validate_goals([{"id": "x", "kind": "steps", "title": "走路"}])
    with pytest.raises(SettingsError):
        validate_goals([{"id": "s", "kind": "sleep_hours", "title": "睡够"}])
    cleaned = validate_goals([{"id": "s", "kind": "sleep_hours", "title": "睡够", "threshold": 7}])
    assert cleaned[0]["threshold"] == 7


# ---------- analytics ----------

def test_note_count_and_bp_slot() -> None:
    assert analytics.note_count("第一组 50 个，第二组 43 个") == (93, "个")
    assert analytics.note_count("共 60 个") == (60, "个")
    assert analytics.note_count("轻松") is None
    assert analytics.bp_slot({"measured_at": "2026-09-14T07:10"}) == "morning"
    assert analytics.bp_slot({"measured_at": "2026-09-14T21:10"}) == "evening"
    assert analytics.bp_slot({"measured_at": "2026-09-14T21:10", "context": {"period": "晨间"}}) == "morning"


def test_bp_stats_include_every_same_slot_measurement() -> None:
    bp = analytics.build_bp([
        {"measured_at": "2026-09-14T07:00", "systolic_mmHg": 120, "diastolic_mmHg": 80},
        {"measured_at": "2026-09-14T07:10", "systolic_mmHg": 140, "diastolic_mmHg": 90},
    ], ["2026-09-14"], {"systolic": [90, 130], "diastolic": [60, 90]})
    assert bp["stats"]["morning_count"] == bp["stats"]["total"] == 2
    assert bp["stats"]["morning_mean"] == [130, 85]
    assert bp["morning"]["2026-09-14"]["sample_count"] == 2


def test_medication_grid_distinguishes_missing_from_skipped() -> None:
    keys = analytics.day_keys(date(2026, 9, 14), 3)
    plans = [{"id": "p1", "medication": "A", "schedule": "每天早上", "dose": "1", "unit": "片", "status": "active", "user_confirmed": True}]
    meds = [
        {"medication": "A", "taken_at": "2026-09-12T07:00", "taken": True, "dose": "1", "unit": "片", "medication_plan_id": "p1"},
        {"medication": "A", "taken_at": "2026-09-13T07:00", "taken": False, "medication_plan_id": "p1"},
    ]
    rows = analytics.build_medications(plans, meds, keys, "2026-09-14")
    cells = rows[0]["cells"]
    assert cells["2026-09-12"]["state"] == "taken" and cells["2026-09-12"]["detail"] == "1片"
    assert cells["2026-09-13"]["state"] == "skipped"
    assert cells["2026-09-14"]["state"] == "pending"
    assert rows[0]["slot"] == "早"


def test_goals_count_only_recorded_days_for_sleep() -> None:
    today = date(2026, 9, 14)
    goals = [{"id": "s", "kind": "sleep_hours", "title": "睡够", "threshold": 7}, {"id": "b", "kind": "bp_twice", "title": "早晚"}]
    sleep = [{"date": "2026-09-13", "duration_minutes": 450}, {"date": "2026-09-14", "duration_minutes": 390}]
    vitals = [{"measured_at": "2026-09-14T07:00", "systolic_mmHg": 120}, {"measured_at": "2026-09-14T21:00", "systolic_mmHg": 118}]
    rows = analytics.build_goals(goals, today=today, vitals=vitals, medications=[], plans=[], activities=[], sleep=sleep)
    assert rows[0]["achieved"] == 1 and rows[0]["of"] == 2
    assert rows[1]["achieved"] == 1 and rows[1]["of"] == 30


# ---------- demo vault + workbench ----------

def test_demo_vault_builds_with_confirmed_and_pending_reports(tmp_path: Path) -> None:
    today = date(2026, 9, 14)
    store = build_demo_vault(tmp_path / "demo.vault", today)
    settings = load_settings(tmp_path / "missing-settings.json")
    model = analytics.build_model(
        store, DEMO_PERSON, settings=settings, today=today, days=14, mode="demo",
        generated_at=datetime(2026, 9, 14, 19, 58), vault_label="demo.vault",
    )
    assert model["bp"]["stats"]["total"] == 25 and model["bp"]["stats"]["highest"]["sys"] == 131
    assert [row["slot"] for row in model["medications"]] == ["早", "晚"]
    assert model["today_block"]["missing"] == ["晚间血压", "用药：调脂药（示例）", "冥想 10 分钟"]
    assert {row["name"] for row in model["activities"]} == {"俯卧撑", "快走", "冥想"}
    assert {row["name"] for row in model["other_activities"]} == {"跑步", "深蹲"}
    assert model["sleep"]["stats"] == {"average": 7.0, "seven_plus": 8, "recorded": 13, "missing": 1}
    assert model["pending"]["candidates"] == 3 and model["pending"]["jobs"][0]["unmapped"] == 1
    assert {row["label"] for row in model["labs"]} >= {"肌酐", "血清尿酸"}
    assert model["encounters"][0]["mentions"] == [{"text": "高血压", "context": "既往"}]
    reopened = VaultStore.open(tmp_path / "demo.vault", DEMO_PASSPHRASE)
    assert reopened.verify_audit_chain()

    html = render_workbench(model)
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    assert not re.search(r"https?://", html), "workbench must not reference any network resource"
    assert "示例数据" in html and "林禾（示例）" in html
    assert "</script" not in json.dumps(model)


def test_cli_demo_workbench_and_status(data_dir: Path, tmp_path: Path, capsys) -> None:
    output = tmp_path / "wb.html"
    code, payload, _ = run(capsys, "--mode", "demo", "workbench", str(output), "--days", "30")
    assert code == 0 and payload["status"] == "generated" and payload["days"] == 30
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert (data_dir / "demo.vault").exists() and (data_dir / "demo.json").exists()
    code, payload, _ = run(capsys, "--mode", "demo", "status")
    assert code == 0 and payload["person_id"] == DEMO_PERSON and payload["passphrase_source"] == "none"


def test_private_write_ignores_a_predictable_temp_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before", encoding="utf-8")
    (tmp_path / ".out.html.tmp").symlink_to(target)
    output = write_private_text(tmp_path / "out.html", "after")
    assert output.read_text(encoding="utf-8") == "after"
    assert target.read_text(encoding="utf-8") == "before"


# ---------- CLI contract ----------

def test_parser_has_no_accept_all_and_rejects_unknown_flags() -> None:
    parser = build_parser()
    assert "--accept-all" not in parser.format_help()
    with pytest.raises(Exception):
        parser.parse_args(["confirm", "job_1", "--accept-all"])
    with pytest.raises(Exception):
        parser.parse_args(["--mode", "typo", "status"])


def test_exit_codes_for_auth_usage_not_found_and_confirmation(data_dir: Path, capsys, tmp_path: Path) -> None:
    code, _, err = run(capsys, "recent")
    assert code == 3 and json.loads(err)["error"] == "auth"
    code, _, err = run(capsys, "record", "vital", "--at", "2026-09-14T07:00")
    assert code in (2, 3)  # usage or auth, never a traceback
    code, _, err = run(capsys, "--mode", "demo", "evidence", "obs_missing")
    assert code == 4 and json.loads(err)["error"] == "not_found"
    code, _, err = run(capsys, "--mode", "demo", "workbench", str(tmp_path / "out.txt"))
    assert code == 2


def test_live_flow_never_exposes_passphrase(live: Path, capsys, tmp_path: Path) -> None:
    code, payload, err = run(capsys, "auth", "status")
    assert code == 0 and payload["configured"] and payload["source"] == "file"
    assert PASSPHRASE not in json.dumps(payload) and PASSPHRASE not in err
    assert stat.S_IMODE((live / "passphrase").stat().st_mode) == 0o600

    today = date.today().isoformat()
    code, payload, _ = run(capsys, "record", "vital", "--at", f"{today}T07:10", "--systolic", "121", "--diastolic", "79")
    assert code == 0 and payload["status"] == "recorded"
    code, payload, _ = run(capsys, "record", "vital", "--at", f"{today}T07:10", "--systolic", "121", "--diastolic", "79")
    assert payload["status"] == "duplicate"
    code, payload, err = run(capsys, "record", "medication", "--at", f"{today}T07:20")
    assert code == 2 and "--medication" in json.loads(err)["message"]
    inp = tmp_path / "sleep.json"
    inp.write_text(json.dumps({"date": today, "bedtime": "23:20", "wake_time": "06:20", "quality": 4}), encoding="utf-8")
    code, payload, _ = run(capsys, "record", "sleep", "--input", str(inp))
    assert code == 0 and payload["type"] == "sleep"
    code, payload, _ = run(capsys, "recent", "--days", "1")
    assert len(payload["vitals"]) == 1 and len(payload["sleep_records"]) == 1 and payload["sleep_records"][0]["duration_minutes"] == 420

    record_id = payload["vitals"][0]["id"]
    code, payload, _ = run(capsys, "edit", "--type", "vital", "--id", record_id, "--changes", json.dumps({"heart_rate_bpm": 66}))
    assert code == 0 and payload["changed_fields"] == ["heart_rate_bpm"]

    output = tmp_path / "wb.html"
    code, payload, _ = run(capsys, "workbench", str(output))
    assert code == 0 and payload["mode"] == "live"
    html = output.read_text(encoding="utf-8")
    assert PASSPHRASE not in html and "个人数据" in html and "测试者" in html


def test_import_candidates_confirm_field_by_field(live: Path, capsys) -> None:
    code, payload, _ = run(capsys, "import", str(ROOT / "fixtures/lab-report-zh.txt"), "--report-date", "2026-08-01")
    assert code == 0 and payload["status"] == "awaiting_review"
    fields = {item["field"] for item in payload["candidates"]}
    assert "creatinine" in fields
    job_id = payload["job_id"]

    code, payload, _ = run(capsys, "labs")
    assert code == 0 and payload["observations"] == []

    code, payload, _ = run(capsys, "confirm", job_id, "--field", "creatinine", "--value", "88.4")
    assert code == 0 and payload["status"] == "confirmed" and payload["confirmation_receipt_id"]
    code, payload, _ = run(capsys, "labs", "--field", "creatinine")
    assert [item["value"] for item in payload["observations"]] == [88.4]
    code, payload, _ = run(capsys, "candidates", job_id)
    statuses = {item["field"]: item["status"] for item in payload["candidates"]}
    assert statuses["creatinine"] == "confirmed" and statuses["egfr"] == "candidate"
    code, labs_payload, _ = run(capsys, "labs")
    code, payload, _ = run(capsys, "evidence", labs_payload["observations"][0]["id"])
    assert code == 0 and payload["evidence"]["locator"]


def test_backup_import_requires_confirmation(live: Path, passphrase_file: Path, capsys, tmp_path: Path) -> None:
    code, payload, _ = run(capsys, "backup", "export", str(tmp_path / "b.hcbackup"), "--passphrase-file", str(passphrase_file))
    assert code == 0 and payload["status"] == "backed_up"
    code, _, err = run(capsys, "backup", "import", str(tmp_path / "b.hcbackup"), "--passphrase-file", str(passphrase_file))
    assert code == 5 and json.loads(err)["error"] == "confirmation_required"


def test_settings_set_and_goals_file(data_dir: Path, capsys, tmp_path: Path) -> None:
    code, payload, _ = run(capsys, "settings", "set", "--bp-target", "100-120/60-80", "--person", "dad")
    assert code == 0 and payload["settings"]["bp_target"]["systolic"] == [100, 120] and payload["settings"]["person_id"] == "dad"
    goals = tmp_path / "goals.json"
    goals.write_text(json.dumps({"goals": [{"id": "walk", "kind": "exercise_daily", "title": "每天走路"}]}), encoding="utf-8")
    code, payload, _ = run(capsys, "settings", "set", "--goals-file", str(goals))
    assert code == 0 and [goal["id"] for goal in payload["settings"]["goals"]] == ["walk"]
    code, _, err = run(capsys, "settings", "set")
    assert code == 2
    assert stat.S_IMODE((data_dir / "settings.json").stat().st_mode) == 0o600


def test_advisor_code_never_deletes_files() -> None:
    """Product rule (2026-09-15): the Skill never deletes or moves user files; WorkBuddy's sandbox forbids it anyway."""
    import re as _re

    package = ROOT / "src" / "healthcare" / "advisor"
    offenders = []
    for path in package.glob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if _re.search(r"\.(unlink|rmdir|rename)\(|os\.replace\(|rmtree|os\.remove\(|shutil\.move\(", code) and "os.replace(temporary, path)" not in code:
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, offenders


def test_auth_revoke_blanks_instead_of_deleting(data_dir: Path, passphrase_file: Path, capsys) -> None:
    code, payload, _ = run(capsys, "auth", "init", str(passphrase_file))
    assert code == 0
    code, payload, _ = run(capsys, "auth", "revoke")
    assert code == 0 and payload["status"] == "revoked" and payload["source_now"] == "none"
    stored = data_dir / "passphrase"
    assert stored.exists() and stored.stat().st_size == 0
    code, _, err = run(capsys, "recent")
    assert code == 3


def test_demo_vault_rebuild_overwrites_without_deleting(tmp_path: Path) -> None:
    from healthcare.advisor.demo import build_demo_vault

    path = tmp_path / "demo.vault"
    build_demo_vault(path, date(2026, 9, 13))
    files_before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    build_demo_vault(path, date(2026, 9, 14))
    reopened = VaultStore.open(path, DEMO_PASSPHRASE)
    assert reopened.verify_audit_chain()
    assert max(v["measured_at"] for v in reopened.vitals(DEMO_PERSON, limit=5000)).startswith("2026-09-14")
    files_after = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    assert files_before <= files_after, "a rebuild overwrites in place and never deletes files"
