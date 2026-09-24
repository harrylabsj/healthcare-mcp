from __future__ import annotations

import json
import hashlib
import os
import socket
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from healthcare.parser import parse_report
from healthcare.control import ControlSession
from healthcare.decoder import DecodedDocument, DecodedPage
from healthcare.models import now_iso
from healthcare.vault import SessionError, VaultError, VaultStore, read_encrypted_export, read_session


REPORT = """检验报告\n报告日期：2026-08-01\n血肌酐 88.4 umol/L\neGFR 72 mL/min/1.73m²\n尿白蛋白/肌酐 32 mg/g\n血钾 4.5 mmol/L\n血红蛋白 13.2 g/dL\n血压 128/82 mmHg\n"""


def _unix_socket_available() -> bool:
    path = f"/tmp/healthcare-test-{os.getpid()}.sock"
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


def test_parser_creates_candidates_without_writing_facts() -> None:
    fields = parse_report(REPORT)
    assert {field.field for field in fields} == {
        "creatinine",
        "egfr",
        "uacr",
        "potassium",
        "hemoglobin",
        "systolic_bp",
        "diastolic_bp",
    }
    assert next(field for field in fields if field.field == "creatinine").value == 88.4
    assert next(field for field in fields if field.field == "systolic_bp").value == 128


def test_parser_preserves_source_unit_reference_and_comparator() -> None:
    fields = parse_report("检验报告\n肌酐 <5.0 mg/dL 参考范围 0.6-1.2\n")
    creatinine = next(field for field in fields if field.field == "creatinine")
    assert creatinine.raw_unit == "mg/dL"
    assert creatinine.unit is None
    assert creatinine.mapping_status == "unmapped"
    assert creatinine.reference_range_original == "0.6-1.2"
    assert creatinine.raw_comparator == "<"
    assert creatinine.precision == 1


def test_review_job_skips_superseded_candidates(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab.txt", "血肌酐 88.4 umol/L\n")
    store.re_extract_document(job.document_id)  # supersedes the old candidate
    # Confirming the old job must not resurrect its superseded candidate.
    with pytest.raises(Exception, match="no matching candidate"):
        ControlSession(store).review_job(job.id, accept_all=True)
    assert store.observations("me") == []


def test_vault_migration_framework(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from healthcare.vault import VaultStore

    ran: list[bool] = []

    def _migrate_v0(state: dict) -> dict:
        ran.append(True)
        state["migrated_marker"] = "yes"
        return state

    monkeypatch.setitem(VaultStore.MIGRATIONS, 0, _migrate_v0)
    store = VaultStore.create(tmp_path / "v.vault", "secret")
    store.state["vault_version"] = 0
    store.save()
    reopened = VaultStore.open(tmp_path / "v.vault", "secret")
    assert reopened.state["vault_version"] == VaultStore.VERSION
    assert reopened.state.get("migrated_marker") == "yes"
    assert ran == [True]


def test_vault_migration_missing_path_raises(tmp_path: Path) -> None:
    from healthcare.vault import VaultStore

    store = VaultStore.create(tmp_path / "v.vault", "secret")
    store.state["vault_version"] = 0
    store.save()
    with pytest.raises(Exception, match="no migration path"):
        VaultStore.open(tmp_path / "v.vault", "secret")


def test_audit_chain_verifies_and_detects_tamper(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.append_audit("first", "control", "success")
    store.append_audit("second", "agent-1", "success", person_id="me", metadata={"n": 1})
    assert store.verify_audit_chain() is True
    # Tamper with the first event's metadata: the chain must break.
    store.state["audit_events"][0]["metadata"]["tampered"] = True
    assert store.verify_audit_chain() is False


def test_audit_chain_backfills_pre_chain_events(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.state["audit_events"] = [
        {"id": "audit_old1", "event": "old", "actor": "control", "outcome": "success", "person_id": None, "metadata": {}, "created_at": now_iso()}
    ]
    store.append_audit("new", "control", "success")
    assert store.verify_audit_chain() is True


def test_re_extract_generates_fresh_candidates_and_supersedes_old(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    job = store.import_text_unassigned("lab.txt", "血肌酐 88.4 umol/L\neGFR 72 mL/min/1.73m²\n")
    old_candidates = list(job.candidate_ids)
    fresh = store.re_extract_document(job.document_id)
    assert fresh.status == "awaiting_identity"
    assert len(fresh.candidate_ids) == 2  # creatinine + egfr
    assert set(fresh.candidate_ids).isdisjoint(old_candidates)
    for candidate_id in old_candidates:
        assert store.state["candidates"][candidate_id]["status"] == "superseded"


def test_re_extract_skips_confirmed_fields(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab.txt", "血肌酐 88.4 umol/L\neGFR 72 mL/min/1.73m²\n")
    ControlSession(store).review_job(job.id, field="creatinine", value=88.4)
    fresh = store.re_extract_document(job.document_id)
    fields = {store.state["candidates"][cid]["field"] for cid in fresh.candidate_ids}
    assert fields == {"egfr"}  # creatinine already confirmed for this document
    assert store.observations("me")


def test_re_extract_on_fully_confirmed_document_yields_nothing(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab.txt", "血肌酐 88.4 umol/L\n")
    ControlSession(store).review_job(job.id, accept_all=True)
    fresh = store.re_extract_document(job.document_id)
    assert fresh.candidate_ids == []


def test_missing_adjacent_unit_is_unmapped_not_silently_canonical() -> None:
    # Table layouts put the unit in a separate column; without an adjacent unit
    # the parser must not assume the canonical one (143 g/L must never become
    # 143 g/dL silently).
    fields = parse_report("检验报告\n★肌酐(CREA) 139.20 ↑\n")
    creatinine = next(field for field in fields if field.field == "creatinine")
    assert creatinine.unit is None
    assert creatinine.raw_unit is None
    assert creatinine.mapping_status == "unmapped"


def test_urine_panel_fields_and_abbrev_value_skip() -> None:
    fields = parse_report(
        "检验报告\n尿肌酐(UCREA) 13.83 mmol/L\n尿微量白蛋白(MA) 35.30 ↑ 0-19 mg/L\n"
        "尿α1-微球蛋白(A1M) 37.30 ↑ 0.00-12.00 mg/L\nMA/UCREA比值(ACR) 22.56 <30 mg/g\n"
    )
    by_field = {field.field: field for field in fields}
    assert by_field["ucrea"].value == 13.83
    assert by_field["ucrea"].unit == "mmol/L"
    # the "1" inside (A1M) must not be read as the value
    assert by_field["a1m"].value == 37.3
    assert by_field["a1m"].unit == "mg/L"
    assert by_field["ualb"].value == 35.3
    assert by_field["uacr"].value == 22.56
    assert by_field["uacr"].unit == "mg/g"


def test_cbc_fields_with_digit_leading_units() -> None:
    fields = parse_report(
        "检验报告\n★ 白细胞计数 WBC 7.73 - 10^9/L 3.5--9.5\n"
        "★ 血小板计数 PLT 281.00 - 10^9/L 125-350\n"
        "★ 红细胞比容 HCT 43.5 - % 40-50\n"
        "★ 空腹血糖 GLU-0h 4.76 - mmol/L 3.9--6.1\n"
        "★ 血清尿酸 UA 450 ↑ μmol/L 208-428\n"
    )
    by_field = {field.field: field for field in fields}
    assert by_field["wbc"].value == 7.73
    assert by_field["wbc"].unit == "10^9/L"
    assert by_field["plt"].value == 281.0
    assert by_field["hct"].value == 43.5
    assert by_field["hct"].unit == "%"
    assert by_field["fpg"].value == 4.76  # the "0" in GLU-0h is not the value
    assert by_field["ua"].value == 450.0
    assert by_field["ua"].unit == "umol/L"


def test_qualitative_text_value_extraction() -> None:
    fields = parse_report(
        "检验报告\n抗中性粒细胞胞浆抗体(IIF)(ANCA) 阴性 阴性\n"
        "抗蛋白酶3(PR3)抗体(ELISA)(PR3) 阳性 <20 RU/ml\n"
    )
    by_field = {field.field: field for field in fields}
    assert by_field["anca"].value_type == "text"
    assert by_field["anca"].text_value == "阴性"
    assert by_field["pr3"].value_type == "text"
    assert by_field["pr3"].text_value == "阳性"


def test_urine_extras_fields_with_comparator() -> None:
    fields = parse_report(
        "检验报告\n尿转铁蛋白(TRU) <2.00 0-2 mg/L 速率散射比浊法\n"
        "尿免疫球蛋白(IGU) 7.00 <8.00 mg/L 速率散射比浊法\n"
        "NAG酶(NAG) 6.90 0.3-12 U/L MNP-G1CNAc底物法\n"
    )
    by_field = {field.field: field for field in fields}
    assert by_field["utrf"].value == 2.0
    assert by_field["utrf"].raw_comparator == "<"
    assert by_field["utrf"].unit == "mg/L"
    assert by_field["uigu"].value == 7.0
    assert by_field["uigu"].unit == "mg/L"
    assert by_field["nag"].value == 6.9
    assert by_field["nag"].unit == "U/L"


def test_immune_panel_fields() -> None:
    fields = parse_report(
        "检验报告\n免疫球蛋白G(IgG.) 12.46 7-16 g/L 免疫比浊\n"
        "免疫球蛋白A(IgA.) 3.57 0.7-4 g/L 免疫比浊\n"
        "补体C3(C3.) 1.360 0.7-1.8 g/L 免疫比浊\n"
        "补体C4(C4.) 0.370 0.1-0.4 g/L 免疫比浊\n"
    )
    by_field = {field.field: field for field in fields}
    assert by_field["igg"].value == 12.46
    assert by_field["igg"].unit == "g/L"
    assert by_field["iga"].value == 3.57
    assert by_field["c3"].value == 1.36
    assert by_field["c3"].precision == 3
    assert by_field["c4"].value == 0.37
    # 尿免疫球蛋白(IGU) is a urine panel, not serum IgG.
    fields2 = parse_report("检验报告\n尿免疫球蛋白(IGU) 7.00 <8.00 mg/L\n")
    assert "igg" not in {field.field for field in fields2}


def test_longer_label_wins_over_prefix_collision() -> None:
    # 尿微量白蛋白/肌酐 is the uACR ratio, not an albumin concentration.
    fields = parse_report("检验报告\n尿微量白蛋白/肌酐 32 mg/g\n")
    assert {field.field for field in fields} == {"uacr"}
    assert next(field for field in fields if field.field == "uacr").value == 32.0


def test_table_row_reads_unit_from_later_column() -> None:
    fields = parse_report(
        "检验报告\n★肌酐(CREA) 139.20 ↑ 44-133 μmol/L 苦味酸法\n"
        "估算肾小球滤过率(eGFR) 50.108 ml/min/1.73㎡ 计算法(CKD-EPI)\n"
    )
    creatinine = next(field for field in fields if field.field == "creatinine")
    assert creatinine.raw_unit == "μmol/L"
    assert creatinine.unit == "umol/L"
    assert creatinine.mapping_status == "mapped"
    assert creatinine.value == 139.2
    egfr = next(field for field in fields if field.field == "egfr")
    assert egfr.raw_unit == "ml/min/1.73㎡"
    assert egfr.unit == "mL/min/1.73m²"
    assert egfr.mapping_status == "mapped"


def test_table_row_truncated_unit_stays_unmapped() -> None:
    fields = parse_report("检验报告\n估算肾小球滤过率(eGFR) 50.108 ml/min/1.7 计算法(CKD-EPI)\n")
    egfr = next(field for field in fields if field.field == "egfr")
    assert egfr.raw_unit == "ml/min/1.7"
    assert egfr.unit is None
    assert egfr.mapping_status == "unmapped"


def test_ocr_flattened_superscript_unit_still_maps() -> None:
    fields = parse_report("检验报告\neGFR 72 mL/min/1.73m2\n")
    egfr = next(field for field in fields if field.field == "egfr")
    assert egfr.unit == "mL/min/1.73m²"
    assert egfr.raw_unit == "mL/min/1.73m2"
    assert egfr.mapping_status == "mapped"


def test_import_skips_empty_pages_but_keeps_readable_text(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    decoded = DecodedDocument(
        "handbook.pdf",
        "application/pdf",
        (
            DecodedPage(1, "报告日期：2026-08-01\n血肌酐 88.4 umol/L", "text/plain", "test"),
            DecodedPage(2, "   \n", "text/plain", "test"),  # scanned/blank page
            DecodedPage(3, "", "text/plain", "test"),
        ),
        "test",
    )
    job = store.import_decoded_document(None, decoded, source_bytes=b"original")
    assert job.status == "awaiting_identity"
    assert job.document_id
    assert job.candidate_ids  # the blank pages were skipped, text page parsed


def test_import_rejects_document_with_no_readable_pages(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    decoded = DecodedDocument(
        "blank.pdf",
        "application/pdf",
        (DecodedPage(1, "", "text/plain", "test"),),
        "test",
    )
    with pytest.raises(Exception, match="non-empty pages"):
        store.import_decoded_document(None, decoded)


def test_unmapped_unit_cannot_be_committed_as_a_normalized_observation(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", "检验报告\n肌酐 <5.0 mg/dL\n")
    with pytest.raises(Exception, match="unmapped unit"):
        ControlSession(store).review_job(job.id, accept_all=True)
    assert store.observations("me") == []


def test_formal_review_requires_trusted_control_approval(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", "报告日期：2026-08-01\n肌酐 88.4 umol/L\n")
    with pytest.raises(Exception, match="trusted Control approval"):
        store.review_job(job.id, accept_all=True)
    assert store.observations("me") == []


def test_encrypted_vault_round_trip_and_evidence(tmp_path: Path) -> None:
    path = tmp_path / "pilot.vault"
    store = VaultStore.create(path, "correct horse battery staple")
    store.ensure_person("me", "我")
    job = store.import_text("me", "lab-report.txt", REPORT)
    assert job.status == "awaiting_review"
    assert len(job.candidate_ids) == 7
    assert store.observations("me") == []

    committed = ControlSession(store).review_job(job.id, accept_all=True)
    assert committed.status == "committed"
    assert committed.confirmation_receipt_id
    ControlSession(store).review_job(job.id, accept_all=True)
    observations = store.observations("me", "creatinine")
    assert len(observations) == 1
    assert observations[0]["value"] == 88.4

    evidence = store.evidence_for("me", observations[0]["id"])
    assert evidence["evidence"]["locator"].startswith("line:")
    assert evidence["document"]["sha256"]

    reopened = VaultStore.open(path, "correct horse battery staple")
    assert reopened.observations("me", "egfr")[0]["value"] == 72
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "healthCare.encrypted-vault"

    with pytest.raises(Exception):
        VaultStore.open(path, "wrong passphrase")


def test_session_is_short_lived_and_single_use(tmp_path: Path) -> None:
    if not _unix_socket_available():
        pytest.skip("sandbox does not permit ephemeral session sockets")
    vault_path = tmp_path / "pilot.vault"
    session_path = tmp_path / "session.json"
    store = VaultStore.create(vault_path, "secret")
    store.ensure_person("me")
    store.issue_session(session_path, "me", "evaluation-host", ["observations.read"], ttl_seconds=60)
    assert session_path.stat().st_mode & 0o777 == 0o600
    session_text = session_path.read_text(encoding="utf-8")
    assert "vault_passphrase" not in session_text
    assert "vault_path" not in session_text
    session = read_session(session_path)
    assert session["person_id"] == "me"
    assert session["version"] == 2
    assert not session_path.exists()
    with pytest.raises(SessionError):
        read_session(session_path)


def test_session_capability_file_and_broker_argv_never_contain_vault_passphrase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStdin:
        def write(self, _value: bytes) -> None:
            return None

        def close(self) -> None:
            return None

    captured: dict[str, object] = {}

    class FakeProcess:
        stdin = FakeStdin()

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("healthcare.vault.subprocess.Popen", fake_popen)
    store = VaultStore.create(tmp_path / "pilot.vault", "super-secret")
    store.ensure_person("me")
    session_path = store.issue_session(tmp_path / "session.json", "me", "test-host", ["observations.read"])
    assert "super-secret" not in session_path.read_text(encoding="utf-8")
    assert "super-secret" not in " ".join(captured["command"])  # type: ignore[arg-type]


def test_quarantined_import_cannot_enter_timeline_before_identity_confirmation(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text_unassigned("lab-report.txt", REPORT, idempotency_key="upload-1")
    assert job.status == "awaiting_identity"
    assert store.state["documents"][job.document_id]["identity_status"] == "quarantined"
    with pytest.raises(Exception, match="identity"):
        ControlSession(store).review_job(job.id, accept_all=True)
    assigned = ControlSession(store).assign_document(job.document_id, "me")
    assert assigned.status == "awaiting_review"
    ControlSession(store).review_job(job.id, accept_all=True)
    assert len(store.observations("me")) == 7


def test_import_idempotency_returns_the_original_job(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    first = store.import_text("me", "lab-report.txt", REPORT, idempotency_key="same-upload")
    second = store.import_text("me", "lab-report.txt", REPORT, idempotency_key="same-upload")
    assert second.id == first.id
    assert len(store.state["documents"]) == 1


def test_export_defaults_to_encrypted_and_plaintext_requires_confirmation(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", REPORT)
    ControlSession(store).review_job(job.id, accept_all=True)

    encrypted = tmp_path / "confirmed.hcexport"
    ControlSession(store).export_confirmed("me", encrypted, export_passphrase="export-secret")
    assert "血肌酐" not in encrypted.read_text(encoding="utf-8")
    payload = read_encrypted_export(encrypted, "export-secret")
    assert payload["person_id"] == "me"
    assert len(payload["items"]) == 7

    with pytest.raises(Exception, match="confirmation"):
        ControlSession(store).export_confirmed("me", tmp_path / "plain.json", plaintext=True)
    plain = tmp_path / "plain.json"
    ControlSession(store).export_confirmed("me", plain, plaintext=True, confirm_plaintext=True)
    assert "血肌酐" in plain.read_text(encoding="utf-8")


def test_export_requires_trusted_control_approval(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    with pytest.raises(Exception, match="trusted Control approval"):
        store.export_confirmed("me", tmp_path / "out.json", plaintext=True, confirm_plaintext=True)


def test_decoded_document_import_keeps_evidence_on_the_source_page(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    raw_source = b"synthetic PDF bytes"
    decoded = DecodedDocument(
        "lab-report.pdf",
        "application/pdf",
        (
            DecodedPage(1, "检验报告\n血肌酐 88.4 umol/L", "text/plain", "test-decoder"),
            DecodedPage(2, "血钾 4.5 mmol/L\n血压 128/82 mmHg", "text/plain", "test-decoder"),
        ),
        "test-decoder",
        hashlib.sha256(raw_source).hexdigest(),
        len(raw_source),
    )
    job = store.import_decoded_document("me", decoded, source_bytes=raw_source)
    ControlSession(store).review_job(job.id, accept_all=True)
    potassium = next(item for item in store.observations("me") if item["field"] == "potassium")
    evidence = store.evidence_for("me", potassium["id"])
    assert evidence["evidence"]["page_number"] == 2
    assert evidence["document"]["media_type"] == "application/pdf"
    assert evidence["document"]["sha256"] == hashlib.sha256(raw_source).hexdigest()
    assert evidence["document"]["object_id"]
    assert store.read_source_object("me", decoded_document_id := job.document_id) == raw_source
    reopened = VaultStore.open(tmp_path / "pilot.vault", "secret")
    assert reopened.read_source_object("me", decoded_document_id) == raw_source


def test_control_confirmation_receipt_contains_no_health_content(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", REPORT)
    committed = ControlSession(store).review_job(job.id, accept_all=True)
    receipt = next(event for event in store.audit_events() if event["id"] == committed.confirmation_receipt_id)
    assert receipt["event"] == "extraction.confirmed"
    assert receipt["actor"] == "control"
    assert "血肌酐" not in json.dumps(receipt, ensure_ascii=False)


def test_agent_import_request_requires_control_fulfillment_and_stays_quarantined(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    request = store.request_document_import("me", "session-1", ["text/plain"], "add a lab report")
    assert request["status"] == "awaiting_control"
    assert "path" not in json.dumps(request)
    decoded = DecodedDocument(
        "lab-report.txt",
        "text/plain",
        (DecodedPage(1, "报告日期：2026-08-01\n肌酐 88.4 umol/L", "text/plain", "test-decoder"),),
        "test-decoder",
    )
    job = store.fulfill_import_request(request["request_id"], decoded, source_bytes=b"report-bytes")
    assert job.status == "awaiting_identity"
    status = store.import_request_status(request["request_id"], "me", "session-1")
    assert status["status"] == "awaiting_identity"
    with pytest.raises(Exception, match="identity"):
        ControlSession(store).review_job(job.id, accept_all=True)


def test_agent_control_review_lifecycle_reaches_committed_receipt(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    request = store.request_document_import("me", "agent-session", ["text/plain"], "add a lab report")
    decoded = DecodedDocument(
        "lab-report.txt",
        "text/plain",
        (DecodedPage(1, "报告日期：2026-08-01\n肌酐 88.4 umol/L", "text/plain", "test-decoder"),),
        "test-decoder",
    )
    job = store.fulfill_import_request(request["request_id"], decoded, source_bytes=b"report-bytes")
    assert store.import_request_status(request["request_id"], "me", "agent-session")["status"] == "awaiting_identity"

    control = ControlSession(store)
    assigned = control.assign_document(job.document_id, "me")
    assert assigned.status == "awaiting_review"
    assert store.import_request_status(request["request_id"], "me", "agent-session")["status"] == "awaiting_review"

    committed = control.review_job(job.id, accept_all=True)
    final_status = store.import_request_status(request["request_id"], "me", "agent-session")
    assert committed.status == "committed"
    assert final_status["status"] == "committed"
    assert final_status["confirmation_receipt_id"] == committed.confirmation_receipt_id
    assert final_status["task_status_projection"] == "completed"
    assert final_status["tasks_handle"] is None
    assert final_status["poll_via"] == "health_get_import_status"
    assert len(store.observations("me")) == 1


def test_review_rejects_accept_all_combined_with_value(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", "肌酐 88.4 umol/L\n血钾 4.5 mmol/L\n")
    with pytest.raises(Exception, match="accept-all"):
        ControlSession(store).review_job(job.id, accept_all=True, field="creatinine", value=999.0)
    with pytest.raises(Exception, match="requires a field"):
        ControlSession(store).review_job(job.id, value=1.0)
    assert store.observations("me") == []


def test_unmapped_unit_can_be_corrected_with_value_and_unit(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", "肌酐 5.0 mg/dL\n")
    with pytest.raises(Exception, match="unmapped unit"):
        ControlSession(store).review_job(job.id, accept_all=True)
    ControlSession(store).review_job(job.id, field="creatinine", value=442.0, unit="umol/L")
    observations = store.observations("me", "creatinine")
    assert observations[0]["value"] == 442.0
    assert observations[0]["unit"] == "umol/L"


def test_mu_variant_unit_confirms_without_correction(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", "血肌酐 88.4 μmol/L\n")
    ControlSession(store).review_job(job.id, accept_all=True)
    observations = store.observations("me", "creatinine")
    assert observations[0]["value"] == 88.4
    assert observations[0]["unit"] == "umol/L"


def test_fabricated_approval_identifiers_are_rejected(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", "肌酐 88.4 umol/L\n")
    with pytest.raises(Exception, match="trusted Control approval"):
        store.review_job(
            job.id,
            accept_all=True,
            control_session_id="forged-session",
            approval_grant_id="forged-grant",
            expected_revision=store.state["data_revision"],
        )
    assert store.observations("me") == []


def test_registered_approval_grant_is_single_use_and_action_bound(tmp_path: Path) -> None:
    from dataclasses import asdict

    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", "肌酐 88.4 umol/L\n血钾 4.5 mmol/L\n")
    control = ControlSession(store)
    selected = job.candidate_ids
    grant = control.issue_grant("records.export", "me", ["wrong-target"], "wrong action grant")
    store.register_approval(asdict(grant))
    with pytest.raises(Exception, match="does not match the requested action"):
        store.review_job(
            job.id,
            accept_all=True,
            control_session_id=control.session_id,
            approval_grant_id=grant.grant_id,
            expected_revision=grant.expected_revision,
        )
    review_grant = control.issue_grant("extraction.review", "me", selected, "confirm candidates")
    store.register_approval(asdict(review_grant))
    store.review_job(
        job.id,
        accept_all=True,
        control_session_id=control.session_id,
        approval_grant_id=review_grant.grant_id,
        expected_revision=review_grant.expected_revision,
    )
    with pytest.raises(Exception, match="trusted Control approval"):
        store.review_job(
            job.id,
            accept_all=True,
            control_session_id=control.session_id,
            approval_grant_id=review_grant.grant_id,
            expected_revision=review_grant.expected_revision,
        )
    assert len(store.observations("me")) == 2


def test_broker_startup_failure_is_detected_and_session_file_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not _unix_socket_available():
        pytest.skip("sandbox does not permit ephemeral session sockets")
    import healthcare.vault as vault_module

    monkeypatch.setattr(vault_module.sys, "executable", "/usr/bin/false")
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    session_path = tmp_path / "session.json"
    with pytest.raises(SessionError, match="broker"):
        store.issue_session(session_path, "me", "host", ["observations.read"])
    assert not session_path.exists()


def test_recent_filters_records_to_day_window(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    today = datetime.now().date()
    yesterday = (today - timedelta(days=1)).isoformat()
    old = (today - timedelta(days=30)).isoformat()
    store.record_vital("me", f"{today.isoformat()}T08:00", 128, 81, 72)
    store.record_vital("me", f"{old}T08:00", 130, 82, 70)
    store.record_medication("me", f"{yesterday}T21:30", "匹伐他汀", "1", "片")
    store.record_activity("me", today.isoformat(), "俯卧撑", note="第一组103个")
    store.record_activity("me", old, "跑步", 30.0)

    one_day = store.recent("me", days=1)
    assert one_day["since"] == today.isoformat()
    assert len(one_day["vitals"]) == 1
    assert one_day["vitals"][0]["systolic_mmHg"] == 128
    assert len(one_day["activities"]) == 1
    assert one_day["activities"][0]["activity_type"] == "俯卧撑"
    assert one_day["medications"] == []

    week = store.recent("me", days=7)
    assert len(week["vitals"]) == 1
    assert len(week["medications"]) == 1
    assert len(week["activities"]) == 1

    month = store.recent("me", days=31)
    assert len(month["vitals"]) == 2
    assert len(month["activities"]) == 2


def test_recent_rejects_out_of_range_days(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    with pytest.raises(VaultError, match="days"):
        store.recent("me", days=0)
    with pytest.raises(VaultError, match="days"):
        store.recent("me", days=367)
