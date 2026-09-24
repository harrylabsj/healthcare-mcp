from __future__ import annotations

import pytest

from healthcare.control import ControlSession
from healthcare.schemas import (
    SchemaError,
    load,
    schema_names,
    validate,
    validate_entity,
    validate_vault_entities,
)
from healthcare.vault import VaultStore

REPORT = """检验报告\n报告日期：2026-08-01\n血肌酐 88.4 umol/L\neGFR 72 mL/min/1.73m²\n尿白蛋白/肌酐 32 mg/g\n血钾 4.5 mmol/L\n血红蛋白 13.2 g/dL\n血压 128/82 mmHg\n"""


def test_all_frozen_schemas_load_and_are_valid() -> None:
    assert schema_names() == (
        "observation",
        "field-candidate",
        "evidence-record",
        "import-job",
        "mcp-envelope",
    )
    for name in schema_names():
        document = load(name)
        assert document["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert document.get("type") == "object"


def test_full_flow_entities_conform_to_frozen_contract(tmp_path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    job = store.import_text("me", "lab-report.txt", REPORT)
    validate_entity(job, "import-job")
    for candidate_id in job.candidate_ids:
        candidate = store.state["candidates"][candidate_id]
        validate(candidate, "field-candidate")
        validate(store.state["evidence"][candidate["evidence_id"]], "evidence-record")
    ControlSession(store).review_job(job.id, accept_all=True)
    observations = store.observations("me")
    assert observations
    for observation in observations:
        validate(observation, "observation")
    validate_vault_entities(store.state)


def test_quarantined_import_job_and_candidate_conform(tmp_path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    job = store.import_text_unassigned("lab-report.txt", REPORT)
    validate_entity(job, "import-job")
    assert job.person_id is None
    for candidate_id in job.candidate_ids:
        validate(store.state["candidates"][candidate_id], "field-candidate")


def test_mcp_envelope_accepts_null_data_revision() -> None:
    envelope = {
        "request_id": "req_1",
        "data_revision": None,
        "data": {"items": [], "person_id": "me"},
        "evidence_refs": [],
        "warnings": [],
        "scope_used": ["observations.read"],
    }
    validate(envelope, "mcp-envelope")


def test_mcp_envelope_rejects_missing_scope_used() -> None:
    with pytest.raises(SchemaError, match="scope_used"):
        validate(
            {
                "request_id": "req_1",
                "data_revision": 1,
                "data": {},
                "evidence_refs": [],
                "warnings": [],
            },
            "mcp-envelope",
        )


def test_observation_rejects_extra_undeclared_field() -> None:
    observation = {
        "id": "obs_1",
        "person_id": "me",
        "field": "creatinine",
        "value": 88.4,
        "measured_at": "2026-08-01",
        "document_id": "doc_1",
        "evidence_id": "evidence_1",
        "provenance_id": "prov_1",
        "verification_status": "user_confirmed",
        "revision": 1,
        "mapping_status": "mapped",
        "created_at": "2026-08-20T10:00:00+00:00",
        "smuggled_field": True,
    }
    with pytest.raises(SchemaError, match="smuggled_field"):
        validate(observation, "observation")


def test_observation_rejects_impossible_date() -> None:
    base = {
        "id": "obs_1",
        "person_id": "me",
        "field": "creatinine",
        "value": 88.4,
        "measured_at": "2026-13-05",
        "document_id": "doc_1",
        "evidence_id": "evidence_1",
        "provenance_id": "prov_1",
        "verification_status": "user_confirmed",
        "revision": 1,
        "mapping_status": "mapped",
        "created_at": "2026-08-20T10:00:00+00:00",
    }
    with pytest.raises(SchemaError, match="measured_at"):
        validate(base, "observation")


def test_candidate_rejects_confidence_out_of_range() -> None:
    candidate = {
        "id": "cand_1",
        "job_id": "job_1",
        "person_id": "me",
        "field": "creatinine",
        "raw_value": "88.4",
        "normalized_value": 88.4,
        "unit": "umol/L",
        "confidence": 1.5,
        "evidence_id": "evidence_1",
        "raw_unit": "umol/L",
        "mapping_status": "mapped",
        "status": "candidate",
        "verification_status": "candidate",
        "created_at": "2026-08-20T10:00:00+00:00",
    }
    with pytest.raises(SchemaError, match="confidence"):
        validate(candidate, "field-candidate")


def test_import_job_rejects_unknown_status() -> None:
    job = {
        "id": "job_1",
        "document_id": "doc_1",
        "status": "flying",
        "candidate_ids": [],
        "created_at": "2026-08-20T10:00:00+00:00",
    }
    with pytest.raises(SchemaError, match="status"):
        validate(job, "import-job")
