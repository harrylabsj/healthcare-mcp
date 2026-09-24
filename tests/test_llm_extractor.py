from __future__ import annotations

import pytest

from healthcare.llm_extractor import (
    LlmExtractionError,
    LlmField,
    _validate_item,
    build_extraction_prompt,
)
from healthcare.vault import VaultStore


def test_prompt_is_extraction_only() -> None:
    prompt = build_extraction_prompt()
    assert "只做提取" in prompt or "只能从图片中提取" in prompt
    assert "JSON 数组" in prompt
    assert "creatinine" in prompt
    assert "不能诊断" in prompt


def test_validate_item_numeric_mapped() -> None:
    item = _validate_item({"field": "creatinine", "value": 138.3, "unit": "umol/L", "raw_value": "138.3"})
    assert item.field == "creatinine"
    assert item.value == 138.3
    assert item.unit == "umol/L"
    assert item.value_type == "numeric"


def test_validate_item_unmapped_unit_keeps_none() -> None:
    item = _validate_item({"field": "hemoglobin", "value": 143, "unit": "g/L", "raw_value": "143"})
    assert item.unit is None  # g/L does not fold-match g/dL; stays unmapped
    assert item.raw_unit == "g/L"


def test_validate_item_text_value() -> None:
    item = _validate_item({"field": "anca", "value": "阴性", "unit": None, "raw_value": "阴性"})
    assert item.value_type == "text"
    assert item.value == "阴性"
    assert item.unit is None


def test_validate_item_rejects_unknown_field() -> None:
    with pytest.raises(LlmExtractionError, match="unknown field"):
        _validate_item({"field": "not-a-field", "value": 1, "unit": "x", "raw_value": "1"})


def test_validate_item_rejects_null_value() -> None:
    with pytest.raises(LlmExtractionError, match="no usable value"):
        _validate_item({"field": "creatinine", "value": None, "unit": "umol/L", "raw_value": "?"})


def test_cli_llm_extract_requires_remote_consent(tmp_path) -> None:
    from healthcare.cli import main

    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    image = tmp_path / "scan.png"
    image.write_bytes(b"not-a-real-image")
    rc = main(["llm-extract", "--vault", str(store.path), "--passphrase", "secret", "--image", str(image)])
    assert rc == 2  # refused before any network/vault access


def test_import_llm_extraction_creates_quarantine_candidates(tmp_path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    fields = [
        LlmField("creatinine", 138.3, "umol/L", "138.3"),
        LlmField("anca", "阴性", None, "阴性", value_type="text"),
    ]
    job = store.import_llm_extraction("scan.png", fields)
    assert job.status == "awaiting_identity"
    assert len(job.candidate_ids) == 2
    candidates = [store.state["candidates"][cid] for cid in job.candidate_ids]
    by_field = {c["field"]: c for c in candidates}
    assert by_field["creatinine"]["mapping_status"] == "mapped"
    assert by_field["creatinine"]["normalized_value"] == 138.3
    assert by_field["anca"]["value_type"] == "text"
    assert by_field["anca"]["text_value"] == "阴性"


def test_llm_import_preserves_source_image_and_requires_a_confirmed_date(tmp_path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    image = b"synthetic-image-bytes"
    job = store.import_llm_extraction(
        "scan.png",
        [LlmField("creatinine", 88.4, "umol/L", "88.4")],
        person_id="me",
        source_bytes=image,
        source_media_type="image/png",
    )
    document = store.state["documents"][job.document_id]
    assert document["report_date"] is None
    assert document["object_id"]
    assert store.object_store.get(document["object_id"]) == image
    from healthcare.control import ControlSession
    with pytest.raises(Exception, match="report date"):
        ControlSession(store).review_job(job.id, accept_all=True)
