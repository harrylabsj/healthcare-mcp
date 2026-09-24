from __future__ import annotations

import pytest

from healthcare.parser import parse_report


@pytest.mark.parametrize(
    ("label", "value", "unit", "field"),
    [
        ("Scr", "80,5", "umol/L", "creatinine"),
        ("CREA", "81.5", "umol/L", "creatinine"),
        ("EGFR", "75", "mL/min/1.73m²", "egfr"),
        ("uACR", "20", "mg/g", "uacr"),
        ("K+", "4,2", "mmol/L", "potassium"),
        ("Hb", "12,8", "g/dL", "hemoglobin"),
    ],
)
def test_common_report_label_and_decimal_variants(label: str, value: str, unit: str, field: str) -> None:
    parsed = parse_report(f"检验报告\n{label} {value} {unit}\n")
    result = next(item for item in parsed if item.field == field)
    assert result.raw_value == value
    assert result.raw_unit == unit
    assert result.mapping_status == "mapped"


def test_bp_alias_variant_preserves_two_observations() -> None:
    parsed = parse_report("Lab report\nBP 120 / 78 mmHg\n")
    values = {item.field: item.value for item in parsed}
    assert values == {"systolic_bp": 120.0, "diastolic_bp": 78.0}


def test_uacr_line_before_creatinine_does_not_cross_capture() -> None:
    parsed = parse_report("尿白蛋白/肌酐 32 mg/g\n血肌酐 88.4 umol/L\n")
    values = {item.field: item.value for item in parsed}
    assert values == {"creatinine": 88.4, "uacr": 32.0}
    lines = {item.field: item.source_line for item in parsed}
    assert lines["uacr"].startswith("尿白蛋白/肌酐")
    assert lines["creatinine"].startswith("血肌酐")


def test_reference_range_before_value_is_not_mistaken_for_the_value() -> None:
    parsed = parse_report("肌酐(参考范围:44-133) 88.4 umol/L\n")
    creatinine = next(item for item in parsed if item.field == "creatinine")
    assert creatinine.value == 88.4
    assert creatinine.reference_range_original == "44-133)"


def test_english_bp_evidence_uses_the_matching_line() -> None:
    parsed = parse_report("Lab report\nBP 120 / 78 mmHg\n")
    for item in parsed:
        assert item.source_line == "BP 120 / 78 mmHg"
        assert item.locator == "line:2"


def test_mu_unit_variant_maps_to_the_canonical_unit() -> None:
    parsed = parse_report("血肌酐 88.4 μmol/L\n")
    creatinine = next(item for item in parsed if item.field == "creatinine")
    assert creatinine.mapping_status == "mapped"
    assert creatinine.unit == "umol/L"
    assert creatinine.raw_unit == "μmol/L"


def test_impossible_report_date_falls_back() -> None:
    from healthcare.parser import parse_report_date

    assert parse_report_date("2026年13月5日", "2026-08-20") == "2026-08-20"
