from __future__ import annotations

from importlib.resources import files

import pytest

from healthcare.vault import VaultConflictError, VaultStore


def test_schema_contracts_are_packaged_with_healthcare_module() -> None:
    resource_root = files("healthcare").joinpath("schema_data")
    assert resource_root.joinpath("observation.schema.json").is_file()
    assert resource_root.joinpath("mcp-envelope.schema.json").is_file()


def test_stale_writer_is_rejected_without_overwriting_committed_record(tmp_path) -> None:
    path = tmp_path / "pilot.vault"
    initial = VaultStore.create(path, "secret")
    initial.ensure_person("me")
    first = VaultStore.open(path, "secret")
    stale = VaultStore.open(path, "secret")
    assert first.record_vital("me", "2026-09-11T08:00", systolic_mmHg=120, diastolic_mmHg=80)
    with pytest.raises(VaultConflictError, match="another local process"):
        stale.record_vital("me", "2026-09-11T20:00", systolic_mmHg=121, diastolic_mmHg=81)
    reopened = VaultStore.open(path, "secret")
    assert len(reopened.vitals("me")) == 1
