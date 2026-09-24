from __future__ import annotations

import pytest

from healthcare.task_mapping import fallback_status_payload, task_status_for_application_status


def test_application_job_status_maps_to_future_tasks_without_handle() -> None:
    assert task_status_for_application_status("awaiting_control") == "input_required"
    assert task_status_for_application_status("committed") == "completed"
    payload = fallback_status_payload({"request_id": "req_1", "status": "awaiting_identity", "job_id": "job_1"})
    assert payload["task_status_projection"] == "input_required"
    assert payload["tasks_handle"] is None
    assert payload["poll_via"] == "health_get_import_status"


def test_unknown_application_status_fails_closed() -> None:
    with pytest.raises(ValueError, match="unmapped"):
        task_status_for_application_status("invented")
