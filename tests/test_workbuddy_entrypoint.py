from __future__ import annotations

import os
import stat

import pytest

from healthcare.workbuddy_entrypoint import load_config, write_config


def test_workbuddy_config_keeps_only_local_routing_fields_and_is_private(tmp_path) -> None:
    config = tmp_path / "workbuddy.json"
    socket = tmp_path / "healthcare.sock"
    written = write_config(config, socket=socket, agent_id="agent_1", host_id="workbuddy-local", person_id="me")
    assert written == config
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert load_config(config) == {
        "socket": str(socket), "agent_id": "agent_1", "host_id": "workbuddy-local", "person_id": "me",
    }
    assert "token" not in config.read_text(encoding="utf-8").casefold()


def test_workbuddy_config_rejects_insecure_permissions_and_relative_socket(tmp_path) -> None:
    config = tmp_path / "bad.json"
    config.write_text('{"socket":"relative.sock","agent_id":"a","host_id":"h","person_id":"p"}', encoding="utf-8")
    config.chmod(0o600)
    with pytest.raises(ValueError, match="absolute"):
        load_config(config)
    config.write_text('{"socket":"/tmp/healthcare.sock","agent_id":"a","host_id":"h","person_id":"p"}', encoding="utf-8")
    if os.name == "posix":
        config.chmod(0o644)
        with pytest.raises(ValueError, match="0600"):
            load_config(config)
