from __future__ import annotations

import threading
from pathlib import Path

import pytest

from healthcare.daemon import LocalHealthDaemon, LocalHealthClient
from healthcare.trust import MemoryKeychain, TrustManager
from healthcare.vault import VaultError, VaultStore


def test_authenticated_daemon_dispatch(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent("Test Agent", "host-1", "sha256:abc", ["me"], ["observations.read"])
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)
    client = LocalHealthClient(tmp_path / "healthcare.sock", profile.agent_id, token, "host-1", "me")

    assert daemon.dispatch({
        "id": "1", "agent_id": profile.agent_id, "token": token, "host_id": "host-1",
        "person_id": "me", "method": "health_ping", "params": {},
    }) == {"status": "ok", "person_id": "me"}
    assert daemon.dispatch({
        "id": "2", "agent_id": profile.agent_id, "token": token, "host_id": "host-1",
        "person_id": "me", "method": "health_get_timeline", "params": {},
    })["items"] == []
    with pytest.raises(VaultError):
        daemon.dispatch({
            "id": "3", "agent_id": profile.agent_id, "token": "wrong", "host_id": "host-1",
            "person_id": "me", "method": "health_ping", "params": {},
        })


def test_daemon_challenge_response_accepts_valid_proof(tmp_path: Path) -> None:
    import hashlib

    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent("CR Agent", "host-1", "sha256:abc", ["me"], ["observations.read"])
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    nonce = "a1b2c3d4"
    proof = TrustManager._challenge_response(digest, nonce, profile.agent_id, "me", "health_ping")
    assert daemon.dispatch({
        "id": "1", "agent_id": profile.agent_id, "host_id": "host-1",
        "person_id": "me", "method": "health_ping", "params": {},
        "nonce": nonce, "proof": proof,
    }) == {"status": "ok", "person_id": "me"}


def test_daemon_challenge_response_rejects_replayed_nonce(tmp_path: Path) -> None:
    import hashlib

    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent("CR Agent", "host-1", "sha256:abc", ["me"], ["observations.read"])
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    nonce = "replay-nonce-1"
    proof = TrustManager._challenge_response(digest, nonce, profile.agent_id, "me", "health_ping")
    request = {
        "id": "1", "agent_id": profile.agent_id, "host_id": "host-1",
        "person_id": "me", "method": "health_ping", "params": {},
        "nonce": nonce, "proof": proof,
    }
    assert daemon.dispatch(request)["status"] == "ok"
    with pytest.raises(VaultError, match="replayed"):
        daemon.dispatch(request)


def test_daemon_challenge_response_rejects_wrong_proof(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, _token = trust.pair_agent("CR Agent", "host-1", "sha256:abc", ["me"], ["observations.read"])
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)
    with pytest.raises(VaultError, match="authentication failed"):
        daemon.dispatch({
            "id": "1", "agent_id": profile.agent_id, "host_id": "host-1",
            "person_id": "me", "method": "health_ping", "params": {},
            "nonce": "bad-nonce", "proof": "0" * 64,
        })


def test_authenticated_unix_socket_round_trip(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent("Socket Agent", "host-1", "sha256:abc", ["me"], ["observations.read"])
    # AF_UNIX has a short path limit; use a relative test socket so the
    # sandbox's long temporary directory does not mask daemon behavior.
    socket_path = Path("healthcare-test.sock")
    daemon = LocalHealthDaemon(socket_path, trust)
    try:
        probe = daemon._make_server()
    except PermissionError:
        pytest.skip("sandbox does not permit Unix socket binding")
    else:
        daemon._close_server(probe)
    worker = threading.Thread(target=daemon.serve_once, daemon=True)
    worker.start()
    client = LocalHealthClient(socket_path, profile.agent_id, token, "host-1", "me")
    assert client.call("health_ping") == {"status": "ok", "person_id": "me"}
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert not socket_path.exists()


def test_daemon_reports_agent_capabilities(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent("Cap Agent", "host-1", "sha256:abc", ["me"], ["observations.read"])
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)
    result = daemon.dispatch({
        "id": "1", "agent_id": profile.agent_id, "token": token, "host_id": "host-1",
        "person_id": "me", "method": "health_get_capabilities", "params": {},
    })
    assert result == {"person_id": "me", "person_ids": ["me"], "scopes": ["observations.read"]}


def test_daemon_rejects_out_of_range_timeline_limit(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent("Limit Agent", "host-1", "sha256:abc", ["me"], ["observations.read"])
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)
    with pytest.raises(VaultError, match="limit"):
        daemon.dispatch({
            "id": "1", "agent_id": profile.agent_id, "token": token, "host_id": "host-1",
            "person_id": "me", "method": "health_get_timeline", "params": {"limit": 0},
        })


def test_daemon_records_and_reads_emotion_with_write_scope(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent(
        "Emotion Agent",
        "host-1",
        "sha256:abc",
        ["me"],
        ["observations.read", "records.write"],
    )
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)
    base = {
        "agent_id": profile.agent_id,
        "token": token,
        "host_id": "host-1",
        "person_id": "me",
    }
    result = daemon.dispatch({
        **base,
        "id": "1",
        "method": "health_record_emotion",
        "params": {
            "occurred_at": "2026-08-29T09:30",
            "name": "紧张",
            "duration_minutes": 25,
            "feelings": "胸口发紧",
            "reflection": "把下一步写下来",
        },
    })
    assert result["status"] == "recorded"
    values = daemon.dispatch({**base, "id": "2", "method": "health_get_emotions", "params": {}})
    assert values["items"][0]["name"] == "紧张"
    assert values["items"][0]["duration_minutes"] == 25.0


def test_daemon_records_and_reads_sleep_with_write_scope(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent(
        "Sleep Agent",
        "host-1",
        "sha256:abc",
        ["me"],
        ["observations.read", "records.write"],
    )
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)
    base = {
        "agent_id": profile.agent_id,
        "token": token,
        "host_id": "host-1",
        "person_id": "me",
    }
    result = daemon.dispatch({
        **base,
        "id": "1",
        "method": "health_record_sleep",
        "params": {"date": "2026-09-12", "bedtime": "23:30", "wake_time": "06:45"},
    })
    assert result["status"] == "recorded"
    assert result["type"] == "sleep"
    duplicate = daemon.dispatch({
        **base,
        "id": "2",
        "method": "health_record_sleep",
        "params": {"date": "2026-09-12", "bedtime": "23:30", "wake_time": "06:45"},
    })
    assert duplicate["status"] == "duplicate"
    values = daemon.dispatch({**base, "id": "3", "method": "health_get_sleep_records", "params": {}})
    assert values["items"][0]["date"] == "2026-09-12"
    assert values["items"][0]["duration_minutes"] == 435.0


def test_daemon_write_survives_commit_by_another_local_process(tmp_path: Path) -> None:
    """A daemon running since before another writer committed must still write.

    Regression: the daemon used to re-read only ``state`` per request and keep
    its startup ``_persisted_revision``, so after any CLI/cron commit every MCP
    write raised VaultConflictError until the daemon was restarted.
    """
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent(
        "Activity Agent", "host-1", "sha256:abc", ["me"], ["observations.read", "records.write"]
    )
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)
    # Another local process (CLI or cron) commits after the daemon started.
    external = VaultStore.open(store.path, "secret")
    assert external.record_activity("me", "2026-09-15", "跑步", 30.0)
    result = daemon.dispatch({
        "id": "1",
        "agent_id": profile.agent_id,
        "token": token,
        "host_id": "host-1",
        "person_id": "me",
        "method": "health_record_activity",
        "params": {"date": "2026-09-15", "activity_type": "俯卧撑", "note": "第一组60个"},
    })
    assert result["status"] == "recorded"
    activities = VaultStore.open(store.path, "secret").activities("me")
    assert sorted(item["activity_type"] for item in activities) == ["俯卧撑", "跑步"]


def test_daemon_replays_once_when_another_process_commits_mid_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit landing between our refresh and our save is retried, not surfaced."""
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent(
        "Activity Agent", "host-1", "sha256:abc", ["me"], ["observations.read", "records.write"]
    )
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)

    original_save = VaultStore.save
    injected = {"done": False}

    def save_with_competing_commit(self: VaultStore, *args: object, **kwargs: object) -> None:
        if not injected["done"] and self.path == store.path:
            injected["done"] = True
            competing = VaultStore.open(store.path, "secret")
            assert competing.record_activity("me", "2026-09-14", "跑步", 30.0)
        return original_save(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(VaultStore, "save", save_with_competing_commit)
    result = daemon.dispatch({
        "id": "1",
        "agent_id": profile.agent_id,
        "token": token,
        "host_id": "host-1",
        "person_id": "me",
        "method": "health_record_activity",
        "params": {"date": "2026-09-15", "activity_type": "俯卧撑", "note": "第一组60个"},
    })
    monkeypatch.undo()
    assert result["status"] == "recorded"
    activities = VaultStore.open(store.path, "secret").activities("me")
    assert sorted(item["activity_type"] for item in activities) == ["俯卧撑", "跑步"]
    assert len([item for item in activities if item["activity_type"] == "俯卧撑"]) == 1


def test_daemon_updates_medication_with_write_scope(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    store.record_medication("me", "2026-09-07T08:00", "阿利沙坦酯片", "1", "片")
    store.save()
    medication_id = store.medications("me")[0]["id"]
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent(
        "Medication Agent", "host-1", "sha256:abc", ["me"], ["observations.read", "records.write"]
    )
    daemon = LocalHealthDaemon(tmp_path / "healthcare.sock", trust)
    base = {
        "agent_id": profile.agent_id,
        "token": token,
        "host_id": "host-1",
        "person_id": "me",
    }
    result = daemon.dispatch({
        **base,
        "id": "1",
        "method": "health_update_medication",
        "params": {"medication_id": medication_id, "changes": {"medication": "匹伐他汀"}},
    })
    assert result["status"] == "updated"
    assert VaultStore.open(store.path, "secret").medications("me")[0]["medication"] == "匹伐他汀"
