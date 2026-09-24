from __future__ import annotations

import json
import os
import socket
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from healthcare import peer
from healthcare.session_broker import EphemeralHealthDaemon
from healthcare.control import ControlSession
from healthcare.decoder import DecodedDocument, DecodedPage
from healthcare.vault import VaultError, VaultStore


def test_ephemeral_broker_authenticates_in_memory_capability(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    daemon = EphemeralHealthDaemon(
        tmp_path / "session.sock",
        store,
        "session-1",
        "one-time-token",
        "host-1",
        "me",
        ["observations.read"],
        (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
    )
    request = {
        "id": "1",
        "session_id": "session-1",
        "token": "one-time-token",
        "host_id": "host-1",
        "person_id": "me",
        "method": "health_ping",
        "params": {},
    }
    assert daemon.dispatch(request) == {"status": "ok", "person_id": "me"}
    with pytest.raises(VaultError, match="authentication"):
        daemon.dispatch({**request, "token": "wrong"})


def test_ephemeral_broker_refreshes_control_updates_before_agent_poll(tmp_path: Path) -> None:
    vault_path = tmp_path / "pilot.vault"
    store = VaultStore.create(vault_path, "secret")
    store.ensure_person("me")
    daemon = EphemeralHealthDaemon(
        tmp_path / "session.sock",
        store,
        "session-1",
        "one-time-token",
        "host-1",
        "me",
        ["observations.read", "documents.ingest"],
        (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
    )
    base = {
        "id": "1",
        "session_id": "session-1",
        "token": "one-time-token",
        "host_id": "host-1",
        "person_id": "me",
    }
    request = daemon.dispatch({
        **base,
        "method": "health_request_document_import",
        "params": {"purpose": "add lab", "file_types": ["text/plain"]},
    })
    external = VaultStore.open(vault_path, "secret")
    decoded = DecodedDocument(
        "lab-report.txt",
        "text/plain",
        (DecodedPage(1, "报告日期：2026-08-01\n肌酐 88.4 umol/L", "text/plain", "test"),),
        "test",
    )
    job = external.fulfill_import_request(request["request_id"], decoded, source_bytes=b"report")
    control = ControlSession(external)
    control.assign_document(job.document_id, "me")
    control.review_job(job.id, accept_all=True)
    status = daemon.dispatch({
        **base,
        "method": "health_get_import_status",
        "params": {"request_id": request["request_id"]},
    })
    assert status["status"] == "committed"
    assert status["confirmation_receipt_id"]


def test_concurrent_import_requests_are_not_lost(tmp_path: Path) -> None:
    import threading

    vault_path = tmp_path / "pilot.vault"
    store = VaultStore.create(vault_path, "secret")
    store.ensure_person("me")
    daemon = EphemeralHealthDaemon(
        tmp_path / "session.sock",
        store,
        "session-1",
        "one-time-token",
        "host-1",
        "me",
        ["observations.read", "documents.ingest"],
        (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
    )
    base = {
        "session_id": "session-1",
        "token": "one-time-token",
        "host_id": "host-1",
        "person_id": "me",
    }
    barrier = threading.Barrier(2)

    def worker(index: int) -> None:
        barrier.wait()
        daemon.dispatch({
            **base,
            "id": f"{index}",
            "method": "health_request_document_import",
            "params": {"purpose": f"request {index}", "file_types": ["text/plain"]},
        })

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    fresh = VaultStore.open(vault_path, "secret")
    assert len(fresh.state["import_requests"]) == 2


def _daemon(tmp_path: Path, **kwargs: object) -> EphemeralHealthDaemon:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    return EphemeralHealthDaemon(
        tmp_path / "session.sock",
        store,
        "session-1",
        "one-time-token",
        "host-1",
        "me",
        ["observations.read"],
        (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
        **kwargs,
    )


def test_peer_pid_binding_accepts_matching_process(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path)
    request = {
        "id": "1",
        "session_id": "session-1",
        "token": "one-time-token",
        "host_id": "host-1",
        "person_id": "me",
        "peer_pid": os.getpid(),
        "method": "health_ping",
        "params": {},
    }
    assert daemon.dispatch(request, kernel_peer_pid=os.getpid()) == {
        "status": "ok",
        "person_id": "me",
    }


def test_peer_pid_binding_rejects_foreign_process(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path)
    request = {
        "id": "1",
        "session_id": "session-1",
        "token": "one-time-token",
        "host_id": "host-1",
        "person_id": "me",
        "peer_pid": os.getpid() + 1_000_000,
        "method": "health_ping",
        "params": {},
    }
    with pytest.raises(VaultError, match="peer process mismatch"):
        daemon.dispatch(request, kernel_peer_pid=os.getpid())


def test_peer_binding_requires_claim_when_kernel_identifies_peer(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path)
    request = {
        "id": "1",
        "session_id": "session-1",
        "token": "one-time-token",
        "host_id": "host-1",
        "person_id": "me",
        "method": "health_ping",
        "params": {},
    }
    with pytest.raises(VaultError, match="peer process not identified"):
        daemon.dispatch(request, kernel_peer_pid=os.getpid())


@pytest.mark.skipif(sys.platform != "darwin", reason="executable digest is macOS-local")
def test_expected_exec_digest_binds_the_adapter_process(tmp_path: Path) -> None:
    digest = peer.executable_digest_for_pid(os.getpid())
    assert digest is not None
    daemon = _daemon(tmp_path, expected_peer_exec_digest=digest)
    request = {
        "id": "1",
        "session_id": "session-1",
        "token": "one-time-token",
        "host_id": "host-1",
        "person_id": "me",
        "peer_pid": os.getpid(),
        "method": "health_ping",
        "params": {},
    }
    assert daemon.dispatch(request, kernel_peer_pid=os.getpid())["status"] == "ok"


@pytest.mark.skipif(sys.platform != "darwin", reason="executable digest is macOS-local")
def test_expected_exec_digest_rejects_foreign_adapter(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, expected_peer_exec_digest="0" * 64)
    request = {
        "id": "1",
        "session_id": "session-1",
        "token": "one-time-token",
        "host_id": "host-1",
        "person_id": "me",
        "peer_pid": os.getpid(),
        "method": "health_ping",
        "params": {},
    }
    with pytest.raises(VaultError, match="peer executable mismatch"):
        daemon.dispatch(request, kernel_peer_pid=os.getpid())


@pytest.mark.skipif(sys.platform != "darwin", reason="LOCAL_PEERPID is macOS-only")
def test_socket_peer_binding_rejects_forged_peer_pid(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    session_file = tmp_path / "session.json"
    store.issue_session(session_file, "me", "host-1", ["observations.read"])
    session = json.loads(session_file.read_text(encoding="utf-8"))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(5)
        connection.connect(session["socket_path"])
        request = {
            "id": "1",
            "session_id": session["session_id"],
            "token": session["token"],
            "host_id": session["host_id"],
            "person_id": session["person_id"],
            "peer_pid": os.getpid() + 1_000_000,
            "method": "health_ping",
            "params": {},
        }
        connection.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        response = json.loads(connection.recv(1024 * 1024).decode("utf-8"))
    assert response["ok"] is False
    assert "peer process mismatch" in response["error"]
    assert "person_id" not in json.dumps(response.get("result", {}))
