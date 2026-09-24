from __future__ import annotations

from pathlib import Path

import pytest

from healthcare.trust import MemoryKeychain, TrustManager
from healthcare.vault import VaultError
from healthcare.control import ControlSession
from healthcare.vault import VaultError, VaultStore


def test_pair_authenticate_and_revoke(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    keychain = MemoryKeychain()
    trust = TrustManager(store, keychain)
    profile, token = trust.pair_agent("Test Agent", "host-1", "sha256:abc", ["me"], ["observations.read"])
    assert keychain.get(TrustManager.SERVICE, profile.agent_id) == token
    assert trust.authenticate(profile.agent_id, token, "host-1", "me", "observations.read").agent_id == profile.agent_id
    with pytest.raises(VaultError):
        trust.authenticate(profile.agent_id, token, "host-1", "other", "observations.read")
    trust.revoke_agent(profile.agent_id)
    with pytest.raises(VaultError):
        trust.authenticate(profile.agent_id, token, "host-1", "me", "observations.read")
    events = store.audit_events()
    assert any(event["event"] == "agent.paired" for event in events)
    assert any(event["event"] == "agent.revoked" for event in events)


def test_encrypted_backup_restore(tmp_path: Path) -> None:
    source = tmp_path / "source.vault"
    restored = tmp_path / "restored.vault"
    backup = tmp_path / "backup.hcbak"
    store = VaultStore.create(source, "vault-secret")
    store.ensure_person("me")
    object_ref = store.object_store.put(b"synthetic source object", "application/pdf")
    ControlSession(store).backup_to(backup, "backup-secret")
    VaultStore.restore_from(backup, restored, "backup-secret", confirmation="RESTORE_VAULT")
    restored_store = VaultStore.open(restored, "vault-secret")
    assert restored_store.state["persons"]["me"]["display_name"] == "me"
    assert restored_store.object_store.get(object_ref.object_id) == b"synthetic source object"
    receipt = restored.with_name(f"{restored.name}.restore-receipt.json")
    assert receipt.exists()
    receipt_text = receipt.read_text(encoding="utf-8")
    assert '"format": "healthCare.restore-receipt"' in receipt_text
    assert "synthetic source object" not in receipt_text
    with pytest.raises(VaultError):
        VaultStore.restore_from(backup, tmp_path / "bad.vault", "wrong", confirmation="RESTORE_VAULT")


def test_restore_requires_explicit_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "source.vault"
    backup = tmp_path / "backup.hcbak"
    store = VaultStore.create(source, "secret")
    ControlSession(store).backup_to(backup, "backup-secret")
    with pytest.raises(VaultError, match="RESTORE_VAULT"):
        VaultStore.restore_from(backup, tmp_path / "restored.vault", "backup-secret")


def test_backup_and_key_rotation_require_control_and_preserve_data(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    backup = tmp_path / "backup"
    store = VaultStore.create(vault, "old-secret")
    store.ensure_person("me")
    with pytest.raises(VaultError, match="trusted Control"):
        store.backup_to(backup, "backup-secret")
    ControlSession(store).rotate_passphrase("new-secret")
    with pytest.raises(VaultError):
        VaultStore.open(vault, "old-secret")
    assert VaultStore.open(vault, "new-secret").state["persons"]["me"]["display_name"] == "me"


def test_crypto_erase_requires_control_and_removes_objects(tmp_path: Path) -> None:
    vault = tmp_path / "pilot.vault"
    store = VaultStore.create(vault, "secret")
    store.ensure_person("me")
    store.object_store.put(b"private source", "text/plain")
    with pytest.raises(VaultError, match="trusted Control"):
        store.crypto_erase("ERASE_VAULT")
    tombstone = ControlSession(store).crypto_erase("ERASE_VAULT")
    assert not vault.exists()
    assert not Path(f"{vault}.objects").exists()
    assert tombstone.exists()
    assert "me" not in tombstone.read_text(encoding="utf-8")


def test_macos_keychain_round_trip(tmp_path: Path) -> None:
    import platform

    if platform.system() != "Darwin":
        pytest.skip("macOS Keychain is only available on macOS")
    from healthcare.trust import MacOSKeychain

    keychain = MacOSKeychain()
    service = f"healthCare.test.{__import__('secrets').token_hex(6)}"
    account = "round-trip-agent"
    try:
        keychain.set(service, account, "probe-token-123")
        assert keychain.get(service, account) == "probe-token-123"
    finally:
        keychain.delete(service, account)
    assert keychain.get(service, account) is None


def test_successful_authentication_does_not_rewrite_the_vault(tmp_path: Path) -> None:
    vault_path = tmp_path / "pilot.vault"
    store = VaultStore.create(vault_path, "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent("Read Agent", "host-1", "sha256:abc", ["me"], ["observations.read"])
    revision_before = store.state["data_revision"]
    bytes_before = vault_path.read_bytes()
    assert trust.authenticate(profile.agent_id, token, "host-1", "me", "observations.read").agent_id == profile.agent_id
    assert store.state["data_revision"] == revision_before
    assert vault_path.read_bytes() == bytes_before


def test_restore_refuses_existing_destination_and_replace_wipes_stale_objects(tmp_path: Path) -> None:
    source = tmp_path / "source.vault"
    backup = tmp_path / "backup.hcbak"
    store = VaultStore.create(source, "secret")
    store.ensure_person("me")
    ControlSession(store).backup_to(backup, "backup-secret")

    existing = tmp_path / "existing.vault"
    VaultStore.create(existing, "other-secret")
    stale_objects = existing.with_name(f"{existing.name}.objects")
    stale_objects.mkdir()
    (stale_objects / "orphan.hobj").write_bytes(b"stale")

    with pytest.raises(VaultError, match="already exists"):
        VaultStore.restore_from(backup, existing, "backup-secret", confirmation="RESTORE_VAULT")

    VaultStore.restore_from(backup, existing, "backup-secret", confirmation="RESTORE_VAULT", replace=True)
    assert not (stale_objects / "orphan.hobj").exists()
    restored = VaultStore.open(existing, "secret")
    assert restored.state["persons"]["me"]["display_name"] == "me"


def test_pair_agent_issues_default_consent_grants(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, _token = trust.pair_agent("Agent", "host-1", "sha256:x", ["me"], ["observations.read"])
    assert store.consent_is_active(profile.agent_id, "me", "observations.read") is True


def test_revoke_consent_blocks_authentication(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent("Agent", "host-1", "sha256:x", ["me"], ["observations.read"])
    assert trust.authenticate(profile.agent_id, token, "host-1", "me", "observations.read") is not None
    grants = store.active_consent_grants(agent_id=profile.agent_id, person_id="me", scope="observations.read")
    store.revoke_consent_grant(grants[0]["id"])
    with pytest.raises(VaultError, match="authentication failed"):
        trust.authenticate(profile.agent_id, token, "host-1", "me", "observations.read")


def test_consent_grant_expiry_blocks_authentication(tmp_path: Path) -> None:
    store = VaultStore.create(tmp_path / "pilot.vault", "secret")
    store.ensure_person("me")
    trust = TrustManager(store, MemoryKeychain())
    profile, token = trust.pair_agent("Agent", "host-1", "sha256:x", ["me"], ["observations.read"])
    grants = store.active_consent_grants(agent_id=profile.agent_id, person_id="me", scope="observations.read")
    store.state["consent_grants"][grants[0]["id"]]["expires_at"] = "2020-01-01T00:00:00+00:00"
    with pytest.raises(VaultError, match="authentication failed"):
        trust.authenticate(profile.agent_id, token, "host-1", "me", "observations.read")
