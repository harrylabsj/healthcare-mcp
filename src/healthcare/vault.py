from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import secrets
import stat
import subprocess
import sys
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, IO

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .models import (
    ActivityRecord,
    ConsentGrant,
    DocumentPage,
    DiagnosisMention,
    Encounter,
    EmotionRecord,
    EvidenceRecord,
    FieldCandidate,
    ImportJob,
    MedicationRecord,
    MedicationPlan,
    Observation,
    PersonProfile,
    ProvenanceRecord,
    ReminderRule,
    ReminderOccurrence,
    SleepRecord,
    SourceDocument,
    VitalMeasurement,
    new_id,
    now_iso,
    serialize,
)
from .object_store import EncryptedObjectStore, ObjectStoreError
from .parser import parse_report, parse_report_date
from .task_mapping import fallback_status_payload


class VaultError(RuntimeError):
    pass


class VaultConflictError(VaultError):
    """A different writer committed after this Vault instance was opened."""


class SessionError(VaultError):
    pass


def _await_session_broker_ready(
    broker: subprocess.Popen[Any],
    ready_path: Path,
    diagnostics: IO[bytes],
    session_file: Path,
) -> None:
    """Detect a broker that dies during startup instead of failing silently.

    The broker touches ``ready_path`` once its socket is bound; if the process
    exits first, surface its stderr tail and remove the orphan session file.
    """
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if ready_path.exists():
            return
        if broker.poll() is not None:
            detail = ""
            try:
                diagnostics.seek(0)
                detail = diagnostics.read().decode("utf-8", errors="replace").strip()[-400:]
            except OSError:
                pass
            for path in (session_file, ready_path):
                try:
                    path.unlink()
                except OSError:
                    pass
            raise SessionError(
                f"ephemeral session broker failed to start: {detail or broker.returncode}"
            )
        time.sleep(0.02)


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    if not passphrase:
        raise VaultError("passphrase must not be empty")
    return PBKDF2HMAC(algorithm=SHA256(), length=32, salt=salt, iterations=600_000).derive(
        passphrase.encode("utf-8")
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def approval_destination_target(destination: Path) -> str:
    return "destination:" + hashlib.sha256(str(destination).encode("utf-8")).hexdigest()


def _secure_mode(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


@contextmanager
def _exclusive_vault_lock(path: Path):
    """Serialize commits across local processes without exposing Vault contents.

    The lock contains no health data and is deliberately adjacent to the Vault,
    so all local CLI and daemon processes contend on the same target.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        _secure_mode(path)
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif os.name == "nt":
            import msvcrt
            import time

            handle.seek(0)
            handle.write("0")
            handle.flush()
            handle.seek(0)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


class VaultStore:
    VERSION = 1
    # Forward migrations: from_version -> callable(state: dict) -> state.
    # A bump to VERSION=N registers MIGRATIONS[N-1] so older vaults are
    # upgraded in order on open. Rollback is always via the last encrypted
    # backup, never an in-place downgrade.
    MIGRATIONS: dict[int, Any] = {}

    def __init__(self, path: Path, passphrase: str, state: dict[str, Any]):
        self.path = path
        self.passphrase = passphrase
        self.state = state
        self.state.setdefault("pages", {})
        self.state.setdefault("identities", {})
        self.state.setdefault("import_idempotency", {})
        self.state.setdefault("import_requests", {})
        self.state.setdefault("approval_grants", {})
        self.state.setdefault("data_revision", 0)
        self.state.setdefault("object_store_key", base64.b64encode(secrets.token_bytes(32)).decode("ascii"))
        self.state.setdefault("agents", {})
        self.state.setdefault("audit_events", [])
        self.state.setdefault("vitals", {})
        self.state.setdefault("medications", {})
        self.state.setdefault("activities", {})
        self.state.setdefault("emotions", {})
        self.state.setdefault("sleep_records", {})
        self.state.setdefault("csv_imports", {})
        self.state.setdefault("consent_grants", {})
        self.state.setdefault("encounters", {})
        self.state.setdefault("diagnoses", {})
        self.state.setdefault("medication_plans", {})
        self.state.setdefault("reminder_rules", {})
        self.state.setdefault("reminder_occurrences", {})
        self._persisted_revision = int(self.state["data_revision"])
        self._defer_saves = False

    @contextmanager
    def deferred_save(self):
        """Batch many record writes into one encrypted save.

        Every record method appends an audit event, and each audit event saves
        the Vault (one PBKDF2 derivation per save). Bulk builders such as the
        demo Vault use this to write once at the end; the revision check in
        ``save`` still runs for that final write.
        """
        if self._defer_saves:
            yield
            return
        self._defer_saves = True
        try:
            yield
        finally:
            self._defer_saves = False
        self.save()

    @classmethod
    def fresh_state(cls) -> dict[str, Any]:
        """The empty state of a newly created Vault (also used to reset a demo Vault in place)."""
        return {
            "vault_version": cls.VERSION,
            "persons": {},
            "documents": {},
            "pages": {},
            "evidence": {},
            "identities": {},
            "jobs": {},
            "candidates": {},
            "observations": {},
            "provenance": {},
            "agents": {},
            "audit_events": [],
            "import_idempotency": {},
            "import_requests": {},
            "approval_grants": {},
            "data_revision": 0,
            "object_store_key": base64.b64encode(secrets.token_bytes(32)).decode("ascii"),
            "vitals": {},
            "medications": {},
            "activities": {},
            "emotions": {},
            "sleep_records": {},
            "csv_imports": {},
            "encounters": {},
            "diagnoses": {},
            "medication_plans": {},
            "reminder_rules": {},
            "reminder_occurrences": {},
        }

    @classmethod
    def create(cls, path: Path, passphrase: str) -> "VaultStore":
        if path.exists():
            raise VaultError(f"vault already exists: {path}")
        store = cls(path, passphrase, cls.fresh_state())
        store.save()
        return store

    @classmethod
    def open(cls, path: Path, passphrase: str) -> "VaultStore":
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            salt = base64.b64decode(envelope["salt"])
            nonce = base64.b64decode(envelope["nonce"])
            ciphertext = base64.b64decode(envelope["ciphertext"])
            key = _derive_key(passphrase, salt)
            plaintext = AESGCM(key).decrypt(nonce, ciphertext, b"healthCare-vault-v1")
            state = json.loads(plaintext.decode("utf-8"))
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise VaultError("unable to open vault") from exc
        except Exception as exc:
            raise VaultError("invalid passphrase or corrupted vault") from exc
        current = int(state.get("vault_version", 0))
        if current > cls.VERSION:
            raise VaultError(f"vault version {current} is newer than supported {cls.VERSION}")
        migrated = False
        while current < cls.VERSION:
            migrator = cls.MIGRATIONS.get(current)
            if migrator is None:
                raise VaultError(f"no migration path from vault version {current}")
            state = migrator(state)
            current += 1
            state["vault_version"] = current
            migrated = True
        store = cls(path, passphrase, state)
        if migrated:
            store.save()
        return store

    def refresh(self) -> None:
        """Re-read the Vault and adopt both its state and its on-disk revision.

        Long-lived holders (the daemon, the session broker) must re-read before
        serving a request. Replacing only ``state`` leaves ``_persisted_revision``
        stale, so every later ``save`` raises ``VaultConflictError`` until the
        process restarts. Refreshing both keeps writes working after any other
        local process (CLI, cron, another agent host) commits.
        """
        fresh = VaultStore.open(self.path, self.passphrase)
        self.state = fresh.state
        self._persisted_revision = fresh._persisted_revision

    def save(self, *, on_disk_passphrase: str | None = None) -> None:
        if self._defer_saves:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(f"{self.path.name}.lock")
        with _exclusive_vault_lock(lock_path):
            current_revision = self._on_disk_revision(on_disk_passphrase)
            if current_revision is not None and current_revision != self._persisted_revision:
                raise VaultConflictError(
                    "Vault changed in another local process; reopen it and retry the action"
                )
            next_revision = self._persisted_revision + 1
            self.state["data_revision"] = next_revision
            salt = secrets.token_bytes(16)
            nonce = secrets.token_bytes(12)
            key = _derive_key(self.passphrase, salt)
            ciphertext = AESGCM(key).encrypt(nonce, _canonical(self.state), b"healthCare-vault-v1")
            envelope = {
                "format": "healthCare.encrypted-vault",
                "version": 1,
                "salt": base64.b64encode(salt).decode("ascii"),
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            }
            temporary = self.path.with_name(f".{self.path.name}.{secrets.token_hex(6)}.tmp")
            temporary.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
            _secure_mode(temporary)
            os.replace(temporary, self.path)
            _secure_mode(self.path)
            self._persisted_revision = next_revision

    def _on_disk_revision(self, passphrase: str | None = None) -> int | None:
        """Read only the encrypted revision while holding the writer lock."""
        if not self.path.exists():
            return None
        try:
            envelope = json.loads(self.path.read_text(encoding="utf-8"))
            key = _derive_key(passphrase or self.passphrase, base64.b64decode(envelope["salt"]))
            plaintext = AESGCM(key).decrypt(
                base64.b64decode(envelope["nonce"]),
                base64.b64decode(envelope["ciphertext"]),
                b"healthCare-vault-v1",
            )
            return int(json.loads(plaintext.decode("utf-8")).get("data_revision", 0))
        except Exception as exc:
            raise VaultError("unable to verify the current Vault revision") from exc

    @property
    def object_store(self) -> EncryptedObjectStore:
        try:
            key = base64.b64decode(self.state["object_store_key"])
            return EncryptedObjectStore(self.path.with_name(f"{self.path.name}.objects"), key)
        except (KeyError, ValueError, ObjectStoreError) as exc:
            raise VaultError("invalid object store key") from exc

    def _audit_backfill(self) -> str:
        """Chain any pre-chain audit events and return the current chain head.

        Events written before the chain existed have no hashes; the first
        append after this change backfills them in order so the whole log is
        verifiable.
        """
        previous = "genesis"
        for event in self.state["audit_events"]:
            if event.get("chain_hash"):
                previous = event["chain_hash"]
                continue
            content = {key: value for key, value in event.items() if key not in ("chain_hash", "prev_hash")}
            event["prev_hash"] = previous
            event["chain_hash"] = _sha256(previous.encode("utf-8") + _canonical(content))
            previous = event["chain_hash"]
        return previous

    def append_audit(
        self,
        event: str,
        actor: str,
        outcome: str,
        person_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a tamper-evident audit event to the hash chain."""
        record = {
            "id": new_id("audit"),
            "event": event,
            "actor": actor,
            "outcome": outcome,
            "person_id": person_id,
            "metadata": metadata or {},
            "created_at": now_iso(),
        }
        previous = self._audit_backfill()
        content = {key: value for key, value in record.items() if key not in ("chain_hash", "prev_hash")}
        record["prev_hash"] = previous
        record["chain_hash"] = _sha256(previous.encode("utf-8") + _canonical(content))
        self.state["audit_events"].append(record)
        self.state["audit_chain_head"] = record["chain_hash"]
        self.save()
        return record

    def verify_audit_chain(self) -> bool:
        """Recompute the audit hash chain and confirm it is unmodified."""
        previous = "genesis"
        for event in self.state["audit_events"]:
            if event.get("prev_hash") != previous or not event.get("chain_hash"):
                return False
            content = {key: value for key, value in event.items() if key not in ("chain_hash", "prev_hash")}
            expected = _sha256(previous.encode("utf-8") + _canonical(content))
            if not hmac.compare_digest(event["chain_hash"], expected):
                return False
            previous = event["chain_hash"]
        return hmac.compare_digest(previous, self.state.get("audit_chain_head", "genesis"))

    def audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1 or limit > 500:
            raise VaultError("limit must be between 1 and 500")
        return list(reversed(self.state["audit_events"][-limit:]))

    def backup_to(
        self,
        destination: Path,
        backup_passphrase: str,
        *,
        control_session_id: str | None = None,
        approval_grant_id: str | None = None,
        expected_revision: int | None = None,
    ) -> Path:
        """Write an independently encrypted backup of the Vault and source objects."""
        self._require_control_approval(
            control_session_id,
            approval_grant_id,
            expected_revision,
            action="backup.create",
            person_id="vault",
            target_ids=[approval_destination_target(destination)],
        )
        self.save()
        source = self.path.read_bytes()
        objects: dict[str, str] = {}
        object_root = self.object_store.root
        if object_root.exists():
            for object_path in object_root.rglob("*.hobj"):
                if not object_path.is_file() or object_path.is_symlink():
                    raise VaultError("object store contains an unsafe entry")
                relative = object_path.relative_to(object_root)
                objects[str(relative)] = base64.b64encode(object_path.read_bytes()).decode("ascii")
        bundle = {
            "format": "healthCare.backup-bundle",
            "version": 2,
            "vault": base64.b64encode(source).decode("ascii"),
            "objects": objects,
        }
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(12)
        key = _derive_key(backup_passphrase, salt)
        ciphertext = AESGCM(key).encrypt(nonce, _canonical(bundle), b"healthCare-backup-v2")
        envelope = {
            "format": "healthCare.encrypted-backup",
            "version": 2,
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(6)}.tmp")
        temporary.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        _secure_mode(temporary)
        os.replace(temporary, destination)
        _secure_mode(destination)
        self.append_audit(
            "backup.created",
            "control",
            "success",
            metadata={
                "backup": str(destination),
                "control_session_id": control_session_id,
                "approval_grant_id": approval_grant_id,
            },
        )
        return destination

    def rotate_passphrase(
        self,
        new_passphrase: str,
        *,
        control_session_id: str | None = None,
        approval_grant_id: str | None = None,
        expected_revision: int | None = None,
    ) -> None:
        self._require_control_approval(
            control_session_id,
            approval_grant_id,
            expected_revision,
            action="vault.key_rotate",
            person_id="vault",
            target_ids=["vault-key"],
        )
        if not new_passphrase:
            raise VaultError("new passphrase must not be empty")
        self.append_audit(
            "vault.key_rotated",
            "control",
            "success",
            metadata={
                "control_session_id": control_session_id,
                "approval_grant_id": approval_grant_id,
            },
        )
        # First commit the audit event under the existing key. Re-encrypting
        # with the new key then verifies the old on-disk revision explicitly.
        old_passphrase = self.passphrase
        self.passphrase = new_passphrase
        self.save(on_disk_passphrase=old_passphrase)

    def crypto_erase(
        self,
        confirmation: str,
        *,
        control_session_id: str | None = None,
        approval_grant_id: str | None = None,
        expected_revision: int | None = None,
    ) -> Path:
        """Destroy this Vault and its adjacent encrypted object store."""
        self._require_control_approval(
            control_session_id,
            approval_grant_id,
            expected_revision,
            action="vault.crypto_erase",
            person_id="vault",
            target_ids=["vault"],
        )
        if confirmation != "ERASE_VAULT":
            raise VaultError("crypto erase requires confirmation ERASE_VAULT")
        object_root = self.object_store.root
        if object_root.exists():
            if object_root.is_symlink() or not object_root.is_dir():
                raise VaultError("object store path is unsafe")
            shutil.rmtree(object_root)
        try:
            self.path.unlink()
        except FileNotFoundError as exc:
            raise VaultError("Vault does not exist") from exc
        tombstone = self.path.with_name(f"{self.path.name}.tombstone")
        tombstone_payload = {
            "format": "healthCare.vault-tombstone",
            "version": 1,
            "operation": "crypto_erase",
            "created_at": now_iso(),
        }
        temporary = tombstone.with_name(f".{tombstone.name}.{secrets.token_hex(6)}.tmp")
        temporary.write_text(json.dumps(tombstone_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _secure_mode(temporary)
        os.replace(temporary, tombstone)
        _secure_mode(tombstone)
        return tombstone

    @classmethod
    def restore_from(
        cls,
        backup: Path,
        destination: Path,
        backup_passphrase: str,
        *,
        confirmation: str | None = None,
        replace: bool = False,
    ) -> Path:
        if confirmation != "RESTORE_VAULT":
            raise VaultError("restore requires confirmation RESTORE_VAULT")
        if destination.exists() and not replace:
            raise VaultError(
                "destination vault already exists; restore to a new path or pass replace=True to overwrite it"
            )
        stale_object_root = destination.with_name(f"{destination.name}.objects")
        if replace and stale_object_root.exists():
            if stale_object_root.is_symlink() or not stale_object_root.is_dir():
                raise VaultError("object store path is unsafe")
            shutil.rmtree(stale_object_root)
        try:
            backup_bytes = backup.read_bytes()
            backup_digest = _sha256(backup_bytes)
            envelope = json.loads(backup_bytes.decode("utf-8"))
            if envelope.get("format") != "healthCare.encrypted-backup" or envelope.get("version") not in {1, 2}:
                raise VaultError("unsupported backup format")
            key = _derive_key(backup_passphrase, base64.b64decode(envelope["salt"]))
            aad = b"healthCare-backup-v2" if envelope["version"] == 2 else b"healthCare-backup-v1"
            source = AESGCM(key).decrypt(
                base64.b64decode(envelope["nonce"]),
                base64.b64decode(envelope["ciphertext"]),
                aad,
            )
            if envelope["version"] == 1:
                vault_source = source
                objects: dict[str, str] = {}
            else:
                bundle = json.loads(source.decode("utf-8"))
                if bundle.get("format") != "healthCare.backup-bundle" or bundle.get("version") != 2:
                    raise VaultError("backup does not contain a valid bundle")
                vault_source = base64.b64decode(bundle["vault"])
                objects = bundle.get("objects", {})
                if not isinstance(objects, dict):
                    raise VaultError("backup object manifest is invalid")
            restored = json.loads(vault_source.decode("utf-8"))
            if restored.get("format") != "healthCare.encrypted-vault":
                raise VaultError("backup does not contain a Vault")
        except VaultError:
            raise
        except Exception as exc:
            raise VaultError("invalid backup or backup passphrase") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(6)}.tmp")
        temporary.write_bytes(vault_source)
        _secure_mode(temporary)
        os.replace(temporary, destination)
        _secure_mode(destination)
        object_root = destination.with_name(f"{destination.name}.objects")
        for relative_name, encoded in objects.items():
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".hobj":
                raise VaultError("backup contains an unsafe object path")
            object_path = object_root / relative
            object_path.parent.mkdir(parents=True, exist_ok=True)
            object_temporary = object_path.with_name(f".{object_path.name}.{secrets.token_hex(6)}.tmp")
            object_temporary.write_bytes(base64.b64decode(encoded))
            _secure_mode(object_temporary)
            os.replace(object_temporary, object_path)
            _secure_mode(object_path)
        receipt = destination.with_name(f"{destination.name}.restore-receipt.json")
        receipt_payload = {
            "format": "healthCare.restore-receipt",
            "version": 1,
            "operation": "restore",
            "backup_sha256": backup_digest,
            "object_count": len(objects),
            "created_at": now_iso(),
        }
        receipt_temporary = receipt.with_name(f".{receipt.name}.{secrets.token_hex(6)}.tmp")
        receipt_temporary.write_text(json.dumps(receipt_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _secure_mode(receipt_temporary)
        os.replace(receipt_temporary, receipt)
        _secure_mode(receipt)
        return destination

    def ensure_person(self, person_id: str, display_name: str | None = None) -> PersonProfile:
        existing = self.state["persons"].get(person_id)
        if existing:
            return PersonProfile(**existing)
        person = PersonProfile(person_id, display_name or person_id)
        self.state["persons"][person_id] = serialize(person)
        self.save()
        return person

    def import_text(
        self,
        person_id: str | None,
        filename: str,
        text: str,
        report_date: str | None = None,
        idempotency_key: str | None = None,
        *,
        media_type: str = "text/plain",
        decoder: str | None = "utf8-text",
        source_sha256: str | None = None,
        source_size_bytes: int | None = None,
        page_specs: list[dict[str, Any]] | None = None,
        source_bytes: bytes | None = None,
    ) -> ImportJob:
        if person_id is not None and person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        if idempotency_key:
            existing_job_id = self.state["import_idempotency"].get(idempotency_key)
            if existing_job_id:
                return ImportJob(**self.state["jobs"][existing_job_id])
        pages = page_specs or [{"page_number": 1, "text": text, "media_type": media_type, "decoder": decoder}]
        # Real reports mix text and scanned pages; an empty page is skipped
        # (the source object still preserves the original), but a document
        # with no readable text at all is rejected.
        pages = [page for page in pages if str(page.get("text", "")).strip()]
        if not pages:
            raise VaultError("decoded document must contain non-empty pages")
        text = "\n".join(str(page["text"]) for page in pages)
        raw = text.encode("utf-8")
        object_id: str | None = None
        if source_bytes is not None:
            if source_sha256 and not hmac.compare_digest(source_sha256, _sha256(source_bytes)):
                raise VaultError("source digest does not match decoded input")
            try:
                object_id = self.object_store.put(source_bytes, media_type).object_id
            except ObjectStoreError as exc:
                raise VaultError("unable to store encrypted source object") from exc
        document = SourceDocument(
            id=new_id("doc"),
            person_id=person_id,
            filename=filename,
            sha256=source_sha256 or _sha256(raw),
            source_text=text,
            report_date=parse_report_date(text, report_date or now_iso()[:10]),
            identity_status="confirmed" if person_id else "quarantined",
            media_type=media_type,
            decoder=decoder,
            source_size_bytes=source_size_bytes or len(source_bytes or raw),
            object_id=object_id,
        )
        self.state["documents"][document.id] = serialize(document)
        page_records: list[tuple[DocumentPage, int, int]] = []
        start_line = 1
        for raw_page in pages:
            page = DocumentPage(
                new_id("page"),
                document.id,
                int(raw_page["page_number"]),
                str(raw_page.get("media_type") or media_type),
                str(raw_page.get("decoder") or decoder) if raw_page.get("decoder") or decoder else None,
            )
            self.state["pages"][page.id] = serialize(page)
            line_count = max(1, len(str(raw_page["text"]).splitlines()))
            page_records.append((page, start_line, start_line + line_count - 1))
            start_line += line_count + 1
        self.state["documents"][document.id]["page_line_ranges"] = [
            [page.page_number, first, last] for page, first, last in page_records
        ]
        candidates: list[str] = []
        for parsed in parse_report(text):
            page_number, page_id = self._page_for_locator(parsed.locator, page_records)
            evidence = EvidenceRecord(
                id=new_id("evidence"),
                document_id=document.id,
                page_number=page_number,
                source_text=parsed.source_line,
                locator=parsed.locator,
                page_id=page_id,
            )
            self.state["evidence"][evidence.id] = serialize(evidence)
            candidate_id = new_id("candidate")
            candidate = FieldCandidate(
                id=candidate_id,
                job_id="pending",
                person_id=person_id,
                field=parsed.field,
                raw_value=parsed.raw_value,
                normalized_value=parsed.value,
                unit=parsed.unit,
                confidence=parsed.confidence,
                evidence_id=evidence.id,
                raw_unit=parsed.raw_unit,
                reference_range_original=parsed.reference_range_original,
                raw_comparator=parsed.raw_comparator,
                precision=parsed.precision,
                mapping_status=parsed.mapping_status,
                value_type=parsed.value_type,
                text_value=parsed.text_value,
            )
            self.state["candidates"][candidate_id] = serialize(candidate)
            candidates.append(candidate_id)
        status = "awaiting_review" if person_id else "awaiting_identity"
        job = ImportJob(
            new_id("job"),
            person_id,
            document.id,
            status,
            candidates,
            state=status,
            idempotency_key=idempotency_key,
        )
        self.state["jobs"][job.id] = serialize(job)
        for candidate_id in candidates:
            self.state["candidates"][candidate_id]["job_id"] = job.id
        if idempotency_key:
            self.state["import_idempotency"][idempotency_key] = job.id
        self.save()
        return ImportJob(**self.state["jobs"][job.id])

    @staticmethod
    def _page_for_locator(locator: str, page_records: list[tuple[DocumentPage, int, int]]) -> tuple[int, str | None]:
        try:
            line_number = int(locator.split(":", 1)[1])
        except (IndexError, ValueError):
            return 1, page_records[0][0].id if page_records else None
        for page, first_line, last_line in page_records:
            if first_line <= line_number <= last_line:
                return page.page_number, page.id
        return page_records[-1][0].page_number, page_records[-1][0].id

    def import_decoded_document(
        self,
        person_id: str | None,
        decoded: Any,
        report_date: str | None = None,
        idempotency_key: str | None = None,
        source_bytes: bytes | None = None,
    ) -> ImportJob:
        """Persist a decoder result while keeping evidence page-aware."""
        if not getattr(decoded, "pages", None):
            raise VaultError("decoded document has no pages")
        page_specs = [
            {
                "page_number": page.page_number,
                "text": page.text,
                "media_type": page.media_type,
                "decoder": page.decoder,
            }
            for page in decoded.pages
        ]
        return self.import_text(
            person_id,
            decoded.filename,
            "\n".join(page["text"] for page in page_specs),
            report_date,
            idempotency_key,
            media_type=decoded.media_type,
            decoder=decoded.decoder,
            source_sha256=decoded.source_sha256,
            source_size_bytes=decoded.source_size_bytes,
            page_specs=page_specs,
            source_bytes=source_bytes,
        )

    def import_llm_extraction(
        self,
        filename: str,
        llm_fields: list[Any],
        *,
        person_id: str | None = None,
        report_date: str | None = None,
        idempotency_key: str | None = None,
        source_bytes: bytes | None = None,
        source_media_type: str = "image/*",
    ) -> ImportJob:
        """Import LLM-extracted fields as quarantine candidates.

        Each ``llm_fields`` item is a dataclass with ``field``, ``value``
        (float or str), ``unit``, ``raw_value``, ``value_type`` and
        ``confidence``. Output is candidates only: identity and confirmation
        are still Control-side, exactly like deterministic extraction.
        """
        if person_id is not None and person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        if idempotency_key and idempotency_key in self.state["import_idempotency"]:
            return ImportJob(**self.state["jobs"][self.state["import_idempotency"][idempotency_key]])
        if source_bytes is not None and not source_bytes:
            raise VaultError("LLM source image must not be empty")
        if report_date is not None:
            try:
                report_date = date.fromisoformat(report_date).isoformat()
            except ValueError as exc:
                raise VaultError("report date must be an ISO date") from exc
        text = "\n".join(f"{item.field} {item.raw_value}" for item in llm_fields) or filename
        object_id: str | None = None
        if source_bytes is not None:
            try:
                object_id = self.object_store.put(source_bytes, source_media_type).object_id
            except ObjectStoreError as exc:
                raise VaultError("unable to store encrypted LLM source object") from exc
        document = SourceDocument(
            id=new_id("doc"),
            person_id=person_id,
            filename=filename,
            sha256=_sha256(source_bytes) if source_bytes is not None else _sha256(text.encode("utf-8")),
            source_text=text,
            # A vision model result cannot establish when the report was made.
            # Keep an unknown date null until a user supplies it in Control.
            report_date=report_date,
            identity_status="confirmed" if person_id else "quarantined",
            media_type="application/llm-extraction",
            decoder="llm-vision",
            source_size_bytes=len(source_bytes) if source_bytes is not None else None,
            object_id=object_id,
        )
        self.state["documents"][document.id] = serialize(document)
        candidate_ids: list[str] = []
        for item in llm_fields:
            evidence = EvidenceRecord(
                id=new_id("evidence"),
                document_id=document.id,
                page_number=1,
                source_text=f"{item.field} {item.raw_value}",
                locator="line:1",
            )
            self.state["evidence"][evidence.id] = serialize(evidence)
            numeric = isinstance(item.value, float)
            mapped = item.unit is not None or not numeric
            candidate_id = new_id("candidate")
            candidate = FieldCandidate(
                id=candidate_id,
                job_id="pending",
                person_id=person_id,
                field=item.field,
                raw_value=str(item.raw_value),
                normalized_value=float(item.value) if numeric else 0.0,
                unit=item.unit,
                confidence=float(item.confidence),
                evidence_id=evidence.id,
                raw_unit=getattr(item, "raw_unit", None) or item.unit,
                mapping_status="mapped" if mapped else "unmapped",
                value_type="numeric" if numeric else "text",
                text_value=str(item.value) if not numeric else None,
            )
            self.state["candidates"][candidate_id] = serialize(candidate)
            candidate_ids.append(candidate_id)
        status = "awaiting_review" if person_id else "awaiting_identity"
        job = ImportJob(
            new_id("job"),
            person_id,
            document.id,
            status,
            candidate_ids,
            state=status,
            idempotency_key=idempotency_key,
        )
        self.state["jobs"][job.id] = serialize(job)
        for candidate_id in candidate_ids:
            self.state["candidates"][candidate_id]["job_id"] = job.id
        if idempotency_key:
            self.state["import_idempotency"][idempotency_key] = job.id
        self.append_audit(
            "document.llm_extracted",
            "control",
            "success",
            person_id=person_id,
            metadata={"document_id": document.id, "candidate_count": len(candidate_ids)},
        )
        return ImportJob(**self.state["jobs"][job.id])

    def _csv_import(
        self,
        record_type: str,
        state_key: str,
        person_id: str,
        records: list[dict[str, Any]],
        idempotency_key: str | None,
        build: Any,
    ) -> int:
        if idempotency_key and idempotency_key in self.state["csv_imports"]:
            return 0
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        for raw in records:
            record = build(raw)
            self.state[state_key][record.id] = serialize(record)
        if idempotency_key:
            self.state["csv_imports"][idempotency_key] = {"type": record_type, "count": len(records)}
        self.append_audit(
            f"csv.{record_type}.imported",
            "control",
            "success",
            person_id=person_id,
            metadata={"count": len(records)},
        )
        return len(records)

    def import_vitals(
        self,
        person_id: str,
        records: list[dict[str, Any]],
        idempotency_key: str | None = None,
    ) -> int:
        def build(raw: dict[str, Any]) -> VitalMeasurement:
            return VitalMeasurement(
                id=new_id("vital"),
                person_id=person_id,
                measured_at=raw["measured_at"],
                systolic_mmHg=raw.get("systolic_mmHg"),
                diastolic_mmHg=raw.get("diastolic_mmHg"),
                heart_rate_bpm=raw.get("heart_rate_bpm"),
                weight_kg=raw.get("weight_kg"),
                context=raw.get("context") or {},
                source=raw.get("note"),
            )

        return self._csv_import("vital", "vitals", person_id, records, idempotency_key, build)

    def import_medications(
        self,
        person_id: str,
        records: list[dict[str, Any]],
        idempotency_key: str | None = None,
    ) -> int:
        def build(raw: dict[str, Any]) -> MedicationRecord:
            return MedicationRecord(
                id=new_id("med"),
                person_id=person_id,
                taken_at=raw["taken_at"],
                medication=raw["medication"],
                dose=raw.get("dose"),
                unit=raw.get("unit"),
                taken=bool(raw.get("taken", True)),
                note=raw.get("note"),
            )

        return self._csv_import("medication", "medications", person_id, records, idempotency_key, build)

    def import_activities(
        self,
        person_id: str,
        records: list[dict[str, Any]],
        idempotency_key: str | None = None,
    ) -> int:
        def build(raw: dict[str, Any]) -> ActivityRecord:
            return ActivityRecord(
                id=new_id("activity"),
                person_id=person_id,
                date=raw["date"],
                activity_type=raw["activity_type"],
                duration_minutes=raw.get("duration_minutes"),
                distance_km=raw.get("distance_km"),
                steps=raw.get("steps"),
                calories=raw.get("calories"),
                note=raw.get("note"),
            )

        return self._csv_import("activity", "activities", person_id, records, idempotency_key, build)

    def record_vital(
        self,
        person_id: str,
        measured_at: str,
        systolic_mmHg: int | None = None,
        diastolic_mmHg: int | None = None,
        heart_rate_bpm: int | None = None,
        weight_kg: float | None = None,
        context: dict[str, Any] | None = None,
        note: str | None = None,
    ) -> bool:
        """Append one daily vital reading; returns False if already recorded."""
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        for existing in self.state["vitals"].values():
            if (
                existing.get("person_id") == person_id
                and existing.get("measured_at") == measured_at
                and existing.get("systolic_mmHg") == systolic_mmHg
                and existing.get("diastolic_mmHg") == diastolic_mmHg
                and existing.get("weight_kg") == weight_kg
            ):
                return False
        record = VitalMeasurement(
            id=new_id("vital"),
            person_id=person_id,
            measured_at=measured_at,
            systolic_mmHg=systolic_mmHg,
            diastolic_mmHg=diastolic_mmHg,
            heart_rate_bpm=heart_rate_bpm,
            weight_kg=weight_kg,
            context=context or {},
            source=note,
        )
        self.state["vitals"][record.id] = serialize(record)
        self.append_audit("vital.recorded", "control", "success", person_id=person_id, metadata={"measured_at": measured_at})
        return True

    def record_medication(
        self,
        person_id: str,
        taken_at: str,
        medication: str,
        dose: str | None = None,
        unit: str | None = None,
        taken: bool = True,
        note: str | None = None,
        medication_plan_id: str | None = None,
    ) -> bool:
        """Append one medication intake event; returns False if already recorded."""
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        if medication_plan_id:
            plan = self.state["medication_plans"].get(medication_plan_id)
            if not plan or plan.get("person_id") != person_id:
                raise VaultError("medication plan is not assigned to this person")
        for existing in self.state["medications"].values():
            if (
                existing.get("person_id") == person_id
                and existing.get("taken_at") == taken_at
                and existing.get("medication") == medication
                and existing.get("dose") == dose
            ):
                return False
        record = MedicationRecord(
            id=new_id("med"),
            person_id=person_id,
            taken_at=taken_at,
            medication=medication,
            dose=dose,
            unit=unit,
            taken=taken,
            note=note,
            medication_plan_id=medication_plan_id,
        )
        self.state["medications"][record.id] = serialize(record)
        self.append_audit("medication.recorded", "control", "success", person_id=person_id, metadata={"medication": medication})
        return True

    def update_record(
        self, person_id: str, record_type: str, record_id: str, changes: dict[str, Any],
        *, control_session_id: str | None = None, approval_grant_id: str | None = None,
        expected_revision: int | None = None,
    ) -> bool:
        """Patch one health record; identity and original evidence are immutable."""
        specs = {
            "vital": ("vitals", {"measured_at", "systolic_mmHg", "diastolic_mmHg", "heart_rate_bpm", "weight_kg", "context", "note"}, ("measured_at", "systolic_mmHg", "diastolic_mmHg", "weight_kg")),
            "medication": ("medications", {"taken_at", "medication", "dose", "unit", "taken", "note", "medication_plan_id"}, ("taken_at", "medication", "dose")),
            "activity": ("activities", {"date", "activity_type", "duration_minutes", "distance_km", "steps", "calories", "note"}, ("date", "activity_type", "duration_minutes")),
            "emotion": ("emotions", {"occurred_at", "name", "duration_minutes", "feelings", "reflection"}, ("occurred_at", "name", "duration_minutes", "feelings", "reflection")),
            "sleep": ("sleep_records", {"date", "bedtime", "wake_time", "duration_minutes", "quality", "note"}, ("date", "bedtime", "wake_time", "duration_minutes")),
            "observation": ("observations", {"field", "value", "unit", "measured_at", "value_type", "text_value"}, ()),
        }
        if not isinstance(record_type, str) or record_type not in specs:
            raise VaultError("unsupported record type")
        if person_id not in self.state["persons"]:
            raise VaultError("unknown person")
        collection, allowed, identity = specs[record_type]
        if not isinstance(changes, dict) or not changes:
            raise VaultError("changes must be a non-empty object")
        if set(changes) - allowed:
            raise VaultError("unsupported record fields: " + ", ".join(sorted(set(changes) - allowed)))
        if not isinstance(record_id, str) or not record_id.strip():
            raise VaultError("record_id is required")
        existing = self.state[collection].get(record_id)
        if existing is None or existing.get("person_id") != person_id:
            return False
        patch = dict(changes)
        if record_type == "vital" and "note" in patch:
            patch["source"] = patch.pop("note")
        candidate = {**existing, **patch}
        required = {"measured_at", "taken_at", "medication", "date", "activity_type", "occurred_at", "name", "field"}
        numeric = {"systolic_mmHg", "diastolic_mmHg", "heart_rate_bpm", "weight_kg", "duration_minutes", "distance_km", "steps", "calories", "value"}
        integers = {"systolic_mmHg", "diastolic_mmHg", "heart_rate_bpm", "steps", "calories"}
        for key, value in patch.items():
            if key in required:
                if not isinstance(value, str) or not value.strip():
                    raise VaultError(f"{key} must be a non-empty string")
                candidate[key] = value.strip()
                if key in {"measured_at", "taken_at", "date", "occurred_at"}:
                    try:
                        datetime.fromisoformat(candidate[key])
                    except ValueError as exc:
                        raise VaultError(f"{key} must be an ISO date or datetime") from exc
            elif key in numeric:
                if value is None and key != "value":
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)):
                    raise VaultError(f"{key} must be a finite number")
                if key in integers and not isinstance(value, int):
                    raise VaultError(f"{key} must be an integer")
                if key != "value" and (value < 0 or (key not in {"steps", "calories", "distance_km"} and value == 0)):
                    raise VaultError(f"{key} must be positive (counts and distance may be zero)")
            elif key == "quality":
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5):
                    raise VaultError("quality must be an integer between 1 and 5")
            elif key == "taken":
                if not isinstance(value, bool):
                    raise VaultError("taken must be a boolean")
            elif key == "context":
                if not isinstance(value, dict):
                    raise VaultError("context must be an object")
                try:
                    json.dumps(value, allow_nan=False)
                except (ValueError, TypeError) as exc:
                    raise VaultError("context must contain valid JSON") from exc
            elif value is not None:
                if not isinstance(value, str):
                    raise VaultError(f"{key} must be a string or null")
                candidate[key] = value.strip() or None
        if record_type == "vital" and not any(candidate.get(k) is not None for k in numeric - {"value", "duration_minutes", "distance_km", "steps", "calories"}):
            raise VaultError("a vital record must retain at least one measurement")
        if record_type == "sleep" and candidate.get("duration_minutes") is None and not (
            candidate.get("bedtime") and candidate.get("wake_time")
        ):
            raise VaultError("a sleep record must retain a duration or both bedtime and wake_time")
        for other_id, other in self.state[collection].items():
            if identity and other_id != record_id and other.get("person_id") == person_id and all(other.get(k) == candidate.get(k) for k in identity):
                raise VaultError("record update conflicts with an existing record")
        if record_type == "observation":
            from .schemas import validate, SchemaError
            if "unit" in changes and "value" not in changes:
                raise VaultError("unit correction requires the converted value")
            if candidate.get("value_type", "numeric") != "numeric" and not candidate.get("text_value"):
                raise VaultError("non-numeric observations require text_value")
            candidate["revision"] = existing.get("revision", 1) + 1
            candidate["verification_status"] = "user_confirmed"
            try:
                validate(candidate, "observation")
            except SchemaError as exc:
                raise VaultError(str(exc)) from exc
            self._require_control_approval(
                control_session_id, approval_grant_id, expected_revision,
                action="observation.correct", person_id=person_id, target_ids=[record_id],
            )
        before = {key: existing.get(key) for key in patch}
        self.state[collection][record_id] = candidate
        self.append_audit(
            f"{record_type}.updated", "control", "success", person_id=person_id,
            metadata={"record_id": record_id, "changed_fields": sorted(changes),
                      "before": before, "after": {key: candidate.get(key) for key in patch}},
        )
        return True

    def delete_record(
        self, person_id: str, record_type: str, record_id: str,
        *, control_session_id: str | None = None, approval_grant_id: str | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any] | None:
        """Physically remove one record; the removed snapshot stays in the audit trail.

        Observation deletion always requires Control approval (same bar as a
        correction). Daily records are gated by the caller (Control grant or the
        daemon's records.write scope). Returns the removed record, or None when
        not found for this person.
        """
        specs = {
            "vital": "vitals",
            "medication": "medications",
            "activity": "activities",
            "emotion": "emotions",
            "sleep": "sleep_records",
            "observation": "observations",
        }
        if not isinstance(record_type, str) or record_type not in specs:
            raise VaultError("unsupported record type")
        if person_id not in self.state["persons"]:
            raise VaultError("unknown person")
        if not isinstance(record_id, str) or not record_id.strip():
            raise VaultError("record_id is required")
        collection = specs[record_type]
        existing = self.state[collection].get(record_id)
        if existing is None or existing.get("person_id") != person_id:
            return None
        if record_type == "observation":
            self._require_control_approval(
                control_session_id, approval_grant_id, expected_revision,
                action="record.delete", person_id=person_id, target_ids=[record_id],
            )
        removed = self.state[collection].pop(record_id)
        self.append_audit(
            f"{record_type}.deleted", "control", "success", person_id=person_id,
            metadata={"record_id": record_id, "removed": removed},
        )
        return removed

    def update_medication(self, person_id: str, medication_id: str, changes: dict[str, Any]) -> bool:
        """Update one medication intake event in place; returns False if not found."""
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        if not changes:
            raise VaultError("at least one medication field must be changed")
        allowed = {"taken_at", "medication", "dose", "unit", "taken", "note"}
        allowed.add("medication_plan_id")
        unknown = set(changes) - allowed
        if unknown:
            raise VaultError(f"unsupported medication fields: {', '.join(sorted(unknown))}")
        existing = self.state["medications"].get(medication_id)
        if existing is None or existing.get("person_id") != person_id:
            return False

        candidate = {**existing, **changes}
        taken_at = str(candidate.get("taken_at", "")).strip()
        medication = str(candidate.get("medication", "")).strip()
        if not taken_at or not medication:
            raise VaultError("taken_at and medication are required")
        candidate["taken_at"] = taken_at
        candidate["medication"] = medication
        if candidate.get("dose") is not None:
            candidate["dose"] = str(candidate["dose"]).strip() or None
        if candidate.get("unit") is not None:
            candidate["unit"] = str(candidate["unit"]).strip() or None
        if candidate.get("note") is not None:
            candidate["note"] = str(candidate["note"]).strip() or None
        candidate["taken"] = bool(candidate.get("taken", True))
        if candidate.get("medication_plan_id"):
            plan = self.state["medication_plans"].get(candidate["medication_plan_id"])
            if not plan or plan.get("person_id") != person_id:
                raise VaultError("medication plan is not assigned to this person")

        for other_id, other in self.state["medications"].items():
            if other_id == medication_id:
                continue
            if (
                other.get("person_id") == person_id
                and other.get("taken_at") == candidate["taken_at"]
                and other.get("medication") == candidate["medication"]
                and other.get("dose") == candidate.get("dose")
            ):
                raise VaultError("medication update conflicts with an existing record")

        existing.update(candidate)
        self.append_audit(
            "medication.updated",
            "control",
            "success",
            person_id=person_id,
            metadata={"medication_id": medication_id, "changed_fields": sorted(changes)},
        )
        return True

    def record_activity(
        self,
        person_id: str,
        date: str,
        activity_type: str,
        duration_minutes: float | None = None,
        distance_km: float | None = None,
        steps: int | None = None,
        calories: int | None = None,
        note: str | None = None,
    ) -> bool:
        """Append one activity session; returns False if already recorded."""
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        for existing in self.state["activities"].values():
            if (
                existing.get("person_id") == person_id
                and existing.get("date") == date
                and existing.get("activity_type") == activity_type
                and existing.get("duration_minutes") == duration_minutes
            ):
                return False
        record = ActivityRecord(
            id=new_id("activity"),
            person_id=person_id,
            date=date,
            activity_type=activity_type,
            duration_minutes=duration_minutes,
            distance_km=distance_km,
            steps=steps,
            calories=calories,
            note=note,
        )
        self.state["activities"][record.id] = serialize(record)
        self.append_audit("activity.recorded", "control", "success", person_id=person_id, metadata={"activity_type": activity_type})
        return True

    def record_emotion(
        self,
        person_id: str,
        occurred_at: str,
        name: str,
        duration_minutes: float | None = None,
        feelings: str | None = None,
        reflection: str | None = None,
        source: str | None = None,
    ) -> bool:
        """Append one emotion record; returns False when the same entry exists."""
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        occurred_at = occurred_at.strip()
        name = name.strip()
        feelings = feelings.strip() if feelings and feelings.strip() else None
        reflection = reflection.strip() if reflection and reflection.strip() else None
        if not occurred_at or not name:
            raise VaultError("occurred_at and name are required")
        if duration_minutes is not None and duration_minutes <= 0:
            raise VaultError("duration_minutes must be positive")
        for existing in self.state["emotions"].values():
            if (
                existing.get("person_id") == person_id
                and existing.get("occurred_at") == occurred_at
                and existing.get("name") == name
                and existing.get("duration_minutes") == duration_minutes
                and existing.get("feelings") == feelings
                and existing.get("reflection") == reflection
            ):
                return False
        record = EmotionRecord(
            id=new_id("emotion"),
            person_id=person_id,
            occurred_at=occurred_at,
            name=name,
            duration_minutes=duration_minutes,
            feelings=feelings,
            reflection=reflection,
            source=source,
        )
        self.state["emotions"][record.id] = serialize(record)
        self.append_audit(
            "emotion.recorded",
            "control",
            "success",
            person_id=person_id,
            metadata={"name": name, "occurred_at": occurred_at},
        )
        return True

    @staticmethod
    def _sleep_duration_minutes(bedtime: str | None, wake_time: str | None) -> float | None:
        """Minutes between bedtime and wake time; a wake time earlier in the
        clock than the bedtime belongs to the next morning. An identical pair
        yields 0 and is rejected as a duration by the caller."""
        if not bedtime or not wake_time:
            return None
        bed = datetime.strptime(bedtime, "%H:%M")
        wake = datetime.strptime(wake_time, "%H:%M")
        minutes = (wake - bed).total_seconds() / 60
        return minutes + 24 * 60 if minutes < 0 else minutes

    def record_sleep(
        self,
        person_id: str,
        date: str,
        duration_minutes: float | None = None,
        bedtime: str | None = None,
        wake_time: str | None = None,
        quality: int | None = None,
        note: str | None = None,
        source: str | None = None,
    ) -> bool:
        """Append one night's sleep; returns False when the same night exists.

        `date` is the wake-up date. Duration may be given directly or derived
        from bedtime plus wake time; at least one of the two is required.
        """
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        date = self._iso_date(date, "date", required=True) or ""
        bedtime = self._clock_time(bedtime, "bedtime")
        wake_time = self._clock_time(wake_time, "wake_time")
        if duration_minutes is None:
            duration_minutes = self._sleep_duration_minutes(bedtime, wake_time)
        if duration_minutes is None:
            raise VaultError("duration_minutes or both bedtime and wake_time are required")
        if isinstance(duration_minutes, bool) or not isinstance(duration_minutes, (int, float)):
            raise VaultError("duration_minutes must be a finite number")
        duration_minutes = float(duration_minutes)
        if not math.isfinite(duration_minutes):
            raise VaultError("duration_minutes must be a finite number")
        if duration_minutes <= 0:
            raise VaultError("duration_minutes must be positive")
        if duration_minutes > 24 * 60:
            raise VaultError("duration_minutes must not exceed 24 hours")
        if isinstance(quality, bool) or (quality is not None and not isinstance(quality, int)):
            raise VaultError("quality must be an integer between 1 and 5")
        if quality is not None and not 1 <= quality <= 5:
            raise VaultError("quality must be an integer between 1 and 5")
        note = note.strip() if note and note.strip() else None
        for existing in self.state["sleep_records"].values():
            if (
                existing.get("person_id") == person_id
                and existing.get("date") == date
                and existing.get("bedtime") == bedtime
                and existing.get("wake_time") == wake_time
                and existing.get("duration_minutes") == duration_minutes
            ):
                return False
        record = SleepRecord(
            id=new_id("sleep"),
            person_id=person_id,
            date=date,
            bedtime=bedtime,
            wake_time=wake_time,
            duration_minutes=duration_minutes,
            quality=quality,
            note=note,
            source=source,
        )
        self.state["sleep_records"][record.id] = serialize(record)
        self.append_audit(
            "sleep.recorded",
            "control",
            "success",
            person_id=person_id,
            metadata={"date": date, "duration_minutes": duration_minutes},
        )
        return True

    def update_activity_note(self, person_id: str, activity_id: str, note: str) -> bool:
        """Update the note of one existing activity record; returns False if not found."""
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        existing = self.state["activities"].get(activity_id)
        if existing is None or existing.get("person_id") != person_id:
            return False
        existing["note"] = note
        self.append_audit("activity.note_updated", "control", "success", person_id=person_id, metadata={"activity_id": activity_id})
        return True

    @staticmethod
    def _iso_date(value: str | None, field_name: str, *, required: bool = False) -> str | None:
        if value is None or not str(value).strip():
            if required:
                raise VaultError(f"{field_name} is required")
            return None
        try:
            return date.fromisoformat(str(value).strip()).isoformat()
        except ValueError as exc:
            raise VaultError(f"{field_name} must be an ISO date") from exc

    @staticmethod
    def _clock_time(value: str | None, field_name: str, *, required: bool = False) -> str | None:
        if value is None or not str(value).strip():
            if required:
                raise VaultError(f"{field_name} is required")
            return None
        try:
            return datetime.strptime(str(value).strip(), "%H:%M").strftime("%H:%M")
        except ValueError as exc:
            raise VaultError(f"{field_name} must use HH:MM local time") from exc

    def _require_person(self, person_id: str) -> None:
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")

    def _require_person_document(self, person_id: str, document_id: str) -> None:
        document = self.state["documents"].get(document_id)
        if not document or document.get("person_id") != person_id:
            raise VaultError("source document is not assigned to this person")

    def create_encounter(
        self,
        person_id: str,
        occurred_on: str,
        *,
        facility: str | None = None,
        department: str | None = None,
        encounter_type: str = "other",
        note: str | None = None,
        document_ids: list[str] | None = None,
    ) -> Encounter:
        self._require_person(person_id)
        allowed_types = {"outpatient", "inpatient", "emergency", "checkup", "telehealth", "other"}
        if encounter_type not in allowed_types:
            raise VaultError("unsupported encounter_type")
        docs = sorted(set(document_ids or []))
        for document_id in docs:
            self._require_person_document(person_id, document_id)
        encounter = Encounter(
            id=new_id("encounter"),
            person_id=person_id,
            occurred_on=self._iso_date(occurred_on, "occurred_on", required=True),
            facility=facility.strip() if facility and facility.strip() else None,
            department=department.strip() if department and department.strip() else None,
            encounter_type=encounter_type,
            note=note.strip() if note and note.strip() else None,
            document_ids=docs,
        )
        self.state["encounters"][encounter.id] = serialize(encounter)
        self.append_audit(
            "encounter.recorded", "control", "success", person_id=person_id,
            metadata={"encounter_id": encounter.id, "document_count": len(docs)},
        )
        return encounter

    def encounters(self, person_id: str, limit: int = 500) -> list[dict[str, Any]]:
        if limit < 1 or limit > 5000:
            raise VaultError("limit must be between 1 and 5000")
        values = [item for item in self.state["encounters"].values() if item.get("person_id") == person_id]
        return sorted(values, key=lambda item: (item["occurred_on"], item["created_at"]))[-limit:]

    def record_diagnosis_mention(
        self,
        person_id: str,
        text: str,
        context: str,
        *,
        occurred_on: str | None = None,
        encounter_id: str | None = None,
        document_id: str | None = None,
        evidence_id: str | None = None,
    ) -> DiagnosisMention:
        self._require_person(person_id)
        if not isinstance(text, str) or not text.strip():
            raise VaultError("diagnosis text is required")
        allowed_contexts = {"current", "suspected", "ruled_out", "history", "family_history", "other"}
        if context not in allowed_contexts:
            raise VaultError("unsupported diagnosis context")
        if encounter_id:
            encounter = self.state["encounters"].get(encounter_id)
            if not encounter or encounter.get("person_id") != person_id:
                raise VaultError("encounter is not assigned to this person")
        if document_id:
            self._require_person_document(person_id, document_id)
        if evidence_id:
            evidence = self.state["evidence"].get(evidence_id)
            if not evidence or evidence.get("document_id") != document_id:
                raise VaultError("evidence does not belong to the source document")
        mention = DiagnosisMention(
            id=new_id("diagnosis"), person_id=person_id, text=text.strip(), context=context,
            occurred_on=self._iso_date(occurred_on, "occurred_on"), encounter_id=encounter_id,
            document_id=document_id, evidence_id=evidence_id,
        )
        self.state["diagnoses"][mention.id] = serialize(mention)
        self.append_audit(
            "diagnosis.mentioned", "control", "success", person_id=person_id,
            metadata={"diagnosis_id": mention.id, "context": context, "has_source": bool(document_id)},
        )
        return mention

    def diagnoses(self, person_id: str, limit: int = 500) -> list[dict[str, Any]]:
        if limit < 1 or limit > 5000:
            raise VaultError("limit must be between 1 and 5000")
        values = [item for item in self.state["diagnoses"].values() if item.get("person_id") == person_id]
        return sorted(values, key=lambda item: (item.get("occurred_on") or "", item["created_at"]))[-limit:]

    def create_medication_plan(
        self,
        person_id: str,
        medication: str,
        schedule: str,
        *,
        dose: str | None = None,
        unit: str | None = None,
        route: str | None = None,
        starts_on: str | None = None,
        ends_on: str | None = None,
        status: str = "draft",
        source_document_id: str | None = None,
        source_evidence_id: str | None = None,
        user_confirmed: bool = False,
    ) -> MedicationPlan:
        self._require_person(person_id)
        if not medication.strip() or not schedule.strip():
            raise VaultError("medication and schedule are required")
        if status not in {"draft", "active", "paused", "ended"}:
            raise VaultError("unsupported medication plan status")
        if status == "active" and not user_confirmed:
            raise VaultError("an active medication plan requires user confirmation")
        start = self._iso_date(starts_on, "starts_on")
        end = self._iso_date(ends_on, "ends_on")
        if start and end and end < start:
            raise VaultError("ends_on cannot be before starts_on")
        if source_document_id:
            self._require_person_document(person_id, source_document_id)
        if source_evidence_id:
            evidence = self.state["evidence"].get(source_evidence_id)
            if not evidence or evidence.get("document_id") != source_document_id:
                raise VaultError("source evidence does not belong to the source document")
        plan = MedicationPlan(
            id=new_id("medplan"), person_id=person_id, medication=medication.strip(), schedule=schedule.strip(),
            dose=dose.strip() if dose and dose.strip() else None,
            unit=unit.strip() if unit and unit.strip() else None,
            route=route.strip() if route and route.strip() else None,
            starts_on=start, ends_on=end, status=status, source_document_id=source_document_id,
            source_evidence_id=source_evidence_id, user_confirmed=user_confirmed,
        )
        self.state["medication_plans"][plan.id] = serialize(plan)
        self.append_audit(
            "medication_plan.created", "control", "success", person_id=person_id,
            metadata={"medication_plan_id": plan.id, "status": status, "user_confirmed": user_confirmed},
        )
        return plan

    def medication_plans(self, person_id: str, limit: int = 500) -> list[dict[str, Any]]:
        if limit < 1 or limit > 5000:
            raise VaultError("limit must be between 1 and 5000")
        values = [item for item in self.state["medication_plans"].values() if item.get("person_id") == person_id]
        return sorted(values, key=lambda item: (item.get("starts_on") or "", item["created_at"]))[-limit:]

    def update_medication_plan(self, person_id: str, plan_id: str, changes: dict[str, Any]) -> bool:
        self._require_person(person_id)
        plan = self.state["medication_plans"].get(plan_id)
        if not plan or plan.get("person_id") != person_id:
            return False
        allowed = {"medication", "schedule", "dose", "unit", "route", "starts_on", "ends_on", "status", "user_confirmed"}
        if not isinstance(changes, dict) or not changes or set(changes) - allowed:
            raise VaultError("unsupported medication plan changes")
        candidate = {**plan, **changes}
        if not str(candidate.get("medication", "")).strip():
            raise VaultError("medication is required")
        candidate["schedule"] = self._clock_time(candidate.get("schedule"), "schedule", required=True)
        candidate["starts_on"] = self._iso_date(candidate.get("starts_on"), "starts_on")
        candidate["ends_on"] = self._iso_date(candidate.get("ends_on"), "ends_on")
        if candidate["starts_on"] and candidate["ends_on"] and candidate["ends_on"] < candidate["starts_on"]:
            raise VaultError("ends_on cannot be before starts_on")
        if candidate.get("status") not in {"draft", "active", "paused", "ended"}:
            raise VaultError("unsupported medication plan status")
        if candidate.get("status") == "active" and not candidate.get("user_confirmed"):
            raise VaultError("an active medication plan requires user confirmation")
        candidate["revision"] = int(plan.get("revision", 1)) + 1
        self.state["medication_plans"][plan_id] = candidate
        self.append_audit(
            "medication_plan.updated", "control", "success", person_id=person_id,
            metadata={"medication_plan_id": plan_id, "changed_fields": sorted(changes)},
        )
        return True

    def create_reminder_rule(
        self,
        person_id: str,
        kind: str,
        schedule: str,
        title: str,
        *,
        medication_plan_id: str | None = None,
        due_on: str | None = None,
        timezone_name: str = "local",
        quiet_start: str | None = None,
        quiet_end: str | None = None,
    ) -> ReminderRule:
        self._require_person(person_id)
        if kind not in {"record", "medication_plan", "followup"}:
            raise VaultError("unsupported reminder kind")
        schedule_time = self._clock_time(schedule, "schedule", required=True)
        if not title.strip():
            raise VaultError("title is required")
        if kind == "medication_plan":
            plan = self.state["medication_plans"].get(medication_plan_id or "")
            if not plan or plan.get("person_id") != person_id:
                raise VaultError("medication reminder requires this person's plan")
            if plan.get("status") != "active" or not plan.get("user_confirmed"):
                raise VaultError("medication reminder requires an active, user-confirmed plan")
        if kind == "followup" and not due_on:
            raise VaultError("followup reminder requires due_on")
        rule = ReminderRule(
            id=new_id("reminder"), person_id=person_id, kind=kind, schedule=schedule_time,
            title=title.strip(), timezone=timezone_name.strip() or "local", medication_plan_id=medication_plan_id,
            due_on=self._iso_date(due_on, "due_on"),
            quiet_start=self._clock_time(quiet_start, "quiet_start"),
            quiet_end=self._clock_time(quiet_end, "quiet_end"),
        )
        self.state["reminder_rules"][rule.id] = serialize(rule)
        self.append_audit(
            "reminder.created", "control", "success", person_id=person_id,
            metadata={"reminder_id": rule.id, "kind": kind},
        )
        return rule

    def reminder_rules(self, person_id: str, include_paused: bool = False) -> list[dict[str, Any]]:
        values = [item for item in self.state["reminder_rules"].values() if item.get("person_id") == person_id]
        if not include_paused:
            values = [item for item in values if item.get("status") == "active"]
        return sorted(values, key=lambda item: (item["created_at"], item["id"]))

    def set_reminder_rule_status(self, person_id: str, rule_id: str, status: str) -> bool:
        if status not in {"active", "paused", "cancelled"}:
            raise VaultError("unsupported reminder status")
        rule = self.state["reminder_rules"].get(rule_id)
        if not rule or rule.get("person_id") != person_id:
            return False
        rule["status"] = status
        self.append_audit(
            "reminder.status_updated", "control", "success", person_id=person_id,
            metadata={"reminder_id": rule_id, "status": status},
        )
        return True

    @staticmethod
    def _inside_quiet_hours(clock: str, start: str | None, end: str | None) -> bool:
        if not start or not end:
            return False
        if start < end:
            return start <= clock < end
        return clock >= start or clock < end

    def due_reminder_occurrences(self, person_id: str, now_at: str | None = None) -> dict[str, Any]:
        """Generate idempotent local due instances; it never delivers a notification."""
        self._require_person(person_id)
        try:
            now = datetime.fromisoformat(now_at) if now_at else datetime.now().astimezone()
        except ValueError as exc:
            raise VaultError("now_at must be an ISO datetime") from exc
        local_date = now.date().isoformat()
        clock = now.strftime("%H:%M")
        generated: list[dict[str, Any]] = []
        suppressed: list[str] = []
        new_count = 0
        for rule in self.reminder_rules(person_id):
            if rule.get("kind") == "followup" and rule.get("due_on") and rule["due_on"] > local_date:
                continue
            if rule["schedule"] > clock:
                continue
            if self._inside_quiet_hours(clock, rule.get("quiet_start"), rule.get("quiet_end")):
                suppressed.append(rule["id"])
                continue
            scheduled_for = f"{local_date}T{rule['schedule']}:00"
            existing = next(
                (
                    item for item in self.state["reminder_occurrences"].values()
                    if item.get("rule_id") == rule["id"] and item.get("scheduled_for") == scheduled_for
                ),
                None,
            )
            if existing:
                generated.append(existing)
                continue
            occurrence = ReminderOccurrence(
                id=new_id("reminder_occurrence"), rule_id=rule["id"], person_id=person_id,
                scheduled_for=scheduled_for,
            )
            self.state["reminder_occurrences"][occurrence.id] = serialize(occurrence)
            generated.append(self.state["reminder_occurrences"][occurrence.id])
            new_count += 1
        if new_count:
            self.append_audit(
                "reminder.occurrences_generated", "scheduler", "success", person_id=person_id,
                metadata={"count": new_count, "scheduled_date": local_date},
            )
        return {"items": generated, "suppressed_rule_ids": suppressed, "evaluated_at": now.isoformat()}

    def complete_reminder_occurrence(self, person_id: str, occurrence_id: str) -> bool:
        occurrence = self.state["reminder_occurrences"].get(occurrence_id)
        if not occurrence or occurrence.get("person_id") != person_id:
            return False
        if occurrence.get("status") != "completed":
            occurrence["status"] = "completed"
            occurrence["completed_at"] = now_iso()
            self.append_audit(
                "reminder.completed", "control", "success", person_id=person_id,
                metadata={"occurrence_id": occurrence_id},
            )
        return True

    def trend_summary(
        self, person_id: str, source: str, field: str, start: str | None = None, end: str | None = None,
    ) -> dict[str, Any]:
        self._require_person(person_id)
        start_date = self._iso_date(start, "start")
        end_date = self._iso_date(end, "end")
        if start_date and end_date and end_date < start_date:
            raise VaultError("end cannot be before start")
        from .trends import observation_trend, vital_trend

        if source == "observation":
            return observation_trend(self.observations(person_id, field), field, start_date, end_date)
        if source == "vital":
            try:
                return vital_trend(self.vitals(person_id, limit=5000), field, start_date, end_date)
            except ValueError as exc:
                raise VaultError(str(exc)) from exc
        raise VaultError("source must be observation or vital")

    def visit_summary(self, person_id: str, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        self._require_person(person_id)
        start_date = self._iso_date(start, "start")
        end_date = self._iso_date(end, "end")
        if start_date and end_date and end_date < start_date:
            raise VaultError("end cannot be before start")
        from .summaries import build_visit_summary

        return build_visit_summary(self, person_id, start_date, end_date)

    def issue_consent_grant(
        self,
        agent_id: str,
        person_id: str,
        scope: str,
        purpose: str,
        ttl_days: int | None = None,
    ) -> ConsentGrant:
        """Issue a standing, revocable consent for an Agent over one person."""
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        if not agent_id.strip() or not scope.strip() or not purpose.strip():
            raise VaultError("agent_id, scope and purpose are required")
        expires_at: str | None = None
        if ttl_days is not None:
            if ttl_days < 1 or ttl_days > 3650:
                raise VaultError("ttl_days must be between 1 and 3650")
            expires_at = (datetime.now(timezone.utc) + timedelta(days=ttl_days)).replace(microsecond=0).isoformat()
        grant = ConsentGrant(
            id=new_id("consent"),
            agent_id=agent_id,
            person_id=person_id,
            scope=scope,
            purpose=purpose,
            expires_at=expires_at,
        )
        self.state["consent_grants"][grant.id] = serialize(grant)
        self.append_audit(
            "consent.granted", "control", "success",
            person_id=person_id,
            metadata={"agent_id": agent_id, "scope": scope},
        )
        return grant

    def revoke_consent_grant(self, grant_id: str) -> ConsentGrant:
        raw = self.state["consent_grants"].get(grant_id)
        if not raw:
            raise VaultError(f"unknown consent grant: {grant_id}")
        if raw.get("status") != "revoked":
            raw["status"] = "revoked"
            raw["revoked_at"] = now_iso()
            self.append_audit(
                "consent.revoked", "control", "success",
                person_id=raw.get("person_id"),
                metadata={"grant_id": grant_id},
            )
        return ConsentGrant(**raw)

    def active_consent_grants(
        self,
        agent_id: str | None = None,
        person_id: str | None = None,
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for raw in self.state["consent_grants"].values():
            if agent_id and raw.get("agent_id") != agent_id:
                continue
            if person_id and raw.get("person_id") != person_id:
                continue
            if scope and raw.get("scope") != scope:
                continue
            if raw.get("status") == "revoked":
                continue
            expires = raw.get("expires_at")
            if expires and datetime.fromisoformat(expires) <= now:
                continue
            items.append(raw)
        return items

    def consent_is_active(self, agent_id: str, person_id: str, scope: str) -> bool:
        return bool(self.active_consent_grants(agent_id=agent_id, person_id=person_id, scope=scope))

    def vitals(self, person_id: str, limit: int = 500) -> list[dict[str, Any]]:
        if limit < 1 or limit > 5000:
            raise VaultError("limit must be between 1 and 5000")
        items = [v for v in self.state["vitals"].values() if v.get("person_id") == person_id]
        return sorted(items, key=lambda v: (v["measured_at"], v["created_at"]))[-limit:]

    def medications(self, person_id: str, limit: int = 500) -> list[dict[str, Any]]:
        if limit < 1 or limit > 5000:
            raise VaultError("limit must be between 1 and 5000")
        items = [v for v in self.state["medications"].values() if v.get("person_id") == person_id]
        return sorted(items, key=lambda v: (v["taken_at"], v["created_at"]))[-limit:]

    def activities(self, person_id: str, limit: int = 500) -> list[dict[str, Any]]:
        if limit < 1 or limit > 5000:
            raise VaultError("limit must be between 1 and 5000")
        items = [v for v in self.state["activities"].values() if v.get("person_id") == person_id]
        return sorted(items, key=lambda v: (v["date"], v["created_at"]))[-limit:]

    def emotions(self, person_id: str, limit: int = 500) -> list[dict[str, Any]]:
        if limit < 1 or limit > 5000:
            raise VaultError("limit must be between 1 and 5000")
        items = [v for v in self.state["emotions"].values() if v.get("person_id") == person_id]
        return sorted(items, key=lambda v: (v["occurred_at"], v["created_at"]))[-limit:]

    def sleep_records(self, person_id: str, limit: int = 500) -> list[dict[str, Any]]:
        if limit < 1 or limit > 5000:
            raise VaultError("limit must be between 1 and 5000")
        items = [v for v in self.state["sleep_records"].values() if v.get("person_id") == person_id]
        return sorted(items, key=lambda v: (v["date"], v["created_at"]))[-limit:]

    def recent(self, person_id: str, days: int = 7) -> dict[str, Any]:
        """Daily vitals, medications, activities, emotions and sleep from recent days.

        Record timestamps are local wall-clock strings ("YYYY-MM-DD[THH:MM]"),
        so the window compares local dates lexicographically. Intended for
        post-record verification: `recent(person, days=1)` answers "did today's
        entry land?" without a plaintext export.
        """
        if days < 1 or days > 366:
            raise VaultError("days must be between 1 and 366")
        cutoff = (datetime.now().date() - timedelta(days=days - 1)).isoformat()

        def within(timestamp: str | None) -> bool:
            return bool(timestamp) and timestamp[:10] >= cutoff

        return {
            "since": cutoff,
            "days": days,
            "vitals": [v for v in self.vitals(person_id, limit=5000) if within(v.get("measured_at"))],
            "medications": [m for m in self.medications(person_id, limit=5000) if within(m.get("taken_at"))],
            "activities": [a for a in self.activities(person_id, limit=5000) if within(a.get("date"))],
            "emotions": [e for e in self.emotions(person_id, limit=5000) if within(e.get("occurred_at"))],
            "sleep_records": [s for s in self.sleep_records(person_id, limit=5000) if within(s.get("date"))],
        }

    def re_extract_document(self, document_id: str) -> ImportJob:
        """Regenerate candidates for a document with the current parser.

        RFC 12.3: re-running a newer parser creates fresh candidates without
        overwriting human-confirmed values. Fields already confirmed for this
        document are skipped, previous unconfirmed candidates are superseded,
        and the new candidates land in a fresh import job.
        """
        document = self.state["documents"].get(document_id)
        if not document:
            raise VaultError(f"unknown document: {document_id}")
        text = str(document.get("source_text", ""))
        if not text.strip():
            raise VaultError("document has no decoded text to re-extract")
        confirmed_fields = {
            observation["field"]
            for observation in self.state["observations"].values()
            if observation.get("document_id") == document_id
        }
        page_records: list[tuple[DocumentPage, int, int]] = []
        for row in document.get("page_line_ranges") or []:
            page = DocumentPage(new_id("page"), document_id, int(row[0]), "text/plain")
            page_records.append((page, int(row[1]), int(row[2])))
        candidate_ids: list[str] = []
        for field in parse_report(text):
            if field.field in confirmed_fields:
                continue
            if page_records:
                page_number, page_id = self._page_for_locator(field.locator, page_records)
            else:
                page_number, page_id = 1, None
            evidence = EvidenceRecord(
                id=new_id("evidence"),
                document_id=document_id,
                page_number=page_number,
                source_text=field.source_line,
                locator=field.locator,
                page_id=page_id,
            )
            self.state["evidence"][evidence.id] = serialize(evidence)
            candidate_id = new_id("candidate")
            candidate = FieldCandidate(
                id=candidate_id,
                job_id="pending",
                person_id=document.get("person_id"),
                field=field.field,
                raw_value=field.raw_value,
                normalized_value=field.value,
                unit=field.unit,
                confidence=field.confidence,
                evidence_id=evidence.id,
                raw_unit=field.raw_unit,
                reference_range_original=field.reference_range_original,
                raw_comparator=field.raw_comparator,
                precision=field.precision,
                mapping_status=field.mapping_status,
                value_type=field.value_type,
                text_value=field.text_value,
            )
            self.state["candidates"][candidate_id] = serialize(candidate)
            candidate_ids.append(candidate_id)
        for raw_job in self.state["jobs"].values():
            if raw_job.get("document_id") != document_id:
                continue
            for candidate_id in raw_job.get("candidate_ids", []):
                candidate = self.state["candidates"].get(candidate_id)
                if candidate and candidate.get("status") != "confirmed":
                    candidate["status"] = "superseded"
        status = "awaiting_review" if document.get("person_id") else "awaiting_identity"
        job = ImportJob(
            new_id("job"),
            document.get("person_id"),
            document_id,
            status,
            candidate_ids,
            state=status,
        )
        self.state["jobs"][job.id] = serialize(job)
        for candidate_id in candidate_ids:
            self.state["candidates"][candidate_id]["job_id"] = job.id
        self.append_audit(
            "document.re_extracted",
            "control",
            "success",
            person_id=document.get("person_id"),
            metadata={"document_id": document_id, "candidate_count": len(candidate_ids)},
        )
        return ImportJob(**self.state["jobs"][job.id])

    def request_document_import(
        self,
        person_id: str,
        requester_id: str,
        file_types: list[str],
        purpose: str,
        ttl_seconds: int = 1800,
    ) -> dict[str, Any]:
        if person_id not in self.state["persons"]:
            raise VaultError("unknown person")
        if not requester_id.strip() or not purpose.strip():
            raise VaultError("requester_id and purpose are required")
        if ttl_seconds < 60 or ttl_seconds > 7200:
            raise VaultError("ttl_seconds must be between 60 and 7200")
        allowed_types = {"text/plain", "application/pdf", "image/*"}
        normalized_types = sorted(set(file_types or ["application/pdf", "image/*"]))
        if set(normalized_types) - allowed_types:
            raise VaultError("unsupported requested file type")
        request_id = new_id("import_request")
        request = {
            "request_id": request_id,
            "requester_id": requester_id,
            "person_id": person_id,
            "file_types": normalized_types,
            "purpose": purpose.strip()[:300],
            "status": "awaiting_control",
            "job_id": None,
            "confirmation_receipt_id": None,
            "created_at": now_iso(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).replace(microsecond=0).isoformat(),
        }
        self.state["import_requests"][request_id] = request
        self.append_audit(
            "document.import_requested",
            requester_id,
            "success",
            person_id=person_id,
            metadata={"request_id": request_id, "file_types": normalized_types},
        )
        return dict(request)

    def import_request_status(self, request_id: str, person_id: str, requester_id: str) -> dict[str, Any]:
        request = self.state["import_requests"].get(request_id)
        if not request or request["person_id"] != person_id or request["requester_id"] != requester_id:
            raise VaultError("import request not found")
        if request["status"] == "awaiting_control" and datetime.fromisoformat(request["expires_at"]) <= datetime.now(timezone.utc):
            request["status"] = "expired"
            self.save()
        return {**dict(request), **fallback_status_payload(request)}

    def _sync_import_request_for_job(self, job_id: str, status: str, receipt_id: str | None = None) -> None:
        for request in self.state["import_requests"].values():
            if request.get("job_id") != job_id:
                continue
            request["status"] = status
            if receipt_id:
                request["confirmation_receipt_id"] = receipt_id

    def fulfill_import_request(
        self,
        request_id: str,
        decoded: Any,
        source_bytes: bytes | None = None,
    ) -> ImportJob:
        request = self.state["import_requests"].get(request_id)
        if not request:
            raise VaultError("import request not found")
        if request["status"] != "awaiting_control":
            raise VaultError("import request is not awaiting Control")
        if datetime.fromisoformat(request["expires_at"]) <= datetime.now(timezone.utc):
            request["status"] = "expired"
            self.save()
            raise VaultError("import request expired")
        if decoded.media_type not in request["file_types"] and not (decoded.media_type.startswith("image/") and "image/*" in request["file_types"]):
            raise VaultError("decoded file type is outside the requested scope")
        job = self.import_decoded_document(
            None,
            decoded,
            idempotency_key=f"request:{request_id}",
            source_bytes=source_bytes,
        )
        request["status"] = "awaiting_identity"
        request["job_id"] = job.id
        self.append_audit(
            "document.import_fulfilled",
            "control",
            "success",
            person_id=request["person_id"],
            metadata={"request_id": request_id, "job_id": job.id},
        )
        return job

    def read_source_object(self, person_id: str, document_id: str) -> bytes:
        document = self.state["documents"].get(document_id)
        if not document or document.get("person_id") != person_id or document.get("identity_status") != "confirmed":
            raise VaultError("source document not found")
        object_id = document.get("object_id")
        if not object_id:
            raise VaultError("source object is not stored")
        try:
            return self.object_store.get(object_id)
        except ObjectStoreError as exc:
            raise VaultError("source object integrity check failed") from exc

    def import_text_unassigned(
        self,
        filename: str,
        text: str,
        report_date: str | None = None,
        idempotency_key: str | None = None,
    ) -> ImportJob:
        """Import into quarantine without assuming the document's patient."""
        return self.import_text(None, filename, text, report_date, idempotency_key)

    def assign_document(
        self,
        document_id: str,
        person_id: str,
        actor: str = "control",
        *,
        control_session_id: str | None = None,
        approval_grant_id: str | None = None,
        expected_revision: int | None = None,
    ) -> ImportJob:
        self._require_control_approval(
            control_session_id,
            approval_grant_id,
            expected_revision,
            action="identity.confirm",
            person_id=person_id,
            target_ids=[document_id],
        )
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        document = self.state["documents"].get(document_id)
        if not document:
            raise VaultError(f"unknown document: {document_id}")
        if document.get("identity_status") == "confirmed":
            if document.get("person_id") != person_id:
                raise VaultError("document identity is already confirmed for another person")
        else:
            document["person_id"] = person_id
            document["identity_status"] = "confirmed"
            document["revision"] = int(document.get("revision", 1)) + 1
            identity_id = new_id("identity")
            self.state["identities"][identity_id] = {
                "id": identity_id,
                "document_id": document_id,
                "person_id": person_id,
                "status": "confirmed",
                "actor": actor,
                "created_at": now_iso(),
                "document_revision": document["revision"],
            }
        matching_job: ImportJob | None = None
        for raw_job in self.state["jobs"].values():
            if raw_job["document_id"] != document_id:
                continue
            raw_job["person_id"] = person_id
            raw_job["state"] = "awaiting_review" if raw_job["status"] == "awaiting_identity" else raw_job.get("state")
            if raw_job["status"] == "awaiting_identity":
                raw_job["status"] = "awaiting_review"
            for candidate_id in raw_job["candidate_ids"]:
                self.state["candidates"][candidate_id]["person_id"] = person_id
            matching_job = ImportJob(**raw_job)
        if matching_job is None:
            raise VaultError("document has no import job")
        self._sync_import_request_for_job(matching_job.id, matching_job.status)
        self.append_audit(
            "document.identity_confirmed",
            actor,
            "success",
            person_id=person_id,
            metadata={
                "document_id": document_id,
                "control_session_id": control_session_id,
                "approval_grant_id": approval_grant_id,
            },
        )
        return matching_job

    def register_approval(self, grant: dict[str, Any]) -> None:
        """Record a single-use Control approval grant inside the Vault state.

        The record persists with the next save; ``_require_control_approval``
        only accepts operations backed by a registered, unconsumed grant whose
        action, person, targets and revision all match.
        """
        self.state["approval_grants"][grant["grant_id"]] = {
            "session_id": grant["session_id"],
            "action": grant["action"],
            "person_id": grant["person_id"],
            "target_ids": list(grant["target_ids"]),
            "expected_revision": int(grant["expected_revision"]),
            "expires_at": grant["expires_at"],
            "consumed": False,
        }

    def _require_control_approval(
        self,
        control_session_id: str | None,
        approval_grant_id: str | None,
        expected_revision: int | None,
        *,
        action: str | None = None,
        person_id: str | None = None,
        target_ids: list[str] | None = None,
    ) -> None:
        if not control_session_id or not approval_grant_id or expected_revision is None:
            raise VaultError("trusted Control approval required")
        record = self.state["approval_grants"].get(approval_grant_id)
        if (
            not record
            or record.get("consumed")
            or record.get("session_id") != control_session_id
            or int(record.get("expected_revision", -1)) != int(expected_revision)
        ):
            raise VaultError("trusted Control approval required")
        if int(expected_revision) != int(self.state.get("data_revision", 0)):
            raise VaultError("CONFLICT_REVISION: approval target changed")
        if action is not None and record.get("action") != action:
            raise VaultError("approval grant does not match the requested action")
        if person_id is not None and record.get("person_id") != person_id:
            raise VaultError("approval grant does not match the requested person")
        if target_ids is not None and [str(item) for item in record.get("target_ids", [])] != sorted(set(target_ids)):
            raise VaultError("approval grant does not match the requested targets")
        try:
            expires = datetime.fromisoformat(str(record.get("expires_at")))
        except ValueError as exc:
            raise VaultError("approval grant has an invalid expiry") from exc
        if expires <= datetime.now(timezone.utc):
            raise VaultError("approval grant expired")
        record["consumed"] = True

    def review_job(
        self,
        job_id: str,
        accept_all: bool = False,
        field: str | None = None,
        value: float | None = None,
        unit: str | None = None,
        *,
        control_session_id: str | None = None,
        approval_grant_id: str | None = None,
        expected_revision: int | None = None,
    ) -> ImportJob:
        if accept_all and (value is not None or unit is not None):
            raise VaultError("value or unit correction cannot be combined with accept-all")
        if (value is not None or unit is not None) and field is None:
            raise VaultError("value or unit correction requires a field")
        if unit is not None and value is None:
            raise VaultError("unit correction requires the converted value")
        raw_job = self.state["jobs"].get(job_id)
        if not raw_job:
            raise VaultError(f"unknown job: {job_id}")
        document = self.state["documents"].get(raw_job["document_id"])
        if not document or document.get("identity_status") != "confirmed":
            raise VaultError("patient identity must be confirmed before review")
        if not document.get("report_date"):
            raise VaultError("report date must be confirmed before review")
        selected = [
            cid
            for cid in raw_job["candidate_ids"]
            if (field is None or self.state["candidates"][cid]["field"] == field)
            and self.state["candidates"][cid].get("status") != "superseded"
        ]
        if not selected:
            raise VaultError("no matching candidate")
        self._require_control_approval(
            control_session_id,
            approval_grant_id,
            expected_revision,
            action="extraction.review",
            person_id=str(raw_job.get("person_id") or ""),
            target_ids=selected,
        )
        job = ImportJob(**raw_job)
        if not job.person_id:
            raise VaultError("patient identity must be confirmed before review")
        confirmed_now = 0
        for candidate_id in selected:
            candidate = self.state["candidates"][candidate_id]
            if candidate.get("status") == "confirmed":
                if value is not None and float(candidate["normalized_value"]) != float(value):
                    raise VaultError("confirmed candidate requires a new correction revision")
                continue
            if value is not None:
                candidate["normalized_value"] = float(value)
            if unit is not None:
                # Control supplies the canonical unit together with the
                # converted value; this is the escape hatch for unmapped units.
                candidate["unit"] = unit
                candidate["mapping_status"] = "mapped"
            if accept_all or value is not None:
                if candidate.get("mapping_status", "mapped") != "mapped":
                    raise VaultError("candidate has an unmapped unit and requires Control correction")
                candidate["status"] = "confirmed"
                candidate["verification_status"] = "user_confirmed"
                observation_id = new_id("obs")
                provenance = ProvenanceRecord(
                    id=new_id("prov"),
                    activity="user_confirmed_extraction",
                    input_id=candidate_id,
                    output_ids=[observation_id],
                    parser="healthcare.numeric-lab-parser",
                    parser_version="0.1.0",
                )
                observation = Observation(
                    id=observation_id,
                    person_id=job.person_id,
                    field=candidate["field"],
                    value=float(candidate["normalized_value"]),
                    unit=candidate["unit"],
                    measured_at=document["report_date"],
                    document_id=job.document_id,
                    evidence_id=candidate["evidence_id"],
                    provenance_id=provenance.id,
                    raw_value=candidate.get("raw_value"),
                    raw_unit=candidate.get("raw_unit") or candidate.get("unit"),
                    reference_range_original=candidate.get("reference_range_original"),
                    raw_comparator=candidate.get("raw_comparator"),
                    mapping_status=candidate.get("mapping_status", "mapped"),
                    value_type=candidate.get("value_type", "numeric"),
                    text_value=candidate.get("text_value"),
                )
                self.state["observations"][observation.id] = serialize(observation)
                self.state["provenance"][provenance.id] = serialize(provenance)
                candidate["observation_id"] = observation_id
                confirmed_now += 1
        statuses = [
            self.state["candidates"][cid]["status"]
            for cid in job.candidate_ids
            if self.state["candidates"][cid].get("status") != "superseded"
        ]
        job.status = "committed" if statuses and all(status == "confirmed" for status in statuses) else "partially_committed"
        job.state = job.status
        job.completed_at = now_iso()
        if confirmed_now:
            receipt = self.append_audit(
                "extraction.confirmed",
                "control",
                "success",
                person_id=job.person_id,
                metadata={
                    "job_id": job.id,
                    "candidate_count": confirmed_now,
                    "control_session_id": control_session_id,
                    "approval_grant_id": approval_grant_id,
                },
            )
            job.confirmation_receipt_id = receipt["id"]
        self._sync_import_request_for_job(job.id, job.status, job.confirmation_receipt_id)
        self.state["jobs"][job.id] = serialize(job)
        self.save()
        return job

    def observations(self, person_id: str, field: str | None = None) -> list[dict[str, Any]]:
        values = [item for item in self.state["observations"].values() if item["person_id"] == person_id]
        if field:
            values = [item for item in values if item["field"] == field]
        return sorted(values, key=lambda item: (item["measured_at"], item["created_at"]))

    def evidence_for(self, person_id: str, observation_id: str) -> dict[str, Any]:
        observation = self.state["observations"].get(observation_id)
        if not observation or observation["person_id"] != person_id:
            raise VaultError("observation not found")
        evidence = self.state["evidence"].get(observation["evidence_id"])
        document = self.state["documents"].get(observation["document_id"])
        if not evidence or not document:
            raise VaultError("evidence chain is incomplete")
        return {"observation": observation, "evidence": evidence, "document": {
            "id": document["id"],
            "filename": document["filename"],
            "sha256": document["sha256"],
            "report_date": document["report_date"],
            "identity_status": document.get("identity_status"),
            "revision": document.get("revision", 1),
            "media_type": document.get("media_type", "text/plain"),
            "decoder": document.get("decoder"),
            "object_id": document.get("object_id"),
        }}

    def source_evidence_for(self, person_id: str, document_id: str, evidence_id: str) -> dict[str, Any]:
        for observation in self.observations(person_id):
            if observation.get("document_id") == document_id and observation.get("evidence_id") == evidence_id:
                return self.evidence_for(person_id, observation["id"])
        raise VaultError("source evidence not found")

    def export_confirmed(
        self,
        person_id: str,
        destination: Path,
        export_passphrase: str | None = None,
        plaintext: bool = False,
        confirm_plaintext: bool = False,
        *,
        control_session_id: str | None = None,
        approval_grant_id: str | None = None,
        expected_revision: int | None = None,
    ) -> Path:
        """Export confirmed facts without exposing the Vault itself."""
        self._require_control_approval(
            control_session_id,
            approval_grant_id,
            expected_revision,
            action="records.export",
            person_id=person_id,
            target_ids=[approval_destination_target(destination)],
        )
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        if plaintext and not confirm_plaintext:
            raise VaultError("plaintext export requires explicit Control confirmation")
        if not plaintext and not export_passphrase:
            raise VaultError("encrypted export requires an export passphrase")
        items = [self.evidence_for(person_id, observation["id"]) for observation in self.observations(person_id)]
        payload = {
            "format": "healthCare.confirmed-export",
            "version": 1,
            "person_id": person_id,
            "data_revision": int(self.state.get("data_revision", 0)),
            "items": items,
        }
        if plaintext:
            output = {
                "format": "healthCare.plaintext-export",
                "version": 1,
                "payload": payload,
            }
        else:
            salt = secrets.token_bytes(16)
            nonce = secrets.token_bytes(12)
            key = _derive_key(export_passphrase or "", salt)
            ciphertext = AESGCM(key).encrypt(nonce, _canonical(payload), b"healthCare-export-v1")
            output = {
                "format": "healthCare.encrypted-export",
                "version": 1,
                "salt": base64.b64encode(salt).decode("ascii"),
                "nonce": base64.b64encode(nonce).decode("ascii"),
                "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{secrets.token_hex(6)}.tmp")
        temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        _secure_mode(temporary)
        os.replace(temporary, destination)
        _secure_mode(destination)
        self.append_audit(
            "export.created",
            "control",
            "success",
            person_id=person_id,
            metadata={
                "format": output["format"],
                "encrypted": not plaintext,
                "control_session_id": control_session_id,
                "approval_grant_id": approval_grant_id,
            },
        )
        return destination

    def search(self, person_id: str, query: str = "") -> list[dict[str, Any]]:
        lowered = query.casefold().strip()
        values = self.observations(person_id)
        if not lowered:
            return values
        return [item for item in values if lowered in item["field"].casefold() or lowered in str(item["value"]).casefold()]

    def issue_session(
        self,
        output: Path,
        person_id: str,
        host_id: str,
        scopes: list[str],
        ttl_seconds: int = 600,
        expected_peer_exec_digest: str | None = None,
    ) -> Path:
        if person_id not in self.state["persons"]:
            raise VaultError(f"unknown person: {person_id}")
        if ttl_seconds < 30 or ttl_seconds > 3600:
            raise VaultError("ttl_seconds must be between 30 and 3600")
        session_id = new_id("session")
        token = secrets.token_urlsafe(32)
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).replace(microsecond=0).isoformat()
        session_stem = secrets.token_hex(10)
        socket_path = Path("/tmp") / f"healthcare-session-{session_stem}.sock"
        ready_path = Path("/tmp") / f"healthcare-session-{session_stem}.ready"
        session = {
            "format": "healthCare.session-capability",
            "version": 2,
            "session_id": session_id,
            "token": token,
            "socket_path": str(socket_path),
            "person_id": person_id,
            "host_id": host_id,
            "scopes": sorted(set(scopes)),
            "issued_at": now_iso(),
            "expires_at": expires_at,
            "expected_peer_exec_digest": expected_peer_exec_digest,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output = output.resolve()
        session["session_file"] = str(output)
        # Create with O_EXCL-style 0600 from the start: the file holds the
        # session token and a chmod-after-write leaves a readable window.
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(session, ensure_ascii=False, indent=2))
        env = os.environ.copy()
        source_root = Path(__file__).resolve().parent.parent
        if source_root.name == "src":
            # Source checkout: make the package importable for the child.
            # An installed environment resolves the package on its own.
            env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(source_root), env.get("PYTHONPATH", ""))))
        bootstrap = {
            "vault_path": str(self.path.resolve()),
            "vault_passphrase": self.passphrase,
            "session_id": session_id,
            "token": token,
            "socket_path": str(socket_path),
            "ready_file": str(ready_path),
            "session_file": str(output),
            "person_id": person_id,
            "host_id": host_id,
            "scopes": sorted(set(scopes)),
            "expires_at": expires_at,
            "expected_peer_exec_digest": expected_peer_exec_digest,
        }
        diagnostics = tempfile.TemporaryFile()
        try:
            broker = subprocess.Popen(
                [sys.executable, "-m", "healthcare.session_broker", "--bootstrap-stdin"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=diagnostics,
                close_fds=True,
                start_new_session=True,
                env=env,
            )
            assert broker.stdin is not None
            broker.stdin.write(json.dumps(bootstrap, ensure_ascii=False).encode("utf-8"))
            broker.stdin.close()
        except (OSError, AssertionError) as exc:
            diagnostics.close()
            try:
                output.unlink()
            except OSError:
                pass
            raise SessionError("unable to start ephemeral session broker") from exc
        try:
            if hasattr(broker, "poll"):
                _await_session_broker_ready(broker, ready_path, diagnostics, output)
        finally:
            diagnostics.close()
        return output


def read_session(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SessionError("invalid session file") from exc
    try:
        path.unlink()
    except OSError as exc:
        raise SessionError("session file must be deleted after consumption") from exc
    if payload.get("format") != "healthCare.session-capability" or payload.get("version") != 2:
        raise SessionError("unsupported session capability")
    try:
        expires = datetime.fromisoformat(payload["expires_at"])
    except (KeyError, ValueError) as exc:
        raise SessionError("invalid session expiry") from exc
    if expires <= datetime.now(timezone.utc):
        raise SessionError("session capability expired")
    required = ("session_id", "token", "socket_path", "person_id", "host_id", "scopes", "session_file")
    if any(not payload.get(key) for key in required):
        raise SessionError("incomplete session capability")
    return payload


def read_encrypted_export(path: Path, passphrase: str) -> dict[str, Any]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("format") != "healthCare.encrypted-export" or envelope.get("version") != 1:
            raise VaultError("unsupported export format")
        key = _derive_key(passphrase, base64.b64decode(envelope["salt"]))
        plaintext = AESGCM(key).decrypt(
            base64.b64decode(envelope["nonce"]),
            base64.b64decode(envelope["ciphertext"]),
            b"healthCare-export-v1",
        )
        payload = json.loads(plaintext.decode("utf-8"))
        if payload.get("format") != "healthCare.confirmed-export":
            raise VaultError("export does not contain confirmed data")
        return payload
    except VaultError:
        raise
    except Exception as exc:
        raise VaultError("invalid export or export passphrase") from exc
