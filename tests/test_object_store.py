from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from healthcare.object_store import EncryptedObjectStore, ObjectStoreError


def test_object_store_encrypts_and_round_trips_without_plain_hash_addressing(tmp_path: Path) -> None:
    content = b"synthetic private report bytes"
    store = EncryptedObjectStore(tmp_path / "objects", b"k" * 32)
    ref = store.put(content, "application/pdf")
    assert ref.object_id != hashlib.sha256(content).hexdigest()
    assert store.get(ref.object_id) == content
    assert store.metadata(ref.object_id).media_type == "application/pdf"
    stored = next((tmp_path / "objects").rglob("*.hobj"))
    assert content not in stored.read_bytes()


def test_object_store_detects_ciphertext_tampering(tmp_path: Path) -> None:
    store = EncryptedObjectStore(tmp_path / "objects", b"k" * 32)
    ref = store.put(b"tamper me", "text/plain")
    path = next((tmp_path / "objects").rglob("*.hobj"))
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["ciphertext"] = base64.b64encode(b"tampered").decode("ascii")
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ObjectStoreError):
        store.get(ref.object_id)
