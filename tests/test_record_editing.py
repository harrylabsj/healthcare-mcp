from copy import deepcopy
import json

import pytest

from healthcare.cli import main
from healthcare.control import ControlSession
from healthcare.daemon import LocalHealthDaemon
from healthcare.trust import MemoryKeychain, TrustManager
from healthcare.vault import VaultError, VaultStore


@pytest.fixture
def store(tmp_path):
    vault = VaultStore.create(tmp_path / 'test.vault', 'secret')
    vault.ensure_person('me')
    vault.ensure_person('other')
    vault.record_vital('me', '2026-09-08T08:00', systolic_mmHg=120, diastolic_mmHg=80, note='before')
    vault.record_medication('me', '2026-09-08T08:00', 'example', dose='1')
    vault.record_activity('me', '2026-09-08', 'walk', duration_minutes=20)
    vault.record_emotion('me', '2026-09-08T08:00', 'calm', feelings='before')
    return vault


@pytest.mark.parametrize('kind,collection,changes', [
    ('vital', 'vitals', {'systolic_mmHg': 125, 'note': None}),
    ('medication', 'medications', {'taken': False, 'dose': None}),
    ('activity', 'activities', {'date': '2026-09-07', 'duration_minutes': 30, 'steps': 0}),
    ('emotion', 'emotions', {'name': 'happy', 'feelings': None}),
])
def test_edit_roundtrip_and_identity(store, kind, collection, changes):
    record = next(iter(store.state[collection].values()))
    before = deepcopy(record)
    assert store.update_record('other', kind, record['id'], changes) is False
    assert store.update_record('me', kind, 'missing', changes) is False
    assert store.update_record('me', kind, record['id'], changes)
    reopened = VaultStore.open(store.path, 'secret')
    updated = reopened.state[collection][record['id']]
    for key, value in changes.items():
        assert updated['source' if kind == 'vital' and key == 'note' else key] == value
    for key in set(before) - set(changes) - ({'source'} if kind == 'vital' else set()):
        assert updated[key] == before[key]
    assert len(reopened.state[collection]) == 1
    audit = reopened.state['audit_events'][-1]
    assert audit['event'] == f'{kind}.updated'
    assert 'before' in audit['metadata']


@pytest.mark.parametrize('changes', [{}, [], {'person_id': 'other'}, {'systolic_mmHg': True}, {'weight_kg': float('nan')}, {'measured_at': None}, {'measured_at': 'invalid'}, {'systolic_mmHg': 1.5}, {'systolic_mmHg': None, 'diastolic_mmHg': None}])
def test_invalid_patch_is_atomic(store, changes):
    record_id = store.vitals('me')[0]['id']
    before = deepcopy(store.state)
    with pytest.raises(VaultError):
        store.update_record('me', 'vital', record_id, changes)
    assert store.state == before


def test_duplicate_rejected(store):
    store.record_activity('me', '2026-09-08', 'walk', duration_minutes=30)
    record_id = store.activities('me')[0]['id']
    before = deepcopy(store.state)
    with pytest.raises(VaultError, match='conflicts'):
        store.update_record('me', 'activity', record_id, {'duration_minutes': 30})
    assert store.state == before


def test_observation_requires_control_and_preserves_evidence(store):
    job = store.import_text('me', 'lab.txt', '血肌酐 88.4 umol/L\n', report_date='2026-09-08')
    ControlSession(store).review_job(job.id, accept_all=True)
    original = deepcopy(store.observations('me')[0])
    with pytest.raises(VaultError, match='Control approval'):
        store.update_record('me', 'observation', original['id'], {'value': 90})
    assert ControlSession(store).update_record('me', 'observation', original['id'], {'value': 90})
    updated = VaultStore.open(store.path, 'secret').observations('me')[0]
    assert updated['revision'] == original['revision'] + 1
    assert updated['value'] == 90
    for key in ('id', 'document_id', 'evidence_id', 'raw_value', 'raw_unit', 'created_at'):
        assert updated[key] == original[key]


def test_cli_edit_and_invalid_json(store, monkeypatch, capsys):
    monkeypatch.setenv('HEALTHCARE_PASSPHRASE', 'secret')
    record_id = store.emotions('me')[0]['id']
    args = ['edit-record', '--vault', str(store.path), '--person', 'me', '--type', 'emotion', '--record-id', record_id, '--changes']
    assert main(args + ['{"reflection":"updated"}']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'updated'
    assert VaultStore.open(store.path, 'secret').emotions('me')[0]['reflection'] == 'updated'
    assert main(args + ['{']) == 2


@pytest.mark.parametrize('writable', [False, True])
def test_daemon_write_scope_and_observation_boundary(store, tmp_path, writable):
    trust = TrustManager(store, MemoryKeychain())
    scopes = ['observations.read'] + (['records.write'] if writable else [])
    profile, token = trust.pair_agent('Editor', 'host', 'sha256:abc', ['me'], scopes)
    daemon = LocalHealthDaemon(tmp_path / 'test.sock', trust)
    request = {'id': '1', 'agent_id': profile.agent_id, 'token': token, 'host_id': 'host', 'person_id': 'me', 'method': 'health_update_record', 'params': {'record_type': 'emotion', 'record_id': store.emotions('me')[0]['id'], 'changes': {'name': 'happy'}}}
    if writable:
        assert daemon.dispatch(request)['status'] == 'updated'
        request['params']['record_type'] = 'observation'
    with pytest.raises(VaultError):
        daemon.dispatch(request)


@pytest.mark.parametrize('kind,collection', [
    ('vital', 'vitals'),
    ('medication', 'medications'),
    ('activity', 'activities'),
    ('emotion', 'emotions'),
])
def test_delete_roundtrip_and_audit(store, kind, collection):
    record = next(iter(store.state[collection].values()))
    assert store.delete_record('other', kind, record['id']) is None
    assert store.delete_record('me', kind, 'missing') is None
    removed = store.delete_record('me', kind, record['id'])
    assert removed is not None and removed['id'] == record['id']
    assert record['id'] not in store.state[collection]
    audit = store.state['audit_events'][-1]
    assert audit['event'] == f'{kind}.deleted'
    assert audit['metadata']['removed']['id'] == record['id']


def test_observation_delete_requires_control(store):
    job = store.import_text('me', 'lab.txt', '血肌酐 88.4 umol/L\n', report_date='2026-09-08')
    ControlSession(store).review_job(job.id, accept_all=True)
    record_id = store.observations('me')[0]['id']
    with pytest.raises(VaultError, match='Control approval'):
        store.delete_record('me', 'observation', record_id)
    assert record_id in store.state['observations']
    removed = ControlSession(store).delete_record('me', 'observation', record_id)
    assert removed['id'] == record_id
    assert record_id not in store.state['observations']


def test_cli_delete(store, monkeypatch, capsys):
    monkeypatch.setenv('HEALTHCARE_PASSPHRASE', 'secret')
    record_id = store.vitals('me')[0]['id']
    args = ['delete-record', '--vault', str(store.path), '--person', 'me', '--type', 'vital', '--record-id', record_id]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'deleted'
    assert record_id not in VaultStore.open(store.path, 'secret').state['vitals']
    assert main(args) == 2


@pytest.mark.parametrize('writable', [False, True])
def test_daemon_delete_scope_and_observation_boundary(store, tmp_path, writable):
    trust = TrustManager(store, MemoryKeychain())
    scopes = ['observations.read'] + (['records.write'] if writable else [])
    profile, token = trust.pair_agent('Deleter', 'host', 'sha256:def', ['me'], scopes)
    daemon = LocalHealthDaemon(tmp_path / 'test.sock', trust)
    request = {'id': '1', 'agent_id': profile.agent_id, 'token': token, 'host_id': 'host', 'person_id': 'me', 'method': 'health_delete_record', 'params': {'record_type': 'vital', 'record_id': store.vitals('me')[0]['id'], 'reason': 'duplicate'}}
    if writable:
        assert daemon.dispatch(request)['status'] == 'deleted'
        audit = store.state['audit_events'][-2:]
        assert {event['event'] for event in audit} == {'vital.deleted', 'agent.delete_record'}
        request['params']['record_type'] = 'observation'
    with pytest.raises(VaultError):
        daemon.dispatch(request)
