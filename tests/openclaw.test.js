import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

import register from '../index.js';

const ROOT = new URL('..', import.meta.url).pathname;
const SRC = join(ROOT, 'src');

function cli(args, env = {}) {
  return spawnSync('python3', ['-m', 'healthcare', ...args], {
    encoding: 'utf8',
    cwd: ROOT,
    env: { ...process.env, PYTHONPATH: SRC, HEALTHCARE_PASSPHRASE: 'secret', ...env },
  });
}

function fakeApi(config = {}) {
  const tools = new Map();
  return {
    tools,
    registerTool(spec) {
      tools.set(spec.name, spec);
    },
    config: { plugins: { entries: { healthcare: { config } } } },
  };
}

function setupVault() {
  const dir = mkdtempSync(join(tmpdir(), 'healthcare-openclaw-'));
  const vault = join(dir, 'pilot.vault');
  const report = join(dir, 'report.txt');
  writeFileSync(report, '检验报告\n报告日期：2026-08-01\n血肌酐 88.4 umol/L\n', 'utf8');
  let r = cli(['init', '--vault', vault]);
  assert.equal(r.status, 0, r.stderr);
  r = cli(['import-text', '--vault', vault, '--person', 'me', '--file', report]);
  assert.equal(r.status, 0, r.stderr);
  const job = JSON.parse(r.stdout).job.id;
  r = cli(['review-job', '--vault', vault, '--job', job, '--accept-all']);
  assert.equal(r.status, 0, r.stderr);
  return { dir, vault };
}

test('OpenClaw plugin registers the healthCare tools', () => {
  const api = fakeApi();
  register(api);
  for (const name of ['healthcare_timeline', 'healthcare_evidence', 'healthcare_import_file', 'healthcare_llm_extract', 'healthcare_record_emotion', 'healthcare_record_sleep', 'healthcare_update_medication', 'healthcare_update_record', 'healthcare_recent']) {
    assert.ok(api.tools.has(name), name);
  }
  assert.equal(api.tools.has('healthcare_review_job'), false);
});

test('healthcare_llm_extract refuses without operator consent', () => {
  const api = fakeApi({ vault: '/nonexistent.vault' }); // allow_remote_llm not set
  register(api);
  const result = api.tools.get('healthcare_llm_extract').fn({ image_path: '/tmp/x.png' }, api);
  assert.equal(result.ok, false);
  assert.equal(result.error_type, 'remote_llm_not_consented');
});

test('healthcare_timeline returns confirmed observations from a real vault', () => {
  const { dir, vault } = setupVault();
  const api = fakeApi({ vault, passphrase: 'secret' });
  register(api);
  const result = api.tools.get('healthcare_timeline').fn({ person_id: 'me' }, api);
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.ok(result.result.some((o) => o.field === 'creatinine'));
});

test('healthcare_record_emotion writes all reflection fields to a real vault', () => {
  const { vault } = setupVault();
  const api = fakeApi({ vault, passphrase: 'secret' });
  register(api);
  const result = api.tools.get('healthcare_record_emotion').fn({
    person_id: 'me',
    occurred_at: '2026-08-29T09:30',
    name: '紧张',
    duration_minutes: 25,
    feelings: '胸口发紧',
    reflection: '把下一步写下来',
  }, api);
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.equal(result.type, 'emotion');
  assert.equal(result.status, 'recorded');
});

test('healthcare_record_sleep writes one night and derives the duration', () => {
  const { vault } = setupVault();
  const api = fakeApi({ vault, passphrase: 'secret' });
  register(api);
  const today = `${new Date().getFullYear()}-${String(new Date().getMonth() + 1).padStart(2, '0')}-${String(new Date().getDate()).padStart(2, '0')}`;
  const result = api.tools.get('healthcare_record_sleep').fn({
    person_id: 'me',
    date: today,
    bedtime: '23:30',
    wake_time: '06:45',
    quality: 4,
    note: '夜里醒过一次',
  }, api);
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.equal(result.type, 'sleep');
  assert.equal(result.status, 'recorded');

  const r = cli(['recent', '--vault', vault, '--person', 'me', '--days', '1']);
  assert.equal(r.status, 0, r.stderr);
  const [sleep] = JSON.parse(r.stdout).sleep_records;
  assert.equal(sleep.duration_minutes, 435);
  assert.equal(sleep.bedtime, '23:30');
  assert.equal(sleep.wake_time, '06:45');
  assert.equal(sleep.quality, 4);
  assert.equal(sleep.note, '夜里醒过一次');
});

test('healthcare_update_medication edits an existing medication by id', () => {
  const { vault } = setupVault();
  let r = cli([
    'record-medication', '--vault', vault, '--person', 'me', '--taken-at', `${new Date().getFullYear()}-${String(new Date().getMonth() + 1).padStart(2, '0')}-${String(new Date().getDate()).padStart(2, '0')}T08:00`,
    '--medication', '阿利沙坦酯片', '--dose', '1', '--unit', '片',
  ]);
  assert.equal(r.status, 0, r.stderr);
  r = cli(['recent', '--vault', vault, '--person', 'me', '--days', '1']);
  assert.equal(r.status, 0, r.stderr);
  const medicationId = JSON.parse(r.stdout).medications[0].id;
  const api = fakeApi({ vault, passphrase: 'secret' });
  register(api);
  const result = api.tools.get('healthcare_update_medication').fn({
    person_id: 'me', medication_id: medicationId, medication: '匹伐他汀',
  }, api);
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.equal(result.status, 'updated');
  r = cli(['recent', '--vault', vault, '--person', 'me', '--days', '1']);
  assert.equal(JSON.parse(r.stdout).medications[0].medication, '匹伐他汀');
});

test('healthcare_update_record patches only the selected daily record', () => {
  const { vault } = setupVault();
  const created = cli(['record-emotion', '--vault', vault, '--person', 'me', '--occurred-at', `${new Date().getFullYear()}-${String(new Date().getMonth() + 1).padStart(2, '0')}-${String(new Date().getDate()).padStart(2, '0')}T08:00`, '--name', 'calm', '--feelings', 'before']);
  assert.equal(created.status, 0, created.stderr);
  const exported = cli(['recent', '--vault', vault, '--person', 'me', '--days', '366']);
  assert.equal(exported.status, 0, exported.stderr);
  const api = fakeApi({ vault, passphrase: 'secret' });
  register(api);
  const recent = api.tools.get('healthcare_recent').fn({person_id: 'me', days: 1}, api);
  assert.equal(recent.ok, true, JSON.stringify(recent));
  const original = recent.emotions[0];
  const result = api.tools.get('healthcare_update_record').fn({person_id: 'me', record_type: 'emotion', record_id: original.id, changes: {name: 'happy', feelings: null}}, api);
  assert.equal(result.status, 'updated', JSON.stringify(result));
  const after = JSON.parse(cli(['recent', '--vault', vault, '--person', 'me', '--days', '366']).stdout).emotions;
  assert.equal(after.length, 1);
  assert.equal(after[0].id, original.id);
  assert.equal(after[0].name, 'happy');
  assert.equal(after[0].feelings, null);
});
