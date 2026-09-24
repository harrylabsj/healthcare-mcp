# healthCare

Local-first personal health management with evidence-linked records and scoped MCP tools.

All health data lives in an encrypted Vault (AES-GCM, passphrase-derived key) on the
user's own machine. Agents never touch the Vault directly: they connect to an
authenticated local IPC daemon over a unix socket with a paired-agent token that lives
only in the macOS Keychain. Every mutation is audit-chained; deletions keep a snapshot
behind an approval grant.

- **MCP tools (24)**: vitals, medications, activities, emotions, sleep, encounters,
  diagnosis mentions, medication plans, reminders, trend summaries, timelines,
  visit-summary preparation, source evidence — all scoped per person and per agent.
- **Zero network by default.** An optional, user-configured LLM endpoint
  (`HEALTHCARE_LLM_API_KEY` / `HEALTHCARE_LLM_ENDPOINT`) exists for assisted report
  import; it is off unless explicitly configured.
- **Zero-config Hermes mode**: `healthcare-hermes` bootstraps Vault, daemon and agent
  pairing automatically under `HEALTHCARE_HOME` (the Hermes plugin data directory),
  then serves stdio MCP.
- macOS only (agent pairing uses the macOS Keychain). Python ≥ 3.11.

## Quick start (Hermes plugin)

Install the `healthcare` Hermes plugin — no manual setup needed. Or standalone:

```sh
uvx --from healthCare==0.2.0 healthcare-hermes
```

State layout under `HEALTHCARE_HOME` (default `~/.local/share/healthcare`):

| Path | What |
| --- | --- |
| `family.vault` | encrypted Vault (AES-GCM) |
| `.vault-passphrase` | generated random passphrase, 0600 |
| `daemon.sock` | local IPC socket |
| `agent.json` | non-secret pairing profile (agent_id / host_id / person_id) |

The agent token never touches disk — macOS Keychain only.

## CLI (operator use)

`healthcare init / record-vital / record-medication / record-activity / record-sleep /
record-emotion / record-encounter / record-diagnosis-mention / create-medication-plan /
create-reminder / edit-record / delete-record / export / backup / restore / daemon /
pair-agent / grant-consent / verify-audit …` — see `healthcare --help`.

## License

Apache-2.0. This repository is the public engine extract of a private monorepo;
`src/` and `tests/` are carried over unchanged.
