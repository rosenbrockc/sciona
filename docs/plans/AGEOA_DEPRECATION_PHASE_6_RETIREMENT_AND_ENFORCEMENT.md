# AGEOA Deprecation Phase 6: Retirement And Enforcement

## Goal

Make the `ageoa` removal durable so it cannot drift back into the active
system.

## Main Changes

### 1. Enforcement checks

- add CI or local validation that fails if canonical provider inventory
  includes `ageo-atoms`
- fail if new `ageoa.*` atoms appear in Supabase seed dry-runs
- fail if matcher code introduces new active `ageoa` runtime references

### 2. Repository retirement

- remove `../ageo-atoms` from active runbooks
- mark it deprecated or archive it once migration is complete
- keep it only as historical source material if needed

### 3. Post-cutover cleanup

- remove temporary migration notes and one-off conversion helpers
- collapse any transitional docs into the canonical provider runbooks

## Exit Criteria

- new `ageoa` references are blocked by validation
- no active runbook still treats `ageo-atoms` as part of the supported system
- the canonical system can be reset, seeded, backfilled, and manifested without
  touching `ageo-atoms`
