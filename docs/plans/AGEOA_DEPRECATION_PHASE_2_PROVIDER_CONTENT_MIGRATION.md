# AGEOA Deprecation Phase 2: Provider Content Migration

## Goal

Move the remaining legacy-owned content out of `ageo-atoms` and into canonical
`sciona-atoms*` repos, while rewriting canonical identities away from
`ageoa.*`.

## Current Problem Shape

Remaining `ageoa` content exists in several forms:

- actual registered atoms still emitted from `../ageo-atoms`
- provider-owned metadata in `sciona-atoms*` that still records legacy
  `ageoa...@ageoa/...` identities
- legacy naming artifacts like `computekurtosissqi`
- old consolidated CDG JSON files that still describe `ageoa.*` modules

## Workstreams

### 1. Source migration by provider family

- audit which `ageoa` modules still have no canonical `sciona` home
- port those modules into the correct sibling repo
- delete or quarantine the legacy source once the canonical version exists

### 2. Metadata identity rewrite

- rewrite `references.json`
- rewrite `matches.json`
- rewrite `uncertainty.json`
- rewrite `cdg.json` / `*_cdg.json`
- rewrite benchmark manifests
- rewrite hyperparam and audit manifests where still applicable

### 3. Naming normalization

- rename remaining legacy-generated names that violate the canonical naming
  policy
- update references, tests, probe plans, and benchmark inputs together
- do this in the canonical repo only; do not preserve the old name as a
  published identity

### 4. Legacy repo retirement staging

- once a family is fully migrated, mark the corresponding `ageo-atoms` content
  as no longer authoritative
- keep a migration ledger so later waves can prove nothing remains unmoved

## Parallelization

Safe parallel ownership is by repo family:

- `sciona-atoms`
- `sciona-atoms-signal`
- `sciona-atoms-bio`
- `sciona-atoms-fintech`
- `sciona-atoms-physics`
- `sciona-atoms-ml`
- `sciona-atoms-robotics`

Do not let multiple workers edit the same family metadata files in the same
wave.

## Exit Criteria

- every supported family has a canonical `sciona` owner repo
- provider-owned metadata no longer names `ageoa` as the canonical identity
- remaining `ageo-atoms` content is only temporary migration residue, not an
  active source of truth

