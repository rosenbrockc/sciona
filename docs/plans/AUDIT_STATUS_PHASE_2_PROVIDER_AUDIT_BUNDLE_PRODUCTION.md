# Audit Status Phase 2: Provider Audit Bundle Production

## Goal

Create the missing provider-owned audit bundles for the remaining
non-publishable atoms.

## Why This Phase Matters

The current blocker is not the Supabase schema or exporter. It is the absence
of provider-owned audit artifacts for large family slices. This phase is where
publishability should actually be earned.

## Scope

This phase covers provider-owned production of:

- IO specs
- parameter manifests
- technical descriptions
- dejargonized descriptions
- audit rollups
- audit evidence where required by the family workflow

This phase does not cover:

- references registry normalization across providers
- Supabase replay
- manifest export

## Primary Repos

- `../sciona-atoms`
- `../sciona-atoms-signal`
- `../sciona-atoms-bio`
- `../sciona-atoms-fintech`
- `../sciona-atoms-physics`
- `../sciona-atoms-robotics`

## Worker Ownership

Parallel by provider repo and family.

Recommended worker slices:

- physics: `tempo_jl` and neighboring physics families
- fintech: `quantfin` and `institutional_quant_engine`
- bio: `molecular_docking`, `pronto`, and neighboring bio families
- robotics: `rust_robotics`
- core repo: `expansion`, `inference`, `state_estimation`, `numpy`, `scipy`
- signal: `signal_processing`

Each worker owns only its assigned provider families and should not edit shared
seed/backfill code during this phase.

## Tasks

1. Generate or finish provider-owned audit manifests for the assigned family
   buckets.
   - Use real family semantics and real review criteria.

2. Ensure each audited atom has a complete metadata bundle in provider-owned
   files.
   - IO specs
   - parameters
   - technical description
   - dejargonized description
   - audit rollup

3. Keep names canonical.
   - No legacy namespace reintroduction.
   - No new ambiguous aliases.

4. Prefer family-batch production over isolated atom fixes.
   - The target is repeatable family coverage, not one-off row repair.

5. Record intentionally deferred families.
   - If a family cannot be meaningfully audited yet, document why instead of
     creating placeholder coverage.

## Validation

- provider-local tests for any touched families pass
- audit artifact files parse cleanly
- every changed family has a complete audited metadata bundle for its promoted
  atoms
- no worker introduces publication fallbacks or fake audit rows

## Exit Criteria

- each assigned provider bucket has real audit bundle outputs or an explicit,
  documented defer reason
- the resulting artifacts are ready for replay into Supabase without manual row
  edits
