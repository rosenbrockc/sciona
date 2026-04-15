# AGEOA Deprecation Phase 3: Metadata And Seed Cutover

## Goal

Stop Supabase population from treating `ageo-atoms` as an active provider and
seed only canonical `sciona` identities.

## Main Changes

### 1. Provider inventory cutover

- remove `ageo-atoms` from active provider discovery
- remove `ageoa` package assumptions from provider inventory helpers
- keep migration helpers only if they are explicitly marked offline/manual

### 2. Seed logic cutover

- make `supabase_seed.py` reject canonical `ageoa.*` rows
- make benchmark/hyperparam/reference loaders expect canonical `sciona.*`
  identities only
- ensure duplicate detection treats any remaining `ageoa` identity as migration
  debt, not a valid provider

### 3. Backfill logic cutover

- remove `ageoa`-specific namespace anchoring logic once Phase 2 content
  migration is complete
- remove legacy root assumptions in reference/match/uncertainty loaders
- fail closed on files that still point to `ageoa` after the cutover point

### 4. Source config cutover in matcher

- remove `ageo-atoms` from [sources.yml](/Users/conrad/personal/sciona-matcher/sources.yml)
- update any helper scripts that still assume `../ageo-atoms`

## Verification

- clean local Supabase reset and reseed yields zero `ageoa.*` atom rows
- backfills run without depending on `../ageo-atoms`
- duplicate-FQDN and skipped-no-atom counts reflect only canonical provider
  content

## Exit Criteria

- `ageo-atoms` is no longer in the active provider inventory
- Supabase seed and backfill scripts no longer produce or expect canonical
  `ageoa.*` identities

