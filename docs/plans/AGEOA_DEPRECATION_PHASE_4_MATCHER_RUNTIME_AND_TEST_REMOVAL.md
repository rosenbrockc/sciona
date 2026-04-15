# AGEOA Deprecation Phase 4: Matcher Runtime And Test Removal

## Goal

Remove `ageoa` as a matcher runtime concept, not just as a provider-content
concept.

## Main Changes

### 1. Runtime and config cleanup

- remove `ageoa` fallback assumptions from ghost simulation
- remove `ageoa` package references from source-catalog helpers
- remove `ageo-atoms` default paths from scripts and config
- update CLI/help text so canonical examples use `sciona.*`

### 2. Test cleanup

- rewrite tests that still treat `ageoa` as the canonical provider namespace
- convert fixtures to canonical `sciona.*` examples
- delete tests that only validate legacy `ageoa` compatibility

### 3. Documentation cleanup

- rewrite architectural docs that describe `ageoa` as the reusable atom store
- archive or mark historical documents that still reference `ageo-atoms`
- ensure active docs point to `sciona-atoms*` instead

## Parallelization

This phase can be split into:

- runtime/config worker
- tests/fixtures worker
- docs worker

But one owner should keep control of shared fixtures and `sources.yml`.

## Exit Criteria

- matcher runtime paths no longer require or prefer `ageoa`
- matcher tests no longer depend on `ageoa` compatibility
- active docs no longer instruct users to use `ageo-atoms` as the canonical
  atom repo

