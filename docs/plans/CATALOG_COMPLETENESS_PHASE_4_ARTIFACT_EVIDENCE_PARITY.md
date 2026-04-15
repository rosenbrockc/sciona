# Catalog Completeness Phase 4: Artifact Evidence Parity

## Goal

Bring published CDG artifacts closer to atom parity on verification,
uncertainty, audit evidence, and benchmark support.

## Why This Phase Matters

Macro artifacts should not be selected just because they exist in the catalog.
They should be selected because they carry evidence that they are publishable,
verifiable, and benchmarked in a way comparable to atoms.

## Scope

This phase covers:

- `artifact_cdg_bindings`
- `artifact_verification_matches`
- `artifact_uncertainty_estimates`
- `artifact_audit_evidence`
- artifact benchmark ingestion/document hydration

It assumes concrete artifacts from Phase 3 already exist.

## Primary Repos

- `sciona-matcher`
- `../sciona-atoms`

## Worker Ownership

Recommended split:

- Worker 1: verification bindings + verification matches
- Worker 2: uncertainty + audit evidence
- Worker 3: artifact benchmark ingestion + document hydration

## Tasks

1. Expand artifact verification ingestion.
   - Derive bindings from concrete artifact leaves.
   - Persist verification matches for bound leaves where available.

2. Expand artifact uncertainty/audit evidence ingestion.
   - Populate artifact uncertainty rows from live or derived artifact state.
   - Populate artifact audit evidence and improve `verified_leaf_coverage`.

3. Integrate benchmark support for artifacts.
   - Ensure CDG benchmark rows are fully hydrated into artifact documents and
     retrieval/ranking inputs.

4. Tighten publishability semantics.
   - Verify published CDGs require real evidence rather than only asset review
     metadata.

5. Add focused artifact parity tests.
   - Tests should assert that a concrete published artifact has the expected
     evidence surfaces in Supabase and matcher document reads.

## Validation

- matcher tests covering:
  - skeleton/artifact sync
  - catalog artifact retrieval
  - catalog API document hydration
- local Supabase row checks for artifact-side evidence tables
- rerun stability checks for idempotent artifact sync

## Exit Criteria

- at least one concrete published artifact family has:
  - bindings
  - verification matches
  - uncertainty
  - audit evidence
  - benchmark rows
- artifact publishability is no longer based only on metadata annotations
