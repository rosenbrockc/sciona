# Catalog Completeness Phase 1: Atom Benchmark Coverage

## Goal

Complete the benchmark coverage loop for concrete atoms so the Supabase catalog
and exported manifest carry useful comparative priors, not just CDG benchmark
rows.

## Why This Phase Matters

The benchmark manifest path exists and CDG benchmark rows already seed cleanly,
but the atom-side coverage is still too sparse. That weakens:

- manifest priors
- catalog ranking
- artifact-versus-atom comparisons inside the same benchmark suite
- trust in macro artifacts as "better than reinventing the wheel" selections

## Scope

This phase covers:

- provider-owned `benchmark_results.json` completion for concrete atom baselines
- matcher-side deterministic result generation where the benchmark data is
  derived in code
- `sciona-atoms` seeding support needed to land atom rows cleanly in
  `atom_benchmarks`

It does not cover artifact evidence parity for CDGs beyond benchmark rows. That
belongs to Phase 4.

## Primary Repos

- `../sciona-atoms`
- `../sciona-atoms-signal`
- `sciona-matcher`

## Worker Ownership

Recommended split:

- Worker 1: `signal.event_rate.ecg.v1` atom baselines
- Worker 2: `state_estimation.kalman.synthetic_tracking.v1` atom baselines
- Worker 3: `state_estimation.particle.synthetic_tracking.v1` atom baselines
- Worker 4: integrator for benchmark result generation + seed validation

## Tasks

1. Inventory benchmarkable atom candidates per suite.
   - Confirm which concrete registered atoms should be compared directly inside
     each existing suite.
   - Reject abstract helper/diagnostic nodes that do not represent a meaningful
     top-level benchmark competitor.

2. Add concrete atom result rows to provider-owned benchmark manifests.
   - Extend `benchmark_results.json` in the owning provider repo(s).
   - Use canonical `(artifact_fqdn, content_hash)` identity.
   - Keep suite IDs unchanged unless the suite definition itself is wrong.

3. Extend deterministic result generation where needed.
   - Update [provider_results.py](/Users/conrad/personal/sciona-matcher/sciona/benchmarks/provider_results.py)
     if a suite’s result generation still only emits CDG rows.
   - Ensure atom results are deterministic and replayable.

4. Validate seeding into Supabase.
   - Replay [supabase_seed.py](</Users/conrad/personal/sciona-atoms/src/sciona/atoms/supabase_seed.py>)
     against a local stack.
   - Confirm `atom_benchmarks` receives rows and remains idempotent.

5. Verify matcher-side readers.
   - Confirm benchmark priors and catalog document reads remain coherent when
     atom and CDG rows coexist in the same suite family.

## Validation

- focused provider seed tests in `../sciona-atoms/tests/test_supabase_seed.py`
- focused benchmark generation tests in matcher
- live local check:
  - non-zero `benchmark_suites`
  - non-zero `benchmark_result_rows`
  - non-zero `atom_benchmark_rows`
  - stable row counts on rerun

## Exit Criteria

- the three current benchmark suites have meaningful concrete atom baselines
- `atom_benchmark_rows > 0` in the clean local replay
- rerunning the seed does not duplicate rows
- matcher benchmark readers remain green
