# Publishability Review Phase 2: Deterministic Metadata And Provenance

## Goal

Exhaust every deterministic route to publishability before asking an LLM or a
human to judge anything.

## Scope

This phase should fill or verify:

- IO specs
- parameters
- description source stubs
- reference registry mappings
- upstream provenance
- license provenance

without asserting final human-style approval.

## Tasks

1. Re-run and harden family-specific deterministic extractors.
2. Normalize provider references and provenance paths.
3. Resolve repo-root and upstream license files where possible.
4. Emit family-level metadata completion reports:
   - fields completed deterministically
   - fields still ambiguous
   - fields blocked on missing source artifacts
5. Fail closed where deterministic evidence is contradictory.

## Worker Rules

- Do not hand-approve semantic quality in this phase.
- Do not weaken audit gates.
- Do capture exact evidence locations so later LLM review can cite them.

## Parallelization

Parallelize by provider-family batch.

Good parallel slices:

- `signal_processing`
- `state_estimation`
- `bio.mint`
- `bio.molecular_docking`
- `fintech.quantfin`
- `fintech.institutional_quant_engine`
- `physics.tempo_jl`
- `robotics.rust_robotics`

## Human Escalation

Only escalate if deterministic evidence conflicts in a way that changes legal
or provenance interpretation.

## Exit Criteria

- all deterministic completions are applied or proposed
- remaining gaps are explicitly labeled as semantic, provenance, or policy
  questions
