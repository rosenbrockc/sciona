# Publishability Review Phase 3: LLM-Assisted Family Review

## Goal

Use LLM workers to draft most of the remaining review bundles so humans only
see residual ambiguity.

## Scope

This phase should produce family-owned draft audit artifacts for atoms that
still fail publication after Phase 2.

## LLM Responsibilities

For each atom or small family cohort, the worker should:

- read the implementation and surrounding family context
- read the deterministic metadata generated in Phase 2
- inspect local references and, when needed, browse upstream docs or papers
- draft:
  - low-jargon English description
  - semantic verdict
  - developer-semantic verdict
  - trust readiness
  - limitations and required actions
  - reference additions or provenance clarifications
  - uncertainty notes

## Required Review Discipline

Workers must:

- cite local evidence or web evidence for factual claims
- distinguish direct evidence from inference
- fail closed when behavior is unclear
- mark unresolved questions explicitly instead of guessing

## Web Usage

Use browsing for:

- upstream algorithm docs
- package documentation
- paper abstracts/official docs
- license/provenance verification

Do not browse casually when the local code and tests already answer the
question.

## Parallelization

This phase is highly parallelizable by family batch.

Recommended ordering:

1. references-only and audit-rollup-only atoms
2. high-volume homogeneous families
3. mixed or abstract helper families

## Output

Provider-owned draft review bundles that are ready either for ingestion or for a
single narrow human question.

## Exit Criteria

- each batch has either a draft review bundle or a bounded human escalation
- most atoms in a batch no longer require open-ended human inspection
