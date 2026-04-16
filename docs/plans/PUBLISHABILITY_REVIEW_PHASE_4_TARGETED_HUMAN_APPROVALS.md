# Publishability Review Phase 4: Targeted Human Approvals

## Goal

Use human time only where the system still cannot safely decide.

## Allowed Human Question Types

- ambiguous provenance
- ambiguous license interpretation
- unresolved semantic boundary or atom split/merge question
- policy decision about publishability under known limitations
- disagreement between deterministic evidence and LLM interpretation

## Forbidden Human Question Types

- “Please review this whole atom”
- “Please check whether this description looks good”
- “Please reread all references and re-explain the algorithm”

Those should already be handled by Phase 2 or Phase 3.

## Required Question Format

Each question must include:

- fqdn
- exact blocker
- evidence summary
- concrete decision options
- recommended answer
- effect on publication state

## Queue Discipline

- deduplicate family-level questions
- prefer one question that unlocks many atoms
- record answers in provider-owned artifacts, not in ad hoc notes

## Parallelization

Human approvals should be triaged centrally, but the underlying family workers
can continue on non-blocked batches while answers are pending.

## Exit Criteria

- all residual human questions are answered or explicitly deferred
- answers are recorded in durable provider-owned review artifacts
