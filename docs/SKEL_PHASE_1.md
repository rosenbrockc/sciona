# Skeleton Proposal Phase 1

## Goal

Add a first-class enrichment proposal model that can represent:

- `primitive_proposal`
- `template_proposal`
- `skeleton_proposal`

This phase must **not** change live enrichment outcomes.

## Scope

In scope:

- proposal data model
- proposal metadata schema
- adapter functions for current primitive/template candidates
- unit tests for proposal construction and validation
- light plumbing in node enrichment boundaries

Out of scope:

- real skeleton proposal generation
- ranking changes
- acceptance logic
- rollback logic
- telemetry/dashboard changes
- Principal-level changes

## Files To Create

- [sciona/architect/proposal_models.py](/Users/conrad/personal/ageo-matcher/sciona/architect/proposal_models.py)
- [tests/test_proposal_models.py](/Users/conrad/personal/ageo-matcher/tests/test_proposal_models.py)

## Files To Update

- [sciona/architect/nodes.py](/Users/conrad/personal/ageo-matcher/sciona/architect/nodes.py)
- optionally [sciona/architect/template_retriever.py](/Users/conrad/personal/ageo-matcher/sciona/architect/template_retriever.py)
- optionally [tests/test_decomposition.py](/Users/conrad/personal/ageo-matcher/tests/test_decomposition.py)

## Implementation Tasks

### 1. Create proposal type definitions

Add a `ProposalType` enum with:

- `primitive`
- `template`
- `skeleton`

### 2. Create the core proposal model

Add `EnrichmentProposal` with fields:

- `proposal_type`
- `source_family`
- `source_label`
- `confidence`
- `compatibility_score`
- `delta_nodes`
- `delta_edges`
- `delta_family_count`
- `delta_concept_type_count`
- `matched_primitive` optional
- `template_fqn` optional
- `skeleton_name` optional
- `payload`

### 3. Add validation and invariants

Primitive proposal:

- may set `matched_primitive`
- must not require `template_fqn` or `skeleton_name`

Template proposal:

- may set `template_fqn`

Skeleton proposal:

- may set `skeleton_name`

General:

- complexity delta fields must always be concrete numbers

### 4. Add constructor helpers

Implement:

- `proposal_from_primitive(...)`
- `proposal_from_template_match(...)`
- `proposal_placeholder_skeleton(...)`

Important:

- `proposal_placeholder_skeleton(...)` exists only so the type is representable
  in tests and future plumbing
- it must not be used in live enrichment yet

### 5. Compute lightweight metadata

Primitive proposals:

- derive `source_family` from primitive name/module prefix
- set `delta_nodes = 0`
- set `delta_edges = 0`

Template proposals:

- derive `source_family` from retrieved exemplar metadata if available
- estimate `delta_nodes` / `delta_edges` from example shape when available

Skeleton placeholder proposals:

- make them representable with explicit type and metadata fields
- do not wire them into live candidate generation yet

### 6. Thread proposal construction into enrichment boundaries

In [sciona/architect/nodes.py](/Users/conrad/personal/ageo-matcher/sciona/architect/nodes.py):

- build proposal objects for primitive candidates
- build a template proposal when a template match is selected

Requirements:

- keep existing control flow unchanged
- do not use proposal objects for selection yet
- do not emit real skeleton proposals yet

### 7. Add tests

Add new unit tests for:

- valid primitive proposal construction
- valid template proposal construction
- valid skeleton placeholder proposal construction
- malformed proposal rejection

Add regression coverage proving:

- `decompose_node()` behavior remains unchanged after proposal plumbing is added

## Guardrails

1. Do not let Phase 1 choose among proposals.
2. Do not generate live skeleton proposals.
3. Do not introduce ranking or complexity penalties yet.
4. Keep proposal objects passive metadata containers in this phase.

## Acceptance Criteria

Phase 1 is complete when:

- the repo has a stable proposal representation for primitive/template/skeleton
- current node enrichment behavior is unchanged
- tests prove proposal objects are valid and usable
- follow-on phases can consume the proposal schema without redesigning it

## Recommended Test Command

```bash
pytest -q tests/test_proposal_models.py tests/test_decomposition.py
```

## Notes For Planner Agent

- Preserve current behavior by default.
- Do not broaden the search surface in this phase.
- Treat this as interface and schema work only.
