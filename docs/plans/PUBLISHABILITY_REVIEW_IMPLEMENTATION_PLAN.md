# Publishability Review Implementation Plan

## Status

Drafted on April 16, 2026 as the execution plan for bringing the existing atom
catalog up to public-publishable quality without spending human time on every
atom by default.

## Purpose

The remaining gap is no longer schema or export mechanics. It is the quality
and review status of the atoms themselves.

The intended workflow is:

- deterministic checks first
- LLM-assisted review second
- targeted web verification when needed
- narrowly scoped human approvals only for unresolved or high-risk questions

The goal is to make public publication rigorous without forcing a human to
manually inspect every field of every atom.

## Publication Rule

An atom should be publicly publishable only when all of the following are true:

- it has canonical IO specs
- it has parameters where required
- it has a reviewed low-jargon English description
- it has references with sufficient provenance
- it has an approved audit rollup under the stricter publication rule
- its current version has acceptable license metadata for public use

Developer mode may still surface atoms that fail one or more of these checks.
That does not change the public publication bar.

## Working Principle

Treat publishability review as an evidence pipeline, not as an unstructured
manual curation task.

For each atom, the system should try to answer these questions in order:

1. Can deterministic scripts prove or derive the required metadata?
2. Can an LLM review the code, artifacts, and references and draft a credible
   review bundle?
3. Can web search or upstream docs close any remaining factual uncertainty?
4. Is there a narrow residual question that truly requires human signoff?

If the answer to 1-3 is yes, do not ask a human.

## Current Backlog Shape

Reference inventory:

- [UNPUBLISHED_ATOM_AUDIT_STATUS.md](/Users/conrad/personal/sciona-matcher/docs/audit/UNPUBLISHED_ATOM_AUDIT_STATUS.md)
- [unpublished_atom_audit_status.json](/Users/conrad/personal/sciona-matcher/docs/audit/unpublished_atom_audit_status.json)

Current known state:

- `504` atoms in the local catalog
- `60` currently publishable by audit status
- `444` still unpublished
- `43` current version rows have approved license metadata
- public manifests currently include only the small intersection of
  publishable-plus-license-approved atoms

This means the remaining work is two-dimensional:

- audit-quality closure
- license/provenance closure

## Review Model

### LLM-first review

Workers should default to producing:

- family-scoped review bundles
- proposed low-jargon descriptions
- parameter and IO completions
- reference completion proposals
- draft semantic/developer-semantic verdicts
- uncertainty and limitation notes

### Human escalation only for specific unresolved questions

Humans should only be asked questions that look like:

- “Which of these two upstream repos is the authoritative provenance source for
  this atom?”
- “Should this atom be treated as public-safe despite an ambiguous weak
  copyleft dependency?”
- “Does this atom represent one primitive or two semantically distinct
  operations that should be split?”

Humans should not be asked to re-review fully checkable bundles line by line.

### Web verification policy

Browsing is appropriate for:

- upstream project/license confirmation
- checking paper or docs provenance
- verifying API behavior when local code is wrapped from an external package
- confirming that a low-jargon explanation matches the upstream method

Browsing is not a substitute for local evidence, but it is an acceptable
supporting evidence source when local artifacts are incomplete.

## Phase Set

1. [PUBLISHABILITY_REVIEW_PHASE_1_INVENTORY_AND_BATCHING.md](/Users/conrad/personal/sciona-matcher/docs/plans/PUBLISHABILITY_REVIEW_PHASE_1_INVENTORY_AND_BATCHING.md)
2. [PUBLISHABILITY_REVIEW_PHASE_2_DETERMINISTIC_METADATA_AND_PROVENANCE.md](/Users/conrad/personal/sciona-matcher/docs/plans/PUBLISHABILITY_REVIEW_PHASE_2_DETERMINISTIC_METADATA_AND_PROVENANCE.md)
3. [PUBLISHABILITY_REVIEW_PHASE_3_LLM_ASSISTED_FAMILY_REVIEW.md](/Users/conrad/personal/sciona-matcher/docs/plans/PUBLISHABILITY_REVIEW_PHASE_3_LLM_ASSISTED_FAMILY_REVIEW.md)
4. [PUBLISHABILITY_REVIEW_PHASE_4_TARGETED_HUMAN_APPROVALS.md](/Users/conrad/personal/sciona-matcher/docs/plans/PUBLISHABILITY_REVIEW_PHASE_4_TARGETED_HUMAN_APPROVALS.md)
5. [PUBLISHABILITY_REVIEW_PHASE_5_SUPABASE_RATCHET_AND_LICENSE_INTERSECTION.md](/Users/conrad/personal/sciona-matcher/docs/plans/PUBLISHABILITY_REVIEW_PHASE_5_SUPABASE_RATCHET_AND_LICENSE_INTERSECTION.md)
6. [PUBLISHABILITY_REVIEW_PHASE_6_BLESSED_PUBLICATION_BASELINE.md](/Users/conrad/personal/sciona-matcher/docs/plans/PUBLISHABILITY_REVIEW_PHASE_6_BLESSED_PUBLICATION_BASELINE.md)

## Parallelization

The work is highly parallelizable after the initial batching step.

### Wave 0

- Phase 1

One integrator should own the initial batching and target-set definition.

### Wave 1

- Phase 2
- Phase 3

These can run in parallel by provider/family once the batches are frozen.

Recommended ownership split:

- Worker A: `signal_processing` and `expansion.signal_*`
- Worker B: `state_estimation` and `expansion.{kalman_filter,particle_filter,sequential_filter}`
- Worker C: `bio` / `alphafold`, `mint`, `molecular_docking`
- Worker D: `fintech` / `quantfin`, `institutional_quant_engine`, `hftbacktest`
- Worker E: `physics`, `numpy`, `scipy`
- Worker F: `robotics`, `inference`, `dynamic_programming`, `ml`
- Worker G: license/provenance and reference normalization across families

### Wave 2

- Phase 4

Human approvals should be integrated in a single queue so the questions stay
small, specific, and deduplicated.

### Wave 3

- Phase 5
- Phase 6

One integrator should own the replay/export ratchet and the blessed baseline.

## Human Question Protocol

Every human question should include:

- atom fqdn
- exact blocker
- evidence already gathered
- 2-3 concrete decision options
- recommended default
- effect of each answer on publication

Example:

- Atom: `sciona.atoms.fintech...`
- Blocker: ambiguous upstream provenance
- Evidence: local wrapper imports package `X`, references point to repos `A`
  and `B`, package metadata names `A` as canonical
- Recommended: treat `A` as canonical upstream

This keeps human review bounded and auditable.

## Exit Criteria

The plan is complete only when all of the following are true:

- every unpublished atom is assigned to a tracked family batch
- most metadata completion is produced through deterministic or LLM-assisted
  review, not manual field entry
- human approvals are limited to genuinely ambiguous or policy-bearing
  questions
- public publishability rises without weakening the stricter audit or license
  gates
- a clean replay/export produces a materially larger public manifest and a
  smaller, well-classified unpublished backlog
