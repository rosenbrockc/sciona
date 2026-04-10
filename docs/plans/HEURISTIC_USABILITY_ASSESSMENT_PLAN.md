# Heuristic Usability Assessment Plan

## Status

Drafted on April 9, 2026 after the first multi-night heuristic cohort reruns
for ECG heart-rate optimization and the realization that the current notion of
"usable night" is still too procedural, too transient, and not yet grounded in
first-class heuristic evidence.

This document is intentionally high-level. It is the ground-truth planning
reference for making usability assessment a deterministic, audited, cross-family
part of the framework. Detailed implementation plans should be audited against
this document.

## Purpose

Sciona now has:

- a canonical heuristic evidence layer
- family heuristic registries
- heuristic-driven proposal guidance
- heuristic outcome memory
- the beginnings of multi-night cohort scoring

That is necessary, but it is not sufficient.

The framework still relies on a weak temporary proxy for deciding whether a
dataset slice, night, or benchmark member is "usable." In practice, the current
logic still depends too much on:

- whether a run happened to complete
- whether a loss was emitted
- whether some heuristics were present
- family-local assumptions hidden in runner behavior

That is not the right contract.

Usability should be a first-class, deterministic assessment derived from
audited heuristic evidence. It should be persisted into long-term memory so the
platform can learn, over time and across build contexts, what kinds of data are
usable for which purposes and why.

## Why This Is Needed

The ECG heart-rate cohort reruns made the gap concrete.

The new cohort scorer can evaluate multiple nights, weight recurring heuristic
patterns, and run in bounded parallel batches. But the framework still lacks a
proper, reusable definition of whether a night is:

- usable for heuristic guidance
- usable for benchmark scoring
- usable for final benchmark acceptance

That distinction matters.

Some dataset members may:

- have enough signal and telemetry to teach the framework about nuisance
  patterns
- be unsuitable for loss-based comparison because required references are
  missing
- be invalid for final benchmark conclusions because mandatory inputs are absent

Without a first-class usability system, Sciona risks:

- conflating execution success with data suitability
- discarding informative nights that should still influence heuristic memory
- over-weighting corner-case failures
- baking ad hoc per-family data filters into runtime code

## Core Thesis

Usability should be a heuristic-derived, audited assessment, not an incidental
side effect of evaluation.

The framework should reason in terms like:

- required evidence present
- coverage sufficient
- timing context coherent
- nuisance structure tolerable
- alignment acceptable
- output plausibility stable
- confidence fragmented

These are cross-family observations about whether a dataset member can support a
given kind of algorithmic decision. Family registries may interpret them
locally, but the shared usability interface must remain de-jargonized and
portable.

## Design Intent

The future system should let Sciona say:

- here are the heuristic observations produced for this dataset member
- here is the auditable usability assessment derived from those observations
- here is whether the member is usable for guidance, scoring, or final
  acceptance
- here is the evidence and rationale behind that decision
- here is how similar decisions have performed historically across runs and
  environments

That is the intended bridge between runtime evidence and long-term platform
learning about data suitability.

## Conceptual Model

Usability assessment should sit above heuristic evidence and below search,
benchmark, and memory systems.

### 1. Runtime and diagnostic atoms produce heuristic evidence

Execution summaries, diagnostic atoms, and audited heuristic-producing atom
outputs generate canonical heuristic observations.

### 2. A family-aware assessor evaluates usability

A deterministic assessor combines:

- canonical heuristics
- family registry rules
- required evidence contracts
- benchmark/task intent

to produce a structured usability assessment.

### 3. The assessment is multi-scope

A dataset member should not collapse to one binary label. The system should
distinguish at least:

- `usable_for_guidance`
- `usable_for_scoring`
- `usable_for_final_benchmark`

### 4. Proposal, benchmarking, and cohort selection consume the assessment

Different subsystems use different scopes:

- cohort guidance may include members unsuitable for final scoring
- scoring should reject members lacking required measurable outputs
- final benchmark summaries should use only fully admissible members

### 5. Outcome memory persists both evidence and decisions

The platform records:

- heuristic signature
- usability assessment
- rationale
- actions attempted
- observed outcome
- build/runtime context

so Sciona can develop stronger deterministic priors over time.

## Cross-Family Principles

The usability layer must preserve the repository's cross-family and
cross-disciplinary purpose.

### 1. Shared usability language must be de-jargonized

Do not encode family-local labels like "bad ECG night" or "RR quality too low"
as canonical concepts.

Prefer cross-family concepts such as:

- `required_input_missing`
- `required_reference_missing`
- `coverage_insufficient`
- `timing_context_incoherent`
- `alignment_error`
- `quality_instability`
- `plausibility_fragmentation`
- `output_density_collapse`

### 2. Families may specialize interpretation, not redefine meaning

Families may specify:

- which heuristics are mandatory
- what thresholds matter locally
- which usability scopes they affect
- what actions are allowed under degraded usability

But they should not redefine the shared meaning of a heuristic or usability
state.

### 3. Usability must remain evidence-backed and auditable

Every usability decision should be reconstructable from:

- the observed heuristics
- the family rules consulted
- the required contracts in force
- the confidence and uncertainty of the underlying evidence

### 4. Usability is not only a hard gate

The system must support:

- hard rejection
- warning-only status
- guidance-only acceptance
- scoring acceptance
- benchmark acceptance

That nuance is necessary for learning from imperfect but informative data.

### 5. Historical learning must remain explicit

The long-term store should not learn opaque labels. It should learn from
structured records linking:

- heuristic signatures
- usability assessments
- selected actions
- resulting improvements or regressions

## Architecture

The usability layer should include the following architectural pieces.

### Canonical usability schema

A first-class schema should define:

- `assessment_id`
- `family`
- `task_intent`
- `heuristic_signature`
- `required_contracts_checked`
- `usable_for_guidance`
- `usable_for_scoring`
- `usable_for_final_benchmark`
- `blocking_reasons`
- `warning_reasons`
- `confidence`
- `uncertainty_notes`
- `provenance`

### Usability rule model

The system needs a deterministic rule representation describing:

- required heuristics or input contracts
- warning thresholds
- blocking thresholds
- scope of impact
- fallback behavior
- family-local rationale

### Runtime usability assessment artifact

Each evaluated dataset member should emit a persisted usability artifact inside
runtime evidence and cohort artifacts so downstream systems can reuse it without
re-deriving the decision ad hoc.

### Cohort-aware usability aggregator

Multi-member evaluation should aggregate:

- how often a usability issue appears
- which issues are common vs corner-case
- whether a member is excluded from guidance, scoring, or only final reporting

### Long-term usability memory

The long-term store should persist, per member and per run:

- heuristic observations
- usability assessment
- family and benchmark context
- selected atoms and expansions
- action classes attempted
- resulting loss deltas or other outcomes
- runtime/build context identifiers

### Benchmark and search integration

Proposal ranking, cohort selection, benchmark summaries, and search policy
should consume usability records explicitly rather than inferring usability from
raw execution behavior.

## Intended Uses

This layer should support all of the following.

### Cohort member selection

Choose dataset members that are truly usable for proposal guidance rather than
merely easy to run.

### Proposal weighting

Weight recurring problems more heavily when they appear across members that are
usable for guidance, not just across all attempted members.

### Benchmark admissibility

Ensure final benchmark conclusions are based only on members suitable for the
declared benchmark purpose.

### Search efficiency

Avoid repeatedly evaluating dataset members that have already been shown to be
unsuitable for a particular kind of comparison.

### Platform learning

Teach Sciona, over time, what kinds of heuristic signatures predict:

- unusable data
- guidance-only usefulness
- productive refinement opportunities
- poor final benchmark validity

## Relationship To Existing Plans

This plan extends, rather than replaces, the existing heuristic and asset
ownership plans.

### With the heuristic evidence layer

The heuristic evidence layer defines how evidence is produced and interpreted.
This plan defines how that evidence is used to decide data suitability.

### With heuristic outcome memory

Outcome memory currently tracks heuristic signatures and action success. This
plan adds usability assessment as an explicit part of that memory.

### With `ageo-atoms` heuristic ownership

Canonical usability concepts and family usability rules should eventually live
with the audited heuristic asset universe in `../ageo-atoms`, while matcher
consumes them.

### With benchmark reform

Benchmark policy should use usability assessment to distinguish informative but
non-final data from truly admissible benchmark members.

## Recommended Phase Structure

The implementation should proceed in staged layers.

### Phase 1. Canonical usability model

Define the cross-family schema for usability assessments and their scopes.

### Phase 2. Family usability rule registries

Add family-local declarative rules mapping heuristic evidence to usability
decisions.

### Phase 3. Runtime assessment emission

Emit persisted usability artifacts during evaluation and cohort scoring.

### Phase 4. Cohort and benchmark integration

Replace ad hoc "usable member" checks with explicit usability scope handling.

### Phase 5. Long-term usability memory

Persist assessments and their downstream outcomes into the platform's memory
store.

### Phase 6. Audit and governance

Add audit rules enforcing de-jargonization, cross-family portability, and
traceable rationale for usability assets.

## Parallelization Analysis

Some phases can run in parallel, but only after the shared schema is stable.

### Critical path

- Phase 1 must happen first.
- Phase 4 depends on meaningful progress in Phases 2 and 3.
- Phase 5 depends on Phase 3 and should ideally follow enough Phase 4 progress
  to know which assessment fields are operationally important.

### Parallel wave after Phase 1

Phases 2 and 3 can proceed in parallel:

- one track defines family usability rule assets
- the other integrates runtime emission and persistence plumbing

### Parallel wave after Phase 4 starts

Once assessment artifacts are stable, Phase 5 and Phase 6 can proceed in
parallel:

- one track builds memory persistence and reporting
- the other builds audit enforcement and governance

## Success Criteria

This plan is successful when:

- usability is represented as a first-class artifact, not hidden runtime logic
- cohort selection uses explicit usability scopes
- benchmark policy distinguishes guidance-only, scoring, and final-benchmark
  members
- long-term memory stores heuristic signatures together with usability
  assessments and outcomes
- usability assets remain cross-family, de-jargonized, and auditable
- the framework can explain why a dataset member was accepted, rejected, or only
  partially admitted

## Non-Goals

This plan does not aim to:

- encode one family's local terminology as canonical usability language
- replace general heuristic evidence with a small set of hard-coded data filters
- force every benchmark member into a single binary good/bad category
- move all implementation details into `ageo-atoms` immediately
- learn opaque black-box usability predictors without explicit evidence

## Immediate Recommendation

The next concrete implementation should introduce a first-class
`usability_assessment` artifact and make cohort selection depend on
`usable_for_guidance` rather than on procedural checks like "loss emitted and
heuristics present."

That is the smallest meaningful step that moves the current ECG cohort work
toward a deterministic, rigorous, and cross-family usability model.
