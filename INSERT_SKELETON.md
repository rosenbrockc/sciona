# Insert Skeleton Proposal

## Goal

Add `skeleton_proposal` as a first-class node-enrichment candidate without
turning enrichment into uncontrolled structural search.

The intended behavior is:

- a node may be enriched by a primitive, a retrieved template, or a skeleton
- skeleton insertion is allowed across families
- skeleton insertion is gated, ranked, and penalized for added complexity
- lower-complexity solutions should beat higher-complexity skeleton proposals
  unless the measured benefit is materially better

## Current State

Today, whole-skeleton insertion is possible, but only indirectly.

- Initial graph bootstrap uses family skeletons directly in
  `sciona.architect.nodes.select_strategy`.
- Node-level enrichment can produce a multi-node subgraph through:
  - retrieved templates / exemplars
  - decomposition output from the LLM or deterministic decomposer
- The deterministic decomposer can emit a skeleton-backed decomposition for a
  single node in `sciona.architect.deterministic_decompose`.

What does **not** exist today:

- a first-class `skeleton_proposal` candidate in the node-enrichment search
- unified ranking of primitive/template/skeleton candidates under one policy
- explicit complexity penalties for skeleton insertion

## Problem

If skeleton insertion becomes a routine ungated enrichment action, the search
space expands too quickly and the optimizer can settle into a structurally
heavier local minimum when a simpler design was better.

The main risks are:

1. Search-space blowup
- each enrichable node could branch into multiple family skeletons
- trial budgets get spent on large structural moves too early

2. Over-decomposition
- a node that needed one corrective primitive may become a 4-8 node subgraph

3. Complexity-biased local minima
- larger structures create more future tuning surface
- that can make them look artificially attractive unless complexity is
  penalized directly

4. Interface brittleness
- skeletons need boundary-port compatibility, not just local primitive IO match

5. Feature overlap
- templates, expansion rules, and skeletons can all introduce structure
- if not separated clearly, the system becomes harder to reason about

## Design Principles

1. `skeleton_proposal` should be first-class but not default.
2. Skeletons should be compared against smaller alternatives, not replace them.
3. Complexity should be an explicit negative term in proposal ranking.
4. Cross-family use is allowed, but family should remain a prior, not a gate.
5. Skeleton insertion should be reversible and observable.

## Proposal Types

For node enrichment, explicitly rank these proposal classes:

1. `primitive_proposal`
- bind the node to a single primitive / atom

2. `template_proposal`
- instantiate a retrieved exemplar or decomposition template

3. `skeleton_proposal`
- instantiate a family skeleton as a candidate subgraph for the node

These three should be treated as sibling proposal types in one ranking pass.

## Gating Rules For `skeleton_proposal`

Only consider skeleton insertion when all of the following are true:

1. The target node is non-atomic or repeatedly failing refinement.
2. Boundary IO is compatible with the candidate skeleton:
- input arity compatible
- output arity compatible
- type-class compatibility acceptable

3. The skeleton is small enough:
- hard cap on inserted node count
- hard cap on inserted edge count

4. The skeleton family is on an allowlist of families with stable boundary
semantics.

5. There is evidence that primitive/template proposals are insufficient:
- low-confidence primitive matches
- repeated critique/retry failures
- poor template confidence
- profiler or expansion diagnostics indicating structural under-specification

If these gates fail, do not generate a `skeleton_proposal`.

## Ranking Policy

Every proposal gets a score:

`proposal_score = objective_gain - complexity_penalty - risk_penalty + prior_bonus`

Where:

- `objective_gain`
  - predicted or measured improvement in the active objective

- `complexity_penalty`
  - penalty for larger structural edits
  - should increase with:
    - inserted node count
    - inserted edge count
    - number of new concept types introduced
    - number of new primitive families introduced

- `risk_penalty`
  - penalty for weak IO/type compatibility or low historical reliability

- `prior_bonus`
  - small positive weight for same-family or historically successful families
  - must not dominate the complexity term

### Complexity Penalty Requirements

This is the most important part.

`skeleton_proposal` should carry a higher default complexity penalty than
primitive or template proposals.

Recommended shape:

- primitive proposal: minimal base penalty
- template proposal: moderate base penalty
- skeleton proposal: highest base penalty

And then add scaled penalties for:

- `delta_nodes`
- `delta_edges`
- `delta_family_count`
- `delta_concept_type_count`

This means a skeleton must clear a materially higher bar to win.

In practice:

- a simpler proposal should win when objective improvements are comparable
- a skeleton should only win when it offers a meaningfully better measured or
  predicted improvement

## Suggested Scoring Heuristic

Start with a conservative heuristic:

`score = gain`
`- 0.20 * delta_nodes`
`- 0.10 * delta_edges`
`- 0.35 * delta_family_count`
`- 0.25 * delta_concept_type_count`
`- 0.50 * skeleton_base_penalty`

With:

- `skeleton_base_penalty = 1` for `skeleton_proposal`
- `0` otherwise

This is only a starting point. The exact constants should be tuned using
telemetry, but the ordering should remain:

- family diversity is good only when it improves the objective
- extra structure is expensive by default

## Acceptance Rule

Do not accept a skeleton proposal unless:

1. It passes all gating checks.
2. Its score is better than the best primitive/template alternative.
3. Its objective gain clears a minimum margin over the best lower-complexity
alternative.

That margin is important. Without it, a slightly better noisy estimate could
cause systematic over-expansion.

Recommended conservative rule:

- if a skeleton is more complex than the best alternative, require a strictly
  positive margin above that alternative before acceptance

## Telemetry

For every `skeleton_proposal`, record:

- target node
- source skeleton family
- inserted node count
- inserted edge count
- distinct concept types introduced
- distinct primitive families introduced
- complexity penalty
- objective estimate
- final rank
- accepted / rejected
- later retained / later reverted

Add summary metrics:

- `skeleton_proposal_trials`
- `accepted_skeleton_proposals`
- `rejected_skeleton_proposals`
- `mean_skeleton_complexity_penalty`
- `mean_skeleton_objective_gain`
- `skeleton_retention_rate`

## Recommended Implementation Path

1. Add `skeleton_proposal` as a formal candidate type in node enrichment.

2. Build a small skeleton-proposal generator:
- takes a target node
- evaluates allowlisted families
- filters by boundary compatibility
- instantiates only bounded-size skeletons

3. Add unified proposal ranking across:
- primitive
- template
- skeleton

4. Add the explicit complexity penalty described above.

5. Start with a narrow allowlist:
- families with stable boundary semantics and moderate skeleton size

6. Add acceptance tests proving:
- skeleton proposals are considered
- lower-complexity candidates win when gains are similar
- skeleton proposals win only when gains are materially better
- harmful skeleton insertions are rejected or rolled back

## Recommendation

Implement `skeleton_proposal` as a first-class enrichment capability, but only
as a constrained, penalized proposal class.

Do **not** make it a default enrichment path.

The default policy should remain:

- prefer simpler proposals
- allow skeleton insertion when it is structurally justified
- require higher measured value from more complex proposals

That preserves the cross-family capability without letting the optimizer drift
into heavier local minima just because larger structures are available.

## Planner-Oriented Phase Plan

This section is meant for focused planner-agent execution. Each phase should be
implemented and reviewed independently.

### Phase 1: Proposal Surface And Data Model

Goal:

- introduce `skeleton_proposal` as a first-class proposal type without changing
  real enrichment behavior yet

Primary owner modules:

- `sciona.architect.nodes`
- `sciona.architect.template_retriever`
- `sciona.architect.deterministic_decompose`
- any new proposal-model module under `sciona.architect`

Required outputs:

- a formal proposal representation that can express:
  - `primitive_proposal`
  - `template_proposal`
  - `skeleton_proposal`
- proposal metadata fields for:
  - source family
  - delta nodes
  - delta edges
  - compatibility score
  - proposal type

Non-goals:

- no live skeleton generation yet
- no ranking change yet
- no telemetry/dashboard change yet

Acceptance criteria:

- existing behavior remains unchanged
- proposal data structures exist and are covered by unit tests
- `skeleton_proposal` can be represented in-memory without special-casing later

### Phase 2: Gated Skeleton Proposal Generation

Goal:

- generate bounded `skeleton_proposal` candidates for eligible nodes

Primary owner modules:

- `sciona.architect.nodes`
- `sciona.architect.skeletons`
- `sciona.architect.deterministic_decompose`

Recommended implementation shape:

- add a small skeleton-proposal generator helper
- input:
  - target node
  - current graph context
  - allowlisted skeleton families / named skeletons
- output:
  - zero or more `skeleton_proposal` candidates

Required gates:

- target node must be non-atomic or refinement-failing
- boundary IO compatibility must pass
- skeleton size caps must pass
- family allowlist must pass

Initial allowlist recommendation:

- start with a very small allowlist of skeletons with stable boundary semantics
- do not start with the full skeleton registry

Non-goals:

- no global search over every skeleton
- no acceptance/ranking against other candidate types yet

Acceptance criteria:

- bounded skeleton candidates can be generated for eligible nodes
- ineligible nodes produce no skeleton proposals
- family allowlist and size caps are enforced by tests

### Phase 3: Unified Proposal Ranking

Goal:

- rank primitive, template, and skeleton proposals together under one policy

Primary owner modules:

- `sciona.architect.nodes`
- possible new proposal-ranking helper under `sciona.architect`

Required behavior:

- compare sibling proposals in a single ranking pass
- add explicit complexity penalty terms
- apply the higher base penalty to `skeleton_proposal`

Required scoring dimensions:

- predicted or proxy objective gain
- complexity penalty
- risk penalty
- family prior bonus

Hard requirement:

- lower-complexity proposals should win when gains are comparable

Non-goals:

- no Principal-wide proposal execution changes yet
- no rollback logic yet

Acceptance criteria:

- ranking function exists and is tested directly
- skeleton proposals lose to simpler candidates when gains are similar
- skeleton proposals can still win when their gain is materially better

### Phase 4: Acceptance, Execution, And Rollback

Goal:

- make `skeleton_proposal` safe in the live enrichment path

Primary owner modules:

- `sciona.architect.nodes`
- possibly `sciona.principal.graph` if skeleton proposals later participate in
  Principal-level proposal selection

Required behavior:

- accept only the highest-ranked candidate that clears the minimum margin rule
- reject skeleton proposals that do not beat lower-complexity alternatives by
  enough
- preserve rollback semantics if a chosen skeleton-based enrichment later harms
  the measured objective

Minimum margin rule:

- if a skeleton proposal is more complex than the best lower-complexity
  alternative, require a strictly positive measured margin before acceptance

Non-goals:

- no broad dashboard work yet

Acceptance criteria:

- harmful skeleton proposals are rejected or reverted
- simpler candidates still win when the margin is not cleared
- live tests cover both acceptance and rejection paths

### Phase 5: Telemetry, Dashboard, And Tuning Loop

Goal:

- make skeleton proposal behavior observable and tuneable

Primary owner modules:

- `sciona.commands.optimize_cmds`
- `sciona.visualizer_api`
- `sciona.static.dashboard.html`
- relevant tests under `tests/test_visualizer_api.py`

Required telemetry:

- `skeleton_proposal_trials`
- `accepted_skeleton_proposals`
- `rejected_skeleton_proposals`
- `mean_skeleton_complexity_penalty`
- `mean_skeleton_objective_gain`
- `skeleton_retention_rate`

Per-trial fields:

- proposal type
- source skeleton family
- complexity deltas
- penalty terms
- accepted / rejected
- later retained / reverted

Acceptance criteria:

- telemetry records are emitted for skeleton proposals
- dashboard summarizes both volume and quality of skeleton use
- tuning the complexity weights becomes possible from observed runs

## Phase Boundaries For Planner Agents

Use these boundaries when assigning work:

- Phase 1 should not change live proposal outcomes
- Phase 2 should only generate skeleton candidates, not broadly accept them
- Phase 3 should change ranking but not rollback semantics
- Phase 4 should add safety and live acceptance behavior
- Phase 5 should make behavior observable and tunable

This sequencing keeps each planner run narrowly scoped and reduces the chance
that a single phase mixes interface design, search behavior, and observability.

## Architecture Constraints For Every Phase

These constraints apply to all phases and should be treated as hard
requirements by planner agents:

1. Do not explode the search space.
- use allowlists
- use size caps
- avoid full skeleton-registry enumeration by default

2. Do not let larger structures win cheaply.
- skeleton proposals must carry the highest base complexity penalty

3. Do not replace simpler proposal types.
- primitive and template proposals remain first-class and should usually be
  preferred

4. Do not use family as a hard routing boundary.
- cross-family skeletons are allowed if compatibility and ranking justify them

5. Keep changes reversible.
- proposal acceptance must remain measurable and rollback-safe

6. Preserve current behavior by default.
- before the ranking/acceptance phases, the mere presence of
  `skeleton_proposal` plumbing should not change outcomes
