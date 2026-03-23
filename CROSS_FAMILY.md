## Cross-Family Expansion Plan

### Goal

Make cross-family and cross-disciplinary CDG expansion a normal optimization path rather than an accidental side effect of retrieval or decomposition.

The desired behavior is:

- a CDG may start from a skeleton in one family
- later expansion may introduce primitives, atoms, or subgraphs from another family
- those mixed-family expansions are evaluated and retained when they improve the objective

### Current Architectural Assessment

The current system is cross-family compatible, but not cross-family seeking.

What already supports cross-family behavior:

- `sciona.principal.expansion.ExpansionEngine` is domain-agnostic
- `sciona.principal.expansion_rules.default_rule_sets()` registers rule sets from many families
- `sciona.architect.graph_rewriter.GraphRewriter` can rewrite heterogeneous graphs
- `sciona.architect.catalog.PrimitiveCatalog.find_matching_primitives()` allows cross-category fallback
- deterministic primitive binding in `sciona.architect.deterministic_decompose` scans all primitives, not just the parent family

What currently suppresses cross-family behavior in practice:

- initial CDGs are seeded from a single selected paradigm skeleton
- `Principal` does not currently use the expansion engine as a live optimization stage
- family-specific variant mutation is the main in-place update path
- refinement/template retrieval is keyed heavily by `concept_type`
- ledger-based mutation currently prefers same-category substitutions

### Target Design

The optimization loop should become:

1. build initial CDG from architect skeleton or decomposition
2. evaluate current artifact
3. build expansion diagnostics from runtime evidence
4. run family-agnostic topology expansion
5. evaluate expanded structure
6. run hyperparameter search on the fixed expanded structure
7. try local variant and primitive substitutions
8. fall back to time-travel re-decomposition only when the above fail

This changes the role of families:

- family is a prior
- structure compatibility is the gate
- measured objective improvement is the decision rule

### Phase 1: Expansion-First Principal

Integrate `ExpansionEngine(default_rule_sets())` into `sciona.principal.graph`.

Required changes:

- add a new Principal stage after evaluation and before variant mutation
- construct `ExpansionContext` from runtime intermediates, evaluation metrics, and signal data
- execute the expansion engine on every trial where runtime evidence is available
- if one or more rules apply, treat the expanded CDG as the next candidate structure
- reset or rescope hyperparameter search when the topology or primitive signature changes

Expected effect:

- the live optimizer will finally try the cross-family expansion machinery that already exists

### Phase 2: Cross-Family Mutation Beyond Expansion

Relax mutation logic so family is not a hard boundary.

Required changes:

- keep curated family variant mutation, but treat it as one plugin among several
- modify ledger-based mutation to consider structurally compatible cross-category candidates
- use a scoring penalty for foreign-family substitutions instead of rejecting them
- rank candidates by:
  - IO compatibility
  - objective history
  - witness/type compatibility
  - family prior

Expected effect:

- even when no explicit expansion rule fires, Principal can still adopt a useful primitive from another family

### Phase 3: Cross-Family Refinement Retrieval

Refinement/template retrieval should not be locked to `concept_type`.

Required changes:

- keep same-`concept_type` retrieval as a high-confidence path
- add a parallel retrieval path keyed by:
  - IO arity
  - child topology
  - witness type overlap
  - abstract type class
- use `concept_type` as a scoring feature, not a hard query filter
- allow refinement templates from another family to seed split/rewrite suggestions

Expected effect:

- failed leaves can be repaired using templates from different disciplines when the structural fit is strong

### Phase 4: Cross-Family Telemetry

Track whether optimization is actually producing interdisciplinary reuse.

For each Principal trial, record:

- `distinct_concept_types`
- `distinct_source_families`
- `cross_family_node_count`
- `cross_family_edge_count`
- `family_entropy`
- `foreign_family_bindings`
- `expansion_rules_applied`
- `topology_changed`
- `primitive_assignment_changed`
- objective value before and after expansion

This should be written into:

- `trial_history.json`
- optimize telemetry run metadata
- dashboard optimize summaries

Expected effect:

- we can measure complexity-versus-accuracy tradeoffs and also diversity-versus-accuracy tradeoffs

### Phase 5: Acceptance Tests

Add live tests for mixed-family optimization, not just unit tests of the expansion engine.

Minimum tests:

- a signal-family CDG expands with a statistics-family diagnostic or cleanup node
- a graph-optimization CDG adopts a linear-algebra subroutine when IO matches
- Principal prefers a mixed-family expansion over a same-family fallback when loss improves
- trial history explicitly records increased family diversity
- a mixed-family expansion can be rolled back when it harms the objective

Important constraint:

- these must exercise the live Principal path, not only `ExpansionEngine` in isolation

### Phase 6: Proposal-Level Selection Instead of Stage-Order Bias

The current loop is expansion-first in execution order. That is acceptable for
bootstrapping cross-family behavior, but it still creates a search-order bias:
with a tight trial budget, Principal may spend budget on an expansion attempt
before evaluating the strongest same-family local mutation from the same
baseline.

Required changes:

- treat expansion and local mutation as sibling proposals from the same trial baseline
- from one evaluated baseline, generate:
  - best expansion candidate
  - best local same-family / ledger mutation candidate
  - optional re-decomposition candidate only after those are exhausted
- evaluate those proposals under the same objective accounting
- choose the next branch by measured loss improvement, not by stage order
- keep rollback semantics for any proposal that underperforms the baseline

Expected effect:

- cross-family exploration remains first-class
- same-family options are not artificially delayed by pipeline ordering
- the optimizer becomes objective-driven at the proposal level rather than
  stage-driven

### Design Guardrails

- do not use family as a hard routing boundary
- do use family as a ranking prior
- structural compatibility and measured objective improvement should dominate
- keep expansion reversible
- avoid introducing family-specific code into Principal core when a plugin interface will do

### Recommended First Implementation Slice

If implementation is staged narrowly, do this first:

1. wire `ExpansionEngine(default_rule_sets())` into `sciona.principal.graph`
2. build `ExpansionContext` from real run artifacts
3. record expansion and family-diversity telemetry

This is the smallest change that converts cross-family expansion from theoretical capability into live optimizer behavior.
