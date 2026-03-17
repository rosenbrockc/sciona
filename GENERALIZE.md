# Generalizing AGEO-Matcher

## Summary

The repo contains the main building blocks for a general synthesis architecture:

- deterministic decomposition and scaffold instantiation
- semantic and lexical retrieval
- Memgraph-backed CDG retrieval
- a notion of "isomorphism" search in the visualizer
- deterministic refinement and verification loops

The main architectural gap is that these pieces are not yet the primary decision path. Domain-shaped control flow bypasses retrieval and general composition in six distinct layers of the system, not just the three most visible ones.

## Where domain knowledge is hardcoded

Domain knowledge appears at every layer of the pipeline, not just in decomposition and refinement. A planner must account for all six layers to avoid leaving domain coupling in place after generalization work.

### Layer 1: Execution routing (run_cmds.py)

[ageom/commands/run_cmds.py](/Users/conrad/personal/ageo-matcher/ageom/commands/run_cmds.py) contains a complete fast-path for signal-event-rate goals.

- `_matches_signal_event_rate_goal()` (line 36): triple-term conjunction — goal must contain a signal term (`ecg`, `ppg`, `eeg`, `sensor`, etc.), a detect term (`detect`, `peak`, `event`), and a rate term (`rate`, `cadence`, `rhythm`).
- `_SIGNAL_EVENT_RATE_DECLARATIONS` (line 48): three hardcoded primitive declarations with full type signatures (`filter_signal_for_detection`, `detect_peaks_in_signal`, `compute_event_rate`).
- `_build_signal_event_rate_cdg()` (line 128): constructs a deterministic 3-node CDG using a skeleton retrieved by `ConceptType.SIGNAL_FILTER` with variant `"event_rate_estimation"`.
- `_build_signal_event_rate_match_results()` (line 74): pre-populates match results with 100% confidence and retrieval method `"curated_signal_event_rate"`, bypassing Hunter entirely.

When this fast-path fires (line 208), the run skips Architect decomposition and Hunter matching completely. It also receives special handling in structured mode (line 248) and verified orchestration (line 684).

### Layer 2: Strategy classification (strategy_classifier.py)

[ageom/architect/strategy_classifier.py](/Users/conrad/personal/ageo-matcher/ageom/architect/strategy_classifier.py) selects the decomposition paradigm before the Architect LLM runs.

- `_PHRASE_RULES` (line 24): 50+ deterministic rules mapping keyword phrases to ConceptTypes with hardcoded weights. Examples: `"bandpass filter"` → `SIGNAL_FILTER` (weight 4.0), `"ecg"` → `SIGNAL_FILTER` (weight 1.5), `"kalman filter"` → `SEQUENTIAL_FILTER` (weight 4.0).
- Special event-rate detection block (line 171): if goal contains signal markers AND detect markers AND rate markers, returns a high-confidence `SIGNAL_FILTER` classification immediately.

This layer is upstream of everything else — it determines which paradigm the Architect uses, which skeleton is retrieved, and whether a fast-path fires. New domains require adding rules to this list before any retrieval-based approach gets a chance.

### Layer 3: Architect decomposition (nodes.py)

[ageom/architect/nodes.py](/Users/conrad/personal/ageo-matcher/ageom/architect/nodes.py) contains a conjugate-update short-circuit that bypasses the entire decompose/critique loop.

- `_CONJUGATE_PAIRS` (line 62): a module-level dict of four conjugate pair specifications — `beta_bernoulli`, `normal_normal`, `gamma_poisson`, `dirichlet_categorical` — each with keywords, library hints, sufficient stat descriptions, update rules, result distribution forms, and type signatures.
- `_detect_conjugate_pair()` (line 154): keyword matching with a dual-gate design — keyword match fires immediately, library-hint match additionally requires a probabilistic keyword (`"prior"`, `"posterior"`, `"conjugate"`, `"bayesian"`, etc.).
- `select_strategy()` (line 620): runs conjugate detection **before the LLM call**. If matched, creates a root node with `ConceptType.CONJUGATE_UPDATE` and `NodeStatus.DECOMPOSED`, sets `short_circuit: True`.
- `route_after_strategy()` (line 1570): routes to `advance_conjugate_node` which emits a fixed 3-node atomic CDG (Data Ingestion → Hyperparameter Update → Distribution Construction) with matched primitives derived from the pair name. All three nodes are `ATOMIC`, bypassing the decompose/critique loop entirely.

The Architect graph topology ([ageom/architect/graph.py](/Users/conrad/personal/ageo-matcher/ageom/architect/graph.py) line 49) has a dedicated `"conjugate"` edge from the router to `advance_conjugate_node` → `END`.

### Layer 4: Hunter query reformulation (query_reformulator.py)

[ageom/hunter/query_reformulator.py](/Users/conrad/personal/ageo-matcher/ageom/hunter/query_reformulator.py) shapes how Hunter searches for primitive matches.

- `_DOMAIN_ANCHORS` (line 72): a set of domain-specific tokens (`"ecg"`, `"bandpass"`, `"dijkstra"`, `"cholesky"`, `"lcs"`, `"spd"`) that receive special treatment in query construction.
- `_phrase_rules()` (line 164): deterministic branching for known problems. Returns 5 pre-computed search query variants for each: ECG bandpass filter (line 168), shortest path + Dijkstra (line 176), SPD/Cholesky (line 186), LCS (line 194), natural number addition commutativity (line 204).
- `_keyword_variants()` (line 229): treats `"ecg"` and `"dp"` as special short tokens that are kept rather than discarded.

This layer is invisible to decomposition-level generalization. A TemplateRetriever that replaces scaffold construction will not affect how Hunter formulates its search queries for leaf-node matching.

### Layer 5: Orchestrator refinement (orchestrator.py)

[ageom/orchestrator.py](/Users/conrad/personal/ageo-matcher/ageom/orchestrator.py) contains 16 hand-authored domain-specific split rules in `_deterministic_split_subnodes()` (line 122).

The function combines node description, failure PDG statement, error summaries, and candidate names into a single lowercased context string, then pattern-matches against:

1. ECG/bandpass/filter (line 135) → "Design Filter" / "Apply Filter"
2. Shortest path/Dijkstra (line 151) → "Initialize Distances" / "Relax Edges"
3. Cholesky/SPD (line 167) → "Cholesky Factor" / "Triangular Solve"
4. Longest common subsequence (line 181) → "Build DP Table" / "Backtrack Subsequence"
5. Matrix factorization/SVD/eigenvalue (line 197) → "Factorize Matrix" / "Extract Components"
6. Optimization/gradient (line 212) → "Initialize Parameters" / "Iterate Optimization" / "Extract Solution"
7. FFT/spectral (line 232) → "Compute Transform" / "Analyze Spectrum"
8. Sorting (line 247) → "Partition Elements" / "Merge Ordered"
9. String matching/edit distance (line 262) → "Build Distance Table" / "Trace Alignment"
10. Signal detection + computation (line 277) → "Detect Features" / "Compute Metric"
11. Interpolation/spline (line 292) → "Fit Model" / "Evaluate Model"
12. Clustering/k-means (line 307) → "Assign Clusters" / "Refine Centroids"
13. Statistical/Bayesian inference (line 322) → "Specify Model" / "Fit Or Sample" / "Summarize Results"
14. Tree/graph traversal (line 342) → "Initialize Traversal" / "Explore Neighbors"
15. Convolution/correlation (line 357) → "Prepare Kernel" / "Apply Convolution"
16. Normalization/standardization (line 372) → "Compute Statistics" / "Apply Transform"

If none match, falls back to `_split_on_connectors()` (line 102) which splits descriptions on discourse connectors (`and then`, `then`, `with`, `plus`, `before`, `after`).

Additionally, the `GENERALIZE` refinement action (line 503) is **stubbed out** — it only sets `critic_notes` and does not actually broaden type signatures or relax constraints. This means the orchestrator has only two functional refinement actions: `UNGROUNDABLE` (mark rejected) and `SPLIT` (deterministic or LLM-driven).

### Layer 6: Ingestion and synthesis

Domain knowledge also appears in:

- **Template abstractor** ([ageom/ingester/template_abstractor.py](/Users/conrad/personal/ageo-matcher/ageom/ingester/template_abstractor.py) line 11): strips domain prefixes (`"ecg"`, `"audio"`, `"financial"`, `"bio"`, `"image"`, `"video"`) and maps concept types to abstract names (`"signal_filter"` → `"Signal Conditioner"`).
- **Ingester prompts** ([ageom/ingester/prompts.py](/Users/conrad/personal/ageo-matcher/ageom/ingester/prompts.py) line 20): SEMANTIC_CHUNK_SYSTEM lists domain-specific ConceptTypes including `signal_transform`, `signal_filter`, `conjugate_update`, `sequential_filter`, `smc_reweight`, `message_passing`.
- **Python synthesizer** ([ageom/synthesizer/python_template.py](/Users/conrad/personal/ageo-matcher/ageom/synthesizer/python_template.py) line 205): `_flatten_inputs()` creates a `flat['signal']` alias when group name contains `('signal', 'ecg', 'ppg', 'wave')`.
- **Polar adapter** ([ageom/principal/adapters/polar.py](/Users/conrad/personal/ageo-matcher/ageom/principal/adapters/polar.py)): ECG-specific enum, SQL queries, and properties.
- **Runtime implementation** ([ageom/runtime_signal_event_rate.py](/Users/conrad/personal/ageo-matcher/ageom/runtime_signal_event_rate.py)): hardcoded butterworth bandpass (3-25 Hz), peak detection thresholds, rate computation. This is the actual signal processing code that the curated fast-path maps to.

## Current retrieval infrastructure

### What already exists

Memgraph-backed retrieval exists and is architecturally promising:

- [ageom/architect/graph_retrieval.py](/Users/conrad/personal/ageo-matcher/ageom/architect/graph_retrieval.py#L1) retrieves similar decomposed subgraphs via a three-layer cascade.
- [ageom/architect/nodes.py](/Users/conrad/personal/ageo-matcher/ageom/architect/nodes.py#L866) injects retrieved examples into the Architect prompt via `format_examples_for_prompt()`.
- [ageom/graph_store.py](/Users/conrad/personal/ageo-matcher/ageom/graph_store.py#L162) stores rich metadata including witness types, abstract type class, statefulness, and contracts.
- [ageom/upsert_cdg.py](/Users/conrad/personal/ageo-matcher/ageom/upsert_cdg.py#L176) supports upserting CDGs into Memgraph with sanitization (dedup, edge normalization, childless-decomposed demotion).

The decompose_node function in nodes.py (line 840) also has three-source primitive retrieval: catalog primitives, lexical fallback, and skill index — deduplicated in a specific order. Plus three context injection channels (lines 888-912): shared context, template context, and failure context, each in separate namespaces.

Graph retrieval is optional — controlled by `retrieval_policy.graph_retrieval_enabled` (run_cmds.py line 442). When disabled, the retriever is `None` and the Architect runs without it.

### Three-layer similarity stack

The current "isomorphism" layer is a three-stage similarity stack, not true graph isomorphism:

**Layer 1: Exact topo_hash match** ([graph_store.py](/Users/conrad/personal/ageo-matcher/ageom/graph_store.py#L282) line 282)
- Queries `MATCH (parent:Atom:Decomposed {topo_hash: $topo_hash})`, returns up to 5 candidates.
- Score: 1.0 (perfect).

**Layer 2: Structural match** ([graph_store.py](/Users/conrad/personal/ageo-matcher/ageom/graph_store.py#L312) line 312)
- Filters by `concept_type` (exact), `n_inputs` (±1), `n_outputs` (±1), `min_children >= 2`.
- Score: always 0.7. The `_io_match_factor()` method ([graph_retrieval.py](/Users/conrad/personal/ageo-matcher/ageom/architect/graph_retrieval.py#L254) line 254) always returns 1.0 — it is a placeholder that does not actually discriminate. Every Layer 2 result gets the same score regardless of match quality.

**Layer 3: Jaccard neighbourhood similarity** ([graph_store.py](/Users/conrad/personal/ageo-matcher/ageom/graph_store.py#L361) line 361)
- Only runs if `exclude_repo` is set. Computes Jaccard overlap on child concept_type lists.
- Filters: `jaccard_score > 0.3`. Score: `0.5 * jaccard_score`.
- Ignores edge structure, node status, primitives, type compatibility, port constraints.

Ranking: Layer 1 (1.0) beats Layer 2 (0.7) beats Layer 3 (up to 0.5). Deduplicates by FQN, returns top N (default 3).

### Topo_hash limitations

The `_topo_hash()` function ([graph_store.py](/Users/conrad/personal/ageo-matcher/ageom/graph_store.py#L12) line 12) works as follows:

1. Finds children of root_id
2. Filters edges to only sibling-to-sibling edges (ignoring phantom edges to `initial`/`final`)
3. For each child, computes `(in_degree, out_degree)` among siblings
4. Hashes the **sorted** degree sequence via SHA-256

This means:
- Two fundamentally different topologies with the same sorted degree sequence will collide.
- Edge labels, node labels, port names, primitive identities, type signatures, and type classes are all ignored.
- Phantom edges (entry/exit patterns) are excluded, so decompositions with different I/O connectivity can hash identically.

### Metadata stored vs. metadata used in retrieval

The graph store persists extensive metadata via `build_atom_params()` (graph_store.py line 162). Most of it is never queried:

| Field | Stored | Used in retrieval |
|-------|--------|-------------------|
| concept_type | Yes | Yes (Layer 2) |
| n_inputs / n_outputs | Yes | Yes (Layer 2, ±1) |
| topo_hash | Yes | Yes (Layer 1) |
| witness_name | Yes | No |
| witness_input_types | Yes | No |
| witness_output_types | Yes | No |
| abstract_type_class | Yes (indexed) | No |
| is_stateful | Yes | No |
| input_contracts | Yes | No |
| output_contracts | Yes | No |
| type_signature | Yes | No |
| depth | Yes | No |
| parallelizable | Yes | No |
| conceptual_summary | Yes | No |
| matched_primitive | Yes | No |

### Auto-upsert of solved runs

**Not implemented.** The only upsert path is the manual CLI command `_cmd_upsert_cdg()` ([ageom/upsert_cdg.py](/Users/conrad/personal/ageo-matcher/ageom/upsert_cdg.py#L176) via upsert_cmds.py line 10) which discovers `*cdg*.json` files in a directory.

Run results end as `OrchestratorResult` or `PlannerRunResult` objects that are never persisted back to CDG JSON or Memgraph. There is no mechanism to capture the final CDG after verification, no "solved run" designation in telemetry, and no conversion pipeline from run result → CDG → upsert.

### Test coverage

Test fixtures ([tests/fixtures/match_cases.json](/Users/conrad/personal/ageo-matcher/tests/fixtures/match_cases.json)) contain only two domains: `pulsar_pipeline` and `biosppy_ecg`. There are no domain-agnostic benchmark cases. Generalization work has no way to measure whether it works for non-signal-processing domains without adding new test cases first.

## Architectural direction

The right direction is not "remove all domain knowledge." The right direction is:

- keep domain knowledge
- move it out of runtime `if goal contains X` branches **at every layer, not just decomposition**
- represent it as retrievable, inspectable, reusable structure

That means turning domain-specific solutions into exemplars, templates, or macro-scaffolds that are retrieved and verified through the same general path as everything else. But it also means generalizing strategy classification, query reformulation, and synthesis — not just the Architect and orchestrator.

## Recommendations

### 1. Fix the broken IO match factor

`_io_match_factor()` in [graph_retrieval.py](/Users/conrad/personal/ageo-matcher/ageom/architect/graph_retrieval.py#L254) (line 254) always returns 1.0. This makes Layer 2 retrieval non-discriminative — every structural match scores 0.7 regardless of quality.

This is a quick fix that makes existing retrieval actually useful before building more infrastructure on top of it. The function should compute a real similarity score based on input/output count proximity, concept type match quality, and child count alignment.

### 2. Add domain-agnostic test cases

Before any generalization work begins, add test fixtures and benchmark goals that are not signal processing or Bayesian inference. Without these, there is no way to measure whether generalization is working.

Candidate domains: graph algorithms (shortest path, MST), string algorithms (edit distance, LCS), linear algebra (matrix factorization, least squares), optimization (gradient descent, LP), control systems, NLP pipelines.

### 3. Introduce a unified TemplateRetriever

Build a single retrieval layer for reusable decomposition scaffolds. The planner path should become:

1. direct atomic match
2. exemplar/template scaffold retrieval
3. deterministic scaffold instantiation when confidence is high
4. ordinary Hunter verification of leaf bindings
5. Architect decomposition only when retrieval confidence is low

This unifies signal-event-rate decomposition, conjugate updates, and the 16 orchestrator split patterns under a single code path. The key change is that these become data-backed exemplars rather than control-flow exceptions.

The TemplateRetriever must account for the existing retrieval sources in decompose_node (catalog, lexical, skill index) and the three context injection channels (shared, template, failure). It should compose with these, not replace them.

Recommended architecture:

- `GraphCandidateRetriever`: Memgraph query layer (replaces current three-layer cascade)
- `GraphAlignmentScorer`: Python structural scorer (replaces topo_hash + Jaccard)
- `TemplateRetriever`: orchestrates candidate generation, scoring, and confidence thresholds

### 4. Generalize strategy classification

The 50+ phrase rules in [strategy_classifier.py](/Users/conrad/personal/ageo-matcher/ageom/architect/strategy_classifier.py) (line 24) are upstream of all decomposition logic. If these remain hardcoded, new domains still require code changes before retrieval-based decomposition can run.

Options:

- Replace phrase rules with a retrieval step: given a goal, find the most similar previously-classified goals and infer the paradigm from their classification.
- Keep deterministic rules but load them from a data file rather than module-level code. New domains add entries to the data file, not the source.
- Use the LLM classifier as the primary path with deterministic rules as high-confidence overrides only.

The phrase rules should not be deleted — they encode useful signal. But they should become data, not code.

### 5. Generalize query reformulation

The pre-computed query variants in [query_reformulator.py](/Users/conrad/personal/ageo-matcher/ageom/hunter/query_reformulator.py) (line 164) and domain anchor tokens (line 72) are invisible to decomposition-level generalization.

Options:

- Move `_phrase_rules()` variants into a data file keyed by concept type or keyword pattern. Load at startup.
- Allow the TemplateRetriever to supply query hints alongside scaffold templates — if a template was successfully used before, its associated Hunter queries can be reused.
- Make `_DOMAIN_ANCHORS` extensible via configuration rather than a hardcoded set.

### 6. Use richer Memgraph metadata in retrieval

The graph store already persists metadata that retrieval never uses (see table above). The retrieval layer should use these fields for ranking and filtering:

- prefer candidates with matching `abstract_type_class`
- prefer candidates whose witness input/output types align with the query node
- prefer candidates with compatible contract structure
- prefer candidates with high prior verification coverage
- filter by statefulness when the query node is stateful or stateless

See [ageom/graph_store.py](/Users/conrad/personal/ageo-matcher/ageom/graph_store.py#L162).

### 7. Add real app-layer graph matching

The repo needs a real near-isomorphism / subgraph-matching step to replace the topo_hash + Jaccard approximation.

Recommended matching features:

- node labels: `concept_type`, `status`
- node metadata: `matched_primitive`, `abstract_type_class`, `is_stateful`
- edge labels: `output_name`, `input_name`
- port counts and port names
- type-signature compatibility
- optional witness and contract overlap

This does not need to be perfect VF2-level graph isomorphism on day one, but it does need to be materially stronger than the current stack.

Architecture: Memgraph returns top-N candidates, Python reranks with a labeled graph matcher, instantiate if alignment is strong, fall back to Architect decomposition otherwise.

### 8. Implement the GENERALIZE refinement action

The `GENERALIZE` action in [orchestrator.py](/Users/conrad/personal/ageo-matcher/ageom/orchestrator.py) (line 503) is currently a stub — it sets `critic_notes` but does not broaden type signatures or relax constraints. The orchestrator effectively has only two functional refinement actions: `UNGROUNDABLE` (reject) and `SPLIT` (deterministic or LLM).

Implementing `GENERALIZE` gives the orchestrator a non-domain-specific refinement path: relax type constraints, broaden the description, or remove specificity from the node to allow a wider Hunter search. This is valuable independent of the TemplateRetriever work.

### 9. Move hard-coded domain paths into exemplar data

The signal-event-rate scaffold, conjugate-update path, and 16 orchestrator split patterns should not disappear immediately. They should be relocated:

- Current runtime branches become declarative templates (JSON/YAML)
- Templates are inserted into a template index or Memgraph store
- The planner retrieves them through the same path used for other decomposition reuse
- Telemetry still records when a solution came from a curated exemplar

The signal-event-rate path (run_cmds.py) already uses real skeleton instantiation via the skeleton library with variant `"event_rate_estimation"`. The conjugate path (nodes.py) does not — it hardcodes the 3-node structure. Both should converge to the same retrieval-based mechanism.

### 10. Auto-upsert solved runs into Memgraph

After a verified or partially verified run, automatically persist:

- the resulting CDG (requires a conversion from OrchestratorResult/PlannerRunResult → CDG JSON)
- final leaf matches with matched primitive names
- verification coverage percentage
- provenance (run ID, timestamp, goal text, execution path)
- benchmark or dataset tags if relevant

This requires building a new persistence pipeline. Today, run results evaporate when the run ends. The pipeline needs to:

1. Capture the final CDG state from the orchestrator or planner result
2. Convert match results into node metadata (matched_primitive, verification status)
3. Compute coverage (fraction of leaf nodes with verified matches)
4. Call `upsert_cdg()` with the enriched CDG
5. Store verification coverage as a queryable field on the root node

### 11. Generalize refinement using retrieved split patterns

The 16 split rules in [orchestrator.py](/Users/conrad/personal/ageo-matcher/ageom/orchestrator.py#L122) should move toward retrieval:

- Search for structurally similar failed nodes in Memgraph
- Retrieve how similar nodes were successfully decomposed elsewhere
- Align the retrieved decomposition to the current failure context
- Apply the retrieved split if confidence is high
- Fall back to LLM refinement otherwise

The orchestrator currently has no cross-round learning — each refinement round starts fresh from the failure context. Retrieval-based refinement would accumulate split precedents over time, especially once auto-upsert (recommendation 10) is in place.

### 12. Address ingestion and synthesis domain coupling

The template abstractor, ingester prompts, and Python synthesizer all embed domain assumptions. These are lower priority than the decomposition and retrieval layers but should not be forgotten:

- **Template abstractor**: `_DOMAIN_PREFIXES` and `_ABSTRACT_NAME_OVERRIDES` should be data-driven, loaded from configuration rather than module-level dicts.
- **Ingester prompts**: the ConceptType list in SEMANTIC_CHUNK_SYSTEM should be generated from the ConceptType enum rather than hardcoded in the prompt string. New concept types added to the enum should automatically appear in the prompt.
- **Python synthesizer**: the signal-specific `flat['signal']` alias in `_flatten_inputs()` should be removed or generalized to a configurable alias mapping.

## Memgraph query direction

The next generation of queries should expose more discriminative structure.

Example coarse candidate query:

```cypher
MATCH (p:Atom:Decomposed)-[:PARENT_OF]->(c:Atom)
WHERE p.repo <> $exclude_repo
  AND p.concept_type IN $candidate_concepts
  AND abs(p.n_inputs - $n_inputs) <= 1
  AND abs(p.n_outputs - $n_outputs) <= 1
WITH p,
     collect(c.concept_type) AS child_types,
     collect(c.abstract_type_class) AS child_type_classes,
     collect(c.input_contracts) AS input_contracts,
     collect(c.output_contracts) AS output_contracts
RETURN p.fqn, p.repo, p.topo_hash, child_types, child_type_classes,
       input_contracts, output_contracts
LIMIT 50
```

Then rerank those 50 in Python using the GraphAlignmentScorer.

A second useful query is for verified exemplars (requires auto-upsert to populate `verified_leaf_coverage`):

```cypher
MATCH (p:Atom:Decomposed)
WHERE p.verified_leaf_coverage >= 0.8
  AND p.concept_type = $concept_type
RETURN p.fqn, p.repo, p.verified_leaf_coverage, p.topo_hash
ORDER BY p.verified_leaf_coverage DESC
LIMIT 20
```

## Migration plan

### Phase 0: Foundation

Establish the ability to measure generalization before changing anything.

Deliverables:

- Fix `_io_match_factor()` to return a real similarity score (recommendation 1)
- Add domain-agnostic test fixtures and benchmark goals (recommendation 2)
- Implement the `GENERALIZE` refinement action stub (recommendation 8)

These are small, low-risk changes that improve the baseline and provide measurement capability.

### Phase 1: Unified retrieval

Build `TemplateRetriever` on top of existing Memgraph candidate retrieval (recommendation 3).

Deliverables:

- Unified scaffold retrieval interface composing with existing catalog/lexical/skill-index sources
- GraphAlignmentScorer replacing topo_hash + Jaccard for candidate ranking (recommendation 7)
- Retrieval uses stored metadata: abstract_type_class, witness types, contracts (recommendation 6)
- Keep existing curated branches as fallback only
- Emit telemetry when retrieval beats the fallback

### Phase 2: Generalize upstream layers

Address strategy classification and query reformulation so new domains don't require code changes (recommendations 4, 5).

Deliverables:

- Strategy classifier loads phrase rules from data, not module-level code
- Query reformulator loads phrase rules and domain anchors from data
- Alternatively: retrieval-based strategy classification using similar prior goals
- ConceptType enum becomes the single source of truth for ingester prompts (recommendation 12)

### Phase 3: Convert hardcoded paths to templates

Move signal-event-rate, conjugate-update, and orchestrator split patterns into declarative template data (recommendation 9).

Targets:

- Signal event-rate scaffold → template in Memgraph or template index
- Conjugate update 3-node CDG → template (currently does not use skeleton instantiation; should converge with signal-event-rate path)
- 16 orchestrator split patterns → template data keyed by failure context patterns
- Template abstractor domain prefixes/overrides → configuration file

### Phase 4: Auto-upsert solved runs

Build the run-result-to-CDG persistence pipeline (recommendation 10).

Deliverables:

- Conversion from OrchestratorResult/PlannerRunResult → enriched CDG JSON
- Automatic upsert after verified or partially verified runs
- Verification coverage stored as queryable field on root nodes
- Provenance tracking (run ID, goal, execution path, timestamp)

### Phase 5: Retrieval-based refinement

Use retrieved exemplars in orchestrator refinement and single-agent planning (recommendation 11).

Deliverables:

- Refinement searches Memgraph for similar previously-failed-then-resolved nodes
- Retrieved split patterns replace the 16 hardcoded rules
- Cross-round and cross-run learning via auto-upserted exemplars
- LLM refinement becomes the fallback, not the first option

This is the point where Memgraph and graph alignment can genuinely replace most of the current benchmark-shaped orchestration logic.

## Bottom line

Domain knowledge is embedded in six layers of the pipeline, not three. Generalizing only the Architect and orchestrator would leave domain coupling in strategy classification, query reformulation, ingestion, and synthesis.

Today the repo has:

- a useful graph candidate retrieval system (but with a broken scoring function and unused metadata)
- a useful template and skeleton system
- a useful graph store (with rich metadata that retrieval ignores)

What it does not yet have:

- a first-class template retrieval layer
- strong structural matching (topo_hash is too weak, IO match factor is a placeholder)
- automatic accumulation of solved decompositions as reusable memory
- data-driven strategy classification and query reformulation
- domain-agnostic test coverage

The right move is to treat domain-specific solutions as retrievable exemplars rather than runtime exceptions — at every layer, not just decomposition. That makes the system more general without discarding the deterministic and practical engineering advantages it already has.
