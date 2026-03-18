# Deterministic Tools

AGEO-Matcher is deterministic-first: regexes, AST transforms, retrieval heuristics,
verification oracles, and typed templates run before any LLM fallback.

The active deterministic prompt wrappers are wired in
`ageom/commands/_helpers.py`.

## Prompt Key Coverage

The system has 17 LLM prompt keys.

| Prompt Key | Phase | Deterministic Tool | Coverage |
|-----------|-------|-------------------|----------|
| `architect_strategy` | Architect | `StrategyClassifier` | Covered |
| `architect_decompose` | Architect | `DeterministicDecomposer` | Partial |
| `architect_critique` | Architect | `DeterministicCritic` + `structural_critique_issues` | Partial |
| `hunter_score` | Hunter | `EmbeddingReranker` + `HeuristicCandidateRanker` | Covered |
| `hunter_reformulate` | Hunter | `HeuristicQueryReformulator` + optional `EmbeddingQueryExpander` | Covered |
| `hunter_analyze_failure` | Hunter | `DeterministicFailureAnalyzer` | Covered |
| `synthesizer_repair` | Synthesizer | `classify_error` + `suggest_deterministic_fix` + `DeterministicFix` | Partial |
| `synthesizer_tactic` | Synthesizer | `DeterministicTacticSuggester` | Partial |
| `orchestrator_refine` | Orchestrator | `_deterministic_split_subnodes` + `_split_on_connectors` | Partial |
| `ingester_chunk` | Ingester | `propose_macro_atoms` + chunker heuristics | Partial |
| `ingester_decompose` | Ingester | `decompose_function` | Covered |
| `ingester_hoist_state` | Ingester | `ASTStateHoister` | Partial |
| `ingester_abstract` | Ingester | `TemplateAbstractor` | Partial |
| `ingester_fix_type` | Ingester | `DeterministicTypeFixer` | Partial |
| `ingester_fix_ghost` | Ingester | `DeterministicGhostFixer` | Partial |
| `ingester_opaque_witness` | Ingester | `TemplateWitnessGenerator` | Partial |
| `ingester_fix_message_cycle` | Ingester | `DeterministicCycleBreaker` | Partial |

Summary: 5 covered, 12 partial, 0 with zero deterministic support.

## Active Deterministic Routers

### Architect

- `StrategyClassifier`
  - File: `ageom/architect/strategy_classifier.py`
  - Role: phrase-rule goal classification for `architect_strategy`

- `DeterministicDecomposer`
  - File: `ageom/architect/deterministic_decompose.py`
  - Role: skeleton-backed decomposition and structured fallback for `architect_decompose`

- `DeterministicCritic`
  - File: `ageom/architect/deterministic_critic.py`
  - Role: deterministic approval/rejection path for `architect_critique`

- `structural_critique_issues`
  - File: `ageom/architect/structural_critic.py`
  - Role: graph structure and IO validation used by architect critique

### Hunter

- `EmbeddingReranker`
  - File: `ageom/hunter/embedding_reranker.py`
  - Role: embedding-based candidate ranking for `hunter_score`

- `HeuristicCandidateRanker`
  - File: `ageom/hunter/candidate_ranker.py`
  - Role: token overlap and domain bonus fallback for `hunter_score`

- `DeterministicFailureAnalyzer`
  - File: `ageom/hunter/failure_analyzer.py`
  - Role: regex-based compiler/runtime failure analysis for `hunter_analyze_failure`

- `HeuristicQueryReformulator`
  - File: `ageom/hunter/query_reformulator.py`
  - Role: phrase-rule and keyword query generation for `hunter_reformulate`

- `EmbeddingQueryExpander`
  - File: `ageom/hunter/embedding_query_expander.py`
  - Role: optional embedding-assisted query diversification for `hunter_reformulate`

### Ingester

- `decompose_function`
  - File: `ageom/ingester/control_flow_decomposer.py`
  - Role: AST control-flow decomposition for `ingester_decompose`

- `propose_macro_atoms`
  - File: `ageom/ingester/chunker.py`
  - Role: deterministic macro-atom fallback for opaque chunking cases

- `ASTStateHoister`
  - File: `ageom/ingester/ast_state_hoister.py`
  - Role: state model extraction for `ingester_hoist_state`

- `TemplateAbstractor`
  - File: `ageom/ingester/template_abstractor.py`
  - Role: template abstraction generation for `ingester_abstract`

- `DeterministicTypeFixer`
  - File: `ageom/ingester/deterministic_type_fixer.py`
  - Role: mypy/type-fix wrapper for `ingester_fix_type`

- `DeterministicGhostFixer`
  - File: `ageom/ingester/deterministic_ghost_fixer.py`
  - Role: witness/ghost error repair for `ingester_fix_ghost`

- `TemplateWitnessGenerator`
  - File: `ageom/ingester/template_witness_generator.py`
  - Role: template witness generation for `ingester_opaque_witness`

- `DeterministicCycleBreaker`
  - File: `ageom/ingester/deterministic_cycle_breaker.py`
  - Role: message-cycle repair for `ingester_fix_message_cycle`

### Synthesizer

- `classify_error` and `suggest_deterministic_fix`
  - File: `ageom/synthesizer/classifier.py`
  - Role: classify and map common compile/type failures to deterministic repairs

- `DeterministicFix`
  - File: `ageom/synthesizer/repair.py`
  - Role: apply deterministic repair batches before LLM repair

- `DeterministicTacticSuggester`
  - File: `ageom/synthesizer/tactic_suggester.py`
  - Role: simple Lean/Coq/Python tactic or implementation fallback for `synthesizer_tactic`

### Orchestrator

- `_deterministic_split_subnodes`
  - File: `ageom/orchestrator.py`
  - Role: domain-template refinement for `orchestrator_refine`

- `_split_on_connectors`
  - File: `ageom/orchestrator.py`
  - Role: connector-based textual split fallback for `orchestrator_refine`

### Planner Policy

- `_select_policy`
  - File: `ageom/services/planner_service.py`
  - Role: route simple vs compound goals using marker and size heuristics

- `_is_compound_goal`
  - File: `ageom/services/planner_service.py`
  - Role: phrase-based compound-goal detection

## Deterministic Oracles

### Verification

| Oracle | File | Method |
|--------|------|--------|
| `LeanEnvironment` | `ageom/judge/lean_env.py` | Lean compile/check |
| `CoqEnvironment` | `ageom/judge/coq_env.py` | Coq compile/check |
| `PythonEnvironment` | `ageom/judge/python_env.py` | importability, callability, and arity/type checks |

### Catalog and Retrieval

| Oracle | File | Method |
|--------|------|--------|
| `PrimitiveCatalog` | `ageom/architect/catalog.py` | exact/alias lookup + token confidence |
| `SkillIndex` | `ageom/architect/embedder.py` | embedding similarity over skill primitives |
| `SemanticIndexImpl` | `ageom/indexer/builder.py` | FAISS-backed declaration retrieval |
| `LexicalSemanticIndex` | `ageom/indexer/fallback_index.py` | FAISS-free lexical fallback retrieval |

## Remaining Gaps

These areas still rely on LLM fallback often enough that more deterministic work would help.

- `architect_decompose`
  - Current wrapper is helpful, but high-confidence goals still often invoke the LLM instead of emitting a full CDG directly.

- `architect_critique`
  - `DeterministicCritic` can approve/reject many cases, but semantic coverage is still heuristic and conservative.

- `synthesizer_repair`
  - Deterministic fixes handle common compile errors, but novel proof obligations and complex implementation gaps still go to the LLM.

- `synthesizer_tactic`
  - `DeterministicTacticSuggester` handles only simple Lean/Coq closers and basic Python delegation cases.

- `ingester_chunk`
  - Outside opaque or structurally simple classes, chunking still depends heavily on LLM decomposition.

- `ingester_*` abstraction/repair tools
  - The new wrappers reduce LLM load, but most are intentionally conservative and fall back on ambiguous inputs.
