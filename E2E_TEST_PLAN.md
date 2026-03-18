# E2E Generalization Test Plan

## Overview

7 end-to-end test files verifying the AGEO-Matcher generalization works across grounding, synthesis, export, and profiling.

---

## 1. `tests/test_e2e_cross_domain_grounding.py`

**Purpose**: Verify the generalized pipeline produces handoff-ready CDGs for non-ECG domains (graph algorithms, linear algebra, sorting). Exercises decomposition → handoff validation → PDG conversion → assembly.

**Tests**:
1. `test_dijkstra_decompose_produces_atomic_nodes` — Decompose "Find shortest path in weighted graph", assert ATOMIC leaves and `cdg.is_handoff_ready()`.
2. `test_cholesky_decompose_produces_atomic_nodes` — Decompose "Solve SPD linear system via Cholesky", assert leaves have non-empty `type_signature`.
3. `test_merge_sort_decompose_produces_atomic_nodes` — Decompose merge sort, assert >= 2 ATOMIC leaves.
4. `test_cross_domain_handoff_validation` — For each domain CDG, `validate_handoff()` and `validate_handoff_strict()` return no issues.
5. `test_cross_domain_pdg_conversion` — `to_pdg_nodes()` returns one PDGNode per ATOMIC leaf with non-empty `statement` and `informal_desc`.
6. `test_cross_domain_assembler_produces_valid_python` — `Assembler.assemble()` output parses with `ast.parse()`, `sorry_count == 0`.
7. `test_non_signal_cdg_has_no_signal_concept_types` — No node has `SIGNAL_FILTER` or `SIGNAL_TRANSFORM` concept type.

**Key mocks**: `DomainAwareLLM` (returns strategy/decompose/critique JSON per domain), `_empty_skill_index()`, `_make_catalog(domain_spec)`.

---

## 2. `tests/test_e2e_refinement_cascade.py`

**Purpose**: Test the full orchestrator 3-layer refinement cascade: retrieval → deterministic → LLM, plus multi-round orchestration.

**Tests**:
1. `test_retrieval_hit_skips_deterministic_and_llm` — Template retriever returns confidence=0.85, assert sub-nodes come from retrieval, telemetry shows `SPLIT_RETRIEVAL`.
2. `test_deterministic_fallback_when_retrieval_below_threshold` — Retriever returns confidence=0.5 (below 0.7), node matches "ecg bandpass filter" pattern, assert deterministic split fires with `SPLIT_DETERMINISTIC`.
3. `test_llm_fallback_when_no_deterministic_match` — No retriever match, no pattern match, LLM returns split JSON. Assert `SPLIT_LLM`.
4. `test_ungroundable_action_marks_rejected` — `FailureAction.UNGROUNDABLE` → node status becomes REJECTED.
5. `test_generalize_action_strips_algorithm_names` — `FailureAction.GENERALIZE` strips "Dijkstra" from description, clears `type_signature`, telemetry shows `GENERALIZE_APPLIED`.
6. `test_run_orchestration_multi_round` — CDG with 3 leaves, Hunter fails 1 in round 1, deterministic split produces 2 sub-nodes, Hunter succeeds all in round 2. Assert `rounds_used == 2`, `all_matched`.
7. `test_run_orchestration_max_rounds_reached` — Hunter always fails, `max_rounds=2`. Assert `ungroundable` is non-empty.

**Key mocks**: `_mock_template_retriever(confidence)`, `_mock_hunter_agent(succeed_ids, fail_ids)`, `_mock_llm_for_split()`.

---

## 3. `tests/test_e2e_synthesis_cross_domain.py`

**Purpose**: Verify synthesizer produces valid Python for non-signal domains with no signal-alias contamination.

**Tests**:
1. `test_graph_algo_skeleton_is_valid_python` — Assemble graph-algo CDG, `ast.parse()` succeeds, `sorry_count == 0`.
2. `test_linalg_skeleton_is_valid_python` — Same for linear algebra CDG.
3. `test_sorting_skeleton_is_valid_python` — Same for sorting CDG.
4. `test_no_signal_alias_in_graph_algo_skeleton` — No `"ecg"`, `"ppg"` strings in source code.
5. `test_no_signal_alias_in_linalg_skeleton` — Same for linalg.
6. `test_composition_function_named_correctly` — Composition function uses `sanitize_name(root.name)`.
7. `test_pipeline_py_no_signal_aliases` — `generate_pipeline_py()` output is valid Python.
8. `test_skeleton_units_match_cdg_leaves` — Every skeleton unit corresponds to a CDG leaf.

**Key helpers**: `_build_graph_algo_cdg()`, `_build_linalg_cdg()`, `_build_sorting_cdg()`, `_make_match()`, `_assemble()`.

---

## 4. `tests/test_e2e_export_roundtrip.py`

**Purpose**: CDG export/import roundtrip, result_to_cdg conversion, exemplar JSON validity.

**Tests**:
1. `test_cdg_save_load_roundtrip` — `save_json → load_json` preserves nodes, edges, metadata.
2. `test_cdg_model_dump_roundtrip` — `model_dump → model_validate` preserves structure.
3. `test_result_to_cdg_preserves_structure` — Correct node count and `verified_leaf_coverage`.
4. `test_result_to_cdg_sanitize_idempotent` — Running `sanitize_cdg()` twice is a no-op.
5. `test_exemplar_json_is_valid_cdg` (parametrized over all exemplar files) — Each exemplar parses, has ATOMIC leaves, edges, decomposed root.
6. `test_conjugate_exemplar_has_3_atomic_nodes` — Beta-Bernoulli exemplar has 3 atomic nodes with expected names.
7. `test_signal_event_rate_exemplar_has_3_stage_pipeline` — 3 atomic nodes, 2 edges, linear pipeline, coverage=1.0.

---

## 5. `tests/test_e2e_telemetry_paths.py`

**Purpose**: Each execution path emits expected telemetry events with correct payloads.

**Tests**:
1. `test_orchestration_emits_round_start_and_done` — Events contain `ROUND_START`, `HUNTER_ROUND_DONE`, `ORCHESTRATION_DONE`.
2. `test_orchestration_emits_round_all_matched` — `ROUND_ALL_MATCHED` when hunter succeeds on all.
3. `test_refinement_split_retrieval_emits_event` — `SPLIT_RETRIEVAL` with confidence and source.
4. `test_refinement_split_deterministic_emits_event` — `SPLIT_DETERMINISTIC` with sub_node_count.
5. `test_refinement_split_llm_emits_event` — `SPLIT_LLM` with sub_node_count.
6. `test_orchestration_max_rounds_emits_event` — `MAX_ROUNDS_REACHED` with ungroundable payload.
7. `test_log_event_populates_all_fields` — Direct `log_event()` call, verify all fields set.
8. `test_telemetry_scope_sets_context` — `telemetry_scope()` propagates `run_id` and `stage`.

**Fixture**: `clean_telemetry` (autouse) calls `reset_telemetry_runtime()` before each test.

---

## 6. `tests/test_e2e_rule_parity.py`

**Purpose**: JSON-loaded rules produce identical behavior to old hardcoded rules.

**Tests**:
1. `test_split_patterns_json_loads_successfully` — `_load_split_patterns()` returns >= 16 patterns.
2. `test_split_patterns_json_has_required_fields` — Each pattern has `name`, `conditions`, `sub_nodes`.
3. `test_deterministic_split_matches_expected_patterns` (parametrized) — 11 known inputs each produce expected sub-node count.
4. `test_phrase_rules_json_loads_successfully` — `_load_phrase_rules()` returns non-empty rules.
5. `test_strategy_classifier_with_json_rules` — Classify ECG/Dijkstra/LCS goals, assert correct ConceptType.
6. `test_query_rules_json_loads_successfully` — `_load_query_rules()` returns non-empty anchors and rules.
7. `test_query_phrase_rules_return_expected_queries` — ECG bandpass rule matches expected input.
8. `test_split_pattern_conditions_match_expected_strings` — Each pattern's conditions match a constructed test string.

---

## 7. `tests/test_e2e_upsert_retrieval_loop.py`

**Purpose**: Validate the flywheel: solved run → auto-upsert → retrieval for similar goals.

**Tests**:
1. `test_result_to_cdg_produces_upsertable_dict` — Output has nodes/edges, coverage=1.0, provenance on root.
2. `test_auto_upsert_skipped_for_low_coverage` — 0/3 matched → coverage=0.0, below threshold.
3. `test_auto_upsert_fires_for_sufficient_coverage` — 3/3 matched → coverage=1.0, mock store upsert called.
4. `test_upserted_run_becomes_retrieval_candidate` — Upserted CDG returned by mock store, TemplateRetriever returns match with confidence >= 0.5.
5. `test_sanitize_cdg_preserves_matched_primitives` — `matched_primitive` values survive sanitization.
6. `test_provenance_metadata_survives_sanitize` — Provenance dict with all fields survives sanitization.
7. `test_flywheel_telemetry_events` — `AUTO_UPSERT_COMPLETED` and `AUTO_UPSERT_SKIPPED_LOW_COVERAGE` events emitted correctly.

---

## Implementation Notes

- **Telemetry isolation**: Every test touching telemetry uses `reset_telemetry_runtime()` in an autouse fixture.
- **No live services**: All tests mock GraphStore, LLM, and Hunter. No Memgraph or external API calls.
- **langgraph guard**: Only test file #1 needs the `importlib.util.find_spec("langgraph")` skip guard.
- **Shared patterns**: Follow `test_e2e_hodges_pipeline.py` for CDG construction, match result building, assembler invocation. Follow `test_orchestrator.py` for refinement mocking patterns.
- **All exemplar files**: Parametrize over `ageom/data/exemplars/*.json` for validation.

## Priority

| # | Test File | Risk Covered |
|---|-----------|-------------|
| 1 | cross_domain_grounding | Core generalization — does it work beyond ECG? |
| 2 | refinement_cascade | Biggest architectural change — 3-layer cascade |
| 3 | upsert_retrieval_loop | Flywheel that makes the system self-improving |
| 4 | synthesis_cross_domain | Signal-alias contamination in generated code |
| 5 | telemetry_paths | Observability for debugging production runs |
| 6 | rule_parity | Guards against regression from JSON externalization |
| 7 | export_roundtrip | Data integrity for CDG persistence |
