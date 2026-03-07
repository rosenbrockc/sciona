# ROADMAP_DONE

## Execution Modes

- Added `rapid`, `structured`, and `verified` execution modes across the main CLI flows.
- Mode selection now gates shared context, graph retrieval, skill-index loading, semantic backend choice, and hunter settings.
- Validation:
  - `conda run -n hpyexec pytest -q`
  - Result at completion time: `1179 passed, 17 skipped`
- Commits:
  - `de1a624` `cli: add execution modes`

## Mode Visibility

- Added runtime mode summaries so command starts show the resolved mode and active feature gates.
- Added run telemetry metadata for `execution_mode` and `mode_features`.
- Validation:
  - `conda run -n hpyexec pytest -q`
  - Result at completion time: `1179 passed, 17 skipped`
- Commits:
  - `7000473` `cli: surface execution mode summaries`

## Direct-Baseline Benchmark Comparisons

- Extended the prompt benchmark harness to run both tuned prompts and a simpler direct-baseline variant.
- Added CLI support with `ageom prompt-benchmark --compare-direct-baseline`.
- Extended unit and E2E regressions to validate the variant split and aggregate reporting.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_prompt_benchmark.py tests/test_e2e_prompt_benchmark.py`
  - Result: `9 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1182 passed, 17 skipped`
- Commits:
  - `c5f42bd` `benchmark: compare direct prompt baselines`

## Conditional Retrieval By Confidence And Mode

- Added a catalog-confidence heuristic so retrieval is now gated by task text instead of always-on defaults.
- `decompose`, `match`, `run`, and `optimize` now derive an effective retrieval policy from execution mode plus catalog confidence.
- Low-confidence tasks now disable heavier retrieval paths and force lexical fallback; high-confidence tasks preserve stronger retrieval paths.
- Added regression coverage for the confidence heuristic and retrieval-policy banding.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_retrieval_policy.py tests/test_execution_modes.py tests/test_cli_skill_index.py tests/test_catalog.py`
  - Result: `49 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1187 passed, 17 skipped`
- Commits:
  - `86a9710` `cli: gate retrieval by catalog confidence`

## Small Full-Flow Benchmark Comparisons

- Added a new `flow_benchmark` harness with small multi-domain end-to-end cases spanning decomposition plus matching.
- The harness compares a `direct_baseline` path against `rapid`, `structured`, and `verified` style full-flow variants.
- Added stable E2E regressions that keep these small task comparisons in-tree.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_flow_benchmark.py tests/test_e2e_flow_benchmark.py`
  - Result: `3 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1190 passed, 17 skipped`
- Commits:
  - `007829b` `benchmark: add small full-flow comparisons`

## Primitive-Binding Confidence Scoring

- Added explicit primitive-binding confidence and provenance fields to algorithmic nodes.
- Deterministic primitive normalization now distinguishes explicit, exact, and token-overlap bindings instead of treating all matches equally.
- Weak token-overlap bindings no longer harden into atomic nodes automatically, and deterministic critique now rejects forced atomic bindings when the score is too weak.
- Added regressions for:
  - confidence metadata on exact normalized bindings
  - weak overlap bindings staying non-atomic
  - deterministic critique rejecting weak forced atomic bindings
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_decomposition.py tests/test_catalog.py`
  - Result: `87 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1192 passed, 17 skipped`
- Commits:
  - `ae5391e` `architect: score primitive binding confidence`

## Graph Invariant Hardening

- Added named invariant failures for deterministic rewrite validation, including:
  - unresolved typed ports
  - primitive-signature violations
  - disconnected children
  - missing typed source-to-sink paths
- Deterministic graph validation now checks rewritten edges, not just node port shapes.
- Added regressions for disconnected-child and invariant-tagged primitive-signature failures.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_decomposition.py`
  - Result: `52 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1194 passed, 17 skipped`
- Commits:
  - `931a4e1` `architect: harden rewrite graph invariants`

## Deterministic Rewrite Action Observability

- Added deterministic rewrite action logs to `build_deterministic_decomposition`, covering:
  - validation wrapper elision
  - primitive normalization
  - routing wrapper elision
  - redundant primitive collapse
  - helper synthesis
  - specialized fallback merges
- Propagated rewrite actions into architect decompose history and snapshot metadata so the non-LLM edits are visible downstream.
- Added regressions for:
  - rewrite action emission during primitive normalization
  - rewrite action emission during routing-wrapper elimination
  - decompose-node history carrying rewrite actions
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_decomposition.py`
  - Result: `53 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1195 passed, 17 skipped`

## Successful Decomposition Template Reuse

- Successful architect decompositions are now persisted into a stable shared-context namespace, `architect/templates`, instead of staying run-local only.
- Future `decompose_node` prompts now search that template namespace and inject compact prior decomposition templates when relevant.
- Added regressions for:
  - decompose prompt injection from the shared template namespace
  - successful `advance_node` writes into the template namespace
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_decomposition.py`
  - Result: `55 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1197 passed, 17 skipped` before a known `hpyexec` post-run bus error on that run; the suite itself completed and reported pass/skip counts
- Commits:
  - `9ca755e` `architect: persist template reuse metrics`

## Shared-Context Reuse Metrics Visibility

- Added explicit template-reuse counters to shared-context metrics:
  - template searches
  - template hits
  - template writes
  - template prompt injections
- Architect template search/write paths now record those counters, and CLI shared-context summaries now print the template-reuse slice directly.
- Added regressions for:
  - standalone template-metric accounting
  - architect decompose template search/injection metrics
  - architect template-write metrics on successful node advancement
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_decomposition.py tests/test_shared_context.py`
  - Result: `62 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1198 passed, 17 skipped`
- Commits:
  - `9ca755e` `architect: persist template reuse metrics`

## Live Provider Failure Reporting

- Prompt-dispatch error events now include provider-supplied failure metadata instead of only a generic exception string.
- Added error metadata capture for:
  - subprocess CLI providers
  - persistent socket shims
  - Gemini socket shim retries/startup failures
- The router now attaches fields such as:
  - failure phase
  - transport type
  - provider/model
  - exit code or timeout context
  - stderr/error excerpts
- Added regressions for:
  - router propagation of client error metadata into `PROMPT_DISPATCH_ERROR`
  - subprocess CLI error metadata capture
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_llm_router.py tests/test_llm.py tests/test_telemetry.py`
  - Result: `71 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1200 passed, 17 skipped`
- Commits:
  - `2f25444` `telemetry: surface live provider failure details`

## Benchmark-Justified Routing Defaults

- Per-prompt LLM overrides are now filtered so only a small benchmark-justified allowlist stays active by default:
  - `architect_strategy`
  - `architect_critique`
  - `hunter_score`
  - `hunter_reformulate`
  - `hunter_analyze_failure`
- Unbenchmarked prompt-specific code defaults no longer create extra provider clients automatically.
- Explicit non-default user overrides are still honored, so this reduces default runtime sprawl without removing escape hatches.
- Added regressions for:
  - benchmark-justified override policy on code defaults
  - explicit override passthrough for unbenchmarked prompt keys
  - router filtering of unbenchmarked default overrides
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_llm_router.py`
  - Result: `50 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1203 passed, 17 skipped`
- Commits:
  - `6b4fa9c` `cli: filter unbenchmarked prompt overrides`

## Prompt Runtime Timeout Profiles

- Added router-enforced timeout profiles for high-volume prompt keys so long-running calls fail on a bounded schedule instead of inheriting one broad client timeout.
- Added default timeout bands for:
  - `architect_strategy`
  - `architect_decompose`
  - `architect_critique`
  - `hunter_score`
  - `hunter_reformulate`
  - `hunter_analyze_failure`
- Added per-prompt environment overrides via `AGEOM_<PROMPT_KEY>_TIMEOUT_S` and a fallback `AGEOM_PROMPT_TIMEOUT_DEFAULT_S`.
- Router timeout failures now surface as explicit `PROMPT_DISPATCH_ERROR` events with `provider_error_phase=router_timeout` and the applied timeout value.
- Added regressions for:
  - router timeout enforcement on a slow prompt client
  - timeout default lookup for benchmarked prompt keys
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_llm_router.py`
  - Result: `52 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1205 passed, 17 skipped`
- Commits:
  - `8d92f04` `llm-router: enforce prompt timeout profiles`

## Runtime Routing Audit Visibility

- Added a CLI routing audit that prints the effective default provider/model plus:
  - active prompt-key overrides
  - suppressed code-default overrides that were filtered out
  - active non-benchmark custom overrides
- Wired the routing audit into the major CLI flows that construct per-round routers:
  - ingester
  - architect
  - hunter
  - synthesizer
  - full `run`
  - optimize/decompose flows that instantiate architect routers
- Added regressions for:
  - structured routing summary content
  - suppressed default override reporting
  - custom non-benchmark override reporting
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_llm_router.py`
  - Result: `54 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1207 passed, 17 skipped`
- Commits:
  - `TBD` `cli: add routing audit summaries`
