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
  - `d00991d` `cli: add routing audit summaries`

## Routing Audit In Run Telemetry

- Persisted the routing audit into `run` telemetry metadata so saved snapshots and the dashboard keep the same provider-routing context printed at startup.
- `algorithm_creation` run metadata now includes round-scoped routing summaries for:
  - architect
  - hunter
- The persisted payload includes:
  - default provider/model
  - active prompt-key overrides
  - suppressed default overrides
  - custom non-benchmark overrides
- Added regression coverage for the telemetry-friendly routing summary transformation.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_llm_router.py`
  - Result: `54 passed`

## Legacy One-Shot Runtime Deprecation

- Marked the legacy one-shot subprocess providers as deprecated:
  - `claude_cli`
  - `codex_cli`
  - `gemini_cli`
- Factory construction now emits a `DeprecationWarning` that points users to the corresponding persistent socket-daemon shim:
  - `claude_shim`
  - `codex_shim`
  - `gemini_shim`
- Kept the legacy providers functional for compatibility while making the primary runtime path explicit.
- Added regressions for:
  - explicit deprecation warnings on legacy provider construction
  - clean router/provider factory tests without expected-warning noise
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_llm.py tests/test_llm_router.py`
  - Result: `66 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1215 passed, 17 skipped`
- Commits:
  - `9b1b33d` `runtime: warn on legacy subprocess providers`

## Release Validation CI Automation

- Added a repo-native GitHub Actions workflow at `.github/workflows/release-validation.yml`.
- The workflow now:
  - installs the project plus the release-validation dependency set
  - runs `ageom release-validate --output build/release_validation`
  - uploads the resulting release-validation bundle as a CI artifact
- This closes the gap between having the validation entrypoint in-tree and actually running it as repeatable repo automation.
- Validation:
  - `conda run -n hpyexec pytest -q`
  - Result: `1215 passed, 17 skipped`
- Commits:
  - `4cbb714` `ci: add release validation workflow`
  - `conda run -n hpyexec pytest -q`
  - Result: `1207 passed, 17 skipped`
- Commits:
  - `99b22e2` `telemetry: persist routing audit metadata`

## Dashboard Routing And Retrieval Summaries

- Added derived dashboard-facing summaries for:
  - retrieval policy
  - architect routing
  - hunter routing
- Exposed those summaries through the dashboard API so the frontend no longer has to parse raw nested metadata blobs to show the active provider mix.
- Updated the telemetry dashboard UI to surface:
  - retrieval confidence/backend on the run list and summary cards
  - architect default LLM plus active override count
  - hunter default LLM plus active override count
- Added regression coverage for the dashboard API summary fields.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_visualizer_api.py`
  - Result: `12 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1208 passed, 17 skipped`
- Commits:
  - `5f0209e` `dashboard: expose routing and retrieval summaries`

## Regression Warning Cleanup

- Set `pytest-asyncio` loop-scope config explicitly to remove the standing deprecation warning.
- Suppressed the known external `torch`-before-`juliacall` warning in pytest so full regressions focus on actionable failures instead of repeated environment noise.
- Validation:
  - `conda run -n hpyexec pytest -q`
  - Result: `1208 passed, 17 skipped`
- Commits:
  - `bb4a4fc` `test: clean regression warning noise`

## Deterministic Benchmark Validation Bundle

- Added a deterministic `benchmark-validate` CLI command that runs:
  - the prompt benchmark suite with an in-repo fixture provider
  - the full-flow benchmark suite
- The command writes a local validation bundle with:
  - `prompt_benchmark.json`
  - `flow_benchmark.json`
  - `summary.json`
- Added a reusable `save_flow_benchmark_report(...)` helper so the flow benchmark now has the same report persistence behavior as the prompt benchmark.
- Added regressions for:
  - flow benchmark report persistence
  - full benchmark validation bundle generation
  - flow benchmark aggregate variants in saved reports
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_benchmark_validation.py tests/test_prompt_benchmark.py tests/test_flow_benchmark.py`
  - Result: `12 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1211 passed, 17 skipped`
- Commits:
  - `66d8281` `benchmark: add deterministic validation bundle`

## Repo-Native Release Validation Entry Point

- Added a deterministic `release-validate` CLI command as a repo-native release validation entrypoint.
- The command wraps the benchmark validation bundle and writes:
  - `release_validation.json`
  - nested benchmark reports under `benchmarks/`
- Added a dedicated `ageom.release_validation` backend so release validation is a product-level API, not just an ad hoc CLI composition.
- Added regressions for:
  - release validation manifest generation
  - benchmark bundle presence inside the release validation directory
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_release_validation.py tests/test_benchmark_validation.py`
  - Result: `4 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1212 passed, 17 skipped`
- Commits:
  - `d018a70` `cli: add release validation entrypoint`

## Persistent Architect Failure Memory

- Added a stable `architect/failures` shared-context namespace for critique rejection patterns.
- Rejected architect critiques now persist compact failure summaries with:
  - parent name/description
  - rejection category
  - rejection reason
  - flagged nodes when present
- Future `decompose_node` prompts now inject relevant prior failure patterns alongside shared context and successful templates.
- Added regressions for:
  - decompose prompt injection from the failure namespace
  - critique rejection persistence into the failure namespace
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_decomposition.py`
  - Result: `57 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1214 passed, 17 skipped`
- Commits:
  - `8827e91` `architect: reuse critique failure patterns`

## Witness-Backed Source Primitive Extraction

- Strengthened source-derived primitive extraction to use witness metadata more effectively.
- Live registry imports now fall back to witness docstrings when wrapper implementations are under-documented.
- AST fallback extraction now:
  - reads the witness function referenced by `@register_atom(...)`
  - uses witness docstrings when the wrapper has no useful description
  - uses witness input/output signatures when the wrapper is generic or untyped
  - strips quoted forward-reference annotations into clean catalog type text
- Added regressions for:
  - live registry description fallback to witness docstrings
  - AST fallback using witness signatures for generic wrappers
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_source_catalog.py tests/test_catalog.py`
  - Result: `44 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1217 passed, 17 skipped`
- Commits:
  - `3c1fcf8` `catalog: enrich witness-backed source extraction`

## Catalog And Atom Registry Alignment Metrics

- Extended `CatalogReport` so source-derived catalog seeding now records:
  - live-registry candidates
  - AST-fallback candidates
  - CDG metadata matches
  - witness-doc fallbacks
  - witness-signature fallbacks
- Wired those counters into architect catalog loading so CLI runs print a source-alignment summary instead of only raw add/merge counts.
- Added regressions for:
  - live-registry candidate accounting
  - CDG metadata match accounting
  - AST-fallback candidate accounting
  - witness-doc and witness-signature fallback accounting
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_source_catalog.py tests/test_catalog.py`
  - Result: `44 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1217 passed, 17 skipped`
- Commits:
  - `00b352a` `catalog: surface source alignment metrics`

## Rapid Mode Router Simplification

- Made `rapid` mode reduce actual runtime complexity instead of only changing surrounding feature gates.
- In `rapid` mode, code-default prompt-key overrides are now suppressed so the router stays on the round default provider/model unless the user explicitly changed an override.
- This keeps:
  - benchmark-tuned default overrides in `structured` and `verified`
  - explicit user escape hatches in `rapid`
- Routing summaries and run telemetry now include the effective execution mode for each round so the simplification is visible.
- Added regressions for:
  - `rapid` mode suppressing benchmark-default overrides in router construction
  - `rapid` mode routing summaries reporting benchmark-default suppression
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_llm_router.py tests/test_execution_modes.py`
  - Result: `60 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1219 passed, 17 skipped`
- Commits:
  - `a933a23` `runtime: simplify rapid mode routing`
