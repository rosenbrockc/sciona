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

## Dashboard Shared-Context Visibility

- Algorithm-creation telemetry runs now persist final shared-context snapshots directly into run metadata instead of requiring separate sidecar inspection.
- Dashboard/API summaries now expose:
  - active shared-context counts
  - total searches, hits, writes, and prompt injections
  - template reuse searches, hits, writes, and injections
  - backend list and metrics file path
- Dashboard UI run cards and detail summaries now show shared-context usage alongside routing, retrieval, and benchmark summaries.
- Added API regressions for shared-context summary derivation from persisted run metadata.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_visualizer_api.py`
  - `conda run -n hpyexec pytest -q`

## Failure-Pattern Reuse Metrics

- Added failure-pattern reuse counters to shared-context metrics, separate from generic context and template reuse:
  - failure searches
  - failure hits
  - failure writes
  - failure prompt injections
- Architect failure-context search/write paths now record those counters explicitly.
- Hunter failure-context retrieval and writeback paths now record the same counters, so repeated mismatch patterns are measurable across refinement loops.
- Dashboard shared-context summaries now surface failure reuse totals alongside generic and template reuse metrics.
- Added regressions for:
  - failure-metric accounting in `SharedContextMetrics`
  - architect failure-context search/write instrumentation
  - hunter failure-context search/write instrumentation
  - dashboard failure-summary derivation
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_shared_context.py tests/test_decomposition.py tests/test_hunter.py tests/test_visualizer_api.py`
  - `conda run -n hpyexec pytest -q`

## Structured-Mode Runtime Simplification

- Tightened prompt-key override policy so `structured` mode now behaves like `rapid` for routing simplicity:
  - benchmark-justified code-default overrides are suppressed
  - explicit user overrides still apply
- This keeps `verified` as the only mode that enables the benchmark-tuned multi-provider routing defaults automatically.
- Updated routing-summary regressions and router-construction tests to assert the new `structured` behavior.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_llm_router.py tests/test_execution_modes.py`
  - `conda run -n hpyexec pytest -q`

## Persistent Shim Prewarm

- Added `warmup()` support to the persistent socket-based shim clients and to `LLMRouter`.
- Main CLI flows now prewarm router/client pools during setup so socket/auth/startup failures surface before long-running stages begin.
- This also reduces first-call jitter for persistent shims by establishing their worker pools up front.
- Added router regressions for warmup de-duplication across shared override clients.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_llm_router.py tests/test_llm.py`
  - `conda run -n hpyexec pytest -q`

## Benchmark Stability Summaries

- Prompt and flow benchmark aggregates now track repeat-group stability in addition to pass/fail and latency.
- Added aggregate fields for:
  - repeat group count
  - stable group count
  - stability rate
- Prompt and flow benchmark summaries now show a `stable` column so repeated-run consistency is visible in the text and JSON artifacts.
- Added regressions for:
  - prompt benchmark stability on repeated runs
  - flow benchmark repeat grouping and summary formatting
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_prompt_benchmark.py tests/test_flow_benchmark.py tests/test_e2e_prompt_benchmark.py tests/test_e2e_flow_benchmark.py`
  - `conda run -n hpyexec pytest -q`

## Flow Benchmark Prompt Volume

- Flow benchmark results now record prompt-call counts in addition to latency and pass/fail outcome.
- Flow benchmark aggregates now summarize:
  - total prompt calls
  - average prompt calls per case
- The text summary now shows `avg prompts`, which makes it easier to compare direct baseline vs `rapid` / `structured` / `verified` on search cost as well as correctness and latency.
- Added regressions for flow-benchmark prompt-call aggregation and summary formatting.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_flow_benchmark.py tests/test_e2e_flow_benchmark.py`
  - `conda run -n hpyexec pytest -q`

## Validation Summary Visibility

- Benchmark validation summaries now persist the newer benchmark signals instead of only the original pass/fail counts:
  - prompt stability summary
  - flow stability summary
  - average flow prompt calls by variant
- CLI telemetry metadata for `benchmark-validate` and `release-validate` now carries those fields through to persisted runs.
- Dashboard/API benchmark summaries now expose those values, and the dashboard UI shows them in run cards/details.
- Added regressions for:
  - benchmark bundle summary contents
  - validation telemetry persistence
  - dashboard API benchmark summary derivation
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_benchmark_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - `conda run -n hpyexec pytest -q`

## Source-Catalog Alignment Visibility

- The architect catalog loader now returns structured source-alignment metadata instead of only printing it.
- Algorithm-creation telemetry runs now persist catalog/source alignment details, including:
  - catalog size
  - total source candidates
  - added and merged primitives
  - structural skips
  - live-registry, AST, and CDG-matched counts
  - witness-doc and witness-signature fallback counts
- Dashboard/API now summarize that alignment data so source-derived catalog quality is visible alongside routing, retrieval, and shared-context usage.
- Added dashboard regressions for catalog-alignment summary derivation.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_visualizer_api.py tests/test_llm_router.py tests/test_execution_modes.py`
  - `conda run -n hpyexec pytest -q`

## Release Validation Gating

- `release_validation.json` is now a real gate instead of always reporting `passed`.
- Release validation now fails when non-baseline benchmark regressions are present:
  - tuned prompt benchmark failures
  - tuned prompt unstable groups
  - non-baseline flow benchmark failures
  - non-baseline flow unstable groups
- Benchmark validation summaries now persist those gating counts directly so release validation and dashboard summaries can consume them without re-reading aggregate files.
- Added regressions for:
  - benchmark summary gating fields
  - failed release-validation manifests when regressions exist
  - telemetry/dashboard persistence of the new gating counts
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - `conda run -n hpyexec pytest -q`

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

## Validation Telemetry And Dashboard Summaries

- Promoted `benchmark-validate` and `release-validate` into first-class telemetry runs instead of leaving them as opaque JSON bundle writers.
- Those commands now persist run metadata for:
  - prompt benchmark counts and summary text
  - flow benchmark counts and summary text
  - release manifest / benchmark bundle locations
- Extended the dashboard API to derive a `benchmark_summary` section from run metadata.
- Updated the dashboard UI so validation runs show benchmark counts and release status directly in the run list and summary panel.
- Added regressions for:
  - benchmark validation telemetry persistence
  - release validation telemetry persistence
  - dashboard benchmark/release summary extraction
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `15 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1222 passed, 17 skipped`
- Commits:
  - `c95ddd6` `telemetry: surface validation benchmark summaries`

## Dotted Source Name Alias Expansion

- Improved source-derived primitive alignment for registry names that use dotted library-style names such as `scipy.linalg.solve`.
- Source seeding now adds generic suffix aliases for dotted names, including:
  - leaf function name, e.g. `solve`
  - two-part dotted suffix, e.g. `linalg.solve`
  - space and underscore variants, e.g. `linalg solve`, `linalg_solve`
- This improves conceptual matching against source-derived atoms without adding task-specific exceptions.
- Added regressions for dotted registration-name alias expansion during AST fallback seeding.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_source_catalog.py tests/test_catalog.py`
  - Result: `45 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1223 passed, 17 skipped`
- Commits:
  - `da55c5a` `catalog: add dotted source name aliases`

## Witness-Derived Alias Expansion

- Extended source-derived alias generation to use witness names in addition to implementation names and registration names.
- Witness aliases now retain both:
  - the raw witness symbol, e.g. `witness_linear_solve`
  - the stripped conceptual alias, e.g. `linear_solve`
- This improves alignment when wrappers are generic but witness names carry the real algorithmic meaning.
- Added regressions for witness-derived aliases during source seeding.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_source_catalog.py tests/test_catalog.py`
  - Result: `45 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1223 passed, 17 skipped`
- Commits:
  - `977dedc` `catalog: add witness source aliases`

## Module-Derived Alias Expansion

- Extended source-derived alias generation to use module path context for simple atom names.
- Atoms defined in namespaced modules now get generic aliases such as:
  - `signal.butter`
  - `signal butter`
  - `signal_butter`
  - `scipy.signal.butter`
  - `scipy signal butter`
  - `scipy_signal_butter`
- This improves matching against source-derived atoms when the registry name is short but the module namespace carries important semantics.
- Added regressions for module-derived alias expansion during source seeding.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_source_catalog.py tests/test_catalog.py`
  - Result: `46 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1224 passed, 17 skipped`
- Commits:
  - `1484cdb` `catalog: add module-derived source aliases`

## Provider Complexity Dashboard Summary

- Added a dashboard-facing provider complexity summary derived from routing metadata.
- Each run now exposes:
  - unique provider count
  - unique provider/model count
  - transport class count
  - provider list
  - provider/model list
  - transport list
- Transport classes are normalized into generic buckets:
  - `persistent_shim`
  - `legacy_cli`
  - `local_server`
  - `api`
  - `other`
- Updated the dashboard UI to show provider counts and transport summaries directly.
- Added regressions for provider-complexity extraction on dashboard run payloads.
- Validation:
  - `conda run -n hpyexec pytest -q tests/test_visualizer_api.py`
  - Result: `13 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1224 passed, 17 skipped`
- Commits:
  - `b28f82a` `dashboard: summarize provider complexity`

## Command Telemetry Persistence

- Promoted `decompose` and `match` into first-class persisted telemetry runs so they show up in the dashboard alongside `run`, validation, and benchmark flows.
- Each command now records:
  - execution mode and mode feature summary
  - retrieval policy summary
  - catalog alignment summary
  - prompt-routing summary for its active round
  - shared-context metrics in run metadata
- Added focused regressions for:
  - persisted `decompose` run snapshots
  - persisted `match` run snapshots
  - stage completion and shared-context metadata for both commands
- Validation:
  - `pytest -q tests/test_cli_command_telemetry.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `18 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1236 passed, 17 skipped`
- Commit:
  - `telemetry: persist decompose and match runs`

## Architect Dashboard Summary

- Added a first-class `architect_summary` to dashboard run payloads so architect failures are visible without reading raw nested metadata.
- The summary now surfaces:
  - unresolved leaf count
  - blocked node count and blocked node names
  - blocked reason and last architect node name
  - Any-type port and edge percentages
  - critique reject totals and top reject categories
  - retry totals and rewrite-action counts
- Updated the dashboard UI to use the derived architect summary in both the run list and the detail panel.
- Added regressions for architect-summary extraction in dashboard API responses.
- Validation:
  - `pytest -q tests/test_visualizer_api.py tests/test_cli_command_telemetry.py tests/test_validation_telemetry.py`
  - Result: `19 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1237 passed, 17 skipped`
- Commit:
  - `dashboard: summarize architect state`

## Hunter Retrieval Quality Summary

- Added first-class `hunter_metrics` telemetry so Hunter search and verification quality is visible in persisted runs.
- Hunter now records:
  - search iterations
  - embedding/type result volume
  - new candidate counts and candidate pool size
  - ranking call counts
  - verification batch counts and success/failure totals
  - reformulation and fallback counts
  - last query and last verified candidate
- Added a derived `hunter_summary` to dashboard run payloads and updated the dashboard UI to surface search yield, verification volume, and reformulation churn directly.
- Added regressions for:
  - Hunter telemetry metadata persistence during a real agent run
  - dashboard extraction of the new hunter summary
- Validation:
  - `pytest -q tests/test_hunter.py tests/test_visualizer_api.py`
  - Result: `26 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1239 passed, 17 skipped`
- Commit:
  - `telemetry: summarize hunter retrieval quality`

## Catalog Source Breakdown Summary

- Extended source-derived catalog reporting to retain per-source contribution counts instead of only global totals.
- Catalog alignment metadata now carries, per source package:
  - added primitive count
  - live-registry candidate count
  - AST-fallback candidate count
- Added dashboard/API summaries for:
  - source package count
  - top contributing sources by added primitives
- Updated the dashboard UI to show the leading contributing source packages directly in run cards and run details.
- Added regressions for:
  - source-breakdown accounting during source-derived catalog seeding
  - dashboard extraction of top source contributors
- Validation:
  - `pytest -q tests/test_source_catalog.py tests/test_visualizer_api.py`
  - Result: `25 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1239 passed, 17 skipped`
- Commit:
  - `dashboard: surface catalog source breakdown`

## Benchmark Latency Summaries

- Extended benchmark validation summaries to carry latency aggregates instead of only correctness and prompt-volume aggregates.
- Validation summaries now persist:
  - prompt average latency by provider/variant
  - flow average latency by mode
- The dashboard API now exposes those latency aggregates in `benchmark_summary`, and the dashboard UI shows them in run cards and run details.
- Added regressions for:
  - benchmark validation bundle latency fields
  - dashboard extraction of prompt and flow latency summaries
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_visualizer_api.py`
  - Result: `19 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1239 passed, 17 skipped`
- Commit:
  - `benchmark: surface latency summaries`

## Legacy Provider Policy

- Tightened runtime policy so legacy one-shot `*_cli` providers are no longer silently available by default.
- `create_llm_client(...)` now rejects `claude_cli`, `codex_cli`, and `gemini_cli` unless legacy subprocess providers are explicitly enabled.
- Added a central config escape hatch:
  - `AGEOM_ALLOW_LEGACY_SUBPROCESS_PROVIDERS=true`
- CLI round/default client creation now threads that policy through from config instead of relying on warning-only behavior.
- Kept the legacy clients available for deliberate compatibility use, but only behind explicit opt-in.
- Added regressions for:
  - default rejection of legacy subprocess providers
  - explicit opt-in still creating the legacy client and preserving the deprecation warning
  - router helper compatibility with config stubs after the new policy field
- Validation:
  - `pytest -q tests/test_llm.py tests/test_llm_router.py`
  - Result: `74 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1241 passed, 17 skipped`
- Commit:
  - `runtime: require opt-in for legacy cli providers`

## Catalog Merge Visibility

- Extended catalog-alignment telemetry to persist concrete merge examples instead of only a merged-count total.
- Catalog run metadata now carries top merge examples from `CatalogReport.merge_details`, including:
  - candidate primitive name
  - incumbent primitive name
  - similarity score
- Added dashboard/API summaries for those merge examples and surfaced them in the run list and detailed run view.
- Added regressions for dashboard extraction of merge examples from catalog-alignment metadata.
- Validation:
  - `pytest -q tests/test_catalog.py tests/test_visualizer_api.py`
  - Result: `53 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1241 passed, 17 skipped`
- Commit:
  - `dashboard: surface catalog merge examples`

## Release Complexity Policing

- Added a first-class runtime-complexity gate to release validation so a release can now fail even when correctness benchmarks pass if the default routing surface becomes too complex.
- Release validation now computes and persists a runtime-complexity summary over the package’s default `verified`-mode routing, including:
  - provider count
  - provider/model count
  - transport count
  - active override count
  - legacy provider presence
- Added explicit budget checks for:
  - maximum provider count
  - maximum provider/model count
  - maximum transport count
  - zero legacy one-shot subprocess providers
- Fixed the release-validation CLI telemetry path to stop hardcoding `status=passed`; it now persists the actual release-validation status and runtime-complexity payload.
- Added regressions for:
  - passing runtime-complexity summaries in the manifest
  - failing release validation when runtime complexity exceeds budget
  - release-validation telemetry carrying the actual status and runtime-complexity details
- Validation:
  - `pytest -q tests/test_release_validation.py tests/test_validation_telemetry.py`
  - Result: `5 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1242 passed, 17 skipped`
- Commit:
  - `release: police runtime complexity budget`

## Benchmark Runtime Complexity Summary

- Promoted runtime-complexity budgeting into the benchmark-validation bundle itself instead of keeping it as release-only logic.
- `benchmark_validation/summary.json` now persists:
  - benchmark status
  - runtime-complexity summary
  - runtime-complexity violations
- Refactored runtime-complexity analysis into the benchmark-validation layer so release validation consumes the same artifact rather than recomputing its own separate check.
- Updated dashboard benchmark summaries to surface:
  - benchmark status
  - runtime provider/transport counts
  - runtime-complexity violations
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `25 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1243 passed, 17 skipped`
- Commit:
  - `benchmark: persist runtime complexity status`

## Benchmark Validation Gating

- `benchmark-validate` is now a real gate instead of a bundle-only report writer.
- The CLI command now:
  - completes successfully only when the combined benchmark status is `passed`
  - marks telemetry as failed when benchmark/runtime-complexity validation fails
  - raises a runtime error on failed benchmark validation
- Fixed telemetry metadata so benchmark runs now carry the full status/runtime-complexity payload used by the dashboard and release validation.
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `25 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1243 passed, 17 skipped`
- Commit:
  - `benchmark: gate validation on runtime budget`

## Mode-Aware Runtime Complexity Budgets

- Promoted runtime-complexity validation from a single global budget to explicit per-mode budgets for:
  - `rapid`
  - `structured`
  - `verified`
- Benchmark validation now persists mode-aware runtime summaries under `runtime_complexity.by_mode`, including:
  - provider count
  - provider/model count
  - transport count
  - active override count
  - budget violations per mode
- Added monotonic simplification checks so benchmark validation now flags cases where:
  - `rapid` is more complex than `structured`
  - `structured` is more complex than `verified`
- Fixed release-style validation to ignore operator-local `.env` overrides and evaluate the repo baseline instead.
- Updated the dashboard to surface per-mode runtime complexity summaries directly from benchmark metadata.
- Added regressions for:
  - mode-aware runtime-complexity summaries
  - benchmark bundle persistence of `by_mode`
  - dashboard extraction of mode-aware runtime complexity
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `26 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1244 passed, 17 skipped`
- Commit:
  - `benchmark: add mode-aware runtime budgets`

## Rapid Mode Direct Path

- Turned `run --mode rapid` into a real direct baseline path instead of a lightly trimmed orchestration flow.
- `rapid` runs now:
  - skip architect setup and decomposition entirely
  - skip orchestration/refinement loops
  - run a single Hunter pass directly against the goal statement
  - emit a minimal one-node CDG artifact that records whether direct matching succeeded
- Added a stable rapid-mode artifact contract:
  - successful direct matches produce an atomic one-node CDG with `matched_primitive`
  - failed direct matches produce a blocked one-node CDG with explicit failure notes
  - run telemetry marks `rapid_direct_path=true`
  - rapid runs no longer record unused architect routing/stages
- Added regressions for:
  - direct-match CDG success/failure semantics
  - rapid direct-match result wrapping
  - CLI telemetry/output behavior for `run --mode rapid`
- Validation:
  - `pytest -q tests/test_rapid_mode.py tests/test_cli_command_telemetry.py tests/test_execution_modes.py`
  - Result: `10 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1248 passed, 17 skipped`
- Commit:
  - `cli: add rapid direct run path`

## Structured Mode Single-Pass Flow

- Turned `run --mode structured` into a real single-pass flow instead of reusing the full verified refine loop.
- `structured` runs now:
  - keep architect decomposition
  - keep one Hunter pass over decomposed leaves
  - skip orchestration refinement rounds entirely
  - emit normal CDG and match artifacts after that single pass
- Added explicit execution-path telemetry:
  - `rapid_direct`
  - `structured_single_pass`
  - `verified_orchestration`
- Structured-mode telemetry now records:
  - architect decomposition stage
  - `structured_match` stage
  - no `orchestration` stage
- Added regressions for:
  - structured single-pass helper semantics
  - CLI telemetry/output behavior for `run --mode structured`
  - interaction with the existing rapid-mode helper tests
- Validation:
  - `pytest -q tests/test_structured_mode.py tests/test_rapid_mode.py tests/test_cli_command_telemetry.py tests/test_execution_modes.py`
  - Result: `12 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1250 passed, 17 skipped`
- Commit:
  - `cli: add structured single-pass run path`
