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

## Flow Benchmark Mode Fidelity

- Refactored the small full-flow benchmark harness so the mode variants now exercise the real product paths instead of a shared placeholder implementation.
- `flow_benchmark` variants now map to:
  - `direct_baseline`: synthetic non-product baseline
  - `rapid`: real rapid direct-match helper path
  - `structured`: real structured single-pass helper path
  - `verified`: real orchestration path
- Fixed benchmark scoring to count unique expected leaf coverage instead of raw successful-match totals, which avoided overcounting duplicate atomic leaves from skeleton expansion.
- Adjusted benchmark validation policy so full-flow correctness gating now requires:
  - `structured`
  - `verified`
  while still reporting `rapid` as a comparison track rather than a release blocker.
- Updated regressions for:
  - flow benchmark variant expectations
  - release/benchmark validation status under the new mode semantics
- Validation:
  - `pytest -q tests/test_flow_benchmark.py tests/test_e2e_flow_benchmark.py tests/test_benchmark_validation.py tests/test_release_validation.py`
  - Result: `11 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1250 passed, 17 skipped`
- Commit:
  - `benchmark: align flow variants with execution modes`

## Benchmark Gate Auditability

- Made the full-flow benchmark gate explicit about which variants are required for release-style validation and which are comparison-only.
- Benchmark-validation artifacts now persist:
  - `flow_required_variants`
  - `flow_comparison_variants`
  - `flow_comparison_failures`
  - `flow_comparison_unstable_groups`
- Release-validation manifests now carry the same required-vs-comparison benchmark metadata.
- Dashboard/API benchmark summaries now expose those fields directly so it is obvious that:
  - `structured` and `verified` are the gated full-flow modes
  - `rapid` and `direct_baseline` are comparison tracks
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `26 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1250 passed, 17 skipped`
- Commit:
  - `benchmark: expose required and comparison flow tracks`

## Mode Distinctness Validation

- Added explicit execution-path metadata to full-flow benchmark results and aggregates.
- Flow benchmark variants now persist the concrete execution path they exercised:
  - `direct_baseline`
  - `rapid_direct`
  - `structured_single_pass`
  - `verified_orchestration`
- Benchmark validation now includes a `flow_execution_paths` summary with:
  - expected path mapping by variant
  - observed path mapping by variant
  - distinctness violations
- Release/benchmark validation now fails if the mode variants collapse onto the wrong execution paths, so routing and benchmark semantics cannot silently drift apart.
- Dashboard/API benchmark summaries now expose those observed execution paths directly.
- Validation:
  - `pytest -q tests/test_flow_benchmark.py tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `29 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1250 passed, 17 skipped`
- Commit:
  - `benchmark: validate mode execution distinctness`

## Live Execution Path Visibility

- Added first-class live run execution summaries to the dashboard/API so mode behavior is visible without opening raw run metadata.
- Dashboard/API run summaries now expose:
  - `execution_mode`
  - `execution_path`
  - `rapid_direct_path`
- Run list cards now show `mode=... path=...`, and the detailed run view shows a dedicated `mode/path` metric.
- Added visualizer regressions for extracting those execution-path fields from persisted algorithm-creation runs.
- Validation:
  - `pytest -q tests/test_visualizer_api.py`
  - Result: `16 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1250 passed, 17 skipped`
- Commit:
  - `dashboard: surface live execution paths`

## Flow Summary Execution Paths

- Updated the human-readable flow benchmark summary to include the execution path for each variant.
- `format_flow_benchmark_summary(...)` now emits a `paths` column so text summaries in benchmark/release artifacts directly show:
  - `direct_baseline`
  - `rapid_direct`
  - `structured_single_pass`
  - `verified_orchestration`
- Added regressions to ensure the summary string continues to carry the mode-specific execution-path markers.
- Validation:
  - `pytest -q tests/test_flow_benchmark.py tests/test_e2e_flow_benchmark.py tests/test_benchmark_validation.py`
  - Result: `8 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1250 passed, 17 skipped`
- Commit:
  - `benchmark: include execution paths in flow summary`

## Benchmark Summary Ergonomics

- Added compact benchmark summary fields so release bundles and dashboard summaries expose the mode policy without requiring nested JSON inspection.
- Benchmark/release metadata now includes:
  - `flow_gate_summary`
  - `flow_execution_path_summary`
- These summarize, in one line each:
  - which flow variants are release-gated vs comparison-only
  - which execution path each flow variant actually exercised
- Dashboard benchmark summaries now surface those compact strings directly.
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `26 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1250 passed, 17 skipped`
- Commit:
  - `benchmark: add compact flow gate summaries`

## Catalog Validation Release Gate

- Added a release-style catalog validation artifact that audits configured source-derived catalog coverage and writes `catalog_validation.json`.
- Release validation now runs that catalog check alongside benchmark validation and fails if configured sources are missing or produce zero source candidates.
- Release telemetry and dashboard summaries now expose:
  - catalog validation status
  - configured vs resolved source counts
  - missing sources
  - zero-candidate sources
  - catalog validation report path
- Added regressions for:
  - successful catalog validation report generation
  - missing/zero-candidate source detection
  - release validation failure when catalog validation fails
  - dashboard/API catalog validation summary extraction
- Validation:
  - `pytest -q tests/test_catalog_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `25 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1253 passed, 17 skipped`
- Commit:
  - `release: validate source catalog coverage`

## Source Registry Alignment Audit

- Added a deterministic source-alignment audit that compares AST-discovered `@register_atom` registrations against live registry registrations on a per-source basis.
- Refactored live registry enumeration into a shared helper so the audit and catalog seeding path use the same source-membership logic.
- Catalog validation now persists alignment details alongside coverage details, including:
  - matched registration count
  - registry-only count
  - AST-only count
  - drift sources
  - registry-error sources
- Dashboard/API catalog validation summaries now surface those alignment drift counts directly.
- Added regressions for:
  - live-registry vs AST drift detection
  - catalog validation persistence of alignment details
  - dashboard/API extraction of alignment summary fields
- Validation:
  - `pytest -q tests/test_source_catalog.py tests/test_catalog_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `35 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1254 passed, 17 skipped`
- Commit:
  - `catalog: audit source registry alignment`

## Rapid And Structured Round-Default Simplification

- Simplified `rapid` and `structured` mode routing so round-level provider/model defaults are now suppressed when they merely restate code defaults.
- In those lighter modes, the effective round default now falls back to the top-level `llm_provider` / `llm_model` unless the operator explicitly changed the round override.
- This reduces default runtime complexity in light modes from mixed round defaults to a single provider/model/transport by default.
- Runtime-complexity benchmarking now reflects that stricter behavior:
  - `rapid` provider count: `1`
  - `structured` provider count: `1`
- Added regressions for:
  - effective round-provider fallback in rapid mode
  - prompt-routing summaries using the simplified default provider/model in rapid and structured modes
  - benchmark runtime-complexity summaries reflecting the collapsed light-mode provider surface
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_llm_router.py tests/test_execution_modes.py tests/test_release_validation.py tests/test_validation_telemetry.py`
  - Result: `77 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1255 passed, 17 skipped`
- Commit:
  - `runtime: simplify light-mode round defaults`

## Light-Mode Runtime Budget Tightening

- Tightened benchmark/release runtime-complexity budgets for `rapid` and `structured` so those modes are now validated as:
  - one provider
  - one provider/model pair
  - one transport
- This converts the earlier simplification work into an explicit release policy instead of leaving the lighter modes on a looser historical budget.
- Added regressions to assert the stricter rapid/structured budgets in the runtime-complexity summary.
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `27 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1255 passed, 17 skipped`
- Commit:
  - `release: tighten light-mode runtime budgets`

## Verified Override Policy Gate

- Added an explicit verified-mode override policy at the benchmark/runtime-complexity layer.
- The verified runtime summary now records:
  - required prompt-key/provider override pairs
  - missing required overrides
  - unexpected active overrides
- Benchmark/release runtime validation now fails if verified-mode routing drifts away from that explicit allowlist, instead of only checking coarse provider/transport counts.
- Dashboard benchmark details now show a compact runtime-policy summary alongside the existing runtime budget metrics.
- Added regressions for:
  - verified mode passing with the current intended override policy
  - verified mode failing when a required override is replaced with an unapproved provider
  - dashboard/API preservation of the runtime override-policy payload
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `28 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1256 passed, 17 skipped`
- Commit:
  - `release: gate verified override policy`

## Runtime Policy Summary Ergonomics

- Added a compact `runtime_override_policy_summary` field to benchmark/release artifacts so runtime override-policy health is visible without opening nested `runtime_complexity` JSON.
- The summary renders per-mode override-policy counts as:
  - required override count
  - missing required override count
  - unexpected active override count
- Propagated that field through:
  - benchmark validation bundle
  - release validation manifest
  - telemetry metadata normalization
  - dashboard/API benchmark summaries
- Added regressions for:
  - benchmark bundle persistence of the runtime policy summary
  - release manifest preservation of the summary
  - telemetry/dashboard extraction of the summary field
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `28 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1256 passed, 17 skipped`
- Commit:
  - `benchmark: summarize runtime override policy`

## Catalog Validation Summary Ergonomics

- Added compact catalog-validation summary fields so release artifacts and dashboards expose source-catalog health without drilling into nested validation payloads.
- The catalog-validation artifact now includes:
  - `coverage_summary`
  - `alignment_summary`
- These summarize, in one line each:
  - resolved/configured sources, added/candidate counts, missing count, zero-candidate count
  - matched registrations, registry-only drift, AST-only drift, drift-source count
- Propagated those fields through:
  - catalog validation reports
  - release-validation manifests
  - telemetry/dashboard API summaries
  - dashboard detail metrics
- Added regressions for:
  - catalog-validation summary generation
  - release manifest preservation of the summary fields
  - dashboard/API extraction of coverage/alignment summaries
- Validation:
  - `pytest -q tests/test_catalog_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `25 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1256 passed, 17 skipped`
- Commit:
  - `catalog: summarize validation health`

## Flow Prompt-Volume Monotonicity Gate

- Added a benchmark validation check that enforces prompt-volume monotonicity across the full-flow execution modes.
- The flow benchmark bundle now records:
  - `flow_prompt_volume`
  - `flow_prompt_volume_summary`
- Benchmark/release validation now fails if:
  - `rapid` uses more prompt calls than `structured`
  - `structured` uses more prompt calls than `verified`
- Propagated the new prompt-volume summary through:
  - benchmark validation bundle
  - release validation manifest
  - telemetry/dashboard API summaries
  - dashboard detail metrics
- Added regressions for:
  - prompt-volume monotonicity failure detection
  - benchmark bundle persistence of prompt-volume summaries
  - release/telemetry/dashboard preservation of the new fields
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `29 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1257 passed, 17 skipped`
- Commit:
  - `benchmark: gate flow prompt volume`

## Catalog Drift Example Visibility

- Surfaced the highest-drift source rows from catalog validation into the dashboard/API summaries.
- Catalog validation summaries now expose `top_drift_sources`, including per-source:
  - registry-only count
  - AST-only count
  - registry-only examples
  - AST-only examples
- Dashboard detail metrics now show a compact top-drift line so alignment regressions can be diagnosed without opening the raw catalog-validation report.
- Added regressions for dashboard/API extraction of top drift-source examples.
- Validation:
  - `pytest -q tests/test_visualizer_api.py tests/test_catalog_validation.py`
  - Result: `18 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1257 passed, 17 skipped`
- Commit:
  - `dashboard: surface top catalog drift examples`

## Catalog Drift Severity Classification

- Added deterministic severity classification for source-registry alignment drift:
  - `healthy`
  - `medium` for AST-only drift
  - `high` for live-registry-only drift
  - `critical` for live registry load failures
- Alignment audits now persist:
  - per-source severity
  - aggregate severity counts
  - highest observed severity
- Catalog validation summaries now include severity in the compact alignment line, and dashboard/API summaries expose severity counts and top-drift severity.
- Added regressions for:
  - high-severity live-registry-only drift classification
  - catalog-validation alignment summaries carrying severity
  - dashboard/API extraction of severity fields
- Validation:
  - `pytest -q tests/test_source_catalog.py tests/test_catalog_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `35 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1257 passed, 17 skipped`
- Commit:
  - `catalog: classify drift severity`

## Critical Catalog Drift Release Gate

- Made the new catalog drift severity actionable in release policy.
- Catalog validation now fails with `critical_alignment_drift` when the source-registry alignment audit reports `highest_severity=critical`, even if source paths and candidate counts otherwise look healthy.
- This makes live registry load failures release-blocking while leaving `medium` and `high` severity drift visible but non-blocking for now.
- Added regressions for:
  - catalog validation failing on critical alignment drift alone
  - release validation failing when catalog validation is critical only
- Validation:
  - `pytest -q tests/test_catalog_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `27 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1259 passed, 17 skipped`
- Commit:
  - `release: fail on critical catalog drift`

## Catalog Warning Summaries For High Drift

- Added explicit non-blocking warning summaries for `high` and `medium` catalog drift.
- Catalog validation now persists:
  - `warnings`
  - `high_severity_sources`
  - `medium_severity_sources`
  - `warning_summary`
- This makes risky-but-non-blocking catalog drift stand out in release artifacts and dashboard summaries without turning it into a release failure.
- Added regressions for:
  - high-severity warning emission during catalog validation
  - warning summary propagation through release fixtures
  - dashboard/API preservation of warning fields
- Validation:
  - `pytest -q tests/test_catalog_validation.py tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `27 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1259 passed, 17 skipped`
- Commit:
  - `catalog: surface drift warnings`

## Release Warning Summary

- Added a compact release-level warning summary that combines:
  - runtime-complexity warning count
  - non-blocking catalog warning count
- Release validation now persists:
  - `warning_summary`
  - `runtime_warning_count`
  - `catalog_warning_count`
- Propagated those fields through:
  - release validation manifest
  - release telemetry metadata
  - dashboard/API benchmark-release summary
- Added regressions for:
  - release manifest warning summary generation
  - release telemetry persistence of warning summary fields
  - dashboard/API extraction of release warning counts
- Validation:
  - `pytest -q tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `24 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1259 passed, 17 skipped`
- Commit:
  - `release: summarize operator warnings`

## Release Top Warning Diagnostics

- Extended the release-level warning summary to include the first concrete warning cause from each side:
  - `top_runtime_warning`
  - `top_catalog_warning`
- The compact release warning string now includes those top causes when present, so the operator-facing summary identifies the first runtime or catalog warning directly instead of only reporting counts.
- Propagated those fields through:
  - release validation manifest
  - release telemetry metadata
  - dashboard/API benchmark-release summaries
  - dashboard detail metrics
- Added regressions for:
  - release manifest top warning fields
  - release telemetry persistence of top warning fields
  - dashboard/API extraction of top warning diagnostics
- Validation:
  - `pytest -q tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `24 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1259 passed, 17 skipped`
- Commit:
  - `release: expose top warning causes`

## Release Top Failure Diagnostics

- Added a compact release-level blocking failure summary parallel to the warning summary:
  - `failure_summary`
  - `top_benchmark_failure`
  - `top_runtime_failure`
  - `top_catalog_failure`
- The release summary now surfaces the first concrete blocking cause from:
  - benchmark validation
  - runtime-complexity validation
  - catalog validation
- Propagated those fields through:
  - release validation manifest
  - release telemetry metadata
  - dashboard/API benchmark-release summaries
  - dashboard detail metrics
- Added regressions for:
  - release manifest top failure fields
  - release telemetry persistence of top failure fields
  - dashboard/API extraction of release failure diagnostics
- Validation:
  - `pytest -q tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `24 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1259 passed, 17 skipped`
- Commit:
  - `release: expose top failure causes`

## Release Top Failed Check Summary

- Extended the compact release failure summary so it names the top failed release check directly:
  - `top_failed_check`
- The failure summary line now starts with `check=<...>` when release validation fails, so operators can immediately see whether the first blocking surface was:
  - `benchmark_validation`
  - `runtime_complexity`
  - `catalog_validation`
- Propagated that field through:
  - release validation manifest
  - release telemetry metadata
  - dashboard/API benchmark-release summaries
  - dashboard detail metrics
- Added regressions for:
  - release manifest top failed-check field
  - release telemetry persistence of top failed-check metadata
  - dashboard/API extraction of top failed-check diagnostics
- Validation:
  - `pytest -q tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `24 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1259 passed, 17 skipped`
- Commit:
  - `release: surface top failed check`

## Release Benchmark Subcheck Diagnostics

- Extended the release failure summary so benchmark-driven failures identify the top failed benchmark subcheck directly:
  - `top_benchmark_subcheck`
- The compact failure line now includes `benchmark_check=<...>` when the benchmark side failed, distinguishing:
  - `runtime_budget`
  - `execution_path`
  - `prompt_volume`
  - `prompt_tuning`
  - `flow_mode`
- Propagated that field through:
  - release validation manifest
  - release telemetry metadata
  - dashboard/API benchmark-release summaries
  - dashboard detail metrics
- Added regressions for:
  - prompt-tuning benchmark failure classification
  - runtime-budget benchmark failure classification
  - execution-path benchmark failure classification
  - release telemetry/dashboard extraction of benchmark subcheck diagnostics
- Validation:
  - `pytest -q tests/test_release_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py`
  - Result: `25 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1259 passed, 17 skipped`
- Commit:
  - `release: surface benchmark failure subcheck`

## Benchmark Bundle Failure Diagnostics

- Promoted benchmark failure classification into the benchmark bundle itself:
  - `top_failed_subcheck`
  - `top_failure`
- The benchmark `summary.json` now records the first failing benchmark surface directly instead of requiring release validation to derive it later.
- Release validation now consumes those benchmark-bundle fields when present, so benchmark and release summaries stay consistent.
- Propagated the benchmark failure fields through:
  - benchmark-validation telemetry metadata
  - dashboard/API benchmark summaries
  - dashboard detail metrics
- Added regressions for:
  - benchmark bundle persistence of top failure diagnostics
  - benchmark-validation telemetry persistence of top failure diagnostics
  - dashboard/API extraction of benchmark failure diagnostics
  - execution-path failure prioritization over prompt-count failures
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py tests/test_release_validation.py`
  - Result: `32 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1260 passed, 17 skipped`
- Commit:
  - `benchmark: surface top failed subcheck`

## Benchmark Failure Summary Line

- Added a compact benchmark-level failure summary line parallel to the release failure summary:
  - `failure_summary`
- The benchmark bundle now records one operator-facing line that combines:
  - `subcheck=<...>`
  - `failure=<...>`
- Release validation now backfills that benchmark failure line when older or synthetic benchmark summaries omit it, so release manifests remain stable across refactors and test fixtures.
- Propagated the benchmark failure summary through:
  - benchmark-validation telemetry metadata
  - release manifest `checks.benchmark_validation`
  - dashboard/API benchmark summaries
  - dashboard detail metrics
- Added regressions for:
  - benchmark bundle persistence of `failure_summary`
  - benchmark command telemetry persistence of `failure_summary`
  - release-manifest fallback synthesis of benchmark failure diagnostics
  - dashboard/API extraction of the compact benchmark failure line
- Validation:
  - `pytest -q tests/test_benchmark_validation.py tests/test_validation_telemetry.py tests/test_visualizer_api.py tests/test_release_validation.py tests/test_cli_command_telemetry.py`
  - Result: `37 passed`
  - `conda run -n hpyexec pytest -q`
  - Result: `1261 passed, 17 skipped`
- Commit:
  - `benchmark: add compact failure summary`
