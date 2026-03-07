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
