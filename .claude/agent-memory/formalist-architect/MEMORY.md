# Formalist Architect Memory - sciona

## Project Overview
- **sciona (sciona)**: Config-driven agentic pipeline for grounding high-level algorithm goals into verified library functions (Lean 4/Mathlib, Coq/Rocq, Python)
- **ageo-atoms (ageoa)**: Companion package with icontract-decorated atom wrappers and ghost witness simulation system
- **Test suite**: ~1200+ tests, well-maintained

## Architecture: Four-Round Pipeline + Principal
- Round 0 (Ingester): Source code -> RawDataFlowGraph -> MacroAtomSpecs -> IngestionBundle
- Round 1 (Architect): Goal -> CDG via LangGraph with checkpointing/time-travel
- Round 2 (Hunter): CDG atomic leaves -> verified library matches via pydantic-graph state machine
- Round 3 (Synthesizer): CDG + matches -> skeleton files with optional ghost simulation
- Principal: NAS-style Optuna loop over the full pipeline with credit assignment

## Core Design Pattern
**Deterministic -> Agentic -> Deterministic sandwich**: LLMs confined to agentic layers; indexer, oracle, ghost sim, assembler, handoff validation, evaluator are pure functions.

## Execution Modes (implemented)
- `rapid`: minimal decomposition, lexical index, no shared context, no GBNF
- `structured`: decomposition + catalog, standard hunter, no shared context
- `verified`: full pipeline with graph retrieval, shared context, GBNF

## Benchmark System
- Prompt benchmarks: per-prompt-key validation with direct-baseline comparison
- Flow benchmarks: full-path validation across all modes (direct_baseline, rapid, structured, verified)
- Benchmark validation bundle: deterministic fixture-based release checks
- Execution path distinctness validation enforced
- See [benchmark-analysis.md](benchmark-analysis.md) for detailed hardening analysis (2026-03-08)
- **Critical gap**: flow benchmarks use fully deterministic mocks; stability=1.0 is trivially true
- **Critical gap**: "direct_baseline" is single-shot Hunter, not true LLM-from-scratch baseline
- **Critical gap**: only 3 flow cases, 2 leaves each; binary pass/fail only; no semantic metrics

## Key Invariants
- Handoff validation: every atomic CDG leaf must have type_signature + description
- Verification oracle: compiler is ground truth (no approximation)
- Mode monotonicity: complexity budget rapid <= structured <= verified
- Ghost simulation: optional but catches structural mismatches pre-compilation

## Catalog System
- PrimitiveCatalog: ~200-500 primitives from CLRS-30, Coq 100 Theorems, Mathlib, source repos
- CatalogConfidence: heuristic gating retrieval by task-text overlap
- De-duplication plan (PLAN-catalog-enrichment.md): embedding + structural dedup, not yet fully implemented
- SkillIndex: FAISS over primitive descriptions for semantic matching

## Key Files
- `/Users/conrad/personal/sciona/ARCH.md` - full architecture doc
- `/Users/conrad/personal/sciona/ROADMAP.md` - strategic direction
- `/Users/conrad/personal/sciona/ROADMAP_DONE.md` - completed items
- `/Users/conrad/personal/sciona/PLAN-catalog-enrichment.md` - next catalog work
- `/Users/conrad/personal/sciona/sciona/config.py` - AgeomConfig + ExecutionModeSettings
- `/Users/conrad/personal/sciona/sciona/benchmark_validation.py` - release validation
- `/Users/conrad/personal/sciona/sciona/flow_benchmark.py` - flow benchmark harness

## CDG Expansion
- [project_cdg_expansion.md](project_cdg_expansion.md) -- Formal analysis of topology expansion mechanism (DPO-based, beyond node swaps)

## Acyclicity Analysis
- See [project_acyclicity_analysis.md](project_acyclicity_analysis.md) for formal analysis of 7 DAG invariants, the MESSAGE_PASSING exception gap, and FixedPoint combinator recommendation

## Algorithmic Commons Foundation
- See [project_algorithmic_commons.md](project_algorithmic_commons.md) for formal analysis of the 6-component marketplace architecture
- Key risk: AST fingerprint 64-bit truncation insufficient at registry scale; Shapley tractable via DAG closed-form; gradient scores leak in Dead-End Flare
