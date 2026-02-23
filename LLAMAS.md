# Local LLM Defaults

AGEO-Matcher uses hybrid per-prompt routing: remote APIs for prompts that
require deep reasoning or formal correctness, local Ollama models for
everything else.

## Default Routing

### Remote API (4 prompts — must be correct)

| Prompt | Why remote |
|--------|-----------|
| `architect_decompose` | Graph design with type reasoning + catalog matching |
| `synthesizer_repair` | Lean4/Coq/Python type-system repair — must be correct |
| `synthesizer_tactic` | Formal proof synthesis — hardest task in the system |
| `ingester_chunk` | 70-line Bayesian state-space prompt, probabilistic expertise |

### Local: `qwen3:14b` (8 prompts — medium complexity)

| Prompt | Task |
|--------|------|
| `architect_critique` | Verification checklist; deterministic phase catches most errors first |
| `hunter_analyze_failure` | Compiler error diagnosis; feeds reformulate |
| `ingester_hoist_state` | Structured state model generation from macro plan |
| `ingester_abstract` | Creative cross-domain abstraction; no correctness constraint |
| `ingester_fix_ghost` | Shape inference repair; short focused task |
| `ingester_opaque_witness` | DL shape propagation; has deterministic fallback |
| `ingester_fix_message_cycle` | BP cycle-breaking; rare, clear rules |
| `orchestrator_refine` | Predicate splitting; simpler than decompose |

### Local: `qwen2.5-coder:7b` (4 prompts — mechanical / high-volume)

| Prompt | Task |
|--------|------|
| `architect_strategy` | Pick one paradigm from a fixed enum |
| `hunter_score` | Rank candidates; GBNF-constrained output |
| `hunter_reformulate` | Generate search queries; GBNF-constrained |
| `ingester_fix_type` | Mechanical mypy fixes from structured errors |

## Available Local Models

| Model | Size | Role |
|-------|------|------|
| `qwen2.5-coder:7b` | ~4.7 GB | Light tier — code transforms, ranking, structured output |
| `qwen3:14b` | ~9.3 GB | Medium tier — diagnostics, structured JSON, creative abstraction |
| `deepseek-r1:32b` | ~20 GB | Optional heavy tier — available if you want to run more prompts locally |

```
llms/
├── qwen2.5-coder-7b/{install,run}.sh
├── qwen3-14b/{install,run}.sh
├── deepseek-r1-32b/{install,run}.sh
├── install_defaults.sh
└── run_defaults.sh
```

## Quickstart

```bash
# 1. Install Ollama + models
bash llms/install_defaults.sh

# 2. Start serving
bash llms/run_defaults.sh

# 3. Point ageom at Ollama's OpenAI-compatible endpoint
export AGEOM_LLAMA_CPP_BASE_URL=http://127.0.0.1:11434/v1
```

## Overriding Defaults

Any prompt can be rerouted via environment variables:

```bash
# Run architect_decompose locally instead of remote
export AGEOM_ARCHITECT_DECOMPOSE_LLM_PROVIDER=llama_cpp
export AGEOM_ARCHITECT_DECOMPOSE_LLM_MODEL=deepseek-r1:32b

# Push a local prompt to remote
export AGEOM_HUNTER_SCORE_LLM_PROVIDER=anthropic
export AGEOM_HUNTER_SCORE_LLM_MODEL=claude-sonnet-4-5-20250929
```
