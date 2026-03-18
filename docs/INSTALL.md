# Installation and Local LLM Setup

This document explains:
- how to install AGEO-Matcher
- how to run supported local LLMs
- how to configure model/provider selection by architecture role

## 1. Install AGEO-Matcher

```bash
# From repo root
pip install -e ".[hunter,architect]"
```

If you also need all optional features:

```bash
pip install -e ".[all]"
```

## 1b. PostgreSQL (optional)

PostgreSQL is used for Architect checkpoint persistence, shared context, and pipeline telemetry. The repo includes a `docker-compose.yml`:

```bash
docker compose up -d postgres
```

This starts PostgreSQL on port 5433 with database `ageom_architect`, user `ageom`, password `ageom_dev`. Add to your `.env`:

```bash
AGEOM_POSTGRES_URI=postgresql://ageom:ageom_dev@localhost:5433/ageom_architect
```

Tables are created automatically on first use. Telemetry backend selection:

| `AGEOM_TELEMETRY_BACKEND` | Behavior |
|---------------------------|----------|
| `auto` (default) | Use Postgres when `AGEOM_POSTGRES_URI` is set, otherwise file-only |
| `postgres` | Require Postgres (falls back silently on connection failure) |
| `file` | File-based persistence only, even when Postgres URI is configured |

You can omit PostgreSQL entirely -- all features fall back to in-memory or file-based storage.

## 2. Supported Local LLM Backend

Local LLM support is implemented through an OpenAI-compatible HTTP endpoint.

Current supported local backend:
- `llama.cpp` server (`llama-server`) with GGUF models

Model families you can run locally (via GGUF):
- Llama 3.1 Instruct (recommended for Hunter)
- Mistral Instruct
- Gemma Instruct

Use instruction-tuned/chat models, not base models.

## 3. Install llama.cpp

### Option A: Build from source

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build -j
```

After build, the server binary is typically:
- `./build/bin/llama-server`

### Option B: Package manager

Use your platform package manager if it provides a recent `llama.cpp` build with `llama-server`.

## 4. Download a Local Model (GGUF)

Download an instruct GGUF model (example targets):
- `Llama-3.1-8B-Instruct` (recommended Hunter default)
- quantization example: `Q4_K_M` (good speed/quality tradeoff)

Place the model somewhere stable, e.g.:
- `/opt/models/llama-3.1-8b-instruct-q4_k_m.gguf`

## 5. Run llama.cpp Server

Example:

```bash
./build/bin/llama-server \
  -m /opt/models/llama-3.1-8b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --alias llama-3.1-8b-instruct
```

AGEO-Matcher expects an OpenAI-compatible base URL:
- `http://127.0.0.1:8080/v1`

## 6. Configure Providers and Model Types

Configuration is read from `.env` with `AGEOM_` prefix.

### 6.1 Recommended local-first config (Hunter default)

```bash
AGEOM_LLAMA_CPP_BASE_URL=http://127.0.0.1:8080/v1
AGEOM_LLAMA_CPP_API_KEY=local

# Global defaults (used when role-specific override is not set)
AGEOM_LLM_PROVIDER=anthropic
AGEOM_LLM_MODEL=claude-sonnet-4-5-20250929

# Hunter (Round 2): local quantized worker
AGEOM_HUNTER_LLM_PROVIDER=llama_cpp
AGEOM_HUNTER_LLM_MODEL=llama-3.1-8b-instruct
AGEOM_HUNTER_LLM_MAX_TOKENS=1024
AGEOM_HUNTER_MODE=speculative_local
AGEOM_HUNTER_USE_GBNF=true
AGEOM_HUNTER_QUERY_BATCH_SIZE=40
AGEOM_HUNTER_TOP_K_PER_QUERY=50
AGEOM_HUNTER_MAX_CANDIDATES_TOTAL=3000
AGEOM_HUNTER_TOP_K_VERIFY=10
```

### 6.2 Role-specific model selection

You can configure providers/models at three levels:

1. Global:
- `AGEOM_LLM_PROVIDER`
- `AGEOM_LLM_MODEL`
- `AGEOM_LLM_MAX_TOKENS`

2. Role-specific overrides:
- Architect (Round 1):
  - `AGEOM_ARCHITECT_LLM_PROVIDER`
  - `AGEOM_ARCHITECT_LLM_MODEL`
- Hunter (Round 2):
  - `AGEOM_HUNTER_LLM_PROVIDER`
  - `AGEOM_HUNTER_LLM_MODEL`
  - `AGEOM_HUNTER_LLM_MAX_TOKENS`
  - plus Hunter mode controls above
- Synthesizer (Round 3):
  - `AGEOM_SYNTHESIZER_LLM_PROVIDER`
  - `AGEOM_SYNTHESIZER_LLM_MODEL`

3. Per-command CLI overrides:
- `--llm-provider`
- `--llm-model`
- `--llm-max-tokens`

## 7. Provider Values

Valid provider values:
- `anthropic`
- `codex`
- `llama_cpp` (local)

## 8. Quick Smoke Tests

### Match (Hunter) with local model

```bash
ageom match \
  --statement "forall n m : Nat, n + m = m + n" \
  --prover lean4 \
  --llm-provider llama_cpp \
  --llm-model llama-3.1-8b-instruct
```

### Decompose with local override

```bash
ageom decompose \
  "Implement merge sort" \
  --llm-provider llama_cpp \
  --llm-model llama-3.1-8b-instruct \
  --no-persist
```

## 9. Notes on GBNF

Hunter supports grammar-constrained decoding for key structured outputs
(ranking indices and query arrays). This is enabled with:

```bash
AGEOM_HUNTER_USE_GBNF=true
```

Use a recent llama.cpp build that supports grammar-constrained generation
through the OpenAI-compatible API path.
