#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-18080}"
ALIAS="${LLAMA_ALIAS:-llama-3.1-8b-instruct}"
API_KEY="${LLAMA_API_KEY:-local}"
N_CTX="${LLAMA_N_CTX:-8192}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi

exec python -m llama_cpp.server \
  --model "$MODEL_PATH" \
  --model_alias "$ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --n_ctx "$N_CTX" \
  --api_key "$API_KEY"
