#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/llama/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
HOST="${LLAMA_HOST:-127.0.0.1}"
PORT="${LLAMA_PORT:-18080}"
ALIAS="${LLAMA_ALIAS:-llama-3.1-8b-instruct}"
API_KEY="${LLAMA_API_KEY:-local}"
N_CTX="${LLAMA_N_CTX:-8192}"
MODE="${LLAMA_SERVER_MODE:-auto}" # auto|native|python
NATIVE_BIN="${LLAMA_SERVER_BIN:-}"
DEVICE="${LLAMA_DEVICE:-none}" # default CPU-only for reliability
GPU_LAYERS="${LLAMA_GPU_LAYERS:-0}"
FIT="${LLAMA_FIT:-off}"
VERBOSITY="${LLAMA_VERBOSITY:-0}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi

resolve_native_bin() {
  if [[ -n "$NATIVE_BIN" ]]; then
    if [[ -x "$NATIVE_BIN" ]]; then
      printf '%s\n' "$NATIVE_BIN"
      return 0
    fi
    echo "LLAMA_SERVER_BIN is set but not executable: $NATIVE_BIN" >&2
    return 1
  fi

  if command -v llama-server >/dev/null 2>&1; then
    command -v llama-server
    return 0
  fi

  if [[ -x "/tmp/llama.cpp/build/bin/llama-server" ]]; then
    printf '%s\n' "/tmp/llama.cpp/build/bin/llama-server"
    return 0
  fi

  return 1
}

if [[ "$MODE" == "native" || "$MODE" == "auto" ]]; then
  if NATIVE_PATH="$(resolve_native_bin)"; then
    echo "Starting native llama-server: $NATIVE_PATH"
    exec "$NATIVE_PATH" \
      --model "$MODEL_PATH" \
      --alias "$ALIAS" \
      --host "$HOST" \
      --port "$PORT" \
      --ctx-size "$N_CTX" \
      --api-key "$API_KEY" \
      --device "$DEVICE" \
      --n-gpu-layers "$GPU_LAYERS" \
      --fit "$FIT" \
      --verbosity "$VERBOSITY"
  elif [[ "$MODE" == "native" ]]; then
    echo "Requested native mode, but no llama-server binary was found." >&2
    echo "Set LLAMA_SERVER_BIN or install/build llama-server." >&2
    exit 1
  fi
fi

echo "Starting python llama_cpp.server fallback"
exec python -m llama_cpp.server \
  --model "$MODEL_PATH" \
  --model_alias "$ALIAS" \
  --host "$HOST" \
  --port "$PORT" \
  --n_ctx "$N_CTX" \
  --n_gpu_layers "$GPU_LAYERS" \
  --api_key "$API_KEY"
