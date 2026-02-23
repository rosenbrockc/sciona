#!/usr/bin/env bash
set -euo pipefail

# --- Check Ollama is installed -------------------------------------------------
if ! command -v ollama &>/dev/null; then
    echo "Error: Ollama is not installed. Run llms/install_defaults.sh first."
    exit 1
fi

# --- Start Ollama server if not running ----------------------------------------
if ! curl -sf http://127.0.0.1:11434/api/tags &>/dev/null; then
    echo "Starting Ollama server …"
    ollama serve &
    OLLAMA_PID=$!
    # Wait for server to be ready
    for i in $(seq 1 30); do
        if curl -sf http://127.0.0.1:11434/api/tags &>/dev/null; then
            break
        fi
        sleep 1
    done
    echo "Ollama server started (PID ${OLLAMA_PID})."
else
    echo "Ollama server already running."
fi

# --- Check each default model --------------------------------------------------
MODELS=("deepseek-r1:32b" "qwen3:14b" "qwen2.5-coder:7b")
AVAILABLE=$(ollama list 2>/dev/null || true)

for model in "${MODELS[@]}"; do
    if echo "${AVAILABLE}" | grep -q "${model}"; then
        echo "✓ ${model} available"
    else
        echo "✗ ${model} not found locally"
        read -rp "  Pull ${model} now? [Y/n] " ans
        if [[ "${ans:-Y}" =~ ^[Yy]$ ]]; then
            ollama pull "${model}"
        else
            echo "  Skipping ${model} — prompts using it will fail until installed."
        fi
    fi
done

echo
echo "Ready. Ollama is serving on http://127.0.0.1:11434"
echo "The llama_cpp provider in ageom connects via AGEOM_LLAMA_CPP_BASE_URL (default http://127.0.0.1:8080/v1)."
echo "If using Ollama's OpenAI-compatible endpoint, set:"
echo "  export AGEOM_LLAMA_CPP_BASE_URL=http://127.0.0.1:11434/v1"
