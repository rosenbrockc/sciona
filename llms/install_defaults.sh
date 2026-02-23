#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Check / install Ollama ---------------------------------------------------
if ! command -v ollama &>/dev/null; then
    echo "Ollama is not installed."
    if command -v brew &>/dev/null; then
        read -rp "Install Ollama via Homebrew? [Y/n] " ans
        if [[ "${ans:-Y}" =~ ^[Yy]$ ]]; then
            brew install ollama
        else
            echo "Please install Ollama manually: https://ollama.com/download"
            exit 1
        fi
    else
        echo "Please install Ollama: https://ollama.com/download"
        exit 1
    fi
fi

echo "Ollama $(ollama --version 2>/dev/null) detected."

# --- Fix Homebrew MLX linkage -------------------------------------------------
# Homebrew's ollama formula depends on mlx-c but doesn't link the dylib into
# its own bin/ directory, which is the only place the binary searches at runtime.
OLLAMA_BIN="$(dirname "$(command -v ollama)")"
OLLAMA_CELLAR_BIN="$(readlink -f "${OLLAMA_BIN}" 2>/dev/null || echo "${OLLAMA_BIN}")"

if [[ "${OLLAMA_CELLAR_BIN}" == */Cellar/ollama/* ]]; then
    for lib in libmlxc.dylib libmlx.dylib; do
        if [[ ! -e "${OLLAMA_CELLAR_BIN}/${lib}" ]] && [[ -e "/opt/homebrew/lib/${lib}" ]]; then
            echo "Symlinking ${lib} into Ollama bin (Homebrew MLX fix)…"
            ln -sf "/opt/homebrew/lib/${lib}" "${OLLAMA_CELLAR_BIN}/${lib}"
        fi
    done
fi
echo

# --- Pull default models ------------------------------------------------------
for model_dir in deepseek-r1-32b qwen3-14b qwen2.5-coder-7b; do
    echo "=== Installing ${model_dir} ==="
    bash "${SCRIPT_DIR}/${model_dir}/install.sh"
    echo
done

echo "All default models installed."
