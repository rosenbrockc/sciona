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

echo "Ollama $(ollama --version) detected."
echo

# --- Pull default models ------------------------------------------------------
for model_dir in deepseek-r1-32b qwen3-14b qwen2.5-coder-7b; do
    echo "=== Installing ${model_dir} ==="
    bash "${SCRIPT_DIR}/${model_dir}/install.sh"
    echo
done

echo "All default models installed."
