#!/usr/bin/env bash
set -euo pipefail
echo "Pulling qwen2.5-coder:7b (~4.7 GB) …"
ollama pull qwen2.5-coder:7b
echo "Done."
