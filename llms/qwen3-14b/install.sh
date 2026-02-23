#!/usr/bin/env bash
set -euo pipefail
echo "Pulling qwen3:14b (~9.3 GB) …"
ollama pull qwen3:14b
echo "Done."
