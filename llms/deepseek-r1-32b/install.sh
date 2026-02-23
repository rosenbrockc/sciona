#!/usr/bin/env bash
set -euo pipefail
echo "Pulling deepseek-r1:32b (~20 GB) …"
ollama pull deepseek-r1:32b
echo "Done."
