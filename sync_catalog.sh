#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -x "${AGEOM_PYTHON:-}" ]]; then
  PYTHON_BIN="$AGEOM_PYTHON"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo "error: could not find a usable Python interpreter" >&2
  exit 1
fi

run_ageom() {
  "$PYTHON_BIN" -m ageom.cli "$@"
}

echo "[sync_catalog] using python: $PYTHON_BIN"
echo "[sync_catalog] syncing sources from sources.yml"
run_ageom sources sync

echo "[sync_catalog] rebuilding python declaration index"
run_ageom index build --prover python --output "$SCRIPT_DIR/data/index"

echo "[sync_catalog] rebuilding skill index from built-ins plus sources.yml"
run_ageom skill index --sources-only --output "$SCRIPT_DIR/data/skill_index"

echo "[sync_catalog] done"
