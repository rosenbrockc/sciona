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

MAX_VERIFIER_WARNINGS="${SYNC_CATALOG_MAX_VERIFIER_WARNINGS:-0}"
VERIFY_IMPORT_SMOKE="${SYNC_CATALOG_VERIFY_IMPORT_SMOKE:-false}"

run_ageom() {
  "$PYTHON_BIN" -m ageom.cli "$@"
}

count_warning_lines() {
  local log_path="$1"
  "$PYTHON_BIN" - "$log_path" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8", errors="replace")
pattern = re.compile(r"(^|\b)(WARNING|WARNINGS)(:|\b)", re.IGNORECASE | re.MULTILINE)
print(len(pattern.findall(text)))
PY
}

run_verifier() {
  local label="$1"
  shift
  local log_path
  log_path="$(mktemp)"
  echo "[sync_catalog] verifying: $label"
  set +e
  "$@" 2>&1 | tee "$log_path"
  local rc=$?
  set -e
  local warning_count
  warning_count="$(count_warning_lines "$log_path")"
  rm -f "$log_path"

  if [[ "$rc" -ne 0 ]]; then
    echo "[sync_catalog] verifier failed: $label (exit $rc)" >&2
    exit "$rc"
  fi
  if [[ "$warning_count" -gt "$MAX_VERIFIER_WARNINGS" ]]; then
    echo "[sync_catalog] verifier produced $warning_count warning(s), exceeding limit $MAX_VERIFIER_WARNINGS: $label" >&2
    exit 1
  fi
}

list_sources() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from ageom.config import AgeomConfig
from ageom.sources import load_sources, resolve_source

config = AgeomConfig()
sources = load_sources(config.sources_file)
for source in sources.sources:
    resolved = resolve_source(source, base_dir=Path.cwd())
    print(f"{source.name}\t{source.package}\t{resolved}")
PY
}

echo "[sync_catalog] using python: $PYTHON_BIN"
echo "[sync_catalog] syncing sources from sources.yml"
run_ageom sources sync

while IFS=$'\t' read -r source_name source_package source_root; do
  [[ -n "${source_name:-}" ]] || continue
  verifier_args=("$PYTHON_BIN" "$SCRIPT_DIR/scripts/verify_atoms_repo.py" "$source_root" "--package" "$source_package")
  if [[ "$(printf '%s' "$VERIFY_IMPORT_SMOKE" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    verifier_args+=("--import-smoke" "--python" "$PYTHON_BIN")
  fi
  run_verifier "static completeness check ($source_name)" "${verifier_args[@]}"

  if [[ -f "$source_root/scripts/audit.py" ]]; then
    run_verifier "source audit ($source_name)" "$PYTHON_BIN" "$source_root/scripts/audit.py"
  fi
done < <(list_sources)

echo "[sync_catalog] rebuilding python declaration index"
run_ageom index build --prover python --output "$SCRIPT_DIR/data/index"

echo "[sync_catalog] rebuilding skill index from built-ins plus sources.yml"
run_ageom skill index --sources-only --output "$SCRIPT_DIR/data/skill_index"

echo "[sync_catalog] done"
