#!/usr/bin/env bash
# e2e_benchmark_all.sh — Run e2e_benchmark.sh for every goal config in e2e_goals/
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${BOLD}[all]${NC} $*"; }
ok()    { echo -e "${GREEN}[pass]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

GOALS_DIR="$SCRIPT_DIR/e2e_goals"
if [ ! -d "$GOALS_DIR" ] || [ -z "$(ls "$GOALS_DIR"/*.yml 2>/dev/null)" ]; then
    echo "No YAML goal configs found in $GOALS_DIR/"
    exit 1
fi

TOTAL=0
PASSED=0
FAILED_GOALS=()

for config in "$GOALS_DIR"/*.yml; do
    goal_name="$(basename "$config" .yml)"
    info "=== Running benchmark: $goal_name ==="
    TOTAL=$((TOTAL + 1))

    if bash "$SCRIPT_DIR/e2e_benchmark.sh" "$config"; then
        ok "$goal_name completed"
        PASSED=$((PASSED + 1))
    else
        fail "$goal_name failed"
        FAILED_GOALS+=("$goal_name")
    fi
    echo ""
done

echo ""
info "=== Summary: $PASSED/$TOTAL goals passed ==="
if [ ${#FAILED_GOALS[@]} -gt 0 ]; then
    fail "Failed goals: ${FAILED_GOALS[*]}"
fi
