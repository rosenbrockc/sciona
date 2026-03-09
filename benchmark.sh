#!/usr/bin/env bash
# benchmark.sh — Run live prompt benchmarks against codex and gemini shims,
# then run deterministic validation, and verify results.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="output/benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

REPEATS="${BENCHMARK_REPEATS:-3}"

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BOLD}[bench]${NC} $*"; }
ok()    { echo -e "${GREEN}[pass]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ---------------------------------------------------------------------------
# 1. Deterministic validation (mocks) — proves harness is healthy
# ---------------------------------------------------------------------------
info "Running deterministic benchmark validation..."
ageom benchmark-validate --output "$OUTPUT_DIR/validation"

VALIDATION_STATUS=$(python -c "
import json, sys
p = json.load(open('$OUTPUT_DIR/validation/summary.json'))
print(p['status'])
")

if [ "$VALIDATION_STATUS" = "passed" ]; then
    ok "Deterministic validation passed"
else
    fail "Deterministic validation failed — check $OUTPUT_DIR/validation/summary.json"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Live prompt benchmark — codex_shim
# ---------------------------------------------------------------------------
info "Running prompt benchmark: codex_shim:gpt-5.3-codex (repeats=$REPEATS)..."
ageom prompt-benchmark \
    --provider codex_shim:gpt-5.3-codex \
    --repeats "$REPEATS" \
    --compare-direct-baseline \
    --output "$OUTPUT_DIR/prompt_codex.json" \
    | tee "$OUTPUT_DIR/prompt_codex_summary.txt"

# ---------------------------------------------------------------------------
# 3. Live prompt benchmark — gemini_shim
# ---------------------------------------------------------------------------
info "Running prompt benchmark: gemini_shim:gemini-2.5-pro (repeats=$REPEATS)..."
ageom prompt-benchmark \
    --provider gemini_shim:gemini-2.5-pro \
    --repeats "$REPEATS" \
    --compare-direct-baseline \
    --output "$OUTPUT_DIR/prompt_gemini.json" \
    | tee "$OUTPUT_DIR/prompt_gemini_summary.txt"

# ---------------------------------------------------------------------------
# 4. Head-to-head — both providers, all prompt keys
# ---------------------------------------------------------------------------
info "Running head-to-head: codex_shim vs gemini_shim (repeats=$REPEATS)..."
ageom prompt-benchmark \
    --provider codex_shim:gpt-5.3-codex \
    --provider gemini_shim:gemini-2.5-pro \
    --repeats "$REPEATS" \
    --compare-direct-baseline \
    --output "$OUTPUT_DIR/prompt_head_to_head.json" \
    | tee "$OUTPUT_DIR/prompt_head_to_head_summary.txt"

# ---------------------------------------------------------------------------
# 5. Verify results
# ---------------------------------------------------------------------------
info "Verifying results..."

ERRORS=0

verify_report() {
    local label="$1"
    local report="$2"

    if [ ! -f "$report" ]; then
        fail "$label: report not found at $report"
        ERRORS=$((ERRORS + 1))
        return
    fi

    local stats
    stats=$(python -c "
import json, sys
p = json.load(open('$report'))
results = p.get('results', [])
aggs = p.get('aggregates', [])
total = len(results)
passed = sum(1 for r in results if r.get('ok'))
failed = total - passed
tuned_aggs = [a for a in aggs if a.get('variant') == 'tuned']
baseline_aggs = [a for a in aggs if a.get('variant') == 'direct_baseline']
tuned_pass = sum(a.get('passed_cases', 0) for a in tuned_aggs)
tuned_total = sum(a.get('total_cases', 0) for a in tuned_aggs)
baseline_pass = sum(a.get('passed_cases', 0) for a in baseline_aggs)
baseline_total = sum(a.get('total_cases', 0) for a in baseline_aggs)
stability = min((a.get('stability_rate', 0) for a in tuned_aggs), default=0)
print(f'{total} {passed} {failed} {tuned_pass} {tuned_total} {baseline_pass} {baseline_total} {stability:.3f}')
")

    read -r total passed failed tuned_pass tuned_total baseline_pass baseline_total stability <<< "$stats"

    echo -e "  ${BOLD}$label${NC}: $passed/$total passed (tuned=$tuned_pass/$tuned_total, baseline=$baseline_pass/$baseline_total, min_stability=$stability)"

    if [ "$tuned_total" -gt 0 ] && [ "$tuned_pass" -eq 0 ]; then
        fail "$label: zero tuned cases passed — shim may be down"
        ERRORS=$((ERRORS + 1))
    elif [ "$tuned_pass" -lt "$tuned_total" ]; then
        warn "$label: $((tuned_total - tuned_pass))/$tuned_total tuned cases failed"
    else
        ok "$label: all tuned cases passed"
    fi

    # Check if tuned beats or matches baseline
    if [ "$baseline_total" -gt 0 ] && [ "$tuned_pass" -lt "$baseline_pass" ]; then
        warn "$label: tuned ($tuned_pass) underperforms baseline ($baseline_pass) — prompt regression?"
    fi
}

verify_report "codex_shim"    "$OUTPUT_DIR/prompt_codex.json"
verify_report "gemini_shim"   "$OUTPUT_DIR/prompt_gemini.json"
verify_report "head-to-head"  "$OUTPUT_DIR/prompt_head_to_head.json"

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
echo ""
info "Results saved to $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    fail "$ERRORS verification error(s)"
    exit 1
fi

echo ""
ok "All benchmarks passed"
