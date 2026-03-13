#!/usr/bin/env bash
# e2e_benchmark.sh — Compare the full ageo-matcher pipeline against a raw LLM
# on real-world goals. Measures: leaf coverage, match quality, latency.
set -uo pipefail  # no -e: individual runs may fail without aborting the whole benchmark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="output/e2e_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
BENCHMARK_PYTHON="${E2E_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
if [ ! -x "$BENCHMARK_PYTHON" ]; then
    BENCHMARK_PYTHON="$(command -v python3)"
fi

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BOLD}[e2e]${NC} $*"; }
ok()    { echo -e "${GREEN}[pass]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GOAL="Detect heart rate from raw ECG signal"
PROVER="python"
export LLM_PROVIDER="${E2E_LLM_PROVIDER:-codex_shim}"
export LLM_MODEL="${E2E_LLM_MODEL:-gpt-5.3-codex}"

# Force all LLM routing through the chosen provider to avoid stale shim
# sockets from other providers (e.g. claude_shim set in .env).
export AGEOM_LLM_PROVIDER="$LLM_PROVIDER"
export AGEOM_LLM_MODEL="$LLM_MODEL"
export AGEOM_HUNTER_LLM_PROVIDER="$LLM_PROVIDER"
export AGEOM_HUNTER_LLM_MODEL="$LLM_MODEL"
export AGEOM_ARCHITECT_LLM_PROVIDER="$LLM_PROVIDER"
export AGEOM_ARCHITECT_LLM_MODEL="$LLM_MODEL"
if [[ "$LLM_PROVIDER" == *_cli ]]; then
    export AGEOM_ALLOW_LEGACY_SUBPROCESS_PROVIDERS="${E2E_ALLOW_LEGACY_SUBPROCESS:-true}"
fi

PROMPT_OVERRIDE_KEYS=(
    ARCHITECT_STRATEGY
    ARCHITECT_DECOMPOSE
    ARCHITECT_CRITIQUE
    HUNTER_SCORE
    HUNTER_REFORMULATE
    HUNTER_ANALYZE_FAILURE
    SYNTHESIZER_REPAIR
    SYNTHESIZER_TACTIC
    INGESTER_CHUNK
    INGESTER_HOIST_STATE
    INGESTER_ABSTRACT
    INGESTER_FIX_TYPE
    INGESTER_FIX_GHOST
    INGESTER_OPAQUE_WITNESS
    INGESTER_FIX_MESSAGE_CYCLE
    INGESTER_DECOMPOSE
    ORCHESTRATOR_REFINE
)
for key in "${PROMPT_OVERRIDE_KEYS[@]}"; do
    export "AGEOM_${key}_LLM_PROVIDER=$LLM_PROVIDER"
    export "AGEOM_${key}_LLM_MODEL=$LLM_MODEL"
done

# Force FAISS semantic index — the default retrieval policy degrades to
# lexical when catalog confidence is < 0.70 (medium band), which prevents
# the benchmark from exercising the full semantic search pipeline.
export AGEOM_SEMANTIC_INDEX_BACKEND=faiss
MODE_TIMEOUT_S="${E2E_MODE_TIMEOUT_S:-240}"
RAW_TIMEOUT_S="${E2E_RAW_TIMEOUT_S:-120}"

# Ground truth: the essential atoms for ECG heart rate detection.
# Each entry is a keyword pattern that should appear in at least one matched
# function name. Order = pipeline order (filter → detect → compute).
GROUND_TRUTH_PATTERNS=(
    "bandpass_filter"
    "r_peak_detection|hamilton_segment"
    "heart_rate_computation"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
elapsed_ms() {
    local start="$1" end="$2"
    "$BENCHMARK_PYTHON" -c "print(int(($end - $start) * 1000))"
}

check_ground_truth() {
    local matches_json="$1"
    local label="$2"
    local hit=0
    local miss=0
    local missed_patterns=()

    for pattern in "${GROUND_TRUTH_PATTERNS[@]}"; do
        if "$BENCHMARK_PYTHON" -c "
import json, re, sys
matches = json.load(open('$matches_json'))
names = []
for m in matches:
    vm = m.get('verified_match')
    if vm and vm.get('verified'):
        decl = vm.get('candidate', {}).get('declaration', {})
        names.append(decl.get('name', ''))
    for c in m.get('all_candidates', []):
        names.append(c.get('declaration', {}).get('name', ''))
pattern = r'$pattern'
found = any(re.search(pattern, n, re.IGNORECASE) for n in names)
sys.exit(0 if found else 1)
" 2>/dev/null; then
            hit=$((hit + 1))
        else
            miss=$((miss + 1))
            missed_patterns+=("$pattern")
        fi
    done

    local total=${#GROUND_TRUTH_PATTERNS[@]}
    echo -e "  ${BOLD}$label${NC}: $hit/$total ground truth atoms found"
    if [ "$miss" -gt 0 ]; then
        warn "$label: missing patterns: ${missed_patterns[*]}"
    else
        ok "$label: all ground truth atoms covered"
    fi
    echo "$hit" > "$OUTPUT_DIR/${label}_hits.txt"
}

# run_pipeline MODE OUTPUT_DIR — runs ageom run, captures latency, tolerates failure
# Uses the project interpreter to invoke ageom.cli.main directly (avoids sandbox issues with
# native library loading when using the ageom entry point script).
run_pipeline() {
    local mode="$1"
    local out_dir="$2"
    local ms_var="${3}"  # name of variable to set with elapsed ms
    mkdir -p "$out_dir"

    info "Running pipeline ($mode): $GOAL"
    local start end elapsed rc timed_out pid deadline now
    start=$("$BENCHMARK_PYTHON" -c "import time; print(time.time())")
    (
        "$BENCHMARK_PYTHON" -c "
import sys; sys.argv = ['ageom', 'run', '$GOAL', '--prover', '$PROVER', '--mode', '$mode', '--llm-provider', '$LLM_PROVIDER', '--llm-model', '$LLM_MODEL', '--output', '$out_dir']
from ageom.cli import main; main()
" 2>&1
    ) | tee "$out_dir/stdout.txt" &
    pid=$!
    rc=0
    timed_out=0
    deadline=$("$BENCHMARK_PYTHON" -c "import time; print(time.time() + float('$MODE_TIMEOUT_S'))")
    while kill -0 "$pid" 2>/dev/null; do
        now=$("$BENCHMARK_PYTHON" -c "import time; print(time.time())")
        if "$BENCHMARK_PYTHON" -c "import sys; sys.exit(0 if float('$now') < float('$deadline') else 1)"; then
            sleep 1
            continue
        fi
        timed_out=1
        warn "$mode exceeded ${MODE_TIMEOUT_S}s; terminating benchmark wrapper"
        kill "$pid" 2>/dev/null || true
        sleep 2
        kill -9 "$pid" 2>/dev/null || true
        break
    done
    if [ "$timed_out" -eq 0 ]; then
        wait "$pid"
        rc=$?
    else
        wait "$pid" 2>/dev/null || true
        rc=124
    fi
    end=$("$BENCHMARK_PYTHON" -c "import time; print(time.time())")
    elapsed=$(elapsed_ms "$start" "$end")

    # Export latency for caller
    eval "$ms_var=$elapsed"

    if [ "$rc" -eq 124 ] && [ -f "$out_dir/matches.json" ]; then
        warn "$mode timed out after ${elapsed}ms but produced matches.json"
    elif [ "$rc" -ne 0 ]; then
        fail "$mode crashed (exit $rc) after ${elapsed}ms — see $out_dir/stdout.txt"
    else
        info "  $mode completed in ${elapsed}ms"
    fi
    return "$rc"
}

# ---------------------------------------------------------------------------
# 1. Pipeline: rapid mode
# ---------------------------------------------------------------------------
RAPID_DIR="$OUTPUT_DIR/pipeline_rapid"
RAPID_MS=0
run_pipeline rapid "$RAPID_DIR" RAPID_MS || true

# ---------------------------------------------------------------------------
# 2. Pipeline: structured mode
# ---------------------------------------------------------------------------
STRUCTURED_DIR="$OUTPUT_DIR/pipeline_structured"
STRUCTURED_MS=0
run_pipeline structured "$STRUCTURED_DIR" STRUCTURED_MS || true

# ---------------------------------------------------------------------------
# 3. Pipeline: verified mode
# ---------------------------------------------------------------------------
VERIFIED_DIR="$OUTPUT_DIR/pipeline_verified"
VERIFIED_MS=0
run_pipeline verified "$VERIFIED_DIR" VERIFIED_MS || true

# ---------------------------------------------------------------------------
# 4. Raw LLM baseline — single-shot function identification
# ---------------------------------------------------------------------------
info "Running raw LLM baseline: $LLM_PROVIDER"
RAW_DIR="$OUTPUT_DIR/raw_llm"
export RAW_DIR
mkdir -p "$RAW_DIR"

# Build function list from index (declarations stored in msgpack)
"$BENCHMARK_PYTHON" -c "
import msgpack
from pathlib import Path

msgpack_path = Path('data/index/declarations.msgpack')
if not msgpack_path.exists():
    print('ERROR: index declarations.msgpack not found — run ageom index build first')
    exit(1)

with open(msgpack_path, 'rb') as f:
    data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)

names = []
for _id, entry in sorted(data.items(), key=lambda x: int(x[0])):
    name = entry.get('name', '')
    sig = entry.get('type_signature', '')
    if name:
        line = f'{name} : {sig}' if sig else name
        names.append(line)

with open('$RAW_DIR/function_list.txt', 'w') as f:
    f.write('\n'.join(names))
print(f'Exported {len(names)} function signatures')
" 2>&1

FUNC_COUNT=$(wc -l < "$RAW_DIR/function_list.txt" | tr -d ' ')
info "  $FUNC_COUNT functions in library index"

# Build the raw LLM prompt
cat > "$RAW_DIR/system_prompt.txt" <<'SYSPROMPT'
You are an expert algorithm engineer. Given a goal and a library of available functions, identify which functions from the library are needed to accomplish the goal. Return a JSON object with a 'functions' key containing an array of the exact function names needed, in pipeline order.
SYSPROMPT

cat > "$RAW_DIR/user_prompt.txt" <<USERPROMPT
Goal: $GOAL

Available functions (name : type_signature):
$(cat "$RAW_DIR/function_list.txt")

Return JSON: {"functions": ["function1", "function2", ...]}
USERPROMPT

RAW_MS=0
START=$("$BENCHMARK_PYTHON" -c "import time; print(time.time())")
(
"$BENCHMARK_PYTHON" << 'PYEOF' 2>&1
import asyncio, json, re, sys, os

RAW_DIR = os.environ.get("RAW_DIR", "")

async def run():
    from ageom.config import AgeomConfig
    from ageom.hunter.llm import create_llm_client

    config = AgeomConfig()
    provider = os.environ.get("LLM_PROVIDER", "codex_shim")
    model = os.environ.get("LLM_MODEL", "gpt-5.3-codex")

    llm = create_llm_client(
        provider=provider,
        model=model,
        max_tokens=config.llm_max_tokens,
        anthropic_api_key=config.anthropic_api_key,
        openai_api_key=config.openai_api_key,
        openai_base_url=config.openai_base_url,
        llama_cpp_base_url=config.llama_cpp_base_url,
        llama_cpp_api_key=config.llama_cpp_api_key,
        allow_legacy_subprocess=getattr(
            config, "allow_legacy_subprocess_providers", False
        ),
    )

    with open(f"{RAW_DIR}/system_prompt.txt") as f:
        system = f.read()
    with open(f"{RAW_DIR}/user_prompt.txt") as f:
        user = f.read()

    response = await llm.complete(system, user)
    with open(f"{RAW_DIR}/response.txt", "w") as f:
        f.write(response)

    # Parse the JSON response
    functions = []
    try:
        text = response.strip()
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            end = len(lines) - 1 if lines[-1].strip().startswith("```") else len(lines)
            text = "\n".join(lines[1:end])
        parsed = json.loads(text)
        functions = parsed.get("functions", [])
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                functions = parsed.get("functions", [])
            except json.JSONDecodeError:
                pass
        if not functions:
            print("WARNING: could not parse JSON from LLM response", file=sys.stderr)

    # Convert to matches.json-compatible format for ground truth checking
    matches = []
    for fn in functions:
        matches.append({
            "verified_match": None,
            "all_candidates": [{
                "declaration": {"name": fn},
                "score": 1.0,
                "retrieval_method": "raw_llm"
            }],
            "all_verifications": []
        })
    with open(f"{RAW_DIR}/matches.json", "w") as f:
        json.dump(matches, f, indent=2)
    print(f"Raw LLM identified {len(functions)} functions: {functions}")

asyncio.run(run())
PYEOF
) | tee "$RAW_DIR/stdout.txt" &
RAW_PID=$!
RAW_RC=0
RAW_TIMED_OUT=0
RAW_DEADLINE=$("$BENCHMARK_PYTHON" -c "import time; print(time.time() + float('$RAW_TIMEOUT_S'))")
while kill -0 "$RAW_PID" 2>/dev/null; do
    RAW_NOW=$("$BENCHMARK_PYTHON" -c "import time; print(time.time())")
    if "$BENCHMARK_PYTHON" -c "import sys; sys.exit(0 if float('$RAW_NOW') < float('$RAW_DEADLINE') else 1)"; then
        sleep 1
        continue
    fi
    RAW_TIMED_OUT=1
    warn "raw LLM exceeded ${RAW_TIMEOUT_S}s; terminating benchmark wrapper"
    kill "$RAW_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$RAW_PID" 2>/dev/null || true
    break
done
if [ "$RAW_TIMED_OUT" -eq 0 ]; then
    wait "$RAW_PID"
    RAW_RC=$?
else
    wait "$RAW_PID" 2>/dev/null || true
    RAW_RC=124
fi
END=$("$BENCHMARK_PYTHON" -c "import time; print(time.time())")
RAW_MS=$(elapsed_ms "$START" "$END")
if [ "$RAW_RC" -eq 124 ] && [ -f "$RAW_DIR/matches.json" ]; then
    warn "raw LLM timed out after ${RAW_MS}ms but produced matches.json"
elif [ "$RAW_RC" -ne 0 ]; then
    fail "raw LLM crashed (exit $RAW_RC) after ${RAW_MS}ms — see $RAW_DIR/stdout.txt"
else
    info "  raw LLM completed in ${RAW_MS}ms"
fi

# ---------------------------------------------------------------------------
# 5. Ground truth evaluation
# ---------------------------------------------------------------------------
echo ""
info "Evaluating ground truth coverage..."
echo ""

for label_dir in \
    "rapid:$RAPID_DIR" \
    "structured:$STRUCTURED_DIR" \
    "verified:$VERIFIED_DIR" \
    "raw_llm:$RAW_DIR"; do
    label="${label_dir%%:*}"
    dir="${label_dir#*:}"
    if [ -f "$dir/matches.json" ]; then
        check_ground_truth "$dir/matches.json" "$label"
    else
        warn "$label: no matches.json found (run may have crashed)"
        echo "0" > "$OUTPUT_DIR/${label}_hits.txt"
    fi
done

# ---------------------------------------------------------------------------
# 6. Summary report
# ---------------------------------------------------------------------------
echo ""
info "Generating summary..."

"$BENCHMARK_PYTHON" -c "
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
total_gt = ${#GROUND_TRUTH_PATTERNS[@]}

def read_hits(label):
    p = output_dir / f'{label}_hits.txt'
    return int(p.read_text().strip()) if p.exists() else 0

def read_matches(label_dir):
    p = Path(label_dir) / 'matches.json'
    if not p.exists():
        return 0, 0
    matches = json.load(open(p))
    total = len(matches)
    verified = sum(1 for m in matches
                   if m.get('verified_match') and m['verified_match'].get('verified'))
    return total, verified

rapid_total, rapid_verified = read_matches('$RAPID_DIR')
structured_total, structured_verified = read_matches('$STRUCTURED_DIR')
verified_total, verified_verified = read_matches('$VERIFIED_DIR')
raw_total, raw_verified = read_matches('$RAW_DIR')

report = {
    'goal': '$GOAL',
    'prover': '$PROVER',
    'llm_provider': '$LLM_PROVIDER',
    'ground_truth_atoms': total_gt,
    'results': {
        'rapid': {
            'latency_ms': $RAPID_MS,
            'matches_total': rapid_total,
            'matches_verified': rapid_verified,
            'ground_truth_hits': read_hits('rapid'),
            'ground_truth_coverage': round(read_hits('rapid') / total_gt, 2),
        },
        'structured': {
            'latency_ms': $STRUCTURED_MS,
            'matches_total': structured_total,
            'matches_verified': structured_verified,
            'ground_truth_hits': read_hits('structured'),
            'ground_truth_coverage': round(read_hits('structured') / total_gt, 2),
        },
        'verified': {
            'latency_ms': $VERIFIED_MS,
            'matches_total': verified_total,
            'matches_verified': verified_verified,
            'ground_truth_hits': read_hits('verified'),
            'ground_truth_coverage': round(read_hits('verified') / total_gt, 2),
        },
        'raw_llm': {
            'latency_ms': $RAW_MS,
            'matches_total': raw_total,
            'matches_verified': 0,
            'ground_truth_hits': read_hits('raw_llm'),
            'ground_truth_coverage': round(read_hits('raw_llm') / total_gt, 2),
        },
    },
}

with open(output_dir / 'summary.json', 'w') as f:
    json.dump(report, f, indent=2)

# Print table
print()
print('variant | latency | matches | verified | GT coverage')
print('--- | ---: | ---: | ---: | ---:')
for variant in ['rapid', 'structured', 'verified', 'raw_llm']:
    r = report['results'][variant]
    lat = f\"{r['latency_ms']}ms\"
    gt = f\"{r['ground_truth_hits']}/{total_gt}\"
    cov = f\"{r['ground_truth_coverage']:.0%}\"
    print(f\"{variant} | {lat} | {r['matches_total']} | {r['matches_verified']} | {gt} ({cov})\")
print()
" 2>&1 | tee "$OUTPUT_DIR/summary_table.txt"

# ---------------------------------------------------------------------------
# 7. Final verdict
# ---------------------------------------------------------------------------
echo ""
info "Results saved to $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/"

VERIFIED_HITS=$(cat "$OUTPUT_DIR/verified_hits.txt" 2>/dev/null || echo "0")
RAW_HITS=$(cat "$OUTPUT_DIR/raw_llm_hits.txt" 2>/dev/null || echo "0")
TOTAL_GT=${#GROUND_TRUTH_PATTERNS[@]}

echo ""
if [ "$VERIFIED_HITS" -ge "$RAW_HITS" ] && [ "$VERIFIED_HITS" -eq "$TOTAL_GT" ]; then
    ok "Pipeline (verified) achieves full ground truth coverage"
elif [ "$VERIFIED_HITS" -ge "$RAW_HITS" ]; then
    ok "Pipeline (verified) matches or beats raw LLM ($VERIFIED_HITS/$TOTAL_GT vs $RAW_HITS/$TOTAL_GT)"
elif [ "$VERIFIED_HITS" -lt "$RAW_HITS" ]; then
    warn "Raw LLM outperforms pipeline ($RAW_HITS/$TOTAL_GT vs $VERIFIED_HITS/$TOTAL_GT) — investigate decomposition quality"
fi

echo ""
ok "E2E benchmark complete"
