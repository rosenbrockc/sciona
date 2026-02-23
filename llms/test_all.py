#!/usr/bin/env python3
"""Smoke-test every installed Ollama model against representative prompts.

Each test mirrors a real prompt tier from the AGEO-Matcher pipeline:
  - qwen2.5-coder:7b  → light tier (enum pick, ranking, query gen, mypy fix)
  - qwen3:14b          → medium tier (critique, diagnosis, abstraction, patches)
  - deepseek-r1:32b    → heavy tier (decomposition, proof, chunking)

Usage:
    python llms/test_all.py                     # test all installed models
    python llms/test_all.py qwen2.5-coder:7b    # test one model
    OLLAMA_BASE_URL=http://host:11434/v1 python llms/test_all.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import httpx

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
TIMEOUT = 300  # seconds per request


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------


@dataclass
class PromptTest:
    name: str  # maps to a prompt key or description
    model: str
    system: str
    user: str
    validate: str  # "json_object" | "json_array" | "json_patch_array" | "text"
    required_keys: list[str] | None = None  # for json_object validation
    max_tokens: int = 2048


TESTS: list[PromptTest] = [
    # === qwen2.5-coder:7b — light tier ===
    PromptTest(
        name="architect_strategy",
        model="qwen2.5-coder:7b",
        system=(
            "You are an expert algorithm designer. Given a high-level algorithmic "
            "goal, select the best algorithmic paradigm for decomposing it.\n\n"
            'You must respond with ONLY a JSON object (no markdown fences):\n'
            '{"paradigm": "<value>", "rationale": "<why>", "variant_hint": "<hint>"}\n\n'
            "The paradigm must be one of: sorting, searching, divide_and_conquer, "
            "greedy, dynamic_programming, graph_traversal"
        ),
        user="Goal: Find the shortest path between two nodes in a weighted graph.",
        validate="json_object",
        required_keys=["paradigm", "rationale"],
    ),
    PromptTest(
        name="hunter_score",
        model="qwen2.5-coder:7b",
        system=(
            "You are a formal mathematics expert. Given a predicate and candidates, "
            "rank them by likelihood of being the correct match.\n"
            "Return a JSON array of candidate indices (0-based), ordered from most "
            "to least likely. Example: [2, 0, 4]"
        ),
        user=(
            "## Predicate\nStatement: ∀ n m : ℕ, n + m = m + n\n"
            "Description: Addition of natural numbers is commutative\n\n"
            "## Candidates\n[0] Nat.mul_comm : ∀ n m, n * m = m * n\n"
            "[1] Nat.add_comm : ∀ n m, n + m = m + n\n"
            "[2] Nat.add_assoc : ∀ n m k, n + (m + k) = (n + m) + k\n\n"
            "Return a JSON array of indices ordered by likelihood:"
        ),
        validate="json_array",
    ),
    PromptTest(
        name="hunter_reformulate",
        model="qwen2.5-coder:7b",
        system=(
            "You are a formal mathematics search expert. Generate new search queries "
            "to find the matching library function.\n"
            "Return a JSON array of 3-5 query strings. "
            'Example: ["Nat.add_comm", "addition commutative", "n + m = m + n"]'
        ),
        user=(
            "## Predicate\nStatement: ∀ n m : ℕ, n + m = m + n\n"
            "Description: Commutativity of natural number addition\nProver: Lean4\n\n"
            "## Previous Queries Tried\n- add_comm\n- Nat.comm\n\n"
            "Generate new search queries as a JSON array:"
        ),
        validate="json_array",
    ),
    PromptTest(
        name="ingester_fix_type",
        model="qwen2.5-coder:7b",
        system=(
            "You are a Python type-checking repair agent. Given mypy errors and "
            "source code, produce minimal line replacements to fix the errors.\n"
            "Return valid JSON only."
        ),
        user=(
            'mypy errors:\ntest.py:3: error: Incompatible return value type '
            '(got "str", expected "int")  [return-value]\n\n'
            "Generated source:\n```python\ndef add(a: int, b: int) -> int:\n"
            '    result = a + b\n    return str(result)\n```\n\n'
            "Return JSON array of fixes:\n"
            '[{"line_start": <int>, "line_end": <int>, "replacement": "<fixed code>"}]'
        ),
        validate="json_patch_array",
    ),
    # === qwen3:14b — medium tier ===
    PromptTest(
        name="architect_critique",
        model="qwen3:14b",
        system=(
            "You are an expert algorithm critic. Evaluate whether a proposed "
            "decomposition is correct and complete.\n\n"
            "You must respond with ONLY a JSON object:\n"
            '{"approved": <true|false>, "reason": "<explanation>", '
            '"io_issues": ["<issue>"], "flagged_nodes": ["<node>"]}'
        ),
        user=(
            "Parent node:\n  Name: shortest_path\n  Description: Dijkstra's algorithm\n"
            "  Inputs: [graph, source]\n  Outputs: [distances]\n\n"
            "Proposed sub-nodes:\n"
            "  1. init_distances: set all to infinity, source to 0\n"
            "  2. relax_edges: iterate and relax\n"
            "  3. extract_min: priority queue extract\n\n"
            "Evaluate this decomposition."
        ),
        validate="json_object",
        required_keys=["approved", "reason"],
    ),
    PromptTest(
        name="hunter_analyze_failure",
        model="qwen3:14b",
        system=(
            "You are a formal mathematics expert analyzing why a candidate function "
            "failed to type-check. Explain why the match failed and suggest what the "
            "correct match might look like. Be concise."
        ),
        user=(
            "## Predicate\nStatement: ∀ n : ℕ, 0 + n = n\n\n"
            "## Failed Candidate\nName: Nat.add_comm\n"
            "Type: ∀ n m : ℕ, n + m = m + n\n\n"
            "## Compiler Output\ntype mismatch: expected ∀ n, 0 + n = n, "
            "got ∀ n m, n + m = m + n\n\nAnalysis:"
        ),
        validate="text",
    ),
    PromptTest(
        name="ingester_abstract",
        model="qwen3:14b",
        system=(
            "You are the Conceptual Abstraction Agent. Describe ingested algorithmic "
            "atoms in a domain-agnostic way for cross-disciplinary semantic retrieval.\n"
            "Return valid JSON only."
        ),
        user=(
            "Atom: bandpass_filter\nDescription: Apply a Butterworth bandpass filter\n"
            "Concept type: signal_filter\n\n"
            "Inputs:\n  signal: np.ndarray (1D time series)\n  low: float\n  high: float\n\n"
            "Outputs:\n  filtered: np.ndarray (1D time series)\n\n"
            "Return JSON:\n"
            '{"abstract_name": "<name>", "conceptual_transform": "<description>", '
            '"abstract_inputs": ["<desc>"], "abstract_outputs": ["<desc>"], '
            '"algorithmic_properties": ["<prop>"], '
            '"cross_disciplinary_applications": ["<app1>", "<app2>", "<app3>"]}'
        ),
        validate="json_object",
        required_keys=["abstract_name", "conceptual_transform"],
    ),
    PromptTest(
        name="ingester_hoist_state",
        model="qwen3:14b",
        system=(
            "You are a software architect generating StateModelSpecs for immutable "
            "state threading between pure macro-atoms.\n"
            "Return valid JSON only. No explanation. No thinking. JSON only."
        ),
        max_tokens=4096,
        user=(
            "Cross-window attributes: [self.x, self.P, self.Q, self.R]\n\n"
            "Macro-atom plan:\n"
            '  [{"name": "predict", "methods": ["_predict"]},\n'
            '   {"name": "update", "methods": ["_update"]}]\n\n'
            "Return JSON:\n"
            '{"state_models": [{"model_name": "<Name>State", '
            '"fields": [["<field>", "<type>"]], '
            '"source_attrs": ["<self.attr>"], "docstring": "<desc>"}]}'
        ),
        validate="json_object",
        required_keys=["state_models"],
    ),
    PromptTest(
        name="orchestrator_refine",
        model="qwen3:14b",
        system=(
            "You are an algorithm decomposition expert. A predicate could not be "
            "grounded to a single library function. Suggest 2-3 finer-grained "
            "sub-predicates that together implement the same functionality.\n"
            "Reply with a JSON array of objects: "
            '[{"name": "...", "description": "...", "type_signature": "..."}]'
        ),
        user=(
            "Predicate: matrix_inverse\n"
            "Description: Compute the inverse of a square matrix\n"
            "Match errors:\n- No exact match for full inverse\n"
            "- Partial match: LU decomposition\n\n"
            "Split this into 2-3 finer sub-predicates."
        ),
        validate="json_array",
    ),
    # === deepseek-r1:32b — heavy tier (optional, for completeness) ===
    PromptTest(
        name="architect_decompose",
        model="deepseek-r1:32b",
        system=(
            "You are an expert algorithm designer. Decompose the given node into "
            "sub-nodes and data-flow edges.\n\n"
            "You must respond with ONLY a JSON object:\n"
            '{"sub_nodes": [{"name": "<name>", "description": "<desc>", '
            '"is_atomic": false}], '
            '"edges": [{"source_name": "<src>", "target_name": "<tgt>"}]}'
        ),
        user=(
            "Node to decompose:\n  Name: merge_sort\n"
            "  Description: Sort an array using merge sort\n"
            "  Inputs: [array: List[int]]\n  Outputs: [sorted: List[int]]\n\n"
            "Decompose into 2 or more sub-nodes with edges."
        ),
        validate="json_object",
        required_keys=["sub_nodes", "edges"],
        max_tokens=4096,
    ),
    PromptTest(
        name="synthesizer_tactic",
        model="deepseek-r1:32b",
        system=(
            "You are a Lean 4 tactic proof specialist. Generate a tactic-mode "
            "proof body to replace `sorry`.\n"
            "Respond with ONLY the tactic body (no `by` prefix, no backticks)."
        ),
        user=(
            "## Goal Type\n∀ n : ℕ, 0 + n = n\n\n"
            "## Available Lemmas\nNat.zero_add : ∀ n, 0 + n = n\n\n"
            "Generate a tactic proof body for this goal."
        ),
        validate="text",
        max_tokens=4096,
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_installed_models() -> set[str]:
    """Return set of model names available in Ollama."""
    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, check=False
    )
    models = set()
    for line in result.stdout.strip().splitlines()[1:]:  # skip header
        name = line.split()[0] if line.strip() else ""
        if name:
            models.add(name)
    return models


def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.index("\n") if "\n" in text else len(text)
        text = text[first_nl + 1 :]
    if text.endswith("```"):
        text = text[: -3]
    return text.strip()


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks (deepseek-r1/qwen3 chain-of-thought)."""
    import re

    # Remove paired <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Handle unclosed <think> (model hit max_tokens inside thinking)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


def call_model(test: PromptTest) -> str:
    """Send a chat completion request and return the response text."""
    # Qwen3 and deepseek-r1 use extended thinking by default, which burns
    # tokens on chain-of-thought.  We keep thinking enabled (the real
    # pipeline uses it) but give the model enough budget.  For JSON
    # prompts we also append a nudge to keep the answer outside <think>.
    system = test.system
    if test.validate != "text":
        system += "\nIMPORTANT: Output the JSON directly. Do not wrap it in think tags."
    payload = {
        "model": test.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": test.user},
        ],
        "max_tokens": test.max_tokens,
        "temperature": 0.2,
    }
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{BASE_URL}/chat/completions", json=payload)
        resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def validate_response(test: PromptTest, raw: str) -> tuple[bool, str]:
    """Validate response against expected format. Returns (ok, detail)."""
    text = strip_think_tags(raw)
    text = strip_markdown_fences(text)

    if test.validate == "text":
        if len(text) < 10:
            return False, f"response too short ({len(text)} chars)"
        return True, f"{len(text)} chars"

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # Show first 200 chars of response for debugging
        preview = text[:200].replace("\n", "\\n")
        return False, f"invalid JSON: {exc} — got: {preview}"

    if test.validate == "json_object":
        if not isinstance(parsed, dict):
            return False, f"expected object, got {type(parsed).__name__}"
        if test.required_keys:
            missing = [k for k in test.required_keys if k not in parsed]
            if missing:
                return False, f"missing keys: {missing}"
        return True, f"object with {len(parsed)} keys"

    if test.validate in ("json_array", "json_patch_array"):
        if not isinstance(parsed, list):
            return False, f"expected array, got {type(parsed).__name__}"
        if len(parsed) == 0:
            return False, "empty array"
        if test.validate == "json_patch_array":
            first = parsed[0]
            if not isinstance(first, dict):
                return False, f"expected array of objects, got {type(first).__name__}"
        return True, f"array with {len(parsed)} items"

    return False, f"unknown validation type: {test.validate}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    filter_model = sys.argv[1] if len(sys.argv) > 1 else None

    # Check Ollama is reachable
    try:
        httpx.get(f"{BASE_URL.replace('/v1', '')}/api/tags", timeout=5)
    except httpx.ConnectError:
        print("ERROR: Cannot reach Ollama. Is it running? (ollama serve)")
        return 1

    installed = get_installed_models()
    if not installed:
        print("ERROR: No models installed. Run: bash llms/install_defaults.sh")
        return 1

    print(f"Installed models: {', '.join(sorted(installed))}")
    print()

    tests = TESTS
    if filter_model:
        tests = [t for t in TESTS if t.model == filter_model]
        if not tests:
            print(f"No tests defined for model: {filter_model}")
            return 1

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        tag = f"[{test.model}] {test.name}"
        if test.model not in installed:
            print(f"  SKIP  {tag}  (model not installed)")
            skipped += 1
            continue

        print(f"  RUN   {tag} ...", end="", flush=True)
        t0 = time.time()
        try:
            raw = call_model(test)
            elapsed = time.time() - t0
            ok, detail = validate_response(test, raw)
            if ok:
                print(f"\r  PASS  {tag}  ({elapsed:.1f}s, {detail})")
                passed += 1
            else:
                print(f"\r  FAIL  {tag}  ({elapsed:.1f}s) — {detail}")
                failed += 1
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"\r  FAIL  {tag}  ({elapsed:.1f}s) — {exc}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
