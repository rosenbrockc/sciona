"""Deterministic failure analyzer — replaces the hunter_analyze_failure LLM call."""

from __future__ import annotations

import re
from typing import Any

_CANDIDATE_RE = re.compile(r"^\s*Name:\s*(.+)$", re.MULTILINE)
_TYPE_RE = re.compile(r"^\s*Type:\s*(.+)$", re.MULTILINE)
_EXPECTED_GOT_RE = re.compile(r"expected\s+['\"]?(\S+)['\"]?.*got\s+['\"]?(\S+)", re.IGNORECASE)
_ARITY_RE = re.compile(r"arity mismatch.*expected\s*~?(\d+).*got\s*(\d+)", re.IGNORECASE)
_UNKNOWN_IDENT_RE = re.compile(r"unknown (?:identifier|constant|namespace) '?@?([A-Za-z_][\w.]*)'?", re.IGNORECASE)
_NO_MODULE_RE = re.compile(r"No module named '([A-Za-z_][\w.]*)'", re.IGNORECASE)
_SYNTAX_RE = re.compile(r"(?:SyntaxError|invalid syntax|parse error|unexpected token)", re.IGNORECASE)
_INCOMPATIBLE_TYPE_RE = re.compile(r"Incompatible (?:return value )?type", re.IGNORECASE)
_ARG_TYPE_RE = re.compile(r'Argument \d+ .* has incompatible type "([^"]+)".*expected "([^"]+)"', re.IGNORECASE)


def _parse_analyze_prompt(user: str) -> tuple[str, str, str, str]:
    """Extract statement, candidate_name, candidate_type, compiler_output."""
    statement = ""
    candidate_name = ""
    candidate_type = ""
    compiler_output = ""

    sections = re.split(r"^##\s+", user, flags=re.MULTILINE)
    for section in sections:
        lines = section.strip()
        if lines.startswith("Predicate"):
            for line in lines.splitlines():
                if line.strip().startswith("Statement:"):
                    statement = line.split(":", 1)[1].strip()
        elif lines.startswith("Failed Candidate"):
            m = _CANDIDATE_RE.search(lines)
            if m:
                candidate_name = m.group(1).strip()
            m = _TYPE_RE.search(lines)
            if m:
                candidate_type = m.group(1).strip()
        elif lines.startswith("Compiler Output"):
            compiler_output = "\n".join(lines.splitlines()[1:]).strip()

    return statement, candidate_name, candidate_type, compiler_output


def _analyze(statement: str, candidate_name: str, candidate_type: str, compiler_output: str) -> str | None:
    """Generate a CAUSE/TARGET/NEXT analysis deterministically."""
    output_lower = compiler_output.lower()

    # Missing import / unknown identifier
    m = _UNKNOWN_IDENT_RE.search(compiler_output)
    if m:
        ident = m.group(1)
        ns = ident.rsplit(".", 1)[0] if "." in ident else ""
        return (
            f"CAUSE: Unknown identifier '{ident}' — missing import or namespace\n"
            f"TARGET: A function with the same semantics that is importable, possibly under namespace '{ns}'\n"
            f"NEXT: Search for '{ns}' namespace variants or alternative implementations"
        )

    m = _NO_MODULE_RE.search(compiler_output)
    if m:
        module = m.group(1)
        return (
            f"CAUSE: Module '{module}' not installed or not importable\n"
            f"TARGET: An equivalent function from an available module\n"
            f"NEXT: Search for alternatives in scipy, numpy, or the target library"
        )

    # Arity mismatch
    m = _ARITY_RE.search(compiler_output)
    if m:
        expected, actual = m.group(1), m.group(2)
        return (
            f"CAUSE: Arity mismatch — predicate expects ~{expected} params, candidate has {actual}\n"
            f"TARGET: A function with {expected} required parameters matching the predicate signature\n"
            f"NEXT: Search for wrapper functions or variants with compatible arity"
        )

    # Argument type incompatibility (Python/mypy)
    m = _ARG_TYPE_RE.search(compiler_output)
    if m:
        got_type, expected_type = m.group(1), m.group(2)
        return (
            f"CAUSE: Argument type mismatch — got '{got_type}', expected '{expected_type}'\n"
            f"TARGET: A function accepting '{got_type}' or a compatible supertype\n"
            f"NEXT: Search for type-compatible variants or conversion wrappers"
        )

    # General type mismatch (expected X got Y)
    m = _EXPECTED_GOT_RE.search(compiler_output)
    if m:
        expected, got = m.group(1), m.group(2)
        return (
            f"CAUSE: Type mismatch — expected '{expected}', got '{got}'\n"
            f"TARGET: A function returning '{expected}' or accepting coercion\n"
            f"NEXT: Search for type-compatible variants of '{candidate_name}'"
        )

    # Incompatible types (Python/mypy)
    if _INCOMPATIBLE_TYPE_RE.search(compiler_output):
        return (
            f"CAUSE: Type incompatibility between candidate '{candidate_name}' and predicate\n"
            f"TARGET: A function with signature matching '{statement}'\n"
            f"NEXT: Search for functions with compatible return and parameter types"
        )

    # Syntax errors
    if _SYNTAX_RE.search(compiler_output):
        return (
            f"CAUSE: Syntax error — candidate name or type may use incompatible notation\n"
            f"TARGET: A function with valid Python/Lean syntax in its declaration\n"
            f"NEXT: Search for alternative formulations avoiding syntax issues"
        )

    # Unsolved goals (Lean/Coq)
    if "unsolved goal" in output_lower or "⊢" in compiler_output:
        return (
            f"CAUSE: Proof obligation remains unsolved after applying '{candidate_name}'\n"
            f"TARGET: A lemma or theorem that directly discharges the goal\n"
            f"NEXT: Search for exact-match lemmas or tactic-friendly variants"
        )

    # Universe mismatch (Lean)
    if "universe" in output_lower:
        return (
            f"CAUSE: Universe level mismatch in '{candidate_name}'\n"
            f"TARGET: A universe-polymorphic variant of the same function\n"
            f"NEXT: Search for @[universe_polymorphic] or explicitly leveled variants"
        )

    return None


class DeterministicFailureAnalyzer:
    """Deterministic compiler error analyzer with LLM fallback.

    Implements the LLMClient protocol so it can be used as a drop-in
    override for the hunter_analyze_failure prompt key.
    """

    _telemetry_provider = "deterministic"
    _telemetry_model = "failure_analyzer_v1"

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        statement, candidate_name, candidate_type, compiler_output = _parse_analyze_prompt(user)

        result = _analyze(statement, candidate_name, candidate_type, compiler_output)
        if result is not None:
            self._last_completion_metadata = {"analysis_source": "deterministic"}
            self._last_error_metadata = {}
            return result

        self._last_completion_metadata = {"analysis_source": "fallback"}
        self._last_error_metadata = {}
        return await self._fallback.complete(system, user)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
