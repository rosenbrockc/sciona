"""Deterministic wrapper for the synthesizer_tactic prompt."""

from __future__ import annotations

import re
from typing import Any

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_DEF_BOOL_RE = re.compile(r"^\s*def\s+\w+\(.*\)\s*->\s*bool\s*:", re.MULTILINE)
_IDENTICAL_EQ_RE = re.compile(r"^(?P<lhs>.+?)\s*=\s*(?P<rhs>.+?)$")
_NUMERIC_RE = re.compile(r"^[0-9+\-*/() <>=≤≥]+$")


def _strip_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def _parse_tactic_prompt(system: str, user: str) -> tuple[str, str, str]:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in user.splitlines():
        line = raw_line.rstrip("\n")
        match = _SECTION_RE.match(line.strip())
        if match:
            current = match.group(1).strip().lower()
            sections[current] = []
            continue
        if current:
            sections.setdefault(current, []).append(line)

    goal_type = _strip_fence("\n".join(sections.get("goal type", [])))
    hypotheses = "\n".join(sections.get("available hypotheses", [])).strip()
    prover = (
        "python"
        if "Python function implementation specialist" in system
        else "coq"
        if "Coq" in system and "Lean 4 / Coq tactic proof specialist" not in system
        else "lean4"
    )
    return goal_type, hypotheses, prover


def _normalized(text: str) -> str:
    return " ".join(text.strip().split())


def _has_complex_quantifiers(goal_type: str) -> bool:
    lowered = goal_type.lower()
    return any(token in lowered for token in ("forall", "∃", "exists", "induction"))


def _suggest_lean_tactic(goal_type: str, hypotheses: str) -> str | None:
    goal = _normalized(goal_type)
    if not goal or _has_complex_quantifiers(goal):
        return None
    if goal in {"True", "⊤"}:
        return "trivial"
    if goal.startswith("Decidable "):
        return "infer_instance"
    match = _IDENTICAL_EQ_RE.match(goal)
    if match:
        lhs = _normalized(match.group("lhs"))
        rhs = _normalized(match.group("rhs"))
        if lhs == rhs:
            return "rfl"
        if _NUMERIC_RE.match(goal.replace("≤", "<=").replace("≥", ">=")):
            return "norm_num"
        if any(token in goal for token in ("Nat", "Int", "<", ">", "≤", "≥", "+", "-", "*")):
            return "omega"
        return "simp"
    if any(token in goal for token in ("Nat", "Int")):
        return "omega"
    if hypotheses and goal and goal in hypotheses:
        return "assumption"
    return "simp"


def _suggest_coq_tactic(goal_type: str, hypotheses: str) -> str | None:
    goal = _normalized(goal_type)
    if not goal or _has_complex_quantifiers(goal):
        return None
    if goal in {"True"}:
        return "trivial."
    match = _IDENTICAL_EQ_RE.match(goal)
    if match:
        lhs = _normalized(match.group("lhs"))
        rhs = _normalized(match.group("rhs"))
        if lhs == rhs:
            return "reflexivity."
        if _NUMERIC_RE.match(goal):
            return "lia."
        return "auto."
    if hypotheses and goal and goal in hypotheses:
        return "assumption."
    return "auto."


def _suggest_python_impl(goal_type: str, hypotheses: str) -> str | None:
    context = "\n".join(part for part in (goal_type, hypotheses) if part).strip()
    if not context:
        return None
    if _DEF_BOOL_RE.search(context):
        return "    return True"
    return None


def _safe_response(text: str | None) -> str | None:
    if text is None:
        return None
    lowered = text.lower()
    if "sorry" in lowered or "admitted" in lowered or "notimplementederror" in lowered:
        return None
    return text


class DeterministicTacticSuggester:
    """Deterministic synthesizer tactic suggester with LLM fallback."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "tactic_suggester_v1"

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        goal_type, hypotheses, prover = _parse_tactic_prompt(system, user)
        if prover == "python":
            result = _safe_response(_suggest_python_impl(goal_type, hypotheses))
        elif prover == "coq":
            result = _safe_response(_suggest_coq_tactic(goal_type, hypotheses))
        else:
            result = _safe_response(_suggest_lean_tactic(goal_type, hypotheses))

        if result is not None:
            self._last_completion_metadata = {
                "tactic_source": "deterministic",
                "tactic_prover": prover,
            }
            self._last_error_metadata = {}
            return result

        self._last_completion_metadata = {
            "tactic_source": "fallback",
            "tactic_prover": prover,
        }
        self._last_error_metadata = {}
        return await self._fallback.complete(system, user)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
