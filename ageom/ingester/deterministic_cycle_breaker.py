"""Deterministic wrapper for the ingester_fix_message_cycle prompt."""

from __future__ import annotations

import json
import re
from typing import Any

_DEADLOCKED_RE = re.compile(r"^Deadlocked nodes:\s*(.+)$", re.MULTILINE)
_CYCLE_RE = re.compile(
    r"^Cycle edges:\s*(?P<edges>.*?)^Current witness source:\s*$",
    re.MULTILINE | re.DOTALL,
)
_SOURCE_RE = re.compile(
    r"Current witness source:\n```python\n(?P<source>.*?)\n```",
    re.DOTALL,
)
_CONVERGED_RE = re.compile(r"^(\s*)converged = False\s*$", re.MULTILINE)
_UNBOUNDED_LOOP_RE = re.compile(r"^(?P<indent>\s*)while True:\s*$", re.MULTILINE)
_NEW_MSG_RE = re.compile(r"^(?P<indent>\s*)new_msg\s*=\s*(?P<expr>.+)$", re.MULTILINE)


def _line_bounds(source: str, match: re.Match[str]) -> tuple[int, int]:
    start_line = source.count("\n", 0, match.start()) + 1
    end_line = source.count("\n", 0, match.end()) + 1
    return start_line, end_line


def _parse_cycle_prompt(user: str) -> tuple[list[str], list[str], str]:
    deadlock_match = _DEADLOCKED_RE.search(user)
    deadlocked_nodes = [
        part.strip()
        for part in (deadlock_match.group(1).split(",") if deadlock_match else [])
        if part.strip()
    ]

    cycle_match = _CYCLE_RE.search(user)
    cycle_edges = [
        line.strip()
        for line in (cycle_match.group("edges").splitlines() if cycle_match else [])
        if line.strip()
    ]

    source_match = _SOURCE_RE.search(user)
    witness_source = source_match.group("source") if source_match else ""
    return deadlocked_nodes, cycle_edges, witness_source


def _iteration_cap_patch(witness_source: str) -> dict[str, Any] | None:
    match = _UNBOUNDED_LOOP_RE.search(witness_source)
    if match is None:
        return None
    indent = match.group("indent")
    loop_indent = indent + "    "
    replacement = "\n".join(
        [
            f"{indent}max_iter = 16",
            f"{indent}iter_count = 0",
            f"{indent}while iter_count < max_iter:",
            f"{loop_indent}iter_count += 1",
        ]
    )
    line_start, line_end = _line_bounds(witness_source, match)
    return {
        "line_start": line_start,
        "line_end": line_end,
        "replacement": replacement,
    }


def _convergence_patch(witness_source: str) -> dict[str, Any] | None:
    match = _CONVERGED_RE.search(witness_source)
    if match is None:
        return None
    indent = match.group(1)
    replacement = (
        f"{indent}converged = bool(var_messages) and "
        "set(var_messages.keys()) == set(factor_messages.keys())"
    )
    line_start, line_end = _line_bounds(witness_source, match)
    return {
        "line_start": line_start,
        "line_end": line_end,
        "replacement": replacement,
    }


def _damping_patch(witness_source: str) -> dict[str, Any] | None:
    old_var = next(
        (name for name in ("old_msg", "prev_msg", "cached_msg") if name in witness_source),
        None,
    )
    if old_var is None:
        return None
    match = _NEW_MSG_RE.search(witness_source)
    if match is None:
        return None
    indent = match.group("indent")
    expr = match.group("expr").strip()
    replacement = "\n".join(
        [
            f"{indent}proposed_msg = {expr}",
            f"{indent}new_msg = 0.5 * proposed_msg + 0.5 * {old_var}",
        ]
    )
    line_start, line_end = _line_bounds(witness_source, match)
    return {
        "line_start": line_start,
        "line_end": line_end,
        "replacement": replacement,
    }


def _break_cycle(
    deadlock_nodes: list[str],
    cycle_edges: list[str],
    witness_source: str,
) -> tuple[list[dict[str, Any]], str] | None:
    if not witness_source or len(deadlock_nodes) > 5:
        return None
    if cycle_edges and len(cycle_edges) > 8:
        return None

    patch = _iteration_cap_patch(witness_source)
    if patch is not None:
        return [patch], "iteration_cap"

    if "memoization state" in witness_source.lower() or "converged = false" in witness_source.lower():
        patch = _convergence_patch(witness_source)
        if patch is not None:
            return [patch], "convergence_check"

    patch = _damping_patch(witness_source)
    if patch is not None:
        return [patch], "damping"
    return None


class DeterministicCycleBreaker:
    """Deterministic message-cycle breaker with LLM fallback."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "cycle_breaker_v1"

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        deadlocked_nodes, cycle_edges, witness_source = _parse_cycle_prompt(user)
        result = _break_cycle(deadlocked_nodes, cycle_edges, witness_source)
        if result is not None:
            patches, strategy = result
            self._last_completion_metadata = {
                "cycle_fix_source": "deterministic",
                "deadlock_node_count": len(deadlocked_nodes),
                "cycle_edge_count": len(cycle_edges),
                "cycle_fix_strategy": strategy,
            }
            self._last_error_metadata = {}
            return json.dumps(patches)

        self._last_completion_metadata = {"cycle_fix_source": "fallback"}
        self._last_error_metadata = {}
        return await self._fallback.complete(system, user)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
