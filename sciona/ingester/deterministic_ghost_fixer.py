"""Deterministic wrapper for the ingester_fix_ghost prompt."""

from __future__ import annotations

import json
import re
from typing import Any

_PROMPT_RE = re.compile(
    r"Ghost simulation error:\n"
    r"\s*Node:\s*(?P<node>.*)\n"
    r"\s*Function:\s*(?P<function>.*)\n"
    r"\s*Error:\s*(?P<error>.*)\n\n"
    r"Generated witnesses:\n```python\n(?P<source>.*?)\n```",
    re.DOTALL,
)
_DEF_RE = re.compile(r"^def\s+(?P<name>\w+)\((?P<params>.*?)\)\s*->\s*(?P<ret>[^:]+):\s*$")


def _parse_fix_ghost_prompt(user: str) -> tuple[str, str, str, str]:
    match = _PROMPT_RE.search(user)
    if match is None:
        return "", "", "", ""
    return (
        match.group("node").strip(),
        match.group("function").strip(),
        match.group("error").strip(),
        match.group("source"),
    )


def _function_span(lines: list[str], function_name: str) -> tuple[int, int] | None:
    start = None
    for idx, line in enumerate(lines):
        if line.startswith(f"def {function_name}("):
            start = idx
            break
    if start is None:
        return None
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("def "):
            end = idx
            break
    return start, end


def _primary_param(params: str) -> str | None:
    for raw in params.split(","):
        name = raw.split(":", 1)[0].strip()
        if not name or name == "state":
            continue
        return name
    return None


def _replacement_return(ret_type: str, params: str) -> str | None:
    ret = ret_type.strip()
    primary = _primary_param(params)
    if ret.startswith("tuple[") and ", AbstractSignal]" in ret:
        if primary:
            return f"    return {primary}, state"
        return "    return state, state"
    if ret in {"AbstractArray", "AbstractSignal", "AbstractDistribution", "AbstractScalar"}:
        if primary:
            return f"    return {primary}"
        return None
    return None


def _fix_none_return(lines: list[str], start: int, end: int, header_match: re.Match) -> bool:
    replacement = _replacement_return(header_match.group("ret"), header_match.group("params"))
    if replacement is None:
        return False
    changed = False
    for idx in range(start + 1, end):
        stripped = lines[idx].strip()
        if stripped in {"return None", "return None, state"}:
            lines[idx] = replacement
            changed = True
    return changed


def _fix_type_error(lines: list[str], start: int, end: int, header_match: re.Match) -> bool:
    """Fix TypeError patterns — typically wrong argument types or missing conversions."""
    params = header_match.group("params")
    primary = _primary_param(params)
    if not primary:
        return False
    changed = False
    for idx in range(start + 1, end):
        stripped = lines[idx].strip()
        # Replace bare raise TypeError with pass-through
        if stripped.startswith("raise TypeError"):
            lines[idx] = f"    return {primary}"
            changed = True
    return changed


def _fix_key_error(lines: list[str], start: int, end: int, error_message: str) -> bool:
    """Fix KeyError — replace dict lookups that raise with .get() defaults."""
    key_match = re.search(r"KeyError:\s*['\"]?(\w+)['\"]?", error_message)
    if not key_match:
        return False
    key = key_match.group(1)
    changed = False
    for idx in range(start + 1, end):
        # Replace dict[key] with dict.get(key, None) for the offending key
        if f'["{key}"]' in lines[idx] or f"['{key}']" in lines[idx]:
            lines[idx] = re.sub(
                rf'\[(["\']){re.escape(key)}\1\]',
                f'.get("{key}", None)',
                lines[idx],
            )
            changed = True
    return changed


def _fix_attribute_error(lines: list[str], start: int, end: int, error_message: str) -> bool:
    """Fix AttributeError — guard attribute access with getattr."""
    attr_match = re.search(r"has no attribute ['\"](\w+)['\"]", error_message)
    if not attr_match:
        return False
    attr = attr_match.group(1)
    changed = False
    for idx in range(start + 1, end):
        pattern = rf"(\w+)\.{re.escape(attr)}\b"
        if re.search(pattern, lines[idx]):
            lines[idx] = re.sub(
                pattern,
                rf'getattr(\1, "{attr}", None)',
                lines[idx],
            )
            changed = True
    return changed


def _rewrite_stub(function_name: str, error_message: str, witness_source: str) -> str | None:
    lines = witness_source.splitlines()
    span = _function_span(lines, function_name)
    if span is None:
        return None
    start, end = span
    header_match = _DEF_RE.match(lines[start].strip())
    if header_match is None:
        return None

    error_lower = error_message.lower()
    changed = False

    if "none" in error_lower or "shape" in error_lower:
        changed = _fix_none_return(lines, start, end, header_match)
    elif "typeerror" in error_lower:
        changed = _fix_type_error(lines, start, end, header_match)
    elif "keyerror" in error_lower:
        changed = _fix_key_error(lines, start, end, error_message)
    elif "attributeerror" in error_lower or "has no attribute" in error_lower:
        changed = _fix_attribute_error(lines, start, end, error_message)
    else:
        # Fallback: try None-return fix for any unrecognized error
        changed = _fix_none_return(lines, start, end, header_match)

    if not changed:
        return None
    return "\n".join(lines)


class DeterministicGhostFixer:
    """Deterministic ingester ghost fixer with LLM fallback."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "ghost_fixer_v1"

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        error_node, error_function, error_message, witness_source = _parse_fix_ghost_prompt(user)
        replacement = _rewrite_stub(error_function, error_message, witness_source)
        if replacement is not None:
            self._last_completion_metadata = {
                "ghost_fix_source": "deterministic",
                "ghost_fix_function": error_function,
                "ghost_fix_node": error_node,
            }
            self._last_error_metadata = {}
            return json.dumps(
                [
                    {
                        "witness_name": error_function,
                        "fix_description": "Replace None-returning witness stub with pass-through abstract value",
                        "replacement": replacement,
                    }
                ]
            )

        self._last_completion_metadata = {"ghost_fix_source": "fallback"}
        self._last_error_metadata = {}
        return await self._fallback.complete(system, user)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
