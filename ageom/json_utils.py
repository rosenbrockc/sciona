"""Robust JSON extraction from LLM responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Matches ```json ... ``` or ``` ... ``` fenced blocks
_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Extract and parse JSON from an LLM response that may contain fences or prose.

    Tries, in order:
    1. Direct ``json.loads`` on the full text (fast path).
    2. Strip markdown fences (````` ```json ... ``` `````) and parse contents.
    3. Find the outermost ``{...}`` or ``[...]`` substring and parse it.

    Raises ``json.JSONDecodeError`` if no valid JSON can be found.
    On failure the first 500 chars of *text* are logged at DEBUG level.
    """
    stripped = text.strip()

    # 1. Fast path: entire response is valid JSON
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. Try markdown-fenced blocks (pick the first one that parses)
    for m in _FENCE_RE.finditer(stripped):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue

    # 3. Find the outermost { ... } or [ ... ] by scanning for balanced braces
    result = _find_balanced(stripped, "{", "}") or _find_balanced(stripped, "[", "]")
    if result is not None:
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            pass

    # All strategies failed — log and re-raise
    logger.debug(
        "extract_json failed; raw response (first 500 chars): %.500s", stripped
    )
    raise json.JSONDecodeError(
        "No valid JSON found in LLM response", stripped, 0
    )


def _find_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced ``open_ch...close_ch`` substring, or *None*."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
