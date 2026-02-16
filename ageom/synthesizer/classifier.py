"""Regex-based compiler error classifier for deterministic pre-filtering."""

from __future__ import annotations

import re
from enum import Enum

from ageom.judge.models import CompilerFeedback


class ErrorCategory(str, Enum):
    """Categories of compiler errors."""

    MISSING_IMPORT = "missing_import"
    TYPE_MISMATCH = "type_mismatch"
    UNSOLVED_GOAL = "unsolved_goal"
    UNIVERSE_MISMATCH = "universe_mismatch"
    SYNTAX = "syntax"
    UNKNOWN = "unknown"


# Patterns checked in order — first match wins
_PATTERNS: list[tuple[re.Pattern[str], ErrorCategory]] = [
    (re.compile(r"unknown identifier", re.IGNORECASE), ErrorCategory.MISSING_IMPORT),
    (re.compile(r"unknown namespace", re.IGNORECASE), ErrorCategory.MISSING_IMPORT),
    (re.compile(r"unknown constant", re.IGNORECASE), ErrorCategory.MISSING_IMPORT),
    (re.compile(r"type mismatch", re.IGNORECASE), ErrorCategory.TYPE_MISMATCH),
    (re.compile(r"expected\b.*\bgot\b", re.IGNORECASE), ErrorCategory.TYPE_MISMATCH),
    (re.compile(r"has type.*but is expected", re.IGNORECASE), ErrorCategory.TYPE_MISMATCH),
    (re.compile(r"unsolved goals", re.IGNORECASE), ErrorCategory.UNSOLVED_GOAL),
    (re.compile(r"^⊢", re.MULTILINE), ErrorCategory.UNSOLVED_GOAL),
    (re.compile(r"universe level", re.IGNORECASE), ErrorCategory.UNIVERSE_MISMATCH),
    (re.compile(r"universe inconsistency", re.IGNORECASE), ErrorCategory.UNIVERSE_MISMATCH),
    (re.compile(r"expected .*(token|command|declaration)", re.IGNORECASE), ErrorCategory.SYNTAX),
    (re.compile(r"parse error", re.IGNORECASE), ErrorCategory.SYNTAX),
    (re.compile(r"unexpected token", re.IGNORECASE), ErrorCategory.SYNTAX),
]


def classify_error(error_line: str) -> ErrorCategory:
    """Classify a single error line into an ErrorCategory."""
    for pattern, category in _PATTERNS:
        if pattern.search(error_line):
            return category
    return ErrorCategory.UNKNOWN


def classify_feedback(
    feedback: CompilerFeedback,
) -> list[tuple[ErrorCategory, str]]:
    """Classify all errors and remaining goals in a CompilerFeedback."""
    results: list[tuple[ErrorCategory, str]] = []
    for error in feedback.errors:
        results.append((classify_error(error), error))
    for goal in feedback.goals_remaining:
        results.append((ErrorCategory.UNSOLVED_GOAL, goal))
    return results


# Regex to extract identifier from "unknown identifier 'Foo.bar'" style messages
_IMPORT_IDENT_RE = re.compile(r"unknown (?:identifier|constant) '?@?([A-Za-z_][\w.]*)'?")
_NAMESPACE_RE = re.compile(r"unknown namespace '?([A-Za-z_][\w.]*)'?")


def suggest_deterministic_fix(
    category: ErrorCategory, error_text: str
) -> str | None:
    """Return a deterministic fix string if one exists, else None.

    Currently handles:
    - MISSING_IMPORT: suggests `open {namespace}` or `import Mathlib.{module}`
    """
    if category != ErrorCategory.MISSING_IMPORT:
        return None

    # Try to extract the identifier
    m = _IMPORT_IDENT_RE.search(error_text)
    if m:
        ident = m.group(1)
        # If it has a namespace prefix, suggest opening that namespace
        parts = ident.rsplit(".", 1)
        if len(parts) == 2:
            return f"open {parts[0]}"
        return f"import Mathlib  -- for {ident}"

    m = _NAMESPACE_RE.search(error_text)
    if m:
        ns = m.group(1)
        return f"open {ns}"

    return None
