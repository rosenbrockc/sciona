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
    (
        re.compile(r"has type.*but is expected", re.IGNORECASE),
        ErrorCategory.TYPE_MISMATCH,
    ),
    (re.compile(r"unsolved goals", re.IGNORECASE), ErrorCategory.UNSOLVED_GOAL),
    (re.compile(r"^⊢", re.MULTILINE), ErrorCategory.UNSOLVED_GOAL),
    (re.compile(r"universe level", re.IGNORECASE), ErrorCategory.UNIVERSE_MISMATCH),
    (
        re.compile(r"universe inconsistency", re.IGNORECASE),
        ErrorCategory.UNIVERSE_MISMATCH,
    ),
    (
        re.compile(r"expected .*(token|command|declaration)", re.IGNORECASE),
        ErrorCategory.SYNTAX,
    ),
    (re.compile(r"parse error", re.IGNORECASE), ErrorCategory.SYNTAX),
    (re.compile(r"unexpected token", re.IGNORECASE), ErrorCategory.SYNTAX),
    # Python / mypy patterns
    (re.compile(r"No module named", re.IGNORECASE), ErrorCategory.MISSING_IMPORT),
    (
        re.compile(
            r"Cannot find implementation or library stub for module named",
            re.IGNORECASE,
        ),
        ErrorCategory.MISSING_IMPORT,
    ),
    (re.compile(r"Incompatible types", re.IGNORECASE), ErrorCategory.TYPE_MISMATCH),
    (
        re.compile(r"Incompatible return value type", re.IGNORECASE),
        ErrorCategory.TYPE_MISMATCH,
    ),
    (
        re.compile(r"Argument \d+ .* has incompatible type", re.IGNORECASE),
        ErrorCategory.TYPE_MISMATCH,
    ),
    (re.compile(r"invalid syntax", re.IGNORECASE), ErrorCategory.SYNTAX),
    (re.compile(r"SyntaxError", re.IGNORECASE), ErrorCategory.SYNTAX),
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
_IMPORT_IDENT_RE = re.compile(
    r"unknown (?:identifier|constant) '?@?([A-Za-z_][\w.]*)'?"
)
_NAMESPACE_RE = re.compile(r"unknown namespace '?([A-Za-z_][\w.]*)'?")
_PYTHON_MODULE_RE = re.compile(r"(?:No module named|module named) '([A-Za-z_][\w.]*)'?")


_TYPE_EXPECTED_GOT_RE = re.compile(
    r"expected\s+['\"]?(\S+)['\"]?.*got\s+['\"]?(\S+)", re.IGNORECASE
)
_INCOMPATIBLE_RETURN_RE = re.compile(
    r'Incompatible return value type \(got "([^"]+)", expected "([^"]+)"\)', re.IGNORECASE
)
_ARG_INCOMPATIBLE_RE = re.compile(
    r'Argument \d+ .* has incompatible type "([^"]+)".*expected "([^"]+)"', re.IGNORECASE
)
_UNDEFINED_NAME_RE = re.compile(r'Name "(\w+)" is not defined', re.IGNORECASE)
_ATTR_MISSING_RE = re.compile(r'"(\w+)" has no attribute "(\w+)"', re.IGNORECASE)

# Known coercions: (from_type, to_type) → fix expression
_PYTHON_COERCIONS: dict[tuple[str, str], str] = {
    ("int", "float"): "float({expr})",
    ("float", "int"): "int({expr})",
    ("str", "int"): "int({expr})",
    ("str", "float"): "float({expr})",
    ("list", "ndarray"): "np.array({expr})",
    ("ndarray", "list"): "{expr}.tolist()",
    ("List[float]", "ndarray"): "np.array({expr})",
    ("List[int]", "ndarray"): "np.array({expr})",
    ("tuple", "list"): "list({expr})",
    ("list", "tuple"): "tuple({expr})",
}

# Lean4 coercions
_LEAN_COERCIONS: dict[tuple[str, str], str] = {
    ("Nat", "Int"): "Int.ofNat",
    ("Int", "Nat"): "Int.toNat",
    ("Fin n", "Nat"): "Fin.val",
    ("List α", "Array α"): "List.toArray",
    ("Array α", "List α"): "Array.toList",
}

# Common Python import suggestions for undefined names
_COMMON_IMPORTS: dict[str, str] = {
    "np": "import numpy as np",
    "numpy": "import numpy",
    "pd": "import pandas as pd",
    "plt": "import matplotlib.pyplot as plt",
    "scipy": "import scipy",
    "signal": "from scipy import signal",
    "linalg": "from scipy import linalg",
    "ndarray": "from numpy import ndarray",
    "Optional": "from typing import Optional",
    "List": "from typing import List",
    "Dict": "from typing import Dict",
    "Tuple": "from typing import Tuple",
    "Callable": "from typing import Callable",
    "Any": "from typing import Any",
}


def suggest_deterministic_fix(category: ErrorCategory, error_text: str) -> str | None:
    """Return a deterministic fix string if one exists, else None.

    Handles:
    - MISSING_IMPORT: suggests `open {namespace}`, `import module`, or common imports
    - TYPE_MISMATCH: suggests coercions for known type pairs
    - SYNTAX: suggests common syntax corrections
    """
    if category == ErrorCategory.MISSING_IMPORT:
        return _fix_missing_import(error_text)
    if category == ErrorCategory.TYPE_MISMATCH:
        return _fix_type_mismatch(error_text)
    if category == ErrorCategory.SYNTAX:
        return _fix_syntax(error_text)
    return None


def _fix_missing_import(error_text: str) -> str | None:
    # Lean: unknown identifier/constant
    m = _IMPORT_IDENT_RE.search(error_text)
    if m:
        ident = m.group(1)
        parts = ident.rsplit(".", 1)
        if len(parts) == 2:
            return f"open {parts[0]}"
        return f"import Mathlib  -- for {ident}"

    m = _NAMESPACE_RE.search(error_text)
    if m:
        ns = m.group(1)
        return f"open {ns}"

    # Python: "No module named 'foo'"
    m = _PYTHON_MODULE_RE.search(error_text)
    if m:
        module = m.group(1)
        return f"import {module}"

    # Python: Name "X" is not defined
    m = _UNDEFINED_NAME_RE.search(error_text)
    if m:
        name = m.group(1)
        if name in _COMMON_IMPORTS:
            return _COMMON_IMPORTS[name]
        return f"# Undefined name '{name}' — add the appropriate import"

    return None


def _fix_type_mismatch(error_text: str) -> str | None:
    # Python: Incompatible return value type
    m = _INCOMPATIBLE_RETURN_RE.search(error_text)
    if m:
        got, expected = m.group(1), m.group(2)
        coercion = _lookup_coercion(got, expected)
        if coercion:
            return f"Wrap return value: {coercion.format(expr='result')}"
        return f"Cast return value from '{got}' to '{expected}'"

    # Python: Argument N has incompatible type
    m = _ARG_INCOMPATIBLE_RE.search(error_text)
    if m:
        got, expected = m.group(1), m.group(2)
        coercion = _lookup_coercion(got, expected)
        if coercion:
            return f"Wrap argument: {coercion.format(expr='arg')}"
        return f"Cast argument from '{got}' to '{expected}'"

    # General expected/got pattern (Lean or Python)
    m = _TYPE_EXPECTED_GOT_RE.search(error_text)
    if m:
        expected, got = m.group(1), m.group(2)
        # Check Lean coercions
        for (from_t, to_t), fix in _LEAN_COERCIONS.items():
            if got.startswith(from_t.split()[0]) and expected.startswith(to_t.split()[0]):
                return f"Apply coercion: {fix}"
        return f"Type coercion needed: '{got}' → '{expected}'"

    # Missing attribute
    m = _ATTR_MISSING_RE.search(error_text)
    if m:
        type_name, attr = m.group(1), m.group(2)
        return f"'{type_name}' has no attribute '{attr}' — check method name or cast to correct type"

    return None


def _fix_syntax(error_text: str) -> str | None:
    lower = error_text.lower()

    # Python: invalid syntax near annotation
    if "perhaps you forgot a comma" in lower:
        return "Check for invalid type annotation syntax — ensure variable annotations use valid types"

    # Python: unexpected indent
    if "unexpected indent" in lower or "indentationerror" in lower:
        return "Fix indentation — ensure consistent use of spaces (not tabs)"

    # Lean: expected token
    if "expected token" in lower:
        return "Check for missing delimiters (parentheses, braces, or keywords)"

    # General parse error
    if "parse error" in lower or "unexpected token" in lower:
        return "Review syntax near the error location — check for missing or extra delimiters"

    return None


def _lookup_coercion(from_type: str, to_type: str) -> str | None:
    """Look up a known coercion between Python types."""
    # Exact match
    key = (from_type, to_type)
    if key in _PYTHON_COERCIONS:
        return _PYTHON_COERCIONS[key]
    # Strip generic params for partial matches
    from_base = from_type.split("[")[0].strip()
    to_base = to_type.split("[")[0].strip()
    key = (from_base, to_base)
    if key in _PYTHON_COERCIONS:
        return _PYTHON_COERCIONS[key]
    return None
