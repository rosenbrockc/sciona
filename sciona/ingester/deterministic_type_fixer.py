"""Deterministic wrapper for the ingester_fix_type prompt."""

from __future__ import annotations

import json
import re
from typing import Any

from sciona.synthesizer.classifier import (
    ErrorCategory,
    classify_error,
    suggest_deterministic_fix,
)

_ERRORS_RE = re.compile(
    r"mypy errors:\n(?P<errors>.*?)\n\nGenerated source:\n```python\n(?P<source>.*?)\n```",
    re.DOTALL,
)
_LINE_NUMBER_RE = re.compile(r"^[^:\n]+:(\d+):\s+(?:error|note):", re.IGNORECASE)
_RETURN_RE = re.compile(r"^(?P<indent>\s*)return\s+(?P<expr>.+?)\s*$")
_WRAP_RETURN_RE = re.compile(r"^Wrap return value:\s*(?P<template>.+?)\s*$")
_UNDEFINED_NAME_RE = re.compile(r'Name "(\w+)" is not defined', re.IGNORECASE)


def _parse_fix_type_prompt(user: str) -> tuple[str, str]:
    match = _ERRORS_RE.search(user)
    if match is None:
        return "", ""
    return match.group("errors").strip(), match.group("source")


def _extract_line_number(error_line: str) -> int | None:
    match = _LINE_NUMBER_RE.match(error_line.strip())
    if match is None:
        return None
    try:
        line_no = int(match.group(1))
    except ValueError:
        return None
    return line_no if line_no > 0 else None


def _import_patch(source_code: str, import_stmt: str) -> dict[str, Any] | None:
    stripped = import_stmt.strip()
    if not stripped:
        return None
    lines = source_code.splitlines()
    if not lines:
        return {
            "line_start": 1,
            "line_end": 1,
            "replacement": stripped,
        }
    if stripped in {line.strip() for line in lines}:
        return None

    insert_after = 1
    for idx, line in enumerate(lines, start=1):
        token = line.strip()
        if not token:
            continue
        if token.startswith(("import ", "from ")):
            insert_after = idx
            continue
        break
    replacement = (
        f"{lines[insert_after - 1]}\n{stripped}" if insert_after > 0 else stripped
    )
    return {
        "line_start": insert_after,
        "line_end": insert_after,
        "replacement": replacement,
    }


def _return_patch(
    source_code: str,
    *,
    error_line: str,
    fix_hint: str,
) -> dict[str, Any] | None:
    line_no = _extract_line_number(error_line)
    if line_no is None:
        return None
    lines = source_code.splitlines()
    if line_no > len(lines):
        return None
    line = lines[line_no - 1]
    match = _RETURN_RE.match(line)
    if match is None:
        return None
    fix_match = _WRAP_RETURN_RE.match(fix_hint.strip())
    if fix_match is None:
        return None
    expr = match.group("expr").strip()
    template = fix_match.group("template").strip()
    wrapped = template.replace("result", expr)
    replacement = f"{match.group('indent')}return {wrapped}"
    return {
        "line_start": line_no,
        "line_end": line_no,
        "replacement": replacement,
    }


def _build_patch(source_code: str, error_line: str) -> dict[str, Any] | None:
    category = classify_error(error_line)
    if category == ErrorCategory.UNKNOWN and _UNDEFINED_NAME_RE.search(error_line):
        category = ErrorCategory.MISSING_IMPORT
    if category == ErrorCategory.UNKNOWN:
        return None
    fix_hint = suggest_deterministic_fix(category, error_line)
    if fix_hint is None:
        return None
    if category == ErrorCategory.MISSING_IMPORT:
        return _import_patch(source_code, fix_hint)
    if category == ErrorCategory.TYPE_MISMATCH:
        return _return_patch(source_code, error_line=error_line, fix_hint=fix_hint)
    return None


class DeterministicTypeFixer:
    """Deterministic ingester type fixer with LLM fallback."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "type_fixer_v1"

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        mypy_errors, source_code = _parse_fix_type_prompt(user)
        error_lines = [
            line.strip()
            for line in mypy_errors.splitlines()
            if line.strip() and "error:" in line.lower()
        ]
        patches: list[dict[str, Any]] = []
        for error_line in error_lines:
            patch = _build_patch(source_code, error_line)
            if patch is None:
                self._last_completion_metadata = {"type_fix_source": "fallback"}
                self._last_error_metadata = {}
                return await self._fallback.complete(system, user)
            patches.append(patch)

        if not patches:
            self._last_completion_metadata = {"type_fix_source": "fallback"}
            self._last_error_metadata = {}
            return await self._fallback.complete(system, user)

        deduped: dict[tuple[int, int, str], dict[str, Any]] = {}
        for patch in patches:
            key = (
                int(patch.get("line_start", 0) or 0),
                int(patch.get("line_end", 0) or 0),
                str(patch.get("replacement", "") or ""),
            )
            deduped[key] = patch
        ordered = sorted(
            deduped.values(),
            key=lambda patch: int(patch.get("line_start", 0) or 0),
            reverse=True,
        )
        self._last_completion_metadata = {
            "type_fix_source": "deterministic",
            "patch_count": len(ordered),
        }
        self._last_error_metadata = {}
        return json.dumps(ordered)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
