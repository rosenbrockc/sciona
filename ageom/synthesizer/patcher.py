"""Patch application and source analysis utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Patch:
    """A line-range replacement in source code."""

    line_start: int  # 1-indexed
    line_end: int  # 1-indexed, inclusive
    replacement: str
    description: str = ""


def apply_patches(source: str, patches: list[Patch]) -> str:
    """Apply patches to source code, bottom-up to preserve line numbers.

    Raises ValueError if any patches overlap.
    """
    if not patches:
        return source

    # Sort by line_start descending so we apply bottom-up
    sorted_patches = sorted(patches, key=lambda p: p.line_start, reverse=True)

    # Check for overlaps
    for i in range(len(sorted_patches) - 1):
        higher = sorted_patches[i]
        lower = sorted_patches[i + 1]
        if lower.line_end >= higher.line_start:
            raise ValueError(
                f"Overlapping patches: lines {lower.line_start}-{lower.line_end} "
                f"and {higher.line_start}-{higher.line_end}"
            )

    lines = source.splitlines(keepends=True)

    for patch in sorted_patches:
        start = patch.line_start - 1  # 0-indexed
        end = patch.line_end  # exclusive (line_end is inclusive, so +1-1=end)

        # Ensure replacement ends with newline if the original section did
        replacement_lines = patch.replacement
        if not replacement_lines.endswith("\n") and end <= len(lines):
            replacement_lines += "\n"

        lines[start:end] = [replacement_lines]

    return "".join(lines)


def find_sorry_locations(
    source: str, prover: str
) -> list[tuple[int, str]]:
    """Find sorry/Admitted placeholders and return (line_number, context).

    Returns a list of (1-indexed line number, surrounding context string).
    """
    if prover == "coq":
        pattern = re.compile(r"\bAdmitted\.\s*$", re.MULTILINE)
    else:
        pattern = re.compile(r"\bsorry\b")

    results: list[tuple[int, str]] = []
    lines = source.splitlines()

    for i, line in enumerate(lines):
        if pattern.search(line):
            context = extract_error_context(source, i + 1, radius=3)
            results.append((i + 1, context))

    return results


def extract_error_context(
    source: str, error_line: int, radius: int = 3
) -> str:
    """Extract lines around an error for LLM context.

    Args:
        source: Full source code.
        error_line: 1-indexed line number of the error.
        radius: Number of lines before and after to include.

    Returns:
        A string with line numbers and content.
    """
    lines = source.splitlines()
    start = max(0, error_line - 1 - radius)
    end = min(len(lines), error_line + radius)

    context_lines: list[str] = []
    for i in range(start, end):
        marker = " >> " if i == error_line - 1 else "    "
        context_lines.append(f"{marker}{i + 1:4d} | {lines[i]}")

    return "\n".join(context_lines)
