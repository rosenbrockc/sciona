"""Ghost simulation failure classification and deterministic fixing.

Re-exports from the ingester's deterministic ghost fixer and verification
classifier.
"""

from __future__ import annotations

from typing import Any


def classify_ghost_failure(
    report: dict[str, Any],
    witness_source: str,
) -> dict[str, Any]:
    """Classify a ghost simulation failure into repairable categories.

    Args:
        report: The ghost simulation failure report dict.
        witness_source: The witness source code that was simulated.

    Returns:
        Classification dict with keys: stage, reason_code, repairable,
        route, summary, issues.
    """
    from sciona.ingester.verification_classifier import (
        classify_ghost_failure as _classify,
    )

    return _classify(report, witness_source)


def build_ghost_fixes(
    error_node: str,
    error_function: str,
    error_message: str,
    witness_source: str,
) -> list[dict[str, Any]] | None:
    """Attempt deterministic fixes for ghost simulation failures.

    Handles: None-return errors, TypeError, KeyError, and
    AttributeError patterns. Returns None if no fix applies.

    Args:
        error_node: The CDG node that failed.
        error_function: The witness function that failed.
        error_message: The error message from simulation.
        witness_source: The full witness source code.

    Returns:
        List of fix dicts with keys {witness_name, fix_description,
        replacement, error_node}, or None.
    """
    from sciona.ingester.deterministic_ghost_fixer import (
        build_deterministic_ghost_fixes,
    )

    return build_deterministic_ghost_fixes(
        error_node, error_function, error_message, witness_source
    )


__all__ = ["classify_ghost_failure", "build_ghost_fixes"]
