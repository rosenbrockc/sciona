"""Baseline measurement for generalization coverage.

Runs strategy classification, deterministic split matching, and phrase rule
matching on domain-agnostic goal descriptions.  Prints a coverage report
showing which goals hit deterministic paths vs. fall through.

This test asserts nothing — it measures baseline coverage before
generalization work changes the numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ageom.architect.models import ConceptType
from ageom.architect.strategy_classifier import StrategyClassifier
from ageom.orchestrator import _deterministic_split_subnodes
from ageom.types import MatchFailureReport, PDGNode

FIXTURES = Path(__file__).parent / "fixtures" / "generalization_goals.json"


def _load_goals() -> list[dict]:
    return json.loads(FIXTURES.read_text())


@pytest.fixture(scope="module")
def goals() -> list[dict]:
    return _load_goals()


def _classifier() -> StrategyClassifier:
    fallback = AsyncMock()
    fallback.complete = AsyncMock(return_value='{"paradigm": "custom", "rationale": "fallback"}')
    return StrategyClassifier(fallback)


def _try_split(goal: str) -> list[dict[str, str]] | None:
    """Attempt a deterministic split using a minimal failure report."""
    from ageom.architect.models import AlgorithmicNode, NodeStatus

    node = AlgorithmicNode(
        node_id="test",
        name="test",
        description=goal,
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
    )
    failure = MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="test",
            statement=goal,
            informal_desc=goal,
        ),
        error_summaries=["no match found"],
    )
    return _deterministic_split_subnodes(failure, node)


def test_generalization_baseline(goals: list[dict]) -> None:
    """Print a coverage report — no assertions."""
    classifier = _classifier()
    results: list[dict] = []

    for g in goals:
        goal_text = g["goal"]
        expected_paradigm = g.get("expected_paradigm")

        # Strategy classification
        decision = classifier.classify(goal_text)
        classified = decision is not None
        paradigm_match = (
            decision is not None
            and expected_paradigm is not None
            and decision[0].value == expected_paradigm
        )

        # Deterministic split
        split = _try_split(goal_text)
        has_split = split is not None

        results.append({
            "id": g["id"],
            "classified": classified,
            "paradigm_match": paradigm_match,
            "has_split": has_split,
            "split_count": len(split) if split else 0,
        })

    # Print coverage report
    total = len(results)
    classified_count = sum(1 for r in results if r["classified"])
    paradigm_match_count = sum(1 for r in results if r["paradigm_match"])
    split_count = sum(1 for r in results if r["has_split"])

    print(f"\n{'='*60}")
    print("GENERALIZATION BASELINE COVERAGE REPORT")
    print(f"{'='*60}")
    print(f"Goals tested:              {total}")
    print(f"Strategy classified:       {classified_count}/{total} ({100*classified_count/total:.0f}%)")
    print(f"Paradigm correct:          {paradigm_match_count}/{total} ({100*paradigm_match_count/total:.0f}%)")
    print(f"Deterministic split found: {split_count}/{total} ({100*split_count/total:.0f}%)")
    print(f"{'='*60}")
    for r in results:
        status = "✓" if r["classified"] else "·"
        split_status = f"split={r['split_count']}" if r["has_split"] else "no split"
        paradigm_status = "paradigm ✓" if r["paradigm_match"] else "paradigm ·"
        print(f"  {status} {r['id']:35s} {paradigm_status:15s} {split_status}")
    print(f"{'='*60}\n")
