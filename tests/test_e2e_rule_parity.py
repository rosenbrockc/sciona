"""End-to-end tests verifying JSON-loaded rules produce identical behavior to old hardcoded rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from ageom.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from ageom.architect.strategy_classifier import (
    StrategyClassifier,
    _load_phrase_rules,
)
from ageom.hunter.query_reformulator import (
    _load_query_rules,
    _match_phrase_rule,
)
from ageom.orchestrator import (
    _deterministic_split_subnodes,
    _load_split_patterns,
    _pattern_matches,
)
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchFailureReport,
    PDGNode,
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "ageom" / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_failure(
    statement: str,
    description: str = "",
    error_summaries: list[str] | None = None,
    candidate_names: list[str] | None = None,
) -> MatchFailureReport:
    """Build a lightweight MatchFailureReport for deterministic split tests."""
    candidates = [
        CandidateMatch(
            declaration=Declaration(name=name, type_signature=""),
            score=0.5,
            retrieval_method="test",
        )
        for name in (candidate_names or [])
    ]
    return MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="test_node",
            statement=statement,
            informal_desc=description,
        ),
        best_candidates=candidates,
        error_summaries=error_summaries or [],
    )


def _make_node(description: str) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id="test_node",
        name="Test Node",
        description=description,
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        depth=0,
    )


class _FallbackLLM:
    """Dummy LLM that should never be called in deterministic tests."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return '{"paradigm":"custom","rationale":"fallback"}'

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


# ---------------------------------------------------------------------------
# 1. test_split_patterns_json_loads_successfully
# ---------------------------------------------------------------------------


def test_split_patterns_json_loads_successfully():
    """_load_split_patterns() returns >= 16 patterns."""
    patterns = _load_split_patterns()
    assert len(patterns) >= 16, f"Expected >= 16 patterns, got {len(patterns)}"


# ---------------------------------------------------------------------------
# 2. test_split_patterns_json_has_required_fields
# ---------------------------------------------------------------------------


def test_split_patterns_json_has_required_fields():
    """Each pattern has name, conditions, sub_nodes with name and description."""
    patterns = _load_split_patterns()
    for i, pattern in enumerate(patterns):
        assert "name" in pattern, f"Pattern {i} missing 'name'"
        assert "conditions" in pattern, f"Pattern {i} ({pattern.get('name')}) missing 'conditions'"
        assert "sub_nodes" in pattern, f"Pattern {i} ({pattern.get('name')}) missing 'sub_nodes'"
        for j, sub in enumerate(pattern["sub_nodes"]):
            assert "name" in sub, f"Pattern {pattern['name']} sub_node {j} missing 'name'"
            assert "description" in sub, f"Pattern {pattern['name']} sub_node {j} missing 'description'"


# ---------------------------------------------------------------------------
# 3. test_deterministic_split_matches_expected_patterns (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("statement", "description", "expected_count"),
    [
        ("ecg bandpass filter", "Design and apply a stable bandpass filter to ECG samples", 2),
        ("shortest path dijkstra", "Compute shortest path in weighted graph", 2),
        ("cholesky symmetric positive definite", "Solve an SPD linear system via Cholesky", 2),
        ("fft fourier spectral", "Compute the FFT of the input signal", 2),
        ("sort order elements", "Sort elements into ascending order", 2),
        ("edit distance levenshtein", "Compute edit distance between two strings", 2),
        ("detect peaks compute rate", "Detect peaks and compute rate", 2),
        ("interpolation spline fit", "Fit a spline interpolation to data", 2),
        ("cluster k-means assign", "Cluster data using k-means", 2),
        ("posterior bayesian inference sampling", "Bayesian inference with posterior sampling", 3),
        ("longest common subsequence dynamic", "Compute LCS using DP", 2),
    ],
)
def test_deterministic_split_matches_expected_patterns(
    statement: str, description: str, expected_count: int
):
    failure = _make_failure(statement=statement, description=description)
    node = _make_node(description)
    result = _deterministic_split_subnodes(failure, node)
    assert result is not None, f"No split pattern matched for: {statement}"
    assert len(result) == expected_count, (
        f"Expected {expected_count} sub-nodes for '{statement}', got {len(result)}"
    )


# ---------------------------------------------------------------------------
# 4. test_phrase_rules_json_loads_successfully
# ---------------------------------------------------------------------------


def test_phrase_rules_json_loads_successfully():
    """_load_phrase_rules() returns non-empty tuple of rules."""
    phrase_rules, conjunction_rules = _load_phrase_rules()
    assert len(phrase_rules) > 0, "phrase_rules should not be empty"
    assert len(conjunction_rules) > 0, "conjunction_rules should not be empty"


# ---------------------------------------------------------------------------
# 5. test_strategy_classifier_with_json_rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("goal", "expected_concept"),
    [
        ("Design and apply a stable bandpass filter to ECG samples.", ConceptType.SIGNAL_FILTER),
        ("Compute shortest path distances from a source node in a weighted graph.", ConceptType.GRAPH_OPTIMIZATION),
        ("Compute the longest common subsequence of two strings.", ConceptType.DYNAMIC_PROGRAMMING),
    ],
)
def test_strategy_classifier_with_json_rules(goal: str, expected_concept: ConceptType):
    """Classify ECG/Dijkstra/LCS goals with explicit rules_path, assert correct ConceptType."""
    rules_path = _DATA_DIR / "phrase_rules.json"
    fallback = _FallbackLLM()
    classifier = StrategyClassifier(fallback, rules_path=rules_path)
    result = classifier.classify(goal)
    assert result is not None, f"Classifier returned None for: {goal}"
    concept, _confidence, _rationale, _variant = result
    assert concept == expected_concept, (
        f"Expected {expected_concept.value} for '{goal}', got {concept.value}"
    )


# ---------------------------------------------------------------------------
# 6. test_query_rules_json_loads_successfully
# ---------------------------------------------------------------------------


def test_query_rules_json_loads_successfully():
    """_load_query_rules() returns non-empty anchors and rules."""
    anchors, rules = _load_query_rules()
    assert len(anchors) > 0, "domain_anchors should not be empty"
    assert len(rules) > 0, "phrase_rules should not be empty"


# ---------------------------------------------------------------------------
# 7. test_query_phrase_rules_return_expected_queries
# ---------------------------------------------------------------------------


def test_query_phrase_rules_return_expected_queries():
    """Check that an ECG-related input matches the expected rule."""
    _anchors, rules = _load_query_rules()
    ecg_rule = next((r for r in rules if r["name"] == "ecg_bandpass_filter"), None)
    assert ecg_rule is not None, "ecg_bandpass_filter rule not found in query_rules.json"

    # The rule has conditions {"all": ["ecg", "bandpass", "filter"]}
    assert _match_phrase_rule(ecg_rule, "ecg bandpass filter design")
    assert not _match_phrase_rule(ecg_rule, "lowpass filter design")


# ---------------------------------------------------------------------------
# 8. test_split_pattern_conditions_match_expected_strings
# ---------------------------------------------------------------------------


def test_split_pattern_conditions_match_expected_strings():
    """For each pattern in JSON, build a test string from its conditions, verify _pattern_matches returns True."""
    patterns = _load_split_patterns()
    assert len(patterns) > 0

    for pattern in patterns:
        conditions = pattern["conditions"]
        # Build a synthetic test string that should satisfy all conditions
        terms: list[str] = []

        if "all" in conditions:
            terms.extend(conditions["all"])

        if "any" in conditions:
            any_val = conditions["any"]
            if any_val and isinstance(any_val[0], list):
                # nested lists: pick first group
                terms.extend(any_val[0])
            else:
                # pick the first term
                terms.append(any_val[0])

        if "any_combo" in conditions:
            # pick the first combo group
            terms.extend(conditions["any_combo"][0])

        if "any_padded" in conditions:
            # add padded terms (strip spaces for inclusion in the test string)
            terms.append(conditions["any_padded"][0].strip())

        if "any_phrase" in conditions:
            terms.append(conditions["any_phrase"][0])

        if "require_any" in conditions:
            terms.append(conditions["require_any"][0])

        test_string = " ".join(terms).lower()
        assert _pattern_matches(conditions, test_string), (
            f"Pattern '{pattern['name']}' conditions did not match synthetic string: '{test_string}'"
        )
