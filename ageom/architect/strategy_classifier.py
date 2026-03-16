"""Deterministic lightweight replacement for the architect strategy prompt."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ageom.architect.models import ConceptType
from ageom.architect.skeletons import SKELETON_TEMPLATES

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class _PhraseRule:
    phrase: str
    concept: ConceptType
    weight: float
    variant_hint: str = ""


_PHRASE_RULES: tuple[_PhraseRule, ...] = (
    _PhraseRule("longest common subsequence", ConceptType.DYNAMIC_PROGRAMMING, 4.0, "lcs"),
    _PhraseRule("edit distance", ConceptType.DYNAMIC_PROGRAMMING, 4.0, "edit_distance"),
    _PhraseRule("dynamic programming", ConceptType.DYNAMIC_PROGRAMMING, 3.0),
    _PhraseRule("memoize", ConceptType.DYNAMIC_PROGRAMMING, 2.0),
    _PhraseRule("recurrence", ConceptType.DYNAMIC_PROGRAMMING, 2.0),
    _PhraseRule("shortest path", ConceptType.GRAPH_OPTIMIZATION, 4.0),
    _PhraseRule("weighted graph", ConceptType.GRAPH_OPTIMIZATION, 2.5),
    _PhraseRule("minimum spanning", ConceptType.GRAPH_OPTIMIZATION, 3.0),
    _PhraseRule("distance map", ConceptType.GRAPH_OPTIMIZATION, 2.0),
    _PhraseRule("bandpass filter", ConceptType.SIGNAL_FILTER, 4.0),
    _PhraseRule("lowpass filter", ConceptType.SIGNAL_FILTER, 4.0),
    _PhraseRule("highpass filter", ConceptType.SIGNAL_FILTER, 4.0),
    _PhraseRule("stable bandpass", ConceptType.SIGNAL_FILTER, 3.0),
    _PhraseRule("ecg", ConceptType.SIGNAL_FILTER, 1.5),
    _PhraseRule("filter", ConceptType.SIGNAL_FILTER, 1.5),
    _PhraseRule("fft", ConceptType.SIGNAL_TRANSFORM, 4.0),
    _PhraseRule("fourier", ConceptType.SIGNAL_TRANSFORM, 4.0),
    _PhraseRule("spectrum", ConceptType.SIGNAL_TRANSFORM, 2.0),
    _PhraseRule("wavelet", ConceptType.SIGNAL_TRANSFORM, 3.0),
    _PhraseRule("symmetric positive definite", ConceptType.ALGEBRA, 4.0, "cholesky"),
    _PhraseRule("positive definite", ConceptType.ALGEBRA, 3.0, "cholesky"),
    _PhraseRule("linear system", ConceptType.ALGEBRA, 2.5),
    _PhraseRule("cholesky", ConceptType.ALGEBRA, 4.0, "cholesky"),
    _PhraseRule("eigenvalue", ConceptType.ALGEBRA, 2.0),
    _PhraseRule("search", ConceptType.SEARCHING, 1.5),
    _PhraseRule("lookup", ConceptType.SEARCHING, 1.5),
    _PhraseRule("binary search", ConceptType.SEARCHING, 4.0),
    _PhraseRule("sort", ConceptType.SORTING, 1.5),
    _PhraseRule("sorting", ConceptType.SORTING, 1.5),
    _PhraseRule("merge sort", ConceptType.DIVIDE_AND_CONQUER, 3.0, "merge_sort"),
    _PhraseRule("quicksort", ConceptType.DIVIDE_AND_CONQUER, 3.0, "quicksort"),
    _PhraseRule("divide and conquer", ConceptType.DIVIDE_AND_CONQUER, 3.0),
    _PhraseRule("recursive", ConceptType.DIVIDE_AND_CONQUER, 1.0),
    _PhraseRule("greedy", ConceptType.GREEDY, 3.0),
    _PhraseRule("bfs", ConceptType.GRAPH_TRAVERSAL, 3.0),
    _PhraseRule("dfs", ConceptType.GRAPH_TRAVERSAL, 3.0),
    _PhraseRule("traverse", ConceptType.GRAPH_TRAVERSAL, 1.5),
    _PhraseRule("regex", ConceptType.STRING_MATCHING, 2.5),
    _PhraseRule("pattern match", ConceptType.STRING_MATCHING, 2.0),
    _PhraseRule("kmp", ConceptType.STRING_MATCHING, 3.0),
    _PhraseRule("convex hull", ConceptType.GEOMETRY, 3.0),
    _PhraseRule("voronoi", ConceptType.GEOMETRY, 3.0),
    _PhraseRule("delaunay", ConceptType.GEOMETRY, 3.0),
    _PhraseRule("prime", ConceptType.NUMBER_THEORY, 1.5),
    _PhraseRule("gcd", ConceptType.NUMBER_THEORY, 2.0),
    _PhraseRule("modular", ConceptType.NUMBER_THEORY, 2.0),
    _PhraseRule("particle filter", ConceptType.SEQUENTIAL_FILTER, 4.0, "particle_filter"),
    _PhraseRule("kalman filter", ConceptType.SEQUENTIAL_FILTER, 4.0, "kalman_filter"),
    _PhraseRule("hamiltonian", ConceptType.MCMC_KERNEL, 4.0, "hmc"),
    _PhraseRule("mcmc", ConceptType.MCMC_KERNEL, 3.0, "hmc"),
    _PhraseRule("variational", ConceptType.VI_ELBO, 3.0, "advi"),
    _PhraseRule("elbo", ConceptType.VI_ELBO, 3.0, "advi"),
    _PhraseRule("belief propagation", ConceptType.MESSAGE_PASSING, 4.0, "belief_propagation"),
    _PhraseRule("message passing", ConceptType.MESSAGE_PASSING, 3.0, "belief_propagation"),
)


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _extract_goal(user: str) -> str:
    for line in user.splitlines():
        if line.lower().startswith("goal:"):
            return line.split(":", 1)[1].strip()
    return user.strip()


def _extract_allowed_concepts(system: str) -> list[ConceptType]:
    allowed = [ct for ct in ConceptType if ct.value in system]
    return allowed or list(SKELETON_TEMPLATES)


class StrategyClassifier:
    """Cheap deterministic strategy classifier with LLM fallback on ambiguity."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "strategy_classifier_v1"

    def __init__(
        self,
        fallback: Any,
        *,
        min_confidence: float = 0.55,
        min_margin: float = 0.15,
    ) -> None:
        self._fallback = fallback
        self._min_confidence = min_confidence
        self._min_margin = min_margin
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        goal = _extract_goal(user)
        allowed = _extract_allowed_concepts(system)
        decision = self._classify(goal, allowed)
        if decision is None:
            self._last_completion_metadata = {
                "strategy_source": "fallback",
                "provider_error_phase": "",
            }
            self._last_error_metadata = {}
            return await self._fallback.complete(system, user)

        concept, confidence, rationale, variant_hint = decision
        payload = {
            "paradigm": concept.value,
            "rationale": rationale,
            "variant_hint": variant_hint,
        }
        self._last_completion_metadata = {
            "strategy_source": "deterministic",
            "strategy_confidence": round(confidence, 3),
            "strategy_variant_hint": variant_hint,
        }
        self._last_error_metadata = {}
        return json.dumps(payload)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)

    def classify(
        self,
        goal: str,
        allowed: list[ConceptType] | None = None,
    ) -> tuple[ConceptType, float, str, str] | None:
        """Classify a goal without going through prompt formatting."""
        allowed_concepts = allowed or list(SKELETON_TEMPLATES)
        return self._classify(goal, allowed_concepts)

    def _classify(
        self,
        goal: str,
        allowed: list[ConceptType],
    ) -> tuple[ConceptType, float, str, str] | None:
        goal_lower = goal.lower()
        goal_tokens = _tokenize(goal)
        if not goal_tokens:
            return None

        signal_markers = {"signal", "waveform", "timeseries", "time_series", "ecg", "ppg", "eeg", "sensor"}
        detect_markers = {"detect", "peak", "event", "events", "feature", "features"}
        rate_markers = {"rate", "cadence", "rhythm"}
        if (
            ConceptType.SIGNAL_FILTER in allowed
            and goal_tokens & signal_markers
            and goal_tokens & detect_markers
            and goal_tokens & rate_markers
        ):
            return (
                ConceptType.SIGNAL_FILTER,
                0.96,
                "deterministic match from signal + detect + rate cues",
                "event_rate_estimation",
            )

        scores: dict[ConceptType, float] = {concept: 0.0 for concept in allowed}
        reasons: dict[ConceptType, list[str]] = {concept: [] for concept in allowed}
        variants: dict[ConceptType, str] = {concept: "" for concept in allowed}

        for rule in _PHRASE_RULES:
            if rule.concept not in scores or rule.phrase not in goal_lower:
                continue
            scores[rule.concept] += rule.weight
            reasons[rule.concept].append(rule.phrase)
            if rule.variant_hint and not variants[rule.concept]:
                variants[rule.concept] = rule.variant_hint

        for concept in allowed:
            skeleton = SKELETON_TEMPLATES.get(concept)
            if skeleton is None:
                continue
            lexical_parts = [skeleton.name, skeleton.description, *skeleton.variants]
            lexical_tokens = _tokenize(" ".join(lexical_parts))
            overlap = goal_tokens & lexical_tokens
            if overlap:
                scores[concept] += min(2.0, 0.35 * len(overlap))
                reasons[concept].extend(sorted(overlap)[:3])
            if not variants[concept]:
                for variant in skeleton.variants:
                    if variant.replace("_", " ") in goal_lower:
                        variants[concept] = variant
                        break

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_concept, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score <= 0:
            return None

        confidence = min(0.99, 0.2 + 0.15 * best_score)
        margin = best_score - second_score
        if confidence < self._min_confidence or margin < self._min_margin:
            return None

        rationale = ", ".join(dict.fromkeys(reasons[best_concept])) or "lexical match"
        return best_concept, confidence, f"deterministic match from {rationale}", variants[best_concept]
