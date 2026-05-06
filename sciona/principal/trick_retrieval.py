"""Gated retrieval over provider solution-trick registries.

Tricks are tactical context for novel or under-covered CDG cases. They are
loaded separately from CDG and expansion assets so they cannot influence base
template ranking unless an architect flow explicitly asks for them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from sciona.atom_identity import candidate_atom_provider_roots


EXTERNAL_TRICK_DIR_CANDIDATES: tuple[tuple[str, ...], ...] = (("data", "solution_tricks"),)
NOVELTY_ASSESSMENTS = {
    "divergent",
    "novel",
    "true_novel",
    "true_novel_composition",
    "novel_cdg_required",
    "under_covered",
    "undercovered",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HIGH_RISK_LEVELS = {"high", "disallowed"}


@dataclass(frozen=True)
class SolutionTrick:
    """One tactic from a provider solution-trick registry."""

    trick_id: str
    name: str
    kind: str
    status: str
    risk_level: str
    generalization_level: str
    summary: str
    applies_when: tuple[str, ...] = ()
    do_not_use_when: tuple[str, ...] = ()
    validation_requirements: tuple[str, ...] = ()
    architect_hint: str = ""
    related_cdgs: tuple[str, ...] = ()
    related_operations: tuple[str, ...] = ()
    source_competitions: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    audit: dict[str, Any] = field(default_factory=dict)
    provider_root: str = ""


@dataclass(frozen=True)
class SolutionTrickMatch:
    """One ranked trick candidate for architect context."""

    trick: SolutionTrick
    score: float
    reasons: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()
    high_risk: bool = False


@dataclass(frozen=True)
class TrickRetrievalQuery:
    """Query for optional trick retrieval after CDG/expansion under-coverage."""

    goal: str = ""
    missing_techniques: tuple[str, ...] = ()
    candidate_cdgs: tuple[str, ...] = ()
    families: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(str(text or "").lower()))


def _token_list(values: Iterable[str]) -> frozenset[str]:
    return frozenset(
        token
        for value in values
        for token in _TOKEN_RE.findall(str(value or "").lower())
    )


def _jaccard(query_tokens: frozenset[str], candidate_tokens: frozenset[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / max(1, len(query_tokens | candidate_tokens))


def _phrase_coverage(phrases: Iterable[str], candidate_tokens: frozenset[str]) -> tuple[str, ...]:
    covered: list[str] = []
    for phrase in phrases:
        phrase_tokens = _tokens(phrase)
        if not phrase_tokens:
            continue
        if len(phrase_tokens & candidate_tokens) / len(phrase_tokens) >= 0.60:
            covered.append(str(phrase))
    return tuple(covered)


def _string_tuple(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return tuple()
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _trick_dir_for_root(root: Path) -> Path:
    for relative in EXTERNAL_TRICK_DIR_CANDIDATES:
        candidate = root.joinpath(*relative)
        if candidate.exists():
            return candidate
    return root.joinpath(*EXTERNAL_TRICK_DIR_CANDIDATES[0])


def _trick_registry_dirs() -> tuple[Path, ...]:
    dirs: list[Path] = []
    for root in candidate_atom_provider_roots():
        dirs.append(_trick_dir_for_root(Path(root).expanduser().resolve()))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for directory in dirs:
        resolved = directory.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return tuple(deduped)


def _trick_text(trick: SolutionTrick) -> str:
    return " ".join(
        [
            trick.trick_id,
            trick.name,
            trick.kind,
            trick.risk_level,
            trick.generalization_level,
            trick.summary,
            *trick.applies_when,
            *trick.do_not_use_when,
            *trick.validation_requirements,
            trick.architect_hint,
            *trick.related_cdgs,
            *trick.related_operations,
            *trick.source_competitions,
            *trick.tags,
        ]
    )


def _parse_trick(raw: dict[str, Any], *, provider_root: Path) -> SolutionTrick:
    return SolutionTrick(
        trick_id=str(raw.get("trick_id", "")).strip(),
        name=str(raw.get("name", "")).strip(),
        kind=str(raw.get("kind", "")).strip(),
        status=str(raw.get("status", "")).strip(),
        risk_level=str(raw.get("risk_level", "")).strip(),
        generalization_level=str(raw.get("generalization_level", "")).strip(),
        summary=str(raw.get("summary", "")).strip(),
        applies_when=_string_tuple(raw.get("applies_when")),
        do_not_use_when=_string_tuple(raw.get("do_not_use_when")),
        validation_requirements=_string_tuple(raw.get("validation_requirements")),
        architect_hint=str(raw.get("architect_hint", "")).strip(),
        related_cdgs=_string_tuple(raw.get("related_cdgs")),
        related_operations=_string_tuple(raw.get("related_operations")),
        source_competitions=_string_tuple(raw.get("source_competitions")),
        source_references=_string_tuple(raw.get("source_references")),
        tags=_string_tuple(raw.get("tags")),
        audit=raw.get("audit") if isinstance(raw.get("audit"), dict) else {},
        provider_root=str(provider_root),
    )


@lru_cache(maxsize=1)
def load_local_solution_tricks() -> tuple[SolutionTrick, ...]:
    """Load provider solution-trick registries from disk."""
    tricks: list[SolutionTrick] = []
    seen_ids: set[str] = set()
    for registry_dir in _trick_registry_dirs():
        if not registry_dir.exists():
            continue
        for path in sorted(registry_dir.glob("*.json")):
            if path.name == "schema.json":
                continue
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict):
                continue
            raw_tricks = raw.get("tricks")
            if not isinstance(raw_tricks, list):
                continue
            for raw_trick in raw_tricks:
                if not isinstance(raw_trick, dict):
                    continue
                trick = _parse_trick(raw_trick, provider_root=registry_dir.parent.parent)
                if not trick.trick_id or trick.trick_id in seen_ids:
                    continue
                seen_ids.add(trick.trick_id)
                tricks.append(trick)
    return tuple(tricks)


def clear_solution_trick_caches() -> None:
    """Clear cached trick registry loaders."""
    load_local_solution_tricks.cache_clear()


def _format_prompt_list(label: str, values: tuple[str, ...]) -> list[str]:
    if not values:
        return [f"  {label}: (none declared)"]
    lines = [f"  {label}:"]
    lines.extend(f"    - {value}" for value in values)
    return lines


def format_solution_trick_prompt_section(
    matches: Iterable[SolutionTrickMatch],
    *,
    max_chars: int = 4500,
) -> str:
    """Render retrieved tricks as a distinct optional architect prompt block."""
    selected = tuple(matches)
    if not selected:
        return ""

    lines: list[str] = [
        "Optional high-risk tactics for novel-CDG cases",
        "",
        "Use this section only after a better-fitting base CDG plus available "
        "expansion/refinement operations remains under-covered or truly novel.",
        "Do not use these tactics as the base topology, and do not let them "
        "override CDG, expansion, validation, or admissibility evidence.",
        "",
    ]
    for index, match in enumerate(selected, start=1):
        trick = match.trick
        lines.extend(
            [
                f"{index}. {trick.name} ({trick.trick_id})",
                f"  kind: {trick.kind}",
                f"  risk_level: {trick.risk_level}",
                f"  generalization_level: {trick.generalization_level}",
                f"  score: {match.score:.3f}",
                f"  reasons: {', '.join(match.reasons) if match.reasons else '(none)'}",
                f"  summary: {trick.summary}",
            ]
        )
        lines.extend(_format_prompt_list("applies_when", trick.applies_when))
        lines.extend(_format_prompt_list("do_not_use_when", trick.do_not_use_when))
        lines.extend(
            _format_prompt_list(
                "validation_requirements",
                trick.validation_requirements,
            )
        )
        lines.append(
            f"  architect_hint: {trick.architect_hint or '(none declared)'}"
        )
        lines.extend(_format_prompt_list("related_cdgs", trick.related_cdgs))
        lines.extend(
            _format_prompt_list("related_operations", trick.related_operations)
        )
        lines.append("")

    rendered = "\n".join(lines).strip()
    if len(rendered) <= max_chars:
        return rendered
    truncated = rendered[: max(0, max_chars - 80)].rstrip()
    return truncated + "\n\n[trick context truncated to prompt budget]"


def should_consult_tricks(novelty_assessment: Any) -> bool:
    """Return whether architect context may include optional tricks.

    This gate is intentionally conservative. Trick retrieval is allowed only
    when the caller explicitly indicates a novel/divergent/under-covered CDG
    case, or provides a counterfactual expansion decision indicating true
    novelty.
    """
    if isinstance(novelty_assessment, str):
        normalized = novelty_assessment.strip().lower().replace("-", "_")
        return normalized in NOVELTY_ASSESSMENTS

    if not isinstance(novelty_assessment, dict):
        return False

    for key in ("novel_cdg_required", "should_compose_novel", "true_novel"):
        if novelty_assessment.get(key) is True:
            return True

    for key in ("assessment", "base_assessment", "decision", "status"):
        value = novelty_assessment.get(key)
        if isinstance(value, str) and should_consult_tricks(value):
            return True

    counterfactual = novelty_assessment.get("counterfactual_expansion")
    if isinstance(counterfactual, dict) and should_consult_tricks(counterfactual):
        return True

    projected = novelty_assessment.get("projected_coverage")
    if isinstance(projected, (int, float)) and projected < 0.50:
        return True

    return False


class SolutionTrickRetriever:
    """Rank optional trick context without modifying CDG or expansion retrieval."""

    def __init__(self, tricks: Iterable[SolutionTrick] | None = None) -> None:
        self._tricks = tuple(tricks) if tricks is not None else load_local_solution_tricks()
        self._indexed = tuple((trick, _tokens(_trick_text(trick))) for trick in self._tricks)

    def retrieve(
        self,
        query: TrickRetrievalQuery,
        *,
        novelty_assessment: Any,
        max_results: int = 5,
        min_score: float = 0.05,
        include_disallowed: bool = False,
    ) -> list[SolutionTrickMatch]:
        if not should_consult_tricks(novelty_assessment):
            return []

        goal_terms = _tokens(query.goal)
        missing_phrases = tuple(query.missing_techniques)
        missing_terms = _token_list(missing_phrases)
        family_terms = _token_list(query.families)
        tag_terms = _token_list(query.tags)
        candidate_cdgs = {str(cdg).strip() for cdg in query.candidate_cdgs if str(cdg).strip()}

        matches: list[SolutionTrickMatch] = []
        for trick, trick_tokens in self._indexed:
            if trick.status == "disallowed" and not include_disallowed:
                continue

            related_hit = bool(candidate_cdgs & set(trick.related_cdgs))
            matched_terms = _phrase_coverage(missing_phrases, trick_tokens)
            goal_score = _jaccard(goal_terms, trick_tokens)
            missing_score = _jaccard(missing_terms, trick_tokens)
            family_score = _jaccard(family_terms, trick_tokens)
            tag_score = _jaccard(tag_terms, trick_tokens)
            related_score = 1.0 if related_hit else 0.0
            if matched_terms:
                missing_score = max(missing_score, min(1.0, len(matched_terms) / max(1, len(missing_phrases))))

            score = (
                0.30 * goal_score
                + 0.30 * missing_score
                + 0.20 * related_score
                + 0.12 * family_score
                + 0.08 * tag_score
            )
            if not goal_terms and not missing_terms and not family_terms and not tag_terms and not candidate_cdgs:
                score = 0.0
            if score < min_score:
                continue

            reasons: list[str] = []
            if matched_terms:
                reasons.append("missing_technique:" + ", ".join(matched_terms))
            if related_hit:
                reasons.append("related_cdg")
            if goal_score > 0:
                reasons.append("goal_overlap")
            if family_score > 0:
                reasons.append("family_overlap")
            if tag_score > 0:
                reasons.append("tag_overlap")

            matches.append(
                SolutionTrickMatch(
                    trick=trick,
                    score=round(min(1.0, score), 6),
                    reasons=tuple(reasons),
                    matched_terms=matched_terms,
                    high_risk=trick.risk_level in _HIGH_RISK_LEVELS,
                )
            )

        matches.sort(
            key=lambda match: (
                match.high_risk,
                -match.score,
                match.trick.risk_level,
                match.trick.trick_id,
            )
        )
        return matches[:max_results]


def retrieve_tricks(
    goal: str,
    candidate_cdgs: Iterable[str] = (),
    novelty_assessment: Any = None,
    *,
    missing_techniques: Iterable[str] = (),
    families: Iterable[str] = (),
    tags: Iterable[str] = (),
    max_results: int = 5,
) -> list[SolutionTrickMatch]:
    """Convenience wrapper for gated solution-trick retrieval."""
    return SolutionTrickRetriever().retrieve(
        TrickRetrievalQuery(
            goal=goal,
            missing_techniques=tuple(missing_techniques),
            candidate_cdgs=tuple(candidate_cdgs),
            families=tuple(families),
            tags=tuple(tags),
        ),
        novelty_assessment=novelty_assessment,
        max_results=max_results,
    )
