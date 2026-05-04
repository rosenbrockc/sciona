"""LLM-based semantic reranking of candidate solution templates."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sciona.architect.solution_index import SolutionTemplate

logger = logging.getLogger(__name__)

TEMPLATE_RERANK_SYSTEM = """\
You are an expert ML architect comparing a problem description against \
candidate solution templates from past Kaggle competitions.

For each candidate, evaluate:
1. Does the problem TYPE match? (classification vs regression vs detection \
vs segmentation vs ranking vs recommendation vs optimization)
2. Does the data MODALITY match? (tabular vs image vs text vs audio vs \
time_series vs 3d_volume vs graph vs geospatial)
3. Are the key CHALLENGES similar? (class imbalance, noisy labels, domain \
shift, large scale, multi-modal fusion, streaming/online, few-shot)
4. Would the template's CRITICAL STAGES apply to this problem?
5. Are there CONTRAINDICATIONS (do_not_use_when) that disqualify this template?

Output JSON only (no markdown fences):
{
  "rankings": [
    {
      "template": "<name>",
      "score": <float 0.0-1.0>,
      "reasoning": "<1-2 sentences>"
    }
  ],
  "best_match": "<name or 'none'>",
  "should_compose_novel": <true|false>,
  "novel_reasoning": "<why no template fits, if applicable>"
}"""


@dataclass(frozen=True)
class RerankResult:
    """Result of LLM template reranking."""

    template_name: str
    score: float
    reasoning: str


@dataclass(frozen=True)
class RerankOutput:
    """Full output from the reranking step."""

    rankings: list[RerankResult]
    best_match: str | None
    should_compose_novel: bool
    novel_reasoning: str


def _format_rerank_prompt(
    prompt: str,
    candidates: list[SolutionTemplate],
) -> str:
    """Format the user prompt for the reranker."""
    parts = [f"## Problem Description\n\n{prompt}\n\n## Candidate Templates\n"]

    for i, tmpl in enumerate(candidates, 1):
        use_when = "\n".join(f"  - {u}" for u in tmpl.use_when) or "  (none)"
        do_not = "\n".join(f"  - {d}" for d in tmpl.do_not_use_when) or "  (none)"
        failure = "\n".join(f"  - {f}" for f in tmpl.failure_modes) or "  (none)"
        stages = ", ".join(tmpl.stage_names[:8])
        if len(tmpl.stage_names) > 8:
            stages += f", ... ({len(tmpl.stage_names)} total)"

        parts.append(f"""### Candidate {i}: {tmpl.name}
- **Family**: {tmpl.family}
- **Paradigm**: {tmpl.paradigm}
- **Summary**: {tmpl.dejargonized_summary or tmpl.summary}
- **Key Insight**: {tmpl.key_insight}
- **Use When**:
{use_when}
- **Do NOT Use When**:
{do_not}
- **Failure Modes**:
{failure}
- **Stages**: {stages}
- **Grounding Rate**: {tmpl.grounding_rate:.0%}
""")

    parts.append(
        "Evaluate each candidate and return the JSON ranking. "
        "If none are a good fit, set should_compose_novel=true."
    )
    return "\n".join(parts)


def _parse_rerank_response(text: str) -> RerankOutput:
    """Parse the LLM's JSON response."""
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse rerank response: %s", text[:200])
        return RerankOutput(
            rankings=[],
            best_match=None,
            should_compose_novel=True,
            novel_reasoning="Failed to parse LLM response",
        )

    rankings = [
        RerankResult(
            template_name=r.get("template", ""),
            score=float(r.get("score", 0.0)),
            reasoning=r.get("reasoning", ""),
        )
        for r in data.get("rankings", [])
    ]
    rankings.sort(key=lambda r: -r.score)

    best = data.get("best_match")
    if best == "none":
        best = None

    return RerankOutput(
        rankings=rankings,
        best_match=best,
        should_compose_novel=data.get("should_compose_novel", False),
        novel_reasoning=data.get("novel_reasoning", ""),
    )


async def rerank_templates(
    prompt: str,
    candidates: list[SolutionTemplate],
    llm: object,
    max_candidates: int = 5,
) -> RerankOutput:
    """Use LLM to rerank candidate templates by semantic fit.

    Parameters
    ----------
    prompt : str
        The original (non-dejargonized) user problem description.
    candidates : list[SolutionTemplate]
        Pre-filtered candidates from keyword retrieval.
    llm : LLMClient
        Any object with ``async complete(system, user)`` method.
    max_candidates : int
        Maximum candidates to send to LLM (cost control).

    Returns
    -------
    RerankOutput
        Ranked templates with confidence scores and reasoning.
    """
    candidates = candidates[:max_candidates]
    if not candidates:
        return RerankOutput(
            rankings=[],
            best_match=None,
            should_compose_novel=True,
            novel_reasoning="No candidates to rerank",
        )

    user_prompt = _format_rerank_prompt(prompt, candidates)
    response = await llm.complete(  # type: ignore[union-attr]
        system=TEMPLATE_RERANK_SYSTEM,
        user=user_prompt,
    )
    return _parse_rerank_response(response.text)  # type: ignore[union-attr]
