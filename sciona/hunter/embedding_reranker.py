"""Embedding-based candidate reranker — replaces the hunter_score LLM call."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from sciona.indexer.embedder import Embedder

_CANDIDATE_RE = re.compile(r"^\[(\d+)\]\s+(.+?)\s+:\s+(.+)$")


def _extract_query(user: str) -> str:
    """Extract the query text (statement + description) from the score prompt."""
    statement = ""
    description = ""
    for line in user.splitlines():
        stripped = line.strip()
        if stripped.startswith("Statement:"):
            statement = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Description:"):
            description = stripped.split(":", 1)[1].strip()
    return f"{statement} {description}".strip()


def _extract_candidates(user: str) -> list[tuple[int, str, str]]:
    """Extract (index, name, type_signature) from the candidates list."""
    candidates: list[tuple[int, str, str]] = []
    for line in user.splitlines():
        match = _CANDIDATE_RE.match(line.strip())
        if not match:
            continue
        candidates.append((int(match.group(1)), match.group(2), match.group(3)))
    return candidates


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower().replace("_", " ").replace(".", " ")))


def _type_bonus(query: str, type_sig: str) -> float:
    """Add a small bonus for overlapping tokens between query and type signature."""
    query_tokens = _tokenize(query)
    type_tokens = _tokenize(type_sig)
    overlap = len(query_tokens & type_tokens)
    return min(overlap * 0.1, 0.3)


class EmbeddingReranker:
    """Rank candidates by embedding similarity — no LLM call.

    Implements the LLMClient protocol so it can be used as a drop-in
    override for the hunter_score prompt key.
    """

    _telemetry_provider = "deterministic"
    _telemetry_model = "embedding_reranker_v1"

    def __init__(
        self,
        embedder: Embedder,
        fallback: Any,
        *,
        confidence_margin: float = 0.05,
    ) -> None:
        self._embedder = embedder
        self._fallback = fallback
        self._confidence_margin = confidence_margin
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        query = _extract_query(user)
        candidates = _extract_candidates(user)

        if not query or not candidates:
            self._last_completion_metadata = {"ranking_source": "fallback"}
            return await self._fallback.complete(system, user)

        query_vec = self._embedder.embed(query)

        scored: list[tuple[float, int]] = []
        for idx, name, type_sig in candidates:
            cand_text = f"{name} : {type_sig}"
            cand_vec = self._embedder.embed(cand_text)
            sim = float(np.dot(query_vec, cand_vec))
            sim += _type_bonus(query, type_sig)
            scored.append((sim, idx))

        scored.sort(key=lambda row: (-row[0], row[1]))

        # Confidence gate: if top-2 are too close, fall back to LLM
        if len(scored) > 1:
            margin = scored[0][0] - scored[1][0]
            if margin < self._confidence_margin:
                self._last_completion_metadata = {
                    "ranking_source": "fallback",
                    "embedding_margin": round(margin, 4),
                }
                return await self._fallback.complete(system, user)

        self._last_completion_metadata = {
            "ranking_source": "embedding",
            "embedding_top_score": round(scored[0][0], 4) if scored else 0.0,
            "embedding_margin": round(
                scored[0][0] - scored[1][0] if len(scored) > 1 else 1.0, 4
            ),
        }
        self._last_error_metadata = {}
        return json.dumps([idx for _, idx in scored])

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)
