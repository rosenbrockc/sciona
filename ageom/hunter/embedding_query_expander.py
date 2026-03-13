"""Embedding-based query expansion — generates search queries from nearest declarations."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ageom.indexer.embedder import Embedder
from ageom.types import Declaration


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower().replace(".", " ").replace("-", " "))


def _namespace_prefix(name: str) -> str:
    parts = [seg for seg in re.split(r"[:./]+", name) if seg]
    if len(parts) <= 1:
        return ""
    return ".".join(parts[:-1])


class EmbeddingQueryExpander:
    """Generate search queries by finding nearest declarations in embedding space.

    Follows the ``EmbeddingReranker`` pattern: takes an ``Embedder`` and a list
    of declarations, embeds lazily on first call, and uses cosine similarity to
    find relevant declaration tokens for query composition.
    """

    _telemetry_provider = "deterministic"
    _telemetry_model = "embedding_query_expander_v1"

    def __init__(
        self,
        embedder: Embedder,
        declarations: list[Declaration],
        *,
        top_k: int = 10,
    ) -> None:
        self._embedder = embedder
        self._declarations = declarations
        self._top_k = top_k
        self._declaration_vecs: np.ndarray | None = None

    def _ensure_vecs(self) -> np.ndarray:
        if self._declaration_vecs is not None:
            return self._declaration_vecs
        if not self._declarations:
            self._declaration_vecs = np.empty((0, 0))
            return self._declaration_vecs
        texts = [
            f"{d.name} {d.type_signature} {d.docstring}".strip()
            for d in self._declarations
        ]
        self._declaration_vecs = self._embedder.embed_batch(texts)
        return self._declaration_vecs

    def expand(self, text: str, max_queries: int = 5) -> list[str]:
        """Return up to *max_queries* search query strings derived from nearest declarations."""
        if not self._declarations or not text.strip():
            return []

        vecs = self._ensure_vecs()
        if vecs.size == 0:
            return []

        query_vec = self._embedder.embed(text)
        sims = vecs @ query_vec
        top_indices = np.argsort(-sims)[: self._top_k]

        seen: set[str] = set()
        queries: list[str] = []

        for idx in top_indices:
            decl = self._declarations[idx]
            tokens = _tokenize(decl.name)
            if not tokens:
                continue

            # Primary query: declaration name tokens
            name_query = " ".join(tokens)
            if name_query not in seen:
                seen.add(name_query)
                queries.append(name_query)

            # Secondary query: namespace prefix + key name tokens
            ns = _namespace_prefix(decl.name)
            if ns:
                ns_query = f"{ns} {' '.join(tokens[-2:])}"
                if ns_query not in seen:
                    seen.add(ns_query)
                    queries.append(ns_query)

            if len(queries) >= max_queries:
                break

        return queries[:max_queries]
