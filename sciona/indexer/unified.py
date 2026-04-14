"""Unified index: single FAISSStore + local embedder implementing SemanticIndex.

Both SkillIndex and SemanticIndexImpl delegate to this common base, ensuring
the Architect and Hunter can share the same underlying index at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sciona.indexer.embedder import DEFAULT_EMBEDDING_BACKEND, Embedder, create_embedder
from sciona.indexer.faiss_store import FAISSStore
from sciona.indexer.models import IndexEntry
from sciona.types import Declaration


class UnifiedIndex:
    """Core index implementing the SemanticIndex protocol.

    Wraps a single FAISSStore + local embedder and exposes the
    ``search_by_embedding``, ``search_by_type``, and ``get_declaration``
    methods required by the protocol.
    """

    def __init__(
        self,
        store: FAISSStore | None = None,
        embedder: Embedder | None = None,
        embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
        embedding_model: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._embedding_backend = embedding_backend
        self._embedding_model = embedding_model
        self._by_name: dict[str, Declaration] = {}
        if store is not None:
            self._by_name = {decl.name: decl for decl in store._declarations.values()}
            if store._metadata is not None:
                self._embedding_backend = store._metadata.embedding_backend
                self._embedding_model = store._metadata.embedding_model
            else:
                self._embedding_backend = "unixcoder"

    def _ensure_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = create_embedder(
                backend=self._embedding_backend,
                model_name=self._embedding_model,
            )
        return self._embedder

    def add_entries(self, entries: list[IndexEntry]) -> None:
        """Add entries to the underlying FAISS store."""
        if self._store is None:
            embedder = self._ensure_embedder()
            self._store = FAISSStore(dim=embedder.dim)
        self._store.add(entries)
        for entry in entries:
            self._by_name[entry.declaration.name] = entry.declaration

    def search_by_embedding(
        self, query_text: str, k: int = 10
    ) -> list[tuple[Declaration, float]]:
        """Search by embedding similarity."""
        if self._store is None or self._store.size == 0:
            return []
        embedder = self._ensure_embedder()
        query_vec = embedder.embed(query_text)
        return self._store.search(query_vec, k=k)

    def search_by_type(self, type_signature: str, k: int = 10) -> list[Declaration]:
        """Search by type signature (falls back to embedding search)."""
        results = self.search_by_embedding(type_signature, k=k)
        return [decl for decl, _score in results]

    def get_declaration(self, name: str) -> Declaration | None:
        """Look up a declaration by fully-qualified name."""
        return self._by_name.get(name)

    @property
    def size(self) -> int:
        return self._store.size if self._store else 0

    def save(self, directory: str | Path) -> None:
        """Persist to disk."""
        if self._store is not None:
            self._store.save(directory)

    @classmethod
    def load(cls, directory: str | Path) -> UnifiedIndex:
        """Load from disk."""
        store = FAISSStore.load(directory)
        idx = cls(store=store)
        return idx


@dataclass(frozen=True)
class SemanticIndexSource:
    """A semantic index plus the embedding space it was built in."""

    index: object
    embedding_space: str
    name: str = ""


class CompositeSemanticIndex:
    """Merge multiple semantic indexes that share an embedding space."""

    def __init__(self, sources: Sequence[SemanticIndexSource]) -> None:
        self._sources = tuple(sources)
        self._embedding_space = self._validate_sources()

    def _validate_sources(self) -> str:
        if not self._sources:
            return ""
        spaces = {source.embedding_space for source in self._sources}
        if len(spaces) > 1:
            details = ", ".join(
                f"{source.name or f'source_{idx}'}={source.embedding_space}"
                for idx, source in enumerate(self._sources)
            )
            raise ValueError(
                "CompositeSemanticIndex cannot merge incompatible embedding spaces: "
                f"{details}"
            )
        return next(iter(spaces))

    @property
    def embedding_space(self) -> str:
        return self._embedding_space

    def search_by_embedding(
        self, query_text: str, k: int = 10
    ) -> list[tuple[Declaration, float]]:
        if k <= 0 or not self._sources:
            return []

        per_source_k = max(k, k * len(self._sources))
        best: dict[str, tuple[Declaration, float, int]] = {}
        for source_idx, source in enumerate(self._sources):
            hits = source.index.search_by_embedding(query_text, k=per_source_k)
            for decl, score in hits:
                current = best.get(decl.name)
                if current is None or score > current[1] or (
                    score == current[1] and source_idx < current[2]
                ):
                    best[decl.name] = (decl, float(score), source_idx)

        ranked = sorted(
            best.values(),
            key=lambda item: (-item[1], item[2], item[0].name),
        )
        return [(decl, score) for decl, score, _ in ranked[:k]]

    def search_by_type(self, type_signature: str, k: int = 10) -> list[Declaration]:
        return [decl for decl, _score in self.search_by_embedding(type_signature, k=k)]

    def get_declaration(self, name: str) -> Declaration | None:
        for source in self._sources:
            decl = source.index.get_declaration(name)
            if decl is not None:
                return decl
        return None
