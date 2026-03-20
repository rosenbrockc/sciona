"""Unified index: single FAISSStore + local embedder implementing SemanticIndex.

Both SkillIndex and SemanticIndexImpl delegate to this common base, ensuring
the Architect and Hunter can share the same underlying index at runtime.
"""

from __future__ import annotations

from pathlib import Path

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
