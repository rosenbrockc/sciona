"""Index builder and SemanticIndex implementation."""

from __future__ import annotations

from ageom.indexer.embedder import UniXcoderEmbedder
from ageom.indexer.faiss_store import FAISSStore
from ageom.indexer.models import IndexEntry, IndexMetadata
from ageom.types import Declaration, Prover


class IndexBuilder:
    """Orchestrates the source -> embed -> store pipeline."""

    def __init__(
        self,
        embedder: UniXcoderEmbedder | None = None,
        store: FAISSStore | None = None,
    ) -> None:
        self._embedder = embedder or UniXcoderEmbedder()
        self._store = store or FAISSStore(dim=self._embedder.dim)

    def build_from_declarations(
        self,
        declarations: list[Declaration],
        source_lib: str = "",
        prover: Prover = Prover.LEAN4,
        batch_size: int = 32,
    ) -> FAISSStore:
        """Build an index from a list of declarations."""
        # Prepare texts for batch embedding
        texts: list[str] = []
        for decl in declarations:
            text = f"{decl.name} : {decl.type_signature}"
            if decl.docstring:
                text += f"\n{decl.docstring}"
            if decl.conceptual_summary:
                text += f"\n{decl.conceptual_summary}"
            texts.append(text)

        # Batch embed
        embeddings = self._embedder.embed_batch(texts, batch_size=batch_size)

        # Create index entries
        entries = [
            IndexEntry(
                declaration=decl,
                embedding=embeddings[i],
                source_text=texts[i],
            )
            for i, decl in enumerate(declarations)
        ]

        self._store.add(entries)
        self._store.set_metadata(
            IndexMetadata(
                num_entries=len(entries),
                prover=prover,
                source_lib=source_lib,
                embedding_model="microsoft/unixcoder-base",
            )
        )
        return self._store


class SemanticIndexImpl:
    """Concrete implementation of the SemanticIndex protocol.

    Combines FAISS vector search with optional lean-explore type search.
    """

    def __init__(
        self,
        store: FAISSStore,
        embedder: UniXcoderEmbedder,
        lean_source: object | None = None,  # LeanDeclarationSource, optional
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._lean_source = lean_source
        # Build name lookup from store
        self._by_name: dict[str, Declaration] = {
            decl.name: decl for decl in store._declarations.values()
        }

    def search_by_embedding(
        self, query_text: str, k: int = 10
    ) -> list[tuple[Declaration, float]]:
        """Search by embedding similarity."""
        query_vec = self._embedder.embed(query_text)
        return self._store.search(query_vec, k=k)

    def search_by_type(
        self, type_signature: str, k: int = 10
    ) -> list[Declaration]:
        """Search by type signature.

        For Lean: delegates to lean-explore's hybrid search.
        For Coq / no lean source: falls back to embedding search.
        """
        if self._lean_source is not None:
            from ageom.indexer.lean_source import LeanDeclarationSource

            if isinstance(self._lean_source, LeanDeclarationSource):
                return self._lean_source.search_by_type(type_signature, k=k)

        # Fallback: embed the type signature and search
        results = self.search_by_embedding(type_signature, k=k)
        return [decl for decl, _score in results]

    def get_declaration(self, name: str) -> Declaration | None:
        """Look up a declaration by fully-qualified name."""
        return self._by_name.get(name)
