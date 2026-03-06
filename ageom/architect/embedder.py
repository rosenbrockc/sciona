"""Skill embedding index for semantic search over algorithmic primitives.

Reuses the existing FAISSStore and UniXcoderEmbedder, building a separate
index specifically for the architect's primitive catalog.
"""

from __future__ import annotations

import json
from pathlib import Path

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.models import AlgorithmicPrimitive
from ageom.indexer.faiss_store import FAISSStore
from ageom.indexer.models import IndexEntry
from ageom.types import Declaration, Prover


class SkillIndex:
    """Semantic search over algorithmic primitives using FAISS embeddings.

    Wraps the existing indexer infrastructure, storing a separate FAISS index
    at data/skill_index/ (distinct from the declaration index at data/index/).
    """

    def __init__(self, index_dir: str | Path = "data/skill_index") -> None:
        self._index_dir = Path(index_dir)
        self._store = None  # Lazy — set by build or load
        self._embedder = None  # Lazy
        self._primitives: list[AlgorithmicPrimitive] = []
        self._id_to_primitive: dict[int, AlgorithmicPrimitive] = {}

    def _ensure_embedder(self):
        if self._embedder is None:
            from ageom.indexer.embedder import UniXcoderEmbedder

            self._embedder = UniXcoderEmbedder()

    def _primitive_to_text(self, prim: AlgorithmicPrimitive) -> str:
        """Format a primitive for embedding."""
        parts = [f"{prim.name}: {prim.description}"]
        if prim.inputs:
            required_inputs = ", ".join(
                f"{io.name}: {io.type_desc}" for io in prim.inputs if io.required
            )
            optional_inputs = ", ".join(
                f"{io.name}: {io.type_desc}={io.default_value_repr or '<default>'}"
                for io in prim.inputs
                if not io.required
            )
            if required_inputs:
                parts.append(f"Required Inputs: {required_inputs}")
            if optional_inputs:
                parts.append(f"Optional Inputs: {optional_inputs}")
        if prim.outputs:
            outputs_str = ", ".join(f"{io.name}: {io.type_desc}" for io in prim.outputs)
            parts.append(f"Outputs: {outputs_str}")
        return "\n".join(parts)

    def build_from_catalog(self, catalog: PrimitiveCatalog) -> None:
        """Build the FAISS index from all primitives in the catalog."""
        self._ensure_embedder()
        primitives = catalog.all_primitives()
        if not primitives:
            return

        self._primitives = primitives

        # Embed all primitives
        texts = [self._primitive_to_text(p) for p in primitives]
        embeddings = self._embedder.embed_batch(texts)

        # Build FAISS store using Declaration wrappers
        store = FAISSStore(dim=self._embedder.dim)
        entries: list[IndexEntry] = []
        for i, prim in enumerate(primitives):
            # Wrap primitive as a Declaration for FAISSStore compatibility
            decl = Declaration(
                name=prim.name,
                type_signature=prim.type_signature,
                docstring=prim.description,
                source_lib=prim.source,
                prover=Prover.LEAN4,  # placeholder — primitives are prover-agnostic
            )
            entries.append(
                IndexEntry(
                    declaration=decl,
                    embedding=embeddings[i],
                    source_text=texts[i],
                )
            )
            self._id_to_primitive[i] = prim

        store.add(entries)
        self._store = store

    # --- SemanticIndex protocol methods ---

    def search_by_embedding(
        self, query_text: str, k: int = 10
    ) -> list[tuple[Declaration, float]]:
        """Search by embedding similarity. Returns (Declaration, score) pairs."""
        if self._store is None:
            return []
        self._ensure_embedder()
        query_vec = self._embedder.embed(query_text)
        return self._store.search(query_vec, k=k)

    def search_by_type(self, type_signature: str, k: int = 10) -> list[Declaration]:
        """Search by type signature (falls back to embedding search)."""
        results = self.search_by_embedding(type_signature, k=k)
        return [decl for decl, _score in results]

    def get_declaration(self, name: str) -> Declaration | None:
        """Look up a declaration by fully-qualified name."""
        if self._store is None:
            return None
        for decl in self._store._declarations.values():
            if decl.name == name:
                return decl
        return None

    # --- Incremental updates ---

    def add_primitive(self, primitive: AlgorithmicPrimitive) -> None:
        """Add a single primitive to the live index.

        Used during catalog seeding so the index stays current as new
        primitives are added.
        """
        self._ensure_embedder()
        text = self._primitive_to_text(primitive)
        vec = self._embedder.embed(text)
        decl = Declaration(
            name=primitive.name,
            type_signature=primitive.type_signature,
            docstring=primitive.description,
            source_lib=primitive.source,
            prover=Prover.LEAN4,
        )
        entry = IndexEntry(declaration=decl, embedding=vec, source_text=text)
        if self._store is None:
            self._store = FAISSStore(dim=self._embedder.dim)
        self._store.add([entry])
        idx = len(self._primitives)
        self._primitives.append(primitive)
        self._id_to_primitive[idx] = primitive

    # --- Original search returning AlgorithmicPrimitive ---

    def search(self, query: str, k: int = 10) -> list[AlgorithmicPrimitive]:
        """Semantic search over the primitive index.

        Returns up to k primitives sorted by descending similarity.
        """
        if self._store is None:
            return []

        self._ensure_embedder()
        query_vec = self._embedder.embed(query)
        results = self._store.search(query_vec, k=k)

        # Map Declaration results back to AlgorithmicPrimitive
        matched: list[AlgorithmicPrimitive] = []
        for decl, _score in results:
            # Find by name
            for prim in self._primitives:
                if prim.name == decl.name:
                    matched.append(prim)
                    break
        return matched

    def save(self, directory: str | Path | None = None) -> None:
        """Persist the skill index to disk."""

        directory = Path(directory) if directory else self._index_dir
        directory.mkdir(parents=True, exist_ok=True)

        if self._store is not None:
            self._store.save(directory)

        # Save the primitive list separately for reconstruction
        prims_data = [p.model_dump() for p in self._primitives]
        with open(directory / "primitives.json", "w") as f:
            json.dump(prims_data, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> SkillIndex:
        """Load a persisted skill index from disk."""

        directory = Path(directory)
        idx = cls(index_dir=directory)
        idx._store = FAISSStore.load(directory)

        prims_path = directory / "primitives.json"
        if prims_path.exists():
            with open(prims_path) as f:
                prims_data = json.load(f)
            idx._primitives = [
                AlgorithmicPrimitive.model_validate(p) for p in prims_data
            ]
            idx._id_to_primitive = {i: p for i, p in enumerate(idx._primitives)}

        return idx
