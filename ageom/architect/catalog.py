"""Primitive catalog — the searchable 'alphabet' of known algorithmic operations."""

from __future__ import annotations

import json
from pathlib import Path

from ageom.architect.models import AlgorithmicNode, AlgorithmicPrimitive, ConceptType


class PrimitiveCatalog:
    """In-memory catalog of algorithmic primitives with category indexing.

    Acts as the stop-condition oracle: if a CDG node matches a primitive,
    the Decomposer marks it as atomic.
    """

    def __init__(self) -> None:
        self._primitives: dict[str, AlgorithmicPrimitive] = {}
        self._by_category: dict[ConceptType, list[AlgorithmicPrimitive]] = {}

    @property
    def size(self) -> int:
        return len(self._primitives)

    def add(self, primitive: AlgorithmicPrimitive) -> None:
        """Add a primitive to the catalog."""
        self._primitives[primitive.name] = primitive
        self._by_category.setdefault(primitive.category, []).append(primitive)

    def get(self, name: str) -> AlgorithmicPrimitive | None:
        """Look up a primitive by name."""
        return self._primitives.get(name)

    def search_by_category(
        self, concept_type: ConceptType
    ) -> list[AlgorithmicPrimitive]:
        """Return all primitives in a given category."""
        return list(self._by_category.get(concept_type, []))

    def all_primitives(self) -> list[AlgorithmicPrimitive]:
        """Return all primitives in the catalog."""
        return list(self._primitives.values())

    def is_atomic(self, node: AlgorithmicNode) -> bool:
        """Check if a node matches any known primitive by name."""
        if node.matched_primitive and node.matched_primitive in self._primitives:
            return True
        # Also try matching by node name (case-insensitive)
        name_lower = node.name.lower().replace(" ", "_")
        for prim_name in self._primitives:
            if prim_name.lower().replace(" ", "_") == name_lower:
                return True
        return False

    def find_matching_primitives(
        self, node: AlgorithmicNode, k: int = 5
    ) -> list[AlgorithmicPrimitive]:
        """Find primitives matching a node by category, then by keyword overlap.

        For full semantic search, use SkillIndex.search() instead.
        """
        # First: exact category matches
        category_matches = self.search_by_category(node.concept_type)

        # Score by keyword overlap between node description and primitive description
        node_words = set(node.description.lower().split())
        scored: list[tuple[float, AlgorithmicPrimitive]] = []
        for prim in category_matches:
            prim_words = set(prim.description.lower().split())
            overlap = len(node_words & prim_words)
            scored.append((overlap, prim))

        # If not enough from same category, add cross-category matches
        if len(scored) < k:
            for prim in self._primitives.values():
                if prim.category != node.concept_type:
                    prim_words = set(prim.description.lower().split())
                    overlap = len(node_words & prim_words)
                    scored.append((overlap, prim))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [prim for _, prim in scored[:k]]

    def save(self, path: str | Path) -> None:
        """Persist the catalog to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [prim.model_dump() for prim in self._primitives.values()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PrimitiveCatalog:
        """Load a catalog from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        catalog = cls()
        for item in data:
            catalog.add(AlgorithmicPrimitive.model_validate(item))
        return catalog
