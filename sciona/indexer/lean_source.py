"""Declaration source for Lean 4 / Mathlib via lean-explore."""

from __future__ import annotations

from sciona.types import Declaration, Prover


class LeanDeclarationSource:
    """Extracts declarations from Lean 4 / Mathlib using lean-explore.

    Requires the `lean-explore` package with local data:
        pip install lean-explore[local]
        lean-explore data fetch
    """

    def __init__(self) -> None:
        from lean_explore.local import Service

        self._service = Service()

    def get_all_declarations(self, library: str = "Mathlib") -> list[Declaration]:
        """Enumerate all declarations from the specified library."""
        results = self._service.search(query="", limit=100_000)
        declarations: list[Declaration] = []
        for item in results:
            decl = Declaration(
                name=item.name,
                type_signature=getattr(item, "type", ""),
                docstring=getattr(item, "docstring", "") or "",
                source_lib=library,
                prover=Prover.LEAN4,
                raw_code=getattr(item, "code", "") or "",
            )
            declarations.append(decl)
        return declarations

    def search_by_type(self, type_sig: str, k: int = 10) -> list[Declaration]:
        """Search using lean-explore's hybrid type search."""
        results = self._service.search(query=type_sig, limit=k)
        return [
            Declaration(
                name=item.name,
                type_signature=getattr(item, "type", ""),
                docstring=getattr(item, "docstring", "") or "",
                source_lib="Mathlib",
                prover=Prover.LEAN4,
                raw_code=getattr(item, "code", "") or "",
            )
            for item in results
        ]
