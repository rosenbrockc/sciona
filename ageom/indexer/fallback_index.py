"""FAISS-free fallback semantic index for environments without faiss.

Loads declarations from index artifacts and performs lightweight lexical
matching so Hunter can continue running when FAISS is unavailable.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import msgpack

from ageom.types import Declaration, Prover


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9_]+", text.lower()) if len(tok) >= 2}


class LexicalSemanticIndex:
    """Minimal SemanticIndex implementation using lexical overlap scoring."""

    def __init__(self, declarations: list[Declaration]) -> None:
        self._declarations = declarations
        self._by_name: dict[str, Declaration] = {decl.name: decl for decl in declarations}
        self._decl_tokens: dict[str, set[str]] = {}
        self._type_tokens: dict[str, set[str]] = {}

        for decl in declarations:
            full_text = " ".join(
                [
                    decl.name,
                    decl.type_signature,
                    decl.docstring,
                    decl.conceptual_summary,
                ]
            )
            self._decl_tokens[decl.name] = _tokenize(full_text)
            self._type_tokens[decl.name] = _tokenize(decl.type_signature)

    @classmethod
    def load(cls, directory: str | Path) -> LexicalSemanticIndex:
        """Load declarations from index files in *directory*."""
        directory = Path(directory)
        msgpack_path = directory / "declarations.msgpack"
        pkl_path = directory / "declarations.pkl"

        declarations: list[Declaration] = []
        if msgpack_path.exists():
            raw = msgpack_path.read_bytes()
            data = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            for row in data.values():
                if not isinstance(row, dict):
                    continue
                prover_raw = str(row.get("prover", "lean4"))
                try:
                    prover = Prover(prover_raw)
                except ValueError:
                    prover = Prover.LEAN4
                declarations.append(
                    Declaration(
                        name=str(row.get("name", "")),
                        type_signature=str(row.get("type_signature", "")),
                        docstring=str(row.get("docstring", "")),
                        conceptual_summary=str(row.get("conceptual_summary", "")),
                        source_lib=str(row.get("source_lib", "")),
                        prover=prover,
                        raw_code=str(row.get("raw_code", "")),
                    )
                )
        elif pkl_path.exists():
            with open(pkl_path, "rb") as f:
                loaded = pickle.load(f)  # noqa: S301
            if isinstance(loaded, dict):
                for row in loaded.values():
                    if isinstance(row, Declaration):
                        declarations.append(row)
            if declarations:
                msgpack_path.write_bytes(msgpack.packb(_declarations_to_msgpack(declarations), use_bin_type=True))
        else:
            raise FileNotFoundError(
                f"No declarations file found in {directory} "
                "(expected declarations.msgpack or declarations.pkl)."
            )

        return cls(declarations)

    def _rank(
        self,
        query: str,
        *,
        k: int,
        use_type_bias: bool,
    ) -> list[tuple[Declaration, float]]:
        if k <= 0 or not self._declarations:
            return []

        query_lower = query.lower().strip()
        query_tokens = _tokenize(query)
        rows: list[tuple[float, Declaration]] = []

        for decl in self._declarations:
            tokens = self._decl_tokens.get(decl.name, set())
            overlap = len(query_tokens & tokens) if query_tokens else 0
            score = float(overlap)

            if query_tokens:
                score += overlap / max(1.0, float(len(query_tokens)))

            name_lower = decl.name.lower()
            if query_lower and query_lower in name_lower:
                score += 4.0

            if use_type_bias:
                t_tokens = self._type_tokens.get(decl.name, set())
                t_overlap = len(query_tokens & t_tokens) if query_tokens else 0
                score += 2.0 * float(t_overlap)

            if score > 0.0:
                rows.append((score, decl))

        if not rows:
            # Keep Hunter functional even if lexical overlap misses everything.
            return [(decl, 0.0) for decl in self._declarations[:k]]

        rows.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        return [(decl, score) for score, decl in rows[:k]]

    def search_by_embedding(
        self, query_text: str, k: int = 10
    ) -> list[tuple[Declaration, float]]:
        return self._rank(query_text, k=k, use_type_bias=False)

    def search_by_type(self, type_signature: str, k: int = 10) -> list[Declaration]:
        ranked = self._rank(type_signature, k=k, use_type_bias=True)
        return [decl for decl, _score in ranked]

    def get_declaration(self, name: str) -> Declaration | None:
        return self._by_name.get(name)


def _declarations_to_msgpack(
    declarations: list[Declaration],
) -> dict[int, dict[str, str]]:
    """Serialize declarations into the msgpack layout used by lexical fallback."""
    rows: dict[int, dict[str, str]] = {}
    for idx, decl in enumerate(declarations):
        rows[idx] = {
            "name": decl.name,
            "type_signature": decl.type_signature,
            "docstring": decl.docstring,
            "conceptual_summary": decl.conceptual_summary,
            "source_lib": decl.source_lib,
            "prover": decl.prover.value,
            "raw_code": decl.raw_code,
        }
    return rows
