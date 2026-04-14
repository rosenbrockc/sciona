"""Index builder and SemanticIndex implementation."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sciona.indexer.embedder import (
    DEFAULT_EMBEDDING_BACKEND,
    Embedder,
    create_embedder,
)
from sciona.indexer.faiss_store import FAISSStore
from sciona.indexer.models import IndexEntry, IndexMetadata
from sciona.types import Declaration, Prover


def _row_get(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _open_manifest_sqlite(manifest_path: str | Path) -> sqlite3.Connection:
    path = Path(manifest_path).expanduser()
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def _fetch_table_rows(con: sqlite3.Connection, table: str) -> list[sqlite3.Row]:
    try:
        con.row_factory = sqlite3.Row
        return list(con.execute(f"SELECT * FROM {table}"))
    except sqlite3.OperationalError:
        return []


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _manifest_source_lib(atom_row: sqlite3.Row) -> str:
    for key in ("source_repo_id", "source_package", "namespace_root", "fqdn"):
        value = _normalize_text(_row_get(atom_row, key, ""))
        if value:
            return f"manifest:{value}"
    return "manifest:unknown"


def _best_description(atom_id: str, descriptions: Iterable[sqlite3.Row]) -> str:
    rows = [
        row
        for row in descriptions
        if _normalize_text(_row_get(row, "atom_id", "")) == atom_id
        and _normalize_text(_row_get(row, "content", ""))
    ]
    if not rows:
        return ""

    preferred_kind = {
        "dejargonized": 0,
        "technical": 1,
        "summary": 2,
        "description": 3,
    }

    def _sort_key(row: sqlite3.Row) -> tuple[Any, ...]:
        kind = _normalize_text(_row_get(row, "kind", "")).lower()
        content = _normalize_text(_row_get(row, "content", ""))
        jargon_score = float(_row_get(row, "jargon_score", 1.0) or 1.0)
        reviewed = int(bool(_row_get(row, "reviewed", 0)))
        updated_at = _normalize_text(_row_get(row, "updated_at", ""))
        return (
            -reviewed,
            preferred_kind.get(kind, 4),
            jargon_score,
            -len(content),
            updated_at,
            _normalize_text(_row_get(row, "description_id", "")),
        )

    return _normalize_text(_row_get(sorted(rows, key=_sort_key)[0], "content", ""))


def _port_from_io_spec(
    row: sqlite3.Row,
) -> tuple[str, str, str, str, bool, str]:
    name = _normalize_text(
        _row_get(row, "port_name", _row_get(row, "name", ""))
    )
    type_desc = _normalize_text(_row_get(row, "type_desc", "")) or "Any"
    constraints = _normalize_text(_row_get(row, "constraints", ""))
    data_kind = _normalize_text(_row_get(row, "data_kind", ""))
    required = bool(_row_get(row, "required", 1))
    default_value_repr = _normalize_text(_row_get(row, "default_value_repr", ""))
    return name, type_desc, constraints, data_kind, required, default_value_repr


def _io_spec_sort_key(row: sqlite3.Row) -> tuple[int, str]:
    ordinal = int(_row_get(row, "ordinal", 0) or 0)
    name = _normalize_text(_row_get(row, "port_name", _row_get(row, "name", "")))
    return ordinal, name


def _format_port_signature(
    name: str,
    type_desc: str,
    constraints: str,
    data_kind: str,
    required: bool,
    default_value_repr: str,
) -> str:
    rendered = name
    if not required:
        rendered += "?"
    rendered += f": {type_desc or 'Any'}"
    extras: list[str] = []
    if constraints:
        extras.append(f"constraints={constraints}")
    if data_kind:
        extras.append(f"kind={data_kind}")
    if not required and default_value_repr:
        extras.append(f"default={default_value_repr}")
    if extras:
        rendered += " [" + ", ".join(extras) + "]"
    return rendered


def _type_signature_from_ports(
    inputs: list[tuple[str, str, str, str, bool, str]],
    outputs: list[tuple[str, str, str, str, bool, str]],
) -> str:
    input_sig = ", ".join(_format_port_signature(*port) for port in inputs) or "void"
    if not outputs:
        output_sig = "void"
    elif len(outputs) == 1:
        output_sig = _format_port_signature(*outputs[0])
    else:
        output_sig = "tuple[" + ", ".join(
            _format_port_signature(*port) for port in outputs
        ) + "]"
    return f"({input_sig}) -> {output_sig}"


def _manifest_decl_from_atom(
    atom_row: sqlite3.Row,
    descriptions: Iterable[sqlite3.Row],
    io_specs: Iterable[sqlite3.Row],
) -> Declaration | None:
    atom_id = _normalize_text(_row_get(atom_row, "atom_id", ""))
    fqdn = _normalize_text(_row_get(atom_row, "fqdn", ""))
    if not atom_id or not fqdn:
        return None

    ports = [
        row
        for row in io_specs
        if _normalize_text(_row_get(row, "atom_id", "")) == atom_id
    ]
    inputs = [
        _port_from_io_spec(row)
        for row in sorted(
            [
                row
                for row in ports
                if _normalize_text(_row_get(row, "direction", "")).lower()
                in {"input", "in"}
            ],
            key=_io_spec_sort_key,
        )
    ]
    outputs = [
        _port_from_io_spec(row)
        for row in sorted(
            [
                row
                for row in ports
                if _normalize_text(_row_get(row, "direction", "")).lower()
                in {"output", "out"}
            ],
            key=_io_spec_sort_key,
        )
    ]

    docstring = _best_description(atom_id, descriptions)
    atom_description = _normalize_text(_row_get(atom_row, "description", ""))
    conceptual_parts = [
        _normalize_text(_row_get(atom_row, "domain_tags", "")),
        _normalize_text(_row_get(atom_row, "visibility_tier", "")),
    ]
    if atom_description and atom_description != docstring:
        conceptual_parts.append(atom_description)
    if inputs or outputs:
        conceptual_parts.append(_type_signature_from_ports(inputs, outputs))

    return Declaration(
        name=fqdn,
        type_signature=_type_signature_from_ports(inputs, outputs),
        docstring=docstring or atom_description,
        conceptual_summary="; ".join(part for part in conceptual_parts if part),
        source_lib=_manifest_source_lib(atom_row),
        prover=Prover.PYTHON,
    )


class IndexBuilder:
    """Orchestrates the source -> embed -> store pipeline."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: FAISSStore | None = None,
        embedding_backend: str = DEFAULT_EMBEDDING_BACKEND,
        embedding_model: str | None = None,
    ) -> None:
        self._embedder = embedder or create_embedder(
            backend=embedding_backend,
            model_name=embedding_model,
        )
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
                embedding_model=self._embedder.model_name,
                embedding_backend=self._embedder.backend,
            )
        )
        return self._store


def build_index_from_manifest_sqlite(
    manifest_path: str | Path,
    embedder: Embedder | None = None,
    existing_store: FAISSStore | None = None,
) -> FAISSStore:
    """Build a FAISS store from a manifest.sqlite declaration snapshot."""
    with _open_manifest_sqlite(manifest_path) as con:
        atoms = _fetch_table_rows(con, "atoms")
        descriptions = _fetch_table_rows(con, "descriptions")
        io_specs = _fetch_table_rows(con, "io_specs")

    declarations: list[Declaration] = []
    for atom_row in atoms:
        status = _normalize_text(_row_get(atom_row, "status", "approved")).lower()
        if status and status != "approved":
            continue
        decl = _manifest_decl_from_atom(atom_row, descriptions, io_specs)
        if decl is not None:
            declarations.append(decl)

    if embedder is None:
        if existing_store is not None and existing_store._metadata is not None:
            embedder = create_embedder(
                backend=existing_store._metadata.embedding_backend,
                model_name=existing_store._metadata.embedding_model,
            )
        else:
            embedder = create_embedder()

    if existing_store is not None:
        store = existing_store
        if store._dim != embedder.dim:
            raise ValueError(
                "manifest embedder dimension does not match the existing FAISS store"
            )
        if store._metadata is not None and (
            store._metadata.embedding_backend != embedder.backend
            or store._metadata.embedding_model != embedder.model_name
        ):
            raise ValueError(
                "manifest embedder does not match the existing FAISS store embedding space"
            )
    else:
        store = FAISSStore(dim=embedder.dim)

    if not declarations:
        if existing_store is None:
            store.set_metadata(
                IndexMetadata(
                    num_entries=0,
                    prover=Prover.PYTHON,
                    source_lib=f"manifest:{Path(manifest_path).name}",
                    embedding_model=embedder.model_name,
                    embedding_backend=embedder.backend,
                )
            )
        return store

    texts: list[str] = []
    for decl in declarations:
        text = f"{decl.name} : {decl.type_signature}"
        if decl.docstring:
            text += f"\n{decl.docstring}"
        if decl.conceptual_summary:
            text += f"\n{decl.conceptual_summary}"
        texts.append(text)

    embeddings = embedder.embed_batch(texts)
    entries = [
        IndexEntry(declaration=decl, embedding=embeddings[i], source_text=texts[i])
        for i, decl in enumerate(declarations)
    ]
    store.add(entries)

    if existing_store is None:
        store.set_metadata(
            IndexMetadata(
                num_entries=len(entries),
                prover=Prover.PYTHON,
                source_lib=f"manifest:{Path(manifest_path).name}",
                embedding_model=embedder.model_name,
                embedding_backend=embedder.backend,
            )
        )

    return store


class SemanticIndexImpl:
    """Concrete implementation of the SemanticIndex protocol.

    Combines FAISS vector search with optional lean-explore type search.
    """

    def __init__(
        self,
        store: FAISSStore,
        embedder: Embedder,
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

    def search_by_type(self, type_signature: str, k: int = 10) -> list[Declaration]:
        """Search by type signature.

        For Lean: delegates to lean-explore's hybrid search.
        For Coq / no lean source: falls back to embedding search.
        """
        if self._lean_source is not None:
            from sciona.indexer.lean_source import LeanDeclarationSource

            if isinstance(self._lean_source, LeanDeclarationSource):
                return self._lean_source.search_by_type(type_signature, k=k)

        # Fallback: embed the type signature and search
        results = self.search_by_embedding(type_signature, k=k)
        return [decl for decl, _score in results]

    def get_declaration(self, name: str) -> Declaration | None:
        """Look up a declaration by fully-qualified name."""
        return self._by_name.get(name)
