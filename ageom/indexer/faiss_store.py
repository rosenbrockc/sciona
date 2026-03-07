"""FAISS-based vector store for declaration embeddings."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import msgpack
import numpy as np

from ageom.indexer.models import IndexEntry, IndexMetadata
from ageom.types import Declaration, Prover


def _declarations_to_msgpack(declarations: dict[int, Declaration]) -> bytes:
    """Serialize declarations dict to msgpack bytes."""
    data = {
        k: {
            "name": d.name,
            "type_signature": d.type_signature,
            "docstring": d.docstring,
            "conceptual_summary": d.conceptual_summary,
            "source_lib": d.source_lib,
            "prover": d.prover.value,
            "raw_code": d.raw_code,
        }
        for k, d in declarations.items()
    }
    return msgpack.packb(data, use_bin_type=True)


def _declarations_from_msgpack(raw: bytes) -> dict[int, Declaration]:
    """Deserialize declarations dict from msgpack bytes."""
    data = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    return {
        int(k): Declaration(
            name=d["name"],
            type_signature=d.get("type_signature", ""),
            docstring=d.get("docstring", ""),
            conceptual_summary=d.get("conceptual_summary", ""),
            source_lib=d.get("source_lib", ""),
            prover=Prover(d.get("prover", "lean4")),
            raw_code=d.get("raw_code", ""),
        )
        for k, d in data.items()
    }


class FAISSStore:
    """Vector store using FAISS IndexFlatIP (inner product on normalized vectors = cosine similarity).

    Uses IndexIDMap to map FAISS internal IDs to our integer keys,
    with a separate dict for ID-to-Declaration metadata.
    """

    def __init__(self, dim: int = 768) -> None:
        import faiss

        self._dim = dim
        base_index = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIDMap(base_index)
        self._declarations: dict[int, Declaration] = {}
        self._next_id = 0
        self._metadata: IndexMetadata | None = None

    @property
    def size(self) -> int:
        return self._index.ntotal

    def add(self, entries: list[IndexEntry]) -> None:
        """Add entries to the index."""

        if not entries:
            return
        ids = []
        vecs = []
        for entry in entries:
            entry_id = self._next_id
            self._next_id += 1
            self._declarations[entry_id] = entry.declaration
            ids.append(entry_id)
            vecs.append(entry.embedding)

        id_array = np.array(ids, dtype=np.int64)
        vec_array = np.ascontiguousarray(np.vstack(vecs), dtype=np.float32)
        self._index.add_with_ids(vec_array, id_array)

    def search(
        self, query_vec: np.ndarray, k: int = 10
    ) -> list[tuple[Declaration, float]]:
        """Search for the k most similar declarations.

        Returns list of (Declaration, score) sorted by descending similarity.
        """
        query = np.ascontiguousarray(query_vec.reshape(1, -1), dtype=np.float32)
        k = min(k, self.size) if self.size > 0 else 0
        if k == 0:
            return []
        scores, ids = self._index.search(query, k)
        results: list[tuple[Declaration, float]] = []
        for score, entry_id in zip(scores[0], ids[0]):
            if entry_id == -1:
                continue
            decl = self._declarations.get(int(entry_id))
            if decl is not None:
                results.append((decl, float(score)))
        return results

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and metadata to disk."""
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "index.faiss"))

        with open(directory / "declarations.msgpack", "wb") as f:
            f.write(_declarations_to_msgpack(self._declarations))

        meta = {
            "next_id": self._next_id,
            "dim": self._dim,
        }
        if self._metadata:
            meta["metadata"] = {
                "num_entries": self._metadata.num_entries,
                "prover": self._metadata.prover.value,
                "source_lib": self._metadata.source_lib,
                "embedding_model": self._metadata.embedding_model,
                "embedding_backend": self._metadata.embedding_backend,
                "created_at": self._metadata.created_at,
            }
        with open(directory / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> FAISSStore:
        """Load a persisted index from disk."""
        import faiss

        directory = Path(directory)

        store = cls.__new__(cls)
        store._index = faiss.read_index(str(directory / "index.faiss"))
        store._dim = store._index.d

        msgpack_path = directory / "declarations.msgpack"
        pkl_path = directory / "declarations.pkl"
        if msgpack_path.exists():
            with open(msgpack_path, "rb") as f:
                store._declarations = _declarations_from_msgpack(f.read())
        elif pkl_path.exists():
            # Legacy fallback -- warn and migrate
            import pickle

            warnings.warn(
                "Loading legacy pickle index. Re-run 'ageom index build' to migrate to msgpack.",
                DeprecationWarning,
                stacklevel=2,
            )
            with open(pkl_path, "rb") as f:
                store._declarations = pickle.load(f)  # noqa: S301
        else:
            store._declarations = {}

        with open(directory / "meta.json") as f:
            meta = json.load(f)
        store._next_id = meta["next_id"]

        if "metadata" in meta:
            md = meta["metadata"]
            store._metadata = IndexMetadata(
                num_entries=md["num_entries"],
                prover=Prover(md["prover"]),
                source_lib=md["source_lib"],
                embedding_model=md["embedding_model"],
                embedding_backend=md.get("embedding_backend", "unixcoder"),
                created_at=md["created_at"],
            )
        else:
            store._metadata = None

        return store

    def set_metadata(self, metadata: IndexMetadata) -> None:
        self._metadata = metadata
