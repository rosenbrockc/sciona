from __future__ import annotations

import numpy as np
import pytest

from sciona.hunter.embedding_query_expander import EmbeddingQueryExpander
from sciona.types import Declaration, Prover


class _FakeEmbedder:
    """Fake embedder using token-overlap heuristic for deterministic testing."""

    backend = "fake"
    model_name = "fake"

    @property
    def dim(self) -> int:
        return 8

    def _vec(self, text: str) -> np.ndarray:
        tokens = set(text.lower().split())
        # Simple bag-of-words hash into a fixed-dim vector
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            idx = hash(token) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.array([self._vec(t) for t in texts])

    def embed_declaration(self, decl: Declaration) -> np.ndarray:
        return self._vec(f"{decl.name} {decl.type_signature} {decl.docstring}")


def _make_declarations() -> list[Declaration]:
    return [
        Declaration(
            name="bandpass_filter",
            type_signature="signal -> signal",
            docstring="Apply a bandpass filter to isolate a frequency band",
            prover=Prover.PYTHON,
        ),
        Declaration(
            name="r_peak_detection",
            type_signature="signal -> list[int]",
            docstring="Detect R-peaks in an ECG signal using Hamilton method",
            prover=Prover.PYTHON,
        ),
        Declaration(
            name="dijkstra",
            type_signature="graph -> node -> distances",
            docstring="Compute shortest path distances using Dijkstra algorithm on a weighted graph",
            prover=Prover.PYTHON,
        ),
        Declaration(
            name="relax_edges",
            type_signature="graph -> distances -> distances",
            docstring="Relax all edges in a weighted graph for shortest path computation",
            prover=Prover.PYTHON,
        ),
        Declaration(
            name="heart_rate_computation",
            type_signature="list[int] -> float",
            docstring="Compute heart rate from detected R-peak intervals",
            prover=Prover.PYTHON,
        ),
    ]


def test_expand_ecg_domain():
    expander = EmbeddingQueryExpander(_FakeEmbedder(), _make_declarations())
    queries = expander.expand("filter ECG signal bandpass cardiac frequency")
    assert len(queries) > 0
    all_text = " ".join(queries).lower()
    assert any(
        term in all_text for term in ("bandpass", "filter", "r_peak", "heart_rate")
    )


def test_expand_graph_domain():
    expander = EmbeddingQueryExpander(_FakeEmbedder(), _make_declarations())
    queries = expander.expand("shortest path distances weighted graph")
    assert len(queries) > 0
    all_text = " ".join(queries).lower()
    assert any(term in all_text for term in ("dijkstra", "relax", "shortest", "path"))


def test_expand_empty_declarations():
    expander = EmbeddingQueryExpander(_FakeEmbedder(), [])
    assert expander.expand("some query text") == []


def test_expand_empty_text():
    expander = EmbeddingQueryExpander(_FakeEmbedder(), _make_declarations())
    assert expander.expand("") == []
    assert expander.expand("   ") == []


def test_expand_respects_max_queries():
    expander = EmbeddingQueryExpander(_FakeEmbedder(), _make_declarations())
    queries = expander.expand("filter signal bandpass", max_queries=2)
    assert len(queries) <= 2
