"""Tests for the Semantic Indexer components."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ageom.indexer.models import IndexEntry, IndexMetadata
from ageom.types import Declaration, Prover


class TestIndexEntry:
    def test_valid_entry(self):
        decl = Declaration(name="test", type_signature="Nat")
        entry = IndexEntry(
            declaration=decl,
            embedding=np.random.randn(768).astype(np.float32),
        )
        assert entry.embedding.shape == (768,)

    def test_invalid_embedding_shape(self):
        decl = Declaration(name="test", type_signature="Nat")
        with pytest.raises(ValueError, match="Expected 1-d"):
            IndexEntry(
                declaration=decl,
                embedding=np.random.randn(2, 768).astype(np.float32),
            )


class TestIndexMetadata:
    def test_creation(self):
        meta = IndexMetadata(
            num_entries=100,
            prover=Prover.LEAN4,
            source_lib="Mathlib",
            embedding_model="microsoft/unixcoder-base",
        )
        assert meta.num_entries == 100
        assert meta.created_at  # auto-set


class TestFAISSStore:
    """Tests for FAISSStore - requires faiss-cpu."""

    @pytest.fixture
    def store(self):
        faiss = pytest.importorskip("faiss")
        from ageom.indexer.faiss_store import FAISSStore

        return FAISSStore(dim=8)

    @pytest.fixture
    def sample_entries(self):
        entries = []
        for i in range(5):
            vec = np.random.randn(8).astype(np.float32)
            vec /= np.linalg.norm(vec)
            entries.append(
                IndexEntry(
                    declaration=Declaration(
                        name=f"decl_{i}",
                        type_signature=f"Type{i}",
                    ),
                    embedding=vec,
                )
            )
        return entries

    def test_add_and_search(self, store, sample_entries):
        store.add(sample_entries)
        assert store.size == 5

        query = sample_entries[0].embedding
        results = store.search(query, k=3)
        assert len(results) == 3
        # First result should be the query itself (exact match)
        assert results[0][0].name == "decl_0"
        assert results[0][1] > 0.99  # cosine sim ~1.0

    def test_search_ordering(self, store, sample_entries):
        store.add(sample_entries)
        query = sample_entries[2].embedding
        results = store.search(query, k=5)
        # Scores should be descending
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_store(self, store):
        results = store.search(np.random.randn(8).astype(np.float32), k=5)
        assert results == []

    def test_save_and_load(self, store, sample_entries):
        store.add(sample_entries)
        store.set_metadata(
            IndexMetadata(
                num_entries=5,
                prover=Prover.LEAN4,
                source_lib="test",
                embedding_model="test-model",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)

            from ageom.indexer.faiss_store import FAISSStore

            loaded = FAISSStore.load(tmpdir)
            assert loaded.size == 5

            # Search should work on loaded store
            query = sample_entries[0].embedding
            results = loaded.search(query, k=1)
            assert results[0][0].name == "decl_0"


class TestUniXcoderEmbedder:
    """Tests for the embedder - requires transformers + torch."""

    @pytest.fixture
    def embedder(self):
        pytest.importorskip("transformers")
        pytest.importorskip("torch")
        from ageom.indexer.embedder import UniXcoderEmbedder

        return UniXcoderEmbedder()

    @pytest.mark.slow
    def test_embed_produces_768_dim(self, embedder):
        vec = embedder.embed("theorem Nat.add_comm : forall n m, n + m = m + n")
        assert vec.shape == (768,)

    @pytest.mark.slow
    def test_embed_l2_normalized(self, embedder):
        vec = embedder.embed("test input")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    @pytest.mark.slow
    def test_similar_code_higher_similarity(self, embedder):
        v1 = embedder.embed("Nat.add_comm : forall n m, n + m = m + n")
        v2 = embedder.embed("addition is commutative for natural numbers")
        v3 = embedder.embed("def quicksort(arr): return sorted(arr)")
        # v1 and v2 should be more similar than v1 and v3
        sim_12 = float(np.dot(v1, v2))
        sim_13 = float(np.dot(v1, v3))
        assert sim_12 > sim_13

    @pytest.mark.slow
    def test_embed_batch(self, embedder):
        texts = ["text one", "text two", "text three"]
        vecs = embedder.embed_batch(texts)
        assert vecs.shape == (3, 768)
        # Each should be normalized
        for i in range(3):
            norm = np.linalg.norm(vecs[i])
            assert abs(norm - 1.0) < 1e-5

    @pytest.mark.slow
    def test_embed_declaration(self, embedder):
        decl = Declaration(
            name="Nat.add_comm",
            type_signature="∀ (n m : ℕ), n + m = m + n",
            docstring="Commutativity of addition",
        )
        vec = embedder.embed_declaration(decl)
        assert vec.shape == (768,)
