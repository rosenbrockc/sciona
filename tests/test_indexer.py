"""Tests for the Semantic Indexer components."""

from __future__ import annotations

import importlib.util
import platform
import sys
import types
import tempfile

import numpy as np
import pytest

from sciona.indexer.models import IndexEntry, IndexMetadata
from sciona.types import Declaration, Prover


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
            embedding_backend="unixcoder",
        )
        assert meta.num_entries == 100
        assert meta.created_at  # auto-set


class TestFAISSStore:
    """Tests for FAISSStore - requires faiss-cpu."""

    @pytest.fixture
    def store(self):
        pytest.importorskip("faiss")
        from sciona.indexer.faiss_store import FAISSStore

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
                embedding_backend="fastembed",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir)

            from sciona.indexer.faiss_store import FAISSStore

            loaded = FAISSStore.load(tmpdir)
            assert loaded.size == 5
            assert loaded._metadata is not None
            assert loaded._metadata.embedding_backend == "fastembed"

            # Search should work on loaded store
            query = sample_entries[0].embedding
            results = loaded.search(query, k=1)
            assert results[0][0].name == "decl_0"


class TestEmbedderFactory:
    def test_create_fastembed_embedder_uses_local_fastembed_backend(self, monkeypatch):
        import sciona.indexer.embedder as embedder_mod

        class _FakeTextEmbedding:
            def __init__(self, model_name: str):
                self.model_name = model_name

            def embed(self, texts, batch_size: int = 32):
                del batch_size
                for text in texts:
                    base = float(len(text))
                    yield np.array([base, 0.0, 4.0], dtype=np.float32)

        monkeypatch.setitem(
            sys.modules,
            "fastembed",
            types.SimpleNamespace(TextEmbedding=_FakeTextEmbedding),
        )

        embedder = embedder_mod.create_embedder("fastembed", "fake-fastembed-model")

        assert embedder.backend == "fastembed"
        assert embedder.model_name == "fake-fastembed-model"
        vec = embedder.embed("abc")
        assert vec.shape == (3,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-6

    def test_create_unixcoder_embedder_uses_guard_before_transformers_import(self, monkeypatch):
        import sciona.indexer.embedder as embedder_mod

        order: list[str] = []

        def _record_juliacall() -> None:
            order.append("juliacall")

        class _FakeTokenizer:
            @staticmethod
            def from_pretrained(_model_name):
                order.append("tokenizer")
                return object()

        class _FakeModel:
            @staticmethod
            def from_pretrained(_model_name):
                order.append("model")

                class _Loaded:
                    def eval(self):
                        order.append("eval")

                return _Loaded()

        fake_transformers = types.SimpleNamespace(
            AutoTokenizer=_FakeTokenizer,
            AutoModel=_FakeModel,
        )

        monkeypatch.setattr(
            embedder_mod,
            "_prefer_juliacall_before_torch",
            _record_juliacall,
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        embedder = embedder_mod.create_embedder("unixcoder", "fake-model")

        assert embedder.dim == 768
        assert order == ["juliacall", "tokenizer", "model", "eval"]


class TestUniXcoderEmbedder:
    """Tests for the unixcoder embedder - requires transformers + torch."""

    @pytest.fixture
    def embedder(self):
        if importlib.util.find_spec("transformers") is None:
            pytest.skip("transformers not installed")
        if importlib.util.find_spec("torch") is None:
            pytest.skip("torch not installed")
        if importlib.util.find_spec("juliacall") is not None and platform.system() == "Darwin":
            pytest.skip(
                "real UniXcoder integration is unstable with juliacall+torch on macOS; "
                "covered by import-order unit test and end-to-end runtime checks"
            )
        from sciona.indexer.embedder import UniXcoderEmbedder

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
