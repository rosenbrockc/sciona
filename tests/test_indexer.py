"""Tests for the Semantic Indexer components."""

from __future__ import annotations

import importlib.util
import sqlite3
import platform
import sys
import types
import tempfile

import numpy as np
import pytest

from sciona.indexer.models import IndexEntry, IndexMetadata
from sciona.types import Declaration, Prover


class _FakeManifestEmbedder:
    backend = "fastembed"
    model_name = "fake-model"

    def __init__(self) -> None:
        self.dim = 4
        self._vectors_by_name: dict[str, np.ndarray] = {}

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        del batch_size
        vectors: list[np.ndarray] = []
        for idx, text in enumerate(texts):
            vec = np.zeros(self.dim, dtype=np.float32)
            vec[idx % self.dim] = 1.0
            vectors.append(vec)
            name = text.split(" : ", 1)[0].splitlines()[0]
            self._vectors_by_name[name] = vec
        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(vectors)

    def embed(self, text: str) -> np.ndarray:
        for name, vec in self._vectors_by_name.items():
            if name in text:
                return vec
        return np.zeros(self.dim, dtype=np.float32)


class _StubSemanticIndex:
    def __init__(
        self,
        *,
        hits_by_query: dict[str, list[tuple[Declaration, float]]],
        declarations: dict[str, Declaration] | None = None,
    ) -> None:
        self._hits_by_query = hits_by_query
        self._declarations = declarations or {}

    def search_by_embedding(
        self, query_text: str, k: int = 10
    ) -> list[tuple[Declaration, float]]:
        return list(self._hits_by_query.get(query_text, []))[:k]

    def search_by_type(self, type_signature: str, k: int = 10) -> list[Declaration]:
        return [decl for decl, _score in self.search_by_embedding(type_signature, k=k)]

    def get_declaration(self, name: str) -> Declaration | None:
        return self._declarations.get(name)


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


class TestManifestIndexBuilder:
    def test_build_index_from_manifest_sqlite_uses_descriptions_and_io_specs(
        self, tmp_path
    ):
        pytest.importorskip("faiss")
        from sciona.indexer.builder import build_index_from_manifest_sqlite

        db_path = tmp_path / "manifest.sqlite"
        con = sqlite3.connect(str(db_path))
        con.executescript(
            """
            CREATE TABLE atoms (
                atom_id TEXT PRIMARY KEY,
                fqdn TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL DEFAULT 'approved',
                domain_tags TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                visibility_tier TEXT NOT NULL DEFAULT 'general',
                source_kind TEXT NOT NULL DEFAULT 'hand_written',
                stateful_kind TEXT NOT NULL DEFAULT 'none',
                is_stochastic INTEGER NOT NULL DEFAULT 0,
                is_ffi INTEGER NOT NULL DEFAULT 0,
                namespace_root TEXT NOT NULL DEFAULT 'sciona.atoms',
                namespace_path TEXT NOT NULL DEFAULT '',
                source_repo_id TEXT NOT NULL DEFAULT '',
                source_package TEXT NOT NULL DEFAULT '',
                source_module_path TEXT NOT NULL DEFAULT '',
                source_symbol TEXT NOT NULL DEFAULT '',
                is_publishable INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE descriptions (
                description_id TEXT PRIMARY KEY,
                atom_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT 'en',
                generated_by TEXT NOT NULL DEFAULT '',
                reviewed INTEGER NOT NULL DEFAULT 0,
                jargon_score REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE io_specs (
                atom_id TEXT NOT NULL,
                port_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                type_desc TEXT NOT NULL DEFAULT '',
                constraints TEXT NOT NULL DEFAULT '',
                data_kind TEXT NOT NULL DEFAULT '',
                required INTEGER NOT NULL DEFAULT 1,
                default_value_repr TEXT NOT NULL DEFAULT '',
                ordinal INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        con.execute(
            """
            INSERT INTO atoms (atom_id, fqdn, status, domain_tags, description, source_repo_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "a1",
                "pkg.filter",
                "approved",
                "signal,analysis",
                "Technical fallback description",
                "repo-123",
            ),
        )
        con.execute(
            """
            INSERT INTO atoms (atom_id, fqdn, status, description, source_repo_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("a2", "pkg.skipme", "draft", "Should not index", "repo-999"),
        )
        con.executemany(
            """
            INSERT INTO descriptions (
                description_id, atom_id, kind, content, reviewed, jargon_score, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("d1", "a1", "technical", "Technical fallback description", 0, 1.0, "2026-01-01T00:00:00Z"),
                ("d2", "a1", "dejargonized", "Keep the signal within range", 1, 0.1, "2026-01-02T00:00:00Z"),
            ],
        )
        con.executemany(
            """
            INSERT INTO io_specs (
                atom_id, port_name, direction, type_desc, required, default_value_repr, ordinal
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("a1", "signal", "input", "ndarray", 1, "", 0),
                ("a1", "threshold", "input", "float", 0, "0.5", 1),
                ("a1", "filtered", "output", "ndarray", 1, "", 0),
            ],
        )
        con.commit()
        con.close()

        store = build_index_from_manifest_sqlite(db_path, embedder=_FakeManifestEmbedder())

        assert store.size == 1
        decl = next(iter(store._declarations.values()))
        assert decl.name == "pkg.filter"
        assert decl.source_lib == "manifest:repo-123"
        assert decl.docstring == "Keep the signal within range"
        assert decl.type_signature == (
            "(signal: ndarray, threshold?: float [default=0.5]) -> filtered: ndarray"
        )
        assert "signal,analysis" in decl.conceptual_summary


class TestCompositeSemanticIndex:
    def test_merges_results_by_best_score_and_name(self):
        from sciona.indexer.unified import CompositeSemanticIndex, SemanticIndexSource

        shared_local = Declaration(name="shared", type_signature="Local")
        shared_manifest = Declaration(name="shared", type_signature="Manifest")
        local_only = Declaration(name="local_only", type_signature="LocalOnly")
        manifest_only = Declaration(name="manifest_only", type_signature="ManifestOnly")

        local = _StubSemanticIndex(
            hits_by_query={"query": [(shared_local, 0.80), (local_only, 0.70)]},
            declarations={"shared": shared_local, "local_only": local_only},
        )
        manifest = _StubSemanticIndex(
            hits_by_query={"query": [(shared_manifest, 0.90), (manifest_only, 0.60)]},
            declarations={"shared": shared_manifest, "manifest_only": manifest_only},
        )

        composite = CompositeSemanticIndex(
            [
                SemanticIndexSource(local, "fastembed:fake-model", name="local"),
                SemanticIndexSource(manifest, "fastembed:fake-model", name="manifest"),
            ]
        )

        hits = composite.search_by_embedding("query", k=3)
        assert [decl.name for decl, _score in hits] == [
            "shared",
            "local_only",
            "manifest_only",
        ]
        assert hits[0][1] == 0.90
        assert composite.get_declaration("shared") is shared_local

    def test_prefers_first_source_on_tie_and_rejects_space_mismatch(self):
        from sciona.indexer.unified import CompositeSemanticIndex, SemanticIndexSource

        local_decl = Declaration(name="shared", type_signature="Local")
        manifest_decl = Declaration(name="shared", type_signature="Manifest")

        local = _StubSemanticIndex(
            hits_by_query={"query": [(local_decl, 0.80)]},
            declarations={"shared": local_decl},
        )
        manifest = _StubSemanticIndex(
            hits_by_query={"query": [(manifest_decl, 0.80)]},
            declarations={"shared": manifest_decl},
        )

        composite = CompositeSemanticIndex(
            [
                SemanticIndexSource(local, "fastembed:fake-model", name="local"),
                SemanticIndexSource(manifest, "fastembed:fake-model", name="manifest"),
            ]
        )
        hits = composite.search_by_embedding("query", k=1)
        assert hits[0][0] is local_decl
        assert composite.get_declaration("shared") is local_decl

        with pytest.raises(ValueError, match="incompatible embedding spaces"):
            CompositeSemanticIndex(
                [
                    SemanticIndexSource(local, "fastembed:model-a", name="local"),
                    SemanticIndexSource(manifest, "fastembed:model-b", name="manifest"),
                ]
            )


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
