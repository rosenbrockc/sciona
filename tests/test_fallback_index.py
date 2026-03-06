"""Tests for FAISS-free lexical fallback index."""

from __future__ import annotations

import pickle

import msgpack
import pytest

from ageom.cli import _load_semantic_index
from ageom.indexer.fallback_index import LexicalSemanticIndex
from ageom.types import Declaration, Prover


def _write_msgpack_index(tmp_path):
    data = {
        0: {
            "name": "fft_forward",
            "type_signature": "ndarray -> ndarray",
            "docstring": "Compute one-sided FFT spectrum from a signal.",
            "conceptual_summary": "frequency transform",
            "source_lib": "dsp",
            "prover": "python",
            "raw_code": "",
        },
        1: {
            "name": "peak_pick",
            "type_signature": "ndarray -> int",
            "docstring": "Pick dominant spectral bin index.",
            "conceptual_summary": "argmax peak detection",
            "source_lib": "dsp",
            "prover": "python",
            "raw_code": "",
        },
    }
    (tmp_path / "declarations.msgpack").write_bytes(
        msgpack.packb(data, use_bin_type=True)
    )


def test_load_and_search_msgpack(tmp_path):
    _write_msgpack_index(tmp_path)
    idx = LexicalSemanticIndex.load(tmp_path)

    hits = idx.search_by_embedding("fft spectrum", k=2)
    assert hits
    assert hits[0][0].name == "fft_forward"
    assert hits[0][1] > 0

    type_hits = idx.search_by_type("ndarray -> int", k=2)
    assert type_hits
    assert type_hits[0].name == "peak_pick"
    assert idx.get_declaration("peak_pick") is not None


def test_load_legacy_pickle(tmp_path):
    decls = {
        0: Declaration(
            name="binary_search",
            type_signature="list[int] -> int -> int",
            docstring="Find target in sorted array",
            prover=Prover.PYTHON,
        )
    }
    with open(tmp_path / "declarations.pkl", "wb") as f:
        pickle.dump(decls, f)

    idx = LexicalSemanticIndex.load(tmp_path)
    hits = idx.search_by_embedding("sorted array target", k=1)
    assert hits
    assert hits[0][0].name == "binary_search"


def test_cli_loader_falls_back_when_faiss_missing(tmp_path, monkeypatch):
    _write_msgpack_index(tmp_path)

    from ageom.indexer.faiss_store import FAISSStore

    def _raise_missing(_directory):
        raise ModuleNotFoundError("No module named 'faiss'")

    monkeypatch.setattr(FAISSStore, "load", staticmethod(_raise_missing))

    class _Cfg:
        embedding_model = "microsoft/unixcoder-base"

    idx, mode = _load_semantic_index(tmp_path, _Cfg())
    assert mode == "lexical_fallback"
    hits = idx.search_by_embedding("fft", k=1)
    assert hits and hits[0][0].name == "fft_forward"


def test_cli_loader_reraises_non_faiss_import_errors(tmp_path, monkeypatch):
    _write_msgpack_index(tmp_path)

    from ageom.indexer.faiss_store import FAISSStore

    def _raise_other(_directory):
        raise ImportError("broken_dependency")

    monkeypatch.setattr(FAISSStore, "load", staticmethod(_raise_other))

    class _Cfg:
        embedding_model = "microsoft/unixcoder-base"

    with pytest.raises(ImportError, match="broken_dependency"):
        _load_semantic_index(tmp_path, _Cfg())
