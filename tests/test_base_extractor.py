"""Tests for BaseExtractor protocol, SourceLanguage, and dispatch factory."""

from __future__ import annotations


from ageom.ingester.base_extractor import (
    EXTENSION_MAP,
    BaseExtractor,
    SourceLanguage,
)
from ageom.ingester.graph import _get_extractor
from ageom.ingester.python_extractor import PythonASTExtractor
from ageom.ingester.treesitter_extractor import TreeSitterExtractor

# ---------------------------------------------------------------------------
# SourceLanguage enum
# ---------------------------------------------------------------------------


class TestSourceLanguage:
    def test_values(self):
        assert SourceLanguage.PYTHON == "python"
        assert SourceLanguage.CPP == "cpp"
        assert SourceLanguage.JULIA == "julia"

    def test_str_enum(self):
        assert isinstance(SourceLanguage.PYTHON, str)
        assert SourceLanguage.CPP.value == "cpp"


# ---------------------------------------------------------------------------
# EXTENSION_MAP
# ---------------------------------------------------------------------------


class TestExtensionMap:
    def test_python(self):
        assert EXTENSION_MAP[".py"] == SourceLanguage.PYTHON

    def test_cpp_extensions(self):
        for ext in [".cpp", ".cc", ".cxx", ".h", ".hpp"]:
            assert EXTENSION_MAP[ext] == SourceLanguage.CPP

    def test_julia(self):
        assert EXTENSION_MAP[".jl"] == SourceLanguage.JULIA

    def test_rust(self):
        assert EXTENSION_MAP[".rs"] == SourceLanguage.RUST

    def test_unknown_not_in_map(self):
        assert ".go" not in EXTENSION_MAP


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_python_extractor_is_base_extractor(self):
        extractor = PythonASTExtractor()
        assert isinstance(extractor, BaseExtractor)

    def test_treesitter_extractor_is_base_extractor(self):
        extractor = TreeSitterExtractor(SourceLanguage.CPP)
        assert isinstance(extractor, BaseExtractor)

    def test_treesitter_julia_is_base_extractor(self):
        extractor = TreeSitterExtractor(SourceLanguage.JULIA)
        assert isinstance(extractor, BaseExtractor)


# ---------------------------------------------------------------------------
# Dispatch factory
# ---------------------------------------------------------------------------


class TestGetExtractor:
    def test_python_dispatch(self):
        ext = _get_extractor("module.py")
        assert isinstance(ext, PythonASTExtractor)

    def test_cpp_dispatch(self):
        ext = _get_extractor("source.cpp")
        assert isinstance(ext, TreeSitterExtractor)

    def test_cc_dispatch(self):
        ext = _get_extractor("source.cc")
        assert isinstance(ext, TreeSitterExtractor)

    def test_julia_dispatch(self):
        ext = _get_extractor("source.jl")
        assert isinstance(ext, TreeSitterExtractor)

    def test_header_dispatch(self):
        ext = _get_extractor("header.hpp")
        assert isinstance(ext, TreeSitterExtractor)

    def test_rust_dispatch(self):
        ext = _get_extractor("module.rs")
        assert isinstance(ext, TreeSitterExtractor)

    def test_unknown_defaults_to_python(self):
        ext = _get_extractor("module.go")
        assert isinstance(ext, PythonASTExtractor)
