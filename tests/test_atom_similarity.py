"""Tests for pre-decomposition atom similarity (AST fingerprint + call-graph overlap)."""

from __future__ import annotations

import textwrap

import pytest

from sciona.architect.atom_similarity import (
    CallGraph,
    SimilarityHit,
    build_fingerprint_index,
    extract_call_graph,
    find_overlapping_atoms,
    fingerprint_function,
)


# ---------------------------------------------------------------------------
# Layer 1 — fingerprint_function
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_identical_functions_same_hash(self):
        src = textwrap.dedent("""\
            def foo(x):
                return x + 1
        """)
        assert fingerprint_function(src) == fingerprint_function(src)

    def test_renamed_vars_same_hash(self):
        a = textwrap.dedent("""\
            def foo(x):
                y = x * 2
                return y + 1
        """)
        b = textwrap.dedent("""\
            def bar(a):
                b = a * 2
                return b + 1
        """)
        assert fingerprint_function(a) == fingerprint_function(b)

    def test_different_structure_different_hash(self):
        a = textwrap.dedent("""\
            def foo(x):
                return x + 1
        """)
        b = textwrap.dedent("""\
            def foo(x):
                return x * x + 1
        """)
        assert fingerprint_function(a) != fingerprint_function(b)

    def test_docstring_stripped(self):
        with_doc = textwrap.dedent('''\
            def foo(x):
                """This is a docstring."""
                return x + 1
        ''')
        without_doc = textwrap.dedent("""\
            def foo(x):
                return x + 1
        """)
        assert fingerprint_function(with_doc) == fingerprint_function(without_doc)

    def test_returns_full_sha256(self):
        src = "x = 1"
        fp = fingerprint_function(src)
        assert len(fp) == 64
        int(fp, 16)  # raises if not valid hex

    def test_build_fingerprint_index(self):
        sources = {
            "atom_a": "def a(x): return x + 1",
            "atom_b": "def b(y): return y * 2",
        }
        index = build_fingerprint_index(sources)
        assert len(index) == 2
        # Both keys should be full SHA-256 hex strings (64 chars).
        for fp in index:
            assert len(fp) == 64

    def test_build_index_skips_bad_syntax(self):
        sources = {"bad": "def (broken syntax"}
        index = build_fingerprint_index(sources)
        assert index == {}


# ---------------------------------------------------------------------------
# Layer 2 — call graph extraction
# ---------------------------------------------------------------------------


class TestCallGraph:
    def test_simple_call(self):
        src = textwrap.dedent("""\
            def foo():
                bar()
        """)
        cg = extract_call_graph(src)
        assert ("foo", "bar") in cg.edges
        assert "bar" in cg.callees

    def test_dotted_call(self):
        src = textwrap.dedent("""\
            def process():
                np.array([1, 2])
                scipy.signal.butter(4, 0.5)
        """)
        cg = extract_call_graph(src)
        assert "np.array" in cg.callees
        assert "scipy.signal.butter" in cg.callees
        # Leaf names also present.
        assert "array" in cg.callees
        assert "butter" in cg.callees

    def test_nested_calls(self):
        src = textwrap.dedent("""\
            def outer():
                foo(bar(x))
        """)
        cg = extract_call_graph(src)
        callees = cg.callees
        assert "foo" in callees
        assert "bar" in callees

    def test_method_call(self):
        src = textwrap.dedent("""\
            def run():
                obj.method()
        """)
        cg = extract_call_graph(src)
        assert "obj.method" in cg.callees

    def test_empty_function(self):
        src = textwrap.dedent("""\
            def noop():
                pass
        """)
        cg = extract_call_graph(src)
        assert cg.edges == []
        assert cg.callees == set()

    def test_adjacency(self):
        src = textwrap.dedent("""\
            def f():
                a()
                b()
        """)
        cg = extract_call_graph(src)
        adj = cg.adjacency()
        assert adj["f"] == {"a", "b"}

    def test_multiple_functions(self):
        src = textwrap.dedent("""\
            def f():
                a()
            def g():
                b()
        """)
        cg = extract_call_graph(src)
        assert ("f", "a") in cg.edges
        assert ("g", "b") in cg.edges


# ---------------------------------------------------------------------------
# Layer 3 — find_overlapping_atoms
# ---------------------------------------------------------------------------


class TestFindOverlappingAtoms:
    def test_call_overlap_finds_known_atom(self):
        src = textwrap.dedent("""\
            def my_pipeline():
                bandpass_filter(data, 3, 25)
                detect_peaks(filtered)
        """)
        catalog = {"bandpass_filter", "kalman_gain_update"}
        hits = find_overlapping_atoms(src, catalog)
        names = [h.atom_name for h in hits]
        assert "bandpass_filter" in names
        assert "kalman_gain_update" not in names

    def test_dotted_callee_matches_leaf(self):
        src = textwrap.dedent("""\
            def run():
                scipy.signal.bandpass_filter(x)
        """)
        catalog = {"sciona.atoms.scipy_signal.bandpass.bandpass_filter"}
        hits = find_overlapping_atoms(src, catalog)
        assert len(hits) >= 1
        assert hits[0].atom_name == "sciona.atoms.scipy_signal.bandpass.bandpass_filter"

    def test_fingerprint_exact_match(self):
        src = "def foo(x): return x + 1"
        index = build_fingerprint_index({"my_atom": src})
        hits = find_overlapping_atoms(src, set(), fingerprint_index=index)
        assert len(hits) == 1
        assert hits[0].confidence == 1.0
        assert hits[0].match_layer == "fingerprint"

    def test_fingerprint_renamed_match(self):
        original = "def foo(x): return x + 1"
        renamed = "def bar(y): return y + 1"
        index = build_fingerprint_index({"my_atom": original})
        hits = find_overlapping_atoms(renamed, set(), fingerprint_index=index)
        assert len(hits) == 1
        assert hits[0].atom_name == "my_atom"

    def test_empty_catalog_returns_empty(self):
        src = "def f(): g()"
        hits = find_overlapping_atoms(src, set())
        assert hits == []

    def test_bad_syntax_returns_empty(self):
        hits = find_overlapping_atoms("def (broken", set())
        assert hits == []

    def test_hits_sorted_by_confidence(self):
        src = "def foo(x): return x + 1"
        index = build_fingerprint_index({"exact_atom": src})
        catalog = {"x"}  # x is also a callee via Name node? No, but let's use a real call.
        src2 = textwrap.dedent("""\
            def foo(x):
                bar()
                return x + 1
        """)
        index2 = build_fingerprint_index({"exact_atom": src2})
        hits = find_overlapping_atoms(
            src2, {"bar"}, fingerprint_index=index2
        )
        # fingerprint match (1.0) should come before call_overlap (0.8)
        assert hits[0].match_layer == "fingerprint"
        if len(hits) > 1:
            assert hits[0].confidence >= hits[1].confidence

    def test_no_duplicate_hits(self):
        src = textwrap.dedent("""\
            def run():
                bandpass_filter()
                bandpass_filter()
        """)
        catalog = {"bandpass_filter"}
        hits = find_overlapping_atoms(src, catalog)
        names = [h.atom_name for h in hits]
        assert names.count("bandpass_filter") == 1
