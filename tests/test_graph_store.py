"""Tests for sciona.graph_store — Cypher param generation, metadata extraction, idempotency.

All tests are pure-Python (no live Memgraph required).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from sciona.graph_store import (
    build_atom_params,
    build_edge_params,
    build_port_params,
    collect_stale_fqns,
    extract_contract_metadata,
    extract_witness_metadata,
    _topo_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_NODE = {
    "node_id": "gamboa_segmenter",
    "parent_id": "EDAProcessor_root",
    "name": "Gamboa Segmenter",
    "description": "Detect phasic EDA responses",
    "concept_type": "segmentation",
    "inputs": [{"name": "signal", "type_desc": "np.ndarray", "constraints": "raw EDA signal"}],
    "outputs": [{"name": "onsets", "type_desc": "np.ndarray", "constraints": "SCR onset indices"}],
    "status": "atomic",
    "children": [],
    "depth": 1,
    "type_signature": "(signal: np.ndarray) -> np.ndarray",
    "is_optional": False,
    "is_opaque": False,
    "is_external": False,
    "parallelizable": False,
    "conceptual_summary": "Gamboa Segmenter",
}

SAMPLE_DECOMPOSED_NODE = {
    "node_id": "EDAProcessor_root",
    "parent_id": None,
    "name": "EDAProcessor",
    "description": "EDA processing pipeline",
    "concept_type": "custom",
    "inputs": [],
    "outputs": [],
    "status": "decomposed",
    "children": ["gamboa_segmenter", "eda_feature_extraction"],
    "depth": 0,
}

SAMPLE_EDGE = {
    "source_id": "gamboa_segmenter",
    "target_id": "eda_feature_extraction",
    "output_name": "onsets",
    "input_name": "onsets",
    "source_type": "np.ndarray",
    "target_type": "np.ndarray",
    "requires_glue": False,
}


# ---------------------------------------------------------------------------
# build_atom_params
# ---------------------------------------------------------------------------

class TestBuildAtomParams:
    def test_basic_properties(self):
        params = build_atom_params("biosppy", SAMPLE_NODE)
        assert params["fqn"] == "biosppy.gamboa_segmenter"
        assert params["repo"] == "biosppy"
        assert params["node_id"] == "gamboa_segmenter"
        assert params["name"] == "Gamboa Segmenter"
        assert params["concept_type"] == "segmentation"
        assert params["status"] == "atomic"
        assert params["n_inputs"] == 1
        assert params["n_outputs"] == 1
        assert params["is_optional"] is False

    def test_witness_metadata_merged(self):
        w_meta = {
            "witness_name": "witness_gamboa_segmenter",
            "witness_input_types": ["AbstractSignal"],
            "witness_output_types": ["AbstractSignal"],
            "abstract_type_class": "Signal",
            "is_stateful": False,
        }
        params = build_atom_params("biosppy", SAMPLE_NODE, witness_meta=w_meta)
        assert params["witness_name"] == "witness_gamboa_segmenter"
        assert params["witness_input_types"] == ["AbstractSignal"]
        assert params["abstract_type_class"] == "Signal"
        assert params["is_stateful"] is False

    def test_contract_metadata_merged(self):
        c_meta = {
            "input_contracts": ["data must not be None", "data must be a numpy array"],
            "output_contracts": ["result must not be None"],
        }
        params = build_atom_params("biosppy", SAMPLE_NODE, contract_meta=c_meta)
        assert params["input_contracts"] == ["data must not be None", "data must be a numpy array"]
        assert params["output_contracts"] == ["result must not be None"]

    def test_no_metadata(self):
        params = build_atom_params("biosppy", SAMPLE_NODE)
        assert "witness_name" not in params
        assert "input_contracts" not in params

    def test_empty_inputs_outputs(self):
        params = build_atom_params("biosppy", SAMPLE_DECOMPOSED_NODE)
        assert params["n_inputs"] == 0
        assert params["n_outputs"] == 0


# ---------------------------------------------------------------------------
# build_port_params
# ---------------------------------------------------------------------------

class TestBuildPortParams:
    def test_input_port(self):
        io_spec = {"name": "signal", "type_desc": "np.ndarray", "constraints": "raw EDA"}
        params = build_port_params("biosppy", "gamboa_segmenter", io_spec, "in")
        assert params["port_id"] == "biosppy.gamboa_segmenter.in.signal"
        assert params["name"] == "signal"
        assert params["type_desc"] == "np.ndarray"

    def test_output_port(self):
        io_spec = {"name": "onsets", "type_desc": "np.ndarray", "constraints": "indices"}
        params = build_port_params("biosppy", "gamboa_segmenter", io_spec, "out")
        assert params["port_id"] == "biosppy.gamboa_segmenter.out.onsets"


# ---------------------------------------------------------------------------
# build_edge_params
# ---------------------------------------------------------------------------

class TestBuildEdgeParams:
    def test_data_flow_edge(self):
        params = build_edge_params(SAMPLE_EDGE)
        assert params["output_name"] == "onsets"
        assert params["input_name"] == "onsets"
        assert params["source_type"] == "np.ndarray"
        assert params["requires_glue"] is False


# ---------------------------------------------------------------------------
# collect_stale_fqns
# ---------------------------------------------------------------------------

class TestCollectStaleFqns:
    def test_no_stale(self):
        stale = collect_stale_fqns(
            "biosppy",
            {"gamboa_segmenter", "eda_feature_extraction"},
            {"biosppy.gamboa_segmenter", "biosppy.eda_feature_extraction"},
        )
        assert stale == set()

    def test_detects_stale(self):
        stale = collect_stale_fqns(
            "biosppy",
            {"gamboa_segmenter"},
            {"biosppy.gamboa_segmenter", "biosppy.old_node"},
        )
        assert stale == {"biosppy.old_node"}

    def test_empty_db(self):
        stale = collect_stale_fqns("biosppy", {"a", "b"}, set())
        assert stale == set()


# ---------------------------------------------------------------------------
# Idempotency: MERGE key construction
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_same_node_produces_same_fqn(self):
        """Two calls with the same repo+node_id produce the same fqn (MERGE key)."""
        p1 = build_atom_params("biosppy", SAMPLE_NODE)
        p2 = build_atom_params("biosppy", SAMPLE_NODE)
        assert p1["fqn"] == p2["fqn"]

    def test_different_repo_different_fqn(self):
        p1 = build_atom_params("biosppy", SAMPLE_NODE)
        p2 = build_atom_params("scipy", SAMPLE_NODE)
        assert p1["fqn"] != p2["fqn"]

    def test_port_idempotency(self):
        io_spec = {"name": "signal", "type_desc": "np.ndarray", "constraints": ""}
        p1 = build_port_params("biosppy", "node_a", io_spec, "in")
        p2 = build_port_params("biosppy", "node_a", io_spec, "in")
        assert p1["port_id"] == p2["port_id"]


# ---------------------------------------------------------------------------
# Topological hash
# ---------------------------------------------------------------------------

class TestTopoHash:
    def test_consistent_hash(self):
        nodes = [
            {"node_id": "root", "parent_id": None},
            {"node_id": "a", "parent_id": "root"},
            {"node_id": "b", "parent_id": "root"},
        ]
        edges = [{"source_id": "a", "target_id": "b"}]
        h1 = _topo_hash(nodes, edges, "root")
        h2 = _topo_hash(nodes, edges, "root")
        assert h1 == h2
        assert len(h1) == 16

    def test_different_topology_different_hash(self):
        nodes1 = [
            {"node_id": "root", "parent_id": None},
            {"node_id": "a", "parent_id": "root"},
        ]
        nodes2 = [
            {"node_id": "root", "parent_id": None},
            {"node_id": "a", "parent_id": "root"},
            {"node_id": "b", "parent_id": "root"},
        ]
        h1 = _topo_hash(nodes1, [], "root")
        h2 = _topo_hash(nodes2, [], "root")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Witness metadata extraction
# ---------------------------------------------------------------------------

class TestExtractWitnessMetadata:
    def test_extracts_from_witness_file(self, tmp_path: Path):
        witness_py = tmp_path / "eda_witnesses.py"
        witness_py.write_text(textwrap.dedent("""\
            from __future__ import annotations

            class AbstractSignal:
                pass

            def witness_gamboa_segmenter(signal: AbstractSignal) -> AbstractSignal:
                pass

            def witness_eda_feature(signal: AbstractSignal, onsets: AbstractSignal) -> tuple[AbstractSignal, AbstractSignal]:
                pass

            def unrelated_function():
                pass
        """))

        meta = extract_witness_metadata(tmp_path, ["gamboa_segmenter", "eda_feature"])
        assert "gamboa_segmenter" in meta
        assert meta["gamboa_segmenter"]["witness_name"] == "witness_gamboa_segmenter"
        assert meta["gamboa_segmenter"]["witness_input_types"] == ["AbstractSignal"]
        assert meta["gamboa_segmenter"]["witness_output_types"] == ["AbstractSignal"]
        assert meta["gamboa_segmenter"]["is_stateful"] is False

    def test_detects_stateful(self, tmp_path: Path):
        witness_py = tmp_path / "witnesses.py"
        witness_py.write_text(textwrap.dedent("""\
            class AbstractSignal:
                pass

            def witness_stateful_node(signal: AbstractSignal, state: AbstractSignal) -> AbstractSignal:
                pass
        """))

        meta = extract_witness_metadata(tmp_path, ["stateful_node"])
        assert meta["stateful_node"]["is_stateful"] is True

    def test_no_match(self, tmp_path: Path):
        witness_py = tmp_path / "witnesses.py"
        witness_py.write_text("def witness_other(x): pass\n")
        meta = extract_witness_metadata(tmp_path, ["nonexistent"])
        assert meta == {}

    def test_tuple_return_parsing(self, tmp_path: Path):
        witness_py = tmp_path / "witnesses.py"
        witness_py.write_text(textwrap.dedent("""\
            def witness_multi(x: int) -> tuple[str, float, bool]:
                pass
        """))
        meta = extract_witness_metadata(tmp_path, ["multi"])
        assert meta["multi"]["witness_output_types"] == ["str", "float", "bool"]


# ---------------------------------------------------------------------------
# Contract metadata extraction
# ---------------------------------------------------------------------------

class TestExtractContractMetadata:
    def test_extracts_contracts(self, tmp_path: Path):
        atoms_py = tmp_path / "atoms.py"
        atoms_py.write_text(textwrap.dedent("""\
            import icontract

            @icontract.require(lambda data: data is not None, "data must not be None")
            @icontract.require(lambda data: data.shape[0] > 0, "data must not be empty")
            @icontract.ensure(lambda result: result is not None, "result must not be None")
            def peak_detection(data):
                pass
        """))

        meta = extract_contract_metadata(tmp_path, ["peak_detection"])
        assert "peak_detection" in meta
        assert "data must not be None" in meta["peak_detection"]["input_contracts"]
        assert "data must not be empty" in meta["peak_detection"]["input_contracts"]
        assert "result must not be None" in meta["peak_detection"]["output_contracts"]

    def test_no_contracts(self, tmp_path: Path):
        atoms_py = tmp_path / "atoms.py"
        atoms_py.write_text("def plain_func(x): pass\n")
        meta = extract_contract_metadata(tmp_path, ["plain_func"])
        assert meta == {}
