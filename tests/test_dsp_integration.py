"""Integration tests for DSP ConceptTypes, skeletons, and contract patterns."""

import pytest

from ageom.architect.models import ConceptType, IOSpec
from ageom.architect.skeletons import (
    SKELETON_TEMPLATES,
    get_skeleton,
    instantiate_skeleton,
)
from ageom.synthesizer.contracts import ContractGenerator, ContractSpec


class TestDspConceptTypes:
    """Test that DSP ConceptType enum values exist."""

    def test_signal_transform_member(self):
        assert ConceptType.SIGNAL_TRANSFORM == "signal_transform"

    def test_signal_filter_member(self):
        assert ConceptType.SIGNAL_FILTER == "signal_filter"

    def test_graph_signal_processing_member(self):
        assert ConceptType.GRAPH_SIGNAL_PROCESSING == "graph_signal_processing"

    def test_all_dsp_types_in_enum(self):
        names = [ct.value for ct in ConceptType]
        assert "signal_transform" in names
        assert "signal_filter" in names
        assert "graph_signal_processing" in names

    def test_custom_and_external_tool_are_last(self):
        members = list(ConceptType)
        assert members[-2] == ConceptType.CUSTOM
        assert members[-1] == ConceptType.EXTERNAL_TOOL


class TestDspSkeletons:
    """Test DSP skeleton template instantiation."""

    @pytest.mark.parametrize("concept_type", [
        ConceptType.SIGNAL_TRANSFORM,
        ConceptType.SIGNAL_FILTER,
        ConceptType.GRAPH_SIGNAL_PROCESSING,
    ])
    def test_skeleton_exists(self, concept_type):
        skeleton = get_skeleton(concept_type)
        assert skeleton is not None

    @pytest.mark.parametrize("concept_type", [
        ConceptType.SIGNAL_TRANSFORM,
        ConceptType.SIGNAL_FILTER,
        ConceptType.GRAPH_SIGNAL_PROCESSING,
    ])
    def test_skeleton_has_nodes_and_edges(self, concept_type):
        skeleton = get_skeleton(concept_type)
        assert len(skeleton.template_nodes) >= 3
        assert len(skeleton.template_edges) >= 2

    @pytest.mark.parametrize("concept_type", [
        ConceptType.SIGNAL_TRANSFORM,
        ConceptType.SIGNAL_FILTER,
        ConceptType.GRAPH_SIGNAL_PROCESSING,
    ])
    def test_skeleton_no_dangling_edges(self, concept_type):
        skeleton = get_skeleton(concept_type)
        node_ids = {n.node_id for n in skeleton.template_nodes}
        for e in skeleton.template_edges:
            assert e.source_id in node_ids, f"Dangling source: {e.source_id}"
            assert e.target_id in node_ids, f"Dangling target: {e.target_id}"

    @pytest.mark.parametrize("concept_type", [
        ConceptType.SIGNAL_TRANSFORM,
        ConceptType.SIGNAL_FILTER,
        ConceptType.GRAPH_SIGNAL_PROCESSING,
    ])
    def test_skeleton_has_variants(self, concept_type):
        skeleton = get_skeleton(concept_type)
        assert len(skeleton.variants) >= 3

    @pytest.mark.parametrize("concept_type", [
        ConceptType.SIGNAL_TRANSFORM,
        ConceptType.SIGNAL_FILTER,
        ConceptType.GRAPH_SIGNAL_PROCESSING,
    ])
    def test_instantiate_produces_fresh_ids(self, concept_type):
        skeleton = get_skeleton(concept_type)
        nodes1, edges1 = instantiate_skeleton(skeleton, "test goal 1")
        nodes2, edges2 = instantiate_skeleton(skeleton, "test goal 2")
        ids1 = {n.node_id for n in nodes1}
        ids2 = {n.node_id for n in nodes2}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.parametrize("concept_type", [
        ConceptType.SIGNAL_TRANSFORM,
        ConceptType.SIGNAL_FILTER,
        ConceptType.GRAPH_SIGNAL_PROCESSING,
    ])
    def test_instantiate_preserves_structure(self, concept_type):
        skeleton = get_skeleton(concept_type)
        nodes, edges = instantiate_skeleton(skeleton, "test goal")
        assert len(nodes) == len(skeleton.template_nodes)
        assert len(edges) == len(skeleton.template_edges)

    def test_signal_transform_variants(self):
        skeleton = get_skeleton(ConceptType.SIGNAL_TRANSFORM)
        assert "fft_filter" in skeleton.variants
        assert "dct_compression" in skeleton.variants

    def test_signal_filter_variants(self):
        skeleton = get_skeleton(ConceptType.SIGNAL_FILTER)
        assert "butterworth_lowpass" in skeleton.variants
        assert "fir_bandpass" in skeleton.variants

    def test_graph_signal_processing_variants(self):
        skeleton = get_skeleton(ConceptType.GRAPH_SIGNAL_PROCESSING)
        assert "heat_diffusion" in skeleton.variants
        assert "graph_denoising" in skeleton.variants

    def test_total_skeleton_count(self):
        assert len(SKELETON_TEMPLATES) == 13  # 10 original + 3 DSP


class TestDspContractPatterns:
    """Test that the contract generator handles DSP constraint strings."""

    def setup_method(self):
        self.gen = ContractGenerator()

    def test_round_trip_pattern(self):
        spec = IOSpec(
            name="x",
            type_desc="np.ndarray",
            constraints="round_trip: IFFT(FFT(x))",
        )
        contract = self.gen._iospec_to_contract(spec, "ensure")
        assert contract is not None
        assert contract.kind == "ensure"
        assert "np.allclose" in contract.lambda_expr
        assert "Round-trip" in contract.description

    def test_poles_pattern(self):
        spec = IOSpec(
            name="coefficients",
            type_desc="tuple",
            constraints="poles inside unit circle",
        )
        contract = self.gen._iospec_to_contract(spec, "ensure")
        assert contract is not None
        assert contract.kind == "ensure"
        assert "_poles_inside_unit_circle" in contract.lambda_expr

    def test_psd_pattern(self):
        spec = IOSpec(
            name="L",
            type_desc="sparse matrix",
            constraints="positive semi-definite",
        )
        contract = self.gen._iospec_to_contract(spec, "ensure")
        assert contract is not None
        assert contract.kind == "ensure"
        assert "_eigenvalues_nonneg" in contract.lambda_expr

    def test_tv_reduction_pattern(self):
        spec = IOSpec(
            name="signal",
            type_desc="np.ndarray",
            constraints="TV(result) <= TV(input)",
        )
        contract = self.gen._iospec_to_contract(spec, "ensure")
        assert contract is not None
        assert contract.kind == "ensure"
        assert "_total_variation" in contract.lambda_expr

    def test_generic_constraint_still_works(self):
        spec = IOSpec(
            name="a",
            type_desc="np.ndarray",
            constraints="a.ndim == 2",
        )
        contract = self.gen._iospec_to_contract(spec, "require")
        assert contract is not None
        assert contract.kind == "require"
        assert "lambda a:" in contract.lambda_expr

    def test_empty_constraint_returns_none(self):
        spec = IOSpec(name="x", type_desc="np.ndarray", constraints="")
        contract = self.gen._iospec_to_contract(spec, "require")
        assert contract is None
