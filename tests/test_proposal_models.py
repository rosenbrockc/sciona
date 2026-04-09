from __future__ import annotations

import pytest

from sciona.architect.graph_alignment import AlignmentScore
from sciona.architect.graph_retrieval import ExampleChild, ExampleDecomposition, ExampleEdge
from sciona.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec
from sciona.architect.proposal_models import (
    EnrichmentProposal,
    ProposalType,
    proposal_from_primitive,
    proposal_from_template_match,
    proposal_placeholder_skeleton,
)
from sciona.architect.template_retriever import TemplateMatch


def test_proposal_from_primitive_builds_passive_metadata() -> None:
    primitive = AlgorithmicPrimitive(
        name="ageoa.signal.filter_signal_basic",
        source="ageoa.signal",
        category=ConceptType.SIGNAL_FILTER,
        description="Filter a signal.",
        inputs=[IOSpec(name="signal", type_desc="ndarray")],
        outputs=[IOSpec(name="filtered_signal", type_desc="ndarray")],
    )

    proposal = proposal_from_primitive(
        primitive,
        confidence=0.42,
        compatibility_score=0.88,
    )

    assert proposal.proposal_type == ProposalType.PRIMITIVE
    assert proposal.matched_primitive == "ageoa.signal.filter_signal_basic"
    assert proposal.source_family == "ageoa.signal"
    assert proposal.delta_nodes == 0
    assert proposal.delta_edges == 0
    assert proposal.template_fqn is None
    assert proposal.skeleton_name is None


def test_proposal_from_template_match_derives_shape_metadata() -> None:
    example = ExampleDecomposition(
        fqn="ageoa.templates.signal_quality_gate",
        name="Signal Quality Gate",
        description="Insert quality scoring before filtering.",
        concept_type=ConceptType.ANALYSIS.value,
        repo="ageo-atoms",
        topo_hash="abc123",
        children=[
            ExampleChild(
                node_id="c1",
                name="Filter Signal",
                description="Filter signal.",
                concept_type=ConceptType.SIGNAL_FILTER.value,
                status="atomic",
                n_inputs=1,
                n_outputs=1,
                type_signature="ndarray -> ndarray",
                matched_primitive="ageoa.signal.filter_signal_basic",
            ),
            ExampleChild(
                node_id="c2",
                name="Score Signal Quality",
                description="Score quality.",
                concept_type=ConceptType.ANALYSIS.value,
                status="atomic",
                n_inputs=1,
                n_outputs=1,
                type_signature="ndarray -> float",
                matched_primitive="ageoa.statistics.score_signal_quality",
            ),
        ],
        edges=[
            ExampleEdge(
                source_id="c1",
                target_id="c2",
                output_name="filtered_signal",
                input_name="filtered_signal",
            )
        ],
        retrieval_layer=2,
        score=0.9,
    )
    match = TemplateMatch(
        example=example,
        alignment=AlignmentScore(
            total=0.84,
            concept_type_match=0.5,
            io_arity_match=1.0,
            child_concept_overlap=0.5,
            topo_match=0.2,
            type_class_match=0.0,
            witness_type_match=0.0,
        ),
        confidence=0.84,
        source="layer_2",
    )

    proposal = proposal_from_template_match(match)

    assert proposal.proposal_type == ProposalType.TEMPLATE
    assert proposal.template_fqn == "ageoa.templates.signal_quality_gate"
    assert proposal.delta_nodes == 2
    assert proposal.delta_edges == 1
    assert proposal.delta_family_count == 2
    assert proposal.delta_concept_type_count == 2
    assert proposal.skeleton_name is None


def test_proposal_from_primitive_supports_federated_namespace_family_inference() -> None:
    primitive = AlgorithmicPrimitive(
        name="sciona.atoms.fintech.options.charfuncoption",
        source="sciona.atoms.fintech",
        category=ConceptType.CUSTOM,
        description="Price an option.",
        inputs=[IOSpec(name="params", type_desc="dict")],
        outputs=[IOSpec(name="price", type_desc="float")],
    )

    proposal = proposal_from_primitive(primitive)

    assert proposal.source_family == "sciona.atoms.fintech"


def test_placeholder_skeleton_proposal_is_representable() -> None:
    proposal = proposal_placeholder_skeleton(
        skeleton_name="kalman_filter",
        source_family="sequential_filter",
        delta_nodes=4,
        delta_edges=3,
        delta_family_count=1,
        delta_concept_type_count=2,
    )

    assert proposal.proposal_type == ProposalType.SKELETON
    assert proposal.skeleton_name == "kalman_filter"
    assert proposal.template_fqn is None
    assert proposal.delta_nodes == 4


def test_invalid_cross_field_combination_is_rejected() -> None:
    with pytest.raises(ValueError):
        EnrichmentProposal(
            proposal_type=ProposalType.PRIMITIVE,
            matched_primitive="ageoa.signal.filter_signal_basic",
            template_fqn="ageoa.templates.signal_quality_gate",
        )

    with pytest.raises(ValueError):
        EnrichmentProposal(
            proposal_type=ProposalType.SKELETON,
            source_family="signal_filter",
            skeleton_name="signal_detect_measure",
            delta_nodes=-1,
        )
