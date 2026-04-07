from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from sciona.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
from sciona.architect.graph_alignment import AlignmentScore
from sciona.architect.graph_retrieval import ExampleChild, ExampleDecomposition
from sciona.architect.models import AlgorithmicNode, ConceptType, IOSpec, NodeStatus
from sciona.architect.proposal_models import (
    proposal_from_template_match,
    proposal_placeholder_skeleton,
)
from sciona.architect.proposal_ranking import ScoredProposal
from sciona.architect.state import DecompositionDeps, DecompositionState
from sciona.architect.template_retriever import TemplateMatch


def _make_parent() -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id="n_parent",
        name="Estimate Event Rate",
        description="Estimate event rate from a sampled signal.",
        concept_type=ConceptType.SIGNAL_FILTER,
        inputs=[
            IOSpec(name="signal", type_desc="np.ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
        ],
        outputs=[IOSpec(name="rate", type_desc="tuple[np.ndarray, np.ndarray]")],
        status=NodeStatus.PENDING,
        depth=1,
    )


def _make_state(parent: AlgorithmicNode) -> DecompositionState:
    return {
        "goal": "Estimate event rate from a signal",
        "max_depth": 8,
        "nodes": [parent],
        "edges": [],
        "history": [],
        "pending_node_ids": [parent.node_id],
        "current_node_id": parent.node_id,
        "paradigm": parent.concept_type.value,
        "skeleton_instantiated": True,
        "critique_passed": False,
        "critique_reason": "",
        "critique_retries": 0,
        "done": False,
        "error": "",
    }


def _make_deps(*, llm: AsyncMock, template_retriever=None) -> DecompositionDeps:
    catalog = PrimitiveCatalog()
    seed_builtin_primitives(catalog)
    skill_index = AsyncMock()
    skill_index.search = lambda query, k=5: []
    return DecompositionDeps(
        catalog=catalog,
        skill_index=skill_index,
        llm=llm,
        template_retriever=template_retriever,
    )


def _make_template_match(*, confidence: float = 0.92) -> TemplateMatch:
    example = ExampleDecomposition(
        fqn="repo.signal.simple_template",
        name="Simple Template",
        description="A simple reusable signal-processing step.",
        concept_type=ConceptType.SIGNAL_FILTER.value,
        repo="repo",
        topo_hash="hash",
        children=[
            ExampleChild(
                node_id="child_1",
                name="Simple Template Step",
                description="Apply a simple signal-processing step.",
                concept_type=ConceptType.SIGNAL_FILTER.value,
                status=NodeStatus.ATOMIC.value,
                n_inputs=2,
                n_outputs=1,
                type_signature="np.ndarray,float -> tuple[np.ndarray,np.ndarray]",
                matched_primitive="filter_signal_for_detection",
            )
        ],
        edges=[],
        retrieval_layer=1,
        score=0.95,
        n_inputs=2,
        n_outputs=1,
    )
    alignment = AlignmentScore(
        total=0.91,
        concept_type_match=1.0,
        io_arity_match=1.0,
        child_concept_overlap=0.8,
        topo_match=0.8,
        type_class_match=0.9,
        witness_type_match=0.9,
    )
    return TemplateMatch(
        example=example,
        alignment=alignment,
        confidence=confidence,
        source="verified_exemplar_same_family",
    )


class _TemplateRetrieverStub:
    def __init__(self, matches: list[TemplateMatch]):
        self._matches = matches

    async def find_templates(self, node, all_nodes, all_edges):
        return list(self._matches)


@pytest.mark.asyncio
async def test_rejects_skeleton_when_simpler_template_is_nearly_as_good(monkeypatch):
    from sciona.architect.nodes import decompose_node

    parent = _make_parent()
    template_match = _make_template_match(confidence=0.95)
    template_proposal = proposal_from_template_match(template_match)
    skeleton_proposal = proposal_placeholder_skeleton(
        skeleton_name="signal_detect_measure",
        source_family=ConceptType.SIGNAL_FILTER.value,
        source_label="Signal Detect and Measure",
        confidence=1.0,
        compatibility_score=1.0,
        delta_nodes=3,
        delta_edges=2,
        delta_family_count=1,
        delta_concept_type_count=3,
    )
    monkeypatch.setattr(
        "sciona.architect.nodes.generate_skeleton_proposals",
        lambda node: [skeleton_proposal],
    )
    monkeypatch.setattr(
        "sciona.architect.nodes.rank_proposals",
        lambda proposals, preferred_family="": [
            ScoredProposal(
                proposal=skeleton_proposal,
                objective_gain=1.20,
                complexity_penalty=1.00,
                risk_penalty=0.00,
                prior_bonus=0.00,
                score=0.35,
            ),
            ScoredProposal(
                proposal=template_proposal,
                objective_gain=1.00,
                complexity_penalty=0.20,
                risk_penalty=0.00,
                prior_bonus=0.00,
                score=0.27,
            ),
        ],
    )

    llm = AsyncMock()
    llm.complete.side_effect = AssertionError("LLM should not run when template wins")
    deps = _make_deps(
        llm=llm,
        template_retriever=_TemplateRetrieverStub([template_match]),
    )

    result = await decompose_node(_make_state(parent), {"configurable": {"deps": deps}})

    assert result["history"][0]["selected_proposal_type"] == "template"
    assert result["history"][0]["skeleton_acceptance_reason"] == (
        "insufficient_margin_over_simpler_alternative"
    )
    assert result["history"][0]["template_used"] == template_match.example.fqn
    assert all(node.name != "Compute Event Rate" for node in result["nodes"])


@pytest.mark.asyncio
async def test_accepts_skeleton_only_when_materially_better(monkeypatch):
    from sciona.architect.nodes import decompose_node

    parent = _make_parent()
    skeleton_proposal = proposal_placeholder_skeleton(
        skeleton_name="signal_detect_measure",
        source_family=ConceptType.SIGNAL_FILTER.value,
        source_label="Signal Detect and Measure",
        confidence=1.0,
        compatibility_score=1.0,
        delta_nodes=3,
        delta_edges=2,
        delta_family_count=1,
        delta_concept_type_count=3,
    )
    monkeypatch.setattr(
        "sciona.architect.nodes.generate_skeleton_proposals",
        lambda node: [skeleton_proposal],
    )
    monkeypatch.setattr(
        "sciona.architect.nodes.rank_proposals",
        lambda proposals, preferred_family="": [
            ScoredProposal(
                proposal=skeleton_proposal,
                objective_gain=2.0,
                complexity_penalty=0.90,
                risk_penalty=0.00,
                prior_bonus=0.00,
                score=0.80,
            )
        ],
    )

    llm = AsyncMock()
    llm.complete.side_effect = AssertionError("LLM should not run when skeleton wins")
    deps = _make_deps(llm=llm, template_retriever=_TemplateRetrieverStub([]))

    result = await decompose_node(_make_state(parent), {"configurable": {"deps": deps}})

    assert result["history"][0]["selected_proposal_type"] == "skeleton"
    assert result["history"][0]["selected_skeleton_name"] == "signal_detect_measure"
    assert result["history"][0]["skeleton_acceptance_reason"].startswith("accepted_")
    assert result["history"][0]["selected_skeleton_asset"]["asset_id"] == "signal_detect_measure"
    assert any(node.name == "Compute Event Rate" for node in result["nodes"])


@pytest.mark.asyncio
async def test_rejects_harmful_skeleton_and_falls_back_to_llm(monkeypatch):
    from sciona.architect.nodes import decompose_node

    parent = _make_parent()
    skeleton_proposal = proposal_placeholder_skeleton(
        skeleton_name="signal_detect_measure",
        source_family=ConceptType.SIGNAL_FILTER.value,
        source_label="Signal Detect and Measure",
        confidence=1.0,
        compatibility_score=1.0,
        delta_nodes=3,
        delta_edges=2,
        delta_family_count=1,
        delta_concept_type_count=3,
    )
    monkeypatch.setattr(
        "sciona.architect.nodes.generate_skeleton_proposals",
        lambda node: [skeleton_proposal],
    )
    monkeypatch.setattr(
        "sciona.architect.nodes.rank_proposals",
        lambda proposals, preferred_family="": [
            ScoredProposal(
                proposal=skeleton_proposal,
                objective_gain=0.10,
                complexity_penalty=0.90,
                risk_penalty=0.10,
                prior_bonus=0.00,
                score=-0.90,
            )
        ],
    )

    llm = AsyncMock()

    async def complete(system: str, user: str) -> str:
        return json.dumps(
            {
                "sub_nodes": [
                    {
                        "name": "Filter Signal",
                        "description": "Filter the signal before downstream processing.",
                        "concept_type": "signal_filter",
                        "inputs": [{"name": "signal", "type_desc": "np.ndarray"}],
                        "outputs": [
                            {"name": "filtered_signal", "type_desc": "np.ndarray"}
                        ],
                    }
                ]
            }
        )

    llm.complete = complete
    deps = _make_deps(llm=llm, template_retriever=_TemplateRetrieverStub([]))

    result = await decompose_node(_make_state(parent), {"configurable": {"deps": deps}})

    assert result["history"][0]["skeleton_acceptance_reason"] == "non_positive_score"
    assert result["history"][0]["num_sub_nodes"] >= 1
    assert result["history"][0]["top_ranked_proposal_type"] == "skeleton"


@pytest.mark.asyncio
async def test_cross_family_skeleton_can_win_when_margin_is_clear(monkeypatch):
    from sciona.architect.nodes import decompose_node

    parent = _make_parent()
    skeleton_proposal = proposal_placeholder_skeleton(
        skeleton_name="signal_detect_measure",
        source_family=ConceptType.SEQUENTIAL_FILTER.value,
        source_label="Signal Detect and Measure",
        confidence=1.0,
        compatibility_score=1.0,
        delta_nodes=3,
        delta_edges=2,
        delta_family_count=1,
        delta_concept_type_count=3,
    )
    monkeypatch.setattr(
        "sciona.architect.nodes.generate_skeleton_proposals",
        lambda node: [skeleton_proposal],
    )
    monkeypatch.setattr(
        "sciona.architect.nodes.rank_proposals",
        lambda proposals, preferred_family="": [
            ScoredProposal(
                proposal=skeleton_proposal,
                objective_gain=2.10,
                complexity_penalty=0.90,
                risk_penalty=0.00,
                prior_bonus=0.00,
                score=0.95,
            )
        ],
    )

    llm = AsyncMock()
    llm.complete.side_effect = AssertionError("LLM should not run when skeleton wins")
    deps = _make_deps(llm=llm, template_retriever=_TemplateRetrieverStub([]))

    result = await decompose_node(_make_state(parent), {"configurable": {"deps": deps}})

    assert result["history"][0]["selected_proposal_type"] == "skeleton"
    assert result["history"][0]["selected_skeleton_name"] == "signal_detect_measure"
    assert any(node.parent_id == parent.node_id for node in result["nodes"][1:])
