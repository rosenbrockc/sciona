from __future__ import annotations

from sciona.architect.models import AlgorithmicNode, ConceptType, IOSpec, NodeStatus
from sciona.architect.proposal_models import ProposalType
from sciona.architect.skeleton_proposals import (
    ALLOWED_SKELETON_PROPOSALS,
    generate_skeleton_proposals,
)


def _make_signal_node() -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id="n_signal",
        name="Estimate Event Rate",
        description="Estimate a rate from a signal stream.",
        concept_type=ConceptType.SIGNAL_FILTER,
        inputs=[
            IOSpec(name="signal", type_desc="any"),
            IOSpec(name="sampling_rate", type_desc="any"),
        ],
        outputs=[IOSpec(name="result", type_desc="any")],
        status=NodeStatus.PENDING,
        depth=1,
    )


def test_generate_skeleton_proposals_respects_allowlist() -> None:
    node = _make_signal_node()
    proposals = generate_skeleton_proposals(node)

    assert proposals
    assert {proposal.skeleton_name for proposal in proposals} <= set(
        ALLOWED_SKELETON_PROPOSALS
    )
    assert all(proposal.proposal_type == ProposalType.SKELETON for proposal in proposals)
    signal_detect = next(
        proposal for proposal in proposals if proposal.skeleton_name == "signal_detect_measure"
    )
    assert signal_detect.payload["source_asset"]["asset_id"] == "signal_detect_measure"


def test_generate_skeleton_proposals_rejects_atomic_nodes() -> None:
    node = _make_signal_node().model_copy(update={"status": NodeStatus.ATOMIC})
    proposals = generate_skeleton_proposals(node)
    assert proposals == []


def test_generate_skeleton_proposals_rejects_boundary_mismatch() -> None:
    node = AlgorithmicNode(
        node_id="n_bad",
        name="Estimate Event Rate",
        description="Estimate a rate from a signal stream.",
        concept_type=ConceptType.SIGNAL_FILTER,
        inputs=[IOSpec(name="signal", type_desc="vector[float]")],
        outputs=[IOSpec(name="result", type_desc="float")],
        status=NodeStatus.PENDING,
        depth=1,
    )

    proposals = generate_skeleton_proposals(node)
    assert proposals == []


def test_generate_skeleton_proposals_enforces_size_caps() -> None:
    node = _make_signal_node()
    proposals = generate_skeleton_proposals(
        node,
        allowlist=("signal_detect_measure", "kalman_filter"),
        max_nodes=3,
        max_edges=2,
    )

    assert proposals
    assert {proposal.skeleton_name for proposal in proposals} == {
        "signal_detect_measure"
    }
