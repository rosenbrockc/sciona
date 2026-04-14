from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.cdg_projection import build_published_cdg_projection


def test_build_published_cdg_projection_derives_deterministic_summary() -> None:
    root = AlgorithmicNode(
        node_id="root",
        name="Detect Rate",
        description="detect rate from signal",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.DECOMPOSED,
        inputs=[IOSpec(name="signal", type_desc="array<float>")],
        outputs=[IOSpec(name="rate", type_desc="float")],
        children=["filter", "score"],
    )
    filter_node = AlgorithmicNode(
        node_id="filter",
        parent_id="root",
        name="Filter",
        description="filter signal",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
    )
    score_node = AlgorithmicNode(
        node_id="score",
        parent_id="root",
        name="Score",
        description="score candidate rate",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
    )
    cdg = CDGExport(
        nodes=[root, filter_node, score_node],
        edges=[
            DependencyEdge(
                source_id="filter",
                target_id="score",
                output_name="filtered",
                input_name="signal",
                source_type="array<float>",
                target_type="array<float>",
            )
        ],
        metadata={"verified_leaf_coverage": 0.75},
    )

    projection = build_published_cdg_projection(
        artifact={"artifact_id": "a1", "fqdn": "pkg.rate_detector", "artifact_kind": "cdg"},
        version={"artifact_version_id": "v1", "semver": "1.0.0", "content_hash": "abc123"},
        cdg=cdg,
    )

    assert projection.artifact_id == "a1"
    assert projection.artifact_version_id == "v1"
    assert projection.fqdn == "pkg.rate_detector"
    assert projection.content_hash == "abc123"
    assert projection.n_inputs == 1
    assert projection.n_outputs == 1
    assert projection.verified_leaf_coverage == 0.75
    assert projection.topo_hash


def test_build_published_cdg_projection_handles_rootless_template_graphs() -> None:
    left = AlgorithmicNode(
        node_id="left",
        name="Left",
        description="left stage",
        concept_type=ConceptType.DIVIDE_AND_CONQUER,
        status=NodeStatus.ATOMIC,
    )
    right = AlgorithmicNode(
        node_id="right",
        name="Right",
        description="right stage",
        concept_type=ConceptType.DIVIDE_AND_CONQUER,
        status=NodeStatus.ATOMIC,
    )
    merge = AlgorithmicNode(
        node_id="merge",
        name="Merge",
        description="merge stage",
        concept_type=ConceptType.DIVIDE_AND_CONQUER,
        status=NodeStatus.ATOMIC,
    )
    linear = CDGExport(
        nodes=[left, right, merge],
        edges=[
            DependencyEdge(
                source_id="left",
                target_id="merge",
                output_name="left_result",
                input_name="left_result",
                source_type="any",
                target_type="any",
            ),
            DependencyEdge(
                source_id="right",
                target_id="merge",
                output_name="right_result",
                input_name="right_result",
                source_type="any",
                target_type="any",
            ),
        ],
        metadata={},
    )
    disconnected = CDGExport(nodes=[left, right, merge], edges=[], metadata={})

    linear_projection = build_published_cdg_projection(
        artifact={"artifact_id": "a1", "fqdn": "cdg.skeleton.linear", "artifact_kind": "cdg"},
        version={"artifact_version_id": "v1", "semver": "1.0.0", "content_hash": "aaa"},
        cdg=linear,
    )
    disconnected_projection = build_published_cdg_projection(
        artifact={"artifact_id": "a2", "fqdn": "cdg.skeleton.disconnected", "artifact_kind": "cdg"},
        version={"artifact_version_id": "v2", "semver": "1.0.0", "content_hash": "bbb"},
        cdg=disconnected,
    )

    assert linear_projection.topo_hash
    assert disconnected_projection.topo_hash
    assert linear_projection.topo_hash != disconnected_projection.topo_hash
