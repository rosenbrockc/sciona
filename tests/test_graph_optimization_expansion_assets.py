from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_assets import (
    asset_backed_rule_sets,
    clear_local_expansion_asset_caches,
    load_local_expansion_assets_by_family,
)
from sciona.principal.expansion_rules.graph_optimization import (
    GraphOptimizationExpansionRuleSet,
)


def _edge(source_id: str, target_id: str) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name="out",
        input_name="in",
        source_type="Graph",
        target_type="Graph",
    )


def _node(node_id: str, name: str) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=name,
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
    )


def _graph_optimization_cdg() -> CDGExport:
    src = AlgorithmicNode(
        node_id="src",
        name="source",
        description="graph input",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
    )
    relax = _node("relax", "Relax Edges")
    extract = _node("extract", "Extract Path")
    return CDGExport(
        nodes=[src, relax, extract],
        edges=[_edge("src", "relax"), _edge("relax", "extract")],
    )


def _asset_backed_rule_set():
    clear_local_expansion_asset_caches()
    return asset_backed_rule_sets([GraphOptimizationExpansionRuleSet()])[0]


def test_graph_optimization_provider_asset_includes_runtime_rules() -> None:
    clear_local_expansion_asset_caches()

    asset = load_local_expansion_assets_by_family()["graph_optimization"]

    assert {operation.rule_name for operation in asset.operations} >= {
        "insert_negative_weight_detection_before_relax",
        "insert_relaxation_convergence_after_relax",
        "insert_distance_overflow_detection_before_extract",
        "insert_graph_density_analysis_before_relax",
        "insert_parallel_path_optimization_before_extract",
    }


def test_parallel_path_optimization_rule_applies_to_graph_pipeline() -> None:
    result = ExpansionEngine([_asset_backed_rule_set()]).expand(
        _graph_optimization_cdg(),
        ExpansionContext(
            intermediates={"requires_parallel_path_optimization": True}
        ),
    )

    assert result.expanded is True
    assert "Parallel Path Optimization" in {node.name for node in result.cdg.nodes}
    assert result.applied_assets[0]["asset_operation_id"] == (
        "insert_parallel_path_optimization_before_extract"
    )
    assert result.applied_assets[0]["asset_operation_type"] == "insert"
