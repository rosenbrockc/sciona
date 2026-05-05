from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext
from sciona.principal.expansion_assets import clear_local_expansion_asset_caches
from sciona.principal.expansion_delta_planner import (
    DeltaAdaptationKind,
    DeltaPlanningQuery,
    plan_expansion_delta,
)
from sciona.principal.expansion_rules.agent_simulation_search_planning import (
    AgentSimulationSearchPlanningRuleSet,
)


def _node(node_id: str, name: str, concept_type: ConceptType) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=name,
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
    )


def _edge(source_id: str, target_id: str) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name="out",
        input_name="in",
        source_type="PlanningArtifact",
        target_type="PlanningArtifact",
    )


def _planning_cdg() -> CDGExport:
    return CDGExport(
        nodes=[
            _node("state_encoder", "State Encoder", ConceptType.DATA_ASSEMBLY),
            _node("candidate_generator", "Candidate Generator", ConceptType.SEARCHING),
            _node("search_or_rollout", "Search or Rollout", ConceptType.OPTIMIZATION),
            _node("constraint_repair", "Constraint Repair", ConceptType.OPTIMIZATION),
            _node("plan_selection", "Plan Selection", ConceptType.OPTIMIZATION),
        ],
        edges=[
            _edge("state_encoder", "candidate_generator"),
            _edge("candidate_generator", "search_or_rollout"),
            _edge("search_or_rollout", "constraint_repair"),
            _edge("constraint_repair", "plan_selection"),
        ],
    )


def test_agent_planning_asset_retrieves_ppo_policy_optimization() -> None:
    clear_local_expansion_asset_caches()

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("agent_simulation_search_planning", "optimization"),
            matched_techniques=("simulation rollout",),
            missing_techniques=(
                "PPO (Proximal Policy Optimization)",
                "Multi-Agent PPO (Proximal Policy Optimization)",
            ),
            stage_names=("State Encoder", "Search or Rollout", "Constraint Repair"),
            base_coverage=0.333,
            min_adapted_coverage=0.50,
            max_operations_per_sequence=3,
        )
    )

    assert plan.decision == DeltaAdaptationKind.EXPANSION
    assert plan.selected.operation_rule_names == (
        "insert_ppo_policy_optimization_after_rollout",
    )
    assert plan.selected.projected_coverage == 1.0


def test_agent_planning_ppo_does_not_cover_transfer_learning_by_weak_tokens() -> None:
    clear_local_expansion_asset_caches()

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("medical_image_tabular", "neural_network"),
            matched_techniques=("image classifier",),
            missing_techniques=(
                "Multi-stage transfer learning",
                "Test-Time Augmentation (TTA)",
            ),
            stage_names=("Image Backbone", "Training Loop", "Prediction Ensemble"),
            base_coverage=0.333,
            min_adapted_coverage=0.50,
            max_operations_per_sequence=3,
        )
    )

    assert "insert_ppo_policy_optimization_after_rollout" not in (
        plan.selected.operation_rule_names
    )


def test_agent_planning_rule_diagnoses_and_applies_ppo_expansion() -> None:
    rule_set = AgentSimulationSearchPlanningRuleSet()
    diagnostics = rule_set.diagnose(
        _planning_cdg(),
        ExpansionContext(
            planning_artifact={
                "techniques": ["Multi-Agent PPO (Proximal Policy Optimization)"]
            }
        ),
    )

    assert [diagnostic.rule_name for diagnostic in diagnostics] == [
        "insert_ppo_policy_optimization_after_rollout"
    ]

    result = rule_set.rules()[0].semantic_apply(_planning_cdg())

    assert not result.is_failure
    assert "PPO Policy Optimization" in {node.name for node in result.unwrap().nodes}
