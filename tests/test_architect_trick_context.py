from __future__ import annotations

from sciona.architect.models import AlgorithmicNode, ConceptType
from sciona.architect.nodes import _solution_trick_context_block
from sciona.architect.state import DecompositionDeps
from sciona.principal.trick_retrieval import SolutionTrick, SolutionTrickMatch


def _deps(**kwargs) -> DecompositionDeps:
    return DecompositionDeps(
        catalog=object(),
        skill_index=object(),
        llm=object(),
        **kwargs,
    )


def _node() -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id="root",
        name="Predict target",
        description="Train a tabular regression model.",
        concept_type=ConceptType.ML_MODEL_SELECTION,
    )


def test_solution_trick_context_block_stays_closed_without_novelty(monkeypatch) -> None:
    called = False

    def fake_retrieve_tricks(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "sciona.architect.nodes.retrieve_tricks",
        fake_retrieve_tricks,
    )

    block = _solution_trick_context_block(
        {"goal": "tabular regression"},
        _deps(),
        _node(),
    )

    assert block == ""
    assert called is False


def test_solution_trick_context_block_uses_planning_family_and_missing_terms(monkeypatch) -> None:
    captured = {}

    def fake_retrieve_tricks(
        goal,
        candidate_cdgs,
        novelty_assessment,
        *,
        missing_techniques,
        families,
        tags,
        max_results,
    ):
        captured.update(
            {
                "goal": goal,
                "candidate_cdgs": tuple(candidate_cdgs),
                "novelty_assessment": novelty_assessment,
                "missing_techniques": tuple(missing_techniques),
                "families": tuple(families),
                "tags": tuple(tags),
                "max_results": max_results,
            }
        )
        return [
            SolutionTrickMatch(
                trick=SolutionTrick(
                    trick_id="trick.test.metric_bound_clipping",
                    name="Metric-bound clipping",
                    kind="metric_hack",
                    status="allowed_with_validation",
                    risk_level="medium",
                    generalization_level="general",
                    summary="Clip predictions to metric bounds.",
                    validation_requirements=("held-out ablation",),
                ),
                score=0.61,
            )
        ]

    monkeypatch.setattr(
        "sciona.architect.nodes.retrieve_tricks",
        fake_retrieve_tricks,
    )

    block = _solution_trick_context_block(
        {
            "goal": "tabular regression",
            "planning_artifact": {
                "family_hint": "classical_tabular_ensemble",
                "paradigm": "ml_model_selection",
            },
        },
        _deps(
            trick_context_novelty_assessment={"assessment": "divergent"},
            trick_context_missing_techniques=("metric-bound clipping",),
            trick_context_candidate_cdgs=(
                "solution.kaggle.classical_tabular_ensemble_topology",
            ),
            trick_context_max_results=2,
        ),
        _node(),
    )

    assert block.startswith("Optional high-risk tactics for novel-CDG cases")
    assert captured["candidate_cdgs"] == (
        "solution.kaggle.classical_tabular_ensemble_topology",
    )
    assert captured["novelty_assessment"] == {"assessment": "divergent"}
    assert captured["missing_techniques"] == ("metric-bound clipping",)
    assert captured["families"] == (
        "classical_tabular_ensemble",
        "ml_model_selection",
    )
    assert captured["tags"] == ("ml_model_selection",)
    assert captured["max_results"] == 2
