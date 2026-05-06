from __future__ import annotations

import json
from pathlib import Path

from sciona.principal.trick_retrieval import (
    SolutionTrick,
    SolutionTrickRetriever,
    TrickRetrievalQuery,
    clear_solution_trick_caches,
    load_local_solution_tricks,
    retrieve_tricks,
    should_consult_tricks,
)


def _write_registry(root: Path) -> None:
    trick_dir = root / "data" / "solution_tricks"
    trick_dir.mkdir(parents=True)
    (trick_dir / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "tricks": [
                    {
                        "trick_id": "trick.test.metric_bound_clipping",
                        "name": "Metric-bound clipping",
                        "kind": "metric_hack",
                        "status": "allowed_with_validation",
                        "risk_level": "medium",
                        "generalization_level": "general",
                        "summary": "Clip regression predictions to metric bounds for RMSLE or MAE.",
                        "applies_when": ["metric bounds define valid prediction ranges"],
                        "do_not_use_when": ["bounds come only from public leaderboard probes"],
                        "validation_requirements": ["ablate clipping on a held-out split"],
                        "architect_hint": "Use only as post-processing after selecting a CDG.",
                        "related_cdgs": ["solution.kaggle.classical_tabular_ensemble_topology"],
                        "related_operations": ["metric_calibration"],
                        "source_competitions": ["synthetic"],
                        "source_references": ["local-test"],
                        "tags": ["metric", "clipping", "postprocessing"],
                        "audit": {
                            "source_kind": "manual_analysis",
                            "review_status": "draft",
                            "notes": "test fixture",
                        },
                    },
                    {
                        "trick_id": "trick.test.public_lb_probe",
                        "name": "Public leaderboard probing",
                        "kind": "public_lb_overfit_risk",
                        "status": "cataloged",
                        "risk_level": "high",
                        "generalization_level": "competition_specific",
                        "summary": "Tune thresholds against public leaderboard feedback.",
                        "applies_when": ["public leaderboard feedback is available"],
                        "do_not_use_when": ["local validation is reliable"],
                        "validation_requirements": ["record probe count"],
                        "architect_hint": "High-risk context only.",
                        "related_cdgs": ["solution.kaggle.classical_tabular_ensemble_topology"],
                        "related_operations": ["metric_calibration"],
                        "source_competitions": ["synthetic"],
                        "source_references": ["local-test"],
                        "tags": ["leaderboard", "thresholding"],
                        "audit": {
                            "source_kind": "manual_analysis",
                            "review_status": "draft",
                            "notes": "test fixture",
                        },
                    },
                ],
            }
        )
    )


def test_should_consult_tricks_only_for_novel_or_undercovered_cases() -> None:
    assert should_consult_tricks("divergent")
    assert should_consult_tricks({"counterfactual_expansion": {"decision": "true_novel"}})
    assert should_consult_tricks({"projected_coverage": 0.25})
    assert not should_consult_tricks("competitive")
    assert not should_consult_tricks({"assessment": "partial", "projected_coverage": 0.75})


def test_provider_loader_reads_solution_trick_registry(tmp_path, monkeypatch) -> None:
    _write_registry(tmp_path)
    monkeypatch.setenv("SCIONA_ATOM_PROVIDER_ROOTS", str(tmp_path))
    clear_solution_trick_caches()

    tricks = load_local_solution_tricks()

    assert any(trick.trick_id == "trick.test.metric_bound_clipping" for trick in tricks)


def test_retrieve_tricks_returns_nothing_when_gate_closed() -> None:
    retriever = SolutionTrickRetriever(
        [
            SolutionTrick(
                trick_id="trick.test.metric_bound_clipping",
                name="Metric-bound clipping",
                kind="metric_hack",
                status="allowed_with_validation",
                risk_level="medium",
                generalization_level="general",
                summary="Clip regression predictions to metric bounds.",
                related_cdgs=("solution.kaggle.classical_tabular_ensemble_topology",),
                tags=("metric", "clipping"),
            )
        ]
    )

    matches = retriever.retrieve(
        TrickRetrievalQuery(
            goal="tabular regression",
            missing_techniques=("metric-bound clipping",),
            candidate_cdgs=("solution.kaggle.classical_tabular_ensemble_topology",),
        ),
        novelty_assessment="competitive",
    )

    assert matches == []


def test_retrieve_tricks_ranks_related_medium_risk_before_high_risk() -> None:
    medium = SolutionTrick(
        trick_id="trick.test.metric_bound_clipping",
        name="Metric-bound clipping",
        kind="metric_hack",
        status="allowed_with_validation",
        risk_level="medium",
        generalization_level="general",
        summary="Clip regression predictions to metric bounds for RMSLE.",
        applies_when=("metric bounds define valid prediction ranges",),
        validation_requirements=("ablate clipping on a held-out split",),
        architect_hint="Use only as post-processing after selecting a CDG.",
        related_cdgs=("solution.kaggle.classical_tabular_ensemble_topology",),
        related_operations=("metric_calibration",),
        tags=("metric", "clipping", "postprocessing"),
    )
    high = SolutionTrick(
        trick_id="trick.test.public_lb_probe",
        name="Public leaderboard probing",
        kind="public_lb_overfit_risk",
        status="cataloged",
        risk_level="high",
        generalization_level="competition_specific",
        summary="Tune thresholds against public leaderboard feedback.",
        related_cdgs=("solution.kaggle.classical_tabular_ensemble_topology",),
        related_operations=("metric_calibration",),
        tags=("leaderboard", "thresholding"),
    )
    retriever = SolutionTrickRetriever([high, medium])

    matches = retriever.retrieve(
        TrickRetrievalQuery(
            goal="tabular regression RMSLE metric clipping",
            missing_techniques=("metric-bound clipping",),
            candidate_cdgs=("solution.kaggle.classical_tabular_ensemble_topology",),
            tags=("metric", "postprocessing"),
        ),
        novelty_assessment={"assessment": "divergent"},
        max_results=5,
    )

    assert [match.trick.trick_id for match in matches][:2] == [
        "trick.test.metric_bound_clipping",
        "trick.test.public_lb_probe",
    ]
    assert matches[0].high_risk is False
    assert matches[1].high_risk is True
    assert "related_cdg" in matches[0].reasons


def test_retrieve_tricks_convenience_wrapper_uses_gate_and_provider(tmp_path, monkeypatch) -> None:
    _write_registry(tmp_path)
    monkeypatch.setenv("SCIONA_ATOM_PROVIDER_ROOTS", str(tmp_path))
    clear_solution_trick_caches()

    gated_out = retrieve_tricks(
        "tabular regression metric clipping",
        ["solution.kaggle.classical_tabular_ensemble_topology"],
        "competitive",
        missing_techniques=["metric-bound clipping"],
    )
    gated_in = retrieve_tricks(
        "tabular regression metric clipping",
        ["solution.kaggle.classical_tabular_ensemble_topology"],
        "divergent",
        missing_techniques=["metric-bound clipping"],
    )

    assert gated_out == []
    assert gated_in
    assert gated_in[0].trick.trick_id == "trick.test.metric_bound_clipping"

