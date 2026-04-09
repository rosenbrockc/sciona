from __future__ import annotations

import pytest

from sciona.heuristic_metadata import (
    AtomHeuristicMetadata,
    AtomHeuristicReference,
    HeuristicOutputContract,
    atom_heuristic_metadata_from_snapshot,
    atom_heuristic_metadata_summary,
)
from sciona.heuristics import (
    CanonicalHeuristic,
    HeuristicActionClass,
    HeuristicApplicabilityScope,
    HeuristicEvidenceType,
    HeuristicProducerKind,
)


def _heuristic(heuristic_id: str) -> CanonicalHeuristic:
    return CanonicalHeuristic(
        heuristic_id=heuristic_id,
        display_name=heuristic_id.replace("_", " ").title(),
        dejargonized_meaning="Generic cross-family heuristic.",
        evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
        producer_kind=HeuristicProducerKind.ATOM_OUTPUT,
        applicability_scope=HeuristicApplicabilityScope.CROSS_FAMILY,
        supported_action_classes=[HeuristicActionClass.GATE_OR_VALIDATE],
    )


def test_output_contract_normalizes_producer_kind_to_atom_output() -> None:
    contract = HeuristicOutputContract(
        output_name="quality_score",
        heuristic=_heuristic("quality_instability").model_copy(
            update={"producer_kind": HeuristicProducerKind.RUNTIME_TRANSFORM}
        ),
    )
    assert contract.heuristic.producer_kind == HeuristicProducerKind.ATOM_OUTPUT


def test_atom_heuristic_metadata_requires_unique_output_names_and_ids() -> None:
    with pytest.raises(ValueError):
        AtomHeuristicMetadata(
            atom_fqdn="ageoa.example.quality_gate",
            summary="Quality scoring helper.",
            dejargonized_summary="Produces a reusable quality score.",
            heuristic_outputs=[
                HeuristicOutputContract(output_name="score", heuristic=_heuristic("quality_instability")),
                HeuristicOutputContract(output_name="score", heuristic=_heuristic("density_collapse")),
            ],
            references=[AtomHeuristicReference(title="Reference")],
        )

    with pytest.raises(ValueError):
        AtomHeuristicMetadata(
            atom_fqdn="ageoa.example.quality_gate",
            summary="Quality scoring helper.",
            dejargonized_summary="Produces a reusable quality score.",
            heuristic_outputs=[
                HeuristicOutputContract(output_name="score", heuristic=_heuristic("quality_instability")),
                HeuristicOutputContract(output_name="mask", heuristic=_heuristic("quality_instability")),
            ],
            references=[AtomHeuristicReference(title="Reference")],
        )


def test_atom_heuristic_metadata_from_snapshot_maps_legacy_metric() -> None:
    metadata = atom_heuristic_metadata_from_snapshot(
        "ageoa.example.quality_gate",
        {
            "summary": "Quality scoring helper.",
            "dejargonized_summary": "Produces a reusable quality score.",
            "source_domain": "signal_event_rate",
            "heuristic_outputs": [
                {
                    "output_name": "quality_score",
                    "metric_name": "signal_quality_variance",
                    "role": "gating",
                }
            ],
            "references": [{"title": "Reference"}],
        },
    )
    assert metadata.heuristic_outputs[0].heuristic.heuristic_id == "quality_instability"
    assert metadata.heuristic_outputs[0].role == "gating"
    assert metadata.heuristic_outputs[0].heuristic.producer_kind == HeuristicProducerKind.ATOM_OUTPUT


def test_summary_reports_heuristic_ids() -> None:
    metadata = AtomHeuristicMetadata(
        atom_fqdn="ageoa.example.quality_gate",
        summary="Quality scoring helper.",
        dejargonized_summary="Produces a reusable quality score.",
        heuristic_outputs=[
            HeuristicOutputContract(output_name="quality_score", heuristic=_heuristic("quality_instability")),
        ],
        references=[AtomHeuristicReference(title="Reference")],
        maintainers=["ageo-atoms"],
    )
    summary = atom_heuristic_metadata_summary(metadata)
    assert summary["heuristic_output_count"] == 1
    assert summary["heuristic_ids"] == ["quality_instability"]


def test_metadata_requires_dejargonized_summary_and_reference() -> None:
    with pytest.raises(ValueError):
        AtomHeuristicMetadata(
            atom_fqdn="ageoa.example.quality_gate",
            summary="Quality scoring helper.",
            dejargonized_summary="",
            heuristic_outputs=[],
            references=[AtomHeuristicReference(title="Reference")],
        )
    with pytest.raises(ValueError):
        AtomHeuristicMetadata(
            atom_fqdn="ageoa.example.quality_gate",
            summary="Quality scoring helper.",
            dejargonized_summary="Produces a reusable quality score.",
            heuristic_outputs=[],
            references=[],
        )
