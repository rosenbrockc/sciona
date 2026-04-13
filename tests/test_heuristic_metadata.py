from __future__ import annotations

import json
from pathlib import Path

import pytest

from sciona.heuristic_metadata import (
    AtomHeuristicMetadata,
    AtomHeuristicReference,
    HeuristicOutputContract,
    atom_heuristic_metadata_from_snapshot,
    atom_heuristic_metadata_summary,
    clear_atom_heuristic_metadata_caches,
    resolve_external_atom_heuristic_metadata,
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


@pytest.fixture(autouse=True)
def _clear_metadata_caches() -> None:
    clear_atom_heuristic_metadata_caches()
    try:
        yield
    finally:
        clear_atom_heuristic_metadata_caches()


def _write_metadata(root: Path, *, atom_fqdn: str, heuristic_id: str) -> None:
    path = root / "ageoa" / "demo" / "fixture" / "heuristic_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "atom_fqdn": atom_fqdn,
                        "summary": "Fixture heuristic metadata.",
                        "dejargonized_summary": "Fixture heuristic metadata for precedence tests.",
                        "heuristic_outputs": [
                            {
                                "output_name": "score",
                                "role": "gating",
                                "heuristic": _heuristic(heuristic_id).model_dump(
                                    mode="json"
                                ),
                            }
                        ],
                        "references": [{"title": "Fixture Reference"}],
                    }
                ]
            }
        )
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
    assert summary["logical_atom_id"] == "example.quality_gate"
    assert summary["provider_id"] == "core.ageo_atoms"


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


def test_resolve_external_atom_heuristic_metadata_loads_ageo_atoms_signal_example() -> None:
    metadata = resolve_external_atom_heuristic_metadata(
        "ageoa.biosppy.ecg_zz2018_d12.assemblezz2018sqi"
    )

    assert metadata is not None
    assert metadata.heuristic_outputs
    assert metadata.heuristic_outputs[0].heuristic.heuristic_id == "quality_instability"


def test_external_metadata_supports_multiple_records_in_one_asset_file() -> None:
    metadata = resolve_external_atom_heuristic_metadata(
        "ageoa.biosppy.ecg_zz2018.calculatefrequencypowersqi"
    )

    assert metadata is not None
    assert metadata.heuristic_outputs[0].heuristic.heuristic_id == (
        "dominant_nuisance_structure"
    )


def test_external_metadata_loads_non_signal_family_example() -> None:
    metadata = resolve_external_atom_heuristic_metadata(
        "ageoa.kalman_filters.filter_rs.evaluatemeasurementoracle"
    )

    assert metadata is not None
    assert metadata.heuristic_outputs[0].heuristic.heuristic_id == (
        "residual_structure_after_transform"
    )


def test_external_metadata_prefers_first_configured_provider_root(tmp_path: Path) -> None:
    atom_fqdn = "ageoa.demo.fixture.score_atom"
    first_root = tmp_path / "provider-one"
    second_root = tmp_path / "provider-two"
    _write_metadata(
        first_root,
        atom_fqdn=atom_fqdn,
        heuristic_id="quality_instability",
    )
    _write_metadata(
        second_root,
        atom_fqdn=atom_fqdn,
        heuristic_id="alignment_error",
    )

    metadata = resolve_external_atom_heuristic_metadata(
        atom_fqdn,
        provider_roots=(first_root, second_root),
    )

    assert metadata is not None
    assert metadata.heuristic_outputs[0].heuristic.heuristic_id == "quality_instability"


def test_external_metadata_uses_later_provider_root_when_first_is_missing(
    tmp_path: Path,
) -> None:
    metadata = resolve_external_atom_heuristic_metadata(
        "ageoa.kalman_filters.filter_rs.evaluatemeasurementoracle",
        provider_roots=(
            tmp_path / "missing-provider",
            Path(__file__).resolve().parents[1].parent / "ageo-atoms",
        ),
    )

    assert metadata is not None
    summary = atom_heuristic_metadata_summary(metadata)
    assert summary["logical_atom_id"] == (
        "kalman_filters.filter_rs.evaluatemeasurementoracle"
    )
    assert summary["heuristic_ids"] == ["residual_structure_after_transform"]
