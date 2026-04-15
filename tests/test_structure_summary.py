from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    NodeStatus,
)
from sciona.principal.structure_summary import _primitive_family, summarize_trial_structure


def _atomic_node(
    node_id: str,
    *,
    concept_type: ConceptType,
    matched_primitive: str,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=node_id,
        description=f"Atomic node {node_id}",
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive,
    )


def test_primitive_family_handles_recognized_atom_namespaces_generically() -> None:
    assert _primitive_family(
        "sciona.atoms.signal_processing.biosppy.ecg.r_peak_detection", None
    ) == (
        "sciona.atoms.signal_processing.biosppy"
    )
    assert _primitive_family("sciona.atoms.demo.ecg.r_peak_detection", None) == (
        "sciona.atoms.demo.ecg"
    )


def test_structure_summary_tracks_distinct_sciona_atom_families_without_catalog() -> None:
    cdg = CDGExport(
        nodes=[
            _atomic_node(
                "filter_signal",
                concept_type=ConceptType.SIGNAL_FILTER,
                matched_primitive="sciona.atoms.demo.filters.bandpass_filter",
            ),
            _atomic_node(
                "detect_events",
                concept_type=ConceptType.ANALYSIS,
                matched_primitive="sciona.atoms.demo.ecg.r_peak_detection",
            ),
        ],
        edges=[
            DependencyEdge(
                source_id="filter_signal",
                target_id="detect_events",
                output_name="filtered_signal",
                input_name="signal",
                source_type="np.ndarray",
                target_type="np.ndarray",
            )
        ],
    )

    summary = summarize_trial_structure(cdg)

    assert summary["distinct_primitive_families"] == [
        "sciona.atoms.demo.ecg",
        "sciona.atoms.demo.filters",
    ]
    assert summary["distinct_primitive_family_count"] == 2
    assert summary["cross_family_edge_count"] == 1
