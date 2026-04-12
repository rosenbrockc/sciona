from __future__ import annotations

from pathlib import Path

from sciona.heuristic_metadata import (
    atom_heuristic_metadata_summary,
    clear_atom_heuristic_metadata_caches,
    resolve_external_atom_heuristic_metadata,
)


def test_src_layout_namespace_provider_metadata_resolves_sciona_kalman_record() -> None:
    provider_root = Path(__file__).resolve().parents[1].parent / "sciona-atoms"
    atom_fqdn = (
        "sciona.atoms.state_estimation.kalman_filters."
        "filter_rs.evaluate_measurement_oracle"
    )

    clear_atom_heuristic_metadata_caches()
    try:
        metadata = resolve_external_atom_heuristic_metadata(
            atom_fqdn,
            provider_roots=(provider_root,),
        )
    finally:
        clear_atom_heuristic_metadata_caches()

    assert metadata is not None
    summary = atom_heuristic_metadata_summary(metadata)
    assert summary["atom_fqdn"] == atom_fqdn
    assert summary["logical_atom_id"] == (
        "state_estimation.kalman_filters.filter_rs.evaluate_measurement_oracle"
    )
    assert summary["provider_id"] == "sciona.atoms.state_estimation"
    assert summary["heuristic_ids"] == ["residual_structure_after_transform"]
