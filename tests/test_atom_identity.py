from __future__ import annotations

from pathlib import Path

from sciona.atom_identity import (
    DEFAULT_PROVIDER_ID,
    atom_provider_id_for_fqdn,
    candidate_atom_provider_roots,
    infer_source_family,
    logical_atom_id_from_fqdn,
)


def test_logical_atom_id_strips_legacy_namespace() -> None:
    assert (
        logical_atom_id_from_fqdn("ageoa.biosppy.ecg.r_peak_detection")
        == "biosppy.ecg.r_peak_detection"
    )


def test_logical_atom_id_strips_federated_namespace() -> None:
    assert (
        logical_atom_id_from_fqdn(
            "sciona.atoms.physics.kalman.filter.updateposteriorstateandcovariance"
        )
        == "physics.kalman.filter.updateposteriorstateandcovariance"
    )


def test_provider_id_infers_legacy_and_federated_prefixes() -> None:
    assert atom_provider_id_for_fqdn("ageoa.biosppy.ecg.r_peak_detection") == (
        DEFAULT_PROVIDER_ID
    )
    assert atom_provider_id_for_fqdn(
        "sciona.atoms.physics.kalman.filter.update"
    ) == "sciona.atoms.physics"


def test_infer_source_family_supports_legacy_and_federated_labels() -> None:
    assert infer_source_family("ageoa.signal.filter_signal_basic") == "ageoa.signal"
    assert infer_source_family(
        "sciona.atoms.fintech.options.charfuncoption"
    ) == "sciona.atoms.fintech"


def test_candidate_provider_roots_include_default_sibling_repo() -> None:
    roots = candidate_atom_provider_roots()
    assert roots
    assert roots[0] == Path("/Users/conrad/personal/ageo-atoms")
