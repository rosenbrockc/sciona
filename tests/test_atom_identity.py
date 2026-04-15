from __future__ import annotations

import os
from pathlib import Path

from sciona.atom_identity import (
    DEFAULT_PROVIDER_ID,
    DEFAULT_NAMESPACE_PROVIDER_ROOT,
    atom_provider_id_for_fqdn,
    candidate_atom_provider_roots,
    infer_source_family,
    known_atom_package_prefixes,
    logical_atom_id_from_fqdn,
)


def test_logical_atom_id_strips_federated_namespace_in_leaf_path() -> None:
    assert (
        logical_atom_id_from_fqdn(
            "sciona.atoms.signal_processing.biosppy.ecg.r_peak_detection"
        )
        == "signal_processing.biosppy.ecg.r_peak_detection"
    )


def test_logical_atom_id_strips_federated_namespace_in_nested_path() -> None:
    assert (
        logical_atom_id_from_fqdn(
            "sciona.atoms.physics.kalman.filter.updateposteriorstateandcovariance"
        )
        == "physics.kalman.filter.updateposteriorstateandcovariance"
    )


def test_provider_id_infers_canonical_and_federated_prefixes() -> None:
    assert atom_provider_id_for_fqdn(
        "sciona.atoms.signal_processing.biosppy.ecg.r_peak_detection"
    ) == (
        "sciona.atoms.signal_processing"
    )
    assert atom_provider_id_for_fqdn(
        "sciona.atoms.physics.kalman.filter.update"
    ) == "sciona.atoms.physics"
    assert atom_provider_id_for_fqdn("custom.provider.ecg.detect") == (
        DEFAULT_PROVIDER_ID
    )


def test_infer_source_family_supports_canonical_and_federated_labels() -> None:
    assert infer_source_family("sciona.atoms.signal.filter_signal_basic") == "sciona.atoms.signal"
    assert infer_source_family(
        "sciona.atoms.fintech.options.charfuncoption"
    ) == "sciona.atoms.fintech"


def test_known_atom_package_prefixes_parse_and_dedupe_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "SCIONA_ATOM_PACKAGE_PREFIXES",
        " custom.provider. , sciona.atoms , custom.provider , sciona.atoms. ",
    )

    assert known_atom_package_prefixes() == (
        "custom.provider",
        "sciona.atoms",
    )


def test_provider_id_prefers_explicit_prefix_mapping(monkeypatch) -> None:
    monkeypatch.setenv(
        "SCIONA_ATOM_PROVIDER_PREFIX_TO_ID",
        ",".join(
            (
                "sciona.atoms=shared.provider",
                "sciona.atoms.physics=physics.provider",
                "invalid-token",
                "=missing.prefix",
            )
        ),
    )

    assert (
        atom_provider_id_for_fqdn("sciona.atoms.physics.kalman.filter.update")
        == "physics.provider"
    )
    assert (
        atom_provider_id_for_fqdn("sciona.atoms.fintech.options.charfuncoption")
        == "shared.provider"
    )


def test_logical_atom_id_leaves_unknown_namespace_unchanged() -> None:
    assert logical_atom_id_from_fqdn("custom.provider.ecg.detect") == (
        "custom.provider.ecg.detect"
    )


def test_candidate_provider_roots_follow_precedence_and_dedupe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "provider-a"
    second = tmp_path / "provider-b"
    third = tmp_path / "provider-c"
    first.mkdir()
    second.mkdir()
    third.mkdir()

    monkeypatch.setenv(
        "SCIONA_ATOM_PROVIDER_ROOTS",
        os.pathsep.join((str(first), str(second), str(first))),
    )
    monkeypatch.setattr(
        "sciona.atom_identity._sources_yml_provider_roots",
        lambda: (third.resolve(), first.resolve()),
    )

    assert candidate_atom_provider_roots() == (
        first.resolve(),
        second.resolve(),
        third.resolve(),
        DEFAULT_NAMESPACE_PROVIDER_ROOT,
    )


def test_candidate_provider_roots_include_default_sibling_repos(monkeypatch) -> None:
    monkeypatch.setattr("sciona.atom_identity._sources_yml_provider_roots", lambda: ())
    roots = candidate_atom_provider_roots()
    assert roots == (DEFAULT_NAMESPACE_PROVIDER_ROOT,)


def test_candidate_provider_roots_include_sources_yml_roots(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "provider-from-sources"
    source_root.mkdir()
    monkeypatch.delenv("SCIONA_ATOM_PROVIDER_ROOTS", raising=False)
    monkeypatch.setattr(
        "sciona.atom_identity._sources_yml_provider_roots",
        lambda: (source_root.resolve(),),
    )

    roots = candidate_atom_provider_roots()

    assert roots[:2] == (
        source_root.resolve(),
        DEFAULT_NAMESPACE_PROVIDER_ROOT,
    )
