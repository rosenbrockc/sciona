from __future__ import annotations

import os
from pathlib import Path

from sciona.atom_identity import (
    DEFAULT_PROVIDER_ID,
    DEFAULT_NAMESPACE_PROVIDER_ROOT,
    DEFAULT_PROVIDER_ROOT,
    atom_provider_id_for_fqdn,
    candidate_atom_provider_roots,
    infer_source_family,
    known_atom_package_prefixes,
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


def test_known_atom_package_prefixes_parse_and_dedupe_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "SCIONA_ATOM_PACKAGE_PREFIXES",
        " custom.provider. , ageoa , custom.provider , sciona.atoms. ",
    )

    assert known_atom_package_prefixes() == (
        "custom.provider",
        "ageoa",
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
                "ageoa=legacy.provider",
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
    assert atom_provider_id_for_fqdn("ageoa.biosppy.ecg.r_peak_detection") == (
        "legacy.provider"
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
    monkeypatch.setenv("SCIONA_AGEO_ATOMS_ROOT", str(second))
    monkeypatch.setattr(
        "sciona.atom_identity._sources_yml_provider_roots",
        lambda: (third.resolve(), first.resolve()),
    )

    assert candidate_atom_provider_roots() == (
        first.resolve(),
        second.resolve(),
        third.resolve(),
        DEFAULT_NAMESPACE_PROVIDER_ROOT,
        DEFAULT_PROVIDER_ROOT,
    )


def test_candidate_provider_roots_include_default_sibling_repos(monkeypatch) -> None:
    monkeypatch.setattr("sciona.atom_identity._sources_yml_provider_roots", lambda: ())
    roots = candidate_atom_provider_roots()
    assert roots[:2] == (
        DEFAULT_NAMESPACE_PROVIDER_ROOT,
        DEFAULT_PROVIDER_ROOT,
    )


def test_candidate_provider_roots_include_sources_yml_roots(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "provider-from-sources"
    source_root.mkdir()
    monkeypatch.delenv("SCIONA_ATOM_PROVIDER_ROOTS", raising=False)
    monkeypatch.delenv("SCIONA_AGEO_ATOMS_ROOT", raising=False)
    monkeypatch.setattr(
        "sciona.atom_identity._sources_yml_provider_roots",
        lambda: (source_root.resolve(),),
    )

    roots = candidate_atom_provider_roots()

    assert roots[:3] == (
        source_root.resolve(),
        DEFAULT_NAMESPACE_PROVIDER_ROOT,
        DEFAULT_PROVIDER_ROOT,
    )
