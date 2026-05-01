from __future__ import annotations

from sciona.physics_ingest.bindings import resolve_publication_artifact_bindings
from sciona.physics_ingest.publication import ArtifactBinding


ARTIFACT_A = "20000000-0000-0000-0000-000000000001"
VERSION_A = "30000000-0000-0000-0000-000000000001"
VERSION_A_OLD = "30000000-0000-0000-0000-000000000099"
ARTIFACT_B = "20000000-0000-0000-0000-000000000002"
VERSION_B = "30000000-0000-0000-0000-000000000002"
ARTIFACT_C = "20000000-0000-0000-0000-000000000003"


def test_resolves_manifest_keys_to_publication_artifact_bindings() -> None:
    result = resolve_publication_artifact_bindings(
        [
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.force",
                "local_artifact_key": "local:sciona-atoms-physics:fixture.force",
                "fqdn": "sciona.atoms.physics.fixture.force",
                "registry_name": "force_atom",
                "atom_name": "force_atom",
            },
            {"artifact_key": "local:sciona-atoms-physics:fixture.angle"},
        ],
        _artifact_rows(),
        _version_rows(),
    )

    assert result.diagnostics == ()
    assert result.bindings[
        "local:sciona-atoms-physics:fixture.force"
    ] == ArtifactBinding(artifact_id=ARTIFACT_A, version_id=VERSION_A)
    assert result.bindings["sciona.atoms.physics.fixture.force"] == ArtifactBinding(
        artifact_id=ARTIFACT_A,
        version_id=VERSION_A,
    )
    assert result.bindings["force_atom"] == ArtifactBinding(
        artifact_id=ARTIFACT_A,
        version_id=VERSION_A,
    )
    assert result.to_publication_bindings()["local:sciona-atoms-physics:fixture.angle"] == (
        ArtifactBinding(artifact_id=ARTIFACT_B, version_id=VERSION_B)
    )


def test_resolves_explicit_version_selectors_from_source_bundle_rows() -> None:
    result = resolve_publication_artifact_bindings(
        [
            {
                "atom_name": "force_atom",
                "semver": "0.9.0",
            },
            {
                "atom_name": "angle_atom",
                "version_id": VERSION_B,
            },
        ],
        _artifact_rows(),
        _version_rows(),
    )

    assert result.diagnostics == ()
    assert result.bindings["force_atom"] == ArtifactBinding(
        artifact_id=ARTIFACT_A,
        version_id=VERSION_A_OLD,
    )
    assert result.bindings["angle_atom"] == ArtifactBinding(
        artifact_id=ARTIFACT_B,
        version_id=VERSION_B,
    )


def test_reports_missing_and_ambiguous_bindings_without_db_calls() -> None:
    result = resolve_publication_artifact_bindings(
        [
            {"artifact_key": "local:missing"},
            {"artifact_key": "local:ambiguous"},
            {"atom_name": "multi_version"},
            {"registry_name": "no_version"},
        ],
        [
            *_artifact_rows(),
            {
                "artifact_id": ARTIFACT_A,
                "artifact_key": "local:ambiguous",
                "atom_name": "ambiguous_a",
            },
            {
                "artifact_id": ARTIFACT_B,
                "artifact_key": "local:ambiguous",
                "atom_name": "ambiguous_b",
            },
            {
                "artifact_id": ARTIFACT_C,
                "atom_name": "multi_version",
                "registry_name": "no_version",
            },
        ],
        [
            *_version_rows(),
            {
                "artifact_id": ARTIFACT_C,
                "version_id": "30000000-0000-0000-0000-000000000003",
                "semver": "1.0.0",
                "is_latest": False,
            },
            {
                "artifact_id": ARTIFACT_C,
                "version_id": "30000000-0000-0000-0000-000000000004",
                "semver": "2.0.0",
                "is_latest": False,
            },
        ],
    )

    assert result.bindings == {}
    assert [(row.row_index, row.reason) for row in result.diagnostics] == [
        (0, "missing_artifact"),
        (1, "ambiguous_artifact"),
        (2, "ambiguous_version"),
        (3, "ambiguous_version"),
    ]
    assert result.diagnostics[1].artifact_ids == (ARTIFACT_A, ARTIFACT_B)
    assert result.diagnostics[2].version_ids == (
        "30000000-0000-0000-0000-000000000003",
        "30000000-0000-0000-0000-000000000004",
    )


def test_reports_key_collisions_without_overwriting_existing_binding() -> None:
    result = resolve_publication_artifact_bindings(
        [
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.force",
                "atom_name": "force_atom",
            },
            {
                "artifact_key": "local:sciona-atoms-physics:fixture.angle",
                "atom_name": "force_atom",
            },
        ],
        _artifact_rows(),
        _version_rows(),
    )

    assert result.bindings["force_atom"] == ArtifactBinding(
        artifact_id=ARTIFACT_A,
        version_id=VERSION_A,
    )
    assert [
        (row.row_index, row.reason, row.key_value) for row in result.diagnostics
    ] == [(1, "ambiguous_artifact", "local:sciona-atoms-physics:fixture.angle")]


def _artifact_rows() -> list[dict[str, object]]:
    return [
        {
            "artifact_id": ARTIFACT_A,
            "artifact_key": "local:sciona-atoms-physics:fixture.force",
            "local_artifact_key": "local:sciona-atoms-physics:fixture.force",
            "fqdn": "sciona.atoms.physics.fixture.force",
            "registry_name": "force_atom",
            "atom_name": "force_atom",
        },
        {
            "artifact_id": ARTIFACT_B,
            "artifact_key": "local:sciona-atoms-physics:fixture.angle",
            "local_artifact_key": "local:sciona-atoms-physics:fixture.angle",
            "fqdn": "sciona.atoms.physics.fixture.angle",
            "registry_name": "angle_atom",
            "atom_name": "angle_atom",
        },
    ]


def _version_rows() -> list[dict[str, object]]:
    return [
        {
            "artifact_id": ARTIFACT_A,
            "version_id": VERSION_A,
            "semver": "1.0.0",
            "content_hash": "hash-a",
            "is_latest": True,
        },
        {
            "artifact_id": ARTIFACT_A,
            "version_id": VERSION_A_OLD,
            "semver": "0.9.0",
            "content_hash": "hash-a-old",
            "is_latest": False,
        },
        {
            "artifact_id": ARTIFACT_B,
            "version_id": VERSION_B,
            "semver": "1.0.0",
            "content_hash": "hash-b",
            "is_latest": False,
        },
    ]
