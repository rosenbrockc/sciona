from __future__ import annotations

from sciona.asset_migration import MigrationReadinessAsset, migration_readiness_summary


def test_migration_readiness_defaults_to_sciona_atoms_target_repository() -> None:
    readiness = MigrationReadinessAsset()

    summary = migration_readiness_summary(readiness)

    assert summary["migration_readiness_target_repository"] == "../sciona-atoms"
