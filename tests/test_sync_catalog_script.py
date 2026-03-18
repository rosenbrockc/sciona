from __future__ import annotations

from pathlib import Path


def test_sync_catalog_runs_verifiers_before_rebuild_steps() -> None:
    script = Path("scripts/sync_catalog.sh").read_text(encoding="utf-8")

    verify_pos = script.index("scripts/verify_atoms_repo.py")
    audit_pos = script.index("scripts/audit.py")
    index_pos = script.index("index build --prover python")
    skill_pos = script.index("skill index --sources-only")

    assert verify_pos < index_pos
    assert audit_pos < index_pos
    assert verify_pos < skill_pos
    assert audit_pos < skill_pos
    assert "MAX_VERIFIER_WARNINGS" in script
