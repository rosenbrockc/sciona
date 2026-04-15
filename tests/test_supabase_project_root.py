from __future__ import annotations

from pathlib import Path

from tests import conftest


def test_supabase_project_root_prefers_env(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "sciona-matcher"
    infra_root = tmp_path / "sciona-infra"
    custom_root = tmp_path / "custom-supabase-root"
    (repo_root / "supabase").mkdir(parents=True)
    (infra_root / "supabase").mkdir(parents=True)
    (custom_root / "supabase").mkdir(parents=True)
    (infra_root / "supabase" / "config.toml").write_text('project_id = "infra"\n')
    (custom_root / "supabase" / "config.toml").write_text('project_id = "custom"\n')
    monkeypatch.setenv("SCIONA_SUPABASE_PROJECT_ROOT", str(custom_root))

    resolved = conftest._supabase_project_root(repo_root)

    assert resolved == custom_root.resolve()


def test_supabase_project_root_defaults_to_infra_when_available(tmp_path: Path) -> None:
    repo_root = tmp_path / "sciona-matcher"
    infra_root = tmp_path / "sciona-infra"
    (repo_root / "supabase").mkdir(parents=True)
    (infra_root / "supabase").mkdir(parents=True)
    (infra_root / "supabase" / "config.toml").write_text('project_id = "infra"\n')

    resolved = conftest._supabase_project_root(repo_root)

    assert resolved == infra_root.resolve()


def test_supabase_project_root_falls_back_to_repo_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "sciona-matcher"
    (repo_root / "supabase").mkdir(parents=True)

    resolved = conftest._supabase_project_root(repo_root)

    assert resolved == repo_root.resolve()


def test_supabase_project_id_reads_selected_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "sciona-infra"
    (project_root / "supabase").mkdir(parents=True)
    (project_root / "supabase" / "config.toml").write_text('project_id = "sciona-infra"\n')

    project_id = conftest._supabase_project_id(project_root)

    assert project_id == "sciona-infra"
