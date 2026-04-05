from __future__ import annotations

import os

from sciona.julia_runtime import configure_juliacall_env


def test_configure_juliacall_env_sets_writable_defaults(tmp_path, monkeypatch):
    project = tmp_path / "julia_project"
    depot = tmp_path / "julia_depot"

    monkeypatch.delenv("PYTHON_JULIAPKG_PROJECT", raising=False)
    monkeypatch.delenv("PYTHON_JULIACALL_PROJECT", raising=False)
    monkeypatch.delenv("JULIA_DEPOT_PATH", raising=False)
    monkeypatch.setenv("PYTHON_JULIACALL_EXE", "/usr/bin/julia")

    cfg = configure_juliacall_env(project=project, depot=depot)

    assert cfg.project == project
    assert cfg.depot == depot
    assert project.is_dir()
    assert depot.is_dir()
    assert os.environ["PYTHON_JULIAPKG_PROJECT"] == str(project)
    assert os.environ["JULIA_DEPOT_PATH"] == str(depot)
    assert os.environ["PYTHON_JULIACALL_EXE"] == "/usr/bin/julia"
    assert "PYTHON_JULIACALL_PROJECT" not in os.environ


def test_configure_juliacall_env_exposes_project_when_pythoncall_is_declared(
    tmp_path, monkeypatch
):
    project = tmp_path / "julia_project"
    depot = tmp_path / "julia_depot"
    project.mkdir(parents=True, exist_ok=True)
    (project / "Project.toml").write_text(
        "[deps]\nPythonCall = \"6099a3de-4c0c-538f-b890-85f7d6d5f1b6\"\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("PYTHON_JULIAPKG_PROJECT", raising=False)
    monkeypatch.delenv("PYTHON_JULIACALL_PROJECT", raising=False)
    monkeypatch.delenv("JULIA_DEPOT_PATH", raising=False)
    monkeypatch.setenv("PYTHON_JULIACALL_EXE", "/usr/bin/julia")

    cfg = configure_juliacall_env(project=project, depot=depot)

    assert cfg.project == project
    assert os.environ["PYTHON_JULIAPKG_PROJECT"] == str(project)
    assert os.environ["PYTHON_JULIACALL_PROJECT"] == str(project)
