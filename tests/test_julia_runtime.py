from __future__ import annotations

import os
import subprocess

from sciona.julia_runtime import configure_juliacall_env, prewarm_juliacall_project


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


def test_prewarm_juliacall_project_instantiates_before_import(tmp_path, monkeypatch):
    project = tmp_path / "julia_project"
    depot = tmp_path / "julia_depot"

    monkeypatch.delenv("PYTHON_JULIAPKG_PROJECT", raising=False)
    monkeypatch.delenv("PYTHON_JULIACALL_PROJECT", raising=False)
    monkeypatch.delenv("JULIA_DEPOT_PATH", raising=False)
    monkeypatch.setenv("PYTHON_JULIACALL_EXE", "/usr/bin/julia")

    captured: dict[str, object] = {}

    def _fake_run(cmd, *, check, env, stdout, stderr):
        captured["cmd"] = list(cmd)
        captured["check"] = check
        captured["env"] = dict(env)
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("sciona.julia_runtime.subprocess.run", _fake_run)

    cfg = prewarm_juliacall_project(project=project, depot=depot)

    bootstrap = str(captured["cmd"][-1])
    instantiate_index = bootstrap.index("Pkg.instantiate()")
    import_index = bootstrap.index("import PythonCall")
    add_index = bootstrap.index('Pkg.add("PythonCall")')
    second_instantiate_index = bootstrap.rindex("Pkg.instantiate()")

    assert cfg.project == project
    assert cfg.depot == depot
    assert instantiate_index < import_index
    assert add_index < second_instantiate_index
    assert captured["env"]["JULIA_DEPOT_PATH"] == str(depot)
    assert captured["env"]["PYTHON_JULIAPKG_PROJECT"] == str(project)
    assert os.environ["PYTHON_JULIACALL_PROJECT"] == str(project)


def test_prewarm_juliacall_project_rebuilds_broken_tmp_runtime(tmp_path, monkeypatch):
    project = tmp_path / "julia_project"
    depot = tmp_path / "julia_depot"
    project.mkdir(parents=True, exist_ok=True)
    depot.mkdir(parents=True, exist_ok=True)
    (project / "stale.txt").write_text("stale", encoding="utf-8")
    (depot / "stale.txt").write_text("stale", encoding="utf-8")

    monkeypatch.delenv("PYTHON_JULIAPKG_PROJECT", raising=False)
    monkeypatch.delenv("PYTHON_JULIACALL_PROJECT", raising=False)
    monkeypatch.delenv("JULIA_DEPOT_PATH", raising=False)
    monkeypatch.setenv("PYTHON_JULIACALL_EXE", "/usr/bin/julia")

    scripts: list[str] = []
    calls = {"count": 0}

    def _fake_run(cmd, *, check, env, stdout, stderr):
        _ = (check, env, stdout, stderr)
        calls["count"] += 1
        scripts.append(str(cmd[-1]))
        if calls["count"] == 1:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("sciona.julia_runtime.subprocess.run", _fake_run)

    cfg = prewarm_juliacall_project(project=project, depot=depot)

    assert cfg.project == project
    assert cfg.depot == depot
    assert calls["count"] == 2
    assert "Pkg.instantiate()" in scripts[0]
    assert 'Pkg.add("PythonCall")' in scripts[1]
    assert not (project / "stale.txt").exists()
    assert not (depot / "stale.txt").exists()
