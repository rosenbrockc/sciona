from pathlib import Path
from types import SimpleNamespace

from ageom.cli import _load_skill_index_or_empty


def test_load_skill_index_can_be_disabled(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setenv("AGEOM_DISABLE_SKILL_INDEX", "1")

    config = SimpleNamespace(skill_index_dir=tmp_path)

    def _fail(_path):
        raise AssertionError("SkillIndex.load should not be called when disabled")

    monkeypatch.setattr("ageom.architect.embedder.SkillIndex.load", _fail)

    index = _load_skill_index_or_empty(config)

    assert index is not None
    assert "AGEOM_DISABLE_SKILL_INDEX" in capsys.readouterr().err


def test_load_skill_index_can_be_disabled_by_execution_mode(
    monkeypatch, tmp_path: Path, capsys
):
    monkeypatch.delenv("AGEOM_DISABLE_SKILL_INDEX", raising=False)
    config = SimpleNamespace(skill_index_dir=tmp_path)

    def _fail(_path):
        raise AssertionError("SkillIndex.load should not be called when mode disables it")

    monkeypatch.setattr("ageom.architect.embedder.SkillIndex.load", _fail)

    index = _load_skill_index_or_empty(config, enabled=False)

    assert index is not None
    assert "execution mode" in capsys.readouterr().err


def test_load_skill_index_uses_persisted_index_when_enabled(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("AGEOM_DISABLE_SKILL_INDEX", raising=False)
    marker = object()
    config = SimpleNamespace(skill_index_dir=tmp_path)

    def _load(path):
        assert path == tmp_path
        return marker

    monkeypatch.setattr("ageom.architect.embedder.SkillIndex.load", _load)

    loaded = _load_skill_index_or_empty(config)

    assert loaded is marker
