from __future__ import annotations

from pathlib import Path

import pytest

from sciona.provider_expansion_declarations import (
    clear_provider_expansion_declaration_caches,
)
from sciona.runtime_paths import _declaration_source_lib, _signal_event_rate_declarations


def _write_provider_registry(module_dir: Path) -> None:
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "signal_event_rate_registry.py").write_text(
        "\n".join(
            [
                "SIGNAL_EVENT_RATE_DECLARATIONS = {",
                "    'filter_signal_for_detection': (",
                "        'provider.signal_event_rate.filter_signal_for_detection',",
                "        'np.ndarray, float -> np.ndarray',",
                "        'Provider filter.',",
                "    ),",
                "}",
            ]
        )
    )


@pytest.fixture
def isolated_provider_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    provider_root = tmp_path / "sciona-atoms-signal"
    registry_dir = provider_root / "src" / "sciona" / "atoms" / "expansion"
    _write_provider_registry(registry_dir)
    monkeypatch.setattr(
        "sciona.provider_expansion_declarations.candidate_atom_provider_roots",
        lambda: (provider_root,),
    )
    clear_provider_expansion_declaration_caches()
    _signal_event_rate_declarations.cache_clear()
    try:
        yield provider_root
    finally:
        clear_provider_expansion_declaration_caches()
        _signal_event_rate_declarations.cache_clear()

def test_signal_event_rate_declarations_prefer_provider_registry(
    isolated_provider_registry: Path,
) -> None:
    declarations = _signal_event_rate_declarations()
    declaration_name = declarations["filter_signal_for_detection"][0]
    assert declaration_name == (
        "provider.signal_event_rate.filter_signal_for_detection"
    )
    assert _declaration_source_lib(
        declaration_name,
        fallback="fallback.module",
    ) == "provider.signal_event_rate"
