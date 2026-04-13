from __future__ import annotations

from pathlib import Path

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, IOSpec, NodeStatus
from sciona.runtime_paths import (
    _build_signal_event_rate_match_results,
    _is_signal_event_rate_scaffold,
    _signal_event_rate_declarations,
)
from sciona.types import Prover


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
        "sciona.runtime_paths.candidate_atom_provider_roots",
        lambda: (provider_root,),
    )
    _signal_event_rate_declarations.cache_clear()
    try:
        yield provider_root
    finally:
        _signal_event_rate_declarations.cache_clear()


def _signal_event_rate_cdg() -> CDGExport:
    node = AlgorithmicNode(
        node_id="filt",
        name="Filter",
        description="filter signal",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        matched_primitive="filter_signal_for_detection",
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
    )
    return CDGExport(nodes=[node], edges=[])


def test_signal_event_rate_declarations_prefer_provider_registry(
    isolated_provider_registry: Path,
) -> None:
    declarations = _signal_event_rate_declarations()
    assert declarations["filter_signal_for_detection"][0] == (
        "provider.signal_event_rate.filter_signal_for_detection"
    )


def test_signal_event_rate_scaffold_uses_provider_registry(
    isolated_provider_registry: Path,
) -> None:
    assert _is_signal_event_rate_scaffold(_signal_event_rate_cdg())


def test_signal_event_rate_match_results_use_provider_registry(
    isolated_provider_registry: Path,
) -> None:
    results = _build_signal_event_rate_match_results(
        _signal_event_rate_cdg(),
        Prover.PYTHON,
    )

    assert len(results) == 1
    assert results[0].all_candidates[0].declaration.name == (
        "provider.signal_event_rate.filter_signal_for_detection"
    )
    assert results[0].all_candidates[0].declaration.source_lib == (
        "provider.signal_event_rate"
    )
