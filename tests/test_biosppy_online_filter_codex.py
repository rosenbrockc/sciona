from __future__ import annotations

import importlib


def test_biosppy_online_filter_codex_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.biosppy.online_filter_codex")
    probes = importlib.import_module("sciona.probes.signal_processing.biosppy_online_filter_codex")
    assert hasattr(atoms, "filterstateinit")
    assert hasattr(atoms, "filterstep")
    assert hasattr(probes, "ONLINE_FILTER_CODEX_PROBE_TARGETS")
