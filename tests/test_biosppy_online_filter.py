from __future__ import annotations

import importlib

from sciona.probes.signal_processing.biosppy_online_filter import probe_records


def test_biosppy_online_filter_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.signal_processing.biosppy.online_filter") is not None
    assert importlib.import_module("sciona.probes.signal_processing.biosppy_online_filter") is not None


def test_biosppy_online_filter_probe_records_resolve_live_symbols() -> None:
    for record in probe_records():
        module = importlib.import_module(str(record["module_import_path"]))
        assert hasattr(module, str(record["wrapper_symbol"]))
        fqdn_parts = str(record["atom_fqdn"]).split(".")
        imported = importlib.import_module(".".join(fqdn_parts[:-1]))
        assert getattr(imported, fqdn_parts[-1]) is getattr(module, str(record["wrapper_symbol"]))
