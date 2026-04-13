from __future__ import annotations

import importlib

from sciona.probes.signal_processing.biosppy_ecg_zz2018 import probe_records


def test_biosppy_ecg_zz2018_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.signal_processing.biosppy.ecg_zz2018") is not None
    assert importlib.import_module("sciona.probes.signal_processing.biosppy_ecg_zz2018") is not None


def test_biosppy_ecg_zz2018_probe_records_resolve_live_symbols() -> None:
    for record in probe_records():
        module = importlib.import_module(str(record["module_import_path"]))
        assert hasattr(module, str(record["wrapper_symbol"]))
        fqdn_parts = str(record["atom_fqdn"]).split(".")
        imported = importlib.import_module(".".join(fqdn_parts[:-1]))
        assert getattr(imported, fqdn_parts[-1]) is getattr(module, str(record["wrapper_symbol"]))
