from __future__ import annotations

import importlib


def test_biosppy_ecg_detectors_import_smoke() -> None:
    atoms = importlib.import_module("sciona.atoms.signal_processing.biosppy.ecg_detectors")
    probes = importlib.import_module("sciona.probes.signal_processing.biosppy_ecg_detectors")
    assert hasattr(atoms, "hamilton_segmenter")
    assert hasattr(atoms, "christov_qrs_segmenter")
    assert hasattr(probes, "ECG_DETECTORS_PROBE_TARGETS")
