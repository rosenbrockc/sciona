"""Probe-side catalog for the BioSPPy ECG ZZ2018 leaves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.signal_processing.biosppy.ecg_zz2018"

ECG_ZZ2018_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.calculatecompositesqi_zz2018", _MODULE, "calculatecompositesqi_zz2018"),
    ProbeTarget(f"{_MODULE}.calculatebeatagreementsqi", _MODULE, "calculatebeatagreementsqi"),
    ProbeTarget(f"{_MODULE}.calculatefrequencypowersqi", _MODULE, "calculatefrequencypowersqi"),
    ProbeTarget(f"{_MODULE}.calculatekurtosissqi", _MODULE, "calculatekurtosissqi"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in ECG_ZZ2018_PROBE_TARGETS
    ]
