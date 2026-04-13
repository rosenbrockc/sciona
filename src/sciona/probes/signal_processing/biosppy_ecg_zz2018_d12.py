"""Probe-side catalog for the BioSPPy ECG ZZ2018 d12 leaves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.signal_processing.biosppy.ecg_zz2018_d12"

ECG_ZZ2018_D12_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.assemblezz2018sqi", _MODULE, "assemblezz2018sqi"),
    ProbeTarget(f"{_MODULE}.computebeatagreementsqi", _MODULE, "computebeatagreementsqi"),
    ProbeTarget(f"{_MODULE}.computefrequencysqi", _MODULE, "computefrequencysqi"),
    ProbeTarget(f"{_MODULE}.computekurtosissqi", _MODULE, "computekurtosissqi"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in ECG_ZZ2018_D12_PROBE_TARGETS
    ]
