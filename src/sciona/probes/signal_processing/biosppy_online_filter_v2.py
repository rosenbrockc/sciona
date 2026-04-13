"""Probe-side catalog for the BioSPPy online filter v2 leaves."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.signal_processing.biosppy.online_filter_v2"

ONLINE_FILTER_V2_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.filterstateinit", _MODULE, "filterstateinit"),
    ProbeTarget(f"{_MODULE}.filterstep", _MODULE, "filterstep"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in ONLINE_FILTER_V2_PROBE_TARGETS
    ]
