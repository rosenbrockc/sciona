"""Probe-side catalog for the BioSPPy online-filter codex variant."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.signal_processing.biosppy.online_filter_codex.atoms"

ONLINE_FILTER_CODEX_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(
        "sciona.atoms.signal_processing.biosppy.online_filter_codex.filterstateinit",
        _MODULE,
        "filterstateinit",
    ),
    ProbeTarget(
        "sciona.atoms.signal_processing.biosppy.online_filter_codex.filterstep",
        _MODULE,
        "filterstep",
    ),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in ONLINE_FILTER_CODEX_PROBE_TARGETS
    ]
