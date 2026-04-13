"""Probe-side catalog for the BioSPPy ECG detector leaf."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.signal_processing.biosppy.ecg_detectors"

ECG_DETECTORS_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"{_MODULE}.thresholdbasedsignalsegmentation", _MODULE, "thresholdbasedsignalsegmentation"),
    ProbeTarget(f"{_MODULE}.asi_signal_segmenter", _MODULE, "asi_signal_segmenter"),
    ProbeTarget(f"{_MODULE}.christovqrsdetect", _MODULE, "christovqrsdetect"),
    ProbeTarget(f"{_MODULE}.christov_qrs_segmenter", _MODULE, "christov_qrs_segmenter"),
    ProbeTarget(f"{_MODULE}.engzee_signal_segmentation", _MODULE, "engzee_signal_segmentation"),
    ProbeTarget(f"{_MODULE}.engzee_qrs_segmentation", _MODULE, "engzee_qrs_segmentation"),
    ProbeTarget(f"{_MODULE}.gamboa_segmentation", _MODULE, "gamboa_segmentation"),
    ProbeTarget(f"{_MODULE}.gamboa_segmenter", _MODULE, "gamboa_segmenter"),
    ProbeTarget(f"{_MODULE}.hamilton_segmentation", _MODULE, "hamilton_segmentation"),
    ProbeTarget(f"{_MODULE}.hamilton_segmenter", _MODULE, "hamilton_segmenter"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in ECG_DETECTORS_PROBE_TARGETS
    ]
