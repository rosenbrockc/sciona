"""Probe-side catalog for the BioSPPy SVM processing family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeTarget:
    atom_fqdn: str
    module_import_path: str
    wrapper_symbol: str
    parity_expected: bool = True


_MODULE = "sciona.atoms.signal_processing.biosppy.svm_proc.atoms"

SVM_PROC_PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(f"sciona.atoms.signal_processing.biosppy.svm_proc.get_auth_rates", _MODULE, "get_auth_rates"),
    ProbeTarget(f"sciona.atoms.signal_processing.biosppy.svm_proc.get_id_rates", _MODULE, "get_id_rates"),
    ProbeTarget(f"sciona.atoms.signal_processing.biosppy.svm_proc.get_subject_results", _MODULE, "get_subject_results"),
    ProbeTarget(f"sciona.atoms.signal_processing.biosppy.svm_proc.assess_classification", _MODULE, "assess_classification"),
    ProbeTarget(f"sciona.atoms.signal_processing.biosppy.svm_proc.assess_runs", _MODULE, "assess_runs"),
    ProbeTarget(f"sciona.atoms.signal_processing.biosppy.svm_proc.combination", _MODULE, "combination"),
    ProbeTarget(f"sciona.atoms.signal_processing.biosppy.svm_proc.majority_rule", _MODULE, "majority_rule"),
    ProbeTarget(f"sciona.atoms.signal_processing.biosppy.svm_proc.cross_validation", _MODULE, "cross_validation"),
)


def probe_records() -> list[dict[str, object]]:
    return [
        {
            "atom_fqdn": target.atom_fqdn,
            "module_import_path": target.module_import_path,
            "wrapper_symbol": target.wrapper_symbol,
            "parity_expected": target.parity_expected,
        }
        for target in SVM_PROC_PROBE_TARGETS
    ]
