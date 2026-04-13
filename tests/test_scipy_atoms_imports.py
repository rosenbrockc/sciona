from __future__ import annotations

import importlib


def test_versionless_scipy_atoms_import() -> None:
    linalg = importlib.import_module("sciona.atoms.scipy.linalg")
    optimize = importlib.import_module("sciona.atoms.scipy.optimize")

    assert linalg.solve.__name__ == "solve"
    assert linalg.inv.__name__ == "inv"
    assert optimize.minimize.__name__ == "minimize"
    assert optimize.shgo.__name__ == "shgo"
    assert optimize.differential_evolution.__name__ == "differential_evolution"


def test_versionless_scipy_probes_import() -> None:
    probes = importlib.import_module("sciona.probes.scipy.witnesses")

    assert hasattr(probes, "witness_scipy_linalg_solve")
    assert hasattr(probes, "witness_scipy_minimize")
    assert hasattr(probes, "witness_scipy_shgo")
