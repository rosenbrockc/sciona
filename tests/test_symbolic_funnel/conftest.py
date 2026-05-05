"""Synthetic data generators for known physical laws at various noise levels."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from sciona.ghost.symbolic import serialize_expr
from sciona.symbolic_funnel.contracts import FunnelConfig
from sciona.symbolic_funnel.dataset import ColumnMetadata, EmpiricalDataset
from sciona.symbolic_funnel.index import FunnelAtomEntry, FunnelIndex


# ---------------------------------------------------------------------------
# Physical law definitions
# ---------------------------------------------------------------------------

# Dispersion delay: t = K * DM / f^2
K_DISPERSION = 4148.808

# Ideal gas: P*V = n*R*T  =>  P = n*R*T/V
R_GAS = 8.314

# Newton's gravity: F = G*m1*m2/r^2
G_GRAV = 6.674e-11

# Ohm's law: V = I * R  (no constants, all data)


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------


def make_dispersion_dataset(
    n: int = 1000,
    noise_frac: float = 0.0,
    seed: int = 42,
) -> EmpiricalDataset:
    """Generate synthetic data for t = K * DM / f^2."""
    rng = np.random.default_rng(seed)
    DM = rng.uniform(1, 500, n)
    f = rng.uniform(100, 2000, n)
    t_clean = K_DISPERSION * DM / f**2
    t = t_clean * (1 + noise_frac * rng.standard_normal(n))
    return EmpiricalDataset(
        data=np.column_stack([t, DM, f]),
        columns=[
            ColumnMetadata(name="t"),
            ColumnMetadata(name="DM"),
            ColumnMetadata(name="f"),
        ],
    )


def make_ideal_gas_dataset(
    n: int = 1000,
    noise_frac: float = 0.0,
    seed: int = 42,
) -> EmpiricalDataset:
    """Generate synthetic data for P = n*R*T/V."""
    rng = np.random.default_rng(seed)
    n_mol = rng.uniform(0.1, 10, n)
    T = rng.uniform(200, 500, n)
    V = rng.uniform(0.001, 1.0, n)
    P_clean = n_mol * R_GAS * T / V
    P = P_clean * (1 + noise_frac * rng.standard_normal(n))
    return EmpiricalDataset(
        data=np.column_stack([P, V, n_mol, T]),
        columns=[
            ColumnMetadata(name="P"),
            ColumnMetadata(name="V"),
            ColumnMetadata(name="n"),
            ColumnMetadata(name="T"),
        ],
    )


def make_gravity_dataset(
    n: int = 1000,
    noise_frac: float = 0.0,
    seed: int = 42,
) -> EmpiricalDataset:
    """Generate synthetic data for F = G*m1*m2/r^2."""
    rng = np.random.default_rng(seed)
    m1 = rng.uniform(1e20, 1e30, n)
    m2 = rng.uniform(1e20, 1e30, n)
    r = rng.uniform(1e6, 1e12, n)
    F_clean = G_GRAV * m1 * m2 / r**2
    F = F_clean * (1 + noise_frac * rng.standard_normal(n))
    return EmpiricalDataset(
        data=np.column_stack([F, m1, m2, r]),
        columns=[
            ColumnMetadata(name="F"),
            ColumnMetadata(name="m1"),
            ColumnMetadata(name="m2"),
            ColumnMetadata(name="r"),
        ],
    )


def make_ohm_dataset(
    n: int = 1000,
    noise_frac: float = 0.0,
    seed: int = 42,
) -> EmpiricalDataset:
    """Generate synthetic data for V = I * R (no constants)."""
    rng = np.random.default_rng(seed)
    I = rng.uniform(0.01, 10, n)
    R_val = rng.uniform(1, 1000, n)
    V_clean = I * R_val
    V = V_clean * (1 + noise_frac * rng.standard_normal(n))
    return EmpiricalDataset(
        data=np.column_stack([V, I, R_val]),
        columns=[
            ColumnMetadata(name="V"),
            ColumnMetadata(name="I"),
            ColumnMetadata(name="R_val"),
        ],
    )


def make_random_dataset(
    n: int = 1000,
    n_cols: int = 4,
    seed: int = 42,
) -> EmpiricalDataset:
    """Generate random noise — should not match any law."""
    rng = np.random.default_rng(seed)
    names = ["col_" + chr(ord("a") + i) for i in range(n_cols)]
    return EmpiricalDataset(
        data=rng.uniform(0.1, 100, (n, n_cols)),
        columns=[ColumnMetadata(name=name) for name in names],
    )


# ---------------------------------------------------------------------------
# FunnelAtomEntry builders
# ---------------------------------------------------------------------------


def _make_entry(
    name: str,
    eq: sp.Equality,
    variables: dict[str, str],
    constants: dict[str, float],
    exponent_sig: dict[str, str] | None = None,
    invariant_forms: list[dict] | None = None,
    validity_bounds: dict[str, tuple] | None = None,
) -> FunnelAtomEntry:
    """Build a FunnelAtomEntry from a SymPy equation."""
    from sciona.atoms.physics.symbolic_publication_manifest import (
        _canonical_zero_form,
        _exponent_signature,
        _exponent_signature_hash,
        _invariant_forms,
    )

    srepr = serialize_expr(eq)
    eq_hash = _canonical_zero_form(srepr)
    if exponent_sig is None:
        exponent_sig = _exponent_signature(srepr, constants, variables)
    if invariant_forms is None:
        invariant_forms = _invariant_forms(srepr, variables, constants)

    return FunnelAtomEntry(
        expression_id=f"test:{name}",
        atom_name=name,
        atom_module="test",
        srepr_str=srepr,
        variables=variables,
        constants=constants,
        dim_signature={},
        validity_bounds=validity_bounds or {},
        equivalence_class_hash=eq_hash,
        is_equivalence_representative=True,
        exponent_signature=exponent_sig,
        exponent_signature_hash=_exponent_signature_hash(exponent_sig),
        invariant_forms=invariant_forms,
        mechanism_tags=[],
        topology_hash="",
        dimensional_hash="",
    )


@pytest.fixture()
def dispersion_entry() -> FunnelAtomEntry:
    K, DM, f, t = sp.symbols("K DM f t")
    return _make_entry(
        "dispersion_delay",
        sp.Eq(t, K * DM * f ** -2),
        variables={"t": "output", "K": "constant", "DM": "input", "f": "input"},
        constants={"K": K_DISPERSION},
        validity_bounds={"DM": (0.0, None), "f": (0.0, None)},
    )


@pytest.fixture()
def ideal_gas_entry() -> FunnelAtomEntry:
    P, V, n, R, T = sp.symbols("P V n R T")
    return _make_entry(
        "ideal_gas",
        sp.Eq(P, n * R * T / V),
        variables={
            "P": "output",
            "V": "input",
            "n": "input",
            "T": "input",
            "R": "constant",
        },
        constants={"R": R_GAS},
    )


@pytest.fixture()
def gravity_entry() -> FunnelAtomEntry:
    G_sym, m1, m2, r, F = sp.symbols("G m1 m2 r F")
    return _make_entry(
        "newton_gravity",
        sp.Eq(F, G_sym * m1 * m2 * r ** -2),
        variables={
            "F": "output",
            "G": "constant",
            "m1": "input",
            "m2": "input",
            "r": "input",
        },
        constants={"G": G_GRAV},
    )


@pytest.fixture()
def ohm_entry() -> FunnelAtomEntry:
    V, I, R_val = sp.symbols("V I R_val")
    return _make_entry(
        "ohm_law",
        sp.Eq(V, I * R_val),
        variables={"V": "output", "I": "input", "R_val": "input"},
        constants={},
    )


@pytest.fixture()
def test_index(
    dispersion_entry: FunnelAtomEntry,
    ideal_gas_entry: FunnelAtomEntry,
    gravity_entry: FunnelAtomEntry,
    ohm_entry: FunnelAtomEntry,
) -> FunnelIndex:
    """Build a FunnelIndex from the test entries."""
    index = FunnelIndex()
    for entry in [dispersion_entry, ideal_gas_entry, gravity_entry, ohm_entry]:
        index._add(entry)
    return index


@pytest.fixture()
def default_config() -> FunnelConfig:
    return FunnelConfig()
