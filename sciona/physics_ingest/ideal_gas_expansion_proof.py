"""Reconstruct the five-step ideal-gas expansion derivation."""
from dataclasses import dataclass
import sympy as sp
from sciona.ghost.symbolic import serialize_expr

SOURCE_VERSION = '060b8ea0-d956-5fbc-9224-a31ccb1229e9'
SOURCE_HASH = '000f7f4e8a1c10b86186f4c34d61bca916f2b721bf6d9f3f31d08dfeeb55bf7c'


def symbols():
    return sp.symbols('T P V n R', positive=True)


@dataclass(frozen=True)
class ExpansionProof:
    steps: tuple


def build_proof():
    T, P, V, n, R = symbols()
    alpha = sp.Symbol('alpha', positive=True)
    return ExpansionProof((
        sp.Eq(alpha, sp.Mul(1/V, n*R/P, sp.Derivative(T,T,evaluate=False), evaluate=False), evaluate=False),
        sp.Eq(alpha, n*R/(V*P), evaluate=False),
        sp.Eq(P*V/T, n*R, evaluate=False),
        sp.Eq(alpha, sp.Mul(P*V/T, 1/(V*P), evaluate=False), evaluate=False),
        sp.Eq(alpha, 1/T, evaluate=False),
    ))


def verify_proof(proof):
    T, P, V, n, R = symbols()
    alpha = sp.Symbol('alpha', positive=True)
    if len(proof.steps) != 5:
        raise ValueError('Five source transitions required')
    # V(T) is determined by the ideal gas law at fixed P,n and molar R.
    volume_path = n*R*T/P
    path_derivative = sp.diff(volume_path,T)
    expected = build_proof()
    for i, (step, target) in enumerate(zip(proof.steps, expected.steps)):
        if step != target:
            raise ValueError('Source transition differs: '+str(i+1))
    first, second, third, fourth, fifth = proof.steps
    if sp.simplify(first.rhs.doit()-path_derivative/V) != 0:
        raise ValueError('Fixed-pressure derivative substitution failed')
    if sp.simplify(first.rhs.doit()-second.rhs) != 0:
        raise ValueError('Temperature self-derivative failed')
    gas_law = sp.Eq(P*V,n*R*T,evaluate=False)
    if sp.simplify(third.lhs-gas_law.lhs/T) != 0 or sp.simplify(third.rhs-gas_law.rhs/T) != 0:
        raise ValueError('Gas law division failed')
    substituted = second.rhs.subs(n*R,third.lhs)
    if sp.simplify(substituted-fourth.rhs) != 0 or sp.simplify(fourth.rhs-fifth.rhs) != 0:
        raise ValueError('Gas law substitution/cancellation failed')
    # Separate direct logarithmic-volume derivative establishes the endpoint.
    if sp.simplify(sp.diff(volume_path,T)/volume_path-fifth.rhs) != 0:
        raise ValueError('Independent volume-path derivative failed')
    return dict(source_version_id=SOURCE_VERSION, source_content_hash=SOURCE_HASH,
                source_ast_parity=False, steps=[serialize_expr(s) for s in proof.steps],
                fixed_pressure_path=serialize_expr(volume_path),
                checks=dict(five_transitions=True, fixed_pressure_derivative=True,
                            gas_law_substitution=True, independent_volume_derivative=True),
                domain='Ideal gas, fixed positive pressure and amount in moles, positive absolute temperature and molar gas constant; volume obeys PV=nRT.',
                corrections=['Explicit differential semantics', 'Restore missing 1/(VP) in intermediate AST', 'Molar amount dimension N1'],
                limitation='Volumetric isobaric expansion coefficient, not linear expansion or a real-gas equation of state.')
