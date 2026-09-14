"""Six-step isothermal ideal-gas compressibility reconstruction."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION='67defffe-886c-5f47-9d9e-e66f7bba6525'
SOURCE_HASH='49a8a2bb648c4a12b7ddbb73dab65009c7b34c66a52e7b2665367256afe6f202'


@dataclass(frozen=True)
class CompressibilityProof:
    steps: tuple


def build_proof():
    P,V,n,R,T=sp.symbols('P V n R T',positive=True);k=sp.Symbol('kappa',positive=True)
    eq=lambda a,b:sp.Eq(a,b,evaluate=False)
    return CompressibilityProof((
        eq(V,n*R*T/P),
        eq(k,-sp.Derivative(n*R*T/P,P,evaluate=False)/V),
        eq(k,-n*R*T*sp.Derivative(1/P,P,evaluate=False)/V),
        eq(k,n*R*T/(V*P**2)),
        eq(k,sp.Mul(P*V,1/V,1/P**2,evaluate=False)),
        eq(k,1/P),
    ))


def verify_proof(proof):
    expected=build_proof();P,V,n,R,T=sp.symbols('P V n R T',positive=True)
    if len(proof.steps)!=6:raise ValueError('Six transitions required')
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected.steps)):
        if actual!=wanted:raise ValueError('Transition differs: '+str(i+1))
    s=proof.steps
    checks=dict(
        gas_law_division=sp.simplify(P*s[0].rhs-n*R*T)==0,
        definition_substitution=sp.simplify(s[1].rhs.doit()+sp.diff(s[0].rhs,P)/V)==0,
        fixed_temperature_and_amount=sp.simplify(s[1].rhs.doit()-s[2].rhs.doit())==0,
        reciprocal_derivative=sp.simplify(s[2].rhs.doit()-s[3].rhs)==0,
        gas_law_substitution=sp.simplify(s[3].rhs.subs(n*R*T,P*V)-s[4].rhs)==0,
        cancellation=sp.simplify(s[4].rhs-s[5].rhs)==0,
        independent_volume_path=sp.simplify(-sp.diff(n*R*T/P,P)/(n*R*T/P)-s[5].rhs)==0)
    if not all(checks.values()):raise ValueError('Compressibility inference failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,source_ast_parity=False,
                steps=[sp.srepr(s) for s in proof.steps],checks=checks,
                domain='Ideal gas PV=nRT, fixed positive absolute temperature and amount in moles; positive absolute pressure, volume and molar R.',
                limitation='Local isothermal volume compressibility, not isentropic compressibility, bulk modulus or a real-gas equation of state.')
