"""Local differential and fixed-pressure path reconstruction of the source."""
from dataclasses import dataclass
import sympy as sp
from sciona.ghost.symbolic import serialize_expr

SOURCE_VERSION='a8f28449-1e31-5c15-b514-b632020fc1a8'
SOURCE_HASH='af502248823bbbd0d4a43f09cd66173c7bdf82d448b3d793d90bfe727a3e7451'


def symbols():
    names='u_T u_V cv pi volume alpha dU dT dV U_path T_path V_path'
    return dict(zip(names.split(),sp.symbols(names,real=True)))


@dataclass(frozen=True)
class IsobaricProof:
    premises: tuple
    steps: tuple


def build_proof():
    s=symbols();ut,uv,cv,pi,V,alpha,dU,dT,dV,U_path,T_path,V_path=s.values()
    return IsobaricProof(
        (sp.Eq(cv,ut,evaluate=False),sp.Eq(pi,uv,evaluate=False),
         sp.Eq(dU,ut*dT+uv*dV,evaluate=False),sp.Eq(alpha,V_path/V,evaluate=False)),
        (sp.Eq(dU,cv*dT+pi*dV,evaluate=False),sp.Eq(U_path,cv*T_path+pi*V_path,evaluate=False),
         sp.Eq(V*alpha,V_path,evaluate=False),sp.Eq(U_path,cv*T_path+pi*V*alpha,evaluate=False),
         sp.Eq(U_path,cv+pi*V*alpha,evaluate=False)))


def verify_proof(proof):
    expected=build_proof();s=symbols()
    if proof.premises!=expected.premises or len(proof.steps)!=5:
        raise ValueError('Four local derivative premises and five transitions required')
    differential=proof.premises[2]
    first=sp.Eq(differential.lhs,differential.rhs.subs({s['u_T']:s['cv'],s['u_V']:s['pi']}),evaluate=False)
    if proof.steps[0]!=first:raise ValueError('Substitute both derivative definitions')
    # Pull back the linear differential to a fixed-pressure path. This gives
    # meaning to the source shorthand of division by dT, without treating
    # differentials as finite numeric increments.
    path={s['dU']:s['U_path'],s['dT']:s['T_path'],s['dV']:s['V_path']}
    second=sp.Eq(first.lhs.xreplace(path),first.rhs.xreplace(path),evaluate=False)
    if proof.steps[1]!=second:raise ValueError('Path derivative differs')
    third=sp.Eq(proof.premises[3].lhs*s['volume'],proof.premises[3].rhs*s['volume'],evaluate=False)
    if proof.steps[2]!=third:raise ValueError('Expansion definition requires nonzero volume')
    fourth=sp.Eq(second.lhs,second.rhs.subs(third.rhs,third.lhs),evaluate=False)
    if proof.steps[3]!=fourth:raise ValueError('Substitute isobaric volume derivative')
    fifth=sp.Eq(fourth.lhs,fourth.rhs.subs(s['T_path'],1),evaluate=False)
    if proof.steps[4]!=fifth or fifth!=expected.steps[4]:raise ValueError('Temperature parametrization differs')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,source_ast_parity=False,
                premises=[serialize_expr(e) for e in proof.premises],steps=[serialize_expr(e) for e in proof.steps],
                checks=dict(coefficient_substitution=True,path_chain_rule=True,expansion_definition=True,temperature_parametrization=True),
                domain='Differentiable U(T,V) and a fixed-pressure path V(T,p), positive V, fixed composition/amount; coefficients evaluated at the same state.',
                dimensional_correction='Source pi_T metadata encodes force; its definition dU/dV requires pressure units J/m^3.',
                limitation='Output is dU/dT at fixed pressure, not dH/dT or Cp.')
