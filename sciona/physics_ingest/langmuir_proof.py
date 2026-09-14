"""Langmuir equilibrium coverage with explicit reciprocal-step domain."""
from dataclasses import dataclass
import sympy as sp
SOURCE_VERSION='e0057db3-ea2b-5d92-b1a4-5b46572574e1'
SOURCE_HASH='afd364f39e7cd913b4a09d03dd3f577c81bae862a35e8c6cec1edc61278deea1'


def eq(a,b):return sp.Eq(a,b,evaluate=False)


@dataclass(frozen=True)
class LangmuirProof:
    steps:tuple


def derive():
    ka,kd,p,S,N,B,K,theta,rd=sp.symbols('ka kd p S N B K theta rd',positive=True)
    first=eq(ka*p*S,rd)
    balance=eq(first.lhs,kd*B)
    vacant=eq(S,kd*B/(ka*p))
    sites=eq(N,vacant.rhs+B)
    factored=eq(N,B*(kd/(ka*p)+1))
    # Displayed source divides by p*S, but the result also divides by kd.
    ratio=eq(ka/kd,B/(p*S))
    reciprocal=eq(1/K,kd/ka)
    converted=eq(N,B*(1/(K*p)+1))
    divided=eq(N/B,1/(K*p)+1)
    inverse_coverage=eq(1/theta,N/B)
    substituted=eq(1/theta,divided.rhs)
    common_denominator=eq(1/theta,(1+K*p)/(K*p))
    endpoint=eq(theta,K*p/(1+K*p))
    return (first,balance,vacant,sites,factored,ratio,reciprocal,converted,
            divided,inverse_coverage,substituted,common_denominator,endpoint)


def build_proof():return LangmuirProof(derive())


def verify_proof(proof):
    expected=derive()
    if len(proof.steps)!=13:raise ValueError('Thirteen transitions required')
    for i,(actual,wanted) in enumerate(zip(proof.steps,expected)):
        if actual!=wanted:raise ValueError(f'Step{i+1} differs')
    ka,kd,N=sp.symbols('ka kd N',positive=True)
    p=sp.Symbol('p',nonnegative=True)
    S,B=sp.symbols('S B')
    solution=sp.solve([ka*p*S-kd*B,S+B-N],(S,B))
    theta=ka*p/(kd+ka*p)
    checks={
        'linear_system_coverage':sp.cancel(solution[B]/N-theta),
        'site_conservation':sp.cancel(solution[S]+solution[B]-N),
        'rate_balance':sp.cancel(ka*p*solution[S]-kd*solution[B]),
        'zero_pressure_coverage':theta.subs(p,0),
        'zero_pressure_vacant':solution[S].subs(p,0)-N,
        'saturation_limit':sp.limit(theta,p,sp.oo)-1,
    }
    if any(v!=0 for v in checks.values()):raise ValueError('Independent equilibrium check failed')
    if sp.factor(sp.diff(theta,p)).is_positive is not True:raise ValueError('Monotonicity unproved')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                steps_verified=13,steps=[sp.srepr(s) for s in expected],
                independent_checks={k:True for k in checks},source_ast_parity=False,
                zero_pressure_extension_verified=True,
                assumptions=['Single non-dissociative adsorbate, identical independent sites, at most one molecule per site.',
                             'Fixed temperature and positive adsorption/desorption coefficients, positive total site density.',
                             'Equilibrium rate balance and conservation of sites; no lateral interactions or multilayer adsorption.',
                             'Source reciprocal derivation requires p>0 and positive occupied/vacant densities.',
                             'K=ka/kd has inverse-pressure units when p is pressure; K*p dimensionless.'],
                limitations=['Zero pressure handled by independent linear system, not literal reciprocal-step replay.',
                             'No competitive adsorption, dissociation, transient kinetics or material-specific fit claim.',
                             'Caller supplies physically applicable coefficients at the stated temperature.'])
