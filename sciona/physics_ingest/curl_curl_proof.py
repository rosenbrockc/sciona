"""Corrected Cartesian Levi-Civita derivation of the curl-curl identity."""
from dataclasses import dataclass
from itertools import product
import sympy as sp
from sciona.physics_ingest.vacuum_wave_proof import curl,divergence,gradient,laplacian,zero

SOURCE_VERSION='080c8bb4-b136-5809-980f-ec6ac28d6fab'
SOURCE_HASH='12d75046fb59a2f06b8f67b51f569e8fbc6104b0d4aba8f198f51636b510607c'


def epsilon_contraction(i,j,m,n):
    return sum(sp.LeviCivita(i,j,k)*sp.LeviCivita(k,m,n) for k in range(3))


def delta_contraction(i,j,m,n):
    return int(i==m)*int(j==n)-int(i==n)*int(j==m)


def check_contraction(candidate=delta_contraction):
    for indices in product(range(3),repeat=4):
        if epsilon_contraction(*indices)!=candidate(*indices):
            raise ValueError('Levi-Civita contraction differs at '+str(indices))
    return 81


@dataclass(frozen=True)
class CurlCurlProof:
    stages: tuple


def build_proof():
    q=sp.symbols('x y z',real=True)
    field=sp.ImmutableMatrix([sp.Function('F'+k)(*q) for k in 'xyz'])
    first_curl=curl(field,q)
    outer=sp.ImmutableMatrix([sum(sp.LeviCivita(i,j,k)*sp.diff(first_curl[k],q[j])
                                 for j,k in product(range(3),repeat=2)) for i in range(3)])
    expanded=sp.ImmutableMatrix([sum(sp.LeviCivita(i,j,k)*sp.LeviCivita(k,m,n)*sp.diff(field[n],q[j],q[m])
                                    for j,k,m,n in product(range(3),repeat=4)) for i in range(3)])
    replaced=sp.ImmutableMatrix([sum(delta_contraction(i,j,m,n)*sp.diff(field[n],q[j],q[m])
                                    for j,m,n in product(range(3),repeat=3)) for i in range(3)])
    separated=sp.ImmutableMatrix([
        sum(int(i==m)*int(j==n)*sp.diff(field[n],q[j],q[m]) for j,m,n in product(range(3),repeat=3))
        -sum(int(i==n)*int(j==m)*sp.diff(field[n],q[j],q[m]) for j,m,n in product(range(3),repeat=3)) for i in range(3)])
    contracted=sp.ImmutableMatrix([sum(sp.diff(field[n],q[n],q[i]) for n in range(3))
                                   -sum(sp.diff(field[i],q[m],2) for m in range(3)) for i in range(3)])
    vector=gradient(divergence(field,q),q)-laplacian(field,q)
    return CurlCurlProof((outer,expanded,replaced,separated,contracted,vector))


def verify_proof(proof):
    count=check_contraction()
    expected=build_proof()
    if len(proof.stages)!=6:
        raise ValueError('Six source transformation stages required')
    for i,(actual,wanted) in enumerate(zip(proof.stages,expected.stages)):
        if not isinstance(actual,sp.MatrixBase) or actual.shape!=(3,1) or not zero(actual-wanted):
            raise ValueError('Stage differs: '+str(i+1))
    for a,b in zip(proof.stages,proof.stages[1:]):
        if not zero(a-b):raise ValueError('Adjacent vector stages differ')
    q=sp.symbols('x y z',real=True)
    field=sp.ImmutableMatrix([sp.Function('F'+k)(*q) for k in 'xyz'])
    if not zero(proof.stages[0]-curl(curl(field,q),q)):
        raise ValueError('Cartesian curl crosscheck differs')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,
                source_ast_parity=False,contraction_cases=count,vector_stages=6,
                stages=[sp.srepr(s) for s in proof.stages],
                identity='sum_k epsilon(i,j,k)*epsilon(k,m,n) = delta(i,m)*delta(j,n)-delta(i,n)*delta(j,m)',
                checks=dict(exhaustive_contraction=True,adjacent_stages=True,cartesian_curl=True),
                domain='C2 three-component fields in a fixed right-handed orthonormal Cartesian frame in Euclidean 3-space; mixed partials commute.',
                limitations='No divergence-free or Maxwell assumption; no curvilinear-coordinate, variable-metric, distributional or boundary-value claim.')
