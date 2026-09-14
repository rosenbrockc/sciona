"""Dependency-ordered reconstruction of frequency/wavelength/wavenumber relations."""
from dataclasses import dataclass
import sympy as sp

SOURCE_VERSION='0873a091-c6f6-56a1-a80d-976b59b95a8c'
SOURCE_HASH='aee08ee2103f0721af8363fcb34fe5b7f5ad172697c3824800d73473370686cf'


def symbols():
    return sp.symbols('f period wavelength speed omega k',positive=True)


@dataclass(frozen=True)
class WaveRelationsProof:
    steps: tuple


def build_proof():
    f,T,L,v,w,k=symbols();eq=lambda a,b:sp.Eq(a,b,evaluate=False)
    # Stored source order, with step 1 depending on step 6.
    return WaveRelationsProof((eq(k,2*sp.pi/(v*T)),eq(L,v*T),eq(k,2*sp.pi/L),
                               eq(T*f,1),eq(f,1/T),eq(w,2*sp.pi/T)))


def verify_proof(proof):
    expected=build_proof();f,T,L,v,w,k=symbols()
    if len(proof.steps)!=6:raise ValueError('Six source steps required')
    if proof.steps!=expected.steps:raise ValueError('Source relation differs')
    period=sp.Eq(T,1/f,evaluate=False)
    wavelength=sp.Eq(L,v/f,evaluate=False)
    angular=sp.Eq(w,2*sp.pi*f,evaluate=False)
    wavenumber=sp.Eq(k,w/v,evaluate=False)
    # Verify the feed correction and propagate actual equations in dependency order.
    computed={}
    computed[4]=sp.Eq(period.lhs*f,period.rhs*f,evaluate=False)
    computed[5]=sp.Eq(computed[4].lhs/T,computed[4].rhs/T,evaluate=False)
    computed[6]=sp.Eq(angular.lhs,angular.rhs.subs(computed[5].lhs,computed[5].rhs),evaluate=False)
    computed[1]=sp.Eq(wavenumber.lhs,wavenumber.rhs.subs(computed[6].lhs,computed[6].rhs),evaluate=False)
    computed[2]=sp.Eq(wavelength.lhs,wavelength.rhs.subs(period.rhs,period.lhs),evaluate=False)
    computed[3]=sp.Eq(computed[1].lhs,computed[1].rhs.subs(computed[2].rhs,computed[2].lhs),evaluate=False)
    for i,equation in computed.items():
        if equation!=proof.steps[i-1]:raise ValueError('Dependent transition failed: '+str(i))
    # Physical phase change over a period and wavelength must each equal 2*pi.
    if sp.simplify(computed[6].rhs*T-2*sp.pi)!=0 or sp.simplify(computed[3].rhs*L-2*sp.pi)!=0:
        raise ValueError('Phase-cycle check failed')
    return dict(source_version_id=SOURCE_VERSION,source_content_hash=SOURCE_HASH,source_ast_parity=False,
                dependency_order=list(computed),steps=[sp.srepr(s) for s in proof.steps],
                checks=dict(six_dependent_transitions=True,frequency_feed_corrected=True,time_phase_cycle=True,space_phase_cycle=True),
                domain='Single positive-frequency periodic wave with positive phase-speed magnitude; f in cycles/s, period in s, wavelength in m, omega in rad/s, k in rad/m.',
                limitation='Angular wavenumber magnitude, not cycles per length or a signed wavevector. Phase speed, not group velocity; no dispersion law or broadband inference.')


def verify_source_equations(records):
    from sciona.physics_ingest.source_symbolic import parse_source_srepr
    f,T,L,v,w,k=symbols()
    mapping=dict(zip(map(sp.Symbol,['pdg0004201','pdg0009491','pdg0001115','pdg0001357','pdg0002321','pdg0005321']),[f,T,L,v,w,k]))
    mapping[sp.Symbol('pdg0003141')]=sp.pi
    expected={identity:e for identity,e in zip(
        ['0934990943','1293923844','3121513111','2131616531','2113211456','3132131132'],build_proof().steps)}
    expected.update({'3131111133':sp.Eq(T,1/f,evaluate=False),'0404050504':sp.Eq(L,v/f,evaluate=False),
                     '3131211131':sp.Eq(w,2*sp.pi*f,evaluate=False),'5900595848':sp.Eq(k,w/v,evaluate=False)})
    if set(records)!=set(expected):raise ValueError('Ten source equations required')
    for identity,record in records.items():
        sides=[parse_source_srepr(record[key]) for key in ['sympy_lhs','sympy_rhs']]
        if any(e.free_symbols-set(mapping) for e in sides):raise ValueError('Unreviewed source expression identity')
        interpreted=[e.xreplace(mapping) for e in sides]
        if any(sp.simplify(a-b)!=0 for a,b in zip(interpreted,expected[identity].args)):
            raise ValueError('Interpreted source equation differs: '+identity)
    return dict(interpreted_equations=10,literal_source_parity=False,all_equation_sides_verified=True)
