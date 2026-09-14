from dataclasses import replace
from decimal import Decimal,localcontext
import pytest
import sympy as sp
from sciona.physics_ingest.gravity_mass_proof import build_proof,verify_proof,eq


def test_ten_steps():assert verify_proof(build_proof())['steps_verified']==10


@pytest.mark.parametrize('i',range(10))
def test_corrupt_step(i):
    p=build_proof();s=list(p.steps);s[i]=eq(s[i].lhs,s[i].rhs+1)
    with pytest.raises(ValueError):verify_proof(replace(p,steps=tuple(s)))


def test_independent_decimal_endpoint():
    with localcontext() as ctx:
        ctx.prec=70
        mass=Decimal('9.80665')*Decimal('6378100')**2/Decimal('6.67430e-11')
        assert abs(mass-Decimal(verify_proof(build_proof())['computed_mass_kg']))<1
        assert mass/Decimal('1e24')>Decimal('5.977')
        assert mass!=Decimal('5.972e24')


def test_effective_gravity_is_not_pure_gravitation():
    G,M,r,w=sp.symbols('G M r w',positive=True)
    effective=G*M/r**2-w*w*r
    assert sp.simplify(effective*r*r/G-M)==-w*w*r**3/G
