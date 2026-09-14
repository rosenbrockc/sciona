from decimal import Decimal,localcontext
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.newton_force import newton_force


def reference(*row):
    with localcontext() as ctx:
        ctx.prec=2500
        G,m1,m2,r=(Decimal.from_float(float(v)) for v in row)
        potential=-G*m1*m2/r
        force=-potential/r
        return tuple(float(v) for v in (force,force/m1,force/m2,potential))


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_reference_and_preservation(shape):
    rng=np.random.default_rng(917)
    args=[np.asarray(np.exp(rng.uniform(-80,80,size=shape))) for _ in range(4)]
    copies=[a.copy() for a in args];out=newton_force(*args)
    for i,row in enumerate(zip(*(a.flat for a in args))):assert tuple(a.flat[i] for a in out)==reference(*row)
    assert all(a.shape==shape and a.dtype==np.float64 for a in out)
    assert all(np.array_equal(a,b) for a,b in zip(args,copies))


@pytest.mark.parametrize('args',[(1.,2.,3.,4.),(1e308,1.,1.,1e308),
    (1e-308,1.,1.,1e-154),(1.,1e150,1e150,1e150),
    (np.nextafter(0.,1.),1.,1.,1.)])
def test_extremes(args):
    assert tuple(float(v) for v in newton_force(*args))==reference(*args)


def test_exchange_acceleration_order_and_scaling():
    F,a1,a2,U=newton_force(1,2,3,4)
    assert (F,a1,a2,U)==(.375,.1875,.125,-1.5)
    assert tuple(float(v) for v in newton_force(1,3,2,4))==(F,a2,a1,U)
    assert tuple(float(v) for v in newton_force(1,2,3,8))==(F/4,a1/4,a2/4,U/2)


@pytest.mark.parametrize('args',[(0,1,1,1),(1,-1,1,1),(1,1,0,1),(1,1,1,0),
    (np.inf,1,1,1),(1,np.nan,1,1),(True,1,1,1),('1',1,1,1),(1j,1,1,1),
    ([],[],[],[]),([1,2],[1],[1],[1]),(1e308,1e308,1e308,1),
    (1e-308,1e-308,1e-308,1e308)])
def test_invalid(args):
    with pytest.raises(ValueError):newton_force(*args)
