from decimal import Decimal, localcontext
import mpmath
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.sound_scale import sound_scale


def reference(*args):
    with localcontext() as ctx:
        ctx.prec=2500
        A,alpha,c,me,mp,f,g=(Decimal.from_float(float(v)) for v in args)
        # Reconstruct energy, density and both moduli at an arbitrary synthetic
        # length, rather than reuse the provider's simplified ratios.
        length=Decimal(3)
        energy=me*(alpha*c)**2/2
        density=A*mp/length**3
        bulk=f*energy/length**3
        shear=g*bulk
        return tuple(float(v) for v in ((energy/(A*mp)).sqrt(),(energy/mp).sqrt(),
            (bulk/density).sqrt(),((bulk+4*shear/3)/density).sqrt(),
            ((bulk+4*shear/3)/bulk).sqrt()))


@pytest.mark.parametrize('shape',[(),(20,),(2,3,4)])
def test_precision_and_preservation(shape):
    rng=np.random.default_rng(538)
    args=[np.asarray(np.exp(rng.uniform(0,20,size=shape)))]
    args += [np.asarray(np.exp(rng.uniform(-80,80,size=shape))) for _ in range(6)]
    copies=[a.copy() for a in args]
    outputs=sound_scale(*args)
    for i,row in enumerate(zip(*(a.flat for a in args))):
        assert tuple(out.flat[i] for out in outputs)==reference(*row)
    assert all(a.shape==shape and a.dtype==np.float64 for a in outputs)
    assert all(np.array_equal(a,b) for a,b in zip(args,copies))


@pytest.mark.parametrize('args',[(1,1,1,2,1,1,0),(4,1,1,2,1,4,3),
    (1,1e308,1e-308,2,1,1,0),(1,1,1,1e308,1e308,1,1),
    (1,1,np.nextafter(0.,1.),2,1,1,0)])
def test_special(args):
    assert tuple(float(v) for v in sound_scale(*args))==reference(*args)


def test_bound_is_conditional_and_diagnostics_expose_dropped_factors():
    estimate,upper,bulk,longitudinal,shear=sound_scale(4,1,1,2,1,4,3)
    assert estimate==0.5 and upper==1 and bulk==1
    assert longitudinal>upper and shear>1
    assert sound_scale(1,1,1,2,1,1,0)[0]==1  # wrong source would be sqrt(2)


@pytest.mark.parametrize('index,value',[(0,.5),(0,np.nan),(1,0),(2,-1),(3,0),
    (4,np.inf),(5,0),(6,-1),(6,np.nan),(0,True),(1,'1'),(2,1j)])
def test_invalid_domain(index,value):
    args=[1,1,1,2,1,1,0];args[index]=value
    with pytest.raises(ValueError):sound_scale(*args)


@pytest.mark.parametrize('args',[(1,1e308,1e308,2,1,1,0),
    (1,1e-308,1e-308,2,1,1,0),([],[],[],[],[],[],[]),
    ([1,2],[1],[1],[2],[1],[1],[0])])
def test_unrepresentable_or_shape(args):
    with pytest.raises(ValueError):sound_scale(*args)


def test_global_context_preserved():
    old=mpmath.mp.dps
    sound_scale(1,1,1,2,1,1,0)
    assert mpmath.mp.dps==old
