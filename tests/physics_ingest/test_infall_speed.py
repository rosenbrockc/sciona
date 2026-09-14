from decimal import Decimal, localcontext
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.infall_speed import infall_speed, witness_infall_speed
from sciona.ghost.abstract import AbstractArray


def reference(G, m, M, r):
    with localcontext() as ctx:
        ctx.prec = 2500
        G, m, M, r = (Decimal.from_float(float(x)) for x in (G, m, M, r))
        # Work over infinity->2r plus 2r->r, then derive speed from kinetic work.
        scale = G*m*M
        work = scale/(2*r)+scale*(1/r-1/(2*r))
        speed = (2*work/m).sqrt()
        return tuple(float(x) for x in (speed, -speed, work, -work))


@pytest.mark.parametrize('row', [
    (1., 1e-6, 1., 2.), (1e308, 1e-6, 1., 1e308),
    (1e-308, 1e-6, 1., 1.), (1., 1e-6, 1., 1e308),
    (1., 1., 1e308, 1e308), (1., np.nextafter(0., 1.), 1., 1.),
])
def test_extremes_against_work_integral_oracle(row):
    actual = infall_speed(*(np.array(x) for x in row))
    np.testing.assert_array_equal([a.item() for a in actual], reference(*row))


def test_arrays_shapes_inputs_and_signs():
    rng = np.random.default_rng(481)
    G, M, r = [np.exp(rng.uniform(-80, 80, (2, 3, 4))) for _ in range(3)]
    args = [G, M*1e-6, M, r]
    before = [a.copy() for a in args]
    actual = infall_speed(*args)
    rows = [reference(*row) for row in zip(*(a.flat for a in args))]
    for j, a in enumerate(actual):
        assert a.shape == (2, 3, 4) and a.dtype == np.float64
        np.testing.assert_array_equal(a, np.array([v[j] for v in rows]).reshape(a.shape))
        assert all(not np.shares_memory(a, x) for x in args)
    assert np.all(actual[0] > 0) and np.all(actual[2] > 0)
    np.testing.assert_array_equal(actual[1], -actual[0])
    np.testing.assert_array_equal(actual[3], -actual[2])
    for a, b in zip(args, before):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('index', range(4))
@pytest.mark.parametrize('bad', [0., -1., np.inf, np.nan])
def test_domain_rejected(index, bad):
    row = [1., 1e-6, 1., 1.]
    row[index] = bad
    with pytest.raises(ValueError):
        infall_speed(*(np.array(x) for x in row))


@pytest.mark.parametrize('row', [(1e308, 1., 1e308, 1.), (1e-308, 1e-6, 1e-308, 1e308)])
def test_output_overflow_underflow_rejected(row):
    with pytest.raises(ValueError):
        infall_speed(*(np.array(x) for x in row))


@pytest.mark.parametrize('bad', [np.array([]), np.array([True]), np.array([1j]),
                               np.array(['1']), np.array([1], dtype=object)])
def test_input_types_rejected(bad):
    with pytest.raises(ValueError):
        infall_speed(bad, np.ones(bad.shape), np.ones(bad.shape), np.ones(bad.shape))


def test_no_broadcasting_and_witness():
    with pytest.raises(ValueError):
        infall_speed(np.ones(2), np.array(1.), np.ones(2), np.ones(2))
    a = AbstractArray(shape=(2, 3), dtype='float64')
    assert len(witness_infall_speed(a, a, a, a)) == 4
    with pytest.raises(ValueError):
        witness_infall_speed(a, a, a, AbstractArray(shape=(1,), dtype='float64'))
