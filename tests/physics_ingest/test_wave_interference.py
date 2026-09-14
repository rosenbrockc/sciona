"""Synthetic complex-field reference, literal phases and invalid-domain cases."""
import mpmath
import numpy as np
import pytest
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.wave_interference import wave_interference, witness_wave_interference
from sciona.ghost.abstract import AbstractArray


def reference(*row):
    ctx = mpmath.mp.clone()
    ctx.dps = 2000
    a, b, phase = (ctx.mpf(float(v)) for v in row)
    field = a+b*ctx.exp(ctx.j*phase)
    intensity = ctx.re(field*ctx.conj(field))
    # Orthogonal phases cancel the interference in the ensemble average.
    opposite = a-b*ctx.exp(ctx.j*phase)
    mean = (intensity+ctx.re(opposite*ctx.conj(opposite)))/2
    cross = intensity-mean
    return tuple(float(v) for v in (intensity, mean, (a+b)*(a+b), (a-b)*(a-b), cross, intensity/mean))


def test_random_multidimensional_complex_reference():
    rng = np.random.default_rng(128)
    args = (np.exp(rng.uniform(-80, 80, (2, 3, 4))), np.exp(rng.uniform(-80, 80, (2, 3, 4))),
            rng.uniform(-100, 100, (2, 3, 4)))
    before = [a.copy() for a in args]
    result = wave_interference(*args)
    expected = [reference(*row) for row in zip(*(a.flat for a in args))]
    for j, actual in enumerate(result):
        np.testing.assert_array_equal(actual, np.array([r[j] for r in expected]).reshape(2, 3, 4))
        assert actual.dtype == np.float64
        assert all(not np.shares_memory(actual, a) for a in args)
    for a, b in zip(args, before):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('row', [(1, 1, 0), (1, 1, np.pi), (1, 2, -np.pi), (0, 2, 1e308),
                               (1, 2, 1e308), (1, 2, -1e308), (1e-160, 0, 0),
                               (1, 1, np.nextafter(np.pi, np.inf)), (3, 1, np.pi/2)])
def test_boundary_and_large_phase_reference(row):
    actual = wave_interference(*(np.array(v, dtype=float) for v in row))
    np.testing.assert_array_equal([a.item() for a in actual], reference(*row))
    assert all(a.shape == () for a in actual)


def test_rounded_pi_does_not_become_exact_cancellation():
    actual = wave_interference(np.array(1.), np.array(1.), np.array(np.pi))
    assert actual[0] > 0
    assert actual[3] == 0
    assert actual[5] > 0


@pytest.mark.parametrize('row', [(0, 0, 0), (-1, 1, 0), (1, -1, 0), (1, 1, np.inf),
                               (np.nan, 1, 0), (True, 1, 0), (1, 1j, 0), (1, 1, '0')])
def test_invalid_inputs(row):
    with pytest.raises(ValueError):
        wave_interference(*(np.array(v) for v in row))


def test_empty_and_broadcast_rejected():
    with pytest.raises(ValueError):
        wave_interference(*(np.array([]) for _ in range(3)))
    with pytest.raises(ValueError, match='shapes'):
        wave_interference(np.ones(2), np.ones(1), np.ones(2))


@pytest.mark.parametrize('row', [(1e308, 1, 0), (1e-308, 0, 0), (1e-160, 1e-160, np.pi)])
def test_nonzero_output_range_failure(row):
    with pytest.raises(ValueError, match='range'):
        wave_interference(*(np.array(v) for v in row))


def test_witness_contract():
    a = AbstractArray(shape=(2, 3), dtype='float64')
    result = witness_wave_interference(a, a, a)
    assert len(result) == 6 and all(v.shape == (2, 3) for v in result)
    with pytest.raises(ValueError, match='shapes'):
        witness_wave_interference(a, a, AbstractArray(shape=(1,), dtype='float64'))
