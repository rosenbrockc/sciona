from decimal import Decimal, localcontext
from dataclasses import replace
import numpy as np
import pytest
import sympy as sp
from sciona.visualizer.runner import _ensure_atoms_imported
_ensure_atoms_imported()
from sciona.atoms.physics.weighted_population_variance import weighted_population_variance
from sciona.physics_ingest.variance_proof import build_proof, verify_proof, symbols, normalized_expectation


def reference(values, weights):
    n = values.shape[-1]; rows = []
    with localcontext() as ctx:
        ctx.prec = 2500
        for values_row, weights_row in zip(values.reshape(-1,n), weights.reshape(-1,n)):
            xs = [Decimal.from_float(float(x)) for x in values_row]
            ws = [Decimal.from_float(float(w)) for w in weights_row]
            total = sum(ws)
            mean = sum(x*w for x,w in zip(xs,ws))/total
            # Independent raw-second-moment form at high precision.
            variance = sum(w*x*x for x,w in zip(xs,ws))/total-mean*mean
            rows.append((float(mean), float(variance)))
    return tuple(np.array([r[i] for r in rows]).reshape(values.shape[:-1]) for i in range(2))


@pytest.mark.parametrize('shape', [(1,), (10,), (2,3,5)])
def test_decimal_reference_and_preserved_inputs(shape):
    rng = np.random.default_rng(414); values = rng.normal(size=shape); weights = rng.uniform(.01,4,size=shape)
    before = values.copy(), weights.copy()
    for actual,expected in zip(weighted_population_variance(values,weights),reference(values,weights)):
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(values,before[0]); np.testing.assert_array_equal(weights,before[1])


@pytest.mark.parametrize('scale', [1., 1e300, 1e-300])
def test_normalization_and_bad_source_counterexample(scale):
    mean,variance = weighted_population_variance(np.array([-1.,1.]),np.array([scale,scale]))
    assert mean == 0 and variance == 1
    # The original intermediate E[x²]-2E[x²]+E[x]² would produce -1.
    assert 1-2*1+float(mean)**2 == -1


def test_large_offset_with_small_spread():
    values = np.array([1e16,1e16+2]); weights = np.ones(2)
    mean, variance = weighted_population_variance(values,weights)
    assert mean == 1e16 and variance == 1.


def test_zero_weight_extreme_and_constant_distribution():
    mean, variance = weighted_population_variance(np.array([1e308,-1e308]),np.array([1e308,0.]))
    assert mean == 1e308 and variance == 0


def test_subnormal_variance():
    values = np.array([-2e-162,2e-162]); weights = np.ones(2)
    for actual,expected in zip(weighted_population_variance(values,weights),reference(values,weights)):
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('v,w', [(np.array([]),np.array([])),(np.array(1.),np.array(1.)),
    (np.ones(2),np.ones((1,2))),(np.ones(2),np.array([-1.,2.])),(np.ones(2),np.zeros(2)),
    (np.array([np.nan]),np.ones(1)),(np.ones(1),np.array([np.inf])),
    (np.array([1j]),np.ones(1)),(np.ones(1),np.array([True])),(np.array(['1']),np.ones(1)),
    (np.ones((2,2)),np.array([[1.,1.],[0.,0.]]))])
def test_invalid_domain(v,w):
    with pytest.raises(ValueError): weighted_population_variance(v,w)


@pytest.mark.parametrize('v', [np.array([-1e308,1e308]),np.array([-1e-300,1e-300])])
def test_unrepresentable_variance_rejected(v):
    with pytest.raises(ValueError): weighted_population_variance(v,np.ones(2))


def test_corrected_proof():
    assert verify_proof(build_proof())['checks']['mean_square_preserved']


@pytest.mark.parametrize('index', range(5))
def test_proof_corruption_rejected(index):
    proof = build_proof(); expressions = list(proof.expressions); expressions[index] = sp.Symbol('unreviewed')
    with pytest.raises(ValueError): verify_proof(replace(proof,expressions=tuple(expressions)))


def test_original_faulty_intermediate_rejected():
    proof = build_proof(); _,mean,second,_ = symbols(); expressions = list(proof.expressions)
    expressions[3] = second-2*second+mean**2
    with pytest.raises(ValueError): verify_proof(replace(proof,expressions=tuple(expressions)))


def test_unassumed_higher_moment_rejected():
    x,*_ = symbols()
    with pytest.raises(ValueError): normalized_expectation(x**3)
